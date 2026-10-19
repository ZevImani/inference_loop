"""LDM loading and the differentiable momentum -> image generator."""
import torch
from omegaconf import OmegaConf

from .paths import ensure_ldm_importable, load_paths, resolve_path

ensure_ldm_importable()
from ldm.util import instantiate_from_config        # noqa: E402
from ldm.models.diffusion.ddim import DDIMSampler   # noqa: E402

BACKGROUND_THRESHOLD = 5e-2


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_ldm(device, config_path=None, checkpoint_path=None):
    """Load the LDM; paths default to ldm_config / ldm_checkpoint in configs/paths.yaml."""
    if config_path is None or checkpoint_path is None:
        paths = load_paths()
        config_path     = config_path or resolve_path(paths.ldm_config)
        checkpoint_path = checkpoint_path or resolve_path(paths.ldm_checkpoint)
    print("Loading LDM...")
    config = OmegaConf.load(config_path)
    # The LDM checkpoint already holds the autoencoder weights, so don't load the (machine-local)
    # autoencoder checkpoint named in the config; the check below confirms nothing is left unloaded.
    config.model.params.first_stage_config.params.ckpt_path = None
    pl_sd  = torch.load(checkpoint_path, map_location="cpu")
    ldm    = instantiate_from_config(config.model)
    missing, _ = ldm.load_state_dict(pl_sd["state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"{checkpoint_path} is missing {len(missing)} LDM weights, e.g. {missing[:5]}")
    ldm.to(device).eval()
    # Inference only optimises momentum, so freeze the weights: backward then skips
    # the weight gradients (and their .grad buffers) and only flows to the inputs.
    ldm.requires_grad_(False)
    print("LDM loaded.\n")
    return ldm


def decode_first_stage_with_grad(model, z):
    """
    Drop-in for model.decode_first_stage() that keeps the gradient graph alive.
    The LDM method is decorated with @torch.no_grad() which silently cuts the
    graph; this calls the underlying VAE decoder directly.
    """
    return model.first_stage_model.decode(z / model.scale_factor)


class DifferentiableLDMGenerator:
    """
    Wrapper for LDM that supports both standard generation and gradient-enabled generation.
    """

    def __init__(self, model, device='cuda', ddim_steps_standard=50, ddim_steps_gradient=10):
        self.model = model
        self.device = device
        self.ddim_steps_standard = ddim_steps_standard
        self.ddim_steps_gradient = ddim_steps_gradient
        self.sampler = DDIMSampler(model)
        self._fixed_z = None  # cached noise vector, allocated on first use with fixed_z=True

    def __call__(self, px, py, pz, batch_size=1, fixed_z=False, z_clean=None, t_renoise=None,
                 renoise_steps=None):
        """
        Generate image from momentum.
        Automatically detects if gradients are needed based on input tensors.

        Args:
            px, py, pz: Momentum components (float or torch.Tensor)
            batch_size: Number of images to generate in parallel (grad mode only).
                        When > 1, returns [batch_size, H, W] with fresh noise per sample.
            fixed_z:    If True, reuse the same noise vector across all gradient calls,
                        making the loss surface a smooth function of momentum.
                        Ignored when batch_size > 1.
            z_clean:    Clean latent [1, C, H_lat, W_lat] encoded from the previous
                        inference step's best image.  When provided with t_renoise, each
                        sample starts from z_clean independently noised to t_renoise
                        instead of from pure Gaussian noise.
            t_renoise:  Integer diffusion timestep to which z_clean is noised before
                        DDIM denoising begins.  Ignored when z_clean is None.
            renoise_steps: DDIM steps for the renoise chain (t_renoise → 0).  None uses
                        ddim_steps_gradient.  Ignored unless z_clean and t_renoise are set.

        Returns:
            Generated image: [H, W] for batch_size=1, [B, H, W] for batch_size>1
        """
        needs_grad = any(
            isinstance(p, torch.Tensor) and p.requires_grad
            for p in [px, py, pz]
        )

        if not isinstance(px, torch.Tensor):
            px = torch.tensor(px, dtype=torch.float32, device=self.device)
        else:
            px = px.to(self.device)

        if not isinstance(py, torch.Tensor):
            py = torch.tensor(py, dtype=torch.float32, device=self.device)
        else:
            py = py.to(self.device)

        if not isinstance(pz, torch.Tensor):
            pz = torch.tensor(pz, dtype=torch.float32, device=self.device)
        else:
            pz = pz.to(self.device)

        momentum = torch.stack([px, py, pz]).unsqueeze(0)  # [1, 3]
        momentum_norm = momentum / 500.0

        if needs_grad:
            return self._generate_with_gradients(momentum_norm, batch_size=batch_size, fixed_z=fixed_z,
                                                  z_clean=z_clean, t_renoise=t_renoise,
                                                  renoise_steps=renoise_steps)
        else:
            return self._generate_standard(momentum_norm)

    def _generate_standard(self, momentum_norm):
        """Standard generation without gradients (full DDIM)."""
        with torch.no_grad():
            conditioning = self.model.get_learned_conditioning(momentum_norm)

            shape = [
                self.model.model.diffusion_model.in_channels,
                self.model.model.diffusion_model.image_size,
                self.model.model.diffusion_model.image_size
            ]

            samples, _ = self.sampler.sample(
                S=self.ddim_steps_standard,
                conditioning=conditioning,
                batch_size=1,
                shape=shape,
                verbose=False,
                eta=0.0
            )

            decoded = self.model.decode_first_stage(samples)
            result = decoded.squeeze()
            result[result < BACKGROUND_THRESHOLD] = 0.0
            return result

    def _generate_with_gradients(self, momentum_norm, batch_size=1, fixed_z=False,
                                 z_clean=None, t_renoise=None, renoise_steps=None):
        """
        Gradient-enabled generation using multi-step DDIM.
        Gradients flow through conditioning → apply_model → DDIM → VAE decode.

        batch_size=1, fixed_z=False : fresh noise each call (unbiased stochastic gradient).
        batch_size=1, fixed_z=True  : pinned noise reused every call (smooth loss surface).
        batch_size>1                : fresh independent noise per sample; fixed_z ignored.
                                      Returns [batch_size, H, W].
        z_clean + t_renoise         : each sample independently noises z_clean to t_renoise
                                      and starts DDIM there; takes priority over fixed_z.
                                      renoise_steps (if set) replaces ddim_steps_gradient.
        """
        conditioning = self.model.get_learned_conditioning(momentum_norm)

        shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size
        ]
        n_steps = self.ddim_steps_gradient

        if z_clean is not None and t_renoise is not None:
            # Renoise path: noise z_clean independently per sample, start DDIM from t_renoise
            alpha_t = self.model.alphas_cumprod[t_renoise]
            noise = torch.randn([batch_size] + shape, device=self.device)
            z = (alpha_t.sqrt() * z_clean.to(self.device).expand(batch_size, -1, -1, -1)
                 + (1 - alpha_t).sqrt() * noise)
            conditioning_in = (conditioning.expand(batch_size, *conditioning.shape[1:])
                               if batch_size > 1 else conditioning)
            t_start = int(t_renoise)
            if renoise_steps is not None:
                n_steps = renoise_steps
        elif batch_size > 1:
            z = torch.randn([batch_size] + shape, device=self.device)
            conditioning_in = conditioning.expand(batch_size, *conditioning.shape[1:])
            t_start = self.model.num_timesteps - 1
        elif fixed_z:
            if self._fixed_z is None or self._fixed_z.shape != torch.Size([1] + shape):
                self._fixed_z = torch.randn([1] + shape, device=self.device)
            z = self._fixed_z
            conditioning_in = conditioning
            t_start = self.model.num_timesteps - 1
        else:
            z = torch.randn([1] + shape, device=self.device)
            conditioning_in = conditioning
            t_start = self.model.num_timesteps - 1

        timesteps = torch.linspace(
            t_start, 0, n_steps,
            dtype=torch.long, device=self.device
        )

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).expand(z.shape[0])
            noise_pred = self.model.apply_model(z, t_batch, conditioning_in)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t      = self.model.alphas_cumprod[t]
                alpha_t_next = self.model.alphas_cumprod[t_next]

                pred_x0 = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
                dir_xt  = (1 - alpha_t_next).sqrt() * noise_pred
                z       = alpha_t_next.sqrt() * pred_x0 + dir_xt

        decoded = decode_first_stage_with_grad(self.model, z)
        result = decoded.squeeze() if batch_size == 1 else decoded.squeeze(1)  # [H, W] or [B, H, W]
        # Hard threshold matching _generate_standard; the mask is constant, so gradients
        # pass through unchanged for kept pixels and are zero for background pixels.
        return result * (result >= BACKGROUND_THRESHOLD)
