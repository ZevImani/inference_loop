import torch
from geomloss import SamplesLoss
from ldm.models.diffusion.ddim import DDIMSampler

BACKGROUND_THRESHOLD = 5e-2


def decode_first_stage_with_grad(model, z):
    """
    Drop-in for model.decode_first_stage() that keeps the gradient graph alive.
    The LDM method is decorated with @torch.no_grad() which silently cuts the
    graph; this calls the underlying VAE decoder directly.
    """
    return model.first_stage_model.decode(z / model.scale_factor)


def emd_loss_with_gradients(generated_img, target_img, blur=0.01):
    """
    Compute Sinkhorn EMD maintaining gradients through generated_img.

    Both images are treated as weighted distributions over a shared pixel
    coordinate grid.  Every pixel's intensity is a weight, so gradients reach
    all pixels — including currently-dark ones that should be lit — giving the
    optimizer a full spatial signal rather than only an intensity signal at
    already-nonzero locations.
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    H, W = generated_img.shape

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    # Shared coordinate grid for both distributions
    ys = torch.arange(H, dtype=torch.float32, device=generated_img.device)
    xs = torch.arange(W, dtype=torch.float32, device=generated_img.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    all_pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # [H*W, 2]

    gen_w = generated_img.flatten()                      # gradients flow through here
    tgt_w = target_img.flatten().detach()

    gen_w = gen_w / (gen_w.sum() + 1e-8)
    tgt_w = tgt_w / tgt_sum

    EMD = SamplesLoss("sinkhorn", p=1, blur=blur)
    return EMD(gen_w, all_pos, tgt_w, all_pos)


def l2_loss_with_gradients(generated_img, target_img):
    """
    Mean squared error between generated and target images.
    Gradients flow through generated_img; target_img is detached.
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()
    return torch.nn.functional.mse_loss(generated_img, target_img.detach())


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

    def __call__(self, px, py, pz, batch_size=1, fixed_z=False):
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
            return self._generate_with_gradients(momentum_norm, batch_size=batch_size, fixed_z=fixed_z)
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

    def _generate_with_gradients(self, momentum_norm, batch_size=1, fixed_z=False):
        """
        Gradient-enabled generation using multi-step DDIM.
        Gradients flow through conditioning → apply_model → DDIM → VAE decode.

        batch_size=1, fixed_z=False : fresh noise each call (unbiased stochastic gradient).
        batch_size=1, fixed_z=True  : pinned noise reused every call (smooth loss surface).
        batch_size>1                : fresh independent noise per sample; fixed_z ignored.
                                      Returns [batch_size, H, W].
        """
        conditioning = self.model.get_learned_conditioning(momentum_norm)

        shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size
        ]

        if batch_size > 1:
            z = torch.randn([batch_size] + shape, device=self.device)
            conditioning_in = conditioning.expand(batch_size, *conditioning.shape[1:])
        elif fixed_z:
            if self._fixed_z is None or self._fixed_z.shape != torch.Size([1] + shape):
                self._fixed_z = torch.randn([1] + shape, device=self.device)
            z = self._fixed_z
            conditioning_in = conditioning
        else:
            z = torch.randn([1] + shape, device=self.device)
            conditioning_in = conditioning

        timesteps = torch.linspace(
            self.model.num_timesteps - 1, 0, self.ddim_steps_gradient,
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
        if batch_size == 1:
            result = decoded.squeeze()
            return torch.nn.functional.relu(result - BACKGROUND_THRESHOLD)
        else:
            result = decoded.squeeze(1)  # [B, H, W]
            return torch.nn.functional.relu(result - BACKGROUND_THRESHOLD)
