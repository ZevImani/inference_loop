import numpy as np
import torch
import ot
import ot.unbalanced
from geomloss import SamplesLoss
from scipy.spatial import ConvexHull
from ldm.models.diffusion.ddim import DDIMSampler

BACKGROUND_THRESHOLD = 5e-2

# ── Convex-hull initial guess ──────────────────────────────────────────────────

_HULL_BACKGROUND_THR = 0.05
_HULL_IMAGE_CENTER   = np.array([32.0, 32.0])
_HULL_DEFAULT_SCALE  = 16.5   # MeV/pixel; override via momentum_scale arg

# Power-law calibration: |p| = _PMAG_PL_SCALE * pixel_sum ** _PMAG_PL_EXPONENT
# Fitted on edep_protons64_v2 validation set via log-log regression.
# Exponent ~0.52 reflects that pixel_sum ∝ track range ∝ |p|^~1.7 (Bethe-Bloch),
# so the inverse mapping has exponent ~1/1.7 ≈ 0.59, empirically ~0.52 here.
_PMAG_PL_SCALE    = 41.025
_PMAG_PL_EXPONENT = 0.5177


def _max_area_triangle(pts):
    """O(n³) largest inscribed triangle among convex-hull vertices (typically 5–30)."""
    n = len(pts)
    best, best_area = (0, 1, 2), -1.0
    for i in range(n):
        for j in range(i + 1, n):
            ab = pts[j] - pts[i]
            for k in range(j + 1, n):
                ac = pts[k] - pts[i]
                area = abs(ab[0] * ac[1] - ab[1] * ac[0])
                if area > best_area:
                    best_area = area
                    best = (i, j, k)
    return best, best_area * 0.5


def _hull_guess_pixels(img, momentum_scale):
    """
    Extract two-track (px, py) guesses from the convex hull of a detector image.

    Algorithm:
      1. Threshold lit pixels and compute convex hull.
      2. Find the max-area inscribed triangle → vertex + two track tips.
      3. Corner closest to image centre is the interaction vertex.
      4. Pixel directions map to momentum:  px = Δrow * scale,  py = Δcol * scale.

    Returns a dict with keys guess1, guess2 (MeV arrays), dir1, dir2, vertex,
    or None on failure (too few pixels, degenerate hull, etc.).
    """
    img_thr = np.array(img, dtype=np.float64)
    img_thr[img_thr < _HULL_BACKGROUND_THR] = 0.0

    ys, xs = np.where(img_thr > 0)
    if len(xs) < 4:
        return None

    pts = np.column_stack([xs.astype(float), ys.astype(float)])   # (col, row)

    try:
        hull = ConvexHull(pts)
    except Exception:
        return None

    hverts = pts[hull.vertices]
    if len(hverts) < 3:
        return None

    (i, j, k), _ = _max_area_triangle(hverts)
    triangle = hverts[[i, j, k]]

    dists  = np.linalg.norm(triangle - _HULL_IMAGE_CENTER, axis=1)
    v_idx  = int(np.argmin(dists))
    t1_idx = (v_idx + 1) % 3
    t2_idx = (v_idx + 2) % 3

    vertex = triangle[v_idx]
    dir1   = triangle[t1_idx] - vertex   # (dcol, drow)
    dir2   = triangle[t2_idx] - vertex

    # px ∝ Δrow, py ∝ Δcol  (empirically verified mapping from calibration)
    guess1 = np.array([dir1[1] * momentum_scale, dir1[0] * momentum_scale])
    guess2 = np.array([dir2[1] * momentum_scale, dir2[0] * momentum_scale])

    return dict(guess1=guess1, guess2=guess2, dir1=dir1, dir2=dir2, vertex=vertex)


def image_pmag_estimate(img):
    """
    Estimate |p| from total above-threshold pixel sum via a power-law calibration.

    Uses |p| = _PMAG_PL_SCALE * pixel_sum ** _PMAG_PL_EXPONENT, fitted by log-log
    regression on the edep_protons64_v2 validation set.  The exponent (~0.52) accounts
    for the nonlinear Bethe-Bloch range-momentum relationship; this reduces std(ratio)
    from ~0.09 (linear) to ~0.006 on the validation set.

    Parameters
    ----------
    img : (H, W) array-like

    Returns
    -------
    float — estimated |p| in MeV
    """
    if hasattr(img, 'cpu'):
        img = img.detach().cpu().numpy()
    img = np.asarray(img, dtype=np.float32)
    pixel_sum = float(img[img > _HULL_BACKGROUND_THR].sum())
    return _PMAG_PL_SCALE * pixel_sum ** _PMAG_PL_EXPONENT


def convex_hull_initial_guess(img, n_tracks=1, momentum_scale=None, pz_init=0.0,
                               use_pmag_estimate=False):
    """
    Estimate initial momentum guess(es) from a detector image using convex hull.

    For single-track, returns the direction of the dominant (longer) track tip.
    For two-track, calls the hull algorithm on the combined sum image and returns
    both track directions.

    When use_pmag_estimate=True, image_pmag_estimate() is used to set |p| from
    total energy deposition via a power-law calibration.  The hull-derived (px, py)
    direction is preserved; pz is set to sqrt(max(|p|^2 - px^2 - py^2, 0)), giving
    a non-zero starting pz when the energy estimate implies |p| > |pt|.
    pz_init is ignored when use_pmag_estimate=True.

    Parameters
    ----------
    img               : (H, W) array-like  — detector image (sum image for 2-track)
    n_tracks          : int  — 1 or 2
    momentum_scale    : float or None  — MeV/pixel; defaults to 15.0 MeV/pixel
    pz_init           : float  — initial pz in MeV when use_pmag_estimate=False (default 0)
    use_pmag_estimate : bool   — use power-law |p| estimator (default False)

    Returns
    -------
    List of n_tracks (px, py, pz) tuples, or None if hull construction fails
    (caller should fall back to random initialisation).
    """
    if momentum_scale is None:
        momentum_scale = _HULL_DEFAULT_SCALE

    if hasattr(img, 'cpu'):
        img = img.detach().cpu().numpy()
    img = np.asarray(img, dtype=np.float32)

    result = _hull_guess_pixels(img, momentum_scale)
    if result is None:
        return None

    if use_pmag_estimate:
        pmag_est = image_pmag_estimate(img)
        def _pz(px, py):
            return float(np.sqrt(max(pmag_est**2 - float(px)**2 - float(py)**2, 0.0)))
    else:
        def _pz(px, py):
            return float(pz_init)

    if n_tracks == 1:
        d1 = np.linalg.norm(result['dir1'])
        d2 = np.linalg.norm(result['dir2'])
        guess = result['guess1'] if d1 >= d2 else result['guess2']
        return [(float(guess[0]), float(guess[1]), _pz(*guess))]

    g1, g2 = result['guess1'], result['guess2']
    return [
        (float(g1[0]), float(g1[1]), _pz(*g1)),
        (float(g2[0]), float(g2[1]), _pz(*g2)),
    ]


def decode_first_stage_with_grad(model, z):
    """
    Drop-in for model.decode_first_stage() that keeps the gradient graph alive.
    The LDM method is decorated with @torch.no_grad() which silently cuts the
    graph; this calls the underlying VAE decoder directly.
    """
    return model.first_stage_model.decode(z / model.scale_factor)


_cost_matrix_cache: dict = {}


def _get_cost_matrix(H, W, device):
    """Build and cache the pairwise L2-distance cost matrix over pixel coordinates."""
    key = (H, W, str(device))
    if key not in _cost_matrix_cache:
        with torch.no_grad():
            ys = torch.arange(H, dtype=torch.float32, device=device)
            xs = torch.arange(W, dtype=torch.float32, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            pos = torch.stack([gy.flatten(), gx.flatten()], dim=1)  # [N, 2]
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)              # [N, N, 2]
            _cost_matrix_cache[key] = (diff ** 2).sum(-1).sqrt()    # [N, N]
    return _cost_matrix_cache[key]


def _prepare_distributions(generated_img, target_img):
    """Flatten images into normalized weight vectors over a shared pixel coordinate grid."""
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    H, W = generated_img.shape
    tgt_sum = target_img.detach().sum()

    ys = torch.arange(H, dtype=torch.float32, device=generated_img.device)
    xs = torch.arange(W, dtype=torch.float32, device=generated_img.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    all_pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # [H*W, 2]

    gen_w = generated_img.flatten() / (generated_img.flatten().sum() + 1e-8)
    tgt_w = target_img.flatten().detach() / (tgt_sum + 1e-8)

    return gen_w, tgt_w, all_pos, tgt_sum


def emd_loss_with_gradients(generated_img, target_img, reg=0.01):
    """
    Sinkhorn EMD (Wasserstein-1) via geomloss SamplesLoss, maintaining gradients
    through generated_img.

    Both images are treated as weighted distributions over a shared pixel
    coordinate grid.  Every pixel's intensity is a weight, so gradients reach
    all pixels — including currently-dark ones that should be lit — giving the
    optimizer a full spatial signal rather than only an intensity signal at
    already-nonzero locations.

    Uses geomloss rather than ot.sinkhorn2 because POT's sinkhorn2 does not
    reliably propagate autograd gradients through the distribution weights.

    Parameters
    ----------
    reg : float — entropic regularisation (geomloss `blur`); default 0.01
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    gen_w, tgt_w, all_pos, _ = _prepare_distributions(generated_img, target_img)
    return SamplesLoss("sinkhorn", p=1, blur=reg)(gen_w, all_pos, tgt_w, all_pos)


def unbalanced_emd_loss(generated_img, target_img, reg=0.05, reg_m=0.5):
    """
    Unbalanced EMD via geomloss SamplesLoss with reach parameter.

    Unlike balanced EMD, total mass need not be conserved: the loss penalises
    both spatial displacement and mass creation/destruction via a KL divergence
    penalty.  Smaller reg_m → more lenient mass imbalance.

    Weights are kept unnormalized so mass differences between generated and
    target images contribute to the loss.

    Uses geomloss rather than ot.unbalanced.sinkhorn_unbalanced2 because POT's
    implementation does not reliably propagate autograd gradients.  For p=1,
    geomloss reach maps directly to POT reg_m (both are the KL penalty weight).

    Parameters
    ----------
    reg   : float — entropic regularisation (geomloss `blur`); default 0.05
    reg_m : float — KL marginal penalty / mass relaxation (geomloss `reach`); default 0.5
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    H, W = generated_img.shape
    ys = torch.arange(H, dtype=torch.float32, device=generated_img.device)
    xs = torch.arange(W, dtype=torch.float32, device=generated_img.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    all_pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)

    gen_w = generated_img.flatten().clamp(min=0)
    tgt_w = target_img.flatten().detach().clamp(min=0)

    return SamplesLoss("sinkhorn", p=1, blur=reg, reach=reg_m)(gen_w, all_pos, tgt_w, all_pos)


def wasserstein2_loss(generated_img, target_img, blur=0.01):
    """
    Sinkhorn Wasserstein-2 loss (p=2).

    Penalizes large spatial displacements more severely than W1, which can
    produce stronger gradients when the generated and target distributions are
    far apart, but may be less stable near convergence.
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    gen_w, tgt_w, all_pos, _ = _prepare_distributions(generated_img, target_img)
    return SamplesLoss("sinkhorn", p=2, blur=blur)(gen_w, all_pos, tgt_w, all_pos)


def energy_distance_loss(generated_img, target_img):
    """
    Energy distance via geomloss.

    An unregularized, scale-free divergence that does not require a blur
    parameter.  Tends to produce sparse, mode-seeking gradients.
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    gen_w, tgt_w, all_pos, _ = _prepare_distributions(generated_img, target_img)
    return SamplesLoss("energy")(gen_w, all_pos, tgt_w, all_pos)


def gaussian_mmd_loss(generated_img, target_img, blur=5.0):
    """
    Gaussian kernel MMD via geomloss.

    Measures the difference between kernel-smoothed versions of both
    distributions.  The blur parameter controls the kernel bandwidth
    (in pixel units); larger values give a smoother, more global signal.
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    gen_w, tgt_w, all_pos, _ = _prepare_distributions(generated_img, target_img)
    return SamplesLoss("gaussian", blur=blur)(gen_w, all_pos, tgt_w, all_pos)


def laplacian_mmd_loss(generated_img, target_img, blur=5.0):
    """
    Laplacian kernel MMD via geomloss.

    Heavier-tailed than Gaussian MMD, making it more robust to outlier
    pixels and sparse detector hits.
    """
    if generated_img.ndim > 2:
        generated_img = generated_img.squeeze()
    if target_img.ndim > 2:
        target_img = target_img.squeeze()

    tgt_sum = target_img.detach().sum()
    if tgt_sum == 0:
        return torch.tensor(0.0, device=generated_img.device, requires_grad=True)

    gen_w, tgt_w, all_pos, _ = _prepare_distributions(generated_img, target_img)
    return SamplesLoss("laplacian", blur=blur)(gen_w, all_pos, tgt_w, all_pos)


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


_LOSS_REGISTRY = {
    'emd':           emd_loss_with_gradients,
    'w1':            emd_loss_with_gradients,
    'sinkhorn':      emd_loss_with_gradients,
    'unbal_emd':     unbalanced_emd_loss,
    'uemd':          unbalanced_emd_loss,
    'w2':            wasserstein2_loss,
    'sinkhorn_p2':   wasserstein2_loss,
    'energy':        energy_distance_loss,
    'gaussian_mmd':  gaussian_mmd_loss,
    'gmmd':          gaussian_mmd_loss,
    'laplacian_mmd': laplacian_mmd_loss,
    'lmmd':          laplacian_mmd_loss,
    'l2':            l2_loss_with_gradients,
    'mse':           l2_loss_with_gradients,
}

LOSS_NAMES = list(_LOSS_REGISTRY.keys())


def get_loss_fn(name_or_callable):
    """
    Resolve a loss function by name string or pass through a callable directly.

    Supported names: 'emd'/'w1'/'sinkhorn' (POT sinkhorn2, param: reg),
                     'unbal_emd'/'uemd' (POT sinkhorn_unbalanced2, params: reg, reg_m),
                     'w2'/'sinkhorn_p2', 'energy',
                     'gaussian_mmd'/'gmmd', 'laplacian_mmd'/'lmmd', 'l2'/'mse'.

    A callable is returned unchanged, so callers can pass in custom functions.
    """
    if callable(name_or_callable):
        return name_or_callable
    key = name_or_callable.lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name_or_callable}'. "
            f"Available: {', '.join(sorted(set(_LOSS_REGISTRY.keys())))}"
        )
    return _LOSS_REGISTRY[key]


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

    def __call__(self, px, py, pz, batch_size=1, fixed_z=False, z_clean=None, t_renoise=None):
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
                                                  z_clean=z_clean, t_renoise=t_renoise)
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
                                 z_clean=None, t_renoise=None):
        """
        Gradient-enabled generation using multi-step DDIM.
        Gradients flow through conditioning → apply_model → DDIM → VAE decode.

        batch_size=1, fixed_z=False : fresh noise each call (unbiased stochastic gradient).
        batch_size=1, fixed_z=True  : pinned noise reused every call (smooth loss surface).
        batch_size>1                : fresh independent noise per sample; fixed_z ignored.
                                      Returns [batch_size, H, W].
        z_clean + t_renoise         : each sample independently noises z_clean to t_renoise
                                      and starts DDIM there; takes priority over fixed_z.
        """
        conditioning = self.model.get_learned_conditioning(momentum_norm)

        shape = [
            self.model.model.diffusion_model.in_channels,
            self.model.model.diffusion_model.image_size,
            self.model.model.diffusion_model.image_size
        ]

        if z_clean is not None and t_renoise is not None:
            # Renoise path: noise z_clean independently per sample, start DDIM from t_renoise
            alpha_t = self.model.alphas_cumprod[t_renoise]
            noise = torch.randn([batch_size] + shape, device=self.device)
            z = (alpha_t.sqrt() * z_clean.to(self.device).expand(batch_size, -1, -1, -1)
                 + (1 - alpha_t).sqrt() * noise)
            conditioning_in = (conditioning.expand(batch_size, *conditioning.shape[1:])
                               if batch_size > 1 else conditioning)
            t_start = int(t_renoise)
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
            t_start, 0, self.ddim_steps_gradient,
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
