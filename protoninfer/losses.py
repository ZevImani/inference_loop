"""Image-space losses between generated and target detector images, plus a name registry."""
import torch
from geomloss import SamplesLoss


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


_sw_cache: dict = {}


def _get_sw_projections(H, W, n_proj, device):
    """Build and cache the per-angle pixel sort order and sorted-coordinate gaps."""
    key = (H, W, n_proj, str(device))
    if key not in _sw_cache:
        with torch.no_grad():
            ys = torch.arange(H, dtype=torch.float32, device=device)
            xs = torch.arange(W, dtype=torch.float32, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            pos = torch.stack([gy.flatten(), gx.flatten()], dim=1)          # [N, 2]
            theta = torch.arange(n_proj, dtype=torch.float32, device=device) * torch.pi / n_proj
            dirs = torch.stack([theta.sin(), theta.cos()], dim=1)          # [L, 2]
            proj_sorted, order = (dirs @ pos.T).sort(dim=1)                # [L, N]
            _sw_cache[key] = (order, proj_sorted.diff(dim=1))              # gaps: [L, N-1]
    return _sw_cache[key]


def sliced_wasserstein_loss(generated_img, target_img, n_proj=64):
    """
    Sliced Wasserstein-1 between images treated as normalized mass distributions
    over the pixel grid, maintaining gradients through generated_img.

    Each image is projected onto n_proj fixed, evenly spaced directions in
    [0, π); the 1D W1 on each line is ∫|F_gen − F_tgt|, and the loss is the mean
    over directions.  Because the support (pixel grid) is fixed, the sort order
    of projected coordinates is precomputed once, so each call is one gather +
    cumsum — no Sinkhorn iterations, no N×N cost matrix.  Angles are
    deterministic so the loss is not stochastic (important for MCMC acceptance).

    Not numerically comparable to emd_loss_with_gradients: SW1 ≤ W1 (≈ 2/π · W1
    for a pure translation), so temperatures/step sizes tuned on EMD need rescaling.

    Accepts [H, W] (returns a scalar) or batched [B, H, W] (returns [B]); a
    single target is broadcast against a batch of generated images.

    Parameters
    ----------
    n_proj : int — number of projection directions; default 64
    """
    gen = generated_img.reshape(-1, *generated_img.shape[-2:])
    tgt = target_img.detach().reshape(-1, *target_img.shape[-2:])
    B, H, W = gen.shape

    tgt_sum = tgt.flatten(1).sum(1, keepdim=True)
    if (tgt_sum == 0).all():
        return torch.zeros(B, device=gen.device, requires_grad=True).squeeze(0)

    order, gaps = _get_sw_projections(H, W, n_proj, gen.device)
    gen_w = gen.flatten(1) / (gen.flatten(1).sum(1, keepdim=True) + 1e-8)
    tgt_w = tgt.flatten(1) / (tgt_sum + 1e-8)

    cdf_diff = (gen_w - tgt_w)[:, order].cumsum(-1)[..., :-1]              # [B, L, N-1]
    sw = (cdf_diff.abs() * gaps).sum(-1).mean(-1)                          # [B]
    return sw.squeeze(0) if generated_img.ndim == 2 else sw


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
    'sliced_w':      sliced_wasserstein_loss,
    'sw':            sliced_wasserstein_loss,
    'swd':           sliced_wasserstein_loss,
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
                     'gaussian_mmd'/'gmmd', 'laplacian_mmd'/'lmmd',
                     'sliced_w'/'sw'/'swd' (param: n_proj), 'l2'/'mse'.

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
