import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import torch
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.append('/n/home11/zimani/reco_model/')
sys.path.append("/n/home11/zimani/latent-diffusion")

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from helper_inference import (
    DifferentiableLDMGenerator,
    emd_loss_with_gradients,
    l2_loss_with_gradients,
)
from ResNet import ResNet50

# ── Default paths ──────────────────────────────────────────────────────────────
LDM_CONFIG_PATH     = "/n/home11/zimani/latent-diffusion/configs/latent-diffusion/protons64-ldm-kl.yaml"
LDM_CHECKPOINT_PATH = "/n/home11/zimani/latent-diffusion/edep_protons64_v2_ldm/runs/checkpoints/epoch=000075.ckpt"
RECO_CHECKPOINT     = '/n/home11/zimani/reco_model/checkpoints/ResNet50_edep_v4/ResNet50_epoch100.pt'


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_ldm(device, config_path=LDM_CONFIG_PATH, checkpoint_path=LDM_CHECKPOINT_PATH):
    print("Loading LDM...")
    config = OmegaConf.load(config_path)
    pl_sd  = torch.load(checkpoint_path, map_location="cpu")
    ldm    = instantiate_from_config(config.model)
    ldm.load_state_dict(pl_sd["state_dict"], strict=False)
    ldm.to(device).eval()
    print("LDM loaded.\n")
    return ldm


def load_reco_model(device, checkpoint_path=RECO_CHECKPOINT):
    model = ResNet50(num_classes=3, channels=1, norm='batch')
    model.to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=True)['model_state_dict']
    )
    model.eval()
    return model


def reco_predict(reco_model, raw_img, device):
    arr = np.array(raw_img, dtype=np.float32)
    if arr.ndim == 2:
        inp = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)
    elif arr.ndim == 3:
        inp = torch.tensor(arr).unsqueeze(0).to(device)
    else:
        inp = torch.tensor(arr).to(device)
    with torch.no_grad():
        pred = reco_model(inp)
    return (pred.squeeze().cpu().numpy() * 500).tolist()


def random_momentum_init(p_min=100, p_max=500, component_range=300, seed=None):
    """Sample a random (px, py, pz) with |p| in [p_min, p_max]."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.uniform(-component_range, component_range, size=3)
        mag = np.linalg.norm(p)
        if p_min <= mag <= p_max:
            return tuple(p.tolist()), mag


# ── Core SGD (generalized to N tracks) ────────────────────────────────────────

def _run_sgd_core(
    generator,
    target_img,
    initial_momenta,        # list of N (px, py, pz) tuples
    n_iterations=100,
    learning_rate=0.1,
    lr_min=0.001,
    gradient_clip=1.0,
    min_distance=0.05,
    optimizer_type='SGD',
    avg_grad=True,
    avg_grad_batch_size=32,
    fixed_z=False,
    device='cuda',
    verbose=True,
    loss_type='emd',
):
    """
    Gradient descent over N sets of (px, py, pz) simultaneously.

    The combined image is the sum of N generated images; EMD is computed on
    the combined image so gradients propagate to all N momentum triples.
    Each track has its own optimizer so Adam moment accumulators remain
    independent across tracks.

    Returns
    -------
    img_path    : list of combined image tensors (CPU), one per step
    dist_path   : list of float EMD losses
    mom_path    : list of tuples — ((px1,py1,pz1), ...) for N>1,
                  or (px, py, pz) for N==1
    explore_path: list of bool (always False)
    lr_path     : list of float learning rates
    """
    loss_fn = l2_loss_with_gradients if loss_type == 'l2' else emd_loss_with_gradients

    n_tracks = len(initial_momenta)
    SCALE    = 500.0
    target_img = target_img.to(device)

    # Build N×3 learnable parameter groups
    params_list = []
    for init_mom in initial_momenta:
        group = [
            torch.tensor(float(init_mom[i]) / SCALE, dtype=torch.float32,
                         requires_grad=True, device=device)
            for i in range(3)
        ]
        params_list.append(group)

    def _make_opt(params):
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(params, lr=learning_rate)
        return torch.optim.SGD(params, lr=learning_rate)

    optimizers = [_make_opt(p) for p in params_list]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=n_iterations, eta_min=lr_min)
        for o in optimizers
    ]

    def _current_mom():
        moms = tuple(
            tuple(p.item() * SCALE for p in group)
            for group in params_list
        )
        return moms[0] if n_tracks == 1 else moms

    # Record starting state without grad
    with torch.no_grad():
        init_imgs = [
            generator(g[0].item() * SCALE, g[1].item() * SCALE, g[2].item() * SCALE)
            for g in params_list
        ]
        init_combined = sum(init_imgs)
        init_loss = loss_fn(init_combined, target_img).item()

    img_path    = [init_combined.cpu()]
    dist_path   = [init_loss]
    mom_path    = [_current_mom()]
    explore_path = [False]
    lr_path     = []

    best_loss    = init_loss
    best_momenta = mom_path[0]

    if verbose:
        for i, g in enumerate(params_list):
            print(f"Track {i+1} init: ({g[0].item()*SCALE:.2f}, {g[1].item()*SCALE:.2f}, {g[2].item()*SCALE:.2f})")
        print(f"Initial {loss_type.upper()} loss: {init_loss:.6f}")
        print(f"Starting {n_tracks}-track SGD ({optimizer_type.upper()}, lr={learning_rate})...")
        print("=" * 70)

    for iteration in range(n_iterations):
        # Enforce lr floor
        for opt in optimizers:
            if opt.param_groups[0]['lr'] < lr_min:
                for g in opt.param_groups:
                    g['lr'] = lr_min
        current_lr = optimizers[0].param_groups[0]['lr']

        for opt in optimizers:
            opt.zero_grad()

        if avg_grad:
            # Generate one batch per track, then sum across tracks element-wise
            batches = [
                generator(g[0] * SCALE, g[1] * SCALE, g[2] * SCALE,
                          batch_size=avg_grad_batch_size)
                for g in params_list
            ]
            losses = torch.stack([
                loss_fn(
                    sum(batches[t][i] for t in range(n_tracks)), target_img
                )
                for i in range(avg_grad_batch_size)
            ])
            losses.mean().backward()
            best_idx = losses.detach().argmin().item()
            combined = sum(batches[t][best_idx] for t in range(n_tracks)).detach().cpu()
            current_loss = losses[best_idx].item()
        else:
            gens     = [generator(g[0] * SCALE, g[1] * SCALE, g[2] * SCALE,
                                  fixed_z=fixed_z)
                        for g in params_list]
            combined = sum(gens)
            loss     = loss_fn(combined, target_img)
            loss.backward()
            current_loss = loss.item()
            combined = combined.detach().cpu()

        # Per-track gradient clipping
        for group in params_list:
            grad_norm = torch.sqrt(
                sum(p.grad.data.norm() ** 2 for p in group if p.grad is not None)
            ).item()
            if grad_norm > gradient_clip:
                scale = gradient_clip / grad_norm
                for p in group:
                    if p.grad is not None:
                        p.grad.data.mul_(scale)

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        current_mom = _current_mom()
        if current_loss < best_loss:
            best_loss    = current_loss
            best_momenta = current_mom

        img_path.append(combined)
        dist_path.append(current_loss)
        mom_path.append(current_mom)
        explore_path.append(False)
        lr_path.append(current_lr)

        if verbose and ((iteration + 1) % 10 == 0 or iteration == 0):
            grad_info = "  ".join(
                f"|g{i+1}|={torch.sqrt(sum(p.grad.data.norm()**2 for p in g if p.grad is not None)).item():.3f}"
                for i, g in enumerate(params_list)
            )
            print(f"Iter {iteration+1:3d}: Loss={current_loss:.6f} | Best={best_loss:.6f} | "
                  f"LR={current_lr:.4f} | {grad_info}")

        if best_loss < min_distance:
            if verbose:
                print(f"\nConverged at iteration {iteration+1} "
                      f"(loss {best_loss:.6f} < {min_distance})")
            break

    if verbose:
        print("=" * 70)
        print(f"{n_tracks}-track SGD complete.  Best loss: {best_loss:.6f}")
        if n_tracks == 1:
            print(f"Best momentum: ({best_momenta[0]:.2f}, {best_momenta[1]:.2f}, {best_momenta[2]:.2f})")
        else:
            for i, m in enumerate(best_momenta):
                print(f"Best track {i+1}: ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f})")

    return img_path, dist_path, mom_path, explore_path, lr_path


# ── Dual-projection SGD (xy + xz views of the same proton) ────────────────────

def _run_dual_proj_sgd(
    xy_target,
    xz_target,
    initial_momentum,
    generator,
    n_iterations=39,
    learning_rate=0.1,
    lr_min=0.001,
    gradient_clip=1.0,
    min_distance=0.05,
    optimizer_type='SGD',
    avg_grad_batch_size=32,
    device='cuda',
    loss_type='emd',
):
    """
    Single (px, py, pz) optimized against both xy and xz projections simultaneously.

    The xy path calls generator(px, py, pz); the xz path calls generator(px, pz, py),
    swapping py/pz so xz gradients accumulate on the correct parameters.
    Both paths share a single backward pass.

    Returns img_path, dist_path, mom_path, lr_path, best_loss, best_mom.
    """
    loss_fn = l2_loss_with_gradients if loss_type == 'l2' else emd_loss_with_gradients

    SCALE = 500.0
    device = torch.device(device) if isinstance(device, str) else device
    xy_target = xy_target.to(device)
    xz_target = xz_target.to(device)

    px, py, pz = [
        torch.tensor(float(initial_momentum[i]) / SCALE, dtype=torch.float32,
                     requires_grad=True, device=device)
        for i in range(3)
    ]
    params = [px, py, pz]

    if optimizer_type.lower() == 'adam':
        opt = torch.optim.Adam(params, lr=learning_rate)
    else:
        opt = torch.optim.SGD(params, lr=learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations, eta_min=lr_min)

    def _mom():
        return tuple(p.item() * SCALE for p in params)

    combined_target = xy_target + xz_target

    with torch.no_grad():
        _px, _py, _pz = (p.item() * SCALE for p in params)
        xy0 = generator(_px, _py, _pz)
        xz0 = generator(_px, _pz, _py)
        init_loss = loss_fn(xy0 + xz0, combined_target).item()

    img_path  = [(xy0 + xz0).cpu()]
    dist_path = [init_loss]
    mom_path  = [_mom()]
    lr_path   = []
    best_loss = init_loss
    best_mom  = mom_path[0]

    for _ in range(n_iterations):
        if opt.param_groups[0]['lr'] < lr_min:
            for g in opt.param_groups:
                g['lr'] = lr_min
        current_lr = opt.param_groups[0]['lr']

        opt.zero_grad()

        xy_batch = generator(px * SCALE, py * SCALE, pz * SCALE,
                             batch_size=avg_grad_batch_size)
        xz_batch = generator(px * SCALE, pz * SCALE, py * SCALE,
                             batch_size=avg_grad_batch_size)

        losses = torch.stack([
            loss_fn(xy_batch[i] + xz_batch[i], combined_target)
            for i in range(avg_grad_batch_size)
        ])
        losses.mean().backward()

        best_idx     = losses.detach().argmin().item()
        combined_img = (xy_batch[best_idx] + xz_batch[best_idx]).detach().cpu()
        current_loss = losses[best_idx].item()

        grad_norm = torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        ).item()
        if grad_norm > gradient_clip:
            scale = gradient_clip / grad_norm
            for p in params:
                if p.grad is not None:
                    p.grad.data.mul_(scale)

        opt.step()
        sched.step()

        current_mom = _mom()
        if current_loss < best_loss:
            best_loss = current_loss
            best_mom  = current_mom

        img_path.append(combined_img)
        dist_path.append(current_loss)
        mom_path.append(current_mom)
        lr_path.append(current_lr)

        if best_loss < min_distance:
            break

    return img_path, dist_path, mom_path, lr_path, best_loss, best_mom


# ── Public entry point ─────────────────────────────────────────────────────────

def run_inference(
    target_img,
    true_momentum,
    n_tracks=1,
    initial_momenta=None,
    # Dual-projection mode
    dual_projection=False,
    xz_img=None,           # required when dual_projection=True
    # SGD hyperparameters
    n_iterations=39,
    learning_rate=0.1,
    lr_min=0.001,
    gradient_clip=1.0,
    min_distance=0.05,
    optimizer_type='SGD',
    avg_grad=True,
    avg_grad_batch_size=32,
    fixed_z=False,
    # LDM settings
    ddim_steps_standard=50,
    ddim_steps_gradient=10,
    # Output
    output_dir='inference_results',
    run_name=None,
    # Misc
    device=None,
    verbose=True,
    save_plots=True,
    generator=None,  # pass a pre-loaded DifferentiableLDMGenerator to skip model loading
    loss_type='emd', # 'emd' for Sinkhorn EMD or 'l2' for MSE
):
    """
    Run gradient-guided LDM inference to recover proton momenta from a detector image.

    Parameters
    ----------
    target_img : np.ndarray or torch.Tensor
        2-D detector image.  In dual_projection mode this is the xy projection.
    true_momentum : tuple or list of tuples
        Ground-truth momenta.  Single track: (px, py, pz).
        Multi-track: ((px1,py1,pz1), ...).
    n_tracks : int
        Number of tracks (1, 2, or 3).  Ignored when dual_projection=True.
    initial_momenta : list of (px, py, pz) or None
        Starting point per track.  Random if None.
    dual_projection : bool
        If True, optimise a single (px, py, pz) against both the xy projection
        (target_img) and the xz projection (xz_img) simultaneously.
        The xz path permutes py/pz so gradients flow to the correct parameters.
    xz_img : np.ndarray or torch.Tensor or None
        xz projection image; required when dual_projection=True.
    n_iterations : int
        Maximum SGD steps.
    learning_rate : float
        Initial learning rate (normalised space, divide by 500 for MeV).
    lr_min : float
        Minimum learning rate for the cosine schedule.
    gradient_clip : float
        Per-track gradient norm clip (normalised space).
    min_distance : float
        Early-stopping EMD threshold.
    optimizer_type : str
        'SGD' or 'adam'.
    avg_grad : bool
        Average gradients over a batch of stochastic LDM samples.
    avg_grad_batch_size : int
        Batch size when avg_grad=True.
    fixed_z : bool
        Fix latent noise across steps (only when avg_grad=False).
    ddim_steps_standard : int
        DDIM steps for standard (no-grad) generation.
    ddim_steps_gradient : int
        DDIM steps for gradient-enabled generation.
    output_dir : str
        Root directory for output files.
    run_name : str or None
        Sub-folder name; auto-generated if None.
    device : str or torch.device or None
        Compute device; auto-detects CUDA if None.
    verbose : bool
        Print progress.
    save_plots : bool
        Generate and save diagnostic plots.
    loss_type : str
        'emd' (default) for Sinkhorn Earth Mover's Distance, or 'l2' for MSE.

    Returns
    -------
    results : dict with keys:
        img_path     - list of image tensors (one per step)
        dist_path    - list of EMD losses
        mom_path     - list of momentum tuples
        explore_path - list of bool
        lr_path      - list of learning rates
        data_dir     - path where .pt / .npy files were saved
        plot_dir     - path where plots were saved (or None)
        elapsed      - wall-clock seconds
    """
    start_time = time.time()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    if verbose:
        print(f"Device: {device}")

    # Convert xy target to tensor
    if isinstance(target_img, np.ndarray):
        raw_xy = target_img.astype(np.float32)
        xy_tensor = torch.tensor(raw_xy, dtype=torch.float32)
    else:
        xy_tensor = target_img.float()
        raw_xy = xy_tensor.cpu().numpy()

    # ── Dual-projection branch ─────────────────────────────────────────────────
    if dual_projection:
        if xz_img is None:
            raise ValueError("xz_img is required when dual_projection=True")

        if isinstance(xz_img, np.ndarray):
            raw_xz = xz_img.astype(np.float32)
            xz_tensor = torch.tensor(raw_xz, dtype=torch.float32)
        else:
            xz_tensor = xz_img.float()
            raw_xz = xz_tensor.cpu().numpy()

        # Normalise single-track true_momentum
        if not isinstance(true_momentum[0], (tuple, list, np.ndarray)):
            true_momentum_list = [tuple(float(v) for v in true_momentum)]
        else:
            true_momentum_list = [tuple(float(v) for v in true_momentum)]

        init_mom = initial_momenta[0] if initial_momenta else None
        if init_mom is None:
            init_mom, mag = random_momentum_init()
            if verbose:
                print(f"Random init: ({init_mom[0]:.1f}, {init_mom[1]:.1f}, {init_mom[2]:.1f})  |p|={mag:.1f}")
        else:
            init_mom = tuple(float(v) for v in init_mom)

        if generator is None:
            ldm = load_ldm(device)
            generator = DifferentiableLDMGenerator(
                ldm, device=str(device),
                ddim_steps_standard=ddim_steps_standard,
                ddim_steps_gradient=ddim_steps_gradient,
            )

        img_path, dist_path, mom_path, lr_path, best_loss, best_mom = _run_dual_proj_sgd(
            xy_target=xy_tensor,
            xz_target=xz_tensor,
            initial_momentum=init_mom,
            generator=generator,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            lr_min=lr_min,
            gradient_clip=gradient_clip,
            min_distance=min_distance,
            optimizer_type=optimizer_type,
            avg_grad_batch_size=avg_grad_batch_size,
            device=str(device),
            loss_type=loss_type,
        )

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nTotal time: {elapsed:.2f}s  ({elapsed / len(img_path):.2f}s per step)")
            print(f"Best loss: {best_loss:.6f}")
            print(f"Best momentum: ({best_mom[0]:.2f}, {best_mom[1]:.2f}, {best_mom[2]:.2f})")

        if run_name is None:
            run_name = "dual_proj" + ("_l2" if loss_type == 'l2' else "")

        data_dir = os.path.join(output_dir, run_name, "data_files")
        plot_dir = os.path.join(output_dir, run_name, "plots")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        raw_combined = raw_xy + raw_xz
        torch.save(torch.stack(img_path), os.path.join(data_dir, "img_path.pt"))
        np.save(os.path.join(data_dir, "mom_path.npy"),   np.array(mom_path))
        np.save(os.path.join(data_dir, "dist_path.npy"),  np.array(dist_path))
        np.save(os.path.join(data_dir, "std_path.npy"),   np.array(lr_path))
        np.save(os.path.join(data_dir, "target_img.npy"), raw_combined)
        np.save(os.path.join(data_dir, "xy_target.npy"),  raw_xy)
        np.save(os.path.join(data_dir, "xz_target.npy"),  raw_xz)
        np.save(os.path.join(data_dir, "truth_mom.npy"),
                np.array(true_momentum_list[0], dtype=np.float32))
        if verbose:
            print(f"Outputs saved to {data_dir}/")

        return {
            'img_path':     img_path,
            'dist_path':    dist_path,
            'mom_path':     mom_path,
            'explore_path': [False] * len(img_path),
            'lr_path':      lr_path,
            'data_dir':     data_dir,
            'plot_dir':     plot_dir,
            'elapsed':      elapsed,
        }

    # ── Standard single/multi-track branch ────────────────────────────────────

    # Validate / normalise true_momentum
    if n_tracks == 1:
        if not isinstance(true_momentum[0], (tuple, list, np.ndarray)):
            true_momentum = (true_momentum,)  # wrap to length-1 list for uniform handling
    true_momentum = [tuple(float(v) for v in m) for m in true_momentum]

    # Initial momenta
    if initial_momenta is None:
        initial_momenta = []
        for i in range(n_tracks):
            p, mag = random_momentum_init()
            if verbose:
                print(f"Random init track {i+1}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})  |p|={mag:.1f}")
            initial_momenta.append(p)
    else:
        initial_momenta = [tuple(float(v) for v in m) for m in initial_momenta]

    # Load LDM (skip if caller passed a pre-loaded generator)
    if generator is None:
        ldm = load_ldm(device)
        generator = DifferentiableLDMGenerator(
            ldm, device=str(device),
            ddim_steps_standard=ddim_steps_standard,
            ddim_steps_gradient=ddim_steps_gradient,
        )

    # Run inference
    img_path, dist_path, mom_path, explore_path, lr_path = _run_sgd_core(
        generator=generator,
        target_img=xy_tensor,
        initial_momenta=initial_momenta,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        lr_min=lr_min,
        gradient_clip=gradient_clip,
        min_distance=min_distance,
        optimizer_type=optimizer_type,
        avg_grad=avg_grad,
        avg_grad_batch_size=avg_grad_batch_size,
        fixed_z=fixed_z,
        device=str(device),
        verbose=verbose,
        loss_type=loss_type,
    )

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTotal time: {elapsed:.2f}s  ({elapsed / len(img_path):.2f}s per step)")

    # ── Save outputs ───────────────────────────────────────────────────────────
    if run_name is None:
        run_name = f"{n_tracks}track" + ("_l2" if loss_type == 'l2' else "")

    data_dir = os.path.join(output_dir, run_name, "data_files")
    plot_dir = os.path.join(output_dir, run_name, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    torch.save(torch.stack(img_path), os.path.join(data_dir, "img_path.pt"))
    np.save(os.path.join(data_dir, "mom_path.npy"),   np.array(mom_path))
    np.save(os.path.join(data_dir, "dist_path.npy"),  np.array(dist_path))
    np.save(os.path.join(data_dir, "std_path.npy"),   np.array(lr_path))
    np.save(os.path.join(data_dir, "target_img.npy"), raw_xy)
    np.save(os.path.join(data_dir, "truth_mom.npy"),  np.array(true_momentum))
    if verbose:
        print(f"Outputs saved to {data_dir}/")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if save_plots:
        _save_plots(
            img_path, dist_path, mom_path, lr_path,
            true_momentum, raw_xy, n_tracks, plot_dir, verbose,
        )

    return {
        'img_path':     img_path,
        'dist_path':    dist_path,
        'mom_path':     mom_path,
        'explore_path': explore_path,
        'lr_path':      lr_path,
        'data_dir':     data_dir,
        'plot_dir':     plot_dir,
        'elapsed':      elapsed,
    }


def _save_plots(img_path, dist_path, mom_path, lr_path,
                true_momentum, raw_img, n_tracks, plot_dir, verbose):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # truth_mom passed to plotting: single tuple for 1-track, pair-tuple for 2+
    truth_mom_plot = true_momentum[0] if n_tracks == 1 else tuple(true_momentum)

    try:
        from plot_inference import plot_sgd_images, create_sgd_gif, plot_sgd_results, plot_initial_final

        plot_sgd_images(
            img_path, dist_path, mom_path, lr_path, truth_mom_plot,
            save_path=os.path.join(plot_dir, 'sgd_images.png'),
            n_cols=5,
        )
        create_sgd_gif(
            img_path, dist_path, mom_path, lr_path,
            save_path=os.path.join(plot_dir, 'sgd_evolution.gif'),
            fps=4, stride=1,
        )
        plot_sgd_results(dist_path, mom_path, truth_mom_plot, save_dir=plot_dir)
        plot_initial_final(
            img_path, dist_path, mom_path,
            target_img=torch.tensor(raw_img),
            truth_mom=truth_mom_plot,
            save_path=os.path.join(plot_dir, 'initial_final.png'),
        )
        if verbose:
            print(f"Plots saved to {plot_dir}/")
    except ImportError as e:
        print(f"[Warning] Could not import plotting module: {e}")
        print("Skipping plots — data files are still saved.")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    DUAL_PROJ_DATASET    = "/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64_xy_xz"
    SINGLE_TRACK_DATASET = "/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64_v2/edep_val"
    PAIRS_PATH           = "/n/home11/zimani/proton64_analysis/double_momentum/angle_separated_pairs_with_emd.npy"

    parser = argparse.ArgumentParser(description="Gradient-guided LDM inference for proton momenta")

    parser.add_argument('--n_tracks', type=int, default=1, choices=[1, 2, 3],
                        help="Track hypothesis: 1, 2, or 3 (default: 1)")
    parser.add_argument('--dual_projection', action='store_true', default=False,
                        help="Optimise a single track against both xy and xz projections")
    parser.add_argument('--batch', type=int, default=0,
                        help="Batch index for dual-projection dataset (default: 0)")
    parser.add_argument('--event', type=int, default=0,
                        help="Event index within the batch for dual-projection dataset (default: 0)")
    parser.add_argument('--n_events', type=int, default=1,
                        help="Number of consecutive events to run inference on (default: 1)")
    parser.add_argument('--angle', type=float, default=None,
                        help="Target pair separation angle for double-track events (e.g. 16.0, 60.8)")
    parser.add_argument('--sample', type=str, default=None,
                        help="Path to a single-event .npy sample file")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset directory for single-track batch files (batch_X.npy, batch_mom_X.npy)")
    parser.add_argument('--output_dir', type=str, default='inference_results')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--no_plots', dest='save_plots', action='store_false', default=True)

    ## Inference hyperparameters
    parser.add_argument('--n_iterations', type=int, default=39) # 40 starting from zero for nice grid, probably overkill
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_min', type=float, default=0.001)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--min_distance', type=float, default=0.05)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'adam'])
    parser.add_argument('--avg_grad', action='store_true', default=True)
    parser.add_argument('--no_avg_grad', dest='avg_grad', action='store_false')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_reco', action='store_true', default=False,
                        help="Seed initial momentum from reco model")
    parser.add_argument('--loss', type=str, default='emd', choices=['emd', 'l2'],
                        help="Loss function: 'emd' (Sinkhorn, default) or 'l2' (MSE)")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Pre-load models once for all events ───────────────────────────────────
    ldm = load_ldm(device)
    generator = DifferentiableLDMGenerator(
        ldm, device=str(device),
        ddim_steps_standard=50,
        ddim_steps_gradient=10,
    )
    reco = load_reco_model(device) if args.use_reco else None

    # ── Dual-projection mode ───────────────────────────────────────────────────
    if args.dual_projection:
        # Track which batch file is currently loaded
        loaded_batch_idx = None
        xy_batch = xz_batch = mom_batch = None

        for ev_i in range(args.n_events):
            global_event = args.event + ev_i

            # Determine which file batch this event lives in
            if xy_batch is not None:
                events_in_batch = len(xy_batch)
                batch_idx  = args.batch + global_event // events_in_batch
                event_idx  = global_event % events_in_batch
            else:
                batch_idx  = args.batch
                event_idx  = global_event

            # Load a new batch file when the index changes
            if batch_idx != loaded_batch_idx:
                xy_batch  = np.load(os.path.join(DUAL_PROJ_DATASET, f"batch_{batch_idx}.npy"))
                xz_batch  = np.load(os.path.join(DUAL_PROJ_DATASET, f"batch_xz_{batch_idx}.npy"))
                mom_batch = np.load(os.path.join(DUAL_PROJ_DATASET, f"batch_mom_{batch_idx}.npy"))
                loaded_batch_idx = batch_idx
                # Recompute event_idx now that we know events_in_batch
                events_in_batch = len(xy_batch)
                batch_idx  = args.batch + global_event // events_in_batch
                event_idx  = global_event % events_in_batch
                if batch_idx != loaded_batch_idx:
                    xy_batch  = np.load(os.path.join(DUAL_PROJ_DATASET, f"batch_{batch_idx}.npy"))
                    xz_batch  = np.load(os.path.join(DUAL_PROJ_DATASET, f"batch_xz_{batch_idx}.npy"))
                    mom_batch = np.load(os.path.join(DUAL_PROJ_DATASET, f"batch_mom_{batch_idx}.npy"))
                    loaded_batch_idx = batch_idx

            raw_img       = xy_batch[event_idx].astype(np.float32)
            xz_raw        = xz_batch[event_idx].astype(np.float32)
            true_momentum = tuple(float(v) for v in mom_batch[event_idx])

            print(f"\n{'='*70}")
            print(f"Event {ev_i+1}/{args.n_events}  (batch={batch_idx}, event={event_idx})")
            print(f"True momentum: ({true_momentum[0]:.1f}, {true_momentum[1]:.1f}, {true_momentum[2]:.1f})")

            initial_momenta_cli = [reco_predict(reco, raw_img, device)] if reco else None

            run_name = args.run_name or (
                f"dual_proj_b{batch_idx}_e{event_idx}" + ("_l2" if args.loss == 'l2' else "")
            )

            run_inference(
                target_img=raw_img,
                true_momentum=true_momentum,
                dual_projection=True,
                xz_img=xz_raw,
                initial_momenta=initial_momenta_cli,
                n_iterations=args.n_iterations,
                learning_rate=args.lr,
                lr_min=args.lr_min,
                gradient_clip=args.gradient_clip,
                min_distance=args.min_distance,
                optimizer_type=args.optimizer,
                avg_grad_batch_size=args.batch_size,
                output_dir=args.output_dir,
                run_name=run_name,
                device=device,
                save_plots=args.save_plots,
                generator=generator,
                verbose=True,
                loss_type=args.loss,
            )

    # ── Multi-track mode ───────────────────────────────────────────────────────
    elif args.n_tracks >= 2:
        def _run_multitrack_event(ev_i, event_imgs, truth_moms):
            if args.n_tracks == 3 and len(event_imgs) == 2:
                event_imgs.append(event_imgs[0])
                truth_moms.append(truth_moms[0])
            raw_img       = sum(img.astype(np.float32) for img in event_imgs)
            true_momentum = truth_moms[:args.n_tracks]
            initial_momenta_cli = (
                [reco_predict(reco, img, device) for img in event_imgs[:args.n_tracks]]
                if reco else None
            )
            run_name = args.run_name or (
                f"{args.n_tracks}track_ev{ev_i}" + ("_l2" if args.loss == 'l2' else "")
            )
            if args.run_name and args.n_events > 1:
                run_name = f"{args.run_name}_ev{ev_i}"
            run_inference(
                target_img=raw_img,
                true_momentum=true_momentum,
                n_tracks=args.n_tracks,
                initial_momenta=initial_momenta_cli,
                n_iterations=args.n_iterations,
                learning_rate=args.lr,
                lr_min=args.lr_min,
                gradient_clip=args.gradient_clip,
                min_distance=args.min_distance,
                optimizer_type=args.optimizer,
                avg_grad=args.avg_grad,
                avg_grad_batch_size=args.batch_size,
                output_dir=args.output_dir,
                run_name=run_name,
                device=device,
                save_plots=args.save_plots,
                generator=generator,
                verbose=True,
                loss_type=args.loss,
            )

        if args.n_events > 1 and args.dataset:
            # Dataset directory: batch_X.npy shape (N, n_tracks, H, W)
            #                    batch_mom_X.npy  shape (N, n_tracks, 3)
            loaded_batch_idx = None
            img_batch = mom_batch_data = None

            for ev_i in range(args.n_events):
                global_event = args.event + ev_i
                if img_batch is not None:
                    events_in_batch = len(img_batch)
                    b_idx = args.batch + global_event // events_in_batch
                    e_idx = global_event % events_in_batch
                else:
                    b_idx = args.batch
                    e_idx = global_event
                if b_idx != loaded_batch_idx:
                    img_batch     = np.load(os.path.join(args.dataset, f"batch_{b_idx}.npy"))
                    mom_batch_data = np.load(os.path.join(args.dataset, f"batch_mom_{b_idx}.npy"))
                    loaded_batch_idx = b_idx
                    events_in_batch = len(img_batch)
                    b_idx = args.batch + global_event // events_in_batch
                    e_idx = global_event % events_in_batch
                    if b_idx != loaded_batch_idx:
                        img_batch     = np.load(os.path.join(args.dataset, f"batch_{b_idx}.npy"))
                        mom_batch_data = np.load(os.path.join(args.dataset, f"batch_mom_{b_idx}.npy"))
                        loaded_batch_idx = b_idx

                # img_batch[e_idx]: shape (n_tracks, H, W); mom_batch_data[e_idx]: (n_tracks, 3)
                event_imgs = [img_batch[e_idx][t] for t in range(args.n_tracks)]
                truth_moms = [tuple(float(v) for v in mom_batch_data[e_idx][t])
                              for t in range(args.n_tracks)]

                print(f"\n{'='*70}")
                print(f"Event {ev_i+1}/{args.n_events}  (batch={b_idx}, event={e_idx})")
                _run_multitrack_event(ev_i, event_imgs, truth_moms)

        else:
            # Pairs-file path (original behavior; also used when n_events==1)
            angle = args.angle if args.angle is not None else 16.0
            colinear = np.load(PAIRS_PATH, allow_pickle=True)
            matching_pairs = [co for co in colinear if np.abs(co['separation'] - angle) < 0.1]
            if not matching_pairs:
                raise RuntimeError(f"No pair found with separation ≈ {angle} deg in {PAIRS_PATH}")

            n_available = len(matching_pairs)
            start = args.event
            if start >= n_available:
                raise RuntimeError(
                    f"--event {start} is out of range: only {n_available} pairs at {angle} deg"
                )
            available_from_start = n_available - start
            if args.n_events > available_from_start:
                print(f"[Warning] Only {available_from_start} pairs available at {angle} deg "
                      f"from event {start}; running {available_from_start} events instead of {args.n_events}.")

            for ev_i, co in enumerate(matching_pairs[start:start + args.n_events]):
                print(f"\n{'='*70}")
                print(f"Event {ev_i+1}/{min(args.n_events, n_available)}  "
                      f"(separation={co['separation']:.2f} deg)")
                event_imgs = [co[k]['image'] for k in ['event1', 'event2']]
                truth_moms = [tuple(co[k].get('momentum', (0., 0., 0.)))
                              for k in ['event1', 'event2']]
                _run_multitrack_event(ev_i, event_imgs, truth_moms)

    # ── Single-track mode ──────────────────────────────────────────────────────
    else:
        # Build an iterable of (raw_img, true_momentum) tuples
        def _iter_single_events():
            if args.dataset:
                # Dataset directory: batch_X.npy + batch_mom_X.npy
                loaded_batch_idx = None
                img_batch = mom_batch = None
                for ev_i in range(args.n_events):
                    global_event = args.event + ev_i
                    if img_batch is not None:
                        events_in_batch = len(img_batch)
                        b_idx = args.batch + global_event // events_in_batch
                        e_idx = global_event % events_in_batch
                    else:
                        b_idx = args.batch
                        e_idx = global_event
                    if b_idx != loaded_batch_idx:
                        img_batch = np.load(os.path.join(args.dataset, f"batch_{b_idx}.npy"))
                        mom_batch = np.load(os.path.join(args.dataset, f"batch_mom_{b_idx}.npy"))
                        loaded_batch_idx = b_idx
                        events_in_batch = len(img_batch)
                        b_idx = args.batch + global_event // events_in_batch
                        e_idx = global_event % events_in_batch
                        if b_idx != loaded_batch_idx:
                            img_batch = np.load(os.path.join(args.dataset, f"batch_{b_idx}.npy"))
                            mom_batch = np.load(os.path.join(args.dataset, f"batch_mom_{b_idx}.npy"))
                            loaded_batch_idx = b_idx
                    yield (img_batch[e_idx].astype(np.float32),
                           [tuple(float(v) for v in mom_batch[e_idx])],
                           b_idx, e_idx)
            elif args.sample:
                data = np.load(args.sample, allow_pickle=True)
                item = data.item() if data.ndim == 0 else {'image': data}
                raw_img = item['image'].astype(np.float32)
                true_momentum = [item.get('momentum', (0., 0., 0.))]
                for _ in range(args.n_events):
                    yield raw_img, true_momentum, None, None
            else:
                # Default: single-track validation dataset; --batch / --event apply
                loaded_batch_idx = None
                img_batch = mom_batch = None
                for ev_i in range(args.n_events):
                    global_event = args.event + ev_i
                    if img_batch is not None:
                        events_in_batch = len(img_batch)
                        b_idx = args.batch + global_event // events_in_batch
                        e_idx = global_event % events_in_batch
                    else:
                        b_idx = args.batch
                        e_idx = global_event
                    if b_idx != loaded_batch_idx:
                        img_batch = np.load(os.path.join(SINGLE_TRACK_DATASET, f"batch_{b_idx}.npy"))
                        mom_batch = np.load(os.path.join(SINGLE_TRACK_DATASET, f"batch_mom_{b_idx}.npy"))
                        loaded_batch_idx = b_idx
                        events_in_batch = len(img_batch)
                        b_idx = args.batch + global_event // events_in_batch
                        e_idx = global_event % events_in_batch
                        if b_idx != loaded_batch_idx:
                            img_batch = np.load(os.path.join(SINGLE_TRACK_DATASET, f"batch_{b_idx}.npy"))
                            mom_batch = np.load(os.path.join(SINGLE_TRACK_DATASET, f"batch_mom_{b_idx}.npy"))
                            loaded_batch_idx = b_idx
                    yield (img_batch[e_idx].astype(np.float32),
                           [tuple(float(v) for v in mom_batch[e_idx])],
                           b_idx, e_idx)

        for ev_i, (raw_img, true_momentum, b_idx, e_idx) in enumerate(_iter_single_events()):
            print(f"\n{'='*70}")
            if b_idx is not None:
                print(f"Event {ev_i+1}/{args.n_events}  (batch={b_idx}, event={e_idx})")
            else:
                print(f"Event {ev_i+1}/{args.n_events}")

            initial_momenta_cli = [reco_predict(reco, raw_img, device)] if reco else None

            if args.run_name:
                run_name = args.run_name if args.n_events == 1 else f"{args.run_name}_ev{ev_i}"
            elif b_idx is not None:
                run_name = f"single_1track_b{b_idx}_e{e_idx}" + ("_l2" if args.loss == 'l2' else "")
            else:
                run_name = f"single_1track_ev{ev_i}" + ("_l2" if args.loss == 'l2' else "")

            run_inference(
                target_img=raw_img,
                true_momentum=true_momentum,
                n_tracks=1,
                initial_momenta=initial_momenta_cli,
                n_iterations=args.n_iterations,
                learning_rate=args.lr,
                lr_min=args.lr_min,
                gradient_clip=args.gradient_clip,
                min_distance=args.min_distance,
                optimizer_type=args.optimizer,
                avg_grad=args.avg_grad,
                avg_grad_batch_size=args.batch_size,
                output_dir=args.output_dir,
                run_name=run_name,
                device=device,
                save_plots=args.save_plots,
                generator=generator,
                verbose=True,
                loss_type=args.loss,
            )
