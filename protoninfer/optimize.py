"""
Gradient-guided LDM inference: recover proton momenta from a detector image.

infer()          config-driven entry point (InferenceConfig; picks the initial guess)
run_inference()  same engine with every hyperparameter as a keyword argument
"""
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from .config import InferenceConfig
from .initial_guess import convex_hull_initial_guess, perturb_momentum, random_momentum_init
from .losses import get_loss_fn
from .model import DifferentiableLDMGenerator, load_ldm
from .plotting import save_event_plots


# ── Core SGD (generalized to N tracks) ────────────────────────────────────────

def _renoise_fraction(renoise, renoise_min, schedule, iteration, n_iterations,
                      lr, learning_rate, lr_min, step=None, step_ref=50.0):
    """
    Renoise fraction for this iteration, decaying from `renoise` to `renoise_min`.

    For schedule='step', `step` is the momentum displacement (MeV) of the previous
    gradient update; the fraction scales linearly with it and reaches `renoise` at
    `step_ref`.  step=None (no update taken yet) gives the full `renoise`.
    """
    if schedule in ('constant', 'fixed'):
        return renoise
    if schedule == 'lr':
        # Map lr ∈ [lr_min, learning_rate] linearly onto [renoise_min, renoise]
        span = learning_rate - lr_min
        w = (lr - lr_min) / span if span > 0 else 1.0
    elif schedule == 'step':
        # Big step → previous image is far from the new momentum → noise it further
        w = 1.0 if step is None else step / step_ref
    elif schedule == 'linear':
        w = 1.0 - iteration / max(1, n_iterations - 1)
    elif schedule == 'cosine':
        w = 0.5 * (1.0 + np.cos(np.pi * iteration / max(1, n_iterations - 1)))
    else:
        raise ValueError(f"Unknown renoise_schedule: {schedule!r}")
    w = min(max(w, 0.0), 1.0)
    return renoise_min + (renoise - renoise_min) * w


def _renoise_ddim_steps(t_renoise, t_max, full_steps, min_steps):
    """
    DDIM steps for a renoise chain t_renoise → 0 that keeps the full chain's timestep
    spacing: the full T_max → 0 chain has full_steps-1 intervals, so a chain starting
    at t_renoise gets ceil((full_steps-1) · t_renoise / t_max) of them, clipped to
    [min_steps, full_steps] steps.
    """
    n_intervals = int(np.ceil((full_steps - 1) * t_renoise / t_max))
    return min(full_steps, max(min_steps, n_intervals + 1))


def _momentum_step(prev_mom, cur_mom):
    """Largest per-track |Δp| (MeV) between two mom_path entries (single or multi-track)."""
    delta = np.asarray(cur_mom, dtype=float).reshape(-1, 3) - np.asarray(prev_mom, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(delta, axis=1).max())


def _encode_latent(generator, img, device):
    """Clean LDM latent [1, C, h, w] of a (H, W) image, used as z_clean for renoising."""
    with torch.no_grad():
        x = img.to(device).reshape(1, 1, *img.shape[-2:])
        return generator.model.get_first_stage_encoding(generator.model.encode_first_stage(x))


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
    loss_fn=None,           # callable override; if set, takes precedence over loss_type
    reparam_pmag=False,     # if True, optimise (px, py, |p|) and derive pz; default: direct (px, py, pz)
    newton_pz=False,        # if True, scale pz (or pmag) grad by diagonal Hessian inverse before stepping
    fix_pz=False,           # if True, hold pz at its initial value and optimise only (px, py)
    renoise=0.0,            # fraction [0,1]; noise previous image to T=renoise*T_max and start DDIM there
    renoise_truth_init=False,        # if True, seed z_clean from the truth image on the first step
    renoise_truth_every_step=False,  # if True, always renoise from truth (never update z_clean)
    renoise_schedule='constant',     # 'constant', 'lr' (track the LR schedule), 'step' (track |Δp|), 'linear', 'cosine'
    renoise_min=0.0,                 # final renoise fraction for non-constant schedules
    renoise_step_ref=50.0,           # 'step' schedule: |Δp| (MeV) at which the fraction reaches `renoise`
    renoise_scale_steps=False,       # if True, scale the gradient DDIM steps with t_renoise / T_max
    renoise_min_steps=2,             # floor on the scaled DDIM steps
    return_renoise_path=False,       # if True, also return the per-step t_renoise list
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
    renoise_path: list of int t_renoise per step (only if return_renoise_path=True)
    """
    if loss_fn is None:
        loss_fn = get_loss_fn(loss_type)
    if fix_pz and (reparam_pmag or newton_pz):
        raise ValueError("fix_pz cannot be combined with reparam_pmag or newton_pz")

    n_tracks = len(initial_momenta)
    SCALE    = 500.0
    target_img = target_img.to(device)

    t_renoise = None
    z_clean   = None
    t_max     = generator.model.num_timesteps - 1

    renoise_steps = None

    def _renoise_frac(iteration, lr, step):
        return _renoise_fraction(renoise, renoise_min, renoise_schedule, iteration, n_iterations,
                                 lr, learning_rate, lr_min, step, renoise_step_ref)

    def _steps_for(t):
        if not renoise_scale_steps:
            return None
        return _renoise_ddim_steps(t, t_max, generator.ddim_steps_gradient, renoise_min_steps)

    if renoise:
        t_renoise = max(1, int(renoise * t_max))
        if verbose:
            sched_str = ("" if renoise_schedule in ('constant', 'fixed') else
                         f", schedule={renoise_schedule} → renoise_min={renoise_min:.2f}")
            if renoise_schedule == 'step':
                sched_str += f", step_ref={renoise_step_ref:.1f} MeV"
            if renoise_scale_steps:
                sched_str += (f", ddim_steps={_steps_for(t_renoise)}/{generator.ddim_steps_gradient}"
                              f" (min {renoise_min_steps})")
            print(f"[renoise] enabled: t_renoise={t_renoise} "
                  f"(renoise={renoise:.2f} × {t_max}{sched_str})")
        if renoise_truth_init or renoise_truth_every_step:
            with torch.no_grad():
                x_truth = target_img
                if x_truth.ndim == 2:
                    x_truth = x_truth.unsqueeze(0).unsqueeze(0)
                z_clean = generator.model.get_first_stage_encoding(
                    generator.model.encode_first_stage(x_truth)
                )
            if verbose:
                tag = "every step" if renoise_truth_every_step else "first step only"
                print(f"[renoise] z_clean initialised from truth image ({tag})")

    # Build N×3 learnable parameter groups: [px/SCALE, py/SCALE, |p|/SCALE]
    # Optimise |p| (total momentum magnitude) instead of pz.  pz is derived as
    # sign_pz * sqrt(|p|^2 - px^2 - py^2), mirroring the reco model's strategy.
    # This gives the optimizer a direct handle on track length (the main pz signal
    # in a single 2D projection), which is otherwise invisible to d(EMD)/d(pz).
    params_list = []
    sign_pz_list = []
    for init_mom in initial_momenta:
        px0, py0, pz0 = float(init_mom[0]), float(init_mom[1]), float(init_mom[2])
        if reparam_pmag:
            p3 = np.sqrt(px0**2 + py0**2 + pz0**2)  # |p|
            sign_pz_list.append(1.0 if pz0 >= 0 else -1.0)
        else:
            p3 = pz0
            sign_pz_list.append(1.0)  # unused in direct mode
        group = [
            torch.tensor(px0 / SCALE, dtype=torch.float32, requires_grad=True, device=device),
            torch.tensor(py0 / SCALE, dtype=torch.float32, requires_grad=True, device=device),
            torch.tensor(p3  / SCALE, dtype=torch.float32, requires_grad=not fix_pz, device=device),
        ]
        params_list.append(group)

    if reparam_pmag:
        def _pz_derived(group, sign_pz):
            px_n, py_n, pmag_n = group
            return sign_pz * torch.sqrt(torch.clamp(pmag_n**2 - px_n**2 - py_n**2, min=1e-8))
    else:
        def _pz_derived(group, sign_pz):
            return group[2]

    def _get_pz_np(i, g):
        if reparam_pmag:
            return sign_pz_list[i] * np.sqrt(max(g[2].item()**2 - g[0].item()**2 - g[1].item()**2, 0.0))
        return g[2].item()

    def _make_opt(params):
        params = [p for p in params if p.requires_grad]   # drops pz when fix_pz
        if optimizer_type.lower() in ('adam', 'adamw'):
            return torch.optim.AdamW(params, lr=learning_rate)
        return torch.optim.SGD(params, lr=learning_rate)

    optimizers = [_make_opt(p) for p in params_list]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=n_iterations, eta_min=lr_min)
        for o in optimizers
    ]

    def _current_mom():
        moms = []
        for i, group in enumerate(params_list):
            px_n = group[0].item()
            py_n = group[1].item()
            pz_n = _get_pz_np(i, group)
            moms.append((px_n * SCALE, py_n * SCALE, pz_n * SCALE))
        moms = tuple(moms)
        return moms[0] if n_tracks == 1 else moms

    # Record starting state without grad
    with torch.no_grad():
        init_imgs = [
            generator(g[0].item() * SCALE, g[1].item() * SCALE, _get_pz_np(i, g) * SCALE)
            for i, g in enumerate(params_list)
        ]
        init_combined = sum(init_imgs)
        init_loss = loss_fn(init_combined, target_img).item()

    img_path    = [init_combined.cpu()]
    dist_path   = [init_loss]
    mom_path    = [_current_mom()]
    explore_path = [False]
    lr_path     = []
    renoise_path = []   # t_renoise used at each SGD step (None when renoise disabled)

    best_loss    = init_loss
    best_momenta = mom_path[0]

    if verbose:
        for i, g in enumerate(params_list):
            pz_n = _get_pz_np(i, g)
            pmag_str = f"  |p|={g[2].item()*SCALE:.2f}" if reparam_pmag else ""
            print(f"Track {i+1} init: ({g[0].item()*SCALE:.2f}, {g[1].item()*SCALE:.2f}, {pz_n*SCALE:.2f}){pmag_str}")
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

        # |Δp| of the update just taken: mom_path[-1] is the momentum before it
        step = _momentum_step(mom_path[-1], _current_mom()) if iteration > 0 else None
        if renoise:
            t_renoise = max(1, int(_renoise_frac(iteration, current_lr, step) * t_max))
            renoise_steps = _steps_for(t_renoise)

        for opt in optimizers:
            opt.zero_grad()

        if avg_grad:
            # Generate one batch per track, then sum across tracks element-wise
            batches = [
                generator(g[0] * SCALE, g[1] * SCALE,
                          _pz_derived(g, sign_pz_list[t]) * SCALE,
                          batch_size=avg_grad_batch_size,
                          z_clean=z_clean, t_renoise=t_renoise,
                          renoise_steps=renoise_steps)
                for t, g in enumerate(params_list)
            ]
            losses = torch.stack([
                loss_fn(
                    sum(batches[t][i] for t in range(n_tracks)), target_img
                )
                for i in range(avg_grad_batch_size)
            ])
            if newton_pz:
                all_params_flat = [p for group in params_list for p in group]
                grads_all = torch.autograd.grad(
                    losses.mean(), all_params_flat, create_graph=True
                )
                for t, group in enumerate(params_list):
                    for j, p in enumerate(group):
                        g = grads_all[t * 3 + j]
                        if j == 2:
                            (h,) = torch.autograd.grad(g, p, retain_graph=True,
                                                       allow_unused=True)
                            # h is None when loss is locally linear in p (H=0); fall back to plain grad
                            p.grad = (g / (h.abs() + 1e-6)).detach() if h is not None else g.detach()
                        else:
                            p.grad = g.detach()
            else:
                losses.mean().backward()
            best_idx = losses.detach().argmin().item()
            combined = sum(batches[t][best_idx] for t in range(n_tracks)).detach().cpu()
            current_loss = losses[best_idx].item()
        else:
            gens = [
                generator(g[0] * SCALE, g[1] * SCALE,
                          _pz_derived(g, sign_pz_list[t]) * SCALE,
                          fixed_z=fixed_z,
                          z_clean=z_clean, t_renoise=t_renoise,
                          renoise_steps=renoise_steps)
                for t, g in enumerate(params_list)
            ]
            combined = sum(gens)
            loss     = loss_fn(combined, target_img)
            if newton_pz:
                all_params_flat = [p for group in params_list for p in group]
                grads_all = torch.autograd.grad(loss, all_params_flat, create_graph=True)
                for t, group in enumerate(params_list):
                    for j, p in enumerate(group):
                        g = grads_all[t * 3 + j]
                        if j == 2:
                            (h,) = torch.autograd.grad(g, p, retain_graph=True,
                                                       allow_unused=True)
                            # h is None when loss is locally linear in p (H=0); fall back to plain grad
                            p.grad = (g / (h.abs() + 1e-6)).detach() if h is not None else g.detach()
                        else:
                            p.grad = g.detach()
            else:
                loss.backward()
            current_loss = loss.item()
            combined = combined.detach().cpu()

        # Encode combined image to update z_clean for the next iteration
        if t_renoise is not None and not renoise_truth_every_step:
            with torch.no_grad():
                x_prev = combined.to(device).unsqueeze(0).unsqueeze(0)
                z_clean = generator.model.get_first_stage_encoding(
                    generator.model.encode_first_stage(x_prev)
                )

        # Record the momentum that produced current_loss, before the optimizer moves it
        current_mom = _current_mom()

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

        if current_loss < best_loss:
            best_loss    = current_loss
            best_momenta = current_mom

        img_path.append(combined)
        dist_path.append(current_loss)
        mom_path.append(current_mom)
        explore_path.append(False)
        lr_path.append(current_lr)
        renoise_path.append(t_renoise)

        if verbose and ((iteration + 1) % 10 == 0 or iteration == 0):
            grad_info = "  ".join(
                f"|g{i+1}|={torch.sqrt(sum(p.grad.data.norm()**2 for p in g if p.grad is not None)).item():.3f}"
                for i, g in enumerate(params_list)
            )
            t_info = f" | T={t_renoise}" if renoise else ""
            if renoise_steps is not None and z_clean is not None:
                t_info += f" ({renoise_steps} DDIM)"
            if renoise and step is not None:
                t_info += f" (|Δp|={step:.1f})"
            print(f"Iter {iteration+1:3d}: Loss={current_loss:.6f} | Best={best_loss:.6f} | "
                  f"LR={current_lr:.4f}{t_info} | {grad_info}")

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

    if return_renoise_path:
        return img_path, dist_path, mom_path, explore_path, lr_path, renoise_path
    return img_path, dist_path, mom_path, explore_path, lr_path


# ── Dual-projection SGD (xy + xz views of the same proton) ────────────────────

def _run_dual_proj_sgd(
    xy_target,
    xz_target,
    initial_momentum,
    generator,
    n_iterations=40,
    learning_rate=0.1,
    lr_min=0.001,
    gradient_clip=1.0,
    min_distance=0.05,
    optimizer_type='SGD',
    avg_grad_batch_size=32,
    device='cuda',
    loss_type='emd',
    loss_fn=None,           # callable override; if set, takes precedence over loss_type
    reparam_pmag=False,     # if True, optimise (px, py, |p|) and derive pz; default: direct (px, py, pz)
    fix_pz=False,           # if True, hold pz at its initial value and optimise only (px, py)
    renoise=0.0,            # fraction [0,1]; see _run_sgd_core
    renoise_truth_init=False,
    renoise_truth_every_step=False,
    renoise_schedule='constant',
    renoise_min=0.0,
    renoise_step_ref=50.0,
    renoise_scale_steps=False,
    renoise_min_steps=2,
    verbose=True,
):
    """
    Single (px, py, pz) optimized against both xy and xz projections simultaneously.

    The xy path calls generator(px, py, pz); the xz path calls generator(px, pz, py),
    swapping py/pz so xz gradients accumulate on the correct parameters.
    Both paths share a single backward pass.

    With renoise, each view keeps its own z_clean, encoded from that view's image
    in the best batch sample of the previous step (or from its target, for the truth options).

    Returns img_path, dist_path, mom_path, lr_path, renoise_path, best_loss, best_mom.
    """
    if loss_fn is None:
        loss_fn = get_loss_fn(loss_type)
    if fix_pz and reparam_pmag:
        raise ValueError("fix_pz cannot be combined with reparam_pmag")

    SCALE = 500.0
    device = torch.device(device) if isinstance(device, str) else device
    xy_target = xy_target.to(device)
    xz_target = xz_target.to(device)

    px0 = float(initial_momentum[0])
    py0 = float(initial_momentum[1])
    pz0 = float(initial_momentum[2])

    px = torch.tensor(px0 / SCALE, dtype=torch.float32, requires_grad=True, device=device)
    py = torch.tensor(py0 / SCALE, dtype=torch.float32, requires_grad=True, device=device)

    if reparam_pmag:
        pmag0   = np.sqrt(px0**2 + py0**2 + pz0**2)
        sign_pz = 1.0 if pz0 >= 0 else -1.0
        pz_param = torch.tensor(pmag0 / SCALE, dtype=torch.float32, requires_grad=True, device=device)
        def _pz_n():
            return sign_pz * torch.sqrt(torch.clamp(pz_param**2 - px**2 - py**2, min=1e-8))
        def _pz_np():
            return sign_pz * np.sqrt(max(pz_param.item()**2 - px.item()**2 - py.item()**2, 0.0))
    else:
        pz_param = torch.tensor(pz0 / SCALE, dtype=torch.float32, requires_grad=not fix_pz, device=device)
        def _pz_n():
            return pz_param
        def _pz_np():
            return pz_param.item()
    params = [px, py, pz_param]
    trainable = [p for p in params if p.requires_grad]   # drops pz when fix_pz

    if optimizer_type.lower() in ('adam', 'adamw'):
        opt = torch.optim.AdamW(trainable, lr=learning_rate)
    else:
        opt = torch.optim.SGD(trainable, lr=learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations, eta_min=lr_min)

    def _mom():
        return (px.item() * SCALE, py.item() * SCALE, _pz_np() * SCALE)

    with torch.no_grad():
        _px_v = px.item() * SCALE
        _py_v = py.item() * SCALE
        _pz_v = _pz_np() * SCALE
        xy0 = generator(_px_v, _py_v, _pz_v)
        xz0 = generator(_px_v, _pz_v, _py_v)
        init_loss = (loss_fn(xy0, xy_target) + loss_fn(xz0, xz_target)).item()

    t_renoise     = None
    renoise_steps = None
    z_xy = z_xz   = None
    t_max         = generator.model.num_timesteps - 1

    def _steps_for(t):
        if not renoise_scale_steps:
            return None
        return _renoise_ddim_steps(t, t_max, generator.ddim_steps_gradient, renoise_min_steps)

    if renoise:
        if verbose:
            t0 = max(1, int(renoise * t_max))
            sched_str = ("" if renoise_schedule in ('constant', 'fixed') else
                         f", schedule={renoise_schedule} → renoise_min={renoise_min:.2f}")
            if renoise_scale_steps:
                sched_str += (f", ddim_steps={_steps_for(t0)}/{generator.ddim_steps_gradient}"
                              f" (min {renoise_min_steps})")
            print(f"[renoise] enabled: t_renoise={t0} "
                  f"(renoise={renoise:.2f} × {t_max}{sched_str})")
        if renoise_truth_init or renoise_truth_every_step:
            z_xy = _encode_latent(generator, xy_target, device)
            z_xz = _encode_latent(generator, xz_target, device)

    img_path     = [(xy0 + xz0).cpu()]
    dist_path    = [init_loss]
    mom_path     = [_mom()]
    lr_path      = []
    renoise_path = []
    best_loss    = init_loss
    best_mom     = mom_path[0]

    for iteration in range(n_iterations):
        if opt.param_groups[0]['lr'] < lr_min:
            for g in opt.param_groups:
                g['lr'] = lr_min
        current_lr = opt.param_groups[0]['lr']

        if renoise:
            # |Δp| of the update just taken: mom_path[-1] is the momentum before it
            step = _momentum_step(mom_path[-1], _mom()) if iteration > 0 else None
            t_renoise = max(1, int(_renoise_fraction(
                renoise, renoise_min, renoise_schedule, iteration, n_iterations,
                current_lr, learning_rate, lr_min, step, renoise_step_ref) * t_max))
            renoise_steps = _steps_for(t_renoise)

        opt.zero_grad()

        pz_t     = _pz_n() * SCALE
        xy_batch = generator(px * SCALE, py * SCALE, pz_t,
                             batch_size=avg_grad_batch_size,
                             z_clean=z_xy, t_renoise=t_renoise,
                             renoise_steps=renoise_steps)
        xz_batch = generator(px * SCALE, pz_t, py * SCALE,
                             batch_size=avg_grad_batch_size,
                             z_clean=z_xz, t_renoise=t_renoise,
                             renoise_steps=renoise_steps)

        losses = torch.stack([
            loss_fn(xy_batch[i], xy_target) + loss_fn(xz_batch[i], xz_target)
            for i in range(avg_grad_batch_size)
        ])
        losses.mean().backward()

        best_idx     = losses.detach().argmin().item()
        combined_img = (xy_batch[best_idx] + xz_batch[best_idx]).detach().cpu()
        current_loss = losses[best_idx].item()

        if t_renoise is not None and not renoise_truth_every_step:
            z_xy = _encode_latent(generator, xy_batch[best_idx].detach(), device)
            z_xz = _encode_latent(generator, xz_batch[best_idx].detach(), device)

        # Record the momentum that produced current_loss, before the optimizer moves it
        current_mom = _mom()

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

        if current_loss < best_loss:
            best_loss = current_loss
            best_mom  = current_mom

        img_path.append(combined_img)
        dist_path.append(current_loss)
        mom_path.append(current_mom)
        lr_path.append(current_lr)
        renoise_path.append(t_renoise)

        if best_loss < min_distance:
            break

    return img_path, dist_path, mom_path, lr_path, renoise_path, best_loss, best_mom


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
    n_iterations=40,
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
    loss_type='emd', # string name resolved via get_loss_fn; ignored when loss_fn is set
    loss_fn=None,    # callable (generated_img, target_img) -> scalar; overrides loss_type
    reparam_pmag=False,  # if True, optimise (px, py, |p|) and derive pz; default: direct (px, py, pz)
    newton_pz=False,     # if True, scale pz (or pmag) grad by diagonal Hessian inverse before stepping
    fix_pz=False,        # if True, hold pz at its initial value and optimise only (px, py)
    init_pz=None,       # if set (float, MeV), override the pz component of every track's initial guess
    renoise=0.0,         # fraction [0,1]; noise previous image to T=renoise*T_max and start DDIM there
    renoise_truth_init=False,       # if True, seed first step's z_clean from the truth image
    renoise_truth_every_step=False, # if True, always renoise from truth (never update z_clean)
    renoise_schedule='constant',    # 'constant', 'lr' (track the LR schedule), 'step' (track |Δp|), 'linear', 'cosine'
    renoise_min=0.0,                # final renoise fraction for non-constant schedules
    renoise_step_ref=50.0,          # 'step' schedule: |Δp| (MeV) at which the fraction reaches `renoise`
    renoise_scale_steps=False,      # if True, scale the gradient DDIM steps with t_renoise / T_max
    renoise_min_steps=2,            # floor on the scaled DDIM steps
):
    """
    Run gradient-guided LDM inference to recover proton momenta from a detector image.

    Parameters
    ----------
    target_img : np.ndarray or torch.Tensor
        2-D detector image.  In dual_projection mode this is the xy projection.
    true_momentum : tuple or list of tuples or None
        Ground-truth momenta.  Single track: (px, py, pz).
        Multi-track: ((px1,py1,pz1), ...).  None for real data (per-event plots are skipped).
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
        'SGD' or 'adamw'.
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
        Name of the loss function (resolved via get_loss_fn).  Supported values:
        'emd'/'w1'/'sinkhorn' (default), 'unbal_emd'/'uemd' (unbalanced W1),
        'w2'/'sinkhorn_p2', 'energy',
        'gaussian_mmd'/'gmmd', 'laplacian_mmd'/'lmmd', 'l2'/'mse'.
        Ignored when loss_fn is provided.
    loss_fn : callable or None
        Custom loss function with signature (generated_img, target_img) -> scalar tensor.
        When provided, overrides loss_type entirely.
    newton_pz : bool
        If True, replace the pz (or |p| when reparam_pmag=True) gradient with a
        diagonal Newton step: g / (|d²L/dpz²| + 1e-6).  This amplifies weak pz
        gradients by the local curvature, helping the optimizer move in the pz
        direction even when d(loss)/d(pz) is small.  Costs one extra backward pass
        per iteration.  Consider reducing avg_grad_batch_size if memory is tight.
    renoise : float
        Initial re-noise fraction [0, 1]; t_renoise = renoise × T_max.  0 disables.
    renoise_schedule : str
        How the re-noise fraction evolves over iterations, decaying from
        `renoise` to `renoise_min`:
        'constant' (default) / 'fixed' — fixed at `renoise`;
        'lr'     — proportional to the current learning rate, mapping
                   [lr_min, learning_rate] onto [renoise_min, renoise];
        'step'   — proportional to the momentum displacement |Δp| of the previous
                   update (largest over tracks), mapping [0, renoise_step_ref] onto
                   [renoise_min, renoise] and saturating above; the first step uses
                   `renoise`.  Unlike 'lr', this sees the actual step after AdamW
                   normalisation and gradient clipping;
        'linear' / 'cosine' — independent linear / cosine decay over n_iterations.
    renoise_min : float
        Final re-noise fraction for non-constant schedules.
    renoise_step_ref : float
        For renoise_schedule='step': |Δp| in MeV that maps to the full `renoise`.
    renoise_scale_steps : bool
        If True, a renoised chain (t_renoise → 0) uses fewer gradient DDIM steps,
        keeping the timestep spacing of the full ddim_steps_gradient chain:
        about ddim_steps_gradient × t_renoise / T_max.  Re-evaluated every step,
        so it follows the renoise schedule.
    renoise_min_steps : int
        Floor on the scaled DDIM steps (>= 2; 2 is a single DDIM update).

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

    # Step 0 is saved as the initial image, so the loop runs n_iterations-1 SGD steps.
    n_iterations = n_iterations - 1

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
        if true_momentum is not None:
            true_momentum_list = [tuple(float(v) for v in true_momentum)]

        init_mom = initial_momenta[0] if initial_momenta else None
        if init_mom is None:
            init_mom, mag = random_momentum_init()
            if verbose:
                print(f"Random init: ({init_mom[0]:.1f}, {init_mom[1]:.1f}, {init_mom[2]:.1f})  |p|={mag:.1f}")
        else:
            init_mom = tuple(float(v) for v in init_mom)

        if init_pz is not None:
            init_mom = (init_mom[0], init_mom[1], float(init_pz))
            if verbose:
                print(f"[init_pz] pz overridden to {float(init_pz):.1f} MeV  "
                      f"→ init ({init_mom[0]:.1f}, {init_mom[1]:.1f}, {init_mom[2]:.1f})")

        if generator is None:
            ldm = load_ldm(device)
            generator = DifferentiableLDMGenerator(
                ldm, device=str(device),
                ddim_steps_standard=ddim_steps_standard,
                ddim_steps_gradient=ddim_steps_gradient,
            )

        img_path, dist_path, mom_path, lr_path, renoise_path, best_loss, best_mom = _run_dual_proj_sgd(
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
            loss_fn=loss_fn,
            reparam_pmag=reparam_pmag,
            fix_pz=fix_pz,
            renoise=renoise,
            renoise_truth_init=renoise_truth_init,
            renoise_truth_every_step=renoise_truth_every_step,
            renoise_schedule=renoise_schedule,
            renoise_min=renoise_min,
            renoise_step_ref=renoise_step_ref,
            renoise_scale_steps=renoise_scale_steps,
            renoise_min_steps=renoise_min_steps,
            verbose=verbose,
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
        if true_momentum is not None:
            np.save(os.path.join(data_dir, "truth_mom.npy"),
                    np.array(true_momentum_list[0], dtype=np.float32))
        if renoise:
            np.save(os.path.join(data_dir, "renoise_path.npy"), np.array(renoise_path))
        if verbose:
            print(f"Outputs saved to {data_dir}/")

        return {
            'img_path':     img_path,
            'dist_path':    dist_path,
            'mom_path':     mom_path,
            'best_mom':     best_mom,
            'best_loss':    best_loss,
            'explore_path': [False] * len(img_path),
            'lr_path':      lr_path,
            'renoise_path': renoise_path,
            'data_dir':     data_dir,
            'plot_dir':     plot_dir,
            'elapsed':      elapsed,
        }

    # ── Standard single/multi-track branch ────────────────────────────────────

    # Validate / normalise true_momentum
    if true_momentum is not None:
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

    if init_pz is not None:
        initial_momenta = [(px, py, float(init_pz)) for px, py, _ in initial_momenta]
        if verbose:
            for i, (px, py, pz) in enumerate(initial_momenta):
                print(f"[init_pz] track {i+1} pz overridden to {pz:.1f} MeV  "
                      f"→ init ({px:.1f}, {py:.1f}, {pz:.1f})")

    # Load LDM (skip if caller passed a pre-loaded generator)
    if generator is None:
        ldm = load_ldm(device)
        generator = DifferentiableLDMGenerator(
            ldm, device=str(device),
            ddim_steps_standard=ddim_steps_standard,
            ddim_steps_gradient=ddim_steps_gradient,
        )

    # Run inference
    img_path, dist_path, mom_path, explore_path, lr_path, renoise_path = _run_sgd_core(
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
        loss_fn=loss_fn,
        reparam_pmag=reparam_pmag,
        newton_pz=newton_pz,
        fix_pz=fix_pz,
        renoise=renoise,
        renoise_truth_init=renoise_truth_init,
        renoise_truth_every_step=renoise_truth_every_step,
        renoise_schedule=renoise_schedule,
        renoise_min=renoise_min,
        renoise_step_ref=renoise_step_ref,
        renoise_scale_steps=renoise_scale_steps,
        renoise_min_steps=renoise_min_steps,
        return_renoise_path=True,
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
    if true_momentum is not None:
        np.save(os.path.join(data_dir, "truth_mom.npy"),  np.array(true_momentum))
    if renoise:
        np.save(os.path.join(data_dir, "renoise_path.npy"), np.array(renoise_path))
    if verbose:
        print(f"Outputs saved to {data_dir}/")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if save_plots and true_momentum is None:
        print("[plots] skipped: per-event plots compare against true_momentum, which is None")
    elif save_plots:
        save_event_plots(
            img_path, dist_path, mom_path, lr_path,
            true_momentum, raw_xy, n_tracks, plot_dir, verbose,
        )

    return {
        'img_path':     img_path,
        'dist_path':    dist_path,
        'mom_path':     mom_path,
        'explore_path': explore_path,
        'lr_path':      lr_path,
        'renoise_path': renoise_path,
        'data_dir':     data_dir,
        'plot_dir':     plot_dir,
        'elapsed':      elapsed,
    }



# ── Config-driven entry point ──────────────────────────────────────────────────

def infer(
    target_img,
    true_momentum=None,
    cfg=None,
    *,
    n_tracks=1,
    initial_momenta=None,
    xz_img=None,
    generator=None,
    device=None,
    output_dir='inference_results',
    run_name=None,
    save_plots=True,
    verbose=True,
):
    """
    Run inference on one image with hyperparameters from an InferenceConfig.

    Parameters
    ----------
    target_img : (H, W) array or tensor — detector image (xy projection if xz_img is given)
    true_momentum : truth for diagnostics/plots, or None for real data
    cfg : InferenceConfig, DictConfig, or None for defaults
    n_tracks : number of tracks to fit
    initial_momenta : explicit starting guesses; otherwise chosen by cfg.init.method
    xz_img : xz projection; when given, fits one track against both projections
    generator : pre-loaded DifferentiableLDMGenerator (loaded from paths.yaml if None)

    Returns the results dict from run_inference().
    """
    if cfg is None:
        cfg = InferenceConfig()
    elif OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_object(cfg)

    initial_momenta = _initial_guess(target_img, cfg.init, n_tracks, initial_momenta, verbose,
                                     true_momentum=true_momentum)
    kwargs = cfg.to_kwargs()
    kwargs['init_pz'] = None   # already applied by _initial_guess

    return run_inference(
        target_img      = target_img,
        true_momentum   = true_momentum,
        n_tracks        = n_tracks,
        initial_momenta = initial_momenta,
        dual_projection = xz_img is not None,
        xz_img          = xz_img,
        generator       = generator,
        device          = device,
        output_dir      = output_dir,
        run_name        = run_name,
        save_plots      = save_plots,
        verbose         = verbose,
        **kwargs,
    )


def _initial_guess(target_img, init, n_tracks, initial_momenta, verbose, true_momentum=None):
    """
    Starting momenta from an InitConfig: hull or random, then init_pz, then the
    pmag_scale / pmag_jitter / angle_jitter_deg perturbations, then truth_pz.
    Randomness comes from numpy's global RNG, so it is reproducible under the run's `seed`.
    """
    rng = np.random.default_rng(np.random.randint(2**31))

    if initial_momenta is None:
        use_hull = init.method == 'hull' and not (init.random_frac > 0 and rng.random() < init.random_frac)
        if use_hull:
            initial_momenta = convex_hull_initial_guess(
                target_img, n_tracks=n_tracks, use_pmag_estimate=init.use_pmag,
            )
            if initial_momenta is None and verbose:
                print("[hull] construction failed — falling back to random init")
            elif verbose:
                for ti, (px0, py0, pz0) in enumerate(initial_momenta):
                    tag = (f"  |p|={float(np.sqrt(px0**2 + py0**2 + pz0**2)):.1f}"
                           if init.use_pmag else "")
                    print(f"[hull] init t{ti+1}: ({px0:.1f}, {py0:.1f}, {pz0:.1f}){tag}")
        if initial_momenta is None:
            initial_momenta = [
                random_momentum_init(init.random_pmin, init.random_pmax,
                                     seed=int(rng.integers(2**31)))[0]
                for _ in range(n_tracks)
            ]
            if verbose:
                for ti, (px0, py0, pz0) in enumerate(initial_momenta):
                    print(f"[random] init t{ti+1}: ({px0:.1f}, {py0:.1f}, {pz0:.1f})"
                          f"  |p|={float(np.sqrt(px0**2 + py0**2 + pz0**2)):.1f}")

    initial_momenta = [tuple(float(v) for v in m) for m in initial_momenta]
    if init.init_pz is not None:
        initial_momenta = [(px, py, float(init.init_pz)) for px, py, _ in initial_momenta]
        if verbose:
            print(f"[init_pz] pz overridden to {float(init.init_pz):.1f} MeV")

    if init.pmag_scale != 1.0 or init.pmag_jitter > 0 or init.angle_jitter_deg > 0:
        initial_momenta = [
            perturb_momentum(m, rng, init.pmag_scale, init.pmag_jitter, init.angle_jitter_deg)
            for m in initial_momenta
        ]
        if verbose:
            for ti, (px0, py0, pz0) in enumerate(initial_momenta):
                print(f"[perturb] init t{ti+1}: ({px0:.1f}, {py0:.1f}, {pz0:.1f})"
                      f"  |p|={float(np.sqrt(px0**2 + py0**2 + pz0**2)):.1f}")

    if init.truth_pz:
        if true_momentum is None:
            raise ValueError("init.truth_pz needs true_momentum")
        truths = np.asarray(true_momentum, dtype=float).reshape(-1, 3)
        if len(truths) != len(initial_momenta):
            raise ValueError(f"init.truth_pz: {len(truths)} truth tracks for {len(initial_momenta)} fitted tracks")
        initial_momenta = [(px, py, float(t[2])) for (px, py, _), t in zip(initial_momenta, truths)]
        if verbose:
            for ti, (px0, py0, pz0) in enumerate(initial_momenta):
                print(f"[truth_pz] init t{ti+1}: ({px0:.1f}, {py0:.1f}, {pz0:.1f})")
    return initial_momenta
