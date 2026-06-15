"""
compute_loss_surface.py

Compute the EMD loss on a 2D (px, py) grid for an existing SGD run, then
save a heatmap alongside the precomputed arrays.  The grid is auto-sized to
cover the full optimization trajectory plus a configurable margin, and pz is
fixed at its truth value (or the final trajectory value if truth is absent).

Saved to <run>/data_files/:
    loss_surface.npy      shape (n_grid, n_grid)  — loss_grid[i,j] at (px_vals[i], py_vals[j])
    loss_surface_px.npy   shape (n_grid,)
    loss_surface_py.npy   shape (n_grid,)
    loss_surface_pz.npy   scalar (the fixed pz value)

Saved to <run>/plots/:
    loss_surface_heatmap.png

plot_inference.py will pick up the .npy files automatically if present.
"""

import argparse, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.append('/n/home11/zimani/reco_model/')
sys.path.append('/n/home11/zimani/latent-diffusion')
sys.path.append('/n/home11/zimani/inference_loop')

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from helper_inference import DifferentiableLDMGenerator, emd_loss_with_gradients

LDM_CONFIG_PATH     = "/n/home11/zimani/latent-diffusion/configs/latent-diffusion/protons64-ldm-kl.yaml"
LDM_CHECKPOINT_PATH = "/n/home11/zimani/latent-diffusion/edep_protons64_v2_ldm/runs/checkpoints/epoch=000075.ckpt"


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_ldm(device):
    config = OmegaConf.load(LDM_CONFIG_PATH)
    pl_sd  = torch.load(LDM_CHECKPOINT_PATH, map_location="cpu")
    ldm    = instantiate_from_config(config.model)
    ldm.load_state_dict(pl_sd["state_dict"], strict=False)
    ldm.to(device).eval()
    return ldm


# ── Core grid sweep ────────────────────────────────────────────────────────────

def compute_loss_surface(
    data_dir,
    n_grid=20,
    n_avg=3,
    margin=0.2,
    ddim_steps=20,
    device=None,
):
    """
    Evaluate EMD on an (n_grid × n_grid) (px, py) grid at fixed pz.

    Returns
    -------
    loss_grid : ndarray (n_grid, n_grid)   loss_grid[i,j] at (px_vals[i], py_vals[j])
    px_vals   : ndarray (n_grid,)
    py_vals   : ndarray (n_grid,)
    pz_fixed  : float
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── load run data ──────────────────────────────────────────────────────────
    mom_path  = np.load(os.path.join(data_dir, "mom_path.npy"))
    target_np = np.load(os.path.join(data_dir, "target_img.npy"))
    target_img = torch.tensor(target_np, dtype=torch.float32).to(device)

    if mom_path.ndim != 2 or mom_path.shape[1] != 3:
        raise ValueError(
            "compute_loss_surface supports single-track runs only "
            f"(expected mom_path shape (n, 3), got {mom_path.shape})"
        )

    px_traj, py_traj = mom_path[:, 0], mom_path[:, 1]

    # fixed pz: truth preferred, else final trajectory step
    truth_path = os.path.join(data_dir, "truth_mom.npy")
    truth_mom  = None
    if os.path.exists(truth_path):
        arr = np.load(truth_path)
        truth_mom = arr.flatten()[:3]
        pz_fixed  = float(truth_mom[2])
    else:
        pz_fixed = float(mom_path[-1, 2])

    # grid bounding box: traj + truth (if available) + margin
    pts_x = list(px_traj)
    pts_y = list(py_traj)
    if truth_mom is not None:
        pts_x.append(float(truth_mom[0]))
        pts_y.append(float(truth_mom[1]))

    px_min, px_max = min(pts_x), max(pts_x)
    py_min, py_max = min(pts_y), max(pts_y)
    px_span = max(px_max - px_min, 100.0)
    py_span = max(py_max - py_min, 100.0)
    px_lo, px_hi = px_min - margin * px_span, px_max + margin * px_span
    py_lo, py_hi = py_min - margin * py_span, py_max + margin * py_span

    px_vals = np.linspace(px_lo, px_hi, n_grid)
    py_vals = np.linspace(py_lo, py_hi, n_grid)

    print(f"Grid: px [{px_lo:.0f}, {px_hi:.0f}]  py [{py_lo:.0f}, {py_hi:.0f}]  pz={pz_fixed:.0f} (fixed)")
    print(f"Points: {n_grid}x{n_grid}={n_grid*n_grid}, averaged over {n_avg} sample(s) each")
    print(f"Total forward passes: {n_grid*n_grid*n_avg}")

    # ── load model ─────────────────────────────────────────────────────────────
    print("Loading LDM...")
    ldm = _load_ldm(device)
    generator = DifferentiableLDMGenerator(
        ldm, device=str(device),
        ddim_steps_standard=ddim_steps,
        ddim_steps_gradient=10,
    )
    print("LDM ready.\n")

    # ── sweep ──────────────────────────────────────────────────────────────────
    loss_grid = np.zeros((n_grid, n_grid))
    t0    = time.time()
    total = n_grid * n_grid
    for i, px in enumerate(px_vals):
        for j, py in enumerate(py_vals):
            samples = []
            for _ in range(n_avg):
                with torch.no_grad():
                    img  = generator(float(px), float(py), float(pz_fixed))
                    loss = emd_loss_with_gradients(img, target_img).item()
                samples.append(loss)
            loss_grid[i, j] = float(np.mean(samples))

            done    = i * n_grid + j + 1
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done)
            print(
                f"  [{done:4d}/{total}]  px={px:+.0f}  py={py:+.0f}  "
                f"loss={loss_grid[i,j]:.4f}  ETA {eta/60:.1f} min   ",
                end='\r',
            )

    print(f"\nSweep done in {(time.time()-t0)/60:.1f} min")

    # ── save arrays ────────────────────────────────────────────────────────────
    np.save(os.path.join(data_dir, "loss_surface.npy"),    loss_grid)
    np.save(os.path.join(data_dir, "loss_surface_px.npy"), px_vals)
    np.save(os.path.join(data_dir, "loss_surface_py.npy"), py_vals)
    np.save(os.path.join(data_dir, "loss_surface_pz.npy"), np.array([pz_fixed]))
    print(f"Arrays saved to {data_dir}/")

    return loss_grid, px_vals, py_vals, pz_fixed


# ── Heatmap ────────────────────────────────────────────────────────────────────

def plot_loss_surface_heatmap(loss_grid, px_vals, py_vals, pz_fixed,
                               mom_path, dist_path, truth_mom, save_path):
    """2D heatmap with the SGD trajectory overlaid."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # loss_grid[i,j]: i→px, j→py.  imshow expects (rows=py, cols=px) → transpose.
    im = ax.imshow(
        loss_grid.T, origin='lower', aspect='auto', cmap='viridis',
        extent=[px_vals[0], px_vals[-1], py_vals[0], py_vals[-1]],
    )
    plt.colorbar(im, ax=ax, label='EMD Loss')

    px_traj, py_traj = mom_path[:, 0], mom_path[:, 1]
    iters = np.arange(len(px_traj))
    ax.plot(px_traj, py_traj, color='white', linewidth=1.0, alpha=0.6, zorder=5)
    ax.scatter(px_traj, py_traj, c=dist_path, cmap='plasma',
               vmin=np.min(dist_path), vmax=np.max(dist_path),
               s=30, zorder=6, edgecolors='none')
    ax.scatter(px_traj[0],  py_traj[0],  s=150, color='cyan',  marker='s',
               zorder=8, edgecolors='black', linewidth=0.8, label='Start')
    ax.scatter(px_traj[-1], py_traj[-1], s=150, color='lime',  marker='D',
               zorder=8, edgecolors='black', linewidth=0.8, label='End')

    if truth_mom is not None:
        ax.scatter(float(truth_mom[0]), float(truth_mom[1]), s=300, color='red',
                   marker='*', zorder=9, edgecolors='white', linewidth=0.8,
                   label=f'Truth ({float(truth_mom[0]):.0f}, {float(truth_mom[1]):.0f})')

    ax.set_xlabel('px (MeV)', fontsize=12)
    ax.set_ylabel('py (MeV)', fontsize=12)
    ax.set_title(f'EMD Loss Surface  (pz = {pz_fixed:.0f} MeV fixed)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save the EMD loss surface for an SGD run."
    )
    parser.add_argument("output_folder", nargs="?", default=None,
                        help="Subfolder name inside results root (default: most recent).")
    parser.add_argument("--results-root", default="inference_results")
    parser.add_argument("--n-grid",    type=int,   default=20,
                        help="Grid resolution per axis (default: 20).")
    parser.add_argument("--n-avg",     type=int,   default=3,
                        help="LDM samples averaged per grid point (default: 3).")
    parser.add_argument("--margin",    type=float, default=0.2,
                        help="Fractional padding beyond trajectory bbox (default: 0.2).")
    parser.add_argument("--ddim-steps", type=int,  default=20,
                        help="DDIM steps per forward pass (default: 20).")
    args = parser.parse_args()

    # resolve results root
    results_root = args.results_root
    if not os.path.isdir(results_root):
        print(f"Results root '{results_root}' not found. Pass --results-root.")
        exit(1)

    # resolve output folder
    if args.output_folder:
        output_folder = args.output_folder
    else:
        candidates = [
            d for d in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, d, "data_files"))
        ]
        if not candidates:
            print(f"No completed runs found under '{results_root}/'.")
            exit(1)
        output_folder = max(
            candidates,
            key=lambda d: os.path.getmtime(os.path.join(results_root, d, "data_files")),
        )
        print(f"Auto-selected most recent run: {output_folder}")

    data_dir = os.path.join(results_root, output_folder, "data_files")
    plot_dir = os.path.join(results_root, output_folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(os.path.join(data_dir, "mom_path.npy")):
        print(f"No SGD outputs found in {data_dir}/.")
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    loss_grid, px_vals, py_vals, pz_fixed = compute_loss_surface(
        data_dir,
        n_grid=args.n_grid,
        n_avg=args.n_avg,
        margin=args.margin,
        ddim_steps=args.ddim_steps,
        device=device,
    )

    mom_path  = np.load(os.path.join(data_dir, "mom_path.npy"))
    dist_path = np.load(os.path.join(data_dir, "dist_path.npy"))

    truth_mom = None
    tp = os.path.join(data_dir, "truth_mom.npy")
    if os.path.exists(tp):
        arr = np.load(tp)
        truth_mom = arr.flatten()[:3]

    plot_loss_surface_heatmap(
        loss_grid, px_vals, py_vals, pz_fixed,
        mom_path=mom_path,
        dist_path=dist_path,
        truth_mom=truth_mom,
        save_path=os.path.join(plot_dir, "loss_surface_heatmap.png"),
    )

    print(f"\nDone. Run plot_inference.py to regenerate the 3D trajectory plot.")
