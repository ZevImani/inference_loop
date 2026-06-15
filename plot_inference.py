import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import torch
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_numpy(img):
    if isinstance(img, torch.Tensor):
        return img.squeeze().cpu().numpy()
    return np.array(img).squeeze()


def _is_double_track(mom_path):
    """True if mom_path holds pairs of momenta — shape (n, 2, 3) after conversion."""
    return np.array(mom_path).ndim == 3


def _extract_mom(m):
    """Single-track: return (px, py, pz)."""
    if isinstance(m, torch.Tensor):
        return tuple(m.cpu().numpy().tolist())
    return tuple(float(v) for v in m)


def _extract_mom_pair(m):
    """Double-track: return ((px1,py1,pz1), (px2,py2,pz2))."""
    arr = np.asarray(m)  # shape (2, 3)
    return tuple(float(v) for v in arr[0]), tuple(float(v) for v in arr[1])


def _truth_scalars(truth_mom):
    if truth_mom is None:
        return None
    if isinstance(truth_mom, (np.ndarray, torch.Tensor)):
        arr = truth_mom.cpu().numpy() if isinstance(truth_mom, torch.Tensor) else truth_mom
        return tuple(float(v) for v in arr)
    return tuple(float(v) for v in truth_mom)


def _pad_lr(lr_path, total_len):
    """lr_path has length n_iters; pad a None at index 0 to align with dist/mom."""
    if len(lr_path) == total_len:
        return list(lr_path)
    return [None] + list(lr_path)


# ── Static image grid ──────────────────────────────────────────────────────────

def plot_sgd_images(
    img_path,
    dist_path,
    mom_path,
    lr_path,
    truth_mom,
    save_path='sgd_images.png',
    n_cols=5,
):
    """Plot a grid of all generated images from the SGD run."""
    double = _is_double_track(mom_path)
    mom_arr = np.array(mom_path)
    lr_padded = _pad_lr(lr_path, len(dist_path))
    dist_arr = np.array(dist_path, dtype=float)
    indices = list(range(len(dist_arr)))

    n_images = len(indices)
    n_cols = min(n_cols, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    row_h = 4.4 if double else 3.8
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, row_h * n_rows),
                              squeeze=False)

    vmax = max(_to_numpy(img_path[i]).max() for i in indices)
    vmax = vmax if vmax > 0 else 1.0

    for plot_idx, data_idx in enumerate(indices):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row][col]
        img_np = _to_numpy(img_path[data_idx])
        ax.imshow(img_np, cmap='gray', aspect='equal', vmin=0, vmax=vmax,
                  interpolation='nearest')

        lr_val = lr_padded[data_idx]
        lr_str = f'LR={lr_val:.3f}' if lr_val is not None else 'Initial'
        label = 'Initial' if data_idx == 0 else f'Step {data_idx}'

        if double:
            m1, m2 = _extract_mom_pair(mom_arr[data_idx])
            title = (f'{label}  EMD={dist_arr[data_idx]:.4f}  {lr_str}\n'
                     f'T1: px={m1[0]:.1f} py={m1[1]:.1f} pz={m1[2]:.1f}\n'
                     f'T2: px={m2[0]:.1f} py={m2[1]:.1f} pz={m2[2]:.1f}')
        else:
            m = _extract_mom(mom_path[data_idx])
            title = (f'{label}  |  EMD={dist_arr[data_idx]:.4f}\n'
                     f'px={m[0]:.1f}  py={m[1]:.1f}  pz={m[2]:.1f}\n'
                     f'{lr_str}')

        ax.set_title(title, fontsize=7 if double else 8, fontweight='bold', pad=4)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for plot_idx in range(n_images, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row][col].set_visible(False)

    if double and truth_mom is not None:
        t1 = tuple(float(v) for v in truth_mom[0])
        t2 = tuple(float(v) for v in truth_mom[1])
        suptitle = (f'Double-Track SGD  —  All Steps\n'
                    f'Truth T1: px={t1[0]:.1f} py={t1[1]:.1f} pz={t1[2]:.1f}  |  '
                    f'T2: px={t2[0]:.1f} py={t2[1]:.1f} pz={t2[2]:.1f}')
    elif not double and truth_mom is not None:
        truth = _truth_scalars(truth_mom)
        suptitle = (f'SGD Optimisation  —  All Steps\n'
                    f'Truth: px={truth[0]:.1f}  py={truth[1]:.1f}  pz={truth[2]:.1f}')
    else:
        suptitle = 'SGD Optimisation  —  All Steps'

    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Image grid saved to {save_path}")
    plt.close(fig)
    return fig


# ── GIF ────────────────────────────────────────────────────────────────────────

def create_sgd_gif(
    img_path,
    dist_path,
    mom_path,
    lr_path,
    save_path='sgd_evolution.gif',
    fps=4,
    stride=1,
):
    """
    Animated GIF of the SGD image sequence.

    stride: show every Nth frame (useful if there are many iterations).
    """
    double = _is_double_track(mom_path)
    mom_arr = np.array(mom_path)
    lr_padded = _pad_lr(lr_path, len(dist_path))
    dist_arr = np.array(dist_path, dtype=float)
    frame_indices = list(range(0, len(img_path), stride))

    vmax = max(_to_numpy(img_path[i]).max() for i in frame_indices)
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.axis('off')
    img_np0 = _to_numpy(img_path[frame_indices[0]])
    im = ax.imshow(img_np0, cmap='gray', aspect='equal', vmin=0, vmax=vmax,
                   interpolation='nearest')

    def update(frame_pos):
        data_idx = frame_indices[frame_pos]
        im.set_array(_to_numpy(img_path[data_idx]))

        lr_val = lr_padded[data_idx]
        lr_str = f'LR={lr_val:.3f}' if lr_val is not None else ''

        if double:
            m1, m2 = _extract_mom_pair(mom_arr[data_idx])
            title = (f'Step {data_idx}  EMD={dist_arr[data_idx]:.5f}  {lr_str}\n'
                     f'T1: px={m1[0]:.1f} py={m1[1]:.1f} pz={m1[2]:.1f}\n'
                     f'T2: px={m2[0]:.1f} py={m2[1]:.1f} pz={m2[2]:.1f}')
        else:
            m = _extract_mom(mom_path[data_idx])
            title = (f'Step {data_idx}\n'
                     f'EMD={dist_arr[data_idx]:.5f}  {lr_str}\n'
                     f'px={m[0]:.1f}  py={m[1]:.1f}  pz={m[2]:.1f}')

        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_visible(True)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=1000 / fps, blit=True, repeat=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    print(f"GIF saved to {save_path}")
    plt.close(fig)
    return anim


# ── Trajectory / scalar plots ──────────────────────────────────────────────────

def plot_sgd_results(
    dist_path,
    mom_path,
    truth_mom,
    save_dir='sgd_plots',
):
    """
    Save trajectory plots:
      - distance_evolution.png
      - momentum_components.png
      - momentum_2d.png
    """
    os.makedirs(save_dir, exist_ok=True)
    double = _is_double_track(mom_path)
    dist_arr = np.array(dist_path, dtype=float)
    iters = np.arange(len(dist_arr))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, dist_arr, color='steelblue', linewidth=1.5, alpha=0.8)
    ax.scatter(iters, dist_arr, color='steelblue', s=60, zorder=5,
               edgecolors='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('EMD Distance', fontsize=12)
    ax.set_title('Distance vs Iteration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distance_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if double:
        mom_arr = np.array(mom_path)  # shape (n, 2, 3)
        _plot_double_momentum(iters, mom_arr, truth_mom, save_dir, dist_arr=dist_arr)
    else:
        mom_arr = np.array([_extract_mom(m) for m in mom_path])
        truth = _truth_scalars(truth_mom)
        _plot_single_momentum(iters, mom_arr[:, 0], mom_arr[:, 1], mom_arr[:, 2],
                              truth, save_dir, dist_arr=dist_arr)

    print(f"All trajectory plots saved to {save_dir}/")


def _add_gradient_line(ax, px, py, iters, cmap):
    points = np.array([px, py]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,
                        norm=plt.Normalize(iters.min() - len(iters) // 2, iters.max()),
                        linewidth=0.8, alpha=0.5, zorder=1)
    lc.set_array(iters[:-1])
    ax.add_collection(lc)


def _plot_single_momentum(iters, px, py, pz, truth, save_dir, dist_arr=None):
    colors = {'x': 'steelblue', 'y': 'seagreen', 'z': 'darkorange'}
    truth_vals = dict(zip(('x', 'y', 'z'), truth if truth else (None, None, None)))
    comp_data  = {'x': px, 'y': py, 'z': pz}

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for ax, comp in zip(axes, ('x', 'y', 'z')):
        vals, color = comp_data[comp], colors[comp]
        ax.plot(iters, vals, color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(iters, vals, color=color, s=40, zorder=5,
                   edgecolors='white', linewidth=0.5)
        tv = truth_vals[comp]
        if tv is not None:
            ax.axhline(tv, color=color, linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'Truth = {tv:.1f}')
            ax.legend(fontsize=9, loc='upper right')
        ax.set_ylabel(f'p{comp}', fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_title('Momentum Components vs Iteration', fontsize=13, fontweight='bold')
    axes[-1].set_xlabel('Iteration', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'momentum_components.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    truth_color = 'crimson'
    track_color = truth_color if truth else 'steelblue'
    fig, ax = plt.subplots(figsize=(7, 7))
    emd_c = dist_arr if dist_arr is not None else iters
    sc = ax.scatter(px, py, c=emd_c, cmap='plasma', vmin=emd_c.min(), vmax=emd_c.max(),
                    s=80, zorder=5, edgecolors='white', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='EMD')
    _add_gradient_line(ax, px, py, iters, 'Blues')
    step = max(1, len(iters) // 20)
    for i in range(0, len(iters) - 1, step):
        ax.annotate('', xy=(px[i + 1], py[i + 1]), xytext=(px[i], py[i]),
                    arrowprops=dict(arrowstyle='->', color=track_color, alpha=0.6, lw=1.2))
    ax.scatter(px[0], py[0], s=200, color=track_color, marker='s',
               zorder=10, edgecolors='black', linewidth=1,
               label=f'Start ({px[0]:.1f}, {py[0]:.1f})')
    ax.scatter(px[-1], py[-1], s=200, color=track_color, marker='D',
               zorder=10, edgecolors='black', linewidth=1,
               label=f'End ({px[-1]:.1f}, {py[-1]:.1f})')
    if truth:
        ax.scatter(truth[0], truth[1], s=350, color=truth_color, marker='*',
                   zorder=11, label=f'Truth ({truth[0]:.1f}, {truth[1]:.1f})',
                   edgecolors='black', linewidth=1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('px', fontsize=12)
    ax.set_ylabel('py', fontsize=12)
    ax.set_title('2D Momentum Trajectory (px vs py)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'momentum_2d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_double_momentum(iters, mom_arr, truth_mom, save_dir, dist_arr=None):
    """Momentum trajectory plots for both tracks. mom_arr shape: (n, 2, 3)."""
    px1, py1, pz1 = mom_arr[:, 0, 0], mom_arr[:, 0, 1], mom_arr[:, 0, 2]
    px2, py2, pz2 = mom_arr[:, 1, 0], mom_arr[:, 1, 1], mom_arr[:, 1, 2]

    t1 = t2 = None
    if truth_mom is not None:
        t1 = tuple(float(v) for v in truth_mom[0])
        t2 = tuple(float(v) for v in truth_mom[1])

    labels     = ['T1 px', 'T1 py', 'T1 pz', 'T2 px', 'T2 py', 'T2 pz']
    all_data   = [px1, py1, pz1, px2, py2, pz2]
    all_colors = ['steelblue', 'seagreen', 'darkorange',
                  'royalblue', 'mediumseagreen', 'tomato']
    truth_vals = (list(t1) + list(t2)) if (t1 and t2) else [None] * 6

    fig, axes = plt.subplots(6, 1, figsize=(7, 14), sharex=True)
    for ax, vals, label, color, tv in zip(axes, all_data, labels, all_colors, truth_vals):
        ax.plot(iters, vals, color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(iters, vals, color=color, s=30, zorder=5,
                   edgecolors='white', linewidth=0.5)
        if tv is not None:
            ax.axhline(tv, color=color, linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'Truth = {tv:.1f}')
            ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_title('Double-Track Momentum Components vs Iteration',
                      fontsize=13, fontweight='bold')
    axes[-1].set_xlabel('Iteration', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'momentum_components.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))

    truth_colors = [plt.get_cmap('Blues')(0.7), plt.get_cmap('Reds')(0.7)]
    if t1 is not None and t2 is not None:
        end1 = np.array([px1[-1], py1[-1]])
        end2 = np.array([px2[-1], py2[-1]])
        tp1  = np.array([t1[0], t1[1]])
        tp2  = np.array([t2[0], t2[1]])
        cost_straight = np.linalg.norm(end1 - tp1) + np.linalg.norm(end2 - tp2)
        cost_swapped  = np.linalg.norm(end1 - tp2) + np.linalg.norm(end2 - tp1)
        end_colors = [truth_colors[0], truth_colors[1]] if cost_straight <= cost_swapped \
                     else [truth_colors[1], truth_colors[0]]
    else:
        end_colors = [plt.get_cmap('Blues')(0.9), plt.get_cmap('Reds')(0.9)]

    track_specs = [
        (px1, py1, 'Blues', 'Track 1', t1, end_colors[0]),
        (px2, py2, 'Reds',  'Track 2', t2, end_colors[1]),
    ]
    emd_c = dist_arr if dist_arr is not None else iters
    emd_vmin, emd_vmax = emd_c.min(), emd_c.max()
    step = max(1, len(iters) // 15)
    sc_last = None
    for px, py, cmap, label, truth, end_color in track_specs:
        ax.plot(px, py, linewidth=0.8, alpha=0.4, zorder=1, color=end_color)
        sc_last = ax.scatter(px, py, c=emd_c, cmap=cmap, vmin=emd_vmin, vmax=emd_vmax,
                             s=60, zorder=5, edgecolors='white', linewidth=0.5)
        _add_gradient_line(ax, px, py, iters, cmap)
        for i in range(0, len(iters) - 1, step):
            ax.annotate('', xy=(px[i + 1], py[i + 1]), xytext=(px[i], py[i]),
                        arrowprops=dict(arrowstyle='->', color=end_color,
                                        alpha=0.5, lw=1.0))
        ax.scatter(px[0], py[0], s=200, color=end_color, marker='s',
                   zorder=10, edgecolors='black', linewidth=1,
                   label=f'{label} start ({px[0]:.1f}, {py[0]:.1f})')
        ax.scatter(px[-1], py[-1], s=200, color=end_color, marker='D',
                   zorder=10, edgecolors='black', linewidth=1,
                   label=f'{label} end ({px[-1]:.1f}, {py[-1]:.1f})')
        if truth is not None:
            truth_c = plt.get_cmap(cmap)(0.7)
            ax.scatter(truth[0], truth[1], s=350, color=truth_c, marker='*',
                       zorder=11, edgecolors='black', linewidth=1,
                       label=f'{label} truth ({truth[0]:.1f}, {truth[1]:.1f})')
    if sc_last is not None:
        plt.colorbar(sc_last, ax=ax, label='EMD')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('px', fontsize=12)
    ax.set_ylabel('py', fontsize=12)
    ax.set_title('2D Momentum Trajectories (px vs py) — Both Tracks',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'momentum_2d.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── 3-D loss surface + trajectory ─────────────────────────────────────────────

def plot_loss_surface_3d(dist_path, mom_path, truth_mom, data_dir, save_dir):
    """
    Load a precomputed loss surface from data_dir (if present) and render a
    3-D figure: the surface as a mesh and the SGD trajectory as a floating
    curve at its actual EMD height.

    Files expected in data_dir:
        loss_surface.npy      (n_px, n_py)
        loss_surface_px.npy   (n_px,)
        loss_surface_py.npy   (n_py,)
        loss_surface_pz.npy   scalar

    Returns True if the plot was produced, False if files were absent.
    """
    needed = ["loss_surface.npy", "loss_surface_px.npy",
              "loss_surface_py.npy", "loss_surface_pz.npy"]
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in needed):
        return False

    loss_grid = np.load(os.path.join(data_dir, "loss_surface.npy"))
    px_vals   = np.load(os.path.join(data_dir, "loss_surface_px.npy"))
    py_vals   = np.load(os.path.join(data_dir, "loss_surface_py.npy"))
    pz_fixed  = float(np.load(os.path.join(data_dir, "loss_surface_pz.npy")).flat[0])

    dist_arr = np.array(dist_path, dtype=float)
    mom_arr  = np.array(mom_path)
    if mom_arr.ndim != 2 or mom_arr.shape[1] != 3:
        return False  # double-track not supported here

    px_traj, py_traj = mom_arr[:, 0], mom_arr[:, 1]

    # meshgrid for plot_surface: PX/PY shape (n_py, n_px)
    PX, PY = np.meshgrid(px_vals, py_vals)
    LOSS   = loss_grid.T          # (n_py, n_px) so it aligns with PX, PY

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        PX, PY, LOSS,
        cmap='viridis', alpha=0.55, linewidth=0, antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, label='EMD Loss')

    # trajectory as a 3-D curve (z = actual EMD at each step)
    ax.plot(px_traj, py_traj, dist_arr,
            color='white', linewidth=1.8, zorder=10, alpha=0.9)

    # colour the trajectory dots by iteration so progression is visible
    iters = np.arange(len(px_traj))
    sc = ax.scatter(px_traj, py_traj, dist_arr,
                    c=iters, cmap='plasma', s=30, zorder=11,
                    edgecolors='none', depthshade=False)

    # start / end markers
    ax.scatter([px_traj[0]],  [py_traj[0]],  [dist_arr[0]],
               s=120, color='cyan', marker='s', zorder=12,
               edgecolors='black', linewidth=0.8,
               label=f'Start  EMD={dist_arr[0]:.3f}')
    ax.scatter([px_traj[-1]], [py_traj[-1]], [dist_arr[-1]],
               s=120, color='lime', marker='D', zorder=12,
               edgecolors='black', linewidth=0.8,
               label=f'End  EMD={dist_arr[-1]:.3f}')

    # truth: project star down to surface level and up to its actual height
    if truth_mom is not None:
        truth = _truth_scalars(truth_mom)
        tx, ty = float(truth[0]), float(truth[1])
        # interpolate surface loss at truth location for the "floor" marker
        ix = int(np.argmin(np.abs(px_vals - tx)))
        iy = int(np.argmin(np.abs(py_vals - ty)))
        tz_surf = loss_grid[ix, iy]
        ax.scatter([tx], [ty], [tz_surf],
                   s=280, color='red', marker='*', zorder=13,
                   edgecolors='white', linewidth=0.6,
                   label=f'Truth ({tx:.0f}, {ty:.0f})')
        # vertical dashed line from surface to max z so it is easy to see
        ax.plot([tx, tx], [ty, ty], [tz_surf, dist_arr.max()],
                color='red', linewidth=1.0, linestyle='--', alpha=0.5)

    ax.set_xlabel('px (MeV)', fontsize=10, labelpad=6)
    ax.set_ylabel('py (MeV)', fontsize=10, labelpad=6)
    ax.set_zlabel('EMD Loss',  fontsize=10, labelpad=6)
    ax.set_title(f'Loss Surface & SGD Trajectory  (pz = {pz_fixed:.0f} MeV fixed)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'loss_surface_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"3D loss surface plot saved to {save_path}")
    return True


# ── Initial / final comparison ─────────────────────────────────────────────────

def plot_initial_final(img_path, dist_path, mom_path, target_img=None,
                       truth_mom=None, save_path='initial_final.png'):
    """Side-by-side of the target image and the final generated image."""
    double = _is_double_track(mom_path)
    mom_arr = np.array(mom_path)

    img0 = _to_numpy(target_img) if target_img is not None else _to_numpy(img_path[0])
    imgf = _to_numpy(img_path[-1])
    vmax = max(img0.max(), imgf.max())
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if double:
        if truth_mom is not None:
            t1 = tuple(float(v) for v in truth_mom[0])
            t2 = tuple(float(v) for v in truth_mom[1])
            left_title = (f'Input (Target)\n'
                          f'T1: px={t1[0]:.1f} py={t1[1]:.1f} pz={t1[2]:.1f}\n'
                          f'T2: px={t2[0]:.1f} py={t2[1]:.1f} pz={t2[2]:.1f}')
        else:
            left_title = 'Input (Target)'
        mf1, mf2 = _extract_mom_pair(mom_arr[-1])
        right_title = (f'Final  EMD={dist_path[-1]:.4f}\n'
                       f'T1: px={mf1[0]:.1f} py={mf1[1]:.1f} pz={mf1[2]:.1f}\n'
                       f'T2: px={mf2[0]:.1f} py={mf2[1]:.1f} pz={mf2[2]:.1f}')
    else:
        truth = _truth_scalars(truth_mom)
        left_title = (f'Input (Truth)\npx={truth[0]:.1f}  py={truth[1]:.1f}  pz={truth[2]:.1f}'
                      if truth else 'Input (Truth)')
        mf = _extract_mom(mom_path[-1])
        right_title = (f'Final\nEMD={dist_path[-1]:.4f}\n'
                       f'px={mf[0]:.1f}  py={mf[1]:.1f}  pz={mf[2]:.1f}')

    for ax, img, title in [(axes[0], img0, left_title), (axes[1], imgf, right_title)]:
        ax.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=vmax,
                  interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Initial/final comparison saved to {save_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate inference plots from saved SGD run outputs. "
                    "Automatically detects single-track vs double-track from the data shape."
    )
    parser.add_argument(
        "output_folder", nargs="?", default=None,
        help="Subfolder name inside the results root. "
             "If omitted, the most recently modified folder is used.",
    )
    parser.add_argument(
        "--results-root", default="inference_results",
        help="Root directory containing output folders (default: inference_results).",
    )
    parser.add_argument("--fps",    type=int, default=4, help="Frames per second for the GIF (default: 4).")
    parser.add_argument("--stride", type=int, default=1, help="Show every Nth frame in the GIF (default: 1).")
    parser.add_argument("--n-cols", type=int, default=5, help="Columns in the image grid (default: 5).")

    # Plot selection flags — all enabled by default; use --skip-X to disable
    parser.add_argument("--skip-images",       action="store_true", help="Skip the image grid plot.")
    parser.add_argument("--skip-gif",          action="store_true", help="Skip the animated GIF.")
    parser.add_argument("--skip-trajectories", action="store_true", help="Skip trajectory/scalar plots.")
    parser.add_argument("--skip-comparison",   action="store_true", help="Skip the initial/final comparison.")
    parser.add_argument("--skip-loss-surface", action="store_true",
                        help="Skip the 3D loss surface plot (no-op if surface files are absent).")

    args = parser.parse_args()

    # Resolve results root
    results_root = args.results_root
    if not os.path.isdir(results_root):
        print(f"Results root '{results_root}' not found. Pass --results-root.")
        exit(1)

    # Resolve output folder
    if args.output_folder:
        output_folder = args.output_folder
    else:
        if not os.path.isdir(results_root):
            print(f"Results root '{results_root}' not found.")
            exit(1)
        candidates = [
            d for d in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, d, "data_files"))
        ]
        if not candidates:
            print(f"No completed SGD runs found under '{results_root}/'.")
            exit(1)
        output_folder = max(
            candidates,
            key=lambda d: os.path.getmtime(os.path.join(results_root, d, "data_files")),
        )
        print(f"Auto-selected most recent run: {output_folder}")

    data_dir = os.path.join(results_root, output_folder, "data_files")
    plot_dir = os.path.join(results_root, output_folder, "plots")

    if not os.path.exists(os.path.join(data_dir, "img_path.pt")):
        print(f"No SGD outputs found in {data_dir}/.")
        exit(1)

    print("Loading SGD outputs...")
    img_tensor = torch.load(os.path.join(data_dir, "img_path.pt"), map_location="cpu")
    img_path   = [img_tensor[i] for i in range(img_tensor.shape[0])]
    mom_path   = np.load(os.path.join(data_dir, "mom_path.npy"))
    dist_path  = np.load(os.path.join(data_dir, "dist_path.npy")).tolist()
    lr_path    = np.load(os.path.join(data_dir, "std_path.npy")).tolist()

    target_img_path = os.path.join(data_dir, "target_img.npy")
    target_img = np.load(target_img_path) if os.path.exists(target_img_path) else None

    pair_path        = os.path.join(data_dir, "truth_mom_pair.npy")
    single_truth_path = os.path.join(data_dir, "truth_mom.npy")
    if os.path.exists(pair_path):
        pair_arr  = np.load(pair_path)  # shape (2, 3)
        truth_mom = (tuple(pair_arr[0].tolist()), tuple(pair_arr[1].tolist()))
    elif os.path.exists(single_truth_path):
        arr = np.load(single_truth_path)
        # shape (3,) for single-track; shape (1, 3) or (N, 3) for multi-track saves
        if arr.ndim == 1:
            truth_mom = tuple(arr.tolist())
        else:
            truth_mom = tuple(arr[0].tolist())
    else:
        truth_mom = None

    double = _is_double_track(mom_path)
    print(f"Mode: {'double-track' if double else 'single-track'}")

    os.makedirs(plot_dir, exist_ok=True)

    if not args.skip_images:
        plot_sgd_images(
            img_path, dist_path, mom_path, lr_path, truth_mom,
            save_path=os.path.join(plot_dir, 'sgd_images.png'),
            n_cols=args.n_cols,
        )

    if not args.skip_gif:
        create_sgd_gif(
            img_path, dist_path, mom_path, lr_path,
            save_path=os.path.join(plot_dir, 'sgd_evolution.gif'),
            fps=args.fps,
            stride=args.stride,
        )

    if not args.skip_trajectories:
        plot_sgd_results(dist_path, mom_path, truth_mom, save_dir=plot_dir)

    if not args.skip_comparison:
        plot_initial_final(
            img_path, dist_path, mom_path,
            target_img=target_img,
            truth_mom=truth_mom,
            save_path=os.path.join(plot_dir, 'initial_final.png'),
        )

    if not args.skip_loss_surface and not double:
        produced = plot_loss_surface_3d(
            dist_path, mom_path, truth_mom,
            data_dir=data_dir,
            save_dir=plot_dir,
        )
        if not produced:
            print("Loss surface files not found in data_dir — skipping 3D plot. "
                  "Run compute_loss_surface.py first.")

    print(f"\nAll plots saved to {plot_dir}/")
