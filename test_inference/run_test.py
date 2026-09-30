#!/usr/bin/env python3
"""
run_test.py — Generalized inference test over N events.

Modes
-----
1-track (--n_tracks 1):
    Loads events from a single-track dataset directory and runs dual-projection
    SGD inference on each event.

1-track pz-degeneracy (--n_tracks 1 --pz_degeneracy):
    Loads events from the pz_degeneracy dataset (.npz).  Use --pz_which ref
    (default) for real events with |pz|>100 MeV/c, or --pz_which pz0 for the
    MCMC-found pz=0 counterparts.  Tests whether gradient-guided inference
    recovers the correct momentum despite the pz degeneracy.

1-track kinked (--n_tracks 1 --kinked):
    Loads events from the kinked-tracks dataset (.npz, keys: "images", "momenta").
    Use --event_offset to select a specific event (equivalent to --sample_idx in
    half_noise_test.py).  E.g. --kinked --event_offset 15 --n_events 1.

2-track (--n_tracks 2):
    Loads angle pairs from the angle dataset, samples N events uniformly
    across [--angle_min, --angle_max] degrees, runs 2-track inference on each,
    and plots final EMD vs separation angle.

Output layout
-------------
<output_dir>/<run_name>/
    events/
        event_000/
            data_files/   (dist_path.npy, mom_path.npy, target_img.npy, truth_mom.npy, final_img.npy, …)
            plots/        (initial_final.png, distance_evolution.png, …)
        event_001/
            …
    summary/
        summary.npz               (final_emds, final_momentum, truth_momentum, [angles])
        plot_loss_distribution.png
        plot_recovered_vs_truth.png   (1-track / dual-projection)
        plot_residual_histograms.png  (1-track / dual-projection)
        plot_emd_vs_angle.png         (2-track only)

Initial guess: convex-hull estimate; falls back to random if hull construction fails.
The large img_path.pt tensor is removed after each event by default; pass
--keep_img_path to retain it.
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, INFERENCE_DIR)
sys.path.append('/n/home11/zimani/latent-diffusion')

from run_inference import load_ldm, run_inference, random_momentum_init
from helper_inference import DifferentiableLDMGenerator, convex_hull_initial_guess

# ── Default paths ───────────────────────────────────────────────────────────────
DEFAULT_SINGLE_DATASET      = '/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64_v2/edep_val'
DEFAULT_ANGLE_PAIRS         = '/n/home11/zimani/inference_loop/angle_dataset/angle_pairs_val.npy'
DEFAULT_OUTPUT_DIR          = os.path.join(SCRIPT_DIR, 'results')
DEFAULT_PZ_DEGENERACY_DATASET = (
    '/n/home11/zimani/inference_loop/paper_plots/pz_degeneracy/'
    'pz_degeneracy_dataset/pz_degeneracy_dataset.npz'
)
DEFAULT_KINKED_DATASET           = '/n/home11/zimani/inference_loop/datasets/kinked_tracks/kinked_tracks.npz'
DEFAULT_LINE_KINKED_DATASET      = '/n/home11/zimani/inference_loop/datasets/line_kinked_tracks/all_line_kinked_tracks.npz'
DEFAULT_LINE_KINKED_DATASET_SMALL = '/n/home11/zimani/inference_loop/datasets/line_kinked_tracks/line_kinked_tracks_64.npz'


# ── Data loaders ────────────────────────────────────────────────────────────────

def _load_pz_degeneracy_events(npz_path, n_events, which='ref', event_offset=0):
    """
    Load events from the pz_degeneracy dataset .npz file.

    which='ref'  → ref_imgs / ref_moms  (real events with |pz| > 100 MeV/c)
    which='pz0'  → pz0_imgs / pz0_moms (MCMC-found pz=0 counterparts)
    """
    data = np.load(npz_path)
    if which == 'pz0':
        imgs = data['pz0_imgs']
        moms = data['pz0_moms']
    else:
        imgs = data['ref_imgs']
        moms = data['ref_moms']

    total = len(imgs)
    end   = min(event_offset + n_events, total)
    if event_offset >= total:
        raise RuntimeError(
            f"event_offset={event_offset} >= dataset size={total}"
        )
    if event_offset + n_events > total:
        print(f"[info] only {total - event_offset} events available after offset "
              f"(requested {n_events}) — using all.")

    events = []
    for i in range(event_offset, end):
        raw_img       = imgs[i].astype(np.float32)
        true_momentum = tuple(float(v) for v in moms[i])
        events.append((raw_img, true_momentum))
    return events


def _load_kinked_events(npz_path, n_events, event_offset=0):
    """
    Load events from the kinked-tracks dataset (.npz with keys 'images', 'momenta').

    images  : (N, 64, 64) float array
    momenta : (N, 3) float array in MeV  (stored un-normalised, same convention as
              the standard single-track dataset)
    """
    data  = np.load(npz_path)
    imgs  = data['images']    # (N, H, W)
    moms  = data['momenta']   # (N, 3)  MeV

    total = len(imgs)
    end   = min(event_offset + n_events, total)
    if event_offset >= total:
        raise RuntimeError(
            f"event_offset={event_offset} >= kinked dataset size={total}"
        )
    if event_offset + n_events > total:
        print(f"[info] only {total - event_offset} kinked events available after offset "
              f"(requested {n_events}) — using all.")

    events = []
    for i in range(event_offset, end):
        raw_img       = imgs[i].astype(np.float32)
        true_momentum = tuple(float(v) for v in moms[i])
        events.append((raw_img, true_momentum))
    return events


def _run_1track_inference(events, ev_prefix, shared_kwargs, keep_img_path, use_pmag=False):
    """
    Run single-track gradient inference on a list of (raw_img, true_momentum) events.
    Returns (final_emds, final_mom_arr, truth_mom_arr) as numpy arrays.
    """
    final_emds    = []
    all_final_mom = []
    all_truth_mom = []

    for ev_i, (raw_img, true_momentum) in enumerate(events):
        print(f"\n{'='*70}")
        print(f"Event {ev_i + 1}/{len(events)}")

        hull_guess = convex_hull_initial_guess(raw_img, n_tracks=1, use_pmag_estimate=use_pmag)
        if hull_guess is None:
            print("[hull] construction failed — falling back to random init")
        else:
            px0, py0, pz0 = hull_guess[0]
            tag = f"  |p|={float(np.sqrt(px0**2+py0**2+pz0**2)):.1f}" if use_pmag else ""
            print(f"[hull] init: ({px0:.1f}, {py0:.1f}, {pz0:.1f}){tag}")

        results = run_inference(
            target_img      = raw_img,
            true_momentum   = [true_momentum],
            n_tracks        = 1,
            run_name        = f"{ev_prefix}_{ev_i:03d}",
            initial_momenta = hull_guess,
            **shared_kwargs,
        )
        final_emds.append(results['dist_path'][-1])
        all_final_mom.append(np.array(results['mom_path'][-1], dtype=np.float32))
        all_truth_mom.append(np.array(true_momentum, dtype=np.float32))

        best_idx = int(np.argmin(results['dist_path']))
        np.save(os.path.join(results['data_dir'], 'final_img.npy'),
                results['img_path'][best_idx].numpy())

        if not keep_img_path:
            _maybe_delete(os.path.join(results['data_dir'], 'img_path.pt'))

    return (
        np.array(final_emds),
        np.array(all_final_mom, dtype=np.float32),
        np.array(all_truth_mom, dtype=np.float32),
    )


def _load_single_track_events(dataset_dir, n_events, event_offset=0):
    """Return list of (raw_img, true_momentum) from a batch dataset directory."""
    events = []
    loaded_batch_idx = None
    img_batch = mom_batch = None

    for ev_i in range(n_events):
        global_event = event_offset + ev_i
        b_idx = 0 if img_batch is None else global_event // len(img_batch)
        e_idx = global_event if img_batch is None else global_event % len(img_batch)

        if b_idx != loaded_batch_idx:
            img_batch = np.load(os.path.join(dataset_dir, f'batch_{b_idx}.npy'))
            mom_batch = np.load(os.path.join(dataset_dir, f'batch_mom_{b_idx}.npy'))
            loaded_batch_idx = b_idx
            b_idx = global_event // len(img_batch)
            e_idx = global_event % len(img_batch)
            if b_idx != loaded_batch_idx:
                img_batch = np.load(os.path.join(dataset_dir, f'batch_{b_idx}.npy'))
                mom_batch = np.load(os.path.join(dataset_dir, f'batch_mom_{b_idx}.npy'))
                loaded_batch_idx = b_idx

        raw_img       = img_batch[e_idx].astype(np.float32)
        true_momentum = tuple(float(v) for v in mom_batch[e_idx])
        events.append((raw_img, true_momentum))

    return events


def _sample_angle_pairs(pairs_path, n_events, angle_min, angle_max, seed=None):
    """
    Load angle pairs, filter to [angle_min, angle_max] deg, and return n_events
    pairs sampled uniformly across the range via stratified bin sampling.
    """
    all_pairs = np.load(pairs_path, allow_pickle=True)
    in_range  = [p for p in all_pairs
                 if angle_min <= float(p['separation']) <= angle_max]

    if not in_range:
        raise RuntimeError(
            f"No pairs found in [{angle_min}, {angle_max}] deg in {pairs_path}"
        )

    if n_events >= len(in_range):
        print(f"[info] {n_events} events requested but only {len(in_range)} "
              f"available in range — using all.")
        return sorted(in_range, key=lambda p: float(p['separation']))

    rng       = np.random.default_rng(seed)
    bin_edges = np.linspace(angle_min, angle_max, n_events + 1)
    sampled   = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include the upper edge in the last bin
        if hi == angle_max:
            bucket = [p for p in in_range if lo <= float(p['separation']) <= hi]
        else:
            bucket = [p for p in in_range if lo <= float(p['separation']) < hi]
        if bucket:
            sampled.append(bucket[rng.integers(len(bucket))])

    # Fill any gap caused by empty bins by drawing from unselected pairs
    if len(sampled) < n_events:
        used      = {id(p) for p in sampled}
        remaining = [p for p in in_range if id(p) not in used]
        n_extra   = min(n_events - len(sampled), len(remaining))
        idxs      = rng.choice(len(remaining), size=n_extra, replace=False)
        sampled.extend(remaining[i] for i in idxs)

    return sorted(sampled, key=lambda p: float(p['separation']))


# ── Summary plots ────────────────────────────────────────────────────────────────

def _plot_emd_vs_angle(angles, final_emds, min_distance, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(angles, final_emds, color='steelblue', s=70, zorder=3)
    ax.axhline(min_distance, color='red', linestyle='--', linewidth=1.2,
               label=f'Convergence cut (EMD = {min_distance})')
    ax.set_xlabel('Separation Angle (deg)', fontsize=13)
    ax.set_ylabel('Final EMD', fontsize=13)
    ax.set_title('2-Track Inference: Final EMD vs Separation Angle', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def _plot_loss_distribution(final_emds, min_distance, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = max(5, len(final_emds) // 5)
    ax.hist(final_emds, bins=n_bins, color='steelblue', edgecolor='white')
    ax.axvline(min_distance, color='red', linestyle='--', linewidth=1.2,
               label=f'Convergence cut ({min_distance})')
    ax.axvline(float(np.median(final_emds)), color='orange', linestyle='-', linewidth=1.5,
               label=f'Median = {np.median(final_emds):.4f}')
    ax.set_xlabel('Final EMD', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── Recovered-vs-truth summary plots (single-track) ─────────────────────────────

def _plot_recovered_vs_truth(final_mom, truth_mom, out_path):
    """Scatter of recovered vs truth for px, py, pz, |p|."""
    mag_final = np.linalg.norm(final_mom, axis=1)
    mag_truth = np.linalg.norm(truth_mom, axis=1)

    cols   = [truth_mom[:, 0], truth_mom[:, 1], truth_mom[:, 2], mag_truth]
    rows   = [final_mom[:, 0], final_mom[:, 1], final_mom[:, 2], mag_final]
    labels = [r'$p_x$', r'$p_y$', r'$p_z$', r'$|\mathbf{p}|$']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, xdata, ydata, lbl in zip(axes, cols, rows, labels):
        lo = min(xdata.min(), ydata.min())
        hi = max(xdata.max(), ydata.max())
        ax.scatter(xdata, ydata, s=20, alpha=0.5, rasterized=True)
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2)
        corr = np.corrcoef(xdata, ydata)[0, 1] if len(xdata) > 1 else float('nan')
        ax.set_xlabel(f'Truth {lbl}  [MeV]', fontsize=11)
        ax.set_ylabel(f'Recovered {lbl}  [MeV]', fontsize=11)
        ax.set_title(f'r = {corr:.3f}', fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Recovered vs truth momentum', y=1.01, fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _plot_residual_histograms(final_mom, truth_mom, out_path):
    """Histograms of Δpx, Δpy, Δpz, Δ|p|."""
    delta     = final_mom - truth_mom
    mag_final = np.linalg.norm(final_mom, axis=1)
    mag_truth = np.linalg.norm(truth_mom, axis=1)
    delta_mag = mag_final - mag_truth

    labels = [r'$\Delta p_x$', r'$\Delta p_y$', r'$\Delta p_z$', r'$\Delta|\mathbf{p}|$']
    deltas = [delta[:, 0], delta[:, 1], delta[:, 2], delta_mag]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, lbl, d in zip(axes, labels, deltas):
        n_bins = max(5, len(d) // 5)
        ax.hist(d, bins=n_bins, edgecolor='none', alpha=0.8)
        ax.axvline(0,        color='k',    lw=1.2, ls='--', label='zero')
        ax.axvline(d.mean(), color='blue', lw=1.5, ls='-',  label=f'mean={d.mean():.1f}')
        ax.set_xlabel(f'{lbl}  [MeV]', fontsize=11)
        ax.set_ylabel('events', fontsize=11)
        ax.set_title(f'MAE={np.abs(d).mean():.1f}  RMS={np.sqrt((d**2).mean()):.1f}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Momentum residuals: recovered − truth', y=1.01, fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _plot_emd_vs_momentum(final_emds, truth_mom, out_path):
    """Scatter of final EMD vs each truth momentum component and |p|."""
    mag_truth = np.linalg.norm(truth_mom, axis=1)

    xlabels = [r'$p_x$  [MeV]', r'$p_y$  [MeV]', r'$p_z$  [MeV]', r'$|\mathbf{p}|$  [MeV]']
    xdata   = [truth_mom[:, 0], truth_mom[:, 1], truth_mom[:, 2], mag_truth]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, x, xlabel in zip(axes, xdata, xlabels):
        ax.scatter(x, final_emds, s=10, alpha=0.4, rasterized=True)
        corr = np.corrcoef(x, final_emds)[0, 1] if len(x) > 1 else float('nan')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Final EMD', fontsize=11)
        ax.set_title(f'r = {corr:.3f}', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Final EMD vs truth momentum', y=1.01, fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _make_summary_plots(final_emds, truth_mom_arr, final_mom_arr,
                        min_distance, n_tracks, summary_dir, angles=None):
    """Generate all summary plots from arrays. Called both after inference and in --plot_only mode."""
    _plot_loss_distribution(
        final_emds.tolist(), min_distance,
        out_path=os.path.join(summary_dir, 'plot_loss_distribution.png'),
        title=f'Final EMD Distribution ({n_tracks}-track, {len(final_emds)} events)',
    )

    # Recovered-vs-truth plots require shape (n_events, 3); skip for multi-track
    if (final_mom_arr is not None and truth_mom_arr is not None
            and final_mom_arr.ndim == 2 and truth_mom_arr.ndim == 2):
        _plot_recovered_vs_truth(
            final_mom_arr, truth_mom_arr,
            out_path=os.path.join(summary_dir, 'plot_recovered_vs_truth.png'),
        )
        _plot_residual_histograms(
            final_mom_arr, truth_mom_arr,
            out_path=os.path.join(summary_dir, 'plot_residual_histograms.png'),
        )
        _plot_emd_vs_momentum(
            final_emds, truth_mom_arr,
            out_path=os.path.join(summary_dir, 'plot_emd_vs_momentum.png'),
        )

    if n_tracks == 2 and angles is not None and len(angles):
        _plot_emd_vs_angle(
            angles, final_emds.tolist(), min_distance,
            out_path=os.path.join(summary_dir, 'plot_emd_vs_angle.png'),
        )


# ── Dual-colour summary plots (pz_both mode) ─────────────────────────────────

_REF_COLOR = 'steelblue'
_PZ0_COLOR = 'crimson'


def _plot_loss_distribution_dual(emds_ref, emds_pz0, min_distance, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = max(5, (len(emds_ref) + len(emds_pz0)) // 8)
    ax.hist(emds_ref, bins=n_bins, color=_REF_COLOR, edgecolor='white', alpha=0.65,
            label=f'ref |pz|>100  (N={len(emds_ref)})')
    ax.hist(emds_pz0, bins=n_bins, color=_PZ0_COLOR, edgecolor='white', alpha=0.65,
            label=f'pz0  (N={len(emds_pz0)})')
    ax.axvline(min_distance, color='black', linestyle='--', linewidth=1.2,
               label=f'Convergence cut ({min_distance})')
    ax.axvline(float(np.median(emds_ref)), color=_REF_COLOR, linestyle='-', linewidth=1.8,
               label=f'ref median = {np.median(emds_ref):.4f}')
    ax.axvline(float(np.median(emds_pz0)), color=_PZ0_COLOR, linestyle='-', linewidth=1.8,
               label=f'pz0 median = {np.median(emds_pz0):.4f}')
    ax.set_xlabel('Final EMD', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Final EMD Distribution: ref vs pz0', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def _plot_recovered_vs_truth_dual(final_mom_ref, truth_mom_ref,
                                   final_mom_pz0, truth_mom_pz0, out_path):
    mag_fr = np.linalg.norm(final_mom_ref, axis=1)
    mag_tr = np.linalg.norm(truth_mom_ref, axis=1)
    mag_fp = np.linalg.norm(final_mom_pz0, axis=1)
    mag_tp = np.linalg.norm(truth_mom_pz0, axis=1)

    labels    = [r'$p_x$', r'$p_y$', r'$p_z$', r'$|\mathbf{p}|$']
    truth_ref = [truth_mom_ref[:, 0], truth_mom_ref[:, 1], truth_mom_ref[:, 2], mag_tr]
    reco_ref  = [final_mom_ref[:, 0], final_mom_ref[:, 1], final_mom_ref[:, 2], mag_fr]
    truth_pz0 = [truth_mom_pz0[:, 0], truth_mom_pz0[:, 1], truth_mom_pz0[:, 2], mag_tp]
    reco_pz0  = [final_mom_pz0[:, 0], final_mom_pz0[:, 1], final_mom_pz0[:, 2], mag_fp]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, xr, yr, xp, yp, lbl in zip(axes, truth_ref, reco_ref, truth_pz0, reco_pz0, labels):
        lo = min(xr.min(), yr.min(), xp.min(), yp.min())
        hi = max(xr.max(), yr.max(), xp.max(), yp.max())
        ax.scatter(xr, yr, s=20, alpha=0.5, color=_REF_COLOR, label='ref', rasterized=True)
        ax.scatter(xp, yp, s=20, alpha=0.5, color=_PZ0_COLOR, label='pz0', rasterized=True)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2)
        corr_r = np.corrcoef(xr, yr)[0, 1] if len(xr) > 1 else float('nan')
        corr_p = np.corrcoef(xp, yp)[0, 1] if len(xp) > 1 else float('nan')
        ax.set_xlabel(f'Truth {lbl}  [MeV]', fontsize=11)
        ax.set_ylabel(f'Recovered {lbl}  [MeV]', fontsize=11)
        ax.set_title(f'r(ref)={corr_r:.3f}  r(pz0)={corr_p:.3f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Recovered vs truth momentum', y=1.01, fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _plot_residual_histograms_dual(final_mom_ref, truth_mom_ref,
                                    final_mom_pz0, truth_mom_pz0, out_path):
    delta_r   = final_mom_ref - truth_mom_ref
    delta_p   = final_mom_pz0 - truth_mom_pz0
    dmag_r    = np.linalg.norm(final_mom_ref, axis=1) - np.linalg.norm(truth_mom_ref, axis=1)
    dmag_p    = np.linalg.norm(final_mom_pz0, axis=1) - np.linalg.norm(truth_mom_pz0, axis=1)

    labels = [r'$\Delta p_x$', r'$\Delta p_y$', r'$\Delta p_z$', r'$\Delta|\mathbf{p}|$']
    sets_r = [delta_r[:, 0], delta_r[:, 1], delta_r[:, 2], dmag_r]
    sets_p = [delta_p[:, 0], delta_p[:, 1], delta_p[:, 2], dmag_p]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, lbl, dr, dp in zip(axes, labels, sets_r, sets_p):
        n_bins = max(5, (len(dr) + len(dp)) // 8)
        ax.hist(dr, bins=n_bins, color=_REF_COLOR, edgecolor='none', alpha=0.65, label='ref')
        ax.hist(dp, bins=n_bins, color=_PZ0_COLOR, edgecolor='none', alpha=0.65, label='pz0')
        ax.axvline(0,          color='k',         lw=1.2, ls='--')
        ax.axvline(dr.mean(),  color=_REF_COLOR,  lw=1.5, ls='-',
                   label=f'ref μ={dr.mean():.1f}')
        ax.axvline(dp.mean(),  color=_PZ0_COLOR,  lw=1.5, ls='-',
                   label=f'pz0 μ={dp.mean():.1f}')
        ax.set_xlabel(f'{lbl}  [MeV]', fontsize=11)
        ax.set_ylabel('events', fontsize=11)
        mae_r = np.abs(dr).mean(); mae_p = np.abs(dp).mean()
        ax.set_title(
            f'MAE ref={mae_r:.1f}  pz0={mae_p:.1f}', fontsize=10
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Momentum residuals: recovered − truth', y=1.01, fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _plot_emd_vs_momentum_dual(emds_ref, truth_mom_ref,
                                emds_pz0, truth_mom_pz0, out_path):
    mag_tr = np.linalg.norm(truth_mom_ref, axis=1)
    mag_tp = np.linalg.norm(truth_mom_pz0, axis=1)

    xlabels   = [r'$p_x$  [MeV]', r'$p_y$  [MeV]', r'$p_z$  [MeV]', r'$|\mathbf{p}|$  [MeV]']
    xdata_ref = [truth_mom_ref[:, 0], truth_mom_ref[:, 1], truth_mom_ref[:, 2], mag_tr]
    xdata_pz0 = [truth_mom_pz0[:, 0], truth_mom_pz0[:, 1], truth_mom_pz0[:, 2], mag_tp]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, xr, xp, xlabel in zip(axes, xdata_ref, xdata_pz0, xlabels):
        ax.scatter(xr, emds_ref, s=10, alpha=0.5, color=_REF_COLOR,
                   label='ref', rasterized=True)
        ax.scatter(xp, emds_pz0, s=10, alpha=0.5, color=_PZ0_COLOR,
                   label='pz0', rasterized=True)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Final EMD', fontsize=11)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('Final EMD vs truth momentum', y=1.01, fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def _make_pz_both_summary_plots(emds_ref, final_mom_ref, truth_mom_ref,
                                  emds_pz0, final_mom_pz0, truth_mom_pz0,
                                  min_distance, summary_dir):
    _plot_loss_distribution_dual(
        emds_ref, emds_pz0, min_distance,
        out_path=os.path.join(summary_dir, 'plot_loss_distribution.png'),
    )
    _plot_recovered_vs_truth_dual(
        final_mom_ref, truth_mom_ref, final_mom_pz0, truth_mom_pz0,
        out_path=os.path.join(summary_dir, 'plot_recovered_vs_truth.png'),
    )
    _plot_residual_histograms_dual(
        final_mom_ref, truth_mom_ref, final_mom_pz0, truth_mom_pz0,
        out_path=os.path.join(summary_dir, 'plot_residual_histograms.png'),
    )
    _plot_emd_vs_momentum_dual(
        emds_ref, truth_mom_ref, emds_pz0, truth_mom_pz0,
        out_path=os.path.join(summary_dir, 'plot_emd_vs_momentum.png'),
    )


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run gradient-guided LDM inference on N events and collect result plots.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Event source
    parser.add_argument('--n_events', type=int, default=5,
                        help='Number of events to run')
    parser.add_argument('--n_tracks', type=int, default=1, choices=[1, 2],
                        help='1 = single-track; 2 = multi-track (angle dataset)')
    parser.add_argument('--dataset', type=str, default=None,
                        help=f'Single-track dataset dir (default: {DEFAULT_SINGLE_DATASET})')
    parser.add_argument('--event_offset', type=int, default=0,
                        help='Start index within the dataset')
    parser.add_argument('--kinked', action='store_true', default=False,
                        help='Use the kinked-tracks dataset (.npz with "images"/"momenta" keys)')
    parser.add_argument('--kinked_dataset', type=str, default=DEFAULT_KINKED_DATASET,
                        help=f'Path to kinked_tracks.npz (default: {DEFAULT_KINKED_DATASET})')
    parser.add_argument('--line_kinked', action='store_true', default=False,
                        help='Use the line-kinked-tracks dataset (.npz with "images"/"momenta" keys)')
    parser.add_argument('--line_kinked_dataset', type=str, default=DEFAULT_LINE_KINKED_DATASET_SMALL,
                        help=f'Path to line_kinked_tracks .npz '
                             f'(default: {DEFAULT_LINE_KINKED_DATASET}; '
                             f'64-event subset: {DEFAULT_LINE_KINKED_DATASET_SMALL})')
    parser.add_argument('--pz_degeneracy', action='store_true', default=False,
                        help='Use the pz degeneracy dataset instead of the standard single-track dataset')
    parser.add_argument('--pz_degeneracy_dataset', type=str, default=DEFAULT_PZ_DEGENERACY_DATASET,
                        help='Path to pz_degeneracy_dataset.npz')
    parser.add_argument('--pz_which', type=str, default='ref', choices=['ref', 'pz0'],
                        help='Which images to use from the pz degeneracy dataset: '
                             '"ref" = real events with |pz|>100 (default); '
                             '"pz0" = MCMC-found pz=0 counterparts')
    parser.add_argument('--pz_both', action='store_true', default=False,
                        help='Run inference on BOTH ref and pz0 subsets of the pz degeneracy '
                             'dataset and produce dual-colour summary plots (blue=ref, red=pz0)')
    parser.add_argument('--pairs_path', type=str, default=DEFAULT_ANGLE_PAIRS,
                        help='Angle pairs .npy for 2-track mode') 
    parser.add_argument('--angle_min', type=float, default=0.0,
                        help='Min separation angle (deg) for 2-track sampling')
    parser.add_argument('--angle_max', type=float, default=30.0,
                        help='Max separation angle (deg) for 2-track sampling')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for angle stratified sampling')

    # Output
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Root output directory (defaults to test_inference/results/)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run subfolder name (auto-generated if omitted)')
    parser.add_argument('--no_plots', dest='save_plots', action='store_false', default=True,
                        help='Skip per-event diagnostic plots')
    parser.add_argument('--keep_img_path', action='store_true', default=False,
                        help='Keep img_path.pt per event (large; deleted by default)')
    parser.add_argument('--plot_only', action='store_true', default=False,
                        help='Load existing summary.npz and regenerate summary plots without rerunning inference')

    # Inference hyperparameters
    parser.add_argument('--n_iterations', type=int, default=39)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_min', type=float, default=0.001)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--min_distance', type=float, default=0.05)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'adam'])
    parser.add_argument('--batch_size', type=int, default=32,
                        help='avg_grad batch size; reduce if GPU memory is tight with --newton_pz')
    parser.add_argument('--newton_pz', action='store_true', default=False,
                        help='Scale pz gradient by diagonal Hessian inverse (costs one extra backward per step)')
    parser.add_argument('--use_pmag', action='store_true', default=False,
                        help='Seed initial |p| from image pixel-sum energy estimate '
                             '(gives non-zero pz start correlated with track length)')
    parser.add_argument('--reparam_pmag', action='store_true', default=False,
                        help='Optimise (px, py, |p|) and derive pz from |p|; '
                             'pairs naturally with --use_pmag')
    parser.add_argument('--init_pz', type=float, default=None,
                        help='Override initial pz (MeV) for every track, replacing the hull/random guess '
                             '(e.g. --init_pz 100 to break the pz=0 saddle)')
    parser.add_argument('--loss', type=str, default='emd',
                        help='Loss: emd, w2, energy, gaussian_mmd, laplacian_mmd, l2')
    parser.add_argument('--renoise', type=float, default=0.0,
                        help='Re-noise fraction [0, 1]: after each SGD step, encode the current '
                             'best image to latent space, noise it to T = renoise × T_max, and '
                             'start the next DDIM generation from that noised latent instead of '
                             'pure Gaussian noise.  0 (default) = disabled.  With a non-constant '
                             '--renoise_schedule this is the starting fraction.')
    parser.add_argument('--renoise_schedule', type=str, default='constant',
                        choices=['constant', 'fixed', 'lr', 'linear', 'cosine'],
                        help='How the renoise fraction evolves from --renoise to --renoise_min: '
                             '"constant"/"fixed" keep it at --renoise every step; "lr" tracks the cosine LR schedule (lr_min ↦ renoise_min, '
                             'lr ↦ renoise); "linear"/"cosine" decay over n_iterations.')
    parser.add_argument('--renoise_min', type=float, default=0.1,
                        help='Final renoise fraction for non-constant --renoise_schedule.')
    parser.add_argument('--renoise_truth_init', action='store_true', default=False,
                        help='Seed the first renoise step from the truth image instead of pure '
                             'noise (requires --renoise > 0).  From step 2 onward the generator\'s '
                             'own output is used as usual.')
    parser.add_argument('--renoise_truth_every_step', action='store_true', default=False,
                        help='Renoise from the truth image on every SGD step (requires --renoise > 0). '
                             'z_clean is never updated from the generator output — the truth latent '
                             'is renoised fresh each iteration.')

    args = parser.parse_args()
    t0 = time.time()

    # ── Plot-only mode ───────────────────────────────────────────────────────────
    if args.plot_only:
        if args.run_name is None:
            # pick the most recently modified run in output_dir
            candidates = [
                d for d in os.listdir(args.output_dir)
                if os.path.isdir(os.path.join(args.output_dir, d))
            ]
            if not candidates:
                raise RuntimeError(f"No run directories found in {args.output_dir}")
            args.run_name = max(
                candidates,
                key=lambda d: os.path.getmtime(os.path.join(args.output_dir, d)),
            )
            print(f"[plot_only] auto-selected run: {args.run_name}")

        summary_dir = os.path.join(args.output_dir, args.run_name, 'summary')
        npz_path    = os.path.join(summary_dir, 'summary.npz')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"No summary.npz found at {npz_path}")

        print(f"[plot_only] loading {npz_path}")
        data = np.load(npz_path)
        final_emds    = data['final_emds']
        final_mom_arr = data['final_momentum'] if 'final_momentum' in data else None
        truth_mom_arr = data['truth_momentum'] if 'truth_momentum' in data else None
        angles        = data['angles'].tolist() if 'angles' in data else None

        n_tracks = args.n_tracks
        if angles is not None:
            n_tracks = 2

        # Fall back to per-event files when summary.npz pre-dates this change
        if final_mom_arr is None or truth_mom_arr is None:
            events_dir = os.path.join(args.output_dir, args.run_name, 'events')
            final_moms, truth_moms = [], []
            for ev_i in range(len(final_emds)):
                ev_dir = os.path.join(events_dir, f'event_{ev_i:03d}', 'data_files')
                mom_path_file  = os.path.join(ev_dir, 'mom_path.npy')
                truth_mom_file = os.path.join(ev_dir, 'truth_mom.npy')
                if os.path.exists(mom_path_file) and os.path.exists(truth_mom_file):
                    mp = np.load(mom_path_file)   # (n_steps, ...) — last step is final
                    tm = np.load(truth_mom_file)
                    final_moms.append(mp[-1].astype(np.float32))
                    truth_moms.append(tm.astype(np.float32))
                else:
                    print(f"[plot_only] warning: missing data files for event_{ev_i:03d}, skipping momentum plots")
                    final_moms, truth_moms = [], []
                    break
            if final_moms:
                final_mom_arr = np.array(final_moms)
                truth_mom_arr = np.array(truth_moms)
                # truth_mom.npy is (1,3) for 1-track → stacked becomes (n_events,1,3); squeeze to (n_events,3)
                if truth_mom_arr.ndim == 3 and truth_mom_arr.shape[1] == 1:
                    truth_mom_arr = truth_mom_arr[:, 0, :]
                print(f"[plot_only] reconstructed momenta from per-event files "
                      f"(final_mom shape={final_mom_arr.shape})")

        print(f"[plot_only] {len(final_emds)} events, n_tracks={n_tracks}")
        _make_summary_plots(
            final_emds, truth_mom_arr, final_mom_arr,
            args.min_distance, n_tracks, summary_dir,
            angles=np.array(angles) if angles is not None else None,
        )
        _save_emd_ranking(final_emds, summary_dir,
                          angles=np.array(angles) if angles is not None else None)
        print(f"\nPlots regenerated in {os.path.abspath(summary_dir)}/")
        return

    # ── Resolve paths ────────────────────────────────────────────────────────────
    if args.run_name is None:
        if args.pz_both:
            args.run_name = f"pzdeg_both_{time.strftime('%Y%m%d_%H%M%S')}"
        elif args.pz_degeneracy:
            args.run_name = f"pzdeg_{args.pz_which}_{time.strftime('%Y%m%d_%H%M%S')}"
        elif args.line_kinked:
            args.run_name = f"line_kinked_{time.strftime('%Y%m%d_%H%M%S')}"
        elif args.kinked:
            args.run_name = f"kinked_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            args.run_name = f"{args.n_tracks}track_{time.strftime('%Y%m%d_%H%M%S')}"

    run_root    = os.path.join(args.output_dir, args.run_name)
    summary_dir = os.path.join(run_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    print(f"Output root : {os.path.abspath(run_root)}")

    # ── Device & model ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}\n")

    ldm = load_ldm(device)
    generator = DifferentiableLDMGenerator(
        ldm, device=str(device),
        ddim_steps_standard=50,
        ddim_steps_gradient=10,
    )

    shared_kwargs = dict(
        n_iterations        = args.n_iterations,
        learning_rate       = args.lr,
        lr_min              = args.lr_min,
        gradient_clip       = args.gradient_clip,
        min_distance        = args.min_distance,
        optimizer_type      = args.optimizer,
        avg_grad_batch_size = args.batch_size,
        device              = str(device),
        generator           = generator,
        save_plots          = args.save_plots,
        loss_type           = args.loss,
        verbose             = True,
        output_dir          = os.path.join(run_root, 'events'),
        newton_pz           = args.newton_pz,
        reparam_pmag        = args.reparam_pmag,
        init_pz             = args.init_pz,
        renoise             = args.renoise,
        renoise_truth_init        = args.renoise_truth_init,
        renoise_truth_every_step  = args.renoise_truth_every_step,
        renoise_schedule          = args.renoise_schedule,
        renoise_min               = args.renoise_min,
    )

    final_emds    = []
    angles        = []   # 2-track mode only
    all_final_mom = []   # (n_events, 3) for 1-track; (n_events, 2, 3) for 2-track
    all_truth_mom = []   # same shape

    # ── Single-track mode ────────────────────────────────────────────────────────
    if args.n_tracks == 1:
        npz_path = args.pz_degeneracy_dataset

        # ── pz_both: run ref and pz0 subsets, dual-colour summary ────────────
        if args.pz_both:
            print(f"=== pz_degeneracy BOTH mode ===")
            print(f"Dataset: {npz_path}\n")

            for which, color in (('ref', 'blue'), ('pz0', 'red')):
                print(f"\n{'#'*70}")
                print(f"# Running subset: {which}  ({color})")
                print(f"{'#'*70}")
                events = _load_pz_degeneracy_events(
                    npz_path, args.n_events, which=which,
                    event_offset=args.event_offset,
                )
                subset_kwargs = {**shared_kwargs,
                                 'output_dir': os.path.join(run_root, 'events', which)}
                emds, fmom, tmom = _run_1track_inference(
                    events, f'event', subset_kwargs, args.keep_img_path, use_pmag=args.use_pmag,
                )
                if which == 'ref':
                    emds_ref, final_mom_ref, truth_mom_ref = emds, fmom, tmom
                else:
                    emds_pz0, final_mom_pz0, truth_mom_pz0 = emds, fmom, tmom

            print(f"\n{'='*70}")
            print(f"pz_both complete — {len(emds_ref)} ref  +  {len(emds_pz0)} pz0 events")
            for label, emds in (('ref', emds_ref), ('pz0', emds_pz0)):
                conv = (emds < args.min_distance).mean() * 100
                print(f"  {label}: mean={emds.mean():.4f}  median={np.median(emds):.4f}"
                      f"  converged={conv:.1f}%")

            np.savez(
                os.path.join(summary_dir, 'summary.npz'),
                final_emds_ref   = emds_ref,
                final_emds_pz0   = emds_pz0,
                final_momentum_ref = final_mom_ref,
                final_momentum_pz0 = final_mom_pz0,
                truth_momentum_ref = truth_mom_ref,
                truth_momentum_pz0 = truth_mom_pz0,
            )

            _save_emd_ranking(emds_ref,  summary_dir, suffix='_ref')
            _save_emd_ranking(emds_pz0,  summary_dir, suffix='_pz0')

            _make_pz_both_summary_plots(
                emds_ref, final_mom_ref, truth_mom_ref,
                emds_pz0, final_mom_pz0, truth_mom_pz0,
                args.min_distance, summary_dir,
            )
            print(f"\nSummary saved to {os.path.abspath(summary_dir)}/")
            return

        # ── single subset ─────────────────────────────────────────────────────
        if args.line_kinked:
            print(f"Loading {args.n_events} line-kinked-track events (offset={args.event_offset}) "
                  f"from:\n  {args.line_kinked_dataset}\n")
            events = _load_kinked_events(
                args.line_kinked_dataset, args.n_events, event_offset=args.event_offset,
            )
        elif args.kinked:
            print(f"Loading {args.n_events} kinked-track events (offset={args.event_offset}) "
                  f"from:\n  {args.kinked_dataset}\n")
            events = _load_kinked_events(
                args.kinked_dataset, args.n_events, event_offset=args.event_offset,
            )
        elif args.pz_degeneracy:
            print(f"Loading {args.n_events} events from pz_degeneracy dataset "
                  f"(which='{args.pz_which}'):\n  {npz_path}\n")
            events = _load_pz_degeneracy_events(
                npz_path, args.n_events, which=args.pz_which,
                event_offset=args.event_offset,
            )
        else:
            dataset_dir = args.dataset or DEFAULT_SINGLE_DATASET
            print(f"Loading {args.n_events} single-track events from:\n  {dataset_dir}\n")
            events = _load_single_track_events(dataset_dir, args.n_events, args.event_offset)

        emds, fmom, tmom = _run_1track_inference(
            events, 'event', shared_kwargs, args.keep_img_path, use_pmag=args.use_pmag,
        )
        final_emds.extend(emds.tolist())
        all_final_mom.extend(fmom.tolist())
        all_truth_mom.extend(tmom.tolist())

    # ── Multi-track (angle sweep) mode ───────────────────────────────────────────
    else:
        print(f"Sampling {args.n_events} angle pairs from "
              f"[{args.angle_min}, {args.angle_max}] deg")
        print(f"Pairs file : {args.pairs_path}\n")
        pairs = _sample_angle_pairs(
            args.pairs_path, args.n_events,
            args.angle_min, args.angle_max, args.seed,
        )
        selected_angles = [f'{float(p["separation"]):.1f}°' for p in pairs]
        print(f"Selected {len(pairs)} pairs at angles: {selected_angles}\n")

        for ev_i, pair in enumerate(pairs):
            angle         = float(pair['separation'])
            target_img    = np.array(pair['sum_image'], dtype=np.float32)
            true_momentum = [pair['event1']['momentum'], pair['event2']['momentum']]

            print(f"\n{'='*70}")
            print(f"Event {ev_i + 1}/{len(pairs)}  (separation = {angle:.2f} deg)")

            hull_guess = convex_hull_initial_guess(target_img, n_tracks=2, use_pmag_estimate=args.use_pmag)
            if hull_guess is None:
                print("[hull] construction failed — falling back to random init")
            else:
                for ti, (px0, py0, pz0) in enumerate(hull_guess):
                    tag = f"  |p|={float(np.sqrt(px0**2+py0**2+pz0**2)):.1f}" if args.use_pmag else ""
                    print(f"[hull] init t{ti+1}: ({px0:.1f}, {py0:.1f}, {pz0:.1f}){tag}")

            results = run_inference(
                target_img      = target_img,
                true_momentum   = true_momentum,
                n_tracks        = 2,
                run_name        = f"event_{ev_i:03d}",
                initial_momenta = hull_guess,
                **shared_kwargs,
            )
            final_emds.append(results['dist_path'][-1])
            angles.append(angle)
            # mom_path[-1] is ((px1,py1,pz1),(px2,py2,pz2)) for 2-track
            all_final_mom.append(np.array(results['mom_path'][-1], dtype=np.float32))
            all_truth_mom.append(np.array(true_momentum, dtype=np.float32))

            best_idx = int(np.argmin(results['dist_path']))
            np.save(os.path.join(results['data_dir'], 'final_img.npy'),
                    results['img_path'][best_idx].numpy())

            if not args.keep_img_path:
                _maybe_delete(os.path.join(results['data_dir'], 'img_path.pt'))

    # ── Summary ──────────────────────────────────────────────────────────────────
    final_emds    = np.array(final_emds)
    final_mom_arr = np.array(all_final_mom, dtype=np.float32) if all_final_mom else None
    truth_mom_arr = np.array(all_truth_mom, dtype=np.float32) if all_truth_mom else None

    print(f"\n{'='*70}")
    print(f"Run complete — {len(final_emds)} events")
    print(f"Final EMD   — mean={final_emds.mean():.4f}  median={np.median(final_emds):.4f}"
          f"  min={final_emds.min():.4f}  max={final_emds.max():.4f}")
    conv_pct = (final_emds < args.min_distance).mean() * 100
    print(f"Converged (< {args.min_distance}): {conv_pct:.1f}%")

    save_dict = {'final_emds': final_emds}
    if angles:
        save_dict['angles'] = np.array(angles)
    if final_mom_arr is not None:
        save_dict['final_momentum'] = final_mom_arr
    if truth_mom_arr is not None:
        save_dict['truth_momentum'] = truth_mom_arr
    np.savez(os.path.join(summary_dir, 'summary.npz'), **save_dict)

    _save_emd_ranking(final_emds, summary_dir,
                      angles=np.array(angles) if angles else None)

    _make_summary_plots(
        final_emds, truth_mom_arr, final_mom_arr,
        args.min_distance, args.n_tracks, summary_dir,
        angles=np.array(angles) if angles else None,
    )

    print(f"\nSummary saved to {os.path.abspath(summary_dir)}/")
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"Total wall time: {h:02d}:{m:02d}:{s:02d}  ({elapsed:.1f} s)")


def _save_emd_ranking(final_emds, summary_dir, angles=None, suffix=''):
    idx_sorted = np.argsort(final_emds)[::-1]
    out_path = os.path.join(summary_dir, f'emd_ranking{suffix}.txt')
    with open(out_path, 'w') as f:
        header = ('%-8s  %-12s  %-12s' % ('Event', 'Angle (deg)', 'Final EMD')
                  if angles is not None else
                  '%-8s  %-12s' % ('Event', 'Final EMD'))
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for i in idx_sorted:
            if angles is not None:
                f.write('%-8d  %-12.2f  %-12.6f\n' % (i, angles[i], final_emds[i]))
            else:
                f.write('%-8d  %-12.6f\n' % (i, final_emds[i]))
    print(f"Saved: {out_path}")


def _maybe_delete(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()
