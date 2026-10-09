"""Event loaders for each study; see registry.py for which study uses which loader."""
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Event:
    img: np.ndarray                     # (H, W) target image (xy projection for dual_proj)
    truth: Any                          # (px, py, pz), or a list of them for multi-track events
    n_tracks: int = 1
    angle: Optional[float] = None       # track separation in degrees (angles study)
    xz_img: Optional[np.ndarray] = None # xz projection (dual_proj study)


# ── Loaders returning (raw_img, true_momentum) lists ────────────────────────────

def load_pz_degeneracy_events(npz_path, n_events, which='ref', event_offset=0):
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


def load_kinked_events(npz_path, n_events, event_offset=0):
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


def load_dual_kinked_events(npz_path, n_events, event_offset=0):
    """Kinked tracks with both projections (.npz with 'images' = xy, 'images_xz', 'momenta')."""
    data   = np.load(npz_path)
    xz     = data['images_xz']
    pairs  = load_kinked_events(npz_path, n_events, event_offset)
    return [Event(img=img, truth=truth, xz_img=xz[event_offset + i].astype(np.float32))
            for i, (img, truth) in enumerate(pairs)]


def load_single_track_events(dataset_dir, n_events, event_offset=0):
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


def sample_angle_pairs(pairs_path, n_events, angle_min, angle_max, seed=None):
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


# ── Loaders returning Event lists ───────────────────────────────────────────────

def load_angle_events(pairs_path, n_events, angle_min, angle_max, seed=None):
    """Two-track events: angle pairs stratified across [angle_min, angle_max] degrees."""
    pairs = sample_angle_pairs(pairs_path, n_events, angle_min, angle_max, seed)
    selected_angles = [f'{float(p["separation"]):.1f}°' for p in pairs]
    print(f"Selected {len(pairs)} pairs at angles: {selected_angles}\n")
    return [
        Event(
            img=np.array(p['sum_image'], dtype=np.float32),
            truth=[p['event1']['momentum'], p['event2']['momentum']],
            n_tracks=2,
            angle=float(p['separation']),
        )
        for p in pairs
    ]


def load_dual_proj_events(dataset_dir, n_events, event_offset=0):
    """xy + xz projections of one track, from equal-size batch_N / batch_xz_N / batch_mom_N files."""
    batch_len = len(np.load(os.path.join(dataset_dir, 'batch_mom_0.npy')))
    events, loaded = [], {}
    for global_event in range(event_offset, event_offset + n_events):
        b_idx, e_idx = divmod(global_event, batch_len)
        if b_idx not in loaded:
            loaded = {b_idx: [np.load(os.path.join(dataset_dir, f'{prefix}_{b_idx}.npy'))
                              for prefix in ('batch', 'batch_xz', 'batch_mom')]}
        xy, xz, mom = loaded[b_idx]
        events.append(Event(
            img=xy[e_idx].astype(np.float32),
            truth=tuple(float(v) for v in mom[e_idx]),
            xz_img=xz[e_idx].astype(np.float32),
        ))
    return events
