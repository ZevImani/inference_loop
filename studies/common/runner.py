"""Run a study: load its events, run inference on each, then write the summary."""
import os
import time

import numpy as np
import torch

from protoninfer import DifferentiableLDMGenerator, infer, load_ldm
from protoninfer.config import to_yaml
from protoninfer.paths import resolve_path

from .registry import STUDIES
from .summary import SubsetResult, replot


def _get_study(name):
    if name not in STUDIES:
        raise ValueError(f"Unknown study {name!r}; available: {', '.join(STUDIES)}")
    return STUDIES[name]


def _maybe_delete(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def run_events(events, cfg, generator, device, events_dir):
    """Run inference on each event; per-event outputs go to <events_dir>/event_NNN/."""
    final_emds, final_moms, truth_moms, angles, n_steps, elapsed = [], [], [], [], [], []

    for ev_i, ev in enumerate(events):
        print(f"\n{'='*70}")
        suffix = f"  (separation = {ev.angle:.2f} deg)" if ev.angle is not None else ""
        print(f"Event {ev_i + 1}/{len(events)}{suffix}")

        results = infer(
            ev.img, ev.truth, cfg.inference,
            n_tracks   = ev.n_tracks,
            xz_img     = ev.xz_img,
            generator  = generator,
            device     = str(device),
            output_dir = events_dir,
            run_name   = f"event_{ev_i:03d}",
            save_plots = cfg.save_plots,
        )
        final_emds.append(results['dist_path'][-1])
        n_steps.append(len(results['dist_path']) - 1)   # dist_path[0] is the initial guess
        elapsed.append(results['elapsed'])
        # mom_path[-1] is (px,py,pz) for 1-track, ((px1,py1,pz1),(px2,py2,pz2)) for 2-track
        final_moms.append(np.array(results['mom_path'][-1], dtype=np.float32))
        truth_moms.append(np.array(ev.truth, dtype=np.float32))
        if ev.angle is not None:
            angles.append(ev.angle)

        best_idx = int(np.argmin(results['dist_path']))
        np.save(os.path.join(results['data_dir'], 'final_img.npy'),
                results['img_path'][best_idx].numpy())
        if not cfg.keep_img_path:
            _maybe_delete(os.path.join(results['data_dir'], 'img_path.pt'))

    return SubsetResult(
        final_emds = np.array(final_emds),
        final_mom  = np.array(final_moms, dtype=np.float32),
        truth_mom  = np.array(truth_moms, dtype=np.float32),
        angles     = np.array(angles) if angles else None,
        n_steps    = np.array(n_steps),
        elapsed    = np.array(elapsed),
    )


def run_study(cfg):
    study = _get_study(cfg.study)
    t0 = time.time()
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    if cfg.run_name is None:
        cfg.run_name = f"{study.prefix(cfg.data)}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_root    = os.path.join(resolve_path(cfg.output_dir), cfg.run_name)
    summary_dir = os.path.join(run_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    print(f"Output root : {os.path.abspath(run_root)}")
    with open(os.path.join(run_root, 'config.yaml'), 'w') as f:
        f.write(to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}\n")
    generator = DifferentiableLDMGenerator(
        load_ldm(device), device=str(device),
        ddim_steps_standard=cfg.inference.sampling.ddim_steps_standard,
        ddim_steps_gradient=cfg.inference.sampling.ddim_steps_gradient,
    )

    subsets = study.load(cfg.data, cfg.n_events, cfg.event_offset)
    results = {}
    for name, events in subsets.items():
        if name:
            print(f"\n{'#'*70}\n# Running subset: {name}\n{'#'*70}")
        events_dir = os.path.join(run_root, 'events', name) if name else os.path.join(run_root, 'events')
        results[name] = run_events(events, cfg, generator, device, events_dir)

    study.summarize(results, cfg.inference.min_distance, summary_dir)
    print(f"\nSummary saved to {os.path.abspath(summary_dir)}/")
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"Total wall time: {h:02d}:{m:02d}:{s:02d}  ({elapsed:.1f} s)")


def plot_only(cfg):
    """Regenerate summary plots for cfg.run_name (default: most recently modified run)."""
    output_dir = resolve_path(cfg.output_dir)
    run_name = cfg.run_name
    if run_name is None:
        candidates = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not candidates:
            raise RuntimeError(f"No run directories found in {output_dir}")
        run_name = max(candidates, key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))
        print(f"[plot_only] auto-selected run: {run_name}")
    replot(os.path.join(output_dir, run_name), cfg.inference.min_distance)
