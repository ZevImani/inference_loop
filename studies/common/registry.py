"""
Study registry. A study says how to load its events, how to name its runs, and
how to summarise the results. To add one: write a loader, register it here, and
add configs/studies/<name>.yaml.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

from protoninfer.paths import resolve_path

from . import datasets as ds
from . import summary
from .datasets import Event

# (data config, n_events, event_offset) -> {subset name: events}; '' for a single subset
Loader = Callable[[dict, int, int], Dict[str, List[Event]]]


@dataclass
class Study:
    load: Loader
    run_prefix: Union[str, Callable[[dict], str]]
    summarize: Callable = summary.summarize

    def prefix(self, data):
        return self.run_prefix(data) if callable(self.run_prefix) else self.run_prefix


def _npz_study(loader, label, **loader_kwargs):
    """Single-subset study over an .npz of (image, momentum) events."""
    def load(data, n_events, event_offset):
        path = resolve_path(data['path'])
        print(f"Loading {n_events} {label} events (offset={event_offset}) from:\n  {path}\n")
        kwargs = {k: data[v] for k, v in loader_kwargs.items()}
        pairs = loader(path, n_events, event_offset=event_offset, **kwargs)
        return {'': [Event(img, truth) for img, truth in pairs]}
    return load


def _load_pzboth(data, n_events, event_offset):
    path = resolve_path(data['path'])
    print(f"=== pz_degeneracy BOTH mode ===\nDataset: {path}\n")
    return {
        which: [Event(img, truth) for img, truth in
                ds.load_pz_degeneracy_events(path, n_events, which=which, event_offset=event_offset)]
        for which in ('ref', 'pz0')
    }


def _load_angles(data, n_events, event_offset):
    path = resolve_path(data['path'])
    print(f"Sampling {n_events} angle pairs from [{data['angle_min']}, {data['angle_max']}] deg")
    print(f"Pairs file : {path}\n")
    return {'': ds.load_angle_events(path, n_events, data['angle_min'], data['angle_max'],
                                     seed=data.get('seed'))}


def _load_dual_proj(data, n_events, event_offset):
    path = resolve_path(data['path'])
    print(f"Loading {n_events} dual-projection events (offset={event_offset}) from:\n  {path}\n")
    return {'': ds.load_dual_proj_events(path, n_events, event_offset)}


def _load_dual_kinked(data, n_events, event_offset):
    path = resolve_path(data['path'])
    print(f"Loading {n_events} dual-projection kinked events (offset={event_offset}) from:\n  {path}\n")
    return {'': ds.load_dual_kinked_events(path, n_events, event_offset)}


STUDIES = {
    'val':         Study(_npz_study(ds.load_single_track_events, 'single-track'), '1track'),
    'kinked':      Study(_npz_study(ds.load_kinked_events, 'kinked-track'), 'kinked'),
    'pzdeg':       Study(_npz_study(ds.load_pz_degeneracy_events, 'pz-degeneracy', which='which'),
                         lambda data: f"pzdeg_{data['which']}"),
    'pzboth':      Study(_load_pzboth, 'pzdeg_both', summarize=summary.summarize_pz_both),
    'angles':      Study(_load_angles, '2track'),
    'dual_proj':   Study(_load_dual_proj, 'dual_proj'),
    'dual_kinked': Study(_load_dual_kinked, 'dual_kinked'),
}
