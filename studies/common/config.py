"""
Run configuration for studies: which study, which events, where outputs go, and
the inference hyperparameters.

Layering (later wins):
    configs/paths.yaml               → `paths` (interpolate as ${paths.datasets.val})
    configs/inference/default.yaml   → `inference`
    each -c file                     → e.g. configs/studies/kinked.yaml
    key=value overrides              → e.g. n_events=64 inference.renoise.frac=0.3
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from protoninfer.config import InferenceConfig, merge_configs
from protoninfer.paths import REPO_ROOT, load_paths

INFERENCE_DEFAULTS = os.path.join(REPO_ROOT, 'configs', 'inference', 'default.yaml')
STUDY_CONFIG_DIR   = os.path.join(REPO_ROOT, 'configs', 'studies')


@dataclass
class RunConfig:
    study: str = 'val'                    # key in common.registry.STUDIES
    n_events: int = 5
    event_offset: int = 0                 # start index within the dataset
    seed: Optional[int] = None            # seeds torch + numpy for a reproducible run
    run_name: Optional[str] = None        # default: <study prefix>_<timestamp>
    output_dir: str = '${paths.output_dir}'
    save_plots: bool = False              # per-event diagnostic plots (off: they dominate disk use)
    keep_img_path: bool = False           # keep the large per-event img_path.pt
    data: Dict[str, Any] = field(default_factory=dict)  # study-specific (path, which, angle_min, ...)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: Dict[str, Any] = field(default_factory=dict)


def resolve_config_file(name):
    """Accept a path, or a bare study name such as `kinked` for configs/studies/kinked.yaml."""
    if os.path.exists(name):
        return name
    candidate = os.path.join(STUDY_CONFIG_DIR, name if name.endswith('.yaml') else f'{name}.yaml')
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"No config file {name!r} (also looked for {candidate})")


def load_run_config(config_files=(), overrides=()):
    sources = [
        {'paths': load_paths()},
        {'inference': OmegaConf.load(INFERENCE_DEFAULTS)},
        *(resolve_config_file(f) for f in config_files),
    ]
    return OmegaConf.to_object(merge_configs(RunConfig, sources, overrides))
