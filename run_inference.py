"""
Compatibility shim: this module moved into the protoninfer package.

Existing scripts that do `from run_inference import ...` keep working. New code
should import from protoninfer (e.g. `from protoninfer import infer, load_ldm`).
"""
from protoninfer.initial_guess import convex_hull_initial_guess, image_pmag_estimate, random_momentum_init  # noqa: F401
from protoninfer.losses import (  # noqa: F401
    LOSS_NAMES, emd_loss_with_gradients, get_loss_fn, l2_loss_with_gradients, unbalanced_emd_loss,
)
from protoninfer.model import DifferentiableLDMGenerator, load_ldm  # noqa: F401
from protoninfer.optimize import _run_dual_proj_sgd, _run_sgd_core, infer, run_inference  # noqa: F401
from protoninfer.paths import load_paths as _load_paths, resolve_path as _resolve_path
from protoninfer.plotting import save_event_plots as _save_plots  # noqa: F401

_paths = _load_paths()
LDM_CONFIG_PATH     = _resolve_path(_paths.ldm_config)
LDM_CHECKPOINT_PATH = _resolve_path(_paths.ldm_checkpoint)
