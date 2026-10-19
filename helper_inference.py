"""
Compatibility shim: this module was split into protoninfer.{losses,model,initial_guess}.

Existing scripts that do `from helper_inference import ...` keep working. New code
should import from protoninfer directly.
"""
from protoninfer.initial_guess import (  # noqa: F401
    _HULL_BACKGROUND_THR, _HULL_DEFAULT_SCALE, _HULL_IMAGE_CENTER, _PMAG_PL_EXPONENT, _PMAG_PL_SCALE,
    _hull_guess_pixels, _max_area_triangle, convex_hull_initial_guess, image_pmag_estimate,
)
from protoninfer.losses import (  # noqa: F401
    LOSS_NAMES, _LOSS_REGISTRY, _get_cost_matrix, _prepare_distributions, emd_loss_with_gradients,
    energy_distance_loss, gaussian_mmd_loss, get_loss_fn, l2_loss_with_gradients, laplacian_mmd_loss,
    sliced_wasserstein_loss, unbalanced_emd_loss, wasserstein2_loss,
)
from protoninfer.model import BACKGROUND_THRESHOLD, DifferentiableLDMGenerator, decode_first_stage_with_grad  # noqa: F401
