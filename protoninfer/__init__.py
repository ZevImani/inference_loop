"""
protoninfer — recover 3D proton momenta from detector images by gradient descent
through a momentum-conditioned latent diffusion model.

    from protoninfer import InferenceConfig, load_inference_config, infer

    cfg = load_inference_config('configs/inference/default.yaml', overrides=['optimizer.lr=0.05'])
    results = infer(image, cfg=cfg, n_tracks=1)
"""
from .config import InferenceConfig, load_inference_config
from .initial_guess import convex_hull_initial_guess, image_pmag_estimate, random_momentum_init
from .losses import LOSS_NAMES, get_loss_fn
from .model import DifferentiableLDMGenerator, load_ldm
from .optimize import infer, run_inference
