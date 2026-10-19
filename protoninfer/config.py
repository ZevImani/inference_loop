"""
Inference hyperparameters as a typed config, loaded from YAML.

The dataclasses below are the schema: YAML files and `key=value` overrides are
merged onto them with OmegaConf, so unknown keys and wrong types raise errors
instead of being silently ignored. configs/inference/default.yaml mirrors these
defaults (tests/test_config.py checks that they stay in sync).

    cfg = load_inference_config('configs/inference/default.yaml',
                                overrides=['optimizer.lr=0.05', 'loss.name=uemd'])
"""
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from .losses import get_loss_fn

OPTIMIZERS        = ('SGD', 'adam', 'adamw')
INIT_METHODS      = ('hull', 'random')
RENOISE_SCHEDULES = ('constant', 'fixed', 'lr', 'step', 'linear', 'cosine')


def _check_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name}={value!r}; expected one of {choices}")


@dataclass
class OptimizerConfig:
    name: str = 'adamw'            # SGD | adam | adamw  (adam and adamw both use AdamW)
    lr: float = 0.1                # normalised units (momentum / 500)
    lr_min: float = 0.001          # floor of the cosine LR schedule
    gradient_clip: float = 1.0     # per-track gradient-norm clip
    reparam_pmag: bool = False     # optimise (px, py, |p|) and derive pz
    newton_pz: bool = False        # scale the pz (or |p|) grad by the inverse diagonal Hessian
    fix_pz: bool = False           # hold pz at its starting value; only px, py are optimised

    def __post_init__(self):
        _check_choice('optimizer.name', self.name, OPTIMIZERS)
        if self.fix_pz and (self.reparam_pmag or self.newton_pz):
            raise ValueError("optimizer.fix_pz cannot be combined with reparam_pmag or newton_pz")


@dataclass
class SamplingConfig:
    avg_grad: bool = True          # average the loss over a batch of stochastic LDM samples
    batch_size: int = 32           # samples per step when avg_grad is true
    fixed_z: bool = False          # reuse one latent noise vector (only when avg_grad is false)
    ddim_steps_standard: int = 50  # DDIM steps for no-grad generation
    ddim_steps_gradient: int = 10  # DDIM steps for gradient-enabled generation


@dataclass
class LossConfig:
    name: str = 'emd'              # see protoninfer.losses.LOSS_NAMES
    kwargs: Dict[str, Any] = field(default_factory=dict)  # e.g. {reg: 0.05, reg_m: 0.5} for uemd

    def __post_init__(self):
        get_loss_fn(self.name)     # raises on unknown names


@dataclass
class InitConfig:
    method: str = 'hull'           # hull (falls back to random if the hull fails) | random
    use_pmag: bool = False         # set the hull guess's |p| from the image energy estimate
    init_pz: Optional[float] = None  # override every track's starting pz (MeV)
    truth_pz: bool = False         # diagnostic: start every track at its true pz (applied after perturbations)
    random_frac: float = 0.0       # with method=hull, fraction of events that use a random init instead
    random_pmin: float = 100.0     # |p| range (MeV) for random inits
    random_pmax: float = 500.0
    pmag_scale: float = 1.0        # multiply every starting |p| by this
    pmag_jitter: float = 0.0       # log-normal std of a random per-track |p| factor (0.1 ≈ ±10%)
    angle_jitter_deg: float = 0.0  # std (deg) of a random 3D rotation of each starting direction

    def __post_init__(self):
        _check_choice('init.method', self.method, INIT_METHODS)
        if not 0.0 <= self.random_frac <= 1.0:
            raise ValueError(f"init.random_frac={self.random_frac}; expected a value in [0, 1]")
        if not 0.0 <= self.random_pmin < self.random_pmax:
            raise ValueError(f"init.random_pmin={self.random_pmin}, init.random_pmax={self.random_pmax}; "
                             "expected 0 <= random_pmin < random_pmax")


@dataclass
class RenoiseConfig:
    frac: float = 0.0              # re-noise fraction in [0, 1]; 0 disables
    schedule: str = 'constant'     # constant | fixed | lr | step | linear | cosine
    min: float = 0.1               # final fraction for non-constant schedules
    step_ref: float = 50.0         # schedule=step: previous-step |Δp| (MeV) that maps to the full frac
    truth_init: bool = False       # seed the first step's latent from the truth image
    truth_every_step: bool = False # always renoise from the truth latent
    scale_steps: bool = False      # scale the gradient DDIM steps with the renoise level (keeps step spacing)
    min_steps: int = 2             # floor on the scaled DDIM steps (2 = one DDIM update)

    def __post_init__(self):
        _check_choice('renoise.schedule', self.schedule, RENOISE_SCHEDULES)
        if self.step_ref <= 0:
            raise ValueError(f"renoise.step_ref={self.step_ref}; expected a positive value (MeV)")
        if self.min_steps < 2:
            raise ValueError(f"renoise.min_steps={self.min_steps}; expected >= 2")


@dataclass
class InferenceConfig:
    n_iterations: int = 40         # frames in the output, including step 0
    min_distance: float = 0.05     # early-stop once the best loss drops below this
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    init: InitConfig = field(default_factory=InitConfig)
    renoise: RenoiseConfig = field(default_factory=RenoiseConfig)

    def loss_fn(self):
        """The configured loss callable, with loss.kwargs bound if any are set."""
        fn = get_loss_fn(self.loss.name)
        return functools.partial(fn, **self.loss.kwargs) if self.loss.kwargs else fn

    def to_kwargs(self):
        """Keyword arguments for optimize.run_inference()."""
        return dict(
            n_iterations             = self.n_iterations,
            min_distance             = self.min_distance,
            optimizer_type           = self.optimizer.name,
            learning_rate            = self.optimizer.lr,
            lr_min                   = self.optimizer.lr_min,
            gradient_clip            = self.optimizer.gradient_clip,
            reparam_pmag             = self.optimizer.reparam_pmag,
            newton_pz                = self.optimizer.newton_pz,
            fix_pz                   = self.optimizer.fix_pz,
            avg_grad                 = self.sampling.avg_grad,
            avg_grad_batch_size      = self.sampling.batch_size,
            fixed_z                  = self.sampling.fixed_z,
            ddim_steps_standard      = self.sampling.ddim_steps_standard,
            ddim_steps_gradient      = self.sampling.ddim_steps_gradient,
            loss_type                = self.loss.name,
            loss_fn                  = self.loss_fn() if self.loss.kwargs else None,
            init_pz                  = self.init.init_pz,
            renoise                  = self.renoise.frac,
            renoise_schedule         = self.renoise.schedule,
            renoise_min              = self.renoise.min,
            renoise_step_ref         = self.renoise.step_ref,
            renoise_truth_init       = self.renoise.truth_init,
            renoise_truth_every_step = self.renoise.truth_every_step,
            renoise_scale_steps      = self.renoise.scale_steps,
            renoise_min_steps        = self.renoise.min_steps,
        )


def merge_configs(schema, sources=(), overrides=()):
    """
    Merge YAML files / dicts, then `key=value` overrides, onto a structured schema.

    Returns an OmegaConf DictConfig (interpolations unresolved); convert with
    OmegaConf.to_object() to get the dataclass instance.
    """
    cfg = OmegaConf.structured(schema)
    for src in sources:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(src) if isinstance(src, str) else src)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def load_inference_config(*yaml_files, overrides=()):
    """InferenceConfig from zero or more YAML files plus `key=value` overrides."""
    return OmegaConf.to_object(merge_configs(InferenceConfig, yaml_files, overrides))


def to_yaml(cfg):
    """YAML text for a config dataclass (or DictConfig)."""
    return OmegaConf.to_yaml(cfg if OmegaConf.is_config(cfg) else OmegaConf.structured(cfg))
