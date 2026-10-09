"""Filesystem locations (LDM checkpoint, datasets, output root), read from configs/paths.yaml."""
import importlib.util
import os
import sys

from omegaconf import OmegaConf

REPO_ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATHS_FILE = os.path.join(REPO_ROOT, 'configs', 'paths.yaml')
PATHS_ENV_VAR      = 'PROTONINFER_PATHS'


def load_paths(path_file=None):
    """Load the paths config: explicit file > $PROTONINFER_PATHS > configs/paths.yaml."""
    path_file = path_file or os.environ.get(PATHS_ENV_VAR) or DEFAULT_PATHS_FILE
    return OmegaConf.load(path_file)


def resolve_path(p):
    """Expand ~ and $VARS; interpret relative paths against the repo root."""
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def ensure_ldm_importable(paths=None):
    """Put latent_diffusion_dir on sys.path if the `ldm` package isn't already importable."""
    if importlib.util.find_spec('ldm') is not None:
        return
    ldm_dir = resolve_path((paths or load_paths()).latent_diffusion_dir)
    if ldm_dir not in sys.path:
        sys.path.append(ldm_dir)
