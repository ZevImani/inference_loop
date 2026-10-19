"""Filesystem locations (LDM checkpoint, datasets, output root), read from configs/paths.yaml."""
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
    """Make `ldm` import from latent_diffusion_dir (the submodule), ahead of any installed copy."""
    ldm_dir = os.path.realpath(resolve_path((paths or load_paths()).latent_diffusion_dir))
    if not os.path.isdir(os.path.join(ldm_dir, 'ldm')):
        raise FileNotFoundError(
            f"No ldm package in {ldm_dir}; run `git submodule update --init` "
            f"or set latent_diffusion_dir in configs/paths.yaml")
    # ldm is a namespace package (no __init__.py), merged from every ldm/ dir on sys.path and
    # searched in sys.path order, so being first is what makes the submodule's modules win.
    if 'ldm' in sys.modules:
        loaded = os.path.realpath(os.path.dirname(list(sys.modules['ldm'].__path__)[0]))
        if loaded != ldm_dir:
            raise ImportError(f"ldm was already imported from {loaded}, not {ldm_dir}; "
                              f"import protoninfer before ldm")
        return
    if ldm_dir in sys.path:
        sys.path.remove(ldm_dir)
    sys.path.insert(0, ldm_dir)
