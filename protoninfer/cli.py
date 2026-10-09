"""
Infer the momentum of the track(s) in one detector image.

    python -m protoninfer event.npy
    python -m protoninfer event.npy --n_tracks 2 -c my_config.yaml optimizer.lr=0.05
    python -m protoninfer xy.npy --xz_image xz.npy --truth 120 -80 200

The image is a (H, W) .npy array, or a 0-d object .npy holding a dict with
'image' (and optionally 'momentum'). Hyperparameters come from
configs/inference/default.yaml, then each -c file, then key=value overrides.
"""
import argparse
import os

import numpy as np
import torch

from .config import load_inference_config, to_yaml
from .model import DifferentiableLDMGenerator, load_ldm
from .optimize import infer
from .paths import REPO_ROOT

DEFAULT_CONFIG = os.path.join(REPO_ROOT, 'configs', 'inference', 'default.yaml')


def _load_image(path):
    data = np.load(path, allow_pickle=True)
    if data.ndim == 0:
        item = data.item()
        return np.asarray(item['image'], dtype=np.float32), item.get('momentum')
    return data.astype(np.float32), None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('image', help='(H, W) .npy image (xy projection when --xz_image is set)')
    parser.add_argument('--xz_image', default=None,
                        help='xz projection .npy; fits one track against both projections')
    parser.add_argument('--n_tracks', type=int, default=1)
    parser.add_argument('--truth', type=float, nargs='+', default=None,
                        help='True momenta px py pz [px2 py2 pz2 ...], for diagnostics and plots')
    parser.add_argument('-c', '--config', action='append', default=[],
                        help='YAML config merged over the defaults (repeatable)')
    parser.add_argument('--output_dir', default='inference_results')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--no_plots', dest='save_plots', action='store_false')
    parser.add_argument('overrides', nargs='*', help='key=value config overrides')
    args = parser.parse_args(argv)

    cfg = load_inference_config(DEFAULT_CONFIG, *args.config, overrides=args.overrides)
    print(to_yaml(cfg))

    image, truth = _load_image(args.image)
    if args.truth is not None:
        if len(args.truth) != 3 * args.n_tracks:
            parser.error(f'--truth needs {3 * args.n_tracks} values for {args.n_tracks} track(s)')
        truth = [tuple(args.truth[i:i + 3]) for i in range(0, len(args.truth), 3)]
    xz_image = _load_image(args.xz_image)[0] if args.xz_image else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = DifferentiableLDMGenerator(
        load_ldm(device), device=str(device),
        ddim_steps_standard=cfg.sampling.ddim_steps_standard,
        ddim_steps_gradient=cfg.sampling.ddim_steps_gradient,
    )

    results = infer(
        image, truth, cfg,
        n_tracks=args.n_tracks, xz_img=xz_image, generator=generator, device=device,
        output_dir=args.output_dir, run_name=args.run_name, save_plots=args.save_plots,
    )

    best = int(np.argmin(results['dist_path']))
    print(f"\nBest loss {results['dist_path'][best]:.6f} at step {best}")
    moms = np.atleast_2d(np.array(results['mom_path'][best]))
    for i, (px, py, pz) in enumerate(moms):
        print(f"Track {i + 1}: px={px:.1f}  py={py:.1f}  pz={pz:.1f}  |p|={np.linalg.norm([px, py, pz]):.1f} MeV")


if __name__ == '__main__':
    main()
