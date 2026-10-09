#!/usr/bin/env python3
"""
Run an inference study over N events: per-event outputs plus summary plots.

    python studies/run_study.py -c val n_events=128 run_name=val128_default
    python studies/run_study.py -c kinked n_events=64 inference.renoise.frac=0.3
    python studies/run_study.py -c angles data.angle_max=25 inference.loss.name=uemd
    python studies/run_study.py --plot_only run_name=val128_default
    python studies/run_study.py -c pzdeg data.which=pz0 --print_config

-c takes a path or a name from configs/studies/ (val, kinked, pzdeg,
pzboth, angles, dual_proj, dual_kinked). Settings are applied in this order, later ones winning:
configs/paths.yaml, configs/inference/default.yaml, each -c file, key=value overrides.
See studies/common/config.py for the run-level keys.

Output layout
-------------
<output_dir>/<run_name>/
    config.yaml                  (fully resolved config for this run)
    events/event_NNN/            (data_files/, plots/; events/ref/, events/pz0/ for pzboth)
    summary/                     (summary.npz, emd_ranking.txt, plot_*.png)
"""
import argparse

from common.config import load_run_config
from common.runner import plot_only, run_study
from protoninfer.config import to_yaml


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-c', '--config', action='append', default=[],
                        help='Study config: a path or a name in configs/studies/ (repeatable)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Regenerate summary plots for run_name (default: latest run) without inference')
    parser.add_argument('--print_config', action='store_true',
                        help='Print the merged config and exit')
    parser.add_argument('overrides', nargs='*', help='key=value overrides, e.g. inference.optimizer.lr=0.05')
    args = parser.parse_args()

    cfg = load_run_config(args.config, args.overrides)
    if args.print_config:
        print(to_yaml(cfg))
    elif args.plot_only:
        plot_only(cfg)
    else:
        run_study(cfg)


if __name__ == '__main__':
    main()
