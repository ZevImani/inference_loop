# inference_loop

Recover the 3D momentum of proton tracks from 2D detector images by running gradient descent through a momentum-conditioned latent diffusion model (LDM).

Given a target image, the loop starts from an initial guess of `(px, py, pz)` for each track. At each step it:

1. Generates images for the current guess with the LDM.
2. Measures how far they are from the target with an image-space loss (Sinkhorn EMD by default).
3. Backpropagates through the diffusion sampler and updates the momenta.

For multi-track events, the image generated for each track is summed before the loss, so a single loss drives all the tracks.

The repo has two parts:

- **`protoninfer/`**: the inference library. Use it to run inference on your own images.
- **`studies/`**: scripts that run inference over whole datasets and write summary plots.

## Installation

Everything runs in the `ldm` conda environment defined in [`latent-diffusion/environment.yaml`](latent-diffusion/environment.yaml). That file pins every package the inference loop and the analysis scripts use. A GPU is strongly recommended.

1. **Clone with submodules:**

   ```bash
   git clone --recurse-submodules git@github.com:zevimani/inference_loop.git
   cd inference_loop
   ```

2. **Create the `ldm` environment** from the submodule:

   ```bash
   cd latent-diffusion
   conda env create -f environment.yaml
   conda activate ldm
   cd ..
   ```

   The environment file ends with `-e .`, which installs the latent-diffusion code in editable mode so that `ldm` can be imported. It has to be run from inside `latent-diffusion/` so that `.` points at that directory.

   The conda step installs `torchvision` 0.8.1, which brings in conda PyTorch 1.7. The pip step then replaces that with `torch` 2.4.1; the inference loop needs PyTorch ≥ 1.10. With this combination, the pure-Python parts of `torchvision` (transforms, models, `make_grid`) work, and those are the only parts this code uses. Its compiled ops, such as `torchvision.ops.nms`, don't work.

3. **Install this package** into the environment:

   ```bash
   pip install --no-deps -e .
   ```

   `--no-deps` stops pip from changing the PyTorch build that is already installed.

4. **Point `configs/paths.yaml` at your trained LDM**: set `ldm_checkpoint` and `ldm_config` (see [Configure paths](#configure-paths)).

### Tested versions

These are the main versions pinned in `environment.yaml`:

| Package | Version |
| --- | --- |
| python | 3.8.12 |
| torch | 2.4.1 (CUDA 12.1) |
| numpy | 1.24.4 |
| scipy | 1.9.3 |
| matplotlib | 3.7.5 |
| omegaconf | 2.1.1 |
| geomloss | 0.2.6 |
| pytorch-lightning | 1.5.0 |
| POT | 0.8.2 |

`geomloss` works without `pykeops`, using its plain-PyTorch backend, which is fast enough for 64×64 images.

### Configure paths

Every filesystem location is set in [`configs/paths.yaml`](configs/paths.yaml): the LDM checkpoint, the datasets, and the output root. Relative paths are resolved against the repo root, and `~` and `$VARS` are expanded.

> **Note:** the committed `paths.yaml` contains the absolute paths used on the author's cluster (`/n/home11/...`, `/n/holystore01/...`). They won't exist on your machine, so replace `latent_diffusion_dir`, `ldm_config`, `ldm_checkpoint` and the `datasets.val` / `datasets.dual_proj` entries with your own locations before running anything.

To keep your own copy outside the repo, point an environment variable at it:

```bash
export PROTONINFER_PATHS=/path/to/my_paths.yaml
```

## Quick start

### Command line: one image

```bash
python -m protoninfer event.npy                              # 1 track, default settings
python -m protoninfer event.npy --truth 120 -80 200          # compare against a known momentum
python -m protoninfer pair.npy --n_tracks 2 optimizer.lr=0.05
python -m protoninfer xy.npy --xz_image xz.npy               # fit one track to both projections
```

- **Input:** the image is a `(H, W)` `.npy` array. It can also be a 0-d object `.npy` holding a dict with `'image'` and, optionally, `'momentum'`.
- **Output:** results go to `inference_results/<run_name>/`, and the best-fit momentum is printed at the end.
- **Plots:** per-event plots need truth momenta, so without `--truth` only the data files are written.

After `pip install -e .`, the same interface is also available as the `protoninfer` command.

### Python

```python
import numpy as np
from protoninfer import infer, load_inference_config

cfg = load_inference_config('configs/inference/default.yaml',
                            overrides=['optimizer.lr=0.05', 'init.use_pmag=true'])

image = np.load('event.npy')
results = infer(image, true_momentum=None, cfg=cfg, n_tracks=1,
                output_dir='inference_results', run_name='my_event')

best = int(np.argmin(results['dist_path']))
print(results['mom_path'][best], results['dist_path'][best])
```

`infer()` loads the LDM from `paths.yaml` on every call unless you give it a generator. When processing many images, load the model once and pass it in:

```python
import torch
from protoninfer import DifferentiableLDMGenerator, load_ldm

device = torch.device('cuda')
generator = DifferentiableLDMGenerator(load_ldm(device), device='cuda')
for image in images:
    infer(image, cfg=cfg, generator=generator, device=device, run_name=...)
```

`infer()` returns a dict with these keys:

| Key | Contents |
| --- | --- |
| `img_path` | Generated image at each step |
| `dist_path` | Loss at each step |
| `mom_path` | Momenta at each step |
| `lr_path` | Learning rate at each step |
| `data_dir` | Directory holding the saved `.npy`/`.pt` files |
| `plot_dir` | Directory holding the plots |
| `elapsed` | Wall-clock time |

## Configuration

Hyperparameters are defined as typed dataclasses in [`protoninfer/config.py`](protoninfer/config.py) and set with YAML. Unknown keys, wrong types and invalid choices raise an error rather than being silently ignored.

[`configs/inference/default.yaml`](configs/inference/default.yaml) lists every setting with a comment. The groups are:

| Group | Settings |
| --- | --- |
| top level | `n_iterations`, `min_distance` (early-stop threshold) |
| `optimizer` | `name` (SGD / adam / adamw), `lr`, `lr_min`, `gradient_clip`, `reparam_pmag`, `newton_pz` |
| `sampling` | `avg_grad`, `batch_size`, `fixed_z`, `ddim_steps_standard`, `ddim_steps_gradient` |
| `loss` | `name` (emd, unbal_emd, w2, energy, gaussian_mmd, laplacian_mmd, l2), `kwargs` |
| `init` | `method` (hull / random), `use_pmag`, `init_pz` |
| `renoise` | `frac`, `schedule`, `min`, `truth_init`, `truth_every_step` |

Override any setting on the command line with dotted `key=value` pairs:

```bash
optimizer.lr=0.05  loss.name=unbal_emd  loss.kwargs.reg_m=1.0  renoise.frac=0.3
```

Studies nest these settings under `inference.`, for example `inference.optimizer.lr=0.05`.

## Running studies

[`studies/run_study.py`](studies/run_study.py) runs inference over `n_events` events from a dataset. It writes each event's outputs and a summary of the whole run.

```bash
python studies/run_study.py -c val n_events=128 run_name=val128_default
python studies/run_study.py -c kinked n_events=64 inference.renoise.frac=0.3
python studies/run_study.py -c angles n_events=100 data.angle_max=25
python studies/run_study.py -c pzdeg data.which=pz0 --print_config   # show the merged config and exit
python studies/run_study.py --plot_only run_name=val128_default      # redraw summary plots only
```

`-c` takes either a path to a YAML file or the name of a file in [`configs/studies/`](configs/studies/). Settings are applied in this order, with later ones winning:

1. `configs/paths.yaml`
2. `configs/inference/default.yaml`
3. each `-c` file
4. `key=value` overrides

### Available studies

| Study | Events |
| --- | --- |
| `val` | Single tracks from the validation set |
| `kinked` | Single kinked tracks (top-64 kink-score events, `kinked_tracks_top64.npz`) |
| `pzdeg` | One subset of the pz-degeneracy set (`data.which: ref` or `pz0`) |
| `pzboth` | Both pz-degeneracy subsets, with two-colour summary plots |
| `angles` | Two-track pairs, stratified across `[data.angle_min, data.angle_max]` degrees |
| `dual_proj` | One track fit against both its xy and xz projections |

### Run-level settings

These are defined in [`studies/common/config.py`](studies/common/config.py):

| Setting | Meaning |
| --- | --- |
| `n_events`, `event_offset` | How many events to run, and where in the dataset to start |
| `run_name` | Output folder name. Defaults to `<study>_<timestamp>` |
| `output_dir` | Output root. Defaults to `studies/results` |
| `seed` | Seeds torch and numpy for a reproducible run. Unseeded by default |
| `save_plots` | Whether to write per-event plots |
| `keep_img_path` | Keep each event's `img_path.pt`, which is large (deleted by default) |
| `data.*` | Study-specific settings, such as the dataset path |

### Output layout

```
<output_dir>/<run_name>/
    config.yaml              # fully resolved config used for this run
    events/event_NNN/
        data_files/          # dist_path.npy, mom_path.npy, target_img.npy, truth_mom.npy, final_img.npy, ...
        plots/               # sgd_images.png, sgd_evolution.gif, initial_final.png, ...
    summary/
        summary.npz          # final_emds, final_momentum, truth_momentum, [angles]
        emd_ranking.txt
        plot_*.png           # loss distribution, recovered vs truth, residuals, EMD vs momentum/angle
```

For `pzboth`, the events are split into `events/ref/` and `events/pz0/`, and the summary keys carry `_ref` / `_pz0` suffixes.

### SLURM

Example batch scripts are in [`studies/submit_scripts/`](studies/submit_scripts/). Each one changes into the repo root, sets a few variables, and calls `run_study.py`:

```bash
sbatch studies/submit_scripts/submit_val128.sh
```

### Adding a study

1. Write a loader in [`studies/common/datasets.py`](studies/common/datasets.py) that returns a list of `Event(img, truth, n_tracks=..., ...)`.
2. Register it in `STUDIES` in [`studies/common/registry.py`](studies/common/registry.py). An entry gives a loader, a run-name prefix and, optionally, a custom summary function.
3. Add `configs/studies/<name>.yaml` with `study: <name>` and a `data:` section.

## Tests

```bash
python tests/test_config.py     # or: pytest tests/
```

These checks run on CPU:

- The YAML defaults match the dataclasses.
- Invalid overrides are rejected.
- Every study config resolves to an existing dataset.
- Every study loader returns events.

They load real data, so `configs/paths.yaml` has to point to your datasets.

## Repository layout

| Path | Contents |
| --- | --- |
| `protoninfer/optimize.py` | `infer()`, `run_inference()`, and the SGD core |
| `protoninfer/model.py` | `load_ldm()` and `DifferentiableLDMGenerator` (differentiable DDIM sampling) |
| `protoninfer/losses.py` | EMD and the other image losses, plus `get_loss_fn()` |
| `protoninfer/initial_guess.py` | Convex-hull direction guess, energy-based \|p\| estimate, random init |
| `protoninfer/plotting.py` | Per-event diagnostic plots |
| `protoninfer/config.py` | Config schema and YAML loading |
| `protoninfer/paths.py` | Reads `configs/paths.yaml` |
| `studies/` | Dataset-level studies and their results |
| `datasets/` | Dataset build scripts and the small datasets (kinked, angle pairs) |
| `pz_degeneracy/`, `rare_tail_events/`, `mcmc_inference/`, `loss_testing/`, `loss_surface_probes/`, `paper_plots/` | Standalone analyses |
| `archive/` | Superseded experiments |
| `latent-diffusion/`, `Proton64_Reco_Model/`, `Proton64_Analysis/` | Git submodules |

`run_inference.py` and `helper_inference.py` in the repo root are compatibility shims. They re-export from `protoninfer` so that older scripts using `from run_inference import ...` keep working. New code should import from `protoninfer`.
