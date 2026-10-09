from setuptools import find_packages, setup

# `ldm` comes from the latent-diffusion submodule's conda env (latent-diffusion/environment.yaml),
# not from PyPI, so it isn't listed here. Install into that env with `pip install --no-deps -e .`.
setup(
    name='protoninfer',
    version='0.1.0',
    description='Proton momentum inference by gradient descent through a latent diffusion model',
    packages=find_packages(include=['protoninfer', 'protoninfer.*']),
    python_requires='>=3.8',
    install_requires=['numpy', 'scipy', 'torch>=1.10', 'matplotlib', 'omegaconf>=2.1', 'geomloss'],
    # Only needed by some standalone scripts (paper_plots/, datasets/, Proton64_Analysis/).
    extras_require={'scripts': ['POT', 'tqdm', 'scikit-learn']},
    entry_points={'console_scripts': ['protoninfer=protoninfer.cli:main']},
)
