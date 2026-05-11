"""
Attack Configuration for ROAST: Reverse-training Offset Attack on Spatial Topologies
=====================================================================================
Central configuration file mapping dataset names to model code, checkpoints,
data directories, output directories and attack hyperparameters.

This repository ships the LBPNet model code (``lbpnet/`` package) inside the
repo, so by default ``model_code_dir`` points to the repo root itself. To
attack your own dataset/model, either:

  1. Edit one of the existing entries below (``mnist`` / ``svhn``) to point to
     a different checkpoint, or
  2. Add a new entry to ``DATASET_CONFIGS`` below following the same schema.

You only need to provide a trained LBPNet checkpoint (``best_model.pth``) that
contains both ``model_state_dict`` and ``config`` keys.

Usage:
    from attack_config import get_config
    cfg = get_config('mnist')
"""

import os
from pathlib import Path

# Repository root = directory containing this file.
REPO_ROOT = Path(__file__).resolve().parent

# Default location for trained checkpoints. Override per-dataset below.
CHECKPOINTS_ROOT = REPO_ROOT / 'checkpoints'

# Default location for downloaded torchvision datasets.
DATA_ROOT = REPO_ROOT / 'data'

# Default location for attack outputs (logs, attacked checkpoints, plots).
OUTPUTS_ROOT = REPO_ROOT / 'attack_outputs'


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {

    # -------------------------------------------------------------------------
    # MNIST (28x28 grayscale digits)
    # -------------------------------------------------------------------------
    'mnist': {
        'name': 'MNIST',
        'description': 'MNIST 28x28 grayscale digits',

        # Directory containing the ``lbpnet`` Python package. The repo
        # vendors lbpnet at the repo root, so this points to REPO_ROOT.
        'model_code_dir': str(REPO_ROOT),

        # Trained model checkpoint. Must include 'model_state_dict' and 'config'.
        'checkpoint_path': str(CHECKPOINTS_ROOT / 'mnist' / 'best_model.pth'),

        # Where torchvision should look for / download the raw dataset.
        'data_dir': str(DATA_ROOT),

        # Where attack artifacts go.
        'output_dir': str(OUTPUTS_ROOT / 'mnist'),

        # Image properties (must match the trained model).
        'image_size': 28,
        'channels': 1,

        # Function name in ``lbpnet.data`` that returns (train, val, test).
        'dataset_loader': 'get_mnist_datasets',

        # Attack hyperparameters.
        'attack_epochs': 50,
        'batch_size': 128,
        'base_lr': 1e-4,
        'offset_lr': 1e-1,
    },

    # -------------------------------------------------------------------------
    # SVHN (32x32, trained as grayscale here)
    # -------------------------------------------------------------------------
    'svhn': {
        'name': 'SVHN',
        'description': 'Street View House Numbers 32x32 (grayscale)',

        'model_code_dir': str(REPO_ROOT),
        'checkpoint_path': str(CHECKPOINTS_ROOT / 'svhn' / 'best_model.pth'),
        'data_dir': str(DATA_ROOT),
        'output_dir': str(OUTPUTS_ROOT / 'svhn'),

        'image_size': 32,
        'channels': 1,

        'dataset_loader': 'get_svhn_datasets',

        'attack_epochs': 50,
        'batch_size': 128,
        'base_lr': 1e-4,
        'offset_lr': 5e-3,
    },
}


# ============================================================================
# CONFIG ACCESS HELPERS
# ============================================================================

def get_config(dataset_name: str) -> dict:
    """Return the config dict for ``dataset_name``."""
    key = dataset_name.lower().replace('-', '_').replace(' ', '_')
    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    cfg = DATASET_CONFIGS[key].copy()

    if not os.path.isdir(cfg['model_code_dir']):
        print(f"WARNING: model_code_dir does not exist: {cfg['model_code_dir']}")
    if not os.path.isfile(cfg['checkpoint_path']):
        print(f"WARNING: checkpoint not found: {cfg['checkpoint_path']}")
        print(f"         Place your trained LBPNet checkpoint there, or edit "
              f"DATASET_CONFIGS['{key}']['checkpoint_path'].")
    return cfg


def list_datasets() -> list:
    """Return all registered dataset names."""
    return list(DATASET_CONFIGS.keys())


def print_config(dataset_name: str):
    cfg = get_config(dataset_name)
    print(f"\n{'='*60}\nConfiguration for: {cfg['name']}\n{'='*60}")
    print(f"Description: {cfg['description']}")
    print(f"\nPaths:")
    print(f"  Model code:  {cfg['model_code_dir']}")
    print(f"  Checkpoint:  {cfg['checkpoint_path']}")
    print(f"  Data dir:    {cfg['data_dir']}")
    print(f"  Output dir:  {cfg['output_dir']}")
    print(f"\nImage properties:")
    print(f"  Size:     {cfg['image_size']}")
    print(f"  Channels: {cfg['channels']}")
    print(f"\nAttack hyperparameters:")
    print(f"  Epochs:    {cfg['attack_epochs']}")
    print(f"  Batch:     {cfg['batch_size']}")
    print(f"  Base LR:   {cfg['base_lr']}")
    print(f"  Offset LR: {cfg['offset_lr']}")
    print(f"{'='*60}\n")


def add_custom_config(name: str, config: dict):
    """Register a new dataset config at runtime."""
    required = ['model_code_dir', 'checkpoint_path', 'data_dir',
                'output_dir', 'image_size', 'channels']
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    defaults = {
        'name': name,
        'description': f'Custom dataset: {name}',
        'attack_epochs': 50,
        'batch_size': 128,
        'base_lr': 1e-4,
        'offset_lr': 5e-4,
    }
    DATASET_CONFIGS[name.lower()] = {**defaults, **config}
    return DATASET_CONFIGS[name.lower()]


if __name__ == '__main__':
    print("Available dataset configurations:")
    for name in list_datasets():
        cfg = DATASET_CONFIGS[name]
        ok_model = "OK" if os.path.isdir(cfg['model_code_dir']) else "MISSING"
        ok_ckpt  = "OK" if os.path.isfile(cfg['checkpoint_path']) else "MISSING"
        print(f"  {name:10s}  model_code={ok_model:7s}  checkpoint={ok_ckpt}")
