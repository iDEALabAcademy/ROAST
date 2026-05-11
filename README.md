# ROAST: Reverse-training Offset Attack on Spatial Sampling Topologies

ROAST is a targeted bit-flip attack against the **learnable sampling offsets**
of LBPNet-style architectures. Unlike classic gradient-based bit-flip
attacks (BFA), ROAST first uses a short *reverse-training* pass to discover
the worst (highest-loss) offset configuration, then translates that
adversarial target into a bit-budget plan via Hamming-distance matching to
the original offsets. The attacker can then apply only those bit flips that
move each offset to its closest pre-attack value, so the total number of
hardware bit-flips is small while the accuracy drop is large.

This repository contains the FP32 reference implementation used in our DAC'26
LBR submission *"Late Breaking Results: ROAST: Reverse-training Offset Attack
on Spatial Sampling Topologies"*.

---

## 1. What is in this repository

```
ROAST/
├── attack_config.py              # Central registry of dataset configs
├── reverse_training_attack.py    # Stage 1: gradient-ascent on offsets
├── find_closest_offsets.py       # Stage 2: 19-bit Hamming bipartite match
├── progressive_attack.py         # Stage 3: bit-flip schedule (cum / indep)
├── evaluate_attacked_offsets.py  # Plug attacked offsets back into clean model
├── visualize_attack_phases.py    # Per-phase original-vs-attacked offset plots
├── visualize_offsets_simple.py   # Quick offset scatter on the LBP window
├── analyze_binary_offsets.py     # Per-tensor binary diff stats
├── run_pipeline.sh               # End-to-end driver
├── requirements.txt
├── configs/                      # YAML LBPNet configs (paper / cropped / size)
└── lbpnet/                       # Vendored LBPNet model code (no checkpoints)
```

The `lbpnet/` package is bundled here so the attack scripts work out of the
box with no external clone required. **Trained checkpoints are not shipped**
— supply your own (see §3).

---

## 2. Installation

```bash
git clone <repo-url> ROAST && cd ROAST
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with PyTorch ≥ 1.9 on a single CUDA GPU (CPU works but is slower).

---

## 3. Bring your own model

ROAST consumes a single PyTorch checkpoint that contains both
`model_state_dict` and the original training `config` dict (the
`build_model(config)` API is in `lbpnet/models/__init__.py`). The attack
targets every parameter named `*offsets_raw` inside the model.

Place your checkpoint somewhere like:

```
ROAST/checkpoints/mnist/best_model.pth
ROAST/checkpoints/svhn/best_model.pth
```

…or edit the `checkpoint_path` field of the corresponding entry in
[`attack_config.py`](attack_config.py).

If you train your own model with the bundled `lbpnet/` package, the config
saved during training is already in the correct format.

---

## 4. Adding a new dataset

Open `attack_config.py` and either edit an existing entry or add a new one:

```python
DATASET_CONFIGS['mydata'] = {
    'name': 'MyDataset',
    'description': '64x64 grayscale frobnicators',

    'model_code_dir':  str(REPO_ROOT),                     # leave as-is
    'checkpoint_path': str(CHECKPOINTS_ROOT / 'mydata' / 'best_model.pth'),
    'data_dir':        str(DATA_ROOT),
    'output_dir':      str(OUTPUTS_ROOT / 'mydata'),

    'image_size': 64,
    'channels':   1,

    # Name of the dataset loader inside lbpnet.data that returns
    # (train_ds, val_ds, test_ds) given a model config dict.
    'dataset_loader': 'get_mydata_datasets',

    'attack_epochs': 50,
    'batch_size':    128,
    'base_lr':       1e-4,
    'offset_lr':     5e-3,
}
```

Then implement `get_mydata_datasets(config)` in
`lbpnet/data/__init__.py` (or a sibling module that gets re-exported).

---

## 5. Running the attack

End-to-end:

```bash
./run_pipeline.sh mnist
```

Or step by step:

```bash
# Stage 1 — Reverse training: gradient ASCENT on offsets, all other params frozen.
python reverse_training_attack.py --dataset mnist
#  -> attack_outputs/mnist/worst_model.pth
#  -> attack_outputs/mnist/Logs/offset_logs_*.json

# Stage 2 — Hamming-distance closest-offset matching (19-bit FP32 truncation).
python find_closest_offsets.py --dataset mnist
#  -> attack_outputs/mnist/closest_analysis/move_originals_to_attacked.csv
#  -> attack_outputs/mnist/closest_analysis/analysis_summary.json

# Stage 3 — Apply the bit-flip schedule, grouped by Hamming distance.
python progressive_attack.py --dataset mnist --mode cumulative
python progressive_attack.py --dataset mnist --mode independent
#  -> attack_outputs/mnist/progressive_attack/{cumulative,independent}/
#       attack_summary.csv, attack_results.json,
#       model_phase{N}_hamming{H}.pth
```

`cumulative` mode keeps each phase's flips and adds the next bucket on top
(monotonically increasing damage). `independent` mode re-applies each
Hamming bucket to a fresh copy of the clean model so each bucket's
individual contribution can be measured.

---

## 6. Helper utilities

| Script | Purpose |
| --- | --- |
| `evaluate_attacked_offsets.py` | Loads the clean model, injects only the attacked `*offsets_raw` tensors, evaluates accuracy / loss delta. |
| `visualize_attack_phases.py`   | Renders 3-column figures (original / attacked / overlay) per attack phase. |
| `visualize_offsets_simple.py`  | Scatter plot of all offsets within the 5×5 LBP window, original vs attacked. |
| `analyze_binary_offsets.py`    | Per-parameter binary-difference statistics between two checkpoints. |

All helpers accept a `--dataset` flag and read paths from `attack_config.py`.

Example:

```bash
python evaluate_attacked_offsets.py \
    --original  checkpoints/mnist/best_model.pth \
    --attacked  attack_outputs/mnist/worst_model.pth

python visualize_attack_phases.py --dataset mnist --mode cumulative

python analyze_binary_offsets.py --dataset mnist
```

---

## 7. Output layout

After running the full pipeline for a dataset `<DS>`:

```
attack_outputs/<DS>/
├── worst_model.pth                                # adversarial target (Stage 1)
├── Logs/offset_logs_<E>epochs.json                # per-offset displacement log
├── closest_analysis/
│   ├── move_originals_to_attacked.csv             # bit-flip plan (Stage 2)
│   └── analysis_summary.json
└── progressive_attack/
    ├── cumulative/
    │   ├── attack_summary.csv                     # phase × accuracy table
    │   ├── attack_results.json
    │   └── model_phase{N}_hamming{H}.pth
    └── independent/
        ├── attack_summary.csv
        ├── attack_results.json
        └── model_phase{N}_hamming{H}.pth
```

---

## 8. Citation

Coming soon ...
```

LBPNet (the victim architecture) is from Lin *et al.*, "Local Binary Pattern
Networks", WACV 2020.
