#!/usr/bin/env python3
"""
Evaluate Model with Attacked Offsets

This script simulates an attacker who:
1. Has the original trained model
2. Knows the worst offset positions (from attack)
3. Manually plugs in those worst offsets
4. Evaluates performance on test set

Everything else (classifier, BN, gates, etc.) stays from the original model.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# The repo vendors the ``lbpnet`` package at the repo root.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lbpnet.models import build_model
import lbpnet.data as _ldata


def get_image_hw(cfg):
    """Return (H, W) from cfg['image_size'] which may be int or (h, w)."""
    sz = cfg.get('image_size', 28)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


def evaluate_with_attacked_offsets(
    original_checkpoint_path: str,
    worst_model_path: str,
    device: str = None,
    batch_size: int = 128
):
    """
    Load original model, plug in worst offsets, evaluate on test set.
    
    Args:
        original_checkpoint_path: Path to the original trained model (best_model.pth)
        worst_model_path: Path to the attacked model with worst offsets (worst_model.pth)
        device: Device to use (cuda or cpu)
        batch_size: Batch size for evaluation
    """
    
    print("=" * 70)
    print("🔍 EVALUATE MODEL WITH ATTACKED OFFSETS")
    print("=" * 70)
    
    # Set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"🔧 Using device: {device}")
    
    # =========================================================================
    # STEP 1: Load Original Model
    # =========================================================================
    print(f"\n📥 Loading ORIGINAL model: {original_checkpoint_path}")
    if not os.path.exists(original_checkpoint_path):
        raise FileNotFoundError(f"Original checkpoint not found: {original_checkpoint_path}")
    
    original_checkpoint = torch.load(original_checkpoint_path, map_location=device, weights_only=False)
    config = original_checkpoint['config']
    print(f"   Config source: {config.get('_source', 'unknown')}")
    
    # Build model
    print("\n🤖 Building model architecture...")
    model = build_model(config).to(device)
    
    # Initialize with dummy forward pass
    H, W = get_image_hw(config)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(8, 1, H, W, device=device))
    
    # Load original weights
    original_state = original_checkpoint.get('model_state_dict', original_checkpoint)
    model.load_state_dict(original_state, strict=False)
    print(f"   ✓ Loaded original model weights")
    
    # Store original offsets for comparison
    original_offsets = {}
    for name, param in model.named_parameters():
        if 'offsets_raw' in name:
            original_offsets[name] = param.data.clone()
    
    print(f"\n📊 Original model offsets:")
    for name in original_offsets:
        data = original_offsets[name]
        print(f"   {name}: mean={data.mean().item():.4f}, std={data.std().item():.4f}")
    
    # =========================================================================
    # STEP 2: Load Worst Offsets from Attacked Model
    # =========================================================================
    print(f"\n📥 Loading ATTACKED offsets from: {worst_model_path}")
    if not os.path.exists(worst_model_path):
        raise FileNotFoundError(f"Worst model not found: {worst_model_path}")
    
    worst_checkpoint = torch.load(worst_model_path, map_location=device, weights_only=False)
    worst_state = worst_checkpoint.get('model_state_dict', worst_checkpoint)
    
    # Extract only offset parameters from worst model
    worst_offsets = {}
    for name, param_tensor in worst_state.items():
        if 'offsets_raw' in name:
            worst_offsets[name] = param_tensor.clone()
    
    print(f"\n📊 Attacked (worst) offsets:")
    for name in worst_offsets:
        data = worst_offsets[name]
        print(f"   {name}: mean={data.mean().item():.4f}, std={data.std().item():.4f}")
    
    # =========================================================================
    # STEP 3: Plug in Worst Offsets (Manual Attack)
    # =========================================================================
    print(f"\n🔧 PLUGGING IN WORST OFFSETS (keeping everything else from original)...")
    
    replaced_count = 0
    for name, param in model.named_parameters():
        if 'offsets_raw' in name and name in worst_offsets:
            # Replace with worst offsets
            with torch.no_grad():
                param.copy_(worst_offsets[name])
            replaced_count += 1
            
            # Calculate displacement
            displacement = torch.sqrt(((param.data - original_offsets[name]) ** 2).sum(dim=-1)).mean().item()
            print(f"   ✓ Replaced {name} (displacement: {displacement:.4f})")
    
    print(f"\n   Total offset tensors replaced: {replaced_count}")
    
    # Verify non-offset params unchanged
    print(f"\n   Verifying other parameters unchanged...")
    other_changed = 0
    for name, param in model.named_parameters():
        if 'offsets_raw' not in name:
            if name in original_state:
                if not torch.allclose(param.data, original_state[name], rtol=1e-5):
                    other_changed += 1
    print(f"   ✓ Non-offset parameters unchanged: {other_changed == 0}")
    
    # =========================================================================
    # STEP 4: Create Test Dataset
    # =========================================================================
    print(f"\n📊 Creating test dataset...")
    loader_name = getattr(_ldata, 'get_mnist_datasets', None) and 'get_mnist_datasets'
    # Allow caller to override via the loaded checkpoint config (key: 'dataset_loader')
    loader_name = config.get('dataset_loader', loader_name) if isinstance(config, dict) else loader_name
    get_datasets = getattr(_ldata, loader_name)
    _, _, test_dataset = get_datasets(config)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"   Test set size: {len(test_dataset)} samples")
    
    # =========================================================================
    # STEP 5: Evaluate with Original Offsets (Baseline)
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("📊 EVALUATION 1: Original Model (Original Offsets)")
    print("=" * 70)
    
    # Restore original offsets temporarily
    for name, param in model.named_parameters():
        if 'offsets_raw' in name and name in original_offsets:
            with torch.no_grad():
                param.copy_(original_offsets[name])
    
    model.eval()
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    
    criterion = nn.CrossEntropyLoss()
    test_loss_original = 0.0
    test_correct_original = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating original"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss_original += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct_original += predicted.eq(target).sum().item()
    
    test_loss_original /= len(test_loader)
    test_acc_original = 100.0 * test_correct_original / test_total
    
    print(f"\n   Original Model Performance:")
    print(f"   Test Loss: {test_loss_original:.4f}")
    print(f"   Test Accuracy: {test_acc_original:.2f}% ({test_correct_original}/{test_total})")
    
    # =========================================================================
    # STEP 6: Evaluate with Attacked Offsets
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("🔥 EVALUATION 2: Attacked Model (Worst Offsets Plugged In)")
    print("=" * 70)
    
    # Plug in worst offsets again
    for name, param in model.named_parameters():
        if 'offsets_raw' in name and name in worst_offsets:
            with torch.no_grad():
                param.copy_(worst_offsets[name])
    
    model.eval()
    test_loss_attacked = 0.0
    test_correct_attacked = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating attacked"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss_attacked += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct_attacked += predicted.eq(target).sum().item()
    
    test_loss_attacked /= len(test_loader)
    test_acc_attacked = 100.0 * test_correct_attacked / test_total
    
    print(f"\n   Attacked Model Performance:")
    print(f"   Test Loss: {test_loss_attacked:.4f}")
    print(f"   Test Accuracy: {test_acc_attacked:.2f}% ({test_correct_attacked}/{test_total})")
    
    # =========================================================================
    # STEP 7: Compare Results
    # =========================================================================
    print(f"\n" + "=" * 70)
    print("📊 ATTACK IMPACT ANALYSIS")
    print("=" * 70)
    
    loss_increase = test_loss_attacked - test_loss_original
    acc_drop = test_acc_original - test_acc_attacked
    acc_drop_pct = (acc_drop / test_acc_original) * 100 if test_acc_original > 0 else 0
    
    print(f"\n   Original Model: {test_acc_original:.2f}% accuracy, {test_loss_original:.4f} loss")
    print(f"   Attacked Model: {test_acc_attacked:.2f}% accuracy, {test_loss_attacked:.4f} loss")
    print(f"\n   📉 Accuracy Drop: {acc_drop:.2f}% (relative: {acc_drop_pct:.1f}%)")
    print(f"   📈 Loss Increase: {loss_increase:.4f}")
    
    if acc_drop > 10:
        print(f"\n   ✅ Attack SUCCESSFUL: Significant performance degradation!")
    elif acc_drop > 5:
        print(f"\n   ⚠️  Attack MODERATE: Noticeable performance degradation")
    else:
        print(f"\n   ❌ Attack WEAK: Minimal performance impact")
    
    print("=" * 70)
    
    # Return results
    return {
        'original': {
            'accuracy': test_acc_original,
            'loss': test_loss_original,
            'correct': test_correct_original,
            'total': test_total
        },
        'attacked': {
            'accuracy': test_acc_attacked,
            'loss': test_loss_attacked,
            'correct': test_correct_attacked,
            'total': test_total
        },
        'impact': {
            'accuracy_drop': acc_drop,
            'accuracy_drop_percent': acc_drop_pct,
            'loss_increase': loss_increase
        }
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model with attacked offsets')
    parser.add_argument('--original', type=str, required=True,
                        help='Path to original (clean) trained checkpoint')
    parser.add_argument('--attacked', type=str, required=True,
                        help='Path to attacked checkpoint produced by reverse_training_attack.py')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    results = evaluate_with_attacked_offsets(
        original_checkpoint_path=args.original,
        worst_model_path=args.attacked,
        device=args.device,
        batch_size=args.batch_size
    )
