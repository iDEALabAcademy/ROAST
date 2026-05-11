#!/usr/bin/env python3
"""
Generalized Progressive Bit-Flip Attack
========================================
Applies offset attacks progressively by Hamming distance.
Works with any dataset configured in attack_config.py

Usage:
    python progressive_attack_general.py --dataset mnist
    python progressive_attack_general.py --dataset svhn --mode cumulative
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import json
import copy
import argparse
import sys
import os
from pathlib import Path

# Import attack configuration
from attack_config import get_config, list_datasets


def get_image_hw(cfg):
    """Return (H, W) from cfg['image_size'] which may be int or (h, w)."""
    sz = cfg.get('image_size', 28)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


def load_model_and_data(attack_config, device):
    """
    Load model and test data based on attack config.
    
    Returns:
        model, model_config, checkpoint, test_loader
    """
    model_code_dir = attack_config['model_code_dir']
    checkpoint_path = attack_config['checkpoint_path']
    
    # Add model code to path
    if model_code_dir not in sys.path:
        sys.path.insert(0, model_code_dir)
    
    from lbpnet.models import build_model
    import lbpnet.data as _ldata
    loader_name = attack_config.get('dataset_loader', 'get_mnist_datasets')
    get_datasets = getattr(_ldata, loader_name)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' not in checkpoint:
        raise ValueError("Config not found in checkpoint")
    model_config = checkpoint['config']
    
    # Build model
    model = build_model(model_config)
    
    # Initialize with dummy forward
    H, W = get_image_hw(model_config)
    # Get channels from model_config first (from checkpoint), fallback to attack_config
    if 'data' in model_config and 'num_classes' in model_config['data']:
        # Try to infer from model config
        channels = model_config.get('in_channels', attack_config.get('channels', 1))
    else:
        # Use attack_config channels
        channels = attack_config.get('channels', 1)
    
    print(f"Using {channels} input channels for model initialization")
    model = model.to(device)  # Move to device BEFORE dummy forward
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, channels, H, W, device=device))
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test data
    try:
        _, _, test_dataset = get_datasets(model_config)
    except Exception as e:
        print(f"Warning: Could not load test dataset via config: {e}")
        print("Falling back to standard MNIST test loader")
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(
            root=attack_config['data_dir'],
            train=False,
            transform=transform,
            download=True
        )
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    return model, model_config, checkpoint, test_loader


def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def parse_offset_id(offset_id):
    """Parse offset ID like 'S0_P1_Pt3_x' into (stage, pattern, point, coord)"""
    parts = offset_id.split('_')
    stage = int(parts[0][1:])
    pattern = int(parts[1][1:])
    point = int(parts[2][2:])
    coord = 0 if parts[3] == 'x' else 1
    return stage, pattern, point, coord


def get_offset_key(stage):
    """Get the state dict key for offsets at given stage"""
    return f'stages.{stage}.lbp_layer.offsets_raw'


def apply_attack_phase(state_dict, attack_rows, original_offsets_backup):
    """Apply attack by replacing attacked positions with original values.
    
    Applies modifications where Hamming_Truncated > 0 (actual bit-flips required).
    Same_Position rows are also attacked if Hamming > 0 (the value shifted during
    reverse training, so moving it back to its own original still requires bit-flips).
    
    Rows with Hamming=0 are skipped because they represent no actual bit-flip change.
    """
    modifications = []
    skipped = 0
    
    for row in attack_rows:
        attacked_pos = row['Attacked_Position']
        original_to_move = row['Original_to_Move']
        same_position = row['Same_Position']
        hamming_truncated = row['Hamming_Truncated']
        
        # Skip Hamming=0 cases - no bit-flips means no actual model change
        if hamming_truncated == 0:
            skipped += 1
            continue
        
        atk_stage, atk_pattern, atk_point, atk_coord = parse_offset_id(attacked_pos)
        src_stage, src_pattern, src_point, src_coord = parse_offset_id(original_to_move)
        
        atk_key = get_offset_key(atk_stage)
        src_key = get_offset_key(src_stage)
        
        if atk_key not in state_dict or src_key not in original_offsets_backup:
            continue
        
        try:
            source_value = original_offsets_backup[src_key][src_pattern, src_point, src_coord].item()
            current_value = state_dict[atk_key][atk_pattern, atk_point, atk_coord].item()
            state_dict[atk_key][atk_pattern, atk_point, atk_coord] = source_value
            
            modifications.append({
                'attacked_position': attacked_pos,
                'source_position': original_to_move,
                'same_position': same_position,
                'old_value': current_value,
                'new_value': source_value,
                'hamming_truncated': row['Hamming_Truncated']
            })
        except IndexError:
            continue
    
    return state_dict, modifications, skipped


def progressive_attack(
    dataset_name: str,
    mode: str = 'cumulative',
    attack_csv_path: str = None,
    device: str = None
):
    """
    Perform progressive bit-flip attack on specified dataset.
    
    Args:
        dataset_name: Dataset name from attack_config.py
        mode: 'cumulative' or 'independent'
        attack_csv_path: Path to attack CSV (default: from output_dir)
        device: Device to use
    """
    
    attack_config = get_config(dataset_name)
    
    print("="*80)
    print(f"PROGRESSIVE BIT-FLIP ATTACK - {attack_config['name']}")
    print(f"Mode: {mode.upper()}")
    print("="*80)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Device: {device}")
    
    # Load model and data
    model, model_config, checkpoint, test_loader = load_model_and_data(attack_config, device)
    
    # Determine attack CSV path
    if attack_csv_path is None:
        attack_csv_path = os.path.join(
            attack_config['output_dir'], 
            'closest_analysis', 
            'move_originals_to_attacked.csv'
        )
    
    print(f"\nLoading attack plan: {attack_csv_path}")
    if not os.path.exists(attack_csv_path):
        raise FileNotFoundError(
            f"Attack CSV not found: {attack_csv_path}\n"
            f"Run find_closest_offsets.py first to generate the attack plan."
        )
    
    attack_df = pd.read_csv(attack_csv_path)
    print(f"Total attack positions: {len(attack_df)}")
    
    # Group by Hamming distance
    phases = attack_df.groupby('Hamming_Truncated')
    phase_groups = {hamming: group for hamming, group in phases}
    unique_hammings = sorted(phase_groups.keys())
    print(f"Attack phases: {unique_hammings}")
    
    # Baseline evaluation
    print("\n" + "-"*80)
    print("BASELINE")
    print("-"*80)
    baseline_acc = evaluate(model, test_loader, device)
    print(f"Accuracy: {baseline_acc:.2f}%")
    
    # Backup original offsets
    original_offsets_backup = {}
    for key in checkpoint['model_state_dict'].keys():
        if 'offsets_raw' in key:
            original_offsets_backup[key] = checkpoint['model_state_dict'][key].clone()
    
    # Results storage
    results = {
        'dataset': dataset_name,
        'mode': mode,
        'baseline_accuracy': baseline_acc,
        'phases': []
    }
    
    # Output directory
    output_dir = Path(attack_config['output_dir']) / 'progressive_attack' / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize cumulative tracking
    if mode == 'cumulative':
        cumulative_state_dict = copy.deepcopy(checkpoint['model_state_dict'])
        cumulative_bitflips = 0
    
    print("\n" + "-"*80)
    print(f"{'Phase':<8} {'Hamming':<10} {'Attacks':<12} {'Skipped':<12} {'Bitflips':<12} {'Accuracy':<12} {'Drop':<12}")
    print("-"*80)
    
    for phase_idx, hamming in enumerate(unique_hammings):
        phase_df = phase_groups[hamming]
        phase_rows = phase_df.to_dict('records')
        
        if mode == 'cumulative':
            cumulative_state_dict, mods, skipped = apply_attack_phase(
                cumulative_state_dict, phase_rows, original_offsets_backup
            )
            cumulative_bitflips += sum(hamming for _ in mods)
            total_bits = cumulative_bitflips
            phase_state_dict = cumulative_state_dict
        else:
            phase_state_dict = copy.deepcopy(checkpoint['model_state_dict'])
            phase_state_dict, mods, skipped = apply_attack_phase(
                phase_state_dict, phase_rows, original_offsets_backup
            )
            total_bits = sum(r['hamming_truncated'] for r in mods)
        
        # Load and evaluate
        model.load_state_dict(phase_state_dict)
        model = model.to(device)
        phase_acc = evaluate(model, test_loader, device)
        acc_drop = baseline_acc - phase_acc
        
        print(f"{phase_idx+1:<8} {hamming:<10} {len(mods):<12} {skipped:<12} {total_bits:<12} "
              f"{phase_acc:<12.2f} {acc_drop:<12.2f}")
        
        results['phases'].append({
            'phase': phase_idx + 1,
            'hamming': int(hamming),
            'attacks': len(mods),
            'skipped': skipped,
            'total_bits': total_bits,
            'accuracy': phase_acc,
            'drop': acc_drop
        })
        
        # Save model for this phase
        model_path = output_dir / f'model_phase{phase_idx+1}_hamming{hamming}.pth'
        torch.save({
            'model_state_dict': phase_state_dict,
            'accuracy': phase_acc,
            'phase': phase_idx + 1,
            'hamming': hamming
        }, model_path)
    
    print("-"*80)
    
    # Calculate totals
    total_attacks = sum(p['attacks'] for p in results['phases'])
    total_skipped = sum(p['skipped'] for p in results['phases'])
    print(f"\nTotal offsets: {len(attack_df)}, Actual attacks: {total_attacks}, Unchanged (skipped): {total_skipped}")
    
    # Save results
    results['total_attacks'] = total_attacks
    results['total_skipped'] = total_skipped
    results_path = output_dir / 'attack_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary CSV
    summary_df = pd.DataFrame(results['phases'])
    summary_path = output_dir / 'attack_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - attack_results.json")
    print(f"  - attack_summary.csv")
    
    # Final summary
    if len(results['phases']) > 0:
        final = results['phases'][-1]
        print(f"\nFinal Results:")
        print(f"  Baseline: {baseline_acc:.2f}%")
        print(f"  Final: {final['accuracy']:.2f}%")
        print(f"  Total drop: {final['drop']:.2f}%")
        print(f"  Total bit flips: {final['total_bits']}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generalized Progressive Bit-Flip Attack'
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        help=f'Dataset to attack. Options: {list_datasets()}')
    parser.add_argument('--mode', type=str, default='cumulative',
                        choices=['cumulative', 'independent'],
                        help='Attack mode')
    parser.add_argument('--attack-csv', type=str, default=None,
                        help='Path to attack CSV')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:", list_datasets())
        sys.exit(0)
    
    progressive_attack(
        dataset_name=args.dataset,
        mode=args.mode,
        attack_csv_path=args.attack_csv,
        device=args.device
    )
