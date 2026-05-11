#!/usr/bin/env python3
"""
Generalized Find Closest Offsets Analysis
==========================================
Analyzes Hamming distance between original and attacked offset values.
Creates attack plan ranked by bit-flip cost.

Usage:
    python find_closest_offsets_general.py --dataset mnist
    python find_closest_offsets_general.py --dataset svhn
"""

import torch
import struct
import json
import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Import attack configuration
from attack_config import get_config, list_datasets

# Truncated binary precision (19 bits = ~3 decimal places)
BITS_TO_KEEP = 19


def float_to_binary(f):
    """Convert float32 to 32-bit binary string."""
    packed = struct.pack('>f', f)
    return ''.join(format(byte, '08b') for byte in packed)


def float_to_binary_truncated(f):
    """Convert float32 to truncated binary (first 19 bits)."""
    return float_to_binary(f)[:BITS_TO_KEEP]


def hamming_distance(bin1, bin2):
    """Compute Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def hamming_distance_truncated(bin1, bin2):
    """Compute Hamming distance using only first 19 bits."""
    return sum(c1 != c2 for c1, c2 in zip(bin1[:BITS_TO_KEEP], bin2[:BITS_TO_KEEP]))


def find_closest_offsets(
    dataset_name: str,
    original_checkpoint: str = None,
    attacked_checkpoint: str = None
):
    """
    Find closest original offset for each attacked offset.
    
    Args:
        dataset_name: Dataset name from attack_config.py
        original_checkpoint: Override original model path
        attacked_checkpoint: Override attacked model path
    """
    
    attack_config = get_config(dataset_name)
    
    print("="*80)
    print(f"FINDING CLOSEST OFFSETS - {attack_config['name']}")
    print("="*80)
    
    # Determine paths
    if original_checkpoint is None:
        original_checkpoint = attack_config['checkpoint_path']
    
    if attacked_checkpoint is None:
        attacked_checkpoint = os.path.join(attack_config['output_dir'], 'worst_model_v2.pth')
    
    output_dir = Path(attack_config['output_dir']) / 'closest_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOriginal checkpoint: {original_checkpoint}")
    print(f"Attacked checkpoint: {attacked_checkpoint}")
    print(f"Output directory: {output_dir}")
    
    # Load checkpoints
    print("\nLoading checkpoints...")
    
    if not os.path.exists(original_checkpoint):
        raise FileNotFoundError(f"Original checkpoint not found: {original_checkpoint}")
    if not os.path.exists(attacked_checkpoint):
        raise FileNotFoundError(f"Attacked checkpoint not found: {attacked_checkpoint}")
    
    original_ckpt = torch.load(original_checkpoint, map_location='cpu', weights_only=False)
    attacked_ckpt = torch.load(attacked_checkpoint, map_location='cpu', weights_only=False)
    
    # Extract state dicts
    if 'model_state_dict' in original_ckpt:
        original_state = original_ckpt['model_state_dict']
    else:
        original_state = original_ckpt
    
    if 'model_state_dict' in attacked_ckpt:
        attacked_state = attacked_ckpt['model_state_dict']
    else:
        attacked_state = attacked_ckpt
    
    # Find offset keys
    offset_keys = [k for k in original_state.keys() if 'offsets_raw' in k]
    print(f"Found {len(offset_keys)} offset layers: {offset_keys}")
    
    # Collect all offsets
    all_original = []
    all_attacked = []
    
    for key in offset_keys:
        stage = int(key.split('.')[1])
        
        if key not in attacked_state:
            print(f"Warning: {key} not in attacked state, skipping")
            continue
        
        orig_tensor = original_state[key]
        attack_tensor = attacked_state[key]
        
        num_patterns = orig_tensor.shape[0]
        num_points = orig_tensor.shape[1]
        
        for p in range(num_patterns):
            for pt in range(num_points):
                for c in range(2):
                    coord = 'x' if c == 0 else 'y'
                    offset_id = f"S{stage}_P{p}_Pt{pt}_{coord}"
                    
                    orig_val = orig_tensor[p, pt, c].item()
                    attack_val = attack_tensor[p, pt, c].item()
                    
                    orig_bin = float_to_binary(orig_val)
                    attack_bin = float_to_binary(attack_val)
                    
                    all_original.append((offset_id, orig_val, orig_bin))
                    all_attacked.append((offset_id, attack_val, attack_bin))
    
    print(f"Total original offsets: {len(all_original)}")
    print(f"Total attacked offsets: {len(all_attacked)}")
    
    # Find closest matches
    print("\nFinding closest original for each attacked offset...")
    
    results = []
    
    for atk_id, atk_val, atk_bin in all_attacked:
        best_match = None
        best_hamming_full = 33
        best_hamming_trunc = 20
        
        for orig_id, orig_val, orig_bin in all_original:
            h_full = hamming_distance(atk_bin, orig_bin)
            h_trunc = hamming_distance_truncated(atk_bin, orig_bin)
            
            if h_trunc < best_hamming_trunc or (h_trunc == best_hamming_trunc and h_full < best_hamming_full):
                best_match = (orig_id, orig_val, orig_bin, h_full, h_trunc)
                best_hamming_full = h_full
                best_hamming_trunc = h_trunc
        
        if best_match:
            orig_id, orig_val, orig_bin, h_full, h_trunc = best_match
            same_position = (atk_id == orig_id)
            
            results.append({
                'Attacked_Position': atk_id,
                'Attacked_Value': atk_val,
                'Attacked_Binary_19b': atk_bin[:19],
                'Original_to_Move': orig_id,
                'Original_Value': orig_val,
                'Original_Binary_19b': orig_bin[:19],
                'Hamming_Full': h_full,
                'Hamming_Truncated': h_trunc,
                'Same_Position': same_position
            })
    
    # Sort by Hamming distance (truncated first, then full)
    results.sort(key=lambda x: (x['Hamming_Truncated'], x['Hamming_Full']))
    
    # Add rank
    for i, r in enumerate(results):
        r['Rank'] = i + 1
    
    # Save results
    df = pd.DataFrame(results)
    cols = ['Rank'] + [c for c in df.columns if c != 'Rank']
    df = df[cols]
    
    csv_path = output_dir / 'move_originals_to_attacked.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved attack plan to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    hamming_counts = df['Hamming_Truncated'].value_counts().sort_index()
    print("\nPositions by Hamming distance (truncated):")
    for h, count in hamming_counts.items():
        print(f"  {h} bits: {count} positions")
    
    same_pos = df['Same_Position'].sum()
    cross_pos = len(df) - same_pos
    print(f"\nSame position: {same_pos} ({100*same_pos/len(df):.1f}%)")
    print(f"Cross position: {cross_pos} ({100*cross_pos/len(df):.1f}%)")
    
    # Save JSON summary
    summary = {
        'dataset': dataset_name,
        'total_positions': len(results),
        'hamming_distribution': hamming_counts.to_dict(),
        'same_position_count': int(same_pos),
        'cross_position_count': int(cross_pos),
        'original_checkpoint': original_checkpoint,
        'attacked_checkpoint': attacked_checkpoint
    }
    
    json_path = output_dir / 'analysis_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {json_path}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find closest original offsets for attacked model'
    )
    parser.add_argument('--dataset', type=str, default='mnist',
                        help=f'Dataset. Options: {list_datasets()}')
    parser.add_argument('--original', type=str, default=None,
                        help='Override original checkpoint path')
    parser.add_argument('--attacked', type=str, default=None,
                        help='Override attacked checkpoint path')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:", list_datasets())
        sys.exit(0)
    
    find_closest_offsets(
        dataset_name=args.dataset,
        original_checkpoint=args.original,
        attacked_checkpoint=args.attacked
    )
