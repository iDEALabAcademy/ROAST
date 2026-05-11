#!/usr/bin/env python3
"""
Analyze binary representation of offset parameters before and after attack.
Computes Hamming distance between original and attacked raw offset values.
"""

import torch
import numpy as np
import struct
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def float_to_binary(f: float) -> str:
    """Convert float32 to 32-bit binary string representation."""
    # Pack as float32, unpack as uint32, convert to binary
    packed = struct.pack('!f', f)
    uint_val = struct.unpack('!I', packed)[0]
    return format(uint_val, '032b')


def hamming_distance(bin1: str, bin2: str) -> int:
    """Compute Hamming distance between two binary strings."""
    assert len(bin1) == len(bin2), "Binary strings must be same length"
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))


def analyze_offset_binary_changes(
    original_ckpt: str,
    attacked_ckpt: str,
    output_dir: str
) -> None:
    """
    Analyze binary representation changes in offset parameters.
    
    Args:
        original_ckpt: Path to original model checkpoint
        attacked_ckpt: Path to attacked model checkpoint
        output_dir: Directory to save analysis results
    """
    
    print("=" * 80)
    print("BINARY OFFSET ANALYSIS: Original vs Attacked")
    print("=" * 80)
    
    # Load checkpoints
    print(f"\n📂 Loading checkpoints...")
    print(f"   Original: {original_ckpt}")
    print(f"   Attacked: {attacked_ckpt}")
    
    orig = torch.load(original_ckpt, map_location='cpu')
    attack = torch.load(attacked_ckpt, map_location='cpu')
    
    orig_state = orig.get('model_state_dict', orig)
    attack_state = attack.get('model_state_dict', attack)
    
    # Extract offset parameters
    offset_keys = [k for k in orig_state.keys() if 'offsets_raw' in k]
    offset_keys.sort()
    
    print(f"\n📊 Found {len(offset_keys)} offset parameters:")
    for k in offset_keys:
        print(f"   - {k}")
    
    # Collect all offset analysis results
    all_results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each offset tensor
    for key in offset_keys:
        # Parse stage/layer info from key
        # Format: stages.{stage}.lbp_layer.offsets_raw
        parts = key.split('.')
        stage_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        
        print(f"\n{'=' * 80}")
        print(f"Analyzing: {key}")
        print(f"Stage: {stage_idx}")
        print(f"{'=' * 80}")
        
        orig_tensor = orig_state[key]  # [P, N, 2]
        attack_tensor = attack_state[key]  # [P, N, 2]
        
        P, N, _ = orig_tensor.shape
        print(f"Shape: {list(orig_tensor.shape)} (P={P} patterns, N={N} points, 2 coords)")
        
        # Flatten to analyze each scalar parameter
        orig_flat = orig_tensor.reshape(-1).numpy()
        attack_flat = attack_tensor.reshape(-1).numpy()
        
        # Process each scalar offset
        stage_results = []
        
        for idx in range(len(orig_flat)):
            # Compute pattern, point, coord indices
            pattern_idx = idx // (N * 2)
            point_idx = (idx % (N * 2)) // 2
            coord_idx = idx % 2
            coord_name = 'x' if coord_idx == 0 else 'y'
            
            orig_val = float(orig_flat[idx])
            attack_val = float(attack_flat[idx])
            
            # Convert to binary
            orig_bin = float_to_binary(orig_val)
            attack_bin = float_to_binary(attack_val)
            
            # Compute Hamming distance
            hamming_dist = hamming_distance(orig_bin, attack_bin)
            
            # Compute value difference
            value_diff = abs(attack_val - orig_val)
            
            result = {
                'stage': stage_idx,
                'pattern': pattern_idx,
                'point': point_idx,
                'coord': coord_name,
                'flat_idx': idx,
                'original_value': orig_val,
                'attacked_value': attack_val,
                'value_diff': value_diff,
                'original_binary': orig_bin,
                'attacked_binary': attack_bin,
                'hamming_distance': hamming_dist,
                'key': key
            }
            
            stage_results.append(result)
            all_results.append(result)
        
        # Sort by Hamming distance for this stage
        stage_results.sort(key=lambda x: x['hamming_distance'])
        
        # Print summary for this stage
        print(f"\nStage {stage_idx} Summary:")
        print(f"   Total offsets: {len(stage_results)}")
        hamming_dists = [r['hamming_distance'] for r in stage_results]
        print(f"   Hamming distance: min={min(hamming_dists)}, max={max(hamming_dists)}, "
              f"mean={np.mean(hamming_dists):.2f}, median={np.median(hamming_dists):.1f}")
        
        value_diffs = [r['value_diff'] for r in stage_results]
        print(f"   Value difference: min={min(value_diffs):.6f}, max={max(value_diffs):.6f}, "
              f"mean={np.mean(value_diffs):.6f}")
        
        # Show top 5 most changed (by Hamming distance)
        print(f"\n   Top 5 most changed offsets (by Hamming distance):")
        for i, r in enumerate(stage_results[-5:], 1):
            print(f"      {i}. Pattern {r['pattern']}, Point {r['point']}, Coord {r['coord']}: "
                  f"hamming={r['hamming_distance']}/32 bits, "
                  f"value: {r['original_value']:.6f} → {r['attacked_value']:.6f} "
                  f"(Δ={r['value_diff']:.6f})")
        
        # Show 5 least changed
        print(f"\n   Top 5 least changed offsets (by Hamming distance):")
        for i, r in enumerate(stage_results[:5], 1):
            print(f"      {i}. Pattern {r['pattern']}, Point {r['point']}, Coord {r['coord']}: "
                  f"hamming={r['hamming_distance']}/32 bits, "
                  f"value: {r['original_value']:.6f} → {r['attacked_value']:.6f} "
                  f"(Δ={r['value_diff']:.6f})")
    
    # Overall analysis
    print(f"\n{'=' * 80}")
    print("OVERALL ANALYSIS (All Stages)")
    print(f"{'=' * 80}")
    
    # Sort all results by Hamming distance
    all_results.sort(key=lambda x: x['hamming_distance'])
    
    total_offsets = len(all_results)
    all_hamming = [r['hamming_distance'] for r in all_results]
    all_value_diffs = [r['value_diff'] for r in all_results]
    
    print(f"\nTotal offset parameters: {total_offsets}")
    print(f"\nHamming Distance Statistics (out of 32 bits):")
    print(f"   Min:    {min(all_hamming)}")
    print(f"   Max:    {max(all_hamming)}")
    print(f"   Mean:   {np.mean(all_hamming):.2f}")
    print(f"   Median: {np.median(all_hamming):.1f}")
    print(f"   Std:    {np.std(all_hamming):.2f}")
    
    print(f"\nValue Difference Statistics:")
    print(f"   Min:    {min(all_value_diffs):.6f}")
    print(f"   Max:    {max(all_value_diffs):.6f}")
    print(f"   Mean:   {np.mean(all_value_diffs):.6f}")
    print(f"   Median: {np.median(all_value_diffs):.6f}")
    print(f"   Std:    {np.std(all_value_diffs):.6f}")
    
    # Distribution by stage
    print(f"\nDistribution by Stage:")
    for stage_idx in sorted(set(r['stage'] for r in all_results)):
        stage_data = [r for r in all_results if r['stage'] == stage_idx]
        stage_hamming = [r['hamming_distance'] for r in stage_data]
        stage_value_diffs = [r['value_diff'] for r in stage_data]
        print(f"   Stage {stage_idx}: n={len(stage_data)}, "
              f"hamming_mean={np.mean(stage_hamming):.2f}, "
              f"value_diff_mean={np.mean(stage_value_diffs):.6f}")
    
    # Save detailed results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")
    
    # Save JSON with all data
    json_path = os.path.join(output_dir, 'binary_offset_analysis.json')
    with open(json_path, 'w') as f:
        json.dump({
            'total_offsets': total_offsets,
            'hamming_distance_stats': {
                'min': int(min(all_hamming)),
                'max': int(max(all_hamming)),
                'mean': float(np.mean(all_hamming)),
                'median': float(np.median(all_hamming)),
                'std': float(np.std(all_hamming))
            },
            'value_diff_stats': {
                'min': float(min(all_value_diffs)),
                'max': float(max(all_value_diffs)),
                'mean': float(np.mean(all_value_diffs)),
                'median': float(np.median(all_value_diffs)),
                'std': float(np.std(all_value_diffs))
            },
            'offsets_ranked_by_hamming': all_results
        }, f, indent=2)
    print(f"✅ Saved JSON: {json_path}")
    
    # Save CSV for easy viewing
    csv_path = os.path.join(output_dir, 'binary_offset_analysis.csv')
    with open(csv_path, 'w') as f:
        f.write('rank,stage,pattern,point,coord,original_value,attacked_value,value_diff,'
                'hamming_distance,original_binary,attacked_binary\n')
        for rank, r in enumerate(all_results, 1):
            f.write(f"{rank},{r['stage']},{r['pattern']},{r['point']},{r['coord']},"
                   f"{r['original_value']:.8f},{r['attacked_value']:.8f},{r['value_diff']:.8f},"
                   f"{r['hamming_distance']},{r['original_binary']},{r['attacked_binary']}\n")
    print(f"✅ Saved CSV: {csv_path}")
    
    # Save human-readable report
    report_path = os.path.join(output_dir, 'binary_offset_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BINARY OFFSET ANALYSIS REPORT: Original vs Attacked\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total offset parameters: {total_offsets}\n\n")
        
        f.write("Hamming Distance Statistics (out of 32 bits):\n")
        f.write(f"   Min:    {min(all_hamming)}\n")
        f.write(f"   Max:    {max(all_hamming)}\n")
        f.write(f"   Mean:   {np.mean(all_hamming):.2f}\n")
        f.write(f"   Median: {np.median(all_hamming):.1f}\n")
        f.write(f"   Std:    {np.std(all_hamming):.2f}\n\n")
        
        f.write("Value Difference Statistics:\n")
        f.write(f"   Min:    {min(all_value_diffs):.6f}\n")
        f.write(f"   Max:    {max(all_value_diffs):.6f}\n")
        f.write(f"   Mean:   {np.mean(all_value_diffs):.6f}\n")
        f.write(f"   Median: {np.median(all_value_diffs):.6f}\n")
        f.write(f"   Std:    {np.std(all_value_diffs):.6f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TOP 20 MOST CHANGED OFFSETS (by Hamming distance)\n")
        f.write("=" * 80 + "\n\n")
        for rank, r in enumerate(all_results[-20:], 1):
            f.write(f"Rank {rank}:\n")
            f.write(f"   Stage: {r['stage']}, Pattern: {r['pattern']}, "
                   f"Point: {r['point']}, Coord: {r['coord']}\n")
            f.write(f"   Value: {r['original_value']:.8f} → {r['attacked_value']:.8f} "
                   f"(Δ = {r['value_diff']:.8f})\n")
            f.write(f"   Hamming distance: {r['hamming_distance']}/32 bits\n")
            f.write(f"   Binary (original): {r['original_binary']}\n")
            f.write(f"   Binary (attacked): {r['attacked_binary']}\n")
            # Highlight differing bits
            diff_markers = ''.join(['^' if b1 != b2 else ' ' 
                                   for b1, b2 in zip(r['original_binary'], r['attacked_binary'])])
            f.write(f"   Differences:       {diff_markers}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TOP 20 LEAST CHANGED OFFSETS (by Hamming distance)\n")
        f.write("=" * 80 + "\n\n")
        for rank, r in enumerate(all_results[:20], 1):
            f.write(f"Rank {rank}:\n")
            f.write(f"   Stage: {r['stage']}, Pattern: {r['pattern']}, "
                   f"Point: {r['point']}, Coord: {r['coord']}\n")
            f.write(f"   Value: {r['original_value']:.8f} → {r['attacked_value']:.8f} "
                   f"(Δ = {r['value_diff']:.8f})\n")
            f.write(f"   Hamming distance: {r['hamming_distance']}/32 bits\n")
            if r['hamming_distance'] > 0:
                f.write(f"   Binary (original): {r['original_binary']}\n")
                f.write(f"   Binary (attacked): {r['attacked_binary']}\n")
                diff_markers = ''.join(['^' if b1 != b2 else ' ' 
                                       for b1, b2 in zip(r['original_binary'], r['attacked_binary'])])
                f.write(f"   Differences:       {diff_markers}\n\n")
            else:
                f.write(f"   Binary (unchanged): {r['original_binary']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DISTRIBUTION BY STAGE\n")
        f.write("=" * 80 + "\n\n")
        for stage_idx in sorted(set(r['stage'] for r in all_results)):
            stage_data = [r for r in all_results if r['stage'] == stage_idx]
            stage_hamming = [r['hamming_distance'] for r in stage_data]
            stage_value_diffs = [r['value_diff'] for r in stage_data]
            
            f.write(f"Stage {stage_idx}:\n")
            f.write(f"   Count: {len(stage_data)}\n")
            f.write(f"   Hamming distance: min={min(stage_hamming)}, max={max(stage_hamming)}, "
                   f"mean={np.mean(stage_hamming):.2f}\n")
            f.write(f"   Value difference: min={min(stage_value_diffs):.6f}, "
                   f"max={max(stage_value_diffs):.6f}, mean={np.mean(stage_value_diffs):.6f}\n\n")
    
    print(f"✅ Saved report: {report_path}")
    
    print(f"\n{'=' * 80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"   - {os.path.basename(json_path)}")
    print(f"   - {os.path.basename(csv_path)}")
    print(f"   - {os.path.basename(report_path)}")


def main():
    """Main entry point."""
    
    # Defaults can be overridden via CLI; by default we use the MNIST entry
    # from attack_config.py.
    import argparse
    from attack_config import get_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--original', default=None,
                        help='Override original checkpoint path')
    parser.add_argument('--attacked', default=None,
                        help='Override attacked checkpoint path')
    args = parser.parse_args()

    cfg = get_config(args.dataset)
    base_out = Path(cfg['output_dir'])
    original_ckpt = Path(args.original) if args.original else Path(cfg['checkpoint_path'])
    attacked_ckpt = Path(args.attacked) if args.attacked else base_out / 'worst_model.pth'
    output_dir = base_out / 'binary_analysis'
    
    # Verify files exist
    if not original_ckpt.exists():
        print(f"❌ Original checkpoint not found: {original_ckpt}")
        print(f"   Please provide the correct path.")
        return
    
    if not attacked_ckpt.exists():
        print(f"❌ Attacked checkpoint not found: {attacked_ckpt}")
        print(f"   Please run the attack first (reverse_training_attack.py)")
        return
    
    # Run analysis
    analyze_offset_binary_changes(
        str(original_ckpt),
        str(attacked_ckpt),
        str(output_dir)
    )


if __name__ == "__main__":
    main()
