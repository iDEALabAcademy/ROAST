#!/usr/bin/env python3
"""
Simple Offset Visualization - Just plot all points on a 5x5 grid.

Usage:
    python visualize_offsets_simple.py --dataset mnist
    python visualize_offsets_simple.py --dataset svhn
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from attack_config import get_config

# ============================================================================
# Dataset-specific paths (derived from attack_config.py)
# ============================================================================

def _paths_for(dataset_name: str) -> dict:
    cfg = get_config(dataset_name)
    output_dir = cfg['output_dir']
    return {
        'model_code_dir': cfg['model_code_dir'],
        'original_path': cfg['checkpoint_path'],
        'attacked_path': os.path.join(output_dir, 'worst_model.pth'),
        'output_dir': os.path.join(output_dir, 'visualizations'),
    }


DATASET_PATHS = {name: _paths_for(name) for name in ('mnist', 'svhn')}


def setup_model_imports(dataset_name):
    """Add the correct binary_Ding directory to sys.path and import build_model."""
    model_code_dir = DATASET_PATHS[dataset_name]['model_code_dir']
    if model_code_dir not in sys.path:
        sys.path.insert(0, model_code_dir)
    # Force reload in case MNIST was loaded before SVHN (or vice versa)
    import importlib
    if 'lbpnet.models' in sys.modules:
        importlib.reload(sys.modules['lbpnet.models'])
    if 'lbpnet' in sys.modules:
        importlib.reload(sys.modules['lbpnet'])
    from lbpnet.models import build_model
    return build_model


def get_image_hw(cfg):
    sz = cfg.get('image_size', 28)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


def load_offsets(checkpoint_path, device, build_model_fn, use_actual=True):
    """
    Load model and extract offset values.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
        build_model_fn: The build_model function (dataset-specific)
        use_actual: If True, return actual sampling offsets (after tanh transform).
                   If False, return raw unconstrained offsets_raw values.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = build_model_fn(config).to(device)
    H, W = get_image_hw(config)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, H, W, device=device))
    
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=False)
    
    offsets = {}
    
    if use_actual:
        # Get actual sampling offsets (after transformation)
        for name, module in model.named_modules():
            if hasattr(module, '_get_offsets'):
                with torch.no_grad():
                    actual = module._get_offsets().cpu().numpy()
                # Convert module name to parameter name format
                param_name = f"{name}.offsets_raw"
                offsets[param_name] = actual
    else:
        # Get raw unconstrained values
        for name, param in model.named_parameters():
            if 'offsets_raw' in name:
                offsets[name] = param.data.cpu().numpy()
    
    return offsets


def main():
    parser = argparse.ArgumentParser(description='Visualize LBPNet offsets')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(DATASET_PATHS.keys()),
                        help='Dataset to visualize (default: mnist)')
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    paths = DATASET_PATHS[dataset_name]
    build_model = setup_model_imports(dataset_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    original_path = paths['original_path']
    attacked_path = paths['attacked_path']
    output_dir = paths['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("LOADING ACTUAL SAMPLING OFFSETS (after tanh transform)")
    print("="*70)
    print("Note: Using actual offsets (radius * tanh(raw/radius)), not raw values")
    print("      For window=5, radius=2, all offsets are in [-2, 2]")
    print("="*70)
    
    # Load both raw and actual offsets
    original_offsets_actual = load_offsets(original_path, device, build_model, use_actual=True)
    attacked_offsets_actual = load_offsets(attacked_path, device, build_model, use_actual=True)
    original_offsets_raw = load_offsets(original_path, device, build_model, use_actual=False)
    attacked_offsets_raw = load_offsets(attacked_path, device, build_model, use_actual=False)
    
    # Use actual offsets for visualization
    original_offsets = original_offsets_actual
    attacked_offsets = attacked_offsets_actual
    
    # Save comprehensive comparison log
    log_path = os.path.join(output_dir, 'offset_comparison_log.txt')
    with open(log_path, 'w') as f:
        f.write("="*90 + "\n")
        f.write("OFFSET COMPARISON: RAW vs CLAMPED (ACTUAL) VALUES\n")
        f.write("="*90 + "\n")
        f.write("\nNote: Raw values are unconstrained parameters (can be any value)\n")
        f.write("      Clamped values = radius * tanh(raw/radius), constrained to [-radius, +radius]\n")
        f.write("      For window=5, radius=2.0\n")
        f.write("="*90 + "\n\n")
        
        layer_names = list(original_offsets_raw.keys())
        
        for layer_name in layer_names:
            orig_raw = original_offsets_raw[layer_name]
            orig_actual = original_offsets_actual[layer_name]
            atk_raw = attacked_offsets_raw[layer_name]
            atk_actual = attacked_offsets_actual[layer_name]
            
            stage_num = layer_name.split('.')[1] if '.' in layer_name else '?'
            f.write(f"\n{'='*90}\n")
            f.write(f"STAGE {stage_num}\n")
            f.write(f"{'='*90}\n")
            
            num_patterns, num_points, _ = orig_raw.shape
            
            for p_idx in range(num_patterns):
                f.write(f"\n  Pattern {p_idx}:\n")
                f.write(f"  {'-'*86}\n")
                f.write(f"  {'Pt':<4} {'Orig Raw X':<12} {'Orig Raw Y':<12} {'Orig Act X':<12} {'Orig Act Y':<12}\n")
                f.write(f"       {'Atk Raw X':<12} {'Atk Raw Y':<12} {'Atk Act X':<12} {'Atk Act Y':<12}\n")
                f.write(f"  {'-'*86}\n")
                
                for pt_idx in range(num_points):
                    # Original values
                    orx_raw = orig_raw[p_idx, pt_idx, 0]
                    ory_raw = orig_raw[p_idx, pt_idx, 1]
                    orx_act = orig_actual[p_idx, pt_idx, 0]
                    ory_act = orig_actual[p_idx, pt_idx, 1]
                    
                    # Attacked values
                    atx_raw = atk_raw[p_idx, pt_idx, 0]
                    aty_raw = atk_raw[p_idx, pt_idx, 1]
                    atx_act = atk_actual[p_idx, pt_idx, 0]
                    aty_act = atk_actual[p_idx, pt_idx, 1]
                    
                    # Calculate differences
                    diff_raw = np.sqrt((atx_raw - orx_raw)**2 + (aty_raw - ory_raw)**2)
                    diff_act = np.sqrt((atx_act - orx_act)**2 + (aty_act - ory_act)**2)
                    
                    f.write(f"  {pt_idx:<4} {orx_raw:>11.4f}  {ory_raw:>11.4f}  {orx_act:>11.4f}  {ory_act:>11.4f}\n")
                    f.write(f"       {atx_raw:>11.4f}  {aty_raw:>11.4f}  {atx_act:>11.4f}  {aty_act:>11.4f}")
                    f.write(f"  [Δraw:{diff_raw:>6.3f}, Δact:{diff_act:>6.3f}]\n")
                
                # Pattern-level statistics
                f.write(f"\n  Pattern {p_idx} Statistics:\n")
                raw_diffs = np.sqrt(((atk_raw[p_idx] - orig_raw[p_idx]) ** 2).sum(axis=-1))
                act_diffs = np.sqrt(((atk_actual[p_idx] - orig_actual[p_idx]) ** 2).sum(axis=-1))
                
                f.write(f"    Raw displacement:    mean={raw_diffs.mean():.4f}, max={raw_diffs.max():.4f}, min={raw_diffs.min():.4f}\n")
                f.write(f"    Actual displacement: mean={act_diffs.mean():.4f}, max={act_diffs.max():.4f}, min={act_diffs.min():.4f}\n")
        
        # Overall summary
        f.write(f"\n\n{'='*90}\n")
        f.write("OVERALL SUMMARY\n")
        f.write(f"{'='*90}\n")
        
        for layer_name in layer_names:
            stage_num = layer_name.split('.')[1] if '.' in layer_name else '?'
            
            orig_raw = original_offsets_raw[layer_name]
            orig_actual = original_offsets_actual[layer_name]
            atk_raw = attacked_offsets_raw[layer_name]
            atk_actual = attacked_offsets_actual[layer_name]
            
            raw_displacement = np.sqrt(((atk_raw - orig_raw) ** 2).sum(axis=-1))
            act_displacement = np.sqrt(((atk_actual - orig_actual) ** 2).sum(axis=-1))
            
            f.write(f"\nStage {stage_num}:\n")
            f.write(f"  Raw offsets range:    [{orig_raw.min():.2f}, {orig_raw.max():.2f}] -> [{atk_raw.min():.2f}, {atk_raw.max():.2f}]\n")
            f.write(f"  Actual offsets range: [{orig_actual.min():.2f}, {orig_actual.max():.2f}] -> [{atk_actual.min():.2f}, {atk_actual.max():.2f}]\n")
            f.write(f"  Raw displacement:     mean={raw_displacement.mean():.4f}, max={raw_displacement.max():.4f}\n")
            f.write(f"  Actual displacement:  mean={act_displacement.mean():.4f}, max={act_displacement.max():.4f}\n")
        
        f.write(f"\n{'='*90}\n")
    
    print(f"✓ Saved comparison log: {log_path}\n")
    
    # Print all offset values to verify
    print("\n" + "="*60)
    print("ORIGINAL OFFSETS:")
    print("="*60)
    for name, data in original_offsets.items():
        print(f"\n{name} - shape: {data.shape}")
        num_patterns, num_points, _ = data.shape
        for p in range(num_patterns):
            print(f"  Pattern {p}:")
            for pt in range(num_points):
                x, y = data[p, pt, 0], data[p, pt, 1]
                print(f"    Point {pt}: ({x:.4f}, {y:.4f})")
    
    print("\n" + "="*60)
    print("ATTACKED OFFSETS:")
    print("="*60)
    for name, data in attacked_offsets.items():
        print(f"\n{name} - shape: {data.shape}")
        num_patterns, num_points, _ = data.shape
        for p in range(num_patterns):
            print(f"  Pattern {p}:")
            for pt in range(num_points):
                x, y = data[p, pt, 0], data[p, pt, 1]
                print(f"    Point {pt}: ({x:.4f}, {y:.4f})")
    
    # Get layer names
    layer_names = list(original_offsets.keys())
    num_layers = len(layer_names)
    
    # Find the global min/max to set proper axis limits (should be around -2 to +2 for actual offsets)
    all_coords = []
    for data in list(original_offsets.values()) + list(attacked_offsets.values()):
        all_coords.append(data.flatten())
    all_coords = np.concatenate(all_coords)
    coord_max = 3.0  # Fixed to show 5x5 window clearly (radius=2 + some margin)
    print(f"\nActual coordinate range: [{all_coords.min():.2f}, {all_coords.max():.2f}]")
    print(f"Using axis limits: [-{coord_max:.1f}, {coord_max:.1f}]")
    print(f"(All actual offsets are within [-2, 2] for window=5)")
    print("="*70)
    
    # =========================================================================
    # ORIGINAL OFFSETS VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(num_layers, 2, figsize=(10, 5 * num_layers))
    fig.suptitle('ORIGINAL Offset Positions', fontsize=16, fontweight='bold')
    
    for layer_idx, layer_name in enumerate(layer_names):
        data = original_offsets[layer_name]  # [num_patterns, num_points, 2]
        num_patterns, num_points, _ = data.shape
        stage_num = layer_name.split('.')[1]
        
        for p_idx in range(num_patterns):
            ax = axes[layer_idx, p_idx]
            
            # Draw 5x5 pixel grid (the actual LBP window)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    gray = 0.7
                    rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                    facecolor=(gray, gray, gray),
                                    edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
            
            # Draw extended grid (lighter) to show where points actually are
            for i in range(-int(coord_max), int(coord_max) + 1):
                for j in range(-int(coord_max), int(coord_max) + 1):
                    if abs(i) > 2 or abs(j) > 2:
                        rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                        facecolor=(0.9, 0.9, 0.9),
                                        edgecolor='lightgray', linewidth=0.3)
                        ax.add_patch(rect)
            
            # Center marker
            ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, zorder=100)
            
            # Draw 5x5 boundary
            rect_boundary = Rectangle((-2.5, -2.5), 5, 5,
                                      fill=False, edgecolor='black', 
                                      linewidth=2, linestyle='--', zorder=90)
            ax.add_patch(rect_boundary)
            
            # Plot ALL 8 points
            for pt_idx in range(num_points):
                x = data[p_idx, pt_idx, 0]
                y = data[p_idx, pt_idx, 1]
                ax.plot(x, y, 'o', color='blue', markersize=14,
                       markeredgecolor='darkblue', markeredgewidth=2, zorder=50)
                ax.text(x, y, str(pt_idx), fontsize=7, fontweight='bold',
                       ha='center', va='center', color='white', zorder=51)
            
            ax.set_xlim(-coord_max, coord_max)
            ax.set_ylim(-coord_max, coord_max)
            ax.set_aspect('equal')
            ax.set_title(f'Stage {stage_num} - Pattern {p_idx}')
            ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'original_offsets_simple.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {save_path}")
    
    # =========================================================================
    # ATTACKED OFFSETS VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(num_layers, 2, figsize=(10, 5 * num_layers))
    fig.suptitle('ATTACKED Offset Positions', fontsize=16, fontweight='bold')
    
    for layer_idx, layer_name in enumerate(layer_names):
        data = attacked_offsets[layer_name]
        num_patterns, num_points, _ = data.shape
        stage_num = layer_name.split('.')[1]
        
        for p_idx in range(num_patterns):
            ax = axes[layer_idx, p_idx]
            
            # Draw 5x5 pixel grid (the actual LBP window)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    gray = 0.7
                    rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                    facecolor=(gray, gray, gray),
                                    edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
            
            # Draw extended grid
            for i in range(-int(coord_max), int(coord_max) + 1):
                for j in range(-int(coord_max), int(coord_max) + 1):
                    if abs(i) > 2 or abs(j) > 2:
                        rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                        facecolor=(0.9, 0.9, 0.9),
                                        edgecolor='lightgray', linewidth=0.3)
                        ax.add_patch(rect)
            
            # Center marker
            ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, zorder=100)
            
            # Draw 5x5 boundary
            rect_boundary = Rectangle((-2.5, -2.5), 5, 5,
                                      fill=False, edgecolor='black', 
                                      linewidth=2, linestyle='--', zorder=90)
            ax.add_patch(rect_boundary)
            
            # Plot ALL 8 points
            for pt_idx in range(num_points):
                x = data[p_idx, pt_idx, 0]
                y = data[p_idx, pt_idx, 1]
                ax.plot(x, y, 'o', color='red', markersize=14,
                       markeredgecolor='darkred', markeredgewidth=2, zorder=50)
                ax.text(x, y, str(pt_idx), fontsize=7, fontweight='bold',
                       ha='center', va='center', color='white', zorder=51)
            
            ax.set_xlim(-coord_max, coord_max)
            ax.set_ylim(-coord_max, coord_max)
            ax.set_aspect('equal')
            ax.set_title(f'Stage {stage_num} - Pattern {p_idx}')
            ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'attacked_offsets_simple.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")
    
    # =========================================================================
    # COMPARISON (side by side for each pattern)
    # =========================================================================
    fig, axes = plt.subplots(num_layers, 4, figsize=(16, 4 * num_layers))
    fig.suptitle('COMPARISON: Original (Blue) vs Attacked (Red)', fontsize=16, fontweight='bold')
    
    for layer_idx, layer_name in enumerate(layer_names):
        orig_data = original_offsets[layer_name]
        atk_data = attacked_offsets[layer_name]
        num_patterns, num_points, _ = orig_data.shape
        stage_num = layer_name.split('.')[1]
        
        for p_idx in range(num_patterns):
            # Original subplot
            ax_orig = axes[layer_idx, p_idx * 2]
            ax_atk = axes[layer_idx, p_idx * 2 + 1]
            
            for ax, data, color, label in [(ax_orig, orig_data, 'blue', 'Original'),
                                            (ax_atk, atk_data, 'red', 'Attacked')]:
                # Draw 5x5 pixel grid
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        gray = 0.7
                        rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                        facecolor=(gray, gray, gray),
                                        edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)
                
                # Draw extended grid
                for i in range(-int(coord_max), int(coord_max) + 1):
                    for j in range(-int(coord_max), int(coord_max) + 1):
                        if abs(i) > 2 or abs(j) > 2:
                            rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                            facecolor=(0.9, 0.9, 0.9),
                                            edgecolor='lightgray', linewidth=0.3)
                            ax.add_patch(rect)
                
                ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=3, zorder=100)
                
                # Draw 5x5 boundary
                rect_boundary = Rectangle((-2.5, -2.5), 5, 5,
                                          fill=False, edgecolor='black', 
                                          linewidth=2, linestyle='--', zorder=90)
                ax.add_patch(rect_boundary)
                
                # Plot all points
                for pt_idx in range(num_points):
                    x = data[p_idx, pt_idx, 0]
                    y = data[p_idx, pt_idx, 1]
                    edge = 'darkblue' if color == 'blue' else 'darkred'
                    ax.plot(x, y, 'o', color=color, markersize=14,
                           markeredgecolor=edge,
                           markeredgewidth=2, zorder=50)
                    ax.text(x, y, str(pt_idx), fontsize=7, fontweight='bold',
                           ha='center', va='center', color='white', zorder=51)
                
                ax.set_xlim(-coord_max, coord_max)
                ax.set_ylim(-coord_max, coord_max)
                ax.set_aspect('equal')
                ax.set_title(f'Stage {stage_num} P{p_idx} - {label}')
                ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_offsets_simple.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
