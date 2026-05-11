#!/usr/bin/env python3
"""
Visualize Attack Phases - Show original vs attacked offsets for each phase
Shows CLAMPED (actual) offsets, not raw values.

Usage:
    python visualize_attack_phases.py --dataset mnist --mode cumulative
    python visualize_attack_phases.py --dataset svhn --mode independent
"""

import os
import sys
import argparse
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

from attack_config import get_config

# ============================================================================
# Dataset-specific paths (derived from attack_config.py + repo layout)
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent


def _paths_for(dataset_name: str) -> dict:
    """Build per-dataset path dict from the central attack_config."""
    cfg = get_config(dataset_name)
    output_dir = Path(cfg['output_dir'])
    return {
        'model_code_dir': cfg['model_code_dir'],
        'original_path': cfg['checkpoint_path'],
        'attack_csv': str(output_dir / 'closest_analysis' / 'move_originals_to_attacked.csv'),
        'checkpoint_dir': str(output_dir / 'progressive_attack'),
        'output_dir': str(output_dir / 'phase_visualizations'),
        'results_filename': 'attack_results.json',
        'ckpt_cumul_fmt': 'model_phase{phase}_hamming{hamming}.pth',
        'ckpt_indep_fmt': 'model_phase{phase}_hamming{hamming}.pth',
        'complete_ckpt': None,
        'results_format': 'progressive',
    }


DATASET_PATHS = {name: _paths_for(name) for name in ('mnist', 'svhn')}


def setup_model_imports(dataset_name):
    """Make the vendored ``lbpnet`` package importable and return ``build_model``."""
    model_code_dir = DATASET_PATHS[dataset_name]['model_code_dir']
    if model_code_dir not in sys.path:
        sys.path.insert(0, model_code_dir)
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


def load_offsets_from_checkpoint(checkpoint_path, device, build_model_fn, config=None):
    """
    Load model and extract CLAMPED (actual) offset values.
    Returns offsets after tanh transform (within [-radius, +radius]).
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        build_model_fn: Function to build model from config
        config: Optional model config dict. If the checkpoint lacks 'config',
                this is used instead (e.g. from the original model checkpoint).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', config)
    if config is None:
        raise ValueError(f"No 'config' in checkpoint and none provided: {checkpoint_path}")
    
    model = build_model_fn(config).to(device)
    H, W = get_image_hw(config)
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, H, W, device=device))
    
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=False)
    
    offsets = {}
    # Get actual sampling offsets (after transformation)
    for name, module in model.named_modules():
        if hasattr(module, '_get_offsets'):
            with torch.no_grad():
                actual = module._get_offsets().cpu().numpy()
            # Convert module name to parameter name format
            param_name = f"{name}.offsets_raw"
            offsets[param_name] = actual
    
    return offsets, config


def parse_offset_id(offset_id):
    """Parse offset ID like 'S0_P1_Pt3_x' into (stage, pattern, point, coord)"""
    parts = offset_id.split('_')
    stage = int(parts[0][1:])  # S0 -> 0
    pattern = int(parts[1][1:])  # P1 -> 1
    point = int(parts[2][2:])  # Pt3 -> 3
    coord = 0 if parts[3] == 'x' else 1  # x -> 0, y -> 1
    return stage, pattern, point, coord


def draw_grid(ax, coord_max=3.0):
    """Draw the 5x5 LBP window grid"""
    # Draw 5x5 pixel grid (the actual LBP window)
    for i in range(-2, 3):
        for j in range(-2, 3):
            gray = 0.85
            rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                            facecolor=(gray, gray, gray),
                            edgecolor='darkgray', linewidth=0.5)
            ax.add_patch(rect)
    
    # Draw extended grid (lighter)
    for i in range(-int(coord_max), int(coord_max) + 1):
        for j in range(-int(coord_max), int(coord_max) + 1):
            if abs(i) > 2 or abs(j) > 2:
                rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                                facecolor=(0.95, 0.95, 0.95),
                                edgecolor='lightgray', linewidth=0.3)
                ax.add_patch(rect)
    
    # Center marker
    ax.plot(0, 0, 'kx', markersize=12, markeredgewidth=2, zorder=100)
    
    # Draw 5x5 boundary
    rect_boundary = Rectangle((-2.5, -2.5), 5, 5,
                              fill=False, edgecolor='black', 
                              linewidth=1.5, linestyle='--', zorder=90)
    ax.add_patch(rect_boundary)
    
    ax.set_xlim(-coord_max, coord_max)
    ax.set_ylim(-coord_max, coord_max)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.3, alpha=0.5)


def visualize_phase(original_offsets, attacked_offsets, phase_info, output_path, 
                    modified_positions=None, title_suffix=""):
    """
    Create a visualization comparing original and attacked offsets for a phase.
    
    Args:
        original_offsets: Dict of original offset arrays
        attacked_offsets: Dict of attacked offset arrays  
        phase_info: Dict with phase number, hamming, accuracy, etc.
        output_path: Where to save the figure
        modified_positions: Set of (stage, pattern, point) that were modified
        title_suffix: Additional text for title
    """
    layer_names = list(original_offsets.keys())
    num_layers = len(layer_names)
    coord_max = 3.0

    # Each stage has 2 patterns, so total rows = num_layers * 2, columns = 3 (Original, Attacked, Movement)
    fig, axes = plt.subplots(num_layers * 2, 3, figsize=(15, 5 * num_layers * 2))

    phase_num = phase_info.get('phase', '?')
    hamming = phase_info.get('hamming', '?')
    accuracy = phase_info.get('accuracy', 0)
    drop = phase_info.get('drop', 0)
    positions = phase_info.get('positions', 0)
    bitflips = phase_info.get('bitflips', 0)

    fig.suptitle(f'Phase {phase_num}: Hamming={hamming} | {positions} positions, {bitflips} bitflips | '
                 f'Accuracy: {accuracy:.2f}% (↓{drop:.2f}%){title_suffix}', 
                 fontsize=14, fontweight='bold')

    for layer_idx, layer_name in enumerate(layer_names):
        orig_data = original_offsets[layer_name]
        atk_data = attacked_offsets[layer_name]
        num_patterns, num_points, _ = orig_data.shape
        stage_num = int(layer_name.split('.')[1])

        for p_idx in range(num_patterns):
            row = layer_idx * 2 + p_idx
            # Column 0: Original
            ax_orig = axes[row, 0]
            draw_grid(ax_orig, coord_max)
            color = 'blue'
            edge = 'darkblue'
            marker = 'o' if p_idx == 0 else 's'
            for pt_idx in range(num_points):
                x = orig_data[p_idx, pt_idx, 0]
                y = orig_data[p_idx, pt_idx, 1]
                ax_orig.plot(x, y, marker, color=color, markersize=12,
                           markeredgecolor=edge, markeredgewidth=1.5, zorder=50)
                ax_orig.text(x, y, str(pt_idx), fontsize=6, fontweight='bold',
                           ha='center', va='center', color='white', zorder=51)
            ax_orig.set_title(f'Stage {stage_num} - Pattern {p_idx} Original', fontsize=11)

            # Column 1: Attacked
            ax_atk = axes[row, 1]
            draw_grid(ax_atk, coord_max)
            for pt_idx in range(num_points):
                x = atk_data[p_idx, pt_idx, 0]
                y = atk_data[p_idx, pt_idx, 1]
                is_modified = modified_positions and (stage_num, p_idx, pt_idx) in modified_positions
                if is_modified:
                    color = 'red'
                    edge = 'darkred'
                    size = 14
                else:
                    color = 'orange'
                    edge = 'darkorange'
                    size = 12
                marker = 'o' if p_idx == 0 else 's'
                ax_atk.plot(x, y, marker, color=color, markersize=size,
                           markeredgecolor=edge, markeredgewidth=1.5, zorder=50)
                ax_atk.text(x, y, str(pt_idx), fontsize=6, fontweight='bold',
                           ha='center', va='center', color='white', zorder=51)
            ax_atk.set_title(f'Stage {stage_num} - Pattern {p_idx} Attacked', fontsize=11)

            # Column 2: Overlay with arrows
            ax_overlay = axes[row, 2]
            draw_grid(ax_overlay, coord_max)
            for pt_idx in range(num_points):
                ox = orig_data[p_idx, pt_idx, 0]
                oy = orig_data[p_idx, pt_idx, 1]
                ax_val = atk_data[p_idx, pt_idx, 0]
                ay = atk_data[p_idx, pt_idx, 1]
                is_modified = modified_positions and (stage_num, p_idx, pt_idx) in modified_positions
                dist = np.sqrt((ax_val - ox)**2 + (ay - oy)**2)
                if dist > 0.01:
                    arrow = FancyArrowPatch((ox, oy), (ax_val, ay),
                                           arrowstyle='->', mutation_scale=15,
                                           color='purple' if is_modified else 'gray',
                                           linewidth=2 if is_modified else 1,
                                           alpha=0.8 if is_modified else 0.4,
                                           zorder=40)
                    ax_overlay.add_patch(arrow)
                # Original position (blue, smaller)
                marker = 'o' if p_idx == 0 else 's'
                ax_overlay.plot(ox, oy, marker, color='blue', markersize=8,
                               markeredgecolor='darkblue', markeredgewidth=1, 
                               alpha=0.6, zorder=45)
                # Attacked position
                if is_modified:
                    ax_overlay.plot(ax_val, ay, marker, color='red', markersize=12,
                                   markeredgecolor='darkred', markeredgewidth=1.5, zorder=50)
                else:
                    ax_overlay.plot(ax_val, ay, marker, color='orange', markersize=10,
                                   markeredgecolor='darkorange', markeredgewidth=1, 
                                   alpha=0.7, zorder=48)
                # Label at attacked position
                ax_overlay.text(ax_val, ay, str(pt_idx), fontsize=5, fontweight='bold',
                               ha='center', va='center', color='white', zorder=51)
            ax_overlay.set_title(f'Stage {stage_num} - Pattern {p_idx} Movement', fontsize=11)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Original'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Modified (attacked)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, label='Unmodified'),
        Line2D([0], [0], color='purple', linewidth=2, label='Movement arrow'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def visualize_summary(all_phases_data, original_offsets, output_path, mode='cumulative'):
    """Create a summary visualization showing accuracy progression"""
    MODE = mode
    phases = [p['phase'] for p in all_phases_data]
    hammings = [p['hamming'] for p in all_phases_data]
    accuracies = [p['accuracy'] for p in all_phases_data]
    drops = [p['drop'] for p in all_phases_data]
    bitflips = [p['bitflips'] for p in all_phases_data]
    positions = [p['positions'] for p in all_phases_data]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy vs Phase
    ax1 = axes[0, 0]
    ax1.plot(phases, accuracies, 'bo-', linewidth=2, markersize=10, label='Accuracy')
    ax1.axhline(y=accuracies[0] if len(accuracies) > 0 else 96.45, 
                color='green', linestyle='--', label='Baseline', alpha=0.7)
    ax1.set_xlabel('Phase', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Degradation by Phase', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for i, (p, acc, h) in enumerate(zip(phases, accuracies, hammings)):
        ax1.annotate(f'H={h}', (p, acc), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Cumulative Bitflips vs Accuracy Drop
    ax2 = axes[0, 1]
    ax2.plot(bitflips, drops, 'rs-', linewidth=2, markersize=10)
    ax2.set_xlabel('Cumulative Bitflips', fontsize=12)
    ax2.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax2.set_title('Accuracy Drop vs Bitflips', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for i, (bf, d, p) in enumerate(zip(bitflips, drops, phases)):
        ax2.annotate(f'P{p}', (bf, d), textcoords="offset points", 
                    xytext=(5, 5), ha='left', fontsize=9)
    
    # Plot 3: Bar chart of bitflips per phase
    ax3 = axes[1, 0]
    phase_labels = [f'P{p}\n(H={h})' for p, h in zip(phases, hammings)]
    if MODE == 'cumulative':
        # Show incremental bitflips
        incremental = [bitflips[0]] + [bitflips[i] - bitflips[i-1] for i in range(1, len(bitflips))]
        ax3.bar(phase_labels, incremental, color='steelblue', edgecolor='navy')
        ax3.set_ylabel('Bitflips (this phase)', fontsize=12)
        ax3.set_title('Bitflips Added per Phase', fontsize=13, fontweight='bold')
    else:
        ax3.bar(phase_labels, bitflips, color='steelblue', edgecolor='navy')
        ax3.set_ylabel('Bitflips', fontsize=12)
        ax3.set_title('Bitflips per Phase (Independent)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Phase', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Positions modified
    ax4 = axes[1, 1]
    ax4.bar(phase_labels, positions, color='coral', edgecolor='darkred')
    ax4.set_xlabel('Phase', fontsize=12)
    if MODE == 'cumulative':
        ax4.set_ylabel('Cumulative Positions Modified', fontsize=12)
        ax4.set_title('Cumulative Positions Modified', fontsize=13, fontweight='bold')
    else:
        ax4.set_ylabel('Positions Modified', fontsize=12)
        ax4.set_title('Positions Modified per Phase', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'{MODE.upper()} Attack Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved summary: {output_path}")


def load_results(results_path, results_format):
    """
    Load results JSON and return (baseline_acc, phase_results, complete_info)
    in a unified format regardless of JSON schema.
    """
    import json
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if results_format == 'bitflip':
        # MNIST bitflip format: results['baseline']['accuracy'], phases have positions_cumulative etc.
        baseline_acc = results['baseline']['accuracy']
        phase_results = []
        for pr in results['phases']:
            phase_results.append({
                'accuracy': pr['accuracy'],
                'drop': pr['drop'],
                'positions': pr.get('positions_cumulative', pr.get('positions_modified', 0)),
                'bitflips': pr.get('bitflips_cumulative', pr.get('bitflips', 0)),
            })
        complete_info = results.get('complete', None)
    else:
        # SVHN progressive format: results['baseline_accuracy'], phases have attacks/total_bits
        baseline_acc = results['baseline_accuracy']
        phase_results = []
        for pr in results['phases']:
            phase_results.append({
                'accuracy': pr['accuracy'],
                'drop': pr['drop'],
                'positions': pr.get('attacks', 0),
                'bitflips': pr.get('total_bits', 0),
            })
        complete_info = None
    
    return baseline_acc, phase_results, complete_info, results


def main():
    parser = argparse.ArgumentParser(description='Visualize Attack Phases')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=list(DATASET_PATHS.keys()),
                        help='Dataset to visualize (default: mnist)')
    parser.add_argument('--mode', type=str, default='cumulative',
                        choices=['cumulative', 'independent'],
                        help='Attack mode (default: cumulative)')
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    MODE = args.mode
    paths = DATASET_PATHS[dataset_name]
    build_model = setup_model_imports(dataset_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up directories
    checkpoint_dir = Path(paths['checkpoint_dir']) / MODE
    OUTPUT_DIR = Path(paths['output_dir']) / MODE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"ATTACK PHASE VISUALIZATION - {dataset_name.upper()} - {MODE.upper()} MODE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load attack plan CSV
    attack_csv = paths['attack_csv']
    print(f"\nLoading attack plan from: {attack_csv}")
    attack_df = pd.read_csv(attack_csv)
    
    # Group by Hamming_Truncated
    phases = attack_df.groupby('Hamming_Truncated')
    phase_groups = {hamming: group for hamming, group in phases}
    unique_hammings = sorted(phase_groups.keys())
    print(f"Attack phases (by Hamming_Truncated): {unique_hammings}")
    
    # Load original model offsets (this checkpoint always has 'config')
    original_path = paths['original_path']
    print(f"\nLoading original model from: {original_path}")
    original_offsets, model_config = load_offsets_from_checkpoint(original_path, device, build_model)
    
    # Load results JSON for accuracy info
    results_path = checkpoint_dir / paths['results_filename']
    if results_path.exists():
        baseline_acc, phase_results, complete_info, raw_results = load_results(
            results_path, paths['results_format']
        )
    else:
        print(f"Warning: Results file not found: {results_path}")
        baseline_acc = 0.0
        phase_results = []
        complete_info = None
        raw_results = {}
    
    all_phases_data = []
    
    # Track cumulative modified positions for cumulative mode
    cumulative_modified = set()
    
    # Visualize baseline
    print(f"\nBaseline accuracy: {baseline_acc:.2f}%")
    
    # Determine checkpoint naming format
    if MODE == 'cumulative':
        ckpt_fmt = paths['ckpt_cumul_fmt']
    else:
        ckpt_fmt = paths['ckpt_indep_fmt']
    
    # Visualize each phase
    for phase_idx, hamming in enumerate(unique_hammings, 1):
        print(f"\n--- Phase {phase_idx} (Hamming={hamming}) ---")
        
        # Find checkpoint for this phase
        ckpt_name = ckpt_fmt.format(phase=phase_idx, hamming=hamming)
        ckpt_path = checkpoint_dir / ckpt_name
        
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue
        
        # Load attacked offsets for this phase (pass model_config as fallback)
        attacked_offsets, _ = load_offsets_from_checkpoint(ckpt_path, device, build_model, config=model_config)
        
        # Get modified positions for this phase
        phase_rows = phase_groups[hamming].to_dict('records')
        phase_modified = set()
        for row in phase_rows:
            attacked_pos = row['Attacked_Position']
            stage, pattern, point, coord = parse_offset_id(attacked_pos)
            phase_modified.add((stage, pattern, point))
        
        if MODE == 'cumulative':
            cumulative_modified.update(phase_modified)
            modified_to_show = cumulative_modified.copy()
        else:
            modified_to_show = phase_modified
        
        # Get accuracy info from results (unified format)
        if phase_idx <= len(phase_results):
            pr = phase_results[phase_idx - 1]
            accuracy = pr['accuracy']
            drop = pr['drop']
            positions = pr['positions']
            bitflips = pr['bitflips']
        else:
            accuracy = 0
            drop = 0
            positions = len(modified_to_show)
            bitflips = 0
        
        phase_info = {
            'phase': phase_idx,
            'hamming': hamming,
            'accuracy': accuracy,
            'drop': drop,
            'positions': positions,
            'bitflips': bitflips
        }
        all_phases_data.append(phase_info)
        
        # Create visualization
        output_path = OUTPUT_DIR / f'phase{phase_idx}_hamming{hamming}.png'
        visualize_phase(original_offsets, attacked_offsets, phase_info, output_path,
                       modified_positions=modified_to_show)
    
    # Visualize complete attack (if checkpoint exists)
    complete_ckpt_name = paths.get('complete_ckpt')
    if complete_ckpt_name:
        complete_ckpt = checkpoint_dir / complete_ckpt_name
        if complete_ckpt.exists():
            num_positions = len(attack_df['Attacked_Position'].unique())
            print(f"\n--- Complete Attack (all {num_positions} positions) ---")
            complete_offsets, _ = load_offsets_from_checkpoint(complete_ckpt, device, build_model, config=model_config)
            
            # All positions modified
            all_modified = set()
            for row in attack_df.to_dict('records'):
                attacked_pos = row['Attacked_Position']
                stage, pattern, point, coord = parse_offset_id(attacked_pos)
                all_modified.add((stage, pattern, point))
            
            if complete_info:
                complete_acc = complete_info['accuracy']
                complete_drop = complete_info['drop']
                complete_bitflips = complete_info['bitflips']
            else:
                # Fallback: use last phase values
                last = all_phases_data[-1] if all_phases_data else {'accuracy': 0, 'drop': 0, 'bitflips': 0}
                complete_acc = last['accuracy']
                complete_drop = last['drop']
                complete_bitflips = last['bitflips']
            
            phase_info = {
                'phase': 'Complete',
                'hamming': 'ALL',
                'accuracy': complete_acc,
                'drop': complete_drop,
                'positions': num_positions,
                'bitflips': complete_bitflips
            }
            
            output_path = OUTPUT_DIR / 'complete_attack.png'
            visualize_phase(original_offsets, complete_offsets, phase_info, output_path,
                           modified_positions=all_modified,
                           title_suffix=f" (All {num_positions} positions)")
    
    # Create summary visualization
    if all_phases_data:
        summary_path = OUTPUT_DIR / 'attack_summary.png'
        visualize_summary(all_phases_data, original_offsets, summary_path, MODE)
    
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
    print(f"All visualizations saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
