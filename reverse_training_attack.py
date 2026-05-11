#!/usr/bin/env python3
"""
Generalized Reverse Training Attack for LBPNet Offsets
=======================================================
Works with any dataset configured in attack_config.py

Usage:
    python reverse_training_attack_general.py --dataset mnist
    python reverse_training_attack_general.py --dataset svhn --epochs 30
    python reverse_training_attack_general.py --dataset mnist_cropped --offset-lr 1e-3
"""

import os
import sys
import copy
import time
import json
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import attack configuration
from attack_config import get_config, list_datasets, print_config


def get_image_hw(cfg):
    """Return (H, W) from cfg['image_size'] which may be int or (h, w)."""
    sz = cfg.get('image_size', 28)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


def load_dataset_module(attack_config):
    """
    Dynamically load the dataset module based on attack config.
    
    Returns:
        get_datasets function
    """
    model_code_dir = attack_config['model_code_dir']
    dataset_loader_name = attack_config['dataset_loader']
    
    # Add model code directory to path
    if model_code_dir not in sys.path:
        sys.path.insert(0, model_code_dir)
    
    # Import the data module
    try:
        from lbpnet.data import get_mnist_datasets, get_mnist_dataloaders
        
        # Try to get the specified loader
        data_module = importlib.import_module('lbpnet.data')
        
        if hasattr(data_module, dataset_loader_name):
            return getattr(data_module, dataset_loader_name)
        else:
            # Fallback to mnist datasets if specific one not found
            print(f"   Warning: {dataset_loader_name} not found, using get_mnist_datasets")
            return get_mnist_datasets
            
    except ImportError as e:
        raise ImportError(
            f"Failed to import dataset loader from {model_code_dir}/lbpnet/data. "
            f"Error: {e}"
        )


def load_model(attack_config, device):
    """
    Load model from checkpoint using attack config.
    
    Returns:
        model, model_config
    """
    model_code_dir = attack_config['model_code_dir']
    checkpoint_path = attack_config['checkpoint_path']
    
    # Add model code directory to path
    if model_code_dir not in sys.path:
        sys.path.insert(0, model_code_dir)
    
    # Import model builder
    from lbpnet.models import build_model
    
    # Load checkpoint
    print(f"\n📥 Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        raise ValueError("Config not found in checkpoint. Cannot rebuild model.")
    
    print(f"   Config source: {model_config.get('_source', 'unknown')}")
    
    # Build model
    print("\n🤖 Building model...")
    model = build_model(model_config).to(device)
    
    # Initialize with dummy forward pass (for RP mappings)
    H, W = get_image_hw(model_config)
    # Use model's expected input channels, not dataset channels
    # Model config should specify this correctly
    model_input_channels = model_config.get('input_channels', 1)
    if model_input_channels is None:
        model_input_channels = 1
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(8, model_input_channels, H, W, device=device))
    
    # Load weights
    raw_state = checkpoint.get('model_state_dict', checkpoint)
    incompatible = model.load_state_dict(raw_state, strict=False)
    print(f"   Loaded state: missing={len(incompatible.missing_keys)}, "
          f"unexpected={len(incompatible.unexpected_keys)}")
    
    return model, model_config, checkpoint


def reverse_training_attack(
    dataset_name: str,
    num_epochs: int = None,
    batch_size: int = None,
    base_lr: float = None,
    offset_lr: float = None,
    device: str = None,
    custom_output_dir: str = None
):
    """
    Perform reverse training attack on specified dataset.
    
    Args:
        dataset_name: Name of dataset (from attack_config.py)
        num_epochs: Override default epochs
        batch_size: Override default batch size
        base_lr: Override default base learning rate
        offset_lr: Override default offset learning rate
        device: Device to use
        custom_output_dir: Override output directory
    """
    
    # Get configuration
    attack_config = get_config(dataset_name)
    print_config(dataset_name)
    
    # Override with command line args if provided
    num_epochs = num_epochs or attack_config['attack_epochs']
    batch_size = batch_size or attack_config['batch_size']
    base_lr = base_lr or attack_config['base_lr']
    offset_lr = offset_lr or attack_config['offset_lr']
    output_dir = custom_output_dir or attack_config['output_dir']
    
    print("=" * 60)
    print(f"🔥 REVERSE TRAINING ATTACK - {attack_config['name']}")
    print("=" * 60)
    
    # Set device
    if device is None:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"🔧 Using device: {device}")
    
    # Load model
    model, model_config, checkpoint = load_model(attack_config, device)
    
    # Set seeds
    seed = model_config.get('reproducibility', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Print initial offset stats
    print("\n📊 Initial offset statistics:")
    for name, param in model.named_parameters():
        if 'offsets_raw' in name:
            print(f"   {name}: mean={param.data.mean().item():.4f}, "
                  f"std={param.data.std().item():.4f}, "
                  f"min={param.data.min().item():.4f}, max={param.data.max().item():.4f}")
    
    # Load dataset
    print(f"\n📊 Loading dataset: {attack_config['name']}...")
    get_datasets = load_dataset_module(attack_config)
    train_dataset, val_dataset, test_dataset = get_datasets(model_config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate original model accuracy on test set
    print("\n" + "=" * 60)
    print("📊 ORIGINAL MODEL PERFORMANCE (Before Attack)")
    print("=" * 60)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    original_test_loss = test_loss / len(test_loader)
    original_test_acc = 100.0 * test_correct / test_total
    
    print(f"Original Model - Test Loss: {original_test_loss:.4f} | Test Acc: {original_test_acc:.2f}%")
    print(f"Test samples: {test_total}")
    print("=" * 60)
    
    # Store initial offsets for comparison
    initial_offsets = {}
    for name, param in model.named_parameters():
        if 'offsets_raw' in name:
            initial_offsets[name] = param.data.clone()
    
    # Separate parameters into offset and non-offset groups
    offset_params = []
    offset_names = []
    other_params = []
    other_names = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'offsets_raw' in name:
            offset_params.append(param)
            offset_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)
    
    print(f"\n⚡ Parameter groups:")
    print(f"   Offset params (ASCENT - maximize loss): {len(offset_params)}")
    for name in offset_names:
        print(f"      ✓ {name}")
    
    print(f"\n   Other params (FROZEN - no updates): {len(other_params)}")
    
    # FREEZE all non-offset parameters - they will NOT be updated
    for param in other_params:
        param.requires_grad = False
    
    print("   🔒 All non-offset parameters FROZEN:")
    for name in other_names:
        print(f"      🔒 {name}")
    
    # Create optimizer for offsets only (gradient ASCENT)
    offset_optimizer = optim.SGD(offset_params, lr=offset_lr, momentum=0.5)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'offset_changes': []
    }
    
    best_attack_loss = float('-inf')  # Track highest loss (worst performance)
    
    # Set model to train mode (but keep non-offset params frozen)
    model.train()
    
    # Verify freeze status
    print("\n🔍 Verifying parameter freeze status:")
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"   Frozen params: {frozen_count}")
    print(f"   Trainable params (offsets only): {trainable_count}")
    
    # Set STE
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    
    print("\n🎯 Starting Reverse Training Attack (FROZEN OTHER PARAMS)...")
    print(f"   Dataset: {attack_config['name']}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Offset LR (ASCENT): {offset_lr}")
    print(f"   Other params: FROZEN (no updates)")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Zero gradients (only offset optimizer now)
            offset_optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Add offset penalty if exists
            if hasattr(model, 'get_offset_penalty'):
                loss = loss + model.get_offset_penalty()
            
            # Check for NaN
            if torch.isnan(loss):
                continue
            
            # Backward pass - computes gradients for ALL params
            loss.backward()
            
            # Clip gradients for stability (relaxed for stronger attack)
            torch.nn.utils.clip_grad_norm_(offset_params, max_norm=50.0)
            
            # === KEY: Negate gradients for offset params ===
            # This converts descent into ascent for offsets only
            with torch.no_grad():
                for p in offset_params:
                    if p.grad is not None:
                        p.grad.neg_()  # Negate gradient for ascent
            
            # Step offset optimizer only (other params are frozen)
            offset_optimizer.step()
            
            # Stats
            train_loss_sum += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.1f}%'})
        
        train_loss = train_loss_sum / len(train_loader)
        train_acc = 100.0 * train_correct / max(1, train_total)
        
        # Validation
        val_loss = train_loss
        val_acc = train_acc
        
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss_sum += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_loss = val_loss_sum / len(val_loader)
            val_acc = 100.0 * val_correct / max(1, val_total)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Calculate offset changes
        offset_change = 0.0
        for name, param in model.named_parameters():
            if name in initial_offsets:
                diff = (param.data - initial_offsets[name]).abs().mean().item()
                offset_change += diff
        history['offset_changes'].append(offset_change)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
              f"Δoffset: {offset_change:.4f} | Time: {epoch_time:.1f}s")
        
        # Save worst model (highest validation loss)
        if val_loss > best_attack_loss:
            best_attack_loss = val_loss
            worst_model_path = os.path.join(output_dir, 'worst_model_v2.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': model_config,
                'attack_config': {
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'base_lr': base_lr,
                    'offset_lr': offset_lr,
                },
                'history': history
            }, worst_model_path)
            print(f"   🎯 New worst model saved (val_loss={val_loss:.4f})")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 ATTACK SUMMARY")
    print("=" * 60)
    
    # =========================================================================
    # BUILD JSON LOG DATA
    # =========================================================================
    offset_logs = {
        'attack_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'base_lr': base_lr,
            'offset_lr': offset_lr,
            'dataset': attack_config['name']
        },
        'statistics': {},
        'detailed_positions': {},
        'displacement_summary': {}
    }
    
    # Collect statistics for each offset parameter
    for name, param in model.named_parameters():
        if 'offsets_raw' in name:
            init_mean = initial_offsets[name].mean().item()
            final_mean = param.data.mean().item()
            init_std = initial_offsets[name].std().item()
            final_std = param.data.std().item()
            
            offset_logs['statistics'][name] = {
                'initial': {'mean': init_mean, 'std': init_std},
                'final': {'mean': final_mean, 'std': final_std},
                'change': {'mean': final_mean - init_mean, 'std': final_std - init_std}
            }
    
    # Collect detailed offset positions for all patterns and layers
    total_disp_all = 0.0
    num_offsets = 0
    
    for name, param in model.named_parameters():
        if 'offsets_raw' in name:
            init_data = initial_offsets[name].cpu().numpy()
            final_data = param.data.cpu().numpy()
            
            # Shape is [num_patterns, num_points, 2] where 2 = (x, y)
            num_patterns, num_points, _ = init_data.shape
            
            layer_data = {'patterns': {}}
            
            for p in range(num_patterns):
                pattern_data = {'points': []}
                
                for pt in range(num_points):
                    init_x, init_y = float(init_data[p, pt, 0]), float(init_data[p, pt, 1])
                    final_x, final_y = float(final_data[p, pt, 0]), float(final_data[p, pt, 1])
                    delta_x, delta_y = final_x - init_x, final_y - init_y
                    
                    pattern_data['points'].append({
                        'point_id': pt,
                        'initial': {'x': init_x, 'y': init_y},
                        'final': {'x': final_x, 'y': final_y},
                        'delta': {'x': delta_x, 'y': delta_y}
                    })
                
                # Calculate displacement for this pattern
                pattern_init = init_data[p]
                pattern_final = final_data[p]
                total_displacement = float(np.sqrt(((pattern_final - pattern_init) ** 2).sum(axis=1)).mean())
                pattern_data['avg_displacement'] = total_displacement
                
                layer_data['patterns'][f'pattern_{p}'] = pattern_data
            
            offset_logs['detailed_positions'][name] = layer_data
            
            # Overall displacement for this layer
            disp = np.sqrt(((final_data - init_data) ** 2).sum(axis=-1)).mean()
            total_disp_all += disp
            num_offsets += 1
            offset_logs['displacement_summary'][name] = float(disp)
    
    offset_logs['displacement_summary']['total_average'] = float(total_disp_all / num_offsets) if num_offsets > 0 else 0.0
    
    # Save to JSON file
    logs_dir = os.path.join(output_dir, 'Logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f'offset_logs_{num_epochs}epochs_v2.json'
    log_path = os.path.join(logs_dir, log_filename)
    
    with open(log_path, 'w') as f:
        json.dump(offset_logs, f, indent=2)
    
    print(f"\n📄 Detailed offset logs saved to: {log_path}")
    
    # Print brief summary to terminal
    print("\n📊 Brief Summary:")
    for name in offset_logs['displacement_summary']:
        if name != 'total_average':
            print(f"   {name}: displacement = {offset_logs['displacement_summary'][name]:.4f}")
    print(f"   TOTAL average displacement: {offset_logs['displacement_summary']['total_average']:.4f}")
    print("=" * 60)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_attack_model_v2.pth')
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'initial_offsets': initial_offsets,
        'config': model_config,
        'attack_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'base_lr': base_lr,
            'offset_lr': offset_lr,
        },
        'history': history
    }, final_model_path)
    
    print(f"\n✅ Attack complete!")
    print(f"   Worst model: {os.path.join(output_dir, 'worst_model_v2.pth')}")
    print(f"   Final model: {final_model_path}")
    print(f"   Best attack loss: {best_attack_loss:.4f}")
    print("=" * 60)
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generalized Reverse Training Attack for LBPNet Offsets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reverse_training_attack_general.py --dataset mnist
  python reverse_training_attack_general.py --dataset svhn --epochs 30
  python reverse_training_attack_general.py --dataset mnist_adaptive_p --offset-lr 1e-3
  python reverse_training_attack_general.py --list-datasets
        """
    )
    
    parser.add_argument('--dataset', type=str, default='mnist',
                        help=f'Dataset to attack. Options: {list_datasets()}')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of attack epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--base-lr', type=float, default=None,
                        help='Base learning rate for non-offset params')
    parser.add_argument('--offset-lr', type=float, default=None,
                        help='Offset learning rate (gradient ascent)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets and exit')
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("\nAvailable datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        sys.exit(0)
    
    reverse_training_attack(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        offset_lr=args.offset_lr,
        device=args.device,
        custom_output_dir=args.output_dir
    )
