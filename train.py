#!/usr/bin/env python3
"""
Complete Training Script for Urban Sound Classification System
Usage: python train.py --data_path /path/to/UrbanSound8K --epochs 150
"""

import argparse
import os
import sys
from pathlib import Path
import traceback

import torch
from torch.utils.data import DataLoader
import pandas as pd

# Import custom modules
# NOTE: The ModelExporter should be imported here if it's used directly in main/test_model/export_model
# Assuming ModelExporter and run_complete_test_suite are correctly defined in urban_sound_utils/urban_sound_system
try:
    from urban_sound_system import (
        Config, EfficientUrbanSoundCNN, UrbanSoundDataset, Trainer
    )
    from urban_sound_utils import run_complete_test_suite, ModelExporter
except ImportError as e:
    print(f"ERROR: Could not import custom modules. Please ensure 'urban_sound_system' and 'urban_sound_utils' are in your PYTHONPATH and contain the required classes/functions.")
    print(f"Details: {e}")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Urban Sound Classification System'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='UrbanSound8K',
        help='Path to UrbanSound8K dataset'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run testing on trained model (requires a saved model or resume path)'
    )
    parser.add_argument(
        '--export_only',
        action='store_true',
        help='Only export model for deployment (requires a saved model or resume path)'
    )

    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        # Safely check if a GPU is available before querying its properties
        if torch.cuda.device_count() > 0:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
             print("  WARNING: 'cuda' device selected, but no GPU found.")

    return device


def load_datasets(args, config):
    """Load and prepare datasets"""
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    # Check if dataset exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("\nTo use this system:")
        print("1. Download UrbanSound8K from: https://urbansounddataset.weebly.com/")
        print("2. Extract the dataset")
        print("3. Run: python train.py --data_path /path/to/UrbanSound8K")
        sys.exit(1)

    # Load metadata
    metadata_path = data_path / config.METADATA_FILE
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found at {metadata_path}")
        sys.exit(1)

    metadata = pd.read_csv(metadata_path)
    print(f"✓ Loaded metadata: {len(metadata)} samples")

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(metadata)}")
    print(f"  Classes: {config.NUM_CLASSES}")
    print(f"  Samples per class:")
    for class_name in config.CLASS_NAMES:
        # Assuming 'class' column holds the class name string
        count = len(metadata[metadata['class'] == class_name])
        print(f"    {class_name}: {count}")

    # Create datasets
    print(f"\nCreating datasets...")
    # NOTE: Assuming config.TRAIN_FOLDS, config.VAL_FOLD, config.TEST_FOLD are defined lists/ints in Config
    train_dataset = UrbanSoundDataset(
        args.data_path, metadata, config.TRAIN_FOLDS, config, augment=True
    )
    val_dataset = UrbanSoundDataset(
        args.data_path, metadata, [config.VAL_FOLD], config, augment=False
    )
    test_dataset = UrbanSoundDataset(
        args.data_path, metadata, [config.TEST_FOLD], config, augment=False
    )

    # Create dataloaders
    # NOTE: Using config.DEVICE from the updated config object
    pin_memory_val = config.DEVICE.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory_val
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory_val
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory_val
    )

    print(f"✓ Datasets created successfully")

    return train_loader, val_loader, test_loader, metadata


def create_model(config):
    """Create and initialize model"""
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)

    model = EfficientUrbanSoundCNN(num_classes=config.NUM_CLASSES)
    param_count = model.count_parameters() # Assuming this method exists on the model

    print(f"Model: EfficientUrbanSoundCNN")
    print(f"Parameters: {param_count:,}")
    print(f"Target: <1,500,000 parameters")
    print(f"Status: {'✓ PASS' if param_count < 1_500_000 else '✗ FAIL'}")

    if param_count >= 1_500_000:
        print("WARNING: Model exceeds parameter budget for edge deployment!")

    # Print model architecture
    print(f"\nModel Architecture:")
    # Move model to device before printing or using it
    model.to(config.DEVICE)
    print(model)

    return model


def train_model(args, config, model, train_loader, val_loader):
    """Train the model"""
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Map location ensures the checkpoint is loaded correctly regardless of save device
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Check if optimizer state exists before loading
        if 'optimizer_state_dict' in checkpoint and trainer.optimizer:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    # Train
    trainer.train()

    return trainer


def load_best_model(model, config):
    """Loads the best model checkpoint into the model object."""
    model_path = os.path.join(config.MODEL_SAVE_PATH, config.BEST_MODEL_NAME)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model (validation accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%)")
        return True
    else:
        print(f"WARNING: No saved best model found at {model_path}")
        print("Using current model state...")
        return False


def test_model(config, model, test_loader):
    """Run complete test suite on the loaded model."""
    print("\n" + "="*80)
    print("TESTING")
    print("="*80)

    # Load best model if it exists, otherwise use current state
    load_best_model(model, config)

    # Run complete test suite (Assuming run_complete_test_suite returns the results dictionary)
    results = run_complete_test_suite(model, test_loader, config)

    return results


def export_model(config, model):
    """Export model for deployment"""
    print("\n" + "="*80)
    print("EXPORTING MODEL")
    print("="*80)

    # Load best model for export
    load_best_model(model, config)

    exporter = ModelExporter(model, config)
    exporter.create_deployment_package()
    print("\n✓ Deployment package created.")


def print_summary(results):
    """Print final summary"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    if results:
        metrics = results.get('metrics', {})
        profile = results.get('profile', {})

        print(f"\n📊 Model Performance:")
        test_acc = metrics.get('overall_accuracy', 0)
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Target Range: 84-87%")

        if 84 <= test_acc <= 87:
            print(f"  Status: ✓ TARGET ACHIEVED")
        elif test_acc > 87:
            print(f"  Status: ⚠ Above target (may indicate overfitting)")
        else:
            print(f"  Status: ✗ Below target")

        print(f"\n🔧 Model Profile:")
        total_params = profile.get('total_params', 0)
        print(f"  Parameters: {total_params:,}")
        print(f"  Model Size: {profile.get('model_size_mb', 0):.2f} MB")
        print(f"  Target: <1.5M parameters - {'✓' if total_params < 1_500_000 else '✗'}")

        print(f"\n📁 Generated Files (Check 'models/' and 'deployment_package/' directories):")
        # These are expected files from run_complete_test_suite and export_model
        print(f"  ✓ models/best_urban_sound_model.pth")
        print(f"  ✓ evaluation_results/confusion_matrix.png")
        print(f"  ✓ evaluation_results/per_class_metrics.png")
        print(f"  ✓ evaluation_results/metrics.json")
        print(f"  ✓ deployment_package/model.onnx")
        print(f"  ✓ deployment_package/model.pt")
        print(f"  ✓ deployment_package/config.json")
        print(f"  ✓ deployment_package/README.md")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Setup configuration
    config = Config()
    config.DATA_PATH = args.data_path
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.DEVICE = setup_device(args.device) # Set the device in config
    # Ensure num_workers is also available in config if needed elsewhere
    # config.NUM_WORKERS = args.num_workers

    # Print header
    print("\n" + "="*80)
    print("URBAN SOUND CLASSIFICATION SYSTEM")
    print("Production-Ready Training Pipeline")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Workers: {args.num_workers}")

    # Create model
    model = create_model(config)
    results = None

    # Handle 'export_only' or 'test_only' modes which don't require full data loading for training
    if args.export_only or args.test_only:
        # Load datasets only to get test_loader for testing, or to ensure data path is valid
        # We only need enough to perform the test or get class names/config info.
        # This is slightly inefficient but necessary to initialize config-dependent loaders/info.
        try:
            _, _, test_loader, _ = load_datasets(args, config)
        except SystemExit:
            # load_datasets handles data not found errors by exiting
            return

        if args.export_only:
            export_model(config, model)
            print("\n✓ Export complete!")
            return

        if args.test_only:
            # test_model handles loading the best model if it exists
            results = test_model(config, model, test_loader)
            print_summary(results)
            return

    # Full training pipeline
    # Load datasets for training
    try:
        train_loader, val_loader, test_loader, _ = load_datasets(args, config)
    except SystemExit:
        # load_datasets handles data not found errors by exiting
        return

    try:
        # Train
        train_model(args, config, model, train_loader, val_loader)

        # Test (Loads best model saved by trainer)
        results = test_model(config, model, test_loader)

        # Export (Loads best model saved by trainer)
        export_model(config, model)

        # Summary
        print_summary(results)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Partial results may be saved in models/ directory")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR during pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Ensure config is initialized before main is called if it were accessed globally,
    # but since everything is passed via main, this is safe.
    main()