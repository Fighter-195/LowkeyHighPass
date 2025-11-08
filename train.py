import torch
import sys
from pathlib import Path
from audio_zoom_transformer import train_model
import time

def main():
    # Configuration
    config = {
        'train_data': 'data/train/',
        'val_data': 'data/val/',
        'n_epochs': 100,
        'batch_size': 4,  # Reduce if GPU memory issues
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'log_file': 'training_log.txt'
    }
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")
    
    # Check data exists
    if not Path(config['train_data']).exists():
        print(f"ERROR: Training data not found at {config['train_data']}")
        sys.exit(1)
    
    # Count training files
    train_files = list(Path(config['train_data']).glob('*.mat'))
    print(f"Found {len(train_files)} training files")
    
    if config['val_data'] and Path(config['val_data']).exists():
        val_files = list(Path(config['val_data']).glob('*.mat'))
        print(f"Found {len(val_files)} validation files")
    
    print("\nStarting training in 3 seconds...")
    time.sleep(3)
    
    # Start training
    try:
        train_model(**config)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Model checkpoints saved in:", config['save_dir'])
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()