from pathlib import Path
import json
import scipy.io.wavfile as wavfile
import numpy as np

data_dir = Path('Abhinav_Former/data/train')
samples = [d for d in data_dir.iterdir() if d.is_dir()]

print(f"Found {len(samples)} samples in {data_dir}")

if len(samples) > 0:
    print(f"\nChecking first sample: {samples[0]}")
    
    # Check files exist
    required = ['mixture.wav', 'target.wav', 'metadata.json']
    for f in required:
        exists = (samples[0] / f).exists()
        print(f"  {f}: {'✓' if exists else '✗ MISSING'}")
    
    # Check mixture.wav is stereo
    if (samples[0] / 'mixture.wav').exists():
        sr, mixture = wavfile.read(samples[0] / 'mixture.wav')
        print(f"\n  mixture.wav: shape={mixture.shape}, sr={sr}")
        if len(mixture.shape) == 2 and mixture.shape[1] == 2:
            print(f"  ✓ Stereo (2 channels)")
        elif len(mixture.shape) == 1:
            print(f"  ✗ Mono - Expected stereo (2 channels)")
        else:
            print(f"  ✗ Expected stereo, got shape {mixture.shape}")
    
    # Check target.wav
    if (samples[0] / 'target.wav').exists():
        sr, target = wavfile.read(samples[0] / 'target.wav')
        print(f"\n  target.wav: shape={target.shape}, sr={sr}")
    
    # Check metadata
    if (samples[0] / 'metadata.json').exists():
        with open(samples[0] / 'metadata.json') as f:
            meta = json.load(f)
        print(f"\n  Metadata: {meta}")
        
        # Validate metadata has required fields
        if 'fov_angle' in meta and 'fov_width' in meta:
            print(f"  ✓ FOV angle: {meta['fov_angle']}°")
            print(f"  ✓ FOV width: {meta['fov_width']}°")
        else:
            print(f"  ✗ Missing fov_angle or fov_width in metadata")
else:
    print("✗ No sample directories found!")

print("\n" + "="*60)
print("Data check complete!")