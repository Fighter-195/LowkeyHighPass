import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import traceback
import torch

print('Starting dry-run test: attempting to import module')
try:
    from audio_zoom_transformer import AudioZoomingTransformer, calculate_dfin_fout
    print('Imported audio_zoom_transformer successfully')
except Exception as e:
    print('Failed to import audio_zoom_transformer:')
    traceback.print_exc()
    raise


def main():
    device = torch.device('cpu')

    # Instantiate model with defaults
    model = AudioZoomingTransformer()
    model.to(device)
    model.eval()

    batch = 2
    n_mics = model.n_mics
    n_freqs = model.n_freqs
    n_frames = 10

    # Create synthetic complex STFT: [batch, n_mics, n_freqs, n_frames]
    real = torch.randn(batch, n_mics, n_freqs, n_frames, dtype=torch.float32, device=device)
    imag = torch.randn(batch, n_mics, n_freqs, n_frames, dtype=torch.float32, device=device)
    mic_stfts = torch.complex(real, imag)

    # Also test calculate_dfin_fout function using the synthetic mic_stfts
    try:
        dfin_from_func = calculate_dfin_fout(mic_stfts)
        print('calculate_dfin_fout output shape =', dfin_from_func.shape, 'dtype =', dfin_from_func.dtype)
    except Exception:
        print('calculate_dfin_fout raised an exception:')
        traceback.print_exc()
        raise

    # Create synthetic DFinFout: [batch, n_frames, n_freqs]
    dfin_fout = torch.randn(batch, n_frames, n_freqs, dtype=torch.float32, device=device)

    # Run forward
    with torch.no_grad():
        try:
            output, mask_in, mask_out, weights = model(mic_stfts, dfin_fout)
        except Exception as e:
            print('Forward pass failed with exception:')
            raise

    print('Forward pass succeeded')
    print('output.shape =', output.shape, 'dtype =', output.dtype)
    print('mask_in.shape =', mask_in.shape, 'dtype =', mask_in.dtype)
    print('mask_out.shape =', mask_out.shape, 'dtype =', mask_out.dtype)
    print('weights.shape =', weights.shape, 'dtype =', weights.dtype)


if __name__ == '__main__':
    main()
