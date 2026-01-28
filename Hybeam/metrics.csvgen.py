"""
generate_metrics.py

This script loads a pre-trained **HyBeam-FT-JNF** model (from Hybeam_train_v3.py)
and runs a full evaluation on the validation set, generating a metrics.csv file.

This version is corrected to match the "Hybrid 2" architecture.

Example usage:
python generate_metrics.py --data_dir "reverb_dataset_2channel" \
                           --model_path "hybeam_ft_jnf_best.pth" \
                           --output_csv "final_metrics.csv"
"""

import glob
import json
import os
import random
from math import radians
import csv
import argparse
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Import metrics libraries with a check ---
try:
    from pesq import pesq
    from pystoi import stoi
    HAVE_METRICS_LIBS = True
    print("Successfully imported pesq and pystoi for metric calculation.")
except ImportError:
    HAVE_METRICS_LIBS = False
    print("="*50)
    print("WARNING: `pesq` or `pystoi` not found.")
    print("Please run: pip install pesq pystoi")
    print("Metrics.csv generation will be skipped.")
    print("="*50)
    sys.exit(1)
# --- END ---

# --------------------------------------------------------------------------- #
# --- HELPER FUNCTIONS (Copied from Hybeam_train_v3.py) ---
# --------------------------------------------------------------------------- #

def compute_das_beamformer(mixture_stft, mic_positions, target_direction, fs=16000, n_fft=512):
    """
    Compute Delay-and-Sum (DAS) beamformer outputs for multiple directions
    """
    c = 343.0
    n_channels, n_freq, n_time = mixture_stft.shape
    n_directions = len(target_direction)

    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    beamformer_outputs = np.zeros((n_directions, n_freq, n_time), dtype=complex)

    for d, direction in enumerate(target_direction):
        theta = np.radians(direction)
        for f_idx, freq in enumerate(freqs):
            if freq == 0:
                continue

            delays = []
            for mic_pos in mic_positions.T:
                u = np.array([np.cos(theta), np.sin(theta), 0])
                delay = np.dot(mic_pos, u) / c
                delays.append(delay)

            phases = np.exp(-1j * 2 * np.pi * freq * np.array(delays))
            steering_vector = phases / np.linalg.norm(phases)
            beamformer_outputs[d, f_idx, :] = np.dot(steering_vector.conj(), mixture_stft[:, f_idx, :])

    return beamformer_outputs

def create_hybrid_input(mixture_stft, beamformer_outputs, cutoff_freq=1500, fs=16000, n_fft=512):
    """
    Create hybrid input: microphones at low frequencies, beamformers at high frequencies
    """
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    cutoff_bin = np.argmax(freqs >= cutoff_freq)

    n_mics, n_freq, n_time = mixture_stft.shape
    n_beams = beamformer_outputs.shape[0]

    hybrid_input = np.zeros((n_mics, n_freq, n_time), dtype=complex)

    for f_idx in range(n_freq):
        if f_idx < cutoff_bin:
            hybrid_input[:, f_idx, :] = mixture_stft[:, f_idx, :]
        else:
            for ch in range(n_mics):
                beam_idx = ch % n_beams
                hybrid_input[ch, f_idx, :] = beamformer_outputs[beam_idx, f_idx, :]

    return hybrid_input

def calculate_spatial_weights(target_angle, fov_center, fov_half_width, num_channels=2):
    """
    Calculate spatial weights based on FOV information
    """
    target_rad = radians(target_angle)
    fov_center_rad = radians(fov_center)

    if num_channels == 2:
        mic_angles = np.array([np.pi, 0.0]) # [Mic 0 (left), Mic 1 (right)]
    else:
        mic_angles = np.linspace(0, 2*np.pi, num_channels, endpoint=False)

    alignment_scores = []
    for mic_angle in mic_angles:
        target_diff = min(abs(mic_angle - target_rad), 2*np.pi - abs(mic_angle - target_rad))
        fov_diff = min(abs(mic_angle - fov_center_rad), 2*np.pi - abs(mic_angle - fov_center_rad))

        if fov_diff <= radians(fov_half_width):
            score = 1.0 - (target_diff / np.pi)
        else:
            score = 0.1 * (1.0 - (target_diff / np.pi))
        alignment_scores.append(score)

    scores_array = np.array(alignment_scores)
    if np.sum(scores_array) > 0:
        weights = scores_array / np.sum(scores_array)
    else:
        weights = np.ones(num_channels) / num_channels

    return torch.tensor(weights, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# --- DATASET (Minimal) ---
# --------------------------------------------------------------------------- #

class HyBeamFTJNFDataset(Dataset):
    """
    Minimal version of the dataset class, only used to find
    the file paths for the validation set.
    """
    def __init__(self, data_dir, max_samples=10000):
        self.data_dir = data_dir
        self.sample_folders = sorted(glob.glob(os.path.join(data_dir, 'sample_*')))[:max_samples]
        print(f"Found {len(self.sample_folders)} total samples in dataset")

    def __len__(self):
        return len(self.sample_folders)

    def normalize_signal(self, signal):
        """Normalize signal to prevent gradient issues"""
        if signal.ndim == 1:
            max_val = np.max(np.abs(signal))
            if max_val > 0: return signal / max_val
            return signal
        else:
            normalized = np.zeros_like(signal)
            for c in range(signal.shape[0]):
                channel_max = np.max(np.abs(signal[c]))
                if channel_max > 0:
                    normalized[c] = signal[c] / channel_max
            return normalized

    def __getitem__(self, idx):
        return self.sample_folders[idx]

# --------------------------------------------------------------------------- #
# --- HyBeam-FT-JNF MODEL (Copied from Hybeam_train_v3.py) ---
# --------------------------------------------------------------------------- #

class HyBeamFTJNF(nn.Module):
    def __init__(self, num_channels=2, num_freq_bins=257, lstm_units=128, fov_feature_dim=4,
                 use_bandwise_processing=True):
        super().__init__()

        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        self.fov_feature_dim = fov_feature_dim
        self.use_bandwise_processing = use_bandwise_processing

        self.fov_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, fov_feature_dim),
            nn.Tanh()
        )

        input_feature_dim = num_channels * 2 + fov_feature_dim
        if use_bandwise_processing:
            self.low_freq_encoder = nn.Sequential(
                nn.Linear(input_feature_dim, lstm_units),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.high_freq_encoder = nn.Sequential(
                nn.Linear(input_feature_dim, lstm_units),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.freq_encoder = nn.Sequential(
                nn.Linear(input_feature_dim, lstm_units),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        self.lstm1 = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )

        self.lstm2 = nn.LSTM(
            input_size=lstm_units * 2,
            hidden_size=lstm_units,
            bidirectional=True,
            batch_first=True,
            num_layers=1
        )

        self.mask_estimator = nn.Sequential(
            nn.Linear(lstm_units * 2, lstm_units),
            nn.ReLU(),
            nn.Linear(lstm_units, 2),
            nn.Tanh()
        )

        self.spatial_attention = nn.Sequential(
            nn.Linear(num_channels, 32),
            nn.ReLU(),
            nn.Linear(32, num_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, spatial_weights, fov_center, fov_width, target_angle, cutoff_freq=1500):
        B, C, F, T, _ = x.shape

        fov_features = torch.stack([
            target_angle / 360.0,
            fov_center / 360.0,
            fov_width / 360.0,
            torch.mean(spatial_weights, dim=-1)
        ], dim=-1)

        encoded_fov = self.fov_encoder(fov_features)

        spatial_attn = self.spatial_attention(spatial_weights)
        spatial_attn = spatial_attn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x * spatial_attn

        x_wide = x.permute(0, 3, 2, 1, 4)
        x_wide = x_wide.reshape(B * T, F, C * 2)

        encoded_fov_expanded = encoded_fov.unsqueeze(1).unsqueeze(1)
        encoded_fov_expanded = encoded_fov_expanded.repeat(1, T, F, 1)
        encoded_fov_expanded = encoded_fov_expanded.reshape(B * T, F, self.fov_feature_dim)

        x_wide_enhanced = torch.cat([x_wide, encoded_fov_expanded], dim=-1)

        if self.use_bandwise_processing:
            cutoff_bin = int((cutoff_freq / 8000) * F)

            low_freq_features = self.low_freq_encoder(x_wide_enhanced[:, :cutoff_bin, :])
            high_freq_features = self.high_freq_encoder(x_wide_enhanced[:, cutoff_bin:, :])

            processed_features = torch.cat([low_freq_features, high_freq_features], dim=1)
        else:
            processed_features = self.freq_encoder(x_wide_enhanced)

        out_wide, _ = self.lstm1(processed_features)

        out_wide = out_wide.reshape(B, T, F, -1)
        x_narrow = out_wide.permute(0, 2, 1, 3)
        x_narrow = x_narrow.reshape(B * F, T, -1)

        out_narrow, _ = self.lstm2(x_narrow)

        out_processed = out_narrow.reshape(B, F, T, -1)
        mask = self.mask_estimator(out_processed)

        return mask

# --------------------------------------------------------------------------- #
# --- SI-SDR METRIC FUNCTION ---
# --------------------------------------------------------------------------- #

def si_sdr_metric(estimate, target, epsilon=1e-8):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.
    """
    if not isinstance(estimate, torch.Tensor):
        estimate = torch.tensor(estimate)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    s_target_energy = torch.sum(target**2, dim=-1, keepdim=True) + epsilon
    s_target = torch.sum(target * estimate, dim=-1, keepdim=True) / s_target_energy * target

    e_noise = estimate - s_target

    sdr = torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + epsilon)

    sdr_db = torch.log10(sdr + epsilon) * 10
    return torch.mean(sdr_db)

# --------------------------------------------------------------------------- #
# --- METRIC EVALUATION FUNCTION (Hybrid 2 Version) ---
# --------------------------------------------------------------------------- #

def evaluate_model_and_save_metrics(model, val_dataset, device, config, output_csv_path):
    """
    Runs a full evaluation on the validation dataset (un-chunked)
    and saves SDR, PESQ, and STOI metrics.
    """
    model.eval()
    results = []
    header = ['sample_id', 'sdr_in', 'sdr_out', 'pesq_in', 'pesq_out', 'stoi_in', 'stoi_out']

    n_fft = config['n_fft']
    hop_length = config['hop_length']
    fs = config['fs']
    max_channels = config['max_channels']
    cutoff_freq = config['cutoff_freq']
    beamformer_directions = config['beamformer_directions']

    hann_win = torch.hann_window(n_fft).to(device)

    normalize_signal = val_dataset.dataset.normalize_signal

    print(f"Running evaluation for metrics on {len(val_dataset.indices)} validation files...")

    with torch.no_grad():
        for idx in tqdm(val_dataset.indices, desc="Evaluating Metrics"):
            try:
                # --- 1. Load Full Files ---
                sample_folder = val_dataset.dataset.sample_folders[idx]
                sample_id = os.path.basename(sample_folder)

                mixture_path = os.path.join(sample_folder, 'mixture.wav')
                target_path = os.path.join(sample_folder, 'target.wav')
                metadata_path = os.path.join(sample_folder, 'metadata.json')

                mixture_audio, fs_mix = sf.read(mixture_path)
                target_audio, fs_target = sf.read(target_path)

                # --- 2. Pre-process Full Files ---
                if fs_mix != fs:
                    mixture_audio = librosa.resample(mixture_audio.T, orig_sr=fs_mix, target_sr=fs).T
                if fs_target != fs:
                    target_audio = librosa.resample(target_audio, orig_sr=fs_target, target_sr=fs)

                if mixture_audio.ndim == 1: mixture_audio = mixture_audio[np.newaxis, :]
                else: mixture_audio = mixture_audio.T
                if target_audio.ndim > 1: target_audio = target_audio[:, 0]

                min_len = min(mixture_audio.shape[1], target_audio.shape[0])
                mixture_audio = mixture_audio[:, :min_len]
                target_audio = target_audio[:min_len]

                n_mics = mixture_audio.shape[0]
                if n_mics > max_channels:
                    mixture_audio = mixture_audio[:max_channels, :]
                elif n_mics < max_channels:
                    padding = np.zeros((max_channels - n_mics, mixture_audio.shape[1]))
                    mixture_audio = np.concatenate([mixture_audio, padding], axis=0)

                mixture_audio = normalize_signal(mixture_audio)
                target_audio = normalize_signal(target_audio)

                with open(metadata_path, 'r') as f: metadata = json.load(f)

                # --- 3. Run HyBeam Pre-processing (STFT, Beamforming, Hybrid Input) ---
                mixture_stfts = []
                for ch in range(mixture_audio.shape[0]):
                    stft = librosa.stft(mixture_audio[ch], n_fft=n_fft, hop_length=hop_length, window='hann')
                    mixture_stfts.append(stft)
                mixture_stft = np.stack(mixture_stfts, axis=0)

                if max_channels == 2:
                    mic_positions = np.array([[-0.04, 0.04], [0.0, 0.0], [0.0, 0.0]])
                else:
                    radius = 0.05
                    mic_angles = np.linspace(0, 2*np.pi, max_channels, endpoint=False)
                    mic_positions = np.array([radius * np.cos(mic_angles), radius * np.sin(mic_angles), np.zeros(max_channels)])

                beamformer_outputs = compute_das_beamformer(mixture_stft, mic_positions, beamformer_directions, fs=fs, n_fft=n_fft)
                hybrid_stft = create_hybrid_input(mixture_stft, beamformer_outputs, cutoff_freq, fs=fs, n_fft=n_fft)

                spatial_weights = calculate_spatial_weights(
                    metadata['targetAngle_deg'], metadata['fov_angle_deg'],
                    metadata['fov_width_deg'] / 2.0, max_channels
                )

                # --- 4. Prepare Tensors ---
                hybrid_stft_tensor = torch.from_numpy(hybrid_stft).cfloat()
                hybrid_stft_ri = torch.stack((hybrid_stft_tensor.real, hybrid_stft_tensor.imag), dim=-1)
                model_input = hybrid_stft_ri.unsqueeze(0).to(device)

                spatial_weights_tensor = spatial_weights.unsqueeze(0).to(device)
                fov_center_tensor = torch.tensor([metadata['fov_angle_deg']], dtype=torch.float32).to(device)
                fov_width_tensor = torch.tensor([metadata['fov_width_deg']], dtype=torch.float32).to(device)
                target_angle_tensor = torch.tensor([metadata['targetAngle_deg']], dtype=torch.float32).to(device)

                # This is the reference for "Hybrid 2"
                ref_stft_complex = torch.from_numpy(beamformer_outputs[0]).cfloat().to(device)

                # --- 5. Run Model Inference ---
                predicted_mask_ri = model(
                    model_input, spatial_weights_tensor, fov_center_tensor,
                    fov_width_tensor, target_angle_tensor, cutoff_freq
                )

                # --- 6. Post-process to get Waveforms ---
                M_S_complex = torch.complex(predicted_mask_ri[0, ..., 0], predicted_mask_ri[0, ..., 1])
                enhanced_stft = M_S_complex * ref_stft_complex

                enhanced_wav_tensor = torch.istft(
                    enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                    window=hann_win, length=target_audio.shape[0]
                )
                enhanced_wav = enhanced_wav_tensor.cpu().numpy()

                # "Input" waveform is the forward beamformer
                ref_beam_wav_tensor = torch.istft(
                    ref_stft_complex, n_fft=n_fft, hop_length=hop_length,
                    window=hann_win, length=target_audio.shape[0]
                )
                ref_beam_wav = ref_beam_wav_tensor.cpu().numpy()

                # --- 7. Calculate All Metrics ---
                target_wav_np = target_audio

                min_len_final = min(len(target_wav_np), len(ref_beam_wav), len(enhanced_wav))
                target_wav_np = target_wav_np[:min_len_final]
                ref_beam_wav = ref_beam_wav[:min_len_final]
                enhanced_wav = enhanced_wav[:min_len_final]

                sdr_in = si_sdr_metric(ref_beam_wav, target_wav_np).item()
                sdr_out = si_sdr_metric(enhanced_wav, target_wav_np).item()

                pesq_in = pesq(fs, target_wav_np, ref_beam_wav, 'wb')
                pesq_out = pesq(fs, target_wav_np, enhanced_wav, 'wb')

                stoi_in = stoi(target_wav_np, ref_beam_wav, fs)
                stoi_out = stoi(target_wav_np, enhanced_wav, fs)

                results.append({
                    'sample_id': sample_id,
                    'sdr_in': sdr_in, 'sdr_out': sdr_out,
                    'pesq_in': pesq_in, 'pesq_out': pesq_out,
                    'stoi_in': stoi_in, 'stoi_out': stoi_out
                })

            except Exception as e:
                print(f"\n[Metrics Error] Skipping {sample_id}: {e}")

    # --- 8. Write to CSV ---
    if results:
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSuccessfully wrote metrics to {output_csv_path}")
    else:
        print("\nNo metric results to write.")

# --------------------------------------------------------------------------- #
# --- MAIN EXECUTION ---
# --------------------------------------------------------------------------- #

def main(args):
    """
    Main function to load model, dataset, and run evaluation.
    """

    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = {
        'n_fft': 512,
        'hop_length': 256,
        'fs': 16000,
        'max_channels': 2,
        'cutoff_freq': 1500.0,
        'beamformer_directions': [0, 90, 180, 270]
    }

    # --- 2. Load Dataset to find Validation Indices ---
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at '{args.data_dir}'")
        sys.exit(1)

    dataset = HyBeamFTJNFDataset(data_dir=args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Found {len(val_dataset)} validation samples to evaluate.")

    # --- 3. Load Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        sys.exit(1)

    print(f"Loading best model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Re-initialize model from checkpoint's config
    model_config = checkpoint.get('hybeam_config', {
        'max_channels': 2,
    })

    model = HyBeamFTJNF(
        num_channels=model_config.get('max_channels', 2),
        num_freq_bins=257,
        lstm_units=128, # Assuming 128, add to checkpoint later if needed
        fov_feature_dim=4 # Assuming 4, add to checkpoint later if needed
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Successfully loaded model from epoch {epoch}.")

    # --- 4. Run Evaluation ---
    evaluate_model_and_save_metrics(
        model, val_dataset, device, config, args.output_csv
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for a trained HyBeam-FT-JNF model.")

    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the *root* dataset directory (e.g., 'reverb_dataset')")

    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained .pth checkpoint file (e.g., 'hybeam_ft_jnf_best.pth')")

    parser.add_argument('--output_csv', type=str, default="final_metrics.csv",
                        help="Path to save the final metrics CSV file.")

    args = parser.parse_args()

    main(args)