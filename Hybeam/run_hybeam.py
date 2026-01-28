"""
run_hybeam_v2.py

This script runs inference using a trained HyBeam-FT-JNF model.
IT IS MODIFIED for a 2-CHANNEL model using the "Hybrid 2" architecture.

It takes the following command-line arguments:
  --input_file: Path to the 2-channel .wav file to process.
  --output_file: Path to save the enhanced single-channel .wav file.
  --model_path: Path to the trained 'hybeam_ft_jnf_best.pth' file.
  --target_angle: The target speaker's angle (e.g., 0).
  --fov_center: The center of the camera's FOV (e.g., 0).
  --fov_width: The full width of the FOV (e.g., 60).

Example usage from your terminal:
python run_hybeam_v2.py --input_file "mixture_from_matlab.wav" --output_file "enhanced.wav" --model_path "hybeam_ft_jnf_best.pth" --target_angle 0 --fov_center 0 --fov_width 60
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import argparse
from math import radians, cos, sin
import sys


# --------------------------------------------------------------------------- #
# --- MODEL CLASS (Copied from Hybeam_train_v2.py) ---
# --------------------------------------------------------------------------- #

class HyBeamFTJNF(nn.Module):
    def __init__(self, num_channels=2, num_freq_bins=257, lstm_units=128, fov_feature_dim=4,
                 use_bandwise_processing=True):
        super().__init__()

        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        self.fov_feature_dim = fov_feature_dim
        self.use_bandwise_processing = use_bandwise_processing

        # FOV feature processing
        self.fov_encoder = nn.Sequential(
            nn.Linear(4, 16),  # target_angle, fov_center, fov_width, spatial_weights_mean
            nn.ReLU(),
            nn.Linear(16, fov_feature_dim),
            nn.Tanh()
        )

        # Bandwise processing components
        input_feature_dim = num_channels * 2 + fov_feature_dim
        if use_bandwise_processing:
            # Low-frequency processing (microphone-dominated)
            self.low_freq_encoder = nn.Sequential(
                nn.Linear(input_feature_dim, lstm_units),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            # High-frequency processing (beamformer-dominated)
            self.high_freq_encoder = nn.Sequential(
                nn.Linear(input_feature_dim, lstm_units),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            # Standard processing for all frequencies
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
            nn.Linear(lstm_units, 2),  # Real and imaginary components
            nn.Tanh()
        )

        # Spatial attention based on FOV
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
# --- HELPER FUNCTIONS (Copied from Hybeam_train_v2.py) ---
# --------------------------------------------------------------------------- #

def compute_das_beamformer(mixture_stft, mic_positions, target_direction, fs=16000, n_fft=512):
    c = 343.0
    n_channels, n_freq, n_time = mixture_stft.shape
    n_directions = len(target_direction)

    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
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
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
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
    target_rad = radians(target_angle)
    fov_center_rad = radians(fov_center)

    if num_channels == 2:
        mic_angles = np.array([np.pi, 0.0])  # [Mic 0 (left), Mic 1 (right)]
    else:
        mic_angles = np.linspace(0, 2 * np.pi, num_channels, endpoint=False)

    alignment_scores = []
    for mic_angle in mic_angles:
        target_diff = min(abs(mic_angle - target_rad), 2 * np.pi - abs(mic_angle - target_rad))
        fov_diff = min(abs(mic_angle - fov_center_rad), 2 * np.pi - abs(mic_angle - fov_center_rad))

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


def normalize_signal(signal):
    if signal.ndim == 1:
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal
    else:
        normalized = np.zeros_like(signal)
        for c in range(signal.shape[0]):
            channel_max = np.max(np.abs(signal[c]))
            if channel_max > 0:
                normalized[c] = signal[c] / channel_max
        return normalized


# --------------------------------------------------------------------------- #
# --- MAIN INFERENCE FUNCTION ---
# --------------------------------------------------------------------------- #

def run_inference(args):
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_fft = 512
    hop_length = 256
    fs = 16000
    cutoff_freq = 1500.0
    beamformer_directions = [0, 90, 180, 270]
    max_channels = 2  # --- MODIFIED: 2-channel constraint ---

    # --- 2. Load Model ---
    print(f"Loading model from {args.model_path}...")
    model = HyBeamFTJNF(num_channels=max_channels, num_freq_bins=(n_fft // 2 + 1))

    # Load state dict
    try:
        # Try loading checkpoint first
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If that fails, assume it's just the state dict
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure --model_path points to the correct .pth file (e.g., 'hybeam_ft_jnf_best.pth')")
        sys.exit(1)

    model.to(device)
    model.eval()
    print("Model loaded.")

    # --- 3. Load Audio ---
    print(f"Loading audio from {args.input_file}...")
    mixture_audio, orig_fs = sf.read(args.input_file)

    # Ensure correct format (channels, samples)
    if mixture_audio.ndim == 1:
        mixture_audio = mixture_audio[np.newaxis, :]
    else:
        mixture_audio = mixture_audio.T

    # Resample if necessary
    if orig_fs != fs:
        print(f"Resampling audio from {orig_fs}Hz to {fs}Hz...")
        mixture_audio = librosa.resample(mixture_audio, orig_sr=orig_fs, target_sr=fs)

    # Pad or truncate channels to match model
    n_mics = mixture_audio.shape[0]
    print(f"Input audio has {n_mics} channels.")
    if n_mics > max_channels:
        print(f"Warning: Truncating audio from {n_mics} to {max_channels} channels.")
        mixture_audio = mixture_audio[:max_channels, :]
    elif n_mics < max_channels:
        print(f"Warning: Padding audio from {n_mics} to {max_channels} channels with silence.")
        padding = np.zeros((max_channels - n_mics, mixture_audio.shape[1]))
        mixture_audio = np.concatenate([mixture_audio, padding], axis=0)

    # Normalize
    mixture_audio = normalize_signal(mixture_audio)

    # --- 4. Pre-processing Pipeline (from Dataset) ---
    print("Running pre-processing...")

    # STFT
    mixture_stfts = []
    for ch in range(mixture_audio.shape[0]):
        stft = librosa.stft(mixture_audio[ch], n_fft=n_fft, hop_length=hop_length, window='hann')
        mixture_stfts.append(stft)
    mixture_stft = np.stack(mixture_stfts, axis=0)

    # --- MODIFIED: Mic Positions ---
    if max_channels == 2:
        mic_positions = np.array([
            [-0.04, 0.04],  # x-coordinates
            [0.0, 0.0],  # y-coordinates
            [0.0, 0.0]  # z-coordinates
        ])
    else:
        # Fallback
        print(f"Warning: Using fallback circular array for {max_channels} channels.")
        radius = 0.05
        mic_angles = np.linspace(0, 2 * np.pi, max_channels, endpoint=False)
        mic_positions = np.array([
            radius * np.cos(mic_angles),
            radius * np.sin(mic_angles),
            np.zeros(max_channels)
        ])

    beamformer_outputs = compute_das_beamformer(
        mixture_stft, mic_positions, beamformer_directions, fs=fs, n_fft=n_fft
    )

    # Hybrid Input
    hybrid_stft = create_hybrid_input(
        mixture_stft, beamformer_outputs, cutoff_freq, fs=fs, n_fft=n_fft
    )

    # Spatial Weights
    spatial_weights = calculate_spatial_weights(
        args.target_angle, args.fov_center, args.fov_width / 2.0, max_channels
    )

    # --- 5. Prepare Tensors for Model ---
    # Add batch dimension (B=1) and move to device
    hybrid_stft_tensor = torch.from_numpy(hybrid_stft).cfloat()
    hybrid_stft_ri = torch.stack((hybrid_stft_tensor.real, hybrid_stft_tensor.imag), dim=-1)
    model_input = hybrid_stft_ri.unsqueeze(0).to(device)  # (1, C, F, T, 2)

    spatial_weights_tensor = spatial_weights.unsqueeze(0).to(device)  # (1, C)
    fov_center_tensor = torch.tensor([args.fov_center], dtype=torch.float32).to(device)
    fov_width_tensor = torch.tensor([args.fov_width], dtype=torch.float32).to(device)
    target_angle_tensor = torch.tensor([args.target_angle], dtype=torch.float32).to(device)

    # --- MODIFIED: Reference STFT for final masking ("Hybrid 2") ---
    ref_stft_complex = torch.from_numpy(beamformer_outputs[0]).cfloat().to(device)

    # --- 6. Run Inference ---
    print("Running model inference...")
    with torch.no_grad():
        predicted_mask_ri = model(
            model_input,
            spatial_weights_tensor,
            fov_center_tensor,
            fov_width_tensor,
            target_angle_tensor,
            cutoff_freq  # Pass scalar cutoff freq
        )

    # --- 7. Post-processing ---
    print("Applying mask and saving audio...")
    # Remove batch dim and convert mask to complex
    M_S_complex = torch.complex(predicted_mask_ri[0, ..., 0], predicted_mask_ri[0, ..., 1])

    # Apply mask
    enhanced_stft = M_S_complex * ref_stft_complex

    # ISTFT
    enhanced_audio = torch.istft(
        enhanced_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(device),
        length=mixture_audio.shape[1]
    )

    # Save audio
    enhanced_audio_cpu = enhanced_audio.cpu().numpy()
    sf.write(args.output_file, enhanced_audio_cpu, fs)
    print(f"Successfully saved enhanced audio to {args.output_file}")

    # --- 8. Print output path for MATLAB ---
    # This is often used by calling scripts to find the output
    print(f"MATLAB_OUTPUT_PATH:{args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HyBeam-FT-JNF inference (v2: 2-Ch, Hybrid 2)")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the multi-channel .wav file to process.")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save the enhanced single-channel .wav file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth checkpoint file.")
    parser.add_argument('--target_angle', type=float, default=0.0, help="Target speaker's angle in degrees.")
    parser.add_argument('--fov_center', type=float, default=0.0, help="Center of the camera's FOV in degrees.")
    parser.add_argument('--fov_width', type=float, default=60.0, help="Full width of the FOV in degrees.")

    args = parser.parse_args()

    run_inference(args)