"""
HyBeam-FT-JNF: Hybrid Microphone-Beamforming with FOV-enhanced FT-JNF
Combining FOV spatial awareness with HyBeam's frequency-band hybrid approach

VERSION 3:
- Added tqdm progress bars for training and validation loops.
- Added automatic metric generation (SDR, PESQ, STOI) to metrics.csv
  when a new best validation model is found.
- Configured for 2-channel training (max_channels = 2)
- Implements "Hybrid 2" (uses forward beamformer as reference)
- Implements SI-SDR loss as described in the paper
- Fixes bug where loss terms were not being added
- Fixes mic geometry to match 2-channel 8cm testbed
"""

import glob
import json
import os
import random
from math import radians
import csv

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- NEW: Import metrics libraries with a check ---
try:
    from pesq import pesq
    from pystoi import stoi
    HAVE_METRICS_LIBS = True
except ImportError:
    HAVE_METRICS_LIBS = False
    print("="*50)
    print("WARNING: `pesq` or `pystoi` not found.")
    print("Please run: pip install pesq pystoi")
    print("Metrics.csv generation will be skipped.")
    print("="*50)
# --- END NEW ---

# --------------------------------------------------------------------------- #
# --- HYBEAM HYBRID PROCESSING WITH FOV INTEGRATION ---
# --------------------------------------------------------------------------- #

def compute_das_beamformer(mixture_stft, mic_positions, target_direction, fs=16000, n_fft=512):
    """
    Compute Delay-and-Sum (DAS) beamformer outputs for multiple directions
    """
    # ... existing code ...
    c = 343.0  # Speed of sound in m/s
    n_channels, n_freq, n_time = mixture_stft.shape
    n_directions = len(target_direction)
    
    # Convert to frequency values
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    
    beamformer_outputs = np.zeros((n_directions, n_freq, n_time), dtype=complex)
    
    for d, direction in enumerate(target_direction):
        # Convert direction to radians and compute steering vector
        theta = np.radians(direction)

        for f_idx, freq in enumerate(freqs):
            if freq == 0:
                continue
                
            # Compute steering delays
            delays = []
            for mic_pos in mic_positions.T:  # mic_positions shape: (3, n_mics)
                # Simple far-field assumption: delay = (d·u)/c
                # where u is unit vector in direction of arrival
                u = np.array([np.cos(theta), np.sin(theta), 0])
                delay = np.dot(mic_pos, u) / c
                delays.append(delay)
            
            # Convert delays to phase shifts
            phases = np.exp(-1j * 2 * np.pi * freq * np.array(delays))
            
            # Normalize steering vector
            steering_vector = phases / np.linalg.norm(phases)
            
            # Apply beamforming
            beamformer_outputs[d, f_idx, :] = np.dot(steering_vector.conj(), mixture_stft[:, f_idx, :])
    
    return beamformer_outputs

def create_hybrid_input(mixture_stft, beamformer_outputs, cutoff_freq=1500, fs=16000, n_fft=512):
    """
    Create hybrid input: microphones at low frequencies, beamformers at high frequencies
    The number of output channels == number of microphone channels.
    """
    # ... existing code ...
    freqs = np.fft.rfftfreq(n_fft, 1/fs)
    cutoff_bin = np.argmax(freqs >= cutoff_freq)
    
    n_mics, n_freq, n_time = mixture_stft.shape
    n_beams = beamformer_outputs.shape[0]
    
    # Hybrid input: use mics for low freqs, beams for high freqs
    hybrid_input = np.zeros((n_mics, n_freq, n_time), dtype=complex)
    
    for f_idx in range(n_freq):
        if f_idx < cutoff_bin:
            # Low frequencies: use microphone signals
            hybrid_input[:, f_idx, :] = mixture_stft[:, f_idx, :]
        else:
            # High frequencies: use beamformer outputs
            # We map the first n_mics beams to the n_mics channels
            for ch in range(n_mics):
                beam_idx = ch % n_beams
                hybrid_input[ch, f_idx, :] = beamformer_outputs[beam_idx, f_idx, :]
    
    return hybrid_input

def calculate_spatial_weights(target_angle, fov_center, fov_half_width, num_channels=2):
    """
    Calculate spatial weights based on FOV information
    Higher weights for channels better aligned with target within FOV
    """
    # ... existing code ...
    # Convert angles to radians
    target_rad = radians(target_angle)
    fov_center_rad = radians(fov_center)
    
    # Simple microphone array geometry (linear horizontal array)
    if num_channels == 2:
        # Mic 0 at -x (left), Mic 1 at +x (right)
        # Corresponds to angles 180 and 0
        mic_angles = np.array([np.pi, 0.0])
    else:
        # Fallback to circular
        mic_angles = np.linspace(0, 2*np.pi, num_channels, endpoint=False)
    
    # Calculate alignment scores for each microphone
    alignment_scores = []
    for mic_angle in mic_angles:
        # Calculate angular distance between mic and target
        target_diff = min(abs(mic_angle - target_rad), 2*np.pi - abs(mic_angle - target_rad))
        
        # Calculate angular distance between mic and FOV center
        fov_diff = min(abs(mic_angle - fov_center_rad), 2*np.pi - abs(mic_angle - fov_center_rad))
        
        # Combined score: prefer mics aligned with target AND within FOV
        if fov_diff <= radians(fov_half_width):
            # Within FOV - higher weight for better target alignment
            score = 1.0 - (target_diff / np.pi)  # Normalize to [0, 1]
        else:
            # Outside FOV - penalize
            score = 0.1 * (1.0 - (target_diff / np.pi))
        
        alignment_scores.append(score)
    
    # Normalize scores to sum to 1
    scores_array = np.array(alignment_scores)
    if np.sum(scores_array) > 0:
        weights = scores_array / np.sum(scores_array)
    else:
        weights = np.ones(num_channels) / num_channels
    
    return torch.tensor(weights, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# --- HYBEAM-FT-JNF DATASET ---
# --------------------------------------------------------------------------- #

class HyBeamFTJNFDataset(Dataset):
    def __init__(self, data_dir, chunk_len_sec=2.0, max_samples=100000,
                 cutoff_freq=1500, beamformer_directions=[0, 90, 180, 270]):
        # ... existing code ...
        self.data_dir = data_dir
        self.chunk_len_samples = int(chunk_len_sec * 16000)
        self.cutoff_freq = cutoff_freq
        self.beamformer_directions = beamformer_directions # e.g., [0, 90, 180, 270]
        
        # Find all sample folders
        self.sample_folders = sorted(glob.glob(os.path.join(data_dir, 'sample_*')))[:max_samples]
        
        print(f"Found {len(self.sample_folders)} samples in dataset")
        print(f"Using HyBeam hybrid approach with cutoff: {cutoff_freq}Hz")
        print(f"Beamformer directions: {beamformer_directions}")
        
    def __len__(self):
        return len(self.sample_folders)
    
    def normalize_signal(self, signal):
        # ... existing code ...
        """Normalize signal to prevent gradient issues"""
        if signal.ndim == 1:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                return signal / max_val
            return signal
        else:
            # Normalize per-channel
            normalized = np.zeros_like(signal)
            for c in range(signal.shape[0]):
                channel_max = np.max(np.abs(signal[c]))
                if channel_max > 0:
                    normalized[c] = signal[c] / channel_max
            return normalized
    
    def __getitem__(self, idx):
        # ... existing code ...
        try:
            sample_folder = self.sample_folders[idx]
            
            # Load audio files
            mixture_path = os.path.join(sample_folder, 'mixture.wav')
            target_path = os.path.join(sample_folder, 'target.wav')
            metadata_path = os.path.join(sample_folder, 'metadata.json')
            
            mixture, fs_mix = sf.read(mixture_path)
            target, fs_target = sf.read(target_path)
            
            # Ensure consistent sampling rate
            if fs_mix != 16000:
                mixture = librosa.resample(mixture.T, orig_sr=fs_mix, target_sr=16000).T
            if fs_target != 16000:
                target = librosa.resample(target, orig_sr=fs_target, target_sr=16000)
            
            # Load and parse metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            target_angle = metadata['targetAngle_deg']
            interferer_angle = metadata['interfererAngle_deg'] 
            fov_center = metadata['fov_angle_deg']
            fov_half_width = metadata['fov_width_deg'] / 2  # Convert to half-width
            
            # Ensure proper shapes
            if mixture.ndim == 1:
                mixture = mixture[np.newaxis, :]  # (1, samples)
            else:
                mixture = mixture.T  # (channels, samples)
                
            if target.ndim > 1:
                target = target[:, 0]  # Take first channel if multi-channel
            
            # Handle different lengths
            min_length = min(mixture.shape[1], len(target))
            mixture = mixture[:, :min_length]
            target = target[:min_length]
            
            # Random chunking for training variety
            if min_length > self.chunk_len_samples:
                start_idx = random.randint(0, min_length - self.chunk_len_samples)
                mixture = mixture[:, start_idx:start_idx + self.chunk_len_samples]
                target = target[start_idx:start_idx + self.chunk_len_samples]
            elif min_length < self.chunk_len_samples:
                # Pad with zeros
                pad_length = self.chunk_len_samples - min_length
                mixture = np.pad(mixture, ((0, 0), (0, pad_length)), mode='constant')
                target = np.pad(target, (0, pad_length), mode='constant')
            
            # Normalize
            mixture = self.normalize_signal(mixture)
            target = self.normalize_signal(target)
            
            # Compute STFTs
            n_fft = 512
            hop_length = 256
            
            mixture_stfts = []
            for ch in range(mixture.shape[0]):
                stft = librosa.stft(mixture[ch], n_fft=n_fft, hop_length=hop_length, window='hann')
                mixture_stfts.append(stft)
            mixture_stft = np.stack(mixture_stfts, axis=0)  # (channels, freq, time)
            
            # --- MODIFIED: Use 2-channel, 8cm array geometry to match testbed ---
            n_mics = mixture.shape[0]
            if n_mics == 2:
                # 8cm spacing, centered at 0
                mic_positions = np.array([
                    [-0.04, 0.04], # x-coordinates
                    [0.0, 0.0],    # y-coordinates
                    [0.0, 0.0]     # z-coordinates
                ])
            else:
                # Fallback for other channel counts (e.g., 4-channel)
                print(f"Warning: Using fallback circular array for {n_mics} channels.")
                radius = 0.05 # 5cm radius
                mic_angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
                mic_positions = np.array([
                    radius * np.cos(mic_angles),
                    radius * np.sin(mic_angles),
                    np.zeros(n_mics)
                ])
            # --- END MODIFICATION ---

            beamformer_outputs = compute_das_beamformer(
                mixture_stft, mic_positions, self.beamformer_directions, fs=16000, n_fft=n_fft
            )
            
            # Create hybrid input
            hybrid_stft = create_hybrid_input(
                mixture_stft, beamformer_outputs, self.cutoff_freq, fs=16000, n_fft=n_fft
            )
            
            # Calculate spatial weights based on FOV
            spatial_weights = calculate_spatial_weights(
                target_angle, fov_center, fov_half_width, n_mics
            )

            # --- MODIFIED: Convert to torch tensors (Corrected) ---
            hybrid_stft_tensor = torch.from_numpy(hybrid_stft).cfloat()
            hybrid_stft_ri = torch.stack((hybrid_stft_tensor.real, hybrid_stft_tensor.imag), dim=-1)

            target_t = torch.from_numpy(target).float()

            # Target STFT (single channel)
            target_stft = librosa.stft(target, n_fft=n_fft, hop_length=hop_length, window='hann')
            target_stft_complex = torch.from_numpy(target_stft).cfloat()

            # --- MODIFIED: Implement "Hybrid 2" ---
            # Reference for loss is the FORWARD BEAMFORMER (index 0)
            # (Assuming 0-degrees is the first direction in self.beamformer_directions)
            ref_beam_stft = beamformer_outputs[0] 
            ref_beam_stft_complex = torch.from_numpy(ref_beam_stft).cfloat()
            # --- END MODIFICATION ---

            return {
                "model_input_ri": hybrid_stft_ri.float(),
                "Y_complex": ref_beam_stft_complex, 
                "S_complex": target_stft_complex,
                "spatial_weights": spatial_weights,
                "fov_center": torch.tensor(fov_center, dtype=torch.float32),
                "fov_width": torch.tensor(fov_half_width * 2, dtype=torch.float32),
                "target_angle": torch.tensor(target_angle, dtype=torch.float32),
                "s_wav": target_t,
                "n_channels": n_mics,
                "cutoff_freq": torch.tensor(self.cutoff_freq, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error loading sample {idx} from {sample_folder}: {e}")
            # Return a different sample
            return self.__getitem__(random.randint(0, len(self)-1))

# --------------------------------------------------------------------------- #
# --- HYBEAM-FT-JNF MODEL ---
# --------------------------------------------------------------------------- #

class HyBeamFTJNF(nn.Module):
    def __init__(self, num_channels=2, num_freq_bins=257, lstm_units=128, fov_feature_dim=4,
                 use_bandwise_processing=True):
        # ... existing code ...
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
        
        # Main processing network (FT-JNF architecture)
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
        
        # Frequency-aware mask estimation
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
        # ... existing code ...
        # x shape: (B, C, F, T, 2)
        B, C, F, T, _ = x.shape
        
        # Process FOV information
        fov_features = torch.stack([
            target_angle / 360.0,  # Normalize angle
            fov_center / 360.0,    # Normalize FOV center
            fov_width / 360.0,     # Normalize FOV width  
            torch.mean(spatial_weights, dim=-1)  # Mean spatial weight
        ], dim=-1)  # (B, 4)
        
        # Encode FOV features
        encoded_fov = self.fov_encoder(fov_features)  # (B, fov_feature_dim)
        
        # Apply spatial attention
        spatial_attn = self.spatial_attention(spatial_weights)  # (B, C)
        spatial_attn = spatial_attn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        x = x * spatial_attn  # Weight channels by spatial attention
        
        # Wide-band processing
        x_wide = x.permute(0, 3, 2, 1, 4)  # (B, T, F, C, 2)
        x_wide = x_wide.reshape(B * T, F, C * 2)  # (B*T, F, C*2)
        
        # Add FOV features to each time step and frequency
        encoded_fov_expanded = encoded_fov.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, fov_feature_dim)
        encoded_fov_expanded = encoded_fov_expanded.repeat(1, T, F, 1)  # (B, T, F, fov_feature_dim)
        encoded_fov_expanded = encoded_fov_expanded.reshape(B * T, F, self.fov_feature_dim)  # (B*T, F, fov_feature_dim)
        
        # Concatenate audio features with FOV features
        x_wide_enhanced = torch.cat([x_wide, encoded_fov_expanded], dim=-1)  # (B*T, F, C*2 + fov_feature_dim)
        
        # Bandwise processing if enabled
        if self.use_bandwise_processing:
            # Estimate cutoff bin (simplified)
            cutoff_bin = int((cutoff_freq / 8000) * F)  # Assuming 16kHz, nyquist=8kHz
            
            # Process low and high frequencies separately
            low_freq_features = self.low_freq_encoder(x_wide_enhanced[:, :cutoff_bin, :])
            high_freq_features = self.high_freq_encoder(x_wide_enhanced[:, cutoff_bin:, :])
            
            # Concatenate features
            processed_features = torch.cat([low_freq_features, high_freq_features], dim=1)
        else:
            # Standard processing
            processed_features = self.freq_encoder(x_wide_enhanced)
        
        # First LSTM (wide-band)
        out_wide, _ = self.lstm1(processed_features)
        
        # Switch to narrow-band arrangement
        out_wide = out_wide.reshape(B, T, F, -1)  # (B, T, F, lstm_units*2)
        x_narrow = out_wide.permute(0, 2, 1, 3)  # (B, F, T, lstm_units*2)
        x_narrow = x_narrow.reshape(B * F, T, -1)  # (B*F, T, lstm_units*2)
        
        # Second LSTM (narrow-band)
        out_narrow, _ = self.lstm2(x_narrow)
        
        # Output processing
        out_processed = out_narrow.reshape(B, F, T, -1)  # (B, F, T, lstm_units*2)
        mask = self.mask_estimator(out_processed)  # (B, F, T, 2)
        
        return mask

# --------------------------------------------------------------------------- #
# --- HYBEAM LOSS FUNCTION (SI-SDR) ---
# --------------------------------------------------------------------------- #

def si_sdr_loss(estimate, target, epsilon=1e-8):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
    Shapes: (batch_size, num_samples)
    """
    # ... existing code ...
    # Remove DC offset
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    
    # s_target = <target, estimate> * target / ||target||^2
    s_target_energy = torch.sum(target**2, dim=-1, keepdim=True) + epsilon
    s_target = torch.sum(target * estimate, dim=-1, keepdim=True) / s_target_energy * target
    
    # e_noise = estimate - s_target
    e_noise = estimate - s_target
    
    # SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
    sdr = torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + epsilon)
    
    # Use -SDR as loss (or -log10(SDR) for stability)
    loss = -torch.log10(sdr + epsilon) * 10
    return torch.mean(loss)

# --- NEW: SI-SDR METRIC FUNCTION ---
def si_sdr_metric(estimate, target, epsilon=1e-8):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric.
    This is the positive dB value, not the loss.
    Shapes: (batch_size, num_samples)
    """
    # Remove DC offset
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    
    # s_target = <target, estimate> * target / ||target||^2
    s_target_energy = torch.sum(target**2, dim=-1, keepdim=True) + epsilon
    s_target = torch.sum(target * estimate, dim=-1, keepdim=True) / s_target_energy * target
    
    # e_noise = estimate - s_target
    e_noise = estimate - s_target
    
    # SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
    sdr = torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + epsilon)
    
    # Return the metric in dB
    sdr_db = torch.log10(sdr + epsilon) * 10
    return torch.mean(sdr_db)
# --- END NEW ---

class HyBeamLoss(nn.Module):
    def __init__(self, si_sdr_weight=1.0, freq_loss_weight=0.1, fov_weight=0.1, bandwise_weight=0.05):
        # ... existing code ...
        super().__init__()
        self.si_sdr_weight = si_sdr_weight
        self.freq_loss_weight = freq_loss_weight
        self.fov_weight = fov_weight
        self.bandwise_weight = bandwise_weight
        self.l1_loss_freq = nn.L1Loss()
        
    def forward(self, predicted_mask_ri, batch):
        # ... existing code ...
        device = predicted_mask_ri.device
        
        Y_complex = batch["Y_complex"].to(device)
        S_complex = batch["S_complex"].to(device)
        s_wav = batch["s_wav"].to(device)
        spatial_weights = batch["spatial_weights"].to(device)
        cutoff_freq = batch["cutoff_freq"].to(device)

        # Apply complex mask
        # (B, F, T, 2) -> (B, F, T)
        M_S_complex = torch.complex(predicted_mask_ri[..., 0], predicted_mask_ri[..., 1])
        # (B, F, T) * (B, F, T) -> (B, F, T)
        S_hat_complex = M_S_complex * Y_complex

        # Frequency-domain loss (Magnitude L1)
        loss_freq = self.l1_loss_freq(torch.abs(S_hat_complex), torch.abs(S_complex))

        # Time-domain SI-SDR loss
        s_hat_wav = torch.istft(S_hat_complex, n_fft=512, hop_length=256, 
                               window=torch.hann_window(512).to(device), length=s_wav.shape[-1])
        
        # Ensure s_hat_wav and s_wav are (B, T)
        if s_hat_wav.dim() == 1:
            s_hat_wav = s_hat_wav.unsqueeze(0)
        if s_wav.dim() == 1:
            s_wav = s_wav.unsqueeze(0)
            
        loss_time_sisdr = si_sdr_loss(s_hat_wav, s_wav)

        # Bandwise consistency loss
        F = S_hat_complex.shape[-2]  # Number of frequency bins
        cutoff_bin = int((cutoff_freq[0] / 8000) * F)  # Simplified cutoff bin calculation
        
        # Encourage smooth transition between low and high frequency processing
        if cutoff_bin > 0 and cutoff_bin < F - 1:
            low_freq_power = torch.mean(torch.abs(S_hat_complex[..., :cutoff_bin, :])**2)
            high_freq_power = torch.mean(torch.abs(S_hat_complex[..., cutoff_bin:, :])**2)
            bandwise_loss = torch.abs(low_freq_power - high_freq_power) / (low_freq_power + high_freq_power + 1e-8)
        else:
            bandwise_loss = torch.tensor(0.0, device=device)

        # FOV-aware spatial consistency loss
        # Encourage use of high-weight channels (low loss if spatial_weights are high)
        spatial_loss = torch.mean(1.0 - spatial_weights) 

        # --- MODIFIED: Total loss (All terms included) ---
        total_loss = (self.si_sdr_weight * loss_time_sisdr) + \
                     (self.freq_loss_weight * loss_freq) + \
                     (self.bandwise_weight * bandwise_loss) + \
                     (self.fov_weight * spatial_loss)
        
        return total_loss

# --------------------------------------------------------------------------- #
# --- NEW: METRIC EVALUATION FUNCTION ---
# --------------------------------------------------------------------------- #

def evaluate_model_and_save_metrics(model, val_dataset, device, config, epoch):
    """
    Runs a full evaluation on the validation dataset (un-chunked)
    and saves SDR, PESQ, and STOI metrics to metrics.csv
    """
    if not HAVE_METRICS_LIBS:
        print("Skipping metrics.csv generation (pystoi or pesq not installed).")
        return

    model.eval()
    results = []
    header = ['sample_id', 'sdr_in', 'sdr_out', 'pesq_in', 'pesq_out', 'stoi_in', 'stoi_out']
    
    # Get config
    n_fft = config['n_fft']
    hop_length = config['hop_length']
    fs = config['fs']
    cutoff_freq = config['cutoff_freq']
    beamformer_directions = config['beamformer_directions']
    max_channels = config['max_channels']

    # Use a specific window for istft
    hann_win = torch.hann_window(n_fft).to(device)

    print(f"Running evaluation for metrics.csv on {len(val_dataset.indices)} validation files...")
    
    with torch.no_grad():
        for idx in tqdm(val_dataset.indices, desc="Evaluating Metrics"):
            try:
                # --- 1. Load Full Files (Bypass Dataset chunking) ---
                sample_folder = val_dataset.dataset.sample_folders[idx]
                sample_id = os.path.basename(sample_folder)
                
                mixture_path = os.path.join(sample_folder, 'mixture.wav')
                target_path = os.path.join(sample_folder, 'target.wav')
                metadata_path = os.path.join(sample_folder, 'metadata.json')

                mixture_audio, fs_mix = sf.read(mixture_path)
                target_audio, fs_target = sf.read(target_path)

                # --- 2. Pre-process Full Files (like in run_inference.py) ---
                
                # Resample
                if fs_mix != fs:
                    mixture_audio = librosa.resample(mixture_audio.T, orig_sr=fs_mix, target_sr=fs).T
                if fs_target != fs:
                    target_audio = librosa.resample(target_audio, orig_sr=fs_target, target_sr=fs)
                
                # Ensure (channels, samples)
                if mixture_audio.ndim == 1:
                    mixture_audio = mixture_audio[np.newaxis, :]
                else:
                    mixture_audio = mixture_audio.T
                if target_audio.ndim > 1:
                    target_audio = target_audio[:, 0]
                
                # Align lengths
                min_len = min(mixture_audio.shape[1], target_audio.shape[0])
                mixture_audio = mixture_audio[:, :min_len]
                target_audio = target_audio[:min_len]

                # Channel padding/truncating
                n_mics = mixture_audio.shape[0]
                if n_mics > max_channels:
                    mixture_audio = mixture_audio[:max_channels, :]
                elif n_mics < max_channels:
                    padding = np.zeros((max_channels - n_mics, mixture_audio.shape[1]))
                    mixture_audio = np.concatenate([mixture_audio, padding], axis=0)
                
                # Normalize
                mixture_audio = val_dataset.dataset.normalize_signal(mixture_audio)
                target_audio = val_dataset.dataset.normalize_signal(target_audio)
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
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
                
                # Get "output" waveform
                enhanced_wav_tensor = torch.istft(
                    enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                    window=hann_win, length=target_audio.shape[0]
                )
                enhanced_wav = enhanced_wav_tensor.cpu().numpy()
                
                # Get "input" waveform (the forward beamformer)
                ref_beam_wav_tensor = torch.istft(
                    ref_stft_complex, n_fft=n_fft, hop_length=hop_length,
                    window=hann_win, length=target_audio.shape[0]
                )
                ref_beam_wav = ref_beam_wav_tensor.cpu().numpy()

                # --- 7. Calculate All Metrics ---
                # Ensure all are numpy arrays for metric libs
                target_wav_np = target_audio
                
                # Align lengths again just in case (istft padding)
                min_len_final = min(len(target_wav_np), len(ref_beam_wav), len(enhanced_wav))
                target_wav_np = target_wav_np[:min_len_final]
                ref_beam_wav = ref_beam_wav[:min_len_final]
                enhanced_wav = enhanced_wav[:min_len_final]

                # SDR
                sdr_in = si_sdr_metric(torch.tensor(ref_beam_wav), torch.tensor(target_wav_np)).item()
                sdr_out = si_sdr_metric(torch.tensor(enhanced_wav), torch.tensor(target_wav_np)).item()
                
                # PESQ (Requires 16kHz or 8kHz, 'wb' for 16k)
                pesq_in = pesq(fs, target_wav_np, ref_beam_wav, 'wb')
                pesq_out = pesq(fs, target_wav_np, enhanced_wav, 'wb')

                # STOI
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
        with open('metrics.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSuccessfully wrote metrics for epoch {epoch} to metrics.csv")
    else:
        print("\nNo metric results to write.")

# --------------------------------------------------------------------------- #
# --- HYBEAM TRAINING FUNCTION ---
# --------------------------------------------------------------------------- #

def train_hybeam_model():
    """
    Train the HyBeam-FT-JNF model
    """
    print("--- Starting HyBeam-FT-JNF Training (2-Channel, Hybrid 2, SI-SDR) ---")

    # Configuration
    data_dir = "/home/priya_intern1/sp_cup/reverb_dataset/"  # Update this path
    num_epochs = 10
    batch_size = 8
    learning_rate = 0.0001
    max_channels = 2  # --- MODIFIED: 2-channel constraint ---
    cutoff_freq = 1500  # HyBeam cutoff frequency

    # Beamformer directions (front, right, back, left)
    beamformer_directions = [0, 90, 180, 270]

    # Load dataset
    dataset = HyBeamFTJNFDataset(
        data_dir=data_dir,
        cutoff_freq=cutoff_freq,
        beamformer_directions=beamformer_directions
    )

    if len(dataset) == 0:
        print(f"Error: No data found in {data_dir}")
        return

    # Create train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    print(f"HyBeam configuration: {max_channels} channels, {cutoff_freq}Hz cutoff")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = HyBeamFTJNF(
        num_channels=max_channels,
        num_freq_bins=257,  # (512 // 2 + 1)
        use_bandwise_processing=True
    ).to(device)

    loss_fn = HyBeamLoss(
        si_sdr_weight=1.0,
        freq_loss_weight=0.1,
        fov_weight=0.1,
        bandwise_weight=0.05
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        num_train_batches = 0

        # --- MODIFIED: Added tqdm ---
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Train", ncols=100)
        for batch_idx, batch in enumerate(train_pbar):
            try:
                # ... existing code ...
                model_input = batch["model_input_ri"].to(device)
                spatial_weights = batch["spatial_weights"].to(device)
                fov_center = batch["fov_center"].to(device)
                fov_width = batch["fov_width"].to(device)
                target_angle = batch["target_angle"].to(device)
                cutoff_freq_tensor = batch["cutoff_freq"].to(device)

                # Handle variable channel numbers (e.g., pad 1-ch to 2-ch)
                B, C, F, T, _ = model_input.shape

                if C > max_channels:
                    model_input = model_input[:, :max_channels, :, :, :]
                    spatial_weights = spatial_weights[:, :max_channels]
                elif C < max_channels:
                    padding = torch.zeros(B, max_channels - C, F, T, 2).to(device)
                    model_input = torch.cat([model_input, padding], dim=1)
                    weight_padding = torch.ones(B, max_channels - C).to(device) / (max_channels - C)
                    spatial_weights = torch.cat([spatial_weights, weight_padding], dim=1)

                # Forward pass
                predicted_mask_ri = model(
                    model_input, spatial_weights, fov_center, fov_width,
                    target_angle, cutoff_freq_tensor.mean()
                )

                loss = loss_fn(predicted_mask_ri, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_epoch_loss += loss.item()
                num_train_batches += 1

                train_pbar.set_postfix(loss=loss.item())

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        num_val_batches = 0

        # --- MODIFIED: Added tqdm ---
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Val  ", ncols=100)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                try:
                    # ... existing code ...
                    model_input = batch["model_input_ri"].to(device)
                    spatial_weights = batch["spatial_weights"].to(device)
                    fov_center = batch["fov_center"].to(device)
                    fov_width = batch["fov_width"].to(device)
                    target_angle = batch["target_angle"].to(device)
                    cutoff_freq_tensor = batch["cutoff_freq"].to(device)

                    # Handle channel dimension
                    B, C, F, T, _ = model_input.shape
                    if C > max_channels:
                        model_input = model_input[:, :max_channels, :, :, :]
                        spatial_weights = spatial_weights[:, :max_channels]
                    elif C < max_channels:
                        padding = torch.zeros(B, max_channels - C, F, T, 2).to(device)
                        model_input = torch.cat([model_input, padding], dim=1)
                        weight_padding = torch.ones(B, max_channels - C).to(device) / (max_channels - C)
                        spatial_weights = torch.cat([spatial_weights, weight_padding], dim=1)

                    predicted_mask_ri = model(
                        model_input, spatial_weights, fov_center, fov_width,
                        target_angle, cutoff_freq_tensor.mean()
                    )
                    loss = loss_fn(predicted_mask_ri, batch)

                    val_epoch_loss += loss.item()
                    num_val_batches += 1
                    val_pbar.set_postfix(loss=loss.item())

                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue

        # Calculate epoch statistics
        avg_train_loss = train_epoch_loss / num_train_batches if num_train_batches > 0 else float('inf')
        avg_val_loss = val_epoch_loss / num_val_batches if num_val_batches > 0 else float('inf')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"--- Epoch {epoch + 1}/{num_epochs} Summary ---")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # --- MODIFIED: Save checkpoint ONLY (Moved evaluation out of loop) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss. Saving model to 'hybeam_ft_jnf_best.pth'")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'cutoff_freq': cutoff_freq,
                'hybeam_config': {
                    'max_channels': max_channels,
                    'beamformer_directions': beamformer_directions
                }
            }
            torch.save(checkpoint, f'hybeam_ft_jnf_best.pth')

    # Save final model
    final_model_path = 'hybeam_ft_jnf_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved as {final_model_path}")

    # --- NEW: Run metrics evaluation ONCE using the Best Model ---
    print("\n" + "=" * 50)
    print("FINAL EVALUATION: Loading best model for metrics.csv generation...")
    print("=" * 50)

    try:
        # Load the best checkpoint
        best_checkpoint = torch.load('hybeam_ft_jnf_best.pth', map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        best_epoch = best_checkpoint['epoch']

        config = {
            'n_fft': 512,
            'hop_length': 256,
            'fs': 16000,
            'cutoff_freq': cutoff_freq,
            'beamformer_directions': beamformer_directions,
            'max_channels': max_channels
        }

        evaluate_model_and_save_metrics(
            model, val_dataset, device, config, best_epoch
        )
    except Exception as e:
        print(f"Failed to run final evaluation: {e}")

    # Plot training history
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('HyBeam-FT-JNF Training History (2-Ch, Hybrid 2)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (SI-SDR + others)')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=min(val_losses) - 2, top=max(val_losses) + 2)  # Adjust y-axis
        plt.savefig('hybeam_training_history_v3.png')
        print("Training history plot saved")
    except Exception as e:
        print(f"Could not plot training history: {e}")

    return model

# --------------------------------------------------------------------------- #
# --- HYBEAM INFERENCE FUNCTION (for testing) ---
# --------------------------------------------------------------------------- #

def enhance_audio_hybeam(model, mixture_audio, target_angle, fov_center, fov_half_width, 
                        cutoff_freq=1500, device='cpu'):
    """
    Enhance audio using the trained HyBeam model
    (Modified for Hybrid 2)
    """
    # ... existing code ...
    model.eval()
    
    # Calculate spatial weights
    num_channels = mixture_audio.shape[0] # Should be 2
    spatial_weights = calculate_spatial_weights(target_angle, fov_center, fov_half_width, num_channels)
    spatial_weights = spatial_weights.unsqueeze(0).to(device)
    
    # Compute STFTs and beamformer outputs
    n_fft = 512
    hop_length = 256
    fs = 16000
    
    mixture_stfts = []
    for ch in range(num_channels):
        stft = librosa.stft(mixture_audio[ch], n_fft=n_fft, hop_length=hop_length, window='hann')
        mixture_stfts.append(stft)
    mixture_stft = np.stack(mixture_stfts, axis=0)
    
    # --- MODIFIED: Use 2-channel, 8cm array geometry ---
    if num_channels == 2:
        mic_positions = np.array([
            [-0.04, 0.04], # x-coordinates
            [0.0, 0.0],    # y-coordinates
            [0.0, 0.0]     # z-coordinates
        ])
    else:
        print(f"Warning: Using fallback circular array for {num_channels} channels.")
        radius = 0.05
        mic_angles = np.linspace(0, 2*np.pi, num_channels, endpoint=False)
        mic_positions = np.array([
            radius * np.cos(mic_angles),
            radius * np.sin(mic_angles),
            np.zeros(num_channels)
        ])
    
    beamformer_directions = [0, 90, 180, 270]
    beamformer_outputs = compute_das_beamformer(
        mixture_stft, mic_positions, beamformer_directions, fs=fs, n_fft=n_fft
    )
    
    # Create hybrid input
    hybrid_stft = create_hybrid_input(mixture_stft, beamformer_outputs, cutoff_freq, fs=fs, n_fft=n_fft)
    
    # Prepare model input
    hybrid_stft_tensor = torch.from_numpy(hybrid_stft).cfloat()
    hybrid_stft_ri = torch.stack((hybrid_stft_tensor.real, hybrid_stft_tensor.imag), dim=-1)
    hybrid_stft_ri = hybrid_stft_ri.unsqueeze(0).to(device) # (B, C, F, T, 2)
    
    # Prepare FOV parameters
    fov_center_tensor = torch.tensor([fov_center], dtype=torch.float32).to(device)
    fov_width_tensor = torch.tensor([fov_half_width * 2], dtype=torch.float32).to(device)
    target_angle_tensor = torch.tensor([target_angle], dtype=torch.float32).to(device)
    cutoff_freq_tensor = torch.tensor([cutoff_freq], dtype=torch.float32).to(device)
    
    # Apply model
    with torch.no_grad():
        predicted_mask_ri = model(
            hybrid_stft_ri, spatial_weights, fov_center_tensor, 
            fov_width_tensor, target_angle_tensor, cutoff_freq_tensor.mean()
        )
    
    # --- MODIFIED: Apply mask to FORWARD BEAM ("Hybrid 2") ---
    ref_stft = beamformer_outputs[0] # Use forward beam
    ref_stft_complex = torch.from_numpy(ref_stft).cfloat().to(device)
    
    M_S_complex = torch.complex(predicted_mask_ri[0, ..., 0], predicted_mask_ri[0, ..., 1])
    enhanced_stft = M_S_complex * ref_stft_complex
    
    # Convert back to time domain
    enhanced_audio = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length,
                               window=torch.hann_window(n_fft).to(device), 
                               length=mixture_audio.shape[1])
    
    return enhanced_audio.cpu().numpy()

if __name__ == "__main__":
    print("HyBeam-FT-JNF Training Pipeline (v3: 2-Ch, Hybrid 2, Metrics)")
    
    # Update the data_dir path to point to your 2-CHANNEL dataset
    data_dir = "/home/priya_intern1/sp_cup/reverb_dataset/"  # <---!!! IMPORTANT: Change this to your 2-CHANNEL dataset path
    
    if os.path.exists(data_dir):
        model = train_hybeam_model()
        print("HyBeam-FT-JNF training completed successfully!")
    else:
        print(f"Dataset directory '{data_dir}' not found.")
        print("Please update the data_dir path in the 'if __name__ == \"__main__\"' block.")
        print("This directory should contain your 2-channel audio samples.")