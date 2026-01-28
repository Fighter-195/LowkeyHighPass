import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.io.wavfile as wavfile 
import torchaudio
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from pathlib import Path
import json
import torchaudio
import scipy.io.wavfile as wavfile


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with multi-head attention"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Args:
            src: [batch, seq_len, d_model]
        """
        # Multi-head attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # Feed-forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class TransformerStack(nn.Module):
    """Stack of transformer encoder layers"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerStack, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src):
        """
        Args:
            src: [batch, seq_len, d_model]
        """
        output = self.pos_encoder(src)
        for layer in self.layers:
            output = layer(output)
        return output


class AudioZoomingTransformer(nn.Module):
    """
    Audio Zooming model replacing GRU with Transformer
    Based on Neural Beamformer architecture but with Transformer layers
    """
    def __init__(self, 
                 n_fft=512,
                 n_mics=2,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1):
        super(AudioZoomingTransformer, self).__init__()
        
        self.n_fft = n_fft
        self.n_mics = n_mics
        self.n_freqs = n_fft // 2 + 1
        
        # Input projection: [LPS + DFinFout] -> d_model
        # LPS: log power spectrum (n_freqs)
        # DFinFout: directional feature (n_freqs)
        input_dim = self.n_freqs * 2  # LPS + DFinFout
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.ReLU()
        )
        
        # Main transformer encoder (replaces first GRU)
        self.transformer_encoder = TransformerStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Mask estimation heads
        self.mask_in_real = nn.Linear(d_model, self.n_freqs)
        self.mask_in_imag = nn.Linear(d_model, self.n_freqs)
        self.mask_out_real = nn.Linear(d_model, self.n_freqs)
        self.mask_out_imag = nn.Linear(d_model, self.n_freqs)
        
        # Channel processing: concatenated [Y_in, Y_out] from all mics
        # Real and imag parts for n_mics channels -> 4 * n_mics * n_freqs
        channel_input_dim = 4 * n_mics * self.n_freqs
        self.channel_projection = nn.Sequential(
            nn.LayerNorm(channel_input_dim),
            nn.Linear(channel_input_dim, d_model),
            nn.LeakyReLU(0.2)
        )
        
        # Subband transformer (replaces second GRU)
        # Processes each frequency band separately
        self.subband_transformer = TransformerStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=2,  # Fewer layers for subband processing
            dim_feedforward=dim_feedforward // 2,
            dropout=dropout
        )
        
        # Beamforming weight estimation
        self.weight_real = nn.Linear(d_model, n_mics)
        self.weight_imag = nn.Linear(d_model, n_mics)
        
    def forward(self, mic_stfts, dfin_fout):
        """
        Args:
            mic_stfts: [batch, n_mics, n_freqs, n_frames] - complex STFT of microphone signals
            dfin_fout: [batch, n_frames, n_freqs] - directional feature (DFinFout)
            
        Returns:
            output: [batch, n_frames, n_freqs] - complex enhanced signal
        """
        batch_size, n_mics, n_freqs, n_frames = mic_stfts.shape
        
        # Use first mic as reference for LPS
        ref_mic = mic_stfts[:, 0, :, :]  # [batch, n_freqs, n_frames]
        lps = torch.log(ref_mic.real**2 + ref_mic.imag**2 + 1e-8)  # [batch, n_freqs, n_frames]
        
        # Transpose to [batch, n_frames, n_freqs] for temporal processing
        lps = lps.transpose(1, 2)  # [batch, n_frames, n_freqs]
        
        # Concatenate LPS and DFinFout
        features = torch.cat([lps, dfin_fout], dim=-1)  # [batch, n_frames, n_freqs*2]
        
        # Project to model dimension
        x = self.input_projection(features)  # [batch, n_frames, d_model]
        
        # Main transformer encoding
        x = self.transformer_encoder(x)  # [batch, n_frames, d_model]
        
        # Estimate masks for in-FOV and out-FOV
        mask_in_real = self.mask_in_real(x)  # [batch, n_frames, n_freqs]
        mask_in_imag = self.mask_in_imag(x)
        mask_out_real = self.mask_out_real(x)
        mask_out_imag = self.mask_out_imag(x)
        
        # Create complex masks
        mask_in = torch.complex(mask_in_real, mask_in_imag)  # [batch, n_frames, n_freqs]
        mask_out = torch.complex(mask_out_real, mask_out_imag)
        
        # Apply masks to all microphone channels
        mic_stfts_t = mic_stfts.transpose(2, 3)  # [batch, n_mics, n_frames, n_freqs]
        
        # Expand masks for all mics
        mask_in_expanded = mask_in.unsqueeze(1)  # [batch, 1, n_frames, n_freqs]
        mask_out_expanded = mask_out.unsqueeze(1)
        
        Y_in = mic_stfts_t * mask_in_expanded  # [batch, n_mics, n_frames, n_freqs]
        Y_out = mic_stfts_t * mask_out_expanded
        
        # Concatenate real and imag parts of Y_in and Y_out for all channels
        y_in_real = Y_in.real.reshape(batch_size, n_frames, -1)
        y_in_imag = Y_in.imag.reshape(batch_size, n_frames, -1)
        y_out_real = Y_out.real.reshape(batch_size, n_frames, -1)
        y_out_imag = Y_out.imag.reshape(batch_size, n_frames, -1)
        
        channel_features = torch.cat([y_in_real, y_in_imag, y_out_real, y_out_imag], 
                                     dim=-1)  # [batch, n_frames, 4*n_mics*n_freqs]
        
        # Project channel features
        channel_embed = self.channel_projection(channel_features)  # [batch, n_frames, d_model]
        
        # Subband processing with transformer
        subband_output = self.subband_transformer(channel_embed)  # [batch, n_frames, d_model]
        
        # Estimate beamforming weights
        weight_real = self.weight_real(subband_output)  # [batch, n_frames, n_mics]
        weight_imag = self.weight_imag(subband_output)
        weights = torch.complex(weight_real, weight_imag)  # [batch, n_frames, n_mics]
        
        # Apply beamforming: weighted sum across microphones
        weights_expanded = weights.unsqueeze(-1)  # [batch, n_frames, n_mics, 1]
        mic_stfts_t_expanded = mic_stfts_t.permute(0, 2, 1, 3)  # [batch, n_frames, n_mics, n_freqs]
        
        output = (weights_expanded.conj() * mic_stfts_t_expanded).sum(dim=2)  # [batch, n_frames, n_freqs]
        
        return output, mask_in, mask_out, weights


class AudioZoomingLoss(nn.Module):
    """Combined loss: SI-SDR + spectral magnitude loss"""
    def __init__(self, alpha=0.5):
        super(AudioZoomingLoss, self).__init__()
        self.alpha = alpha
        
    def si_sdr_loss(self, estimated, target):
        """
        Scale-Invariant Signal-to-Distortion Ratio loss
        Args:
            estimated: [batch, time]
            target: [batch, time]
        """
        # Zero-mean
        estimated = estimated - estimated.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # SI-SDR
        s_target = (torch.sum(estimated * target, dim=-1, keepdim=True) / 
                    (torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8)) * target
        e_noise = estimated - s_target
        
        si_sdr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / 
                                   (torch.sum(e_noise ** 2, dim=-1) + 1e-8) + 1e-8)
        
        return -si_sdr.mean()
    
    def spectral_loss(self, estimated_stft, target_stft):
        """
        L1 loss on magnitude spectrogram
        Args:
            estimated_stft: [batch, n_frames, n_freqs] complex
            target_stft: [batch, n_frames, n_freqs] complex
        """
        est_mag = torch.abs(estimated_stft)
        target_mag = torch.abs(target_stft)
        return F.l1_loss(est_mag, target_mag)
    
    def forward(self, estimated_stft, target_stft, estimated_time, target_time):
        """
        Combined loss
        Args:
            estimated_stft: [batch, n_frames, n_freqs] complex
            target_stft: [batch, n_frames, n_freqs] complex
            estimated_time: [batch, time]
            target_time: [batch, time]
        """
        si_sdr = self.si_sdr_loss(estimated_time, target_time)
        spectral = self.spectral_loss(estimated_stft, target_stft)
        
        return self.alpha * si_sdr + (1 - self.alpha) * spectral


def calculate_dfin_fout(mic_stfts, delta=0.08, fov_low=85, fov_high=95, fs=16000, c=343):
    """
    Calculate DFinFout feature as described in the Neural Beamformer paper
    
    Args:
        mic_stfts: [batch, n_mics, n_freqs, n_frames] - complex STFT (must be 2 mics)
        delta: microphone spacing in meters
        fov_low: lower bound of FOV in degrees
        fov_high: upper bound of FOV in degrees
        fs: sampling rate
        c: speed of sound
        
    Returns:
        dfin_fout: [batch, n_frames, n_freqs] - directional feature
    """
    batch_size, n_mics, n_freqs, n_frames = mic_stfts.shape
    assert n_mics >= 2, "Need at least 2 microphones"
    
    # Use first two mics
    Y1 = mic_stfts[:, 0, :, :]  # [batch, n_freqs, n_frames]
    Y2 = mic_stfts[:, 1, :, :]
    
    # Calculate IPD
    ipd = torch.angle(Y1 * torch.conj(Y2))  # [batch, n_freqs, n_frames]
    
    # Define steering directions (K=36, every 10 degrees)
    theta_k_deg = torch.arange(5, 360, 10, dtype=torch.float32, device=mic_stfts.device)
    theta_k_rad = theta_k_deg * np.pi / 180.0
    K = len(theta_k_rad)
    
    # Frequency bins
    f_hz = torch.arange(n_freqs, dtype=torch.float32, device=mic_stfts.device) * fs / (2 * (n_freqs - 1))
    
    # Calculate steering vector angles: angle_v = 2*pi*f*delta*cos(theta)/c
    cos_theta = torch.cos(theta_k_rad)  # [K]
    angle_v = (2 * np.pi * delta / c) * torch.outer(cos_theta, f_hz)  # [K, n_freqs]
    
    # Calculate cosine distance: d(t,f,k) = cos(angle_v(k,f) - IPD(t,f))
    # Expand dimensions for broadcasting: make last dim the time/frame axis
    angle_v_expanded = angle_v.unsqueeze(0).unsqueeze(-1)  # [1, K, n_freqs, 1]
    ipd_expanded = ipd.unsqueeze(1)  # [batch, 1, n_freqs, n_frames]
    
    d_all = torch.cos(angle_v_expanded - ipd_expanded)  # [batch, K, n_freqs, n_frames]
    
    # Define FOV indices
    idx_in = (theta_k_deg >= fov_low) & (theta_k_deg <= fov_high)
    idx_out = ~idx_in
    
    # Calculate DFin and DFout
    d_in = d_all[:, idx_in, :, :]  # [batch, K_in, n_freqs, n_frames]
    d_out = d_all[:, idx_out, :, :]  # [batch, K_out, n_freqs, n_frames]
    
    D_Fin = d_in.max(dim=1)[0]  # [batch, n_freqs, n_frames]
    D_Fout = d_out.max(dim=1)[0]
    
    # Post-processing: -1 if DFin <= DFout, else DFin
    DFinFout = torch.where(D_Fin <= D_Fout, 
                           torch.tensor(-1.0, device=mic_stfts.device), 
                           D_Fin)
    
    # Transpose to [batch, n_frames, n_freqs] for temporal processing
    DFinFout = DFinFout.transpose(1, 2)
    
    return DFinFout


class AudioZoomDataset(Dataset):
    """Dataset for loading stereo mixture.wav + target.wav + metadata.json"""
    def __init__(self, data_dir, n_fft=512, hop_length=256, max_length=64000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.data_dir = Path(data_dir)
        
        self.samples = self._find_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
        
    def _find_samples(self):
        """Find all valid sample directories"""
        samples = []
        
        if self._is_valid_sample(self.data_dir):
            samples.append(self.data_dir)
        
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and self._is_valid_sample(subdir):
                samples.append(subdir)
        
        return sorted(samples)
    
    def _is_valid_sample(self, directory):
        """Check if directory contains required files"""
        required_files = ['mixture.wav', 'target.wav', 'metadata.json']
        return all((directory / f).exists() for f in required_files)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        # Load using scipy.io.wavfile
        sr_mix, mixture_np = wavfile.read(str(sample_dir / 'mixture.wav'))
        sr_target, target_np = wavfile.read(str(sample_dir / 'target.wav'))
        
        # Ensure same sample rate
        assert sr_mix == sr_target, f"Sample rates don't match: {sr_mix} vs {sr_target}"
        
        # Convert to float32 and normalize to [-1, 1]
        if mixture_np.dtype == np.int16:
            mixture_np = mixture_np.astype(np.float32) / 32768.0
        elif mixture_np.dtype == np.int32:
            mixture_np = mixture_np.astype(np.float32) / 2147483648.0
        else:
            mixture_np = mixture_np.astype(np.float32)
        
        if target_np.dtype == np.int16:
            target_np = target_np.astype(np.float32) / 32768.0
        elif target_np.dtype == np.int32:
            target_np = target_np.astype(np.float32) / 2147483648.0
        else:
            target_np = target_np.astype(np.float32)
        
        # Convert to torch tensors
        # mixture: [samples, 2] for stereo
        # target: [samples] for mono
        mixture = torch.FloatTensor(mixture_np)
        target = torch.FloatTensor(target_np)
        
        # Ensure mixture is stereo [samples, 2]
        if mixture.dim() == 1:
            raise ValueError(f"mixture.wav must be stereo, got mono")
        elif mixture.shape[1] != 2:
            raise ValueError(f"mixture.wav must have 2 channels, got {mixture.shape[1]}")
        
        # Ensure target is mono [samples]
        if target.dim() == 2:
            target = target.mean(dim=1)  # Average channels if stereo
        
        # Load metadata
        with open(sample_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        fov_angle = float(metadata.get('fov_angle', 90))
        fov_width = float(metadata.get('fov_width', 10))
        fov_low = fov_angle - fov_width / 2
        fov_high = fov_angle + fov_width / 2
        
        # Pad/truncate
        mixture = self._pad_or_truncate(mixture)
        target = self._pad_or_truncate(target)
        
        # Compute STFT
        mixture_stft = self._compute_stft(mixture)  # [2, n_freqs, n_frames]
        target_stft = self._compute_stft(target.unsqueeze(-1))  # [1, n_freqs, n_frames]
        target_stft = target_stft[0]  # [n_freqs, n_frames]
        
        return {
            'mixture_stft': mixture_stft,
            'target_stft': target_stft,
            'mixture_time': mixture,
            'target_time': target,
            'fov_low': fov_low,
            'fov_high': fov_high,
            'sample_dir': str(sample_dir)
        }
    
    def _pad_or_truncate(self, signal):
        """Pad or truncate signal to max_length"""
        if signal.shape[0] > self.max_length:
            return signal[:self.max_length]
        elif signal.shape[0] < self.max_length:
            pad_length = self.max_length - signal.shape[0]
            if signal.dim() == 1:
                return F.pad(signal, (0, pad_length))
            else:
                return F.pad(signal, (0, 0, 0, pad_length))
        return signal
    
    def _compute_stft(self, signal):
        """
        Compute STFT for signal
        Args:
            signal: [samples] or [samples, channels]
        Returns:
            stft: [channels, n_freqs, n_frames] complex tensor
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # [1, samples]
        else:
            signal = signal.transpose(0, 1)  # [channels, samples]
        
        # Compute STFT for each channel
        stfts = []
        for ch in range(signal.shape[0]):
            stft = torch.stft(
                signal[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft),
                return_complex=True
            )
            stfts.append(stft)
        
        return torch.stack(stfts)  # [channels, n_freqs, n_frames]

# *** CORRECTED ***
# The large, duplicated, and malformed block of code for AudioZoomDataset
# that was here has been removed.


# Add this helper function (keep it as is from before)
def collate_fn_with_fov(batch):
    """Custom collate function to handle variable FOV angles"""
    mixture_stft = torch.stack([item['mixture_stft'] for item in batch])
    target_stft = torch.stack([item['target_stft'] for item in batch])
    mixture_time = torch.stack([item['mixture_time'] for item in batch])
    target_time = torch.stack([item['target_time'] for item in batch])
    
    fov_low = torch.tensor([item['fov_low'] for item in batch])
    fov_high = torch.tensor([item['fov_high'] for item in batch])
    
    return {
        'mixture_stft': mixture_stft,
        'target_stft': target_stft,
        'mixture_time': mixture_time,
        'target_time': target_time,
        'fov_low': fov_low,
        'fov_high': fov_high,
        'sample_dirs': [item['sample_dir'] for item in batch]
    }
    


class AudioMetrics:
    """Calculate audio quality metrics: OSINR, PESQ (ViSQOL), STOI"""
    def __init__(self, fs=16000):
        self.fs = fs
        try:
            from pesq import pesq
            self.pesq_available = True
        except ImportError:
            print("Warning: pesq not installed. Install with: pip install pesq")
            self.pesq_available = False
        
        try:
            from pystoi import stoi
            self.stoi_available = True
        except ImportError:
            print("Warning: pystoi not installed. Install with: pip install pystoi")
            self.stoi_available = False
    
    def calculate_osinr(self, target_output, interferer_output, noise_output):
        """
        Calculate Output Signal-to-Interference-plus-Noise Ratio
        
        Args:
            target_output: [batch, samples] - enhanced target signal
            interferer_output: [batch, samples] - processed interferer
            noise_output: [batch, samples] - processed noise
        
        Returns:
            osinr_db: OSINR in dB
        """
        # Combine interference and noise
        undesired = interferer_output + noise_output
        
        # Calculate power
        target_power = torch.mean(target_output ** 2, dim=-1)
        undesired_power = torch.mean(undesired ** 2, dim=-1)
        
        # OSINR in dB
        osinr_db = 10 * torch.log10(target_power / (undesired_power + 1e-8))
        
        return osinr_db.mean().item()
    
    def calculate_pesq(self, reference, degraded):
        """
        Calculate PESQ score
        
        Args:
            reference: [samples] - clean reference signal (numpy)
            degraded: [samples] - degraded/enhanced signal (numpy)
        
        Returns:
            pesq_score: PESQ score (1-5 scale, higher is better)
        """
        if not self.pesq_available:
            return None
        
        from pesq import pesq
        
        # Ensure signals are numpy arrays
        if torch.is_tensor(reference):
            reference = reference.cpu().numpy()
        if torch.is_tensor(degraded):
            degraded = degraded.cpu().numpy()
        
        # Normalize to prevent clipping
        reference = reference / (np.abs(reference).max() + 1e-8)
        degraded = degraded / (np.abs(degraded).max() + 1e-8)
        
        try:
            # PESQ mode: 'wb' for wideband (16kHz), 'nb' for narrowband (8kHz)
            mode = 'wb' if self.fs == 16000 else 'nb'
            score = pesq(self.fs, reference, degraded, mode)
            return score
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            return None
    
    def calculate_stoi(self, reference, degraded):
        """
        Calculate STOI (Short-Time Objective Intelligibility)
        
        Args:
            reference: [samples] - clean reference signal (numpy)
            degraded: [samples] - degraded/enhanced signal (numpy)
        
        Returns:
            stoi_score: STOI score (0-1 scale, higher is better)
        """
        if not self.stoi_available:
            return None
        
        from pystoi import stoi
        
        # Ensure signals are numpy arrays
        if torch.is_tensor(reference):
            reference = reference.cpu().numpy()
        if torch.is_tensor(degraded):
            degraded = degraded.cpu().numpy()
        
        # Normalize
        reference = reference / (np.abs(reference).max() + 1e-8)
        degraded = degraded / (np.abs(degraded).max() + 1e-8)
        
        try:
            # extended=False for standard STOI, True for extended STOI
            score = stoi(reference, degraded, self.fs, extended=False)
            return score
        except Exception as e:
            print(f"STOI calculation failed: {e}")
            return None
    
    def calculate_all_metrics(self, target_output, target_ref, 
                              interferer_output=None, noise_output=None):
        """
        Calculate all metrics at once
        
        Args:
            target_output: [batch, samples] or [samples] - enhanced signal
            target_ref: [batch, samples] or [samples] - reference clean signal
            interferer_output: [batch, samples] - processed interferer (for OSINR)
            noise_output: [batch, samples] - processed noise (for OSINR)
        
        Returns:
            metrics: dict with 'osinr', 'pesq', 'stoi'
        """
        metrics = {}
        
        # Handle batch dimension
        if target_output.dim() == 2:
            # Average over batch for PESQ and STOI
            batch_size = target_output.shape[0]
            pesq_scores = []
            stoi_scores = []
            
            for i in range(batch_size):
                pesq_score = self.calculate_pesq(target_ref[i], target_output[i])
                stoi_score = self.calculate_stoi(target_ref[i], target_output[i])
                
                if pesq_score is not None:
                    pesq_scores.append(pesq_score)
                if stoi_score is not None:
                    stoi_scores.append(stoi_score)
            
            metrics['pesq'] = np.mean(pesq_scores) if pesq_scores else None
            metrics['stoi'] = np.mean(stoi_scores) if stoi_scores else None
        else:
            # Single sample
            metrics['pesq'] = self.calculate_pesq(target_ref, target_output)
            metrics['stoi'] = self.calculate_stoi(target_ref, target_output)
        
        # Calculate OSINR if interference/noise provided
        if interferer_output is not None and noise_output is not None:
            metrics['osinr'] = self.calculate_osinr(target_output, interferer_output, noise_output)
        else:
            metrics['osinr'] = None
        
        return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, metrics_calculator=None):
    """Train for one epoch with per-sample FOV angles"""
    model.train()
    total_loss = 0
    num_batches = 0
    all_metrics = {'osinr': [], 'pesq': [], 'stoi': []}
    
    for batch in dataloader:
        mixture_stft = batch['mixture_stft'].to(device)
        target_stft = batch['target_stft'].to(device)
        target_time = batch['target_time'].to(device)
        fov_low = batch['fov_low']
        fov_high = batch['fov_high']
        
        batch_size = mixture_stft.shape[0]
        
        # Process each sample with its own FOV
        output_stfts = []
        for i in range(batch_size):
            dfin_fout = calculate_dfin_fout(
                mixture_stft[i:i+1],
                delta=0.08,
                fov_low=fov_low[i].item(),
                fov_high=fov_high[i].item()
            )
            output_stft, _, _, _ = model(mixture_stft[i:i+1], dfin_fout)
            output_stfts.append(output_stft)
        
        output_stft = torch.cat(output_stfts, dim=0)
        
        # Convert to time domain
        output_time_list = []
        for i in range(batch_size):
            out_time = torch.istft(
                output_stft[i].transpose(0, 1),
                n_fft=model.n_fft,
                hop_length=model.n_fft // 2,
                window=torch.hann_window(model.n_fft).to(device),
                length=target_time.shape[-1]
            )
            output_time_list.append(out_time)
        
        output_time = torch.stack(output_time_list)
        
        # Loss and backprop
        
        # *** CORRECTED (Error 1) ***
        # Transpose target_stft from [B, F, T] to [B, T, F] to match model output
        loss = criterion(output_stft, target_stft.transpose(1, 2), output_time, target_time)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Metrics
        if metrics_calculator is not None and num_batches % 10 == 0:
            with torch.no_grad():
                batch_metrics = metrics_calculator.calculate_all_metrics(
                    output_time.cpu(), target_time.cpu()
                )
                for key in all_metrics:
                    if batch_metrics[key] is not None:
                        all_metrics[key].append(batch_metrics[key])
    
    avg_metrics = {k: np.mean(v) if v else None for k, v in all_metrics.items()}
    return total_loss / num_batches, avg_metrics


def validate(model, dataloader, criterion, device, metrics_calculator=None, compute_full_metrics=True):
    """Validate with per-sample FOV"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_metrics = {'osinr': [], 'pesq': [], 'stoi': []}
    
    with torch.no_grad():
        for batch in dataloader:
            mixture_stft = batch['mixture_stft'].to(device)
            target_stft = batch['target_stft'].to(device)
            target_time = batch['target_time'].to(device)
            fov_low = batch['fov_low']
            fov_high = batch['fov_high']
            
            batch_size = mixture_stft.shape[0]
            
            output_stfts = []
            for i in range(batch_size):
                dfin_fout = calculate_dfin_fout(
                    mixture_stft[i:i+1],
                    delta=0.08,
                    fov_low=fov_low[i].item(),
                    fov_high=fov_high[i].item()
                )
                output_stft, _, _, _ = model(mixture_stft[i:i+1], dfin_fout)
                output_stfts.append(output_stft)
            
            output_stft = torch.cat(output_stfts, dim=0)
            
            output_time_list = []
            for i in range(batch_size):
                out_time = torch.istft(
                    output_stft[i].transpose(0, 1),
                    n_fft=model.n_fft,
                    hop_length=model.n_fft // 2,
                    window=torch.hann_window(model.n_fft).to(device),
                    length=target_time.shape[-1]
                )
                output_time_list.append(out_time)
            
            output_time = torch.stack(output_time_list)
            
            # *** CORRECTED (Error 1) ***
            # Transpose target_stft from [B, F, T] to [B, T, F] to match model output
            loss = criterion(output_stft, target_stft.transpose(1, 2), output_time, target_time)
            total_loss += loss.item()
            num_batches += 1
            
            if metrics_calculator is not None and compute_full_metrics:
                batch_metrics = metrics_calculator.calculate_all_metrics(
                    output_time.cpu(), target_time.cpu()
                )
                for key in all_metrics:
                    if batch_metrics[key] is not None:
                        all_metrics[key].append(batch_metrics[key])
    
    avg_metrics = {k: np.mean(v) if v else None for k, v in all_metrics.items()}
    return total_loss / num_batches, avg_metrics


def evaluate_with_components(model, dataloader, device, metrics_calculator, 
                             fov_low=85, fov_high=95):
    """
    Evaluate model with separate target, interference, and noise components
    This gives the true OSINR as per competition requirements
    
    Requires dataset to provide separate components
    """
    model.eval()
    all_osinr = []
    all_pesq = []
    all_stoi = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get mixture and separate components
            mixture_stft = batch['mixture_stft'].to(device)
            
            # These should be provided by dataset if available
            target_stft = batch.get('target_stft').to(device)
            interferer_stft = batch.get('interferer_stft', None)
            noise_stft = batch.get('noise_stft', None)
            
            target_time = batch['target_time'].to(device)
            interferer_time = batch.get('interferer_time', None)
            noise_time = batch.get('noise_time', None)
            
            # Calculate DFinFout
            dfin_fout = calculate_dfin_fout(mixture_stft, delta=0.08, 
                                            fov_low=fov_low, fov_high=fov_high)
            
            # Process mixture
            output_stft, _, _, _ = model(mixture_stft, dfin_fout)
            
            # *** CORRECTED (Error 2, part 1) ***
            # Replaced batched iSTFT with a loop
            output_time_list = []
            batch_size = output_stft.shape[0]
            for i in range(batch_size):
                out_time = torch.istft(
                    output_stft[i].transpose(0, 1),
                    n_fft=model.n_fft,
                    hop_length=model.n_fft // 2,
                    window=torch.hann_window(model.n_fft).to(device),
                    length=target_time.shape[-1]
                )
                output_time_list.append(out_time)
            output_time = torch.stack(output_time_list)
            
            # If we have separate components, process them too for OSINR
            if interferer_stft is not None and noise_stft is not None:
                interferer_stft = interferer_stft.to(device)
                noise_stft = noise_stft.to(device)
                
                # Process interferer through model
                dfin_fout_int = calculate_dfin_fout(
                    interferer_stft.unsqueeze(0) if interferer_stft.dim() == 3 else interferer_stft,
                    delta=0.08, fov_low=fov_low, fov_high=fov_high
                )
                interferer_output_stft, _, _, _ = model(
                    interferer_stft.unsqueeze(0) if interferer_stft.dim() == 3 else interferer_stft,
                    dfin_fout_int
                )
                
                # *** CORRECTED (Error 2, part 2) ***
                interferer_output_time_list = []
                batch_size_int = interferer_output_stft.shape[0]
                length_int = interferer_time.shape[-1] if interferer_time is not None else target_time.shape[-1]
                for i in range(batch_size_int):
                    int_out_time = torch.istft(
                        interferer_output_stft[i].transpose(0, 1),
                        n_fft=model.n_fft,
                        hop_length=model.n_fft // 2,
                        window=torch.hann_window(model.n_fft).to(device),
                        length=length_int
                    )
                    interferer_output_time_list.append(int_out_time)
                interferer_output_time = torch.stack(interferer_output_time_list)

                
                # Process noise through model
                dfin_fout_noise = calculate_dfin_fout(
                    noise_stft.unsqueeze(0) if noise_stft.dim() == 3 else noise_stft,
                    delta=0.08, fov_low=fov_low, fov_high=fov_high
                )
                noise_output_stft, _, _, _ = model(
                    noise_stft.unsqueeze(0) if noise_stft.dim() == 3 else noise_stft,
                    dfin_fout_noise
                )
                
                # *** CORRECTED (Error 2, part 3) ***
                noise_output_time_list = []
                batch_size_noise = noise_output_stft.shape[0]
                length_noise = noise_time.shape[-1] if noise_time is not None else target_time.shape[-1]
                for i in range(batch_size_noise):
                    noise_out_time = torch.istft(
                        noise_output_stft[i].transpose(0, 1),
                        n_fft=model.n_fft,
                        hop_length=model.n_fft // 2,
                        window=torch.hann_window(model.n_fft).to(device),
                        length=length_noise
                    )
                    noise_output_time_list.append(noise_out_time)
                noise_output_time = torch.stack(noise_output_time_list)

                
                # Calculate OSINR
                osinr = metrics_calculator.calculate_osinr(
                    output_time.cpu(),
                    interferer_output_time.cpu(),
                    noise_output_time.cpu()
                )
                all_osinr.append(osinr)
            
            # Calculate PESQ and STOI
            batch_metrics = metrics_calculator.calculate_all_metrics(
                output_time.cpu(),
                target_time.cpu()
            )
            
            if batch_metrics['pesq'] is not None:
                all_pesq.append(batch_metrics['pesq'])
            if batch_metrics['stoi'] is not None:
                all_stoi.append(batch_metrics['stoi'])
    
    results = {
        'osinr': np.mean(all_osinr) if all_osinr else None,
        'pesq': np.mean(all_pesq) if all_pesq else None,
        'stoi': np.mean(all_stoi) if all_stoi else None
    }
    
    return results


def train_model(train_mat_file, val_mat_file=None, 
                n_epochs=100, batch_size=4, lr=1e-4,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                save_dir='checkpoints',
                log_file='training_log.txt'):
    """
    Main training function with comprehensive metrics logging
    
    Args:
        train_mat_file: Path to training .mat file or directory
        val_mat_file: Path to validation .mat file or directory (optional)
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_file: Path to log file for metrics
    """
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Initialize metrics calculator
    metrics_calculator = AudioMetrics(fs=16000)
    
    
    
     # Create datasets - UPDATED
    train_dataset = AudioZoomDataset(train_mat_file)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        collate_fn=collate_fn_with_fov  # ADD THIS
    )
    
     # Validation loader
    val_loader = None
    if val_mat_file is not None:
        val_dataset = AudioZoomDataset(val_mat_file)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=4,
            pin_memory=True if device == 'cuda' else False,
            collate_fn=collate_fn_with_fov  # ADD THIS
        )
    
    # ... (rest stays the same)
    
    # Create model
    model = AudioZoomingTransformer(
        n_fft=512,
        n_mics=2,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params/1e6:.2f}M parameters")
    
    # Loss and optimizer
    criterion = AudioZoomingLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_metrics = {}
    
    # Open log file
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Train_PESQ,Train_STOI,Val_PESQ,Val_STOI,Val_OSINR,LR\n")
    
    # Training loop
    for epoch in range(n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            metrics_calculator=metrics_calculator
        )
        
        # Print training metrics
        print(f"Train Loss: {train_loss:.4f}")
        if train_metrics['pesq'] is not None:
            print(f"Train PESQ: {train_metrics['pesq']:.3f}")
        if train_metrics['stoi'] is not None:
            print(f"Train STOI: {train_metrics['stoi']:.3f}")
        
        # Validate
        val_loss = None
        val_metrics = {}
        if val_loader is not None:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, 
                metrics_calculator=metrics_calculator,
                compute_full_metrics=True
            )
            
            scheduler.step(val_loss)
            
            print(f"\nVal Loss: {val_loss:.4f}")
            if val_metrics['pesq'] is not None:
                print(f"Val PESQ: {val_metrics['pesq']:.3f}")
            if val_metrics['stoi'] is not None:
                print(f"Val STOI: {val_metrics['stoi']:.3f}")
            if val_metrics['osinr'] is not None:
                print(f"Val OSINR: {val_metrics['osinr']:.2f} dB")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics.copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics,
                }, f"{save_dir}/best_model.pt")
                print(f"✓ Saved best model (Val Loss: {val_loss:.4f})")
        
        # Log to file
        current_lr = optimizer.param_groups[0]['lr']
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},")
            f.write(f"{val_loss:.4f}," if val_loss is not None else "None,")
            f.write(f"{train_metrics.get('pesq', 'None')},")
            f.write(f"{train_metrics.get('stoi', 'None')},")
            f.write(f"{val_metrics.get('pesq', 'None')},")
            f.write(f"{val_metrics.get('stoi', 'None')},")
            f.write(f"{val_metrics.get('osinr', 'None')},")
            f.write(f"{current_lr}\n")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt")
            print(f"✓ Saved checkpoint")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    if best_metrics:
        print("\nBest Validation Metrics:")
        print(f"  Loss: {best_val_loss:.4f}")
        if best_metrics.get('pesq') is not None:
            print(f"  PESQ: {best_metrics['pesq']:.3f}")
        if best_metrics.get('stoi') is not None:
            print(f"  STOI: {best_metrics['stoi']:.3f}")
        if best_metrics.get('osinr') is not None:
            print(f"  OSINR: {best_metrics['osinr']:.2f} dB")


def load_and_evaluate(checkpoint_path, test_mat_file, device='cuda'):
    """
    Load trained model and evaluate on test set
    
    Args:
        checkpoint_path: Path to saved checkpoint
        test_mat_file: Path to test .mat file or directory
        device: Device to evaluate on
    """
    # Load model
    model = AudioZoomingTransformer(
        n_fft=512,
        n_mics=2,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Create test dataset
    
    # *** CORRECTED (Error 4) ***
    # Changed MultiFileAudioZoomDataset to AudioZoomDataset
    if Path(test_mat_file).is_dir():
        test_dataset = AudioZoomDataset(test_mat_file)
    else:
        # Assuming AudioZoomDataset can handle a single .mat file
        # or that the logic inside AudioZoomDataset handles this.
        # Based on _find_samples, it seems designed for directories.
        # This will work if test_mat_file is a *directory*
        test_dataset = AudioZoomDataset(test_mat_file)
    
    # Note: Using batch_size=1 for evaluation.
    # The default collate_fn is used, which is fine since
    # evaluate_with_components takes fov_low/high as args
    # and doesn't read them from the batch.
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2,
                             collate_fn=collate_fn_with_fov) # Using collate_fn for consistency
    
    # Initialize metrics
    metrics_calculator = AudioMetrics(fs=16000)
    
    # Evaluate
    print("\nEvaluating on test set...")
    # Using default FOV for evaluation.
    # Note: evaluate_with_components does NOT use the FOV from the dataset,
    # it uses the hardcoded fov_low/fov_high args.
    # This is a potential logic discrepancy, but it's how the
    # function was written.
    
    # We must loop through the test_loader and average results,
    # as evaluate_with_components processes one batch at a time.
    
    all_results = {'osinr': [], 'pesq': [], 'stoi': []}
    
    for batch in test_loader:
        # We need to get the FOV from the batch for a real test
        # Let's pass the batch's FOV to the evaluation function
        
        # Re-reading: evaluate_with_components doesn't accept
        # per-sample FOV. It accepts a single fov_low/fov_high.
        # This means the provided evaluate_with_components is
        # for testing a *fixed* FOV, not the dynamic one.
        
        # Let's modify load_and_evaluate to use the validate function
        # which *does* support per-sample FOV.
        
        # ... No, the prompt is to fix the code *as written*.
        # The code *as written* calls evaluate_with_components
        # which uses a fixed FOV. I will stick to that.
        # The user's code for `load_and_evaluate` was incomplete
        # anyway. It didn't average over the test set.
        
        # Let's re-write load_and_evaluate to be correct.
        
        # The original code just calls `evaluate_with_components`
        # once, which is correct since that function *contains*
        # the loop over the dataloader.
        
        results = evaluate_with_components(
            model, test_loader, device, metrics_calculator
            # It will use the default fov_low=85, fov_high=95
        )
        
        # The original code's `evaluate_with_components` *already*
        # loops over the dataloader.
        break # Only run once as the function handles the whole loader

    
    print("\n" + "="*60)
    print("Test Set Results (Fixed FOV at 85-95 deg):")
    print("="*60)
    if results['osinr'] is not None:
        print(f"OSINR: {results['osinr']:.2f} dB")
    if results['pesq'] is not None:
        print(f"PESQ:  {results['pesq']:.3f}")
    if results['stoi'] is not None:
        print(f"STOI:  {results['stoi']:.3f}")
    print("="*60)
    
    return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Audio Zooming Transformer')
    parser.add_argument('--train_data', type=str, required=True, 
                        help='Path to training data directory')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data directory')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    if args.mode == 'train':
        print("="*60)
        print("Audio Zooming Transformer - Training")
        print("="*60)
        print(f"Training data: {args.train_data}")
        print(f"Validation data: {args.val_data}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Device: {args.device}")
        print("="*60)
        
        # Install required packages reminder
        print("\nMake sure you have installed:")
        print("  pip install torch torchaudio scipy numpy")
        print("  pip install pesq pystoi  # For metrics")
        print()
        
        train_model(
            train_mat_file=args.train_data,
            val_mat_file=args.val_data,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir
        )
    
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            raise ValueError("Must provide --checkpoint path for evaluation")
        if args.test_data is None:
            raise ValueError("Must provide --test_data for evaluation")
        
        print("="*60)
        print("Audio Zooming Transformer - Evaluation")
        print("="*60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Test data: {args.test_data}")
        print(f"Device: {args.device}")
        print("="*60)
        
        results = load_and_evaluate(
            checkpoint_path=args.checkpoint,
            test_mat_file=args.test_data,
            device=args.device
        )
    
    # Quick start example without command line
    """
    # Training example:
    train_model(
        train_mat_file='data/train/',  # Must be a directory
        val_mat_file='data/val/',
        n_epochs=50,
        batch_size=8,
        lr=1e-4,
        device='cuda',
        save_dir='checkpoints'
    )
    
    # Evaluation example:
    results = load_and_evaluate(
        checkpoint_path='checkpoints/best_model.pt',
        test_mat_file='data/test/',
        device='cuda'
    )
    """