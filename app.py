import io
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import modal

# =====================================================================
# 1. MODAL APP & CONTAINER IMAGE CONFIGURATION (L40S + NVENC)
# =====================================================================

app = modal.App("gpu-music-visualizer-l40s")

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "fonts-dejavu-core",
        "fonts-liberation",
        "curl",
        "xz-utils",
        "ca-certificates",
    )
    # Install static FFmpeg 7.1 build with av1_nvenc, h264_nvenc, libx264, and libsvtav1
    .run_commands(
        "curl -sSL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz "
        "| tar -xJ --strip-components=2 -C /usr/local/bin/ ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffmpeg ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffprobe "
        "|| true"
    )
    .pip_install(
        "torch==2.4.1",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "pillow>=10.3.0",
        "fastapi[standard]>=0.115.0",
        "python-multipart>=0.0.9",
    )
    # Expose NVIDIA Video Codec SDK (NVENC) inside the Modal GPU container
    .env({"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video"})
)


# =====================================================================
# 2. DATA STRUCTURES & COLOR PALETTE EXTRACTION
# =====================================================================

@dataclass
class ColorPalette:
    bg_dark: Tuple[int, int, int]
    panel_bg: Tuple[int, int, int]
    panel_border: Tuple[int, int, int]
    grid_line: Tuple[int, int, int]
    text_main: Tuple[int, int, int]
    text_dim: Tuple[int, int, int]
    primary: Tuple[int, int, int]      # Dominant vibrant accent
    secondary: Tuple[int, int, int]    # Contrasting / analogous accent
    highlight: Tuple[int, int, int]    # Bright transient / peak accent
    warning: Tuple[int, int, int]      # Amber meter zone
    danger: Tuple[int, int, int]       # Red clip / phase warning


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


def _rgb_to_hsv_np(rgb: "np.ndarray") -> "np.ndarray":
    import numpy as np

    rgb_norm = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc
    s = np.where(maxc == 0, 0.0, deltac / np.maximum(maxc, 1e-6))

    h = np.zeros_like(maxc)
    mask = deltac > 1e-6
    r_mask = mask & (r == maxc)
    g_mask = mask & (g == maxc)
    b_mask = mask & (b == maxc)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / deltac[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / deltac[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / deltac[b_mask]) + 4.0
    h = h / 6.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb_tuple(h: float, s: float, v: float) -> Tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def extract_palette_from_cover(cover_bytes: Optional[bytes]) -> Tuple[ColorPalette, "Image.Image"]:
    """
    Extracts a cohesive 5-swatch studio color palette from album cover art using K-Means,
    or generates a sleek procedural vinyl cover + studio palette if none is uploaded.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if not cover_bytes:
        palette = ColorPalette(
            bg_dark=(9, 12, 20),
            panel_bg=(15, 20, 32),
            panel_border=(38, 52, 82),
            grid_line=(26, 36, 56),
            text_main=(240, 246, 255),
            text_dim=(130, 148, 178),
            primary=(0, 232, 255),     # Electric Cyan
            secondary=(185, 66, 255),  # Neon Violet
            highlight=(255, 196, 46),  # Warm Gold
            warning=(255, 170, 30),
            danger=(255, 58, 92),
        )
        # Generate procedural album cover art
        art = Image.new("RGB", (600, 600), palette.panel_bg)
        draw = ImageDraw.Draw(art)
        cx, cy = 300, 300
        for r in range(280, 20, -12):
            t = r / 280.0
            col = (
                int(palette.primary[0] * (1 - t) + palette.secondary[0] * t * 0.5),
                int(palette.primary[1] * (1 - t) * 0.6 + palette.secondary[1] * t * 0.4),
                int(palette.primary[2] * (1 - t) * 0.8 + palette.secondary[2] * t),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
        draw.ellipse([cx - 85, cy - 85, cx + 85, cy + 85], fill=palette.bg_dark, outline=palette.highlight, width=4)
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=palette.highlight)
        return palette, art

    img = Image.open(io.BytesIO(cover_bytes)).convert("RGB")
    # Center-crop to square for clean album art display
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    square_img = img.crop((left, top, left + side, top + side)).resize((600, 600), Image.Resampling.LANCZOS)

    # Downsample for fast K-Means clustering
    small = square_img.resize((64, 64), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32).reshape(-1, 3)
    hsv = _rgb_to_hsv_np(pixels)

    # Filter for colorful / non-extreme pixels if enough exist
    vivid_mask = (hsv[:, 1] > 0.18) & (hsv[:, 2] > 0.18) & (hsv[:, 2] < 0.98)
    sample_pixels = pixels[vivid_mask] if np.sum(vivid_mask) >= 120 else pixels

    # Deterministic K-Means (k=6)
    k = 6
    rng = np.random.default_rng(42)
    indices = rng.choice(len(sample_pixels), size=min(k, len(sample_pixels)), replace=False)
    centers = sample_pixels[indices].copy()

    counts = np.ones(len(centers), dtype=np.float32)
    for _ in range(15):
        dists = np.sum((sample_pixels[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dists, axis=1)
        for idx in range(len(centers)):
            cluster_pts = sample_pixels[labels == idx]
            if len(cluster_pts) > 0:
                centers[idx] = cluster_pts.mean(axis=0)
                counts[idx] = len(cluster_pts)

    center_hsv = _rgb_to_hsv_np(centers)
    # Score clusters by saturation, brightness, and population
    scores = (center_hsv[:, 1] ** 0.8 + 0.25) * (center_hsv[:, 2] ** 0.7 + 0.2) * (counts ** 0.25)
    order = np.argsort(-scores)
    sorted_hsv = center_hsv[order]

    # Pick Primary (most vibrant)
    h1, s1, v1 = sorted_hsv[0]
    primary = _hsv_to_rgb_tuple(float(h1), max(0.65, float(s1)), max(0.88, float(v1)))

    # Pick Secondary (distinct hue from primary if available)
    h2, s2, v2 = sorted_hsv[1] if len(sorted_hsv) > 1 else ((h1 + 0.33) % 1.0, s1, v1)
    for cand in sorted_hsv[1:]:
        hue_dist = min(abs(float(cand[0]) - float(h1)), 1.0 - abs(float(cand[0]) - float(h1)))
        if hue_dist > 0.10:
            h2, s2, v2 = cand
            break
    if min(abs(float(h2) - float(h1)), 1.0 - abs(float(h2) - float(h1))) < 0.07:
        h2 = (float(h1) + 0.35) % 1.0
    secondary = _hsv_to_rgb_tuple(float(h2), max(0.65, float(s2)), max(0.85, float(v2)))

    # Pick Highlight (brightest / contrasting third color)
    h3 = (float(h1) + 0.55) % 1.0
    s3, v3 = 0.80, 0.98
    for cand in sorted_hsv[2:]:
        d1 = min(abs(float(cand[0]) - float(h1)), 1.0 - abs(float(cand[0]) - float(h1)))
        d2 = min(abs(float(cand[0]) - float(h2)), 1.0 - abs(float(cand[0]) - float(h2)))
        if d1 > 0.12 and d2 > 0.12:
            h3, s3, v3 = float(cand[0]), max(0.70, float(cand[1])), max(0.95, float(cand[2]))
            break
    highlight = _hsv_to_rgb_tuple(h3, s3, v3)

    # Dark studio surfaces tinted with dominant hue
    bg_dark = _hsv_to_rgb_tuple(float(h1), min(0.45, float(s1) * 0.6 + 0.15), 0.06)
    panel_bg = _hsv_to_rgb_tuple(float(h1), min(0.42, float(s1) * 0.5 + 0.15), 0.10)
    panel_border = _hsv_to_rgb_tuple(float(h1), min(0.50, float(s1) * 0.6 + 0.20), 0.28)
    grid_line = _hsv_to_rgb_tuple(float(h1), min(0.38, float(s1) * 0.4 + 0.12), 0.18)

    palette = ColorPalette(
        bg_dark=bg_dark,
        panel_bg=panel_bg,
        panel_border=panel_border,
        grid_line=grid_line,
        text_main=(244, 247, 252),
        text_dim=(145, 160, 185),
        primary=primary,
        secondary=secondary,
        highlight=highlight,
        warning=(255, 180, 40),
        danger=(255, 62, 96),
    )
    return palette, square_img


def build_spectrogram_lut(palette: ColorPalette, device: "torch.device") -> "torch.Tensor":
    """Builds a 256x3 uint8 colormap LUT on GPU from the extracted album cover palette."""
    import numpy as np
    import torch

    stops = [
        (0.00, palette.panel_bg),
        (0.18, (
            int(palette.panel_bg[0] * 0.5 + palette.secondary[0] * 0.35),
            int(palette.panel_bg[1] * 0.5 + palette.secondary[1] * 0.35),
            int(palette.panel_bg[2] * 0.5 + palette.secondary[2] * 0.35),
        )),
        (0.45, palette.secondary),
        (0.74, palette.primary),
        (0.92, palette.highlight),
        (1.00, (255, 255, 255)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for s_idx in range(len(stops) - 1):
            t0, c0 = stops[s_idx]
            t1, c1 = stops[s_idx + 1]
            if t0 <= t <= t1:
                local_t = (t - t0) / max(1e-6, (t1 - t0))
                lut[i, 0] = int(round(c0[0] * (1 - local_t) + c1[0] * local_t))
                lut[i, 1] = int(round(c0[1] * (1 - local_t) + c1[1] * local_t))
                lut[i, 2] = int(round(c0[2] * (1 - local_t) + c1[2] * local_t))
                break
    return torch.from_numpy(lut).to(device=device, dtype=torch.uint8)


# =====================================================================
# 3. AUDIO DECODING & DSP PRE-COMPUTATION (ITU-R BS.1770-4, STFT, PITCH)
# =====================================================================

SAMPLE_RATE = 48000


def decode_audio_stereo_48k(audio_path: str) -> "np.ndarray":
    """Decodes any audio file via FFmpeg into stereo 48kHz float32 numpy array of shape (2, N)."""
    import numpy as np

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", audio_path,
        "-vn",
        "-ac", "2",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if raw.size < 2:
        raise ValueError("Uploaded audio file contains no valid audio samples.")
    stereo = raw.reshape(-1, 2).T.copy()  # (2, N)
    return stereo


def compute_audio_features(
    stereo: "np.ndarray",
    fps: int,
    spec_width: int,
    device: "torch.device",
) -> Dict[str, any]:
    """
    Pre-computes frame-accurate audio features:
    - ITU-R BS.1770-4 K-weighted Momentary, Short-Term, and Integrated LUFS
    - Left/Right Peak (with ballistic decay) & RMS dBFS
    - High-resolution scrolling waveform envelope (min, max, RMS, transient flux)
    - GPU-accelerated 20Hz-20kHz log-frequency Spectrum & Spectrogram matrices
    """
    import numpy as np
    import scipy.signal as signal
    import torch

    n_samples = stereo.shape[1]
    duration = n_samples / float(SAMPLE_RATE)
    n_frames = max(1, int(math.ceil(duration * fps)))

    # 1. ITU-R BS.1770-4 K-Weighting Filter at 48kHz
    # Stage 1: High-shelf pre-filter
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    # Stage 2: RLB high-pass filter
    b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)

    k_filtered = signal.lfilter(b2, a2, signal.lfilter(b1, a1, stereo.astype(np.float64), axis=1), axis=1)
    energy_k = (k_filtered[0] ** 2) + (k_filtered[1] ** 2)
    cumsum_k = np.concatenate([[0.0], np.cumsum(energy_k)])

    frame_centers = np.clip((np.arange(n_frames) * SAMPLE_RATE / fps).astype(np.int64), 0, n_samples - 1)

    # Momentary LUFS (400ms window = 19,200 samples) & Short-Term LUFS (3.0s window = 144,000 samples)
    win_m = int(0.400 * SAMPLE_RATE)
    win_s = int(3.000 * SAMPLE_RATE)

    m_start = np.clip(frame_centers - win_m, 0, n_samples - 1)
    m_end = np.clip(frame_centers + 1, 1, n_samples)
    m_mean = (cumsum_k[m_end] - cumsum_k[m_start]) / np.maximum(1, m_end - m_start)
    momentary_lufs = (-0.691 + 10.0 * np.log10(np.maximum(m_mean, 1e-12))).astype(np.float32)

    s_start = np.clip(frame_centers - win_s, 0, n_samples - 1)
    s_end = np.clip(frame_centers + 1, 1, n_samples)
    s_mean = (cumsum_k[s_end] - cumsum_k[s_start]) / np.maximum(1, s_end - s_start)
    short_term_lufs = (-0.691 + 10.0 * np.log10(np.maximum(s_mean, 1e-12))).astype(np.float32)

    # Integrated LUFS (gated BS.1770 approximation)
    valid_blocks = momentary_lufs[momentary_lufs > -70.0]
    if len(valid_blocks) > 0:
        rel_gate = -0.691 + 10.0 * np.log10(np.mean(10.0 ** ((valid_blocks + 0.691) / 10.0))) - 10.0
        gated_blocks = valid_blocks[valid_blocks > rel_gate]
        if len(gated_blocks) > 0:
            integrated_lufs = float(-0.691 + 10.0 * np.log10(np.mean(10.0 ** ((gated_blocks + 0.691) / 10.0))))
        else:
            integrated_lufs = float(np.mean(valid_blocks))
    else:
        integrated_lufs = -70.0

    # 2. Per-Frame L/R Sample Peak, Peak-Hold, and RMS (in dBFS)
    hop = max(1, int(SAMPLE_RATE / fps))
    peak_l_db = np.full(n_frames, -60.0, dtype=np.float32)
    peak_r_db = np.full(n_frames, -60.0, dtype=np.float32)
    rms_l_db = np.full(n_frames, -60.0, dtype=np.float32)
    rms_r_db = np.full(n_frames, -60.0, dtype=np.float32)

    for i in range(n_frames):
        s0 = max(0, frame_centers[i] - hop)
        s1 = min(n_samples, frame_centers[i] + hop)
        seg_l = stereo[0, s0:s1]
        seg_r = stereo[1, s0:s1]
        if seg_l.size > 0:
            peak_l_db[i] = 20.0 * math.log10(max(float(np.max(np.abs(seg_l))), 1e-6))
            peak_r_db[i] = 20.0 * math.log10(max(float(np.max(np.abs(seg_r))), 1e-6))
            rms_l_db[i] = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg_l ** 2))), 1e-6))
            rms_r_db[i] = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg_r ** 2))), 1e-6))

    # Apply smooth meter ballistics (18 dB/sec release, plus 0.6s peak hold)
    release_per_frame = 18.0 / fps
    hold_frames = int(0.6 * fps)
    hold_l_db = peak_l_db.copy()
    hold_r_db = peak_r_db.copy()
    l_timer, r_timer = 0, 0

    for i in range(1, n_frames):
        peak_l_db[i] = max(peak_l_db[i], peak_l_db[i - 1] - release_per_frame)
        peak_r_db[i] = max(peak_r_db[i], peak_r_db[i - 1] - release_per_frame)
        rms_l_db[i] = max(rms_l_db[i], rms_l_db[i - 1] - release_per_frame * 0.8)
        rms_r_db[i] = max(rms_r_db[i], rms_r_db[i - 1] - release_per_frame * 0.8)

        if peak_l_db[i] >= hold_l_db[i - 1]:
            hold_l_db[i] = peak_l_db[i]
            l_timer = hold_frames
        else:
            if l_timer > 0:
                hold_l_db[i] = hold_l_db[i - 1]
                l_timer -= 1
            else:
                hold_l_db[i] = max(peak_l_db[i], hold_l_db[i - 1] - release_per_frame * 0.5)

        if peak_r_db[i] >= hold_r_db[i - 1]:
            hold_r_db[i] = peak_r_db[i]
            r_timer = hold_frames
        else:
            if r_timer > 0:
                hold_r_db[i] = hold_r_db[i - 1]
                r_timer -= 1
            else:
                hold_r_db[i] = max(peak_r_db[i], hold_r_db[i - 1] - release_per_frame * 0.5)

    # 3. High-Resolution Envelope Timeline for Scrolling Waveform Meter (240 slices/sec)
    env_rate = 240
    env_step = SAMPLE_RATE // env_rate
    n_env = max(1, int(math.ceil(n_samples / env_step)))
    pad_len = n_env * env_step - n_samples
    mid_wave = 0.5 * (stereo[0] + stereo[1])
    if pad_len > 0:
        mid_padded = np.pad(mid_wave, (0, pad_len))
    else:
        mid_padded = mid_wave[: n_env * env_step]
    mid_blocks = mid_padded.reshape(n_env, env_step)

    env_max = np.max(mid_blocks, axis=1).astype(np.float32)
    env_min = np.min(mid_blocks, axis=1).astype(np.float32)
    env_rms = np.sqrt(np.mean(mid_blocks ** 2, axis=1)).astype(np.float32)

    # Compute transient onset score (ratio of fast attack to slow decay envelope)
    fast_env = np.maximum(np.abs(env_max), np.abs(env_min))
    slow_env = np.zeros_like(fast_env)
    acc = 0.0
    for i in range(n_env):
        acc = 0.92 * acc + 0.08 * fast_env[i]
        slow_env[i] = acc
    env_transient = np.clip((fast_env - slow_env * 0.85) / (slow_env + 0.04), 0.0, 1.0).astype(np.float32)

    # 4. GPU-Accelerated Log-Frequency STFT for Spectrum Analyzer & Spectrogram
    n_fft = 4096
    mid_gpu = torch.from_numpy(mid_wave).to(device=device, dtype=torch.float32)
    window_gpu = torch.hann_window(n_fft, device=device)

    # Pad audio so frame_centers align directly with STFT frames
    padded_gpu = torch.nn.functional.pad(mid_gpu, (n_fft // 2, n_fft // 2))
    # Extract exact windows around frame_centers in batches on GPU
    centers_gpu = torch.from_numpy(frame_centers).to(device=device, dtype=torch.long)
    offsets = torch.arange(n_fft, device=device, dtype=torch.long)

    # Build log-frequency interpolation matrix (20 Hz to 20,000 Hz) with 3 dB/octave pink-noise tilt
    freqs_fft = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    target_freqs = np.geomspace(20.0, 20000.0, num=spec_width)
    bin_coords = np.interp(target_freqs, freqs_fft, np.arange(len(freqs_fft))).astype(np.float32)
    bin_low = np.floor(bin_coords).astype(np.int64)
    bin_high = np.minimum(bin_low + 1, len(freqs_fft) - 1)
    bin_frac = (bin_coords - bin_low).astype(np.float32)

    # 3 dB/octave pink noise compensation referenced to 200 Hz
    tilt_db = (3.0 * np.log2(np.maximum(target_freqs, 20.0) / 200.0)).astype(np.float32)

    bin_low_gpu = torch.from_numpy(bin_low).to(device=device)
    bin_high_gpu = torch.from_numpy(bin_high).to(device=device)
    bin_frac_gpu = torch.from_numpy(bin_frac).to(device=device)
    tilt_db_gpu = torch.from_numpy(tilt_db).to(device=device)

    spec_raw_norm = torch.empty((n_frames, spec_width), device=device, dtype=torch.float32)
    batch_size = 1024
    for b0 in range(0, n_frames, batch_size):
        b1_idx = min(n_frames, b0 + batch_size)
        idx_mat = centers_gpu[b0:b1_idx, None] + offsets[None, :]
        frames_batch = padded_gpu[idx_mat] * window_gpu[None, :]
        fft_mag = torch.abs(torch.fft.rfft(frames_batch, n=n_fft, dim=1)) / (n_fft * 0.25)

        interp_mag = (1.0 - bin_frac_gpu[None, :]) * fft_mag[:, bin_low_gpu] + bin_frac_gpu[None, :] * fft_mag[:, bin_high_gpu]
        mag_db = 20.0 * torch.log10(torch.clamp(interp_mag, min=1e-6)) + tilt_db_gpu[None, :]
        # Map [-72 dB .. -4 dB] -> [0.0 .. 1.0]
        norm_val = torch.clamp((mag_db + 72.0) / 68.0, 0.0, 1.0)
        # Apply slight spatial 3-tap smoothing across frequency columns
        norm_pad = torch.nn.functional.pad(norm_val.unsqueeze(1), (1, 1), mode="replicate")
        norm_smooth = (norm_pad[:, 0, :-2] * 0.22 + norm_pad[:, 0, 1:-1] * 0.56 + norm_pad[:, 0, 2:] * 0.22)
        spec_raw_norm[b0:b1_idx] = norm_smooth

    # Temporal smoothing for Spectrum Analyzer + Peak-Hold trace
    spec_smooth = torch.empty_like(spec_raw_norm)
    spec_peak = torch.empty_like(spec_raw_norm)
    curr_s = spec_raw_norm[0].clone()
    curr_p = spec_raw_norm[0].clone()
    decay_alpha = math.exp(-8.5 / fps)
    peak_decay = 0.32 / fps

    for i in range(n_frames):
        raw_i = spec_raw_norm[i]
        curr_s = torch.maximum(raw_i * 0.72 + curr_s * 0.28, curr_s * decay_alpha + raw_i * (1.0 - decay_alpha))
        curr_p = torch.maximum(curr_s, curr_p - peak_decay)
        spec_smooth[i] = curr_s
        spec_peak[i] = curr_p

    return {
        "n_frames": n_frames,
        "duration": duration,
        "frame_centers": frame_centers,
        "momentary_lufs": momentary_lufs,
        "short_term_lufs": short_term_lufs,
        "integrated_lufs": integrated_lufs,
        "peak_l_db": peak_l_db,
        "peak_r_db": peak_r_db,
        "hold_l_db": hold_l_db,
        "hold_r_db": hold_r_db,
        "rms_l_db": rms_l_db,
        "rms_r_db": rms_r_db,
        "env_rate": env_rate,
        "env_max": torch.from_numpy(env_max).to(device=device),
        "env_min": torch.from_numpy(env_min).to(device=device),
        "env_rms": torch.from_numpy(env_rms).to(device=device),
        "env_transient": torch.from_numpy(env_transient).to(device=device),
        "spec_raw_norm": spec_raw_norm,
        "spec_smooth": spec_smooth,
        "spec_peak": spec_peak,
    }


# =====================================================================
# 4. LAYOUT ENGINE & STATIC STUDIO UI OVERLAY GENERATOR
# =====================================================================

def compute_layout(width: int, height: int, orientation: str) -> Dict[str, Rect]:
    """Computes pixel-aligned panel rectangles for Landscape (16:9) or Vertical (9:16)."""
    def r(x0: float, y0: float, x1: float, y1: float) -> Rect:
        px0 = int(round(x0 * width))
        py0 = int(round(y0 * height))
        px1 = int(round(x1 * width))
        py1 = int(round(y1 * height))
        return Rect(px0, py0, max(16, px1 - px0), max(16, py1 - py0))

    if orientation == "vertical":
        return {
            "header": r(0.035, 0.022, 0.965, 0.155),
            "waveform_meter": r(0.035, 0.170, 0.755, 0.275),
            "oscilloscope": r(0.035, 0.290, 0.385, 0.425),
            "waveform_trace": r(0.405, 0.290, 0.755, 0.425),
            "stereometer": r(0.035, 0.440, 0.755, 0.605),
            "spectrum": r(0.035, 0.620, 0.755, 0.790),
            "spectrogram": r(0.035, 0.805, 0.755, 0.975),
            "lufs_meter": r(0.775, 0.170, 0.965, 0.975),
        }
    else:
        return {
            "header": r(0.018, 0.032, 0.285, 0.340),
            "stereometer": r(0.018, 0.362, 0.285, 0.675),
            "oscilloscope": r(0.018, 0.697, 0.147, 0.968),
            "waveform_trace": r(0.156, 0.697, 0.285, 0.968),
            "waveform_meter": r(0.298, 0.032, 0.852, 0.235),
            "spectrum": r(0.298, 0.257, 0.852, 0.615),
            "spectrogram": r(0.298, 0.637, 0.852, 0.968),
            "lufs_meter": r(0.865, 0.032, 0.982, 0.968),
        }


def _load_font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
    from PIL import ImageFont

    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=max(8, size))
    return ImageFont.load_default()


def _load_mono_font(size: int, bold: bool = True) -> "ImageFont.FreeTypeFont":
    from PIL import ImageFont

    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=max(8, size))
    return _load_font(size, bold=bold)


def build_static_layers_and_glyphs(
    width: int,
    height: int,
    orientation: str,
    layout: Dict[str, Rect],
    palette: ColorPalette,
    cover_img: "Image.Image",
    title: str,
    artist: str,
    codec: str,
    fps: int,
    integrated_lufs: float,
    device: "torch.device",
) -> Dict[str, any]:
    """
    Pre-renders the static studio background, panel cards, album art, typography,
    graticules, and a GPU bitmap glyph atlas for zero-overhead live digital readouts.
    """
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFilter

    scale = min(width, height) / 1080.0
    base_img = Image.new("RGB", (width, height), palette.bg_dark)

    # Subtle ambient background glow from album cover art
    ambient = cover_img.resize((width, height), Image.Resampling.BILINEAR).filter(
        ImageFilter.GaussianBlur(radius=int(80 * scale))
    )
    base_img = Image.blend(base_img, ambient, alpha=0.14)
    base_draw = ImageDraw.Draw(base_img)

    # Foreground overlay (drawn on top of dynamic graphs: borders, tick labels, crosshairs)
    fg_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    fg_draw = ImageDraw.Draw(fg_img)

    font_title = _load_font(int(26 * scale), bold=True)
    font_artist = _load_font(int(18 * scale), bold=False)
    font_panel = _load_font(int(12 * scale), bold=True)
    font_tick = _load_mono_font(int(10 * scale), bold=False)

    header_pad = int(24 * scale)
    radius = max(6, int(10 * scale))
    border_w = max(1, int(1.5 * scale))

    panel_titles = {
        "header": "TRACK METADATA & PALETTE",
        "waveform_meter": "WAVEFORM METER  //  AMPLITUDE ENVELOPE & TRANSIENTS",
        "oscilloscope": "OSCILLOSCOPE",
        "waveform_trace": "WAVEFORM TRACE (1T)",
        "stereometer": "STEREOMETER  //  GONIOMETER & PHASE SCOPE",
        "spectrum": "SPECTRUM ANALYZER  //  RTA 20Hz - 20kHz",
        "spectrogram": "SPECTROGRAM  //  FREQUENCY INTENSITY WATERFALL",
        "lufs_meter": "PEAK & LUFS",
    }

    inner_rects: Dict[str, Rect] = {}

    for key, rect in layout.items():
        # Panel card fill
        base_draw.rounded_rectangle(
            [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
            radius=radius,
            fill=palette.panel_bg,
        )
        # Header bar strip
        base_draw.rounded_rectangle(
            [rect.x, rect.y, rect.x + rect.w, rect.y + header_pad],
            radius=radius,
            fill=(
                int(palette.panel_bg[0] * 0.7 + palette.panel_border[0] * 0.3),
                int(palette.panel_bg[1] * 0.7 + palette.panel_border[1] * 0.3),
                int(palette.panel_bg[2] * 0.7 + palette.panel_border[2] * 0.3),
            ),
        )
        # Panel title text on foreground overlay
        fg_draw.text(
            (rect.x + int(10 * scale), rect.y + int(5 * scale)),
            panel_titles.get(key, key.upper()),
            font=font_panel,
            fill=(*palette.text_dim, 235),
        )
        # Outer rounded border on foreground overlay
        fg_draw.rounded_rectangle(
            [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
            radius=radius,
            outline=(*palette.panel_border, 220),
            width=border_w,
        )

        pad = max(4, int(6 * scale))
        ix = rect.x + pad
        iy = rect.y + header_pad + pad
        iw = max(8, rect.w - 2 * pad)
        ih = max(8, rect.h - header_pad - 2 * pad)
        inner_rects[key] = Rect(ix, iy, iw, ih)

    # 1. Populate Header Card (Album Art, Title, Artist, Technical Badges & Palette Swatches)
    hr = inner_rects["header"]
    art_size = min(hr.h - int(6 * scale), int(hr.w * 0.42))
    art_x = hr.x + int(4 * scale)
    art_y = hr.y + (hr.h - art_size) // 2

    art_resized = cover_img.resize((art_size, art_size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (art_size, art_size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, art_size, art_size], radius=int(8 * scale), fill=255)
    base_img.paste(art_resized, (art_x, art_y), mask)
    fg_draw.rounded_rectangle(
        [art_x, art_y, art_x + art_size, art_y + art_size],
        radius=int(8 * scale),
        outline=(*palette.primary, 190),
        width=max(1, int(1.5 * scale)),
    )

    tx = art_x + art_size + int(14 * scale)
    ty = art_y + int(4 * scale)
    max_tw = hr.x + hr.w - tx - int(6 * scale)

    def _fit_text(text: str, font: "ImageFont.FreeTypeFont", max_w: int) -> str:
        if base_draw.textlength(text, font=font) <= max_w:
            return text
        ellipsis = "..."
        for cut in range(len(text), 0, -1):
            candidate = text[:cut].rstrip() + ellipsis
            if base_draw.textlength(candidate, font=font) <= max_w:
                return candidate
        return text[:12]

    base_draw.text((tx, ty), _fit_text(title, font_title, max_tw), font=font_title, fill=palette.text_main)
    ty += int(32 * scale)
    base_draw.text((tx, ty), _fit_text(artist, font_artist, max_tw), font=font_artist, fill=palette.primary)
    ty += int(28 * scale)

    meta_str = f"{width}x{height} @ {fps}fps  |  {codec.upper()}  |  INT: {integrated_lufs:.1f} LUFS"
    base_draw.text((tx, ty), _fit_text(meta_str, font_tick, max_tw), font=font_tick, fill=palette.text_dim)
    ty += int(20 * scale)

    # Extracted Color Palette Swatches
    swatch_w = max(12, int(20 * scale))
    swatch_h = max(8, int(10 * scale))
    swatch_gap = int(6 * scale)
    swatches = [palette.primary, palette.secondary, palette.highlight, palette.panel_border, palette.bg_dark]
    for s_idx, col in enumerate(swatches):
        sx = tx + s_idx * (swatch_w + swatch_gap)
        if sx + swatch_w <= hr.x + hr.w:
            base_draw.rounded_rectangle(
                [sx, ty, sx + swatch_w, ty + swatch_h],
                radius=max(2, int(3 * scale)),
                fill=col,
                outline=palette.text_dim,
                width=1,
            )

    # 2. Oscilloscope & Waveform Trace Graticules
    for key in ("oscilloscope", "waveform_trace"):
        ir = inner_rects[key]
        cy = ir.y + ir.h // 2
        for frac in (0.25, 0.5, 0.75):
            gy = ir.y + int(ir.h * frac)
            base_draw.line([(ir.x, gy), (ir.x + ir.w, gy)], fill=palette.grid_line, width=1)
        for frac in (0.25, 0.5, 0.75):
            gx = ir.x + int(ir.w * frac)
            base_draw.line([(gx, ir.y), (gx, ir.y + ir.h)], fill=palette.grid_line, width=1)
        fg_draw.line([(ir.x, cy), (ir.x + ir.w, cy)], fill=(*palette.text_dim, 75), width=1)

    # 3. Scrolling Waveform Meter Grid & Playhead Marker
    wmr = inner_rects["waveform_meter"]
    wm_cy = wmr.y + wmr.h // 2
    base_draw.line([(wmr.x, wm_cy), (wmr.x + wmr.w, wm_cy)], fill=palette.grid_line, width=1)
    playhead_x = wmr.x + int(wmr.w * 0.70)
    for dt_sec in (-4, -3, -2, -1, 0, 1):
        # Window spans -4.5s to +1.9s (6.4s total, playhead at 70%)
        norm_x = 0.70 + (dt_sec / 6.4)
        if 0.02 <= norm_x <= 0.98:
            gx = wmr.x + int(wmr.w * norm_x)
            base_draw.line([(gx, wmr.y), (gx, wmr.y + wmr.h)], fill=palette.grid_line, width=1)
            lbl = "NOW" if dt_sec == 0 else f"{dt_sec:+d}s"
            fg_draw.text((gx + int(3 * scale), wmr.y + wmr.h - int(14 * scale)), lbl, font=font_tick, fill=(*palette.text_dim, 180))
    fg_draw.line([(playhead_x, wmr.y), (playhead_x, wmr.y + wmr.h)], fill=(*palette.highlight, 230), width=max(1, int(2 * scale)))

    # 4. Spectrum Analyzer & Spectrogram Shared Log-Frequency Grid (20 Hz - 20 kHz)
    spr = inner_rects["spectrum"]
    sgr = inner_rects["spectrogram"]
    freq_ticks = [
        (30, "30"), (60, "60"), (125, "125"), (250, "250"),
        (500, "500"), (1000, "1k"), (2000, "2k"), (4000, "4k"),
        (8000, "8k"), (16000, "16k"),
    ]
    log_min = math.log10(20.0)
    log_max = math.log10(20000.0)
    for f_hz, f_lbl in freq_ticks:
        norm_x = (math.log10(f_hz) - log_min) / (log_max - log_min)
        gx = spr.x + int(norm_x * spr.w)
        base_draw.line([(gx, spr.y), (gx, spr.y + spr.h)], fill=palette.grid_line, width=1)
        fg_draw.line([(gx, sgr.y), (gx, sgr.y + int(8 * scale))], fill=(*palette.text_dim, 150), width=1)
        fg_draw.text((gx + int(3 * scale), spr.y + spr.h - int(14 * scale)), f_lbl, font=font_tick, fill=(*palette.text_dim, 200))

    for db_val, db_frac in [("-6dB", 0.12), ("-18dB", 0.30), ("-36dB", 0.56), ("-54dB", 0.82)]:
        gy = spr.y + int(spr.h * db_frac)
        base_draw.line([(spr.x, gy), (spr.x + spr.w, gy)], fill=palette.grid_line, width=1)
        fg_draw.text((spr.x + int(4 * scale), gy - int(12 * scale)), db_val, font=font_tick, fill=(*palette.text_dim, 165))

    # 5. Stereometer / Goniometer Semi-Circular Polar Grid + Phase Correlation Scale
    str_r = inner_rects["stereometer"]
    fan_h = int(str_r.h * 0.80)
    fan_cx = str_r.x + str_r.w // 2
    fan_cy = str_r.y + fan_h - int(6 * scale)
    fan_rad = min(int(str_r.w * 0.45), fan_h - int(16 * scale))

    for r_frac in (0.33, 0.66, 1.0):
        rr = int(fan_rad * r_frac)
        bbox = [fan_cx - rr, fan_cy - rr, fan_cx + rr, fan_cy + rr]
        col = palette.panel_border if r_frac == 1.0 else palette.grid_line
        fg_draw.arc(bbox, start=180, end=360, fill=(*col, 200), width=max(1, int(1.3 * scale)))

    for deg, lbl in [(-90, "-S"), (-45, "L"), (0, "M"), (45, "R"), (90, "+S")]:
        rad_ang = math.radians(deg)
        x_end = fan_cx + int(fan_rad * math.sin(rad_ang))
        y_end = fan_cy - int(fan_rad * math.cos(rad_ang))
        base_draw.line([(fan_cx, fan_cy), (x_end, y_end)], fill=palette.grid_line, width=1)
        lx = fan_cx + int((fan_rad + 12 * scale) * math.sin(rad_ang)) - int(6 * scale)
        ly = fan_cy - int((fan_rad + 11 * scale) * math.cos(rad_ang)) - int(6 * scale)
        fg_draw.text((lx, ly), lbl, font=font_tick, fill=(*palette.text_main, 215))

    # Phase Correlation Meter Bar Frame (-1 .. 0 .. +1)
    corr_y = str_r.y + fan_h + int(8 * scale)
    corr_h = max(8, int(12 * scale))
    corr_x0 = str_r.x + int(str_r.w * 0.10)
    corr_w = int(str_r.w * 0.80)
    fg_draw.rounded_rectangle(
        [corr_x0, corr_y, corr_x0 + corr_w, corr_y + corr_h],
        radius=max(2, int(4 * scale)),
        outline=(*palette.panel_border, 220),
        width=1,
    )
    for c_val, c_lbl in [(-1.0, "-1"), (0.0, "0"), (1.0, "+1")]:
        cx_tick = corr_x0 + int((c_val + 1.0) * 0.5 * corr_w)
        fg_draw.line([(cx_tick, corr_y - 2), (cx_tick, corr_y + corr_h + 2)], fill=(*palette.text_main, 200), width=1)
        fg_draw.text((cx_tick - int(5 * scale), corr_y + corr_h + int(3 * scale)), c_lbl, font=font_tick, fill=(*palette.text_dim, 200))

    # 6. Peak & LUFS Meter Scale & Bar Labels
    lur = inner_rects["lufs_meter"]
    readout_h = int(lur.h * 0.16)
    bars_y0 = lur.y + readout_h + int(8 * scale)
    bars_h = lur.h - readout_h - int(28 * scale)
    bars_x0 = lur.x + int(28 * scale)
    bars_w = max(24, lur.w - int(34 * scale))

    # Readout box background
    base_draw.rounded_rectangle(
        [lur.x + int(2 * scale), lur.y, lur.x + lur.w - int(2 * scale), lur.y + readout_h],
        radius=int(6 * scale),
        fill=palette.bg_dark,
        outline=palette.panel_border,
        width=1,
    )

    # dB scale ticks (0 to -60 dB / LUFS)
    for db_tick in (0, -6, -14, -24, -36, -48, -60):
        norm_y = abs(db_tick) / 60.0
        ty_bar = bars_y0 + int(norm_y * bars_h)
        t_col = palette.highlight if db_tick == -14 else palette.grid_line
        base_draw.line([(bars_x0, ty_bar), (bars_x0 + bars_w, ty_bar)], fill=t_col, width=1)
        fg_draw.text((lur.x + int(2 * scale), ty_bar - int(6 * scale)), f"{db_tick:d}", font=font_tick, fill=(*palette.text_dim, 210))

    col_names = ["L", "R", "M", "S"]
    col_step = bars_w / 4.0
    for idx, cname in enumerate(col_names):
        cx_bar = int(bars_x0 + (idx + 0.5) * col_step) - int(4 * scale)
        fg_draw.text((cx_bar, bars_y0 + bars_h + int(4 * scale)), cname, font=font_panel, fill=(*palette.text_main, 230))

    # 7. Pre-render GPU Bitmap Font Atlas for Live Digital Readouts (-14.1 LU, -0.8 dB, Hz)
    glyph_chars = "0123456789-+.:/LUFSdBHzmsPK "
    font_readout_lg = _load_mono_font(max(11, int(19 * scale)), bold=True)
    font_readout_sm = _load_mono_font(max(9, int(12 * scale)), bold=True)

    def _build_glyph_dict(font_obj: "ImageFont.FreeTypeFont") -> Tuple[Dict[str, "torch.Tensor"], int, int]:
        bbox = font_obj.getbbox("0")
        gw = max(6, int(font_obj.getlength("0")))
        gh = max(10, int((bbox[3] - bbox[1]) * 1.45))
        g_dict: Dict[str, torch.Tensor] = {}
        for ch in glyph_chars:
            g_img = Image.new("L", (gw, gh), 0)
            g_draw = ImageDraw.Draw(g_img)
            g_draw.text((0, 0), ch, font=font_obj, fill=255)
            arr = np.asarray(g_img, dtype=np.float32) / 255.0
            g_dict[ch] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        return g_dict, gw, gh

    glyphs_lg, gw_lg, gh_lg = _build_glyph_dict(font_readout_lg)
    glyphs_sm, gw_sm, gh_sm = _build_glyph_dict(font_readout_sm)

    base_np = np.array(base_img, dtype=np.uint8)
    fg_np = np.array(fg_img, dtype=np.uint8)

    base_gpu = torch.from_numpy(base_np).to(device=device, dtype=torch.uint8)
    fg_rgb_gpu = torch.from_numpy(fg_np[:, :, :3]).to(device=device, dtype=torch.float32)
    fg_alpha_gpu = (torch.from_numpy(fg_np[:, :, 3:4]).to(device=device, dtype=torch.float32) / 255.0)

    return {
        "inner_rects": inner_rects,
        "base_gpu": base_gpu,
        "fg_rgb_gpu": fg_rgb_gpu,
        "fg_alpha_gpu": fg_alpha_gpu,
        "glyphs_lg": glyphs_lg,
        "gw_lg": gw_lg,
        "gh_lg": gh_lg,
        "glyphs_sm": glyphs_sm,
        "gw_sm": gw_sm,
        "gh_sm": gh_sm,
        "stereo_meta": {
            "fan_h": fan_h,
            "fan_cx_local": fan_cx - str_r.x,
            "fan_cy_local": fan_cy - str_r.y,
            "fan_rad": fan_rad,
            "corr_y_local": corr_y - str_r.y,
            "corr_h": corr_h,
            "corr_x0_local": corr_x0 - str_r.x,
            "corr_w": corr_w,
        },
        "lufs_meta": {
            "readout_h": readout_h,
            "bars_y0_local": bars_y0 - lur.y,
            "bars_h": bars_h,
            "bars_x0_local": bars_x0 - lur.x,
            "bars_w": bars_w,
        },
    }


# =====================================================================
# 5. GPU REAL-TIME FRAME RENDERER (PYTORCH ON L40S CUDA)
# =====================================================================

class VisualizerGPURenderer:
    """
    Renders all 7 pro-audio visualizer instruments directly inside GPU tensors
    and converts frames to YUV420P (BT.709) on the L40S GPU.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        stereo_audio: "np.ndarray",
        features: Dict[str, any],
        ui_data: Dict[str, any],
        palette: ColorPalette,
        device: "torch.device",
    ):
        import numpy as np
        import torch

        self.width = width
        self.height = height
        self.fps = fps
        self.stereo_np = stereo_audio
        self.mid_np = 0.5 * (stereo_audio[0] + stereo_audio[1])
        self.n_samples = stereo_audio.shape[1]
        self.features = features
        self.ui = ui_data
        self.palette = palette
        self.device = device
        self.scale = min(width, height) / 1080.0

        # Palette color tensors on GPU (1, 1, 3)
        def _c(rgb: Tuple[int, int, int]) -> torch.Tensor:
            return torch.tensor(rgb, device=device, dtype=torch.float32).view(1, 1, 3)

        self.col_bg = _c(palette.bg_dark)
        self.col_panel = _c(palette.panel_bg)
        self.col_primary = _c(palette.primary)
        self.col_secondary = _c(palette.secondary)
        self.col_highlight = _c(palette.highlight)
        self.col_warning = _c(palette.warning)
        self.col_danger = _c(palette.danger)
        self.col_text = _c(palette.text_main)

        self.spec_lut = build_spectrogram_lut(palette, device=device)
        self.frame_gpu = torch.empty((height, width, 3), device=device, dtype=torch.float32)

        # Precompute coordinate grids for each panel on GPU
        self.rects: Dict[str, Rect] = ui_data["inner_rects"]

        # 1. Oscilloscope & Waveform Trace vertical grids
        osc_r = self.rects["oscilloscope"]
        self.osc_y = torch.arange(osc_r.h, device=device, dtype=torch.float32).view(osc_r.h, 1)

        tr_r = self.rects["waveform_trace"]
        self.tr_y = torch.arange(tr_r.h, device=device, dtype=torch.float32).view(tr_r.h, 1)
        self.prev_single_cycle = np.zeros(tr_r.w, dtype=np.float32)
        self.prev_freq_hz = 110.0

        # 2. Waveform Meter coordinate grids
        wm_r = self.rects["waveform_meter"]
        self.wm_y = torch.arange(wm_r.h, device=device, dtype=torch.float32).view(wm_r.h, 1)
        self.wm_x = torch.arange(wm_r.w, device=device, dtype=torch.float32).view(1, wm_r.w)
        # Window covers [-4.48s .. +1.92s] around playhead at 0.70 * width
        env_rate = features["env_rate"]
        rel_secs = (torch.arange(wm_r.w, device=device, dtype=torch.float32) / wm_r.w - 0.70) * 6.4
        self.wm_rel_idx = (rel_secs * env_rate).to(torch.long)

        # 3. Spectrum Analyzer vertical grid
        sp_r = self.rects["spectrum"]
        self.sp_y_norm = 1.0 - (torch.arange(sp_r.h, device=device, dtype=torch.float32).view(sp_r.h, 1) / max(1, sp_r.h - 1))
        self.sp_y_px = torch.arange(sp_r.h, device=device, dtype=torch.float32).view(sp_r.h, 1)
        self.sp_x_px = torch.arange(sp_r.w, device=device, dtype=torch.float32).view(1, sp_r.w)

        # 4. Stereometer / Goniometer Polar Coordinate Maps on GPU
        st_r = self.rects["stereometer"]
        smeta = ui_data["stereo_meta"]
        sy = torch.arange(st_r.h, device=device, dtype=torch.float32).view(st_r.h, 1)
        sx = torch.arange(st_r.w, device=device, dtype=torch.float32).view(1, st_r.w)
        dx = sx - smeta["fan_cx_local"]
        dy = smeta["fan_cy_local"] - sy  # Positive upward into semi-circle
        self.st_r_norm = torch.sqrt(dx * dx + dy * dy) / max(1.0, float(smeta["fan_rad"]))
        self.st_theta = torch.atan2(dx, torch.clamp(dy, min=1e-5))  # [-pi/2 .. +pi/2]
        self.st_in_fan = (dy >= 0.0) & (self.st_r_norm <= 1.0)
        self.st_bin_idx = torch.clamp(
            ((self.st_theta + (math.pi / 2.0)) / math.pi * 89.99).to(torch.long), 0, 89
        )
        self.gonio_persist = torch.zeros((st_r.h, st_r.w), device=device, dtype=torch.float32)
        self.fan_bins_smooth = torch.zeros(90, device=device, dtype=torch.float32)
        self.corr_smooth = 1.0

        # 5. YUV420P Pre-allocated Output Tensor on GPU
        self.yuv_flat = torch.empty((height * width * 3) // 2, device=device, dtype=torch.uint8)

    def _draw_segment_curve(
        self,
        panel: "torch.Tensor",
        y_pts: "torch.Tensor",
        y_grid: "torch.Tensor",
        color: "torch.Tensor",
        core_width: float = 1.4,
        glow_width: float = 6.0,
        glow_strength: float = 0.55,
    ) -> None:
        """Vectorized anti-aliased continuous line + phosphor glow shader on GPU."""
        import torch

        y_next = torch.cat([y_pts[1:], y_pts[-1:]], dim=0)
        y_min = torch.minimum(y_pts, y_next).view(1, -1)
        y_max = torch.maximum(y_pts, y_next).view(1, -1)
        dist = torch.clamp(torch.maximum(y_min - y_grid, y_grid - y_max), min=0.0)

        core = torch.exp(-(dist * dist) / max(0.25, (core_width * self.scale) ** 2))
        glow = torch.exp(-dist / max(0.5, glow_width * self.scale)) * glow_strength
        alpha = torch.clamp(core + glow, 0.0, 1.0).unsqueeze(-1)
        panel.copy_(panel * (1.0 - alpha) + color * alpha)

    def _draw_gpu_text(
        self,
        canvas: "torch.Tensor",
        text: str,
        x: int,
        y: int,
        color_rgb: Tuple[int, int, int],
        large: bool = True,
    ) -> None:
        """Blits pre-rendered bitmap glyphs directly onto a GPU tensor slice."""
        import torch

        glyphs = self.ui["glyphs_lg"] if large else self.ui["glyphs_sm"]
        gw = self.ui["gw_lg"] if large else self.ui["gw_sm"]
        gh = self.ui["gh_lg"] if large else self.ui["gh_sm"]
        col = torch.tensor(color_rgb, device=self.device, dtype=torch.float32).view(1, 1, 3)

        H, W, _ = canvas.shape
        if y < 0 or y + gh > H:
            return
        cx = x
        for ch in text:
            if cx + gw > W:
                break
            if cx >= 0 and ch in glyphs:
                mask = glyphs[ch].unsqueeze(-1)
                sub = canvas[y : y + gh, cx : cx + gw]
                sub.copy_(sub * (1.0 - mask) + col * mask)
            cx += gw

    def _render_oscilloscope(self, center_sample: int) -> None:
        """1. Oscilloscope: Displays the raw real-time electrical waveform shape."""
        import numpy as np
        import torch

        r = self.rects["oscilloscope"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]

        win_len = int(0.024 * SAMPLE_RATE)  # 24ms real-time window
        s0 = max(0, center_sample - win_len // 2)
        s1 = min(self.n_samples, s0 + win_len)
        seg_l = self.stereo_np[0, s0:s1]
        seg_r = self.stereo_np[1, s0:s1]
        if len(seg_l) < 16:
            return

        x_src = np.linspace(0, 1, len(seg_l), dtype=np.float32)
        x_dst = np.linspace(0, 1, r.w, dtype=np.float32)
        l_res = torch.from_numpy(np.interp(x_dst, x_src, seg_l)).to(self.device, dtype=torch.float32)
        r_res = torch.from_numpy(np.interp(x_dst, x_src, seg_r)).to(self.device, dtype=torch.float32)
        mid_res = 0.5 * (l_res + r_res)

        cy = r.h * 0.5
        amp = r.h * 0.43
        y_l = torch.clamp(cy - l_res * amp, 1.0, r.h - 2.0)
        y_r = torch.clamp(cy - r_res * amp, 1.0, r.h - 2.0)
        y_m = torch.clamp(cy - mid_res * amp, 1.0, r.h - 2.0)

        self._draw_segment_curve(panel, y_l, self.osc_y, self.col_secondary, core_width=1.0, glow_width=3.5, glow_strength=0.30)
        self._draw_segment_curve(panel, y_r, self.osc_y, self.col_highlight, core_width=1.0, glow_width=3.5, glow_strength=0.25)
        self._draw_segment_curve(panel, y_m, self.osc_y, self.col_primary, core_width=1.6, glow_width=6.5, glow_strength=0.60)

    def _render_waveform_trace(self, center_sample: int) -> None:
        """2. Waveform Trace: Single-cycle / pitch-stabilized view of the signal's period."""
        import numpy as np
        import torch

        r = self.rects["waveform_trace"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]

        win_len = 2048
        s0 = max(0, center_sample - win_len // 2)
        s1 = min(self.n_samples, s0 + win_len)
        seg = self.mid_np[s0:s1]
        if len(seg) < 512:
            return

        # Autocorrelation pitch period detection (40 Hz to 1000 Hz)
        windowed = seg * np.hanning(len(seg))
        fft_spec = np.fft.rfft(windowed)
        autocorr = np.fft.irfft(np.abs(fft_spec) ** 2)
        min_lag = int(SAMPLE_RATE / 1000.0)  # 48 samples
        max_lag = min(len(autocorr) // 2, int(SAMPLE_RATE / 40.0))  # 1200 samples

        period = 240
        if max_lag > min_lag:
            lag_slice = autocorr[min_lag:max_lag]
            peak_lag = int(np.argmax(lag_slice)) + min_lag
            if autocorr[0] > 1e-6 and (autocorr[peak_lag] / autocorr[0]) > 0.12:
                period = peak_lag
                self.prev_freq_hz = 0.80 * self.prev_freq_hz + 0.20 * (SAMPLE_RATE / float(period))

        # Find rising zero-crossing near window midpoint for phase stabilization
        mid_idx = len(seg) // 2
        search_start = max(0, mid_idx - period)
        search_end = min(len(seg) - 2 * period - 2, mid_idx + period)
        zc_idx = search_start
        if search_end > search_start:
            sub = seg[search_start:search_end]
            crossings = np.where((sub[:-1] <= 0.0) & (sub[1:] > 0.0))[0]
            if len(crossings) > 0:
                best = crossings[np.argmin(np.abs((search_start + crossings) - mid_idx))]
                zc_idx = search_start + int(best)

        # Extract 2 stabilized fundamental cycles
        cycle_len = max(32, min(len(seg) - zc_idx, 2 * period))
        cycle_seg = seg[zc_idx : zc_idx + cycle_len]
        peak_val = max(float(np.max(np.abs(cycle_seg))), 0.04)
        norm_cycle = np.clip(cycle_seg / max(0.25, peak_val), -1.0, 1.0)

        x_src = np.linspace(0, 1, len(norm_cycle), dtype=np.float32)
        x_dst = np.linspace(0, 1, r.w, dtype=np.float32)
        res_cycle = np.interp(x_dst, x_src, norm_cycle).astype(np.float32)
        smooth_cycle = 0.62 * self.prev_single_cycle + 0.38 * res_cycle
        self.prev_single_cycle = smooth_cycle

        cy = r.h * 0.5
        amp = r.h * 0.40
        y_pts = torch.clamp(
            cy - torch.from_numpy(smooth_cycle).to(self.device, dtype=torch.float32) * amp,
            2.0,
            r.h - 3.0,
        )

        # Subtle shaded fill between trace and center zero line
        y_top = torch.minimum(y_pts.view(1, -1), torch.tensor(cy, device=self.device))
        y_bot = torch.maximum(y_pts.view(1, -1), torch.tensor(cy, device=self.device))
        fill_mask = ((self.tr_y >= y_top) & (self.tr_y <= y_bot)).unsqueeze(-1).to(torch.float32) * 0.22
        panel.copy_(panel * (1.0 - fill_mask) + self.col_secondary * fill_mask)

        self._draw_segment_curve(panel, y_pts, self.tr_y, self.col_highlight, core_width=1.6, glow_width=6.0, glow_strength=0.60)

        # Live fundamental frequency & period readout
        period_ms = 1000.0 / max(20.0, self.prev_freq_hz)
        lbl = f"{int(round(self.prev_freq_hz)):3d}Hz {period_ms:4.1f}ms"
        self._draw_gpu_text(panel, lbl, int(6 * self.scale), int(6 * self.scale), self.palette.text_main, large=False)

    def _render_waveform_meter(self, center_sample: int) -> None:
        """3. Waveform Meter: Scrolling timeline of amplitude envelope, transients & decay."""
        import torch

        r = self.rects["waveform_meter"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]

        env_rate = self.features["env_rate"]
        center_env_idx = int(center_sample * env_rate / SAMPLE_RATE)
        idx = self.wm_rel_idx + center_env_idx
        n_env = self.features["env_max"].shape[0]
        valid = (idx >= 0) & (idx < n_env)
        idx_clamped = torch.clamp(idx, 0, n_env - 1)

        vmax = torch.where(valid, self.features["env_max"][idx_clamped], torch.zeros_like(idx, dtype=torch.float32))
        vmin = torch.where(valid, self.features["env_min"][idx_clamped], torch.zeros_like(idx, dtype=torch.float32))
        vrms = torch.where(valid, self.features["env_rms"][idx_clamped], torch.zeros_like(idx, dtype=torch.float32))
        vtrans = torch.where(valid, self.features["env_transient"][idx_clamped], torch.zeros_like(idx, dtype=torch.float32))

        cy = r.h * 0.5
        amp = r.h * 0.46
        y_top_pk = cy - vmax.view(1, -1) * amp
        y_bot_pk = cy - vmin.view(1, -1) * amp
        y_top_rms = cy - vrms.view(1, -1) * amp
        y_bot_rms = cy + vrms.view(1, -1) * amp

        in_peak = ((self.wm_y >= y_top_pk) & (self.wm_y <= y_bot_pk)).unsqueeze(-1).to(torch.float32)
        in_rms = ((self.wm_y >= y_top_rms) & (self.wm_y <= y_bot_rms)).unsqueeze(-1).to(torch.float32)

        trans_w = vtrans.view(1, -1, 1)
        peak_col = self.col_secondary * (1.0 - trans_w) + self.col_highlight * trans_w
        rms_col = self.col_primary * 0.85 + self.col_highlight * 0.15

        # Dim upcoming audio to the right of the playhead (70% mark)
        playhead_px = int(r.w * 0.70)
        future_dim = torch.where(self.wm_x <= playhead_px, 1.0, 0.42).unsqueeze(-1)

        out = panel * (1.0 - in_peak * 0.85) + peak_col * (in_peak * 0.85 * future_dim)
        out = out * (1.0 - in_rms * 0.90) + rms_col * (in_rms * 0.90 * future_dim)
        panel.copy_(out)

    def _render_spectrum_analyzer(self, frame_idx: int) -> None:
        """4. Spectrum Analyzer: Graphs volume level across the audible frequency spectrum."""
        import torch

        r = self.rects["spectrum"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]

        spec_val = self.features["spec_smooth"][frame_idx]  # (w,)
        peak_val = self.features["spec_peak"][frame_idx]    # (w,)

        # Filled gradient under the spectrum curve with subtle RTA bar texture
        under_mask = (self.sp_y_norm <= spec_val.view(1, -1)).unsqueeze(-1).to(torch.float32)
        v_grad = self.sp_y_norm.unsqueeze(-1)
        bar_tex = torch.where((self.sp_x_px % 4) == 0, 0.75, 1.0).unsqueeze(-1)
        fill_col = (
            self.col_secondary * (1.0 - v_grad) + self.col_primary * v_grad
        ) * (0.28 + 0.62 * v_grad) * bar_tex

        panel.copy_(panel * (1.0 - under_mask * 0.82) + fill_col * (under_mask * 0.82))

        # Glowing top spectrum curve + floating peak-hold trace
        y_spec_px = torch.clamp((1.0 - spec_val) * (r.h - 1), 1.0, r.h - 2.0)
        y_peak_px = torch.clamp((1.0 - peak_val) * (r.h - 1), 1.0, r.h - 2.0)

        self._draw_segment_curve(panel, y_peak_px, self.sp_y_px, self.col_highlight, core_width=1.0, glow_width=2.5, glow_strength=0.35)
        self._draw_segment_curve(panel, y_spec_px, self.sp_y_px, self.col_primary, core_width=1.7, glow_width=7.0, glow_strength=0.65)

    def _render_spectrogram(self, frame_idx: int) -> None:
        """5. Spectrogram: Displays frequency intensity over time as a vertical heatmap."""
        import torch

        r = self.rects["spectrogram"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]

        # Row 0 (top) is current frame_idx, flowing downward over r.h frames of history
        row_offsets = torch.arange(r.h, device=self.device, dtype=torch.long)
        hist_indices = frame_idx - row_offsets
        valid_rows = (hist_indices >= 0).view(r.h, 1)
        clamped_indices = torch.clamp(hist_indices, 0, self.features["n_frames"] - 1)

        spec_slice = self.features["spec_raw_norm"][clamped_indices]  # (r.h, r.w)
        spec_slice = torch.where(valid_rows, spec_slice, torch.zeros_like(spec_slice))

        lut_idx = torch.clamp((spec_slice * 255.0).to(torch.long), 0, 255)
        heatmap_rgb = self.spec_lut[lut_idx].to(torch.float32)
        panel.copy_(heatmap_rgb)

    def _render_stereometer(self, center_sample: int) -> None:
        """6. Stereometer / Goniometer / Phase Scope: Semi-circular fan display + phase correlation."""
        import numpy as np
        import torch

        r = self.rects["stereometer"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]
        smeta = self.ui["stereo_meta"]

        win_len = 2048
        s0 = max(0, center_sample - win_len // 2)
        s1 = min(self.n_samples, s0 + win_len)
        seg_l = self.stereo_np[0, s0:s1]
        seg_r = self.stereo_np[1, s0:s1]
        if len(seg_l) < 16:
            return

        l_gpu = torch.from_numpy(seg_l).to(self.device, dtype=torch.float32)
        r_gpu = torch.from_numpy(seg_r).to(self.device, dtype=torch.float32)

        # Phase Correlation (-1.0 to +1.0)
        num = torch.sum(l_gpu * r_gpu)
        den = torch.sqrt(torch.sum(l_gpu * l_gpu) * torch.sum(r_gpu * r_gpu) + 1e-9)
        corr_val = float(torch.clamp(num / den, -1.0, 1.0).item()) if float(den.item()) > 1e-7 else 1.0
        self.corr_smooth = 0.82 * self.corr_smooth + 0.18 * corr_val

        # Convert (L, R) to Mid/Side and map to Upper Semi-Circle Polar Coordinates
        # Angle = 0 (Center Mono), -pi/4 (Pure Left), +pi/4 (Pure Right), +-pi/2 (Out-of-Phase Side)
        mid = (l_gpu + r_gpu) * 0.70710678
        side = (r_gpu - l_gpu) * 0.70710678
        mag = torch.sqrt(mid * mid + side * side)
        max_mag = torch.clamp(torch.max(mag), min=0.12)
        r_norm_pts = torch.clamp((mag / max_mag) ** 0.75, 0.0, 0.98)
        theta_pts = torch.atan2(side, torch.abs(mid) + 1e-6)  # [-pi/2 .. +pi/2]

        # 1) Semi-Circular Polar Fan Wedges (90 angular bins)
        bin_pts = torch.clamp(((theta_pts + (math.pi / 2.0)) / math.pi * 89.99).to(torch.long), 0, 89)
        bin_energy = torch.zeros(90, device=self.device, dtype=torch.float32)
        bin_energy.scatter_reduce_(0, bin_pts, r_norm_pts, reduce="amax", include_self=True)
        self.fan_bins_smooth = torch.maximum(bin_energy, self.fan_bins_smooth * 0.84)

        pixel_wedge_r = self.fan_bins_smooth[self.st_bin_idx]
        fan_active = self.st_in_fan & (self.st_r_norm <= pixel_wedge_r) & (pixel_wedge_r > 0.03)
        fan_intensity = torch.where(
            fan_active,
            (1.0 - 0.35 * (self.st_r_norm / torch.clamp(pixel_wedge_r, min=0.05))),
            torch.zeros_like(self.st_r_norm),
        ).unsqueeze(-1)

        angle_ratio = (torch.abs(self.st_theta) / (math.pi * 0.5)).unsqueeze(-1)
        fan_col = self.col_primary * (1.0 - angle_ratio) + self.col_secondary * angle_ratio
        panel.copy_(panel * (1.0 - fan_intensity * 0.52) + fan_col * (fan_intensity * 0.52))

        # 2) High-Density Lissajous / Goniometer Scatter Cloud inside Semi-Circle
        fan_rad = float(smeta["fan_rad"])
        px = (smeta["fan_cx_local"] + r_norm_pts * fan_rad * torch.sin(theta_pts)).to(torch.long)
        py = (smeta["fan_cy_local"] - r_norm_pts * fan_rad * torch.cos(theta_pts)).to(torch.long)
        valid_pts = (px >= 1) & (px < r.w - 1) & (py >= 1) & (py < r.h - 1)
        px = px[valid_pts]
        py = py[valid_pts]

        self.gonio_persist.mul_(0.80)
        if px.numel() > 0:
            flat_idx = py * r.w + px
            ones = torch.full_like(flat_idx, 0.35, dtype=torch.float32)
            self.gonio_persist.view(-1).scatter_add_(0, flat_idx, ones)
            self.gonio_persist.clamp_(0.0, 1.0)

        # 3x3 glow pooling for phosphor persistence
        glow_map = torch.nn.functional.avg_pool2d(
            self.gonio_persist.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0).squeeze(0)
        gonio_alpha = torch.clamp(self.gonio_persist + glow_map * 1.4, 0.0, 1.0).unsqueeze(-1)
        gonio_col = self.col_highlight * 0.65 + self.col_primary * 0.35
        panel.copy_(panel * (1.0 - gonio_alpha) + gonio_col * gonio_alpha)

        # 3) Phase Correlation Meter Bar (-1 .. 0 .. +1)
        cy0 = smeta["corr_y_local"] + 2
        cy1 = smeta["corr_y_local"] + smeta["corr_h"] - 2
        cx_zero = smeta["corr_x0_local"] + smeta["corr_w"] // 2
        cx_val = smeta["corr_x0_local"] + int((self.corr_smooth + 1.0) * 0.5 * smeta["corr_w"])

        x_left = min(cx_zero, cx_val)
        x_right = max(cx_zero + 2, cx_val)
        bar_col = self.col_primary if self.corr_smooth >= 0.0 else self.col_danger
        if cy1 > cy0 and x_right > x_left:
            panel[cy0:cy1, x_left:x_right] = bar_col

    def _render_lufs_meter(self, frame_idx: int) -> None:
        """7. Peak & LUFS Meter: Vertical level indicators + real-time numerical loudness readout."""
        import torch

        r = self.rects["lufs_meter"]
        panel = self.frame_gpu[r.y : r.y + r.h, r.x : r.x + r.w]
        lmeta = self.ui["lufs_meta"]

        m_lufs = float(self.features["momentary_lufs"][frame_idx])
        s_lufs = float(self.features["short_term_lufs"][frame_idx])
        pk_l = float(self.features["peak_l_db"][frame_idx])
        pk_r = float(self.features["peak_r_db"][frame_idx])
        hold_l = float(self.features["hold_l_db"][frame_idx])
        hold_r = float(self.features["hold_r_db"][frame_idx])
        max_pk = max(pk_l, pk_r)

        # 1) Live Digital Numerical Readout (-14.1 LU & Peak dB)
        lufs_disp = max(-60.0, min(0.0, s_lufs if s_lufs > -55.0 else m_lufs))
        lufs_str = f"{lufs_disp:+5.1f} LU"
        pk_str = f"PK {max(-60.0, min(0.0, max_pk)):+5.1f}dB"

        pad_x = int(8 * self.scale)
        pad_y1 = int(8 * self.scale)
        pad_y2 = pad_y1 + self.ui["gh_lg"] + int(4 * self.scale)
        self._draw_gpu_text(panel, lufs_str, pad_x, pad_y1, self.palette.highlight, large=True)
        pk_color = self.palette.danger if max_pk > -1.0 else self.palette.text_main
        self._draw_gpu_text(panel, pk_str, pad_x, pad_y2, pk_color, large=False)

        # 2) Vertical Bar Meters (L Peak, R Peak, Momentary LUFS, Short-Term LUFS)
        by0 = lmeta["bars_y0_local"]
        bh = lmeta["bars_h"]
        bx0 = lmeta["bars_x0_local"]
        bw = lmeta["bars_w"]
        col_step = bw / 4.0
        bar_w = max(4, int(col_step * 0.62))

        values_db = [pk_l, pk_r, m_lufs, s_lufs]
        holds_db = [hold_l, hold_r, None, None]

        y_rows = torch.arange(bh, device=self.device, dtype=torch.float32).view(bh, 1, 1)
        row_db = -(y_rows / max(1.0, float(bh))) * 60.0  # 0 dB at top, -60 dB at bottom
        led_gap = ((torch.arange(bh, device=self.device) % max(3, int(4 * self.scale))) != 0).view(bh, 1, 1).to(torch.float32)

        # Color zones along vertical bar: Primary -> Secondary -> Warning (-9 dB) -> Danger (-1.5 dB)
        t_norm = 1.0 - (y_rows / max(1.0, float(bh)))
        base_bar_col = self.col_secondary * (1.0 - t_norm) + self.col_primary * t_norm
        bar_col = torch.where(row_db > -9.0, self.col_warning, base_bar_col)
        bar_col = torch.where(row_db > -1.5, self.col_danger, bar_col)

        for idx, val_db in enumerate(values_db):
            cx0 = bx0 + int(idx * col_step + (col_step - bar_w) * 0.5)
            cx1 = min(r.w, cx0 + bar_w)
            if cx1 <= cx0:
                continue

            bar_slice = panel[by0 : by0 + bh, cx0:cx1]
            active = (row_db <= max(-60.0, min(0.0, val_db))).to(torch.float32) * led_gap
            dim_bg = bar_col * 0.12 * led_gap
            bar_slice.copy_(bar_slice * (1.0 - led_gap * 0.85) + dim_bg * (1.0 - active) + bar_col * active)

            # Floating Peak-Hold Indicator Tick
            h_db = holds_db[idx]
            if h_db is not None and h_db > -59.0:
                hy = int(max(0.0, min(1.0, abs(min(0.0, h_db)) / 60.0)) * (bh - 3))
                bar_slice[hy : hy + max(2, int(3 * self.scale)), :] = self.col_highlight

    def render_frame_to_yuv420p(self, frame_idx: int) -> "torch.Tensor":
        """Renders all 7 instruments for `frame_idx` and converts RGB -> YUV420P (BT.709) on GPU."""
        import torch

        # 1. Reset canvas from static background layer
        self.frame_gpu.copy_(self.ui["base_gpu"].to(torch.float32))
        center_sample = int(self.features["frame_centers"][frame_idx])

        # 2. Render all 7 dynamic audio visualizers on GPU
        self._render_waveform_meter(center_sample)
        self._render_oscilloscope(center_sample)
        self._render_waveform_trace(center_sample)
        self._render_spectrum_analyzer(frame_idx)
        self._render_spectrogram(frame_idx)
        self._render_stereometer(center_sample)
        self._render_lufs_meter(frame_idx)

        # 3. Alpha-composite static foreground overlay (borders, tick labels, crosshairs)
        fg_a = self.ui["fg_alpha_gpu"]
        self.frame_gpu.copy_(self.frame_gpu * (1.0 - fg_a) + self.ui["fg_rgb_gpu"] * fg_a)

        # 4. Fast GPU RGB -> YUV420P (BT.709 Limited/TV Range) conversion
        r = self.frame_gpu[:, :, 0]
        g = self.frame_gpu[:, :, 1]
        b = self.frame_gpu[:, :, 2]

        y_plane = torch.clamp(0.182586 * r + 0.614231 * g + 0.062007 * b + 16.0, 16.0, 235.0).to(torch.uint8)

        # 2x2 chroma subsampling for U and V planes
        r_sub = 0.25 * (r[0::2, 0::2] + r[1::2, 0::2] + r[0::2, 1::2] + r[1::2, 1::2])
        g_sub = 0.25 * (g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2])
        b_sub = 0.25 * (b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2])

        u_plane = torch.clamp(-0.100644 * r_sub - 0.338572 * g_sub + 0.439216 * b_sub + 128.0, 16.0, 240.0).to(torch.uint8)
        v_plane = torch.clamp(0.439216 * r_sub - 0.398942 * g_sub - 0.040274 * b_sub + 128.0, 16.0, 240.0).to(torch.uint8)

        n_y = self.height * self.width
        n_uv = n_y // 4
        self.yuv_flat[:n_y].copy_(y_plane.reshape(-1))
        self.yuv_flat[n_y : n_y + n_uv].copy_(u_plane.reshape(-1))
        self.yuv_flat[n_y + n_uv :].copy_(v_plane.reshape(-1))
        return self.yuv_flat


# =====================================================================
# 6. HARDWARE-ACCELERATED VIDEO ENCODING PIPELINE (NVENC AV1 / H.264)
# =====================================================================

def select_video_encoder(codec: str, width: int, height: int) -> Tuple[str, List[str]]:
    """
    Probes L40S NVENC hardware encoder (`av1_nvenc` or `h264_nvenc`) at target resolution,
    with automatic fallback to `libsvtav1` / `libx264` if needed.
    """
    codec_norm = codec.strip().upper()
    is_av1 = codec_norm == "AV1"

    nvenc_name = "av1_nvenc" if is_av1 else "h264_nvenc"
    test_cmd = [
        "ffmpeg",
        "-v", "error",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r=30:d=0.05",
        "-pix_fmt", "yuv420p",
        "-c:v", nvenc_name,
        "-f", "null",
        "-",
    ]
    res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        if is_av1:
            return nvenc_name, ["-preset", "p4", "-cq", "24", "-b:v", "0"]
        else:
            return nvenc_name, ["-preset", "p4", "-cq", "20", "-b:v", "0", "-profile:v", "high"]

    # Fallback to fast multi-threaded CPU encoding if NVENC is unavailable
    if is_av1:
        return "libsvtav1", ["-preset", "10", "-crf", "28"]
    else:
        return "libx264", ["-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"]


def render_visualizer_pipeline(
    audio_path: str,
    output_path: str,
    title: str,
    artist: str,
    cover_bytes: Optional[bytes],
    orientation: str,
    resolution: str,
    fps: int,
    codec: str,
) -> Dict[str, any]:
    """End-to-end GPU audio analysis, frame rendering, and FFmpeg video encoding pipeline."""
    import torch

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_map = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }
    base_w, base_h = res_map.get(resolution.lower(), (1920, 1080))
    if orientation.lower() in ("vertical", "mobile", "9:16", "portrait"):
        width, height = base_h, base_w
        orient_norm = "vertical"
    else:
        width, height = base_w, base_h
        orient_norm = "landscape"

    palette, cover_img = extract_palette_from_cover(cover_bytes)
    stereo_audio = decode_audio_stereo_48k(audio_path)
    layout = compute_layout(width, height, orient_norm)

    pad = max(4, int(6 * (min(width, height) / 1080.0)))
    spec_width = max(16, layout["spectrum"].w - 2 * pad)
    features = compute_audio_features(stereo_audio, fps=fps, spec_width=spec_width, device=device)

    ui_data = build_static_layers_and_glyphs(
        width=width,
        height=height,
        orientation=orient_norm,
        layout=layout,
        palette=palette,
        cover_img=cover_img,
        title=title,
        artist=artist,
        codec=codec,
        fps=fps,
        integrated_lufs=features["integrated_lufs"],
        device=device,
    )

    renderer = VisualizerGPURenderer(
        width=width,
        height=height,
        fps=fps,
        stereo_audio=stereo_audio,
        features=features,
        ui_data=ui_data,
        palette=palette,
        device=device,
    )

    encoder_name, encoder_flags = select_video_encoder(codec=codec, width=width, height=height)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", encoder_name,
        *encoder_flags,
        "-c:a", "aac",
        "-b:a", "320k",
        "-movflags", "+faststart",
        "-shortest",
        output_path,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n_frames = features["n_frames"]

    # Pinned host memory buffer for high-speed DMA transfer from L40S GPU to FFmpeg pipe
    host_buf = torch.empty(
        (height * width * 3) // 2,
        dtype=torch.uint8,
        pin_memory=(device.type == "cuda"),
    )

    try:
        for f_idx in range(n_frames):
            yuv_gpu = renderer.render_frame_to_yuv420p(f_idx)
            host_buf.copy_(yuv_gpu, non_blocking=False)
            try:
                proc.stdin.write( memoryview(host_buf.numpy()) )
            except BrokenPipeError:
                # FFmpeg finished encoding via the -shortest flag and closed the pipe
                break
        proc.stdin.close()
        stderr_bytes = proc.stderr.read()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg encoding failed ({encoder_name}): {stderr_bytes.decode('utf-8', errors='replace')}")
    finally:
        if proc.poll() is None:
            proc.kill()

    elapsed = time.time() - t0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": n_frames,
        "duration_sec": round(features["duration"], 2),
        "encoder": encoder_name,
        "integrated_lufs": round(features["integrated_lufs"], 2),
        "render_time_sec": round(elapsed, 2),
        "effective_fps": round(n_frames / max(0.01, elapsed), 1),
        "palette_primary_rgb": palette.primary,
        "palette_secondary_rgb": palette.secondary,
        "palette_highlight_rgb": palette.highlight,
    }


# =====================================================================
# 7. MODAL GPU FUNCTION + FASTAPI WEB APPLICATION
# =====================================================================

HTML_STUDIO_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>L40S GPU Music Visualizer Studio</title>
  <style>
    :root { --bg: #090c14; --panel: #111726; --border: #263452; --accent: #00e8ff; --text: #f0f6ff; }
    body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 32px; }
    .container { max-width: 780px; margin: 0 auto; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 28px; }
    h1 { margin-top: 0; font-size: 24px; color: var(--accent); }
    p.sub { color: #8da0c4; font-size: 14px; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    label { display: block; font-size: 12px; font-weight: 600; text-transform: uppercase; color: #9ab0d8; margin-bottom: 6px; }
    input, select { width: 100%; box-sizing: border-box; background: #090c14; border: 1px solid var(--border); color: var(--text); padding: 10px 12px; border-radius: 6px; margin-bottom: 16px; }
    button { width: 100%; background: linear-gradient(90deg, #00e8ff, #b942ff); color: #090c14; font-weight: 700; border: none; padding: 14px; border-radius: 8px; cursor: pointer; font-size: 15px; }
    button:disabled { opacity: 0.5; cursor: wait; }
    #status { margin-top: 18px; font-family: monospace; font-size: 13px; color: var(--accent); white-space: pre-wrap; }
    video { width: 100%; margin-top: 20px; border-radius: 8px; border: 1px solid var(--border); display: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>NVIDIA L40S Pro-Audio Music Visualizer</h1>
    <p class="sub">Oscilloscope • Stabilized Waveform Trace • Scrolling Envelope • ITU-R BS.1770-4 LUFS & Peak • Semi-Circular Stereometer • RTA Spectrum • Waterfall Spectrogram</p>
    <form id="vizForm">
      <div class="grid">
        <div><label>Track Title *</label><input type="text" name="title" value="Midnight Telemetry" required /></div>
        <div><label>Artist Name *</label><input type="text" name="artist" value="Subharmonic Labs" required /></div>
      </div>
      <div class="grid">
        <div><label>Audio File (WAV, MP3, FLAC, M4A) *</label><input type="file" name="audio_file" accept="audio/*" required /></div>
        <div><label>Album Cover Art (Optional - Extracts Palette)</label><input type="file" name="cover_art" accept="image/*" /></div>
      </div>
      <div class="grid">
        <div>
          <label>Orientation</label>
          <select name="orientation">
            <option value="landscape" selected>Landscape (16:9)</option>
            <option value="vertical">Vertical / Mobile (9:16)</option>
          </select>
        </div>
        <div>
          <label>Resolution</label>
          <select name="resolution">
            <option value="720p">720p HD</option>
            <option value="1080p" selected>1080p Full HD</option>
            <option value="4K">4K UHD (3840×2160)</option>
          </select>
        </div>
      </div>
      <div class="grid">
        <div>
          <label>Frame Rate (FPS)</label>
          <select name="fps">
            <option value="24">24 FPS</option>
            <option value="30" selected>30 FPS</option>
            <option value="60">60 FPS</option>
          </select>
        </div>
        <div>
          <label>Video Codec (NVENC Hardware Accelerated)</label>
          <select name="codec">
            <option value="H.264" selected>H.264 (AVC)</option>
            <option value="AV1">AV1</option>
          </select>
        </div>
      </div>
      <button type="submit" id="submitBtn">Render GPU Visualizer Video</button>
    </form>
    <div id="status"></div>
    <video id="previewVideo" controls></video>
  </div>
  <script>
    const form = document.getElementById('vizForm');
    const btn = document.getElementById('submitBtn');
    const status = document.getElementById('status');
    const video = document.getElementById('previewVideo');
    form.onsubmit = async (e) => {
      e.preventDefault();
      btn.disabled = true;
      video.style.display = 'none';
      status.textContent = 'Uploading & rendering on NVIDIA L40S GPU...';
      try {
        const fd = new FormData(form);
        const res = await fetch('/generate', { method: 'POST', body: fd });
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(errText);
        }
        const stats = res.headers.get('X-Render-Stats') || '';
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        video.src = url;
        video.style.display = 'block';
        status.textContent = 'Render Complete! ' + stats;
        const a = document.createElement('a');
        a.href = url;
        a.download = 'visualizer.mp4';
        a.click();
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
      } finally {
        btn.disabled = false;
      }
    };
  </script>
</body>
</html>
"""


@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from starlette.background import BackgroundTask

    web_app = FastAPI(
        title="L40S GPU Pro-Audio Music Visualizer API",
        description="Generates 720p/1080p/4K music visualizers at 24/30/60 FPS in AV1 or H.264 on NVIDIA L40S GPUs.",
        version="1.0.0",
    )

    @web_app.get("/", response_class=HTMLResponse)
    async def studio_ui():
        return HTML_STUDIO_UI

    @web_app.get("/health")
    async def health():
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return {"status": "ok", "gpu": gpu_name, "cuda_available": torch.cuda.is_available()}

    @web_app.post("/generate")
    async def generate_visualizer(
        audio_file: UploadFile = File(..., description="Input audio file (WAV, MP3, FLAC, M4A, OGG)"),
        title: str = Form(..., description="Song title"),
        artist: str = Form(..., description="Artist name"),
        cover_art: Optional[UploadFile] = File(None, description="Optional album cover art (PNG/JPEG/WebP)"),
        orientation: Literal["landscape", "vertical"] = Form("landscape", description="Video orientation: landscape (16:9) or vertical (9:16)"),
        resolution: Literal["720p", "1080p", "4K"] = Form("1080p", description="Output resolution: 720p, 1080p, or 4K"),
        fps: int = Form(30, description="Frame rate: 24, 30, or 60"),
        codec: Literal["AV1", "H.264"] = Form("H.264", description="Video codec: AV1 or H.264"),
    ):
        if fps not in (24, 30, 60):
            raise HTTPException(status_code=400, detail="FPS must be 24, 30, or 60.")
        if resolution not in ("720p", "1080p", "4K"):
            raise HTTPException(status_code=400, detail="Resolution must be '720p', '1080p', or '4K'.")
        if codec not in ("AV1", "H.264"):
            raise HTTPException(status_code=400, detail="Codec must be 'AV1' or 'H.264'.")

        workdir = tempfile.mkdtemp(prefix="viz_l40s_")
        try:
            ext = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
            audio_path = os.path.join(workdir, f"input_audio{ext}")
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)

            cover_bytes: Optional[bytes] = None
            if cover_art is not None and cover_art.filename:
                raw_cover = await cover_art.read()
                if len(raw_cover) > 0:
                    cover_bytes = raw_cover

            output_path = os.path.join(workdir, "visualizer_output.mp4")
            stats = render_visualizer_pipeline(
                audio_path=audio_path,
                output_path=output_path,
                title=title.strip() or "Untitled",
                artist=artist.strip() or "Unknown Artist",
                cover_bytes=cover_bytes,
                orientation=orientation,
                resolution=resolution,
                fps=fps,
                codec=codec,
            )

            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in f"{artist}_{title}").strip("_") or "visualizer"
            download_filename = f"{safe_name}_{resolution}_{fps}fps_{codec.replace('.', '')}.mp4"

            return FileResponse(
                path=output_path,
                media_type="video/mp4",
                filename=download_filename,
                headers={"X-Render-Stats": str(stats)},
                background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
            )
        except Exception as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return web_app


# =====================================================================
# 8. CLI LOCAL ENTRYPOINT FOR TERMINAL TESTING (`modal run`)
# =====================================================================

@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=3600,
)
def render_cli_job(
    audio_bytes: bytes,
    audio_filename: str,
    title: str,
    artist: str,
    cover_bytes: Optional[bytes],
    orientation: str,
    resolution: str,
    fps: int,
    codec: str,
) -> Tuple[bytes, Dict[str, any]]:
    workdir = tempfile.mkdtemp(prefix="viz_cli_")
    try:
        ext = os.path.splitext(audio_filename)[1] or ".wav"
        audio_path = os.path.join(workdir, f"input{ext}")
        output_path = os.path.join(workdir, "output.mp4")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        stats = render_visualizer_pipeline(
            audio_path=audio_path,
            output_path=output_path,
            title=title,
            artist=artist,
            cover_bytes=cover_bytes,
            orientation=orientation,
            resolution=resolution,
            fps=fps,
            codec=codec,
        )
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes, stats
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.local_entrypoint()
def main(
    audio: str,
    title: str = "Midnight Telemetry",
    artist: str = "Subharmonic Labs",
    cover: Optional[str] = None,
    orientation: str = "landscape",
    resolution: str = "1080p",
    fps: int = 30,
    codec: str = "H.264",
    out: str = "visualizer_output.mp4",
):
    with open(audio, "rb") as f:
        audio_bytes = f.read()
    cover_bytes = None
    if cover and os.path.exists(cover):
        with open(cover, "rb") as f:
            cover_bytes = f.read()

    print(f"Dispatching render job to Modal NVIDIA L40S GPU ({resolution} @ {fps}fps, {codec}, {orientation})...")
    video_bytes, stats = render_cli_job.remote(
        audio_bytes=audio_bytes,
        audio_filename=os.path.basename(audio),
        title=title,
        artist=artist,
        cover_bytes=cover_bytes,
        orientation=orientation,
        resolution=resolution,
        fps=fps,
        codec=codec,
    )
    with open(out, "wb") as f:
        f.write(video_bytes)
    print(f"Saved video to {out} | Stats: {stats}")