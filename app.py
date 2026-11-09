import colorsys
import hmac
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

app = modal.App("gpu-music-visualizer")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core", "ca-certificates")
    .pip_install(
        "torch==2.7.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "numpy==2.2.6",
        "scipy==1.15.3",
        "Pillow>=11.3,<13",
        "fastapi>=0.115,<1",
        "python-multipart>=0.0.20",
    )
    .env(
        {
            # Best effort only: this does not force Modal to expose NVENC.
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
            "OMP_NUM_THREADS": "8",
        }
    )
)

with image.imports():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from scipy.signal import sosfilt


ROOT = "/jobs"
SR = 48_000
MAX_SECONDS = 20 * 60
MAX_AUDIO_BYTES = 512 * 1024 * 1024
MAX_COVER_BYTES = 16 * 1024 * 1024

volume = modal.Volume.from_name("music-visualizer-files", create_if_missing=True)
states = modal.Dict.from_name("music-visualizer-jobs", create_if_missing=True)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


# ---------------------------------------------------------------------------
# Artwork and palette
# ---------------------------------------------------------------------------

def load_artwork(path):
    """Return an optional RGB cover and two readable, artwork-derived accents."""
    if path is None:
        return None, ((76, 213, 255), (211, 113, 255))

    Image.MAX_IMAGE_PIXELS = 20_000_000

    with Image.open(path) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError("Cover art exceeds 20 megapixels.")

        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (10, 14, 22, 255))
        background.alpha_composite(rgba)
        art = background.convert("RGB")

    small = art.copy()
    small.thumbnail((160, 160))
    quantized = small.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()

    candidates = []
    for count, index in quantized.getcolors():
        rgb = tuple(palette[index * 3:index * 3 + 3])
        hsv = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
        candidates.append((count, rgb, hsv))

    # Prefer visible colors over a dominant black border/background.
    visible = [c for c in candidates if max(c[1]) > 40]
    candidates = visible or candidates
    candidates.sort(key=lambda c: c[0] * (0.4 + c[2][1]), reverse=True)

    first = candidates[0]
    second = max(
        candidates,
        key=lambda c: (
            sum((a - b) ** 2 for a, b in zip(c[1], first[1]))
            * math.sqrt(c[0])
        ),
    )

    def brighten(candidate):
        h, s, v = candidate[2]
        # Preserve hue/saturation, including monochrome artwork.
        rgb = colorsys.hsv_to_rgb(h, s, max(v, 0.72))
        return tuple(round(c * 255) for c in rgb)

    accents = (brighten(first), brighten(second))
    art.thumbnail((1024, 1024))
    return art, accents


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def gather_audio(array, indices):
    """Zero-padded gathering for either [samples, channels] or [samples]."""
    valid = (indices >= 0) & (indices < array.shape[0])
    values = array[indices.clamp(0, array.shape[0] - 1)]
    if array.ndim == 2:
        valid = valid[..., None]
    return values * valid


def sample_audio(array, positions):
    """Linear interpolation, preserving raw PCM rather than an envelope."""
    lower = positions.floor().long()
    fraction = positions - lower
    if array.ndim == 2:
        fraction = fraction[..., None]
    return (
        gather_audio(array, lower) * (1 - fraction)
        + gather_audio(array, lower + 1) * fraction
    )


def momentary_lufs(pcm, sample_ends):
    """
    ITU-R BS.1770 K-weighting coefficients for 48 kHz.

    Stereo channel energies are summed, not averaged. The trailing window is
    400 ms, with silence before the beginning of the track. This is momentary
    LUFS, not gated integrated LUFS.
    """
    sos = np.array(
        [
            [
                1.53512485958697,
                -2.69169618940638,
                1.19839281085285,
                1.0,
                -1.69065929318241,
                0.73248077421585,
            ],
            [
                1.0,
                -2.0,
                1.0,
                1.0,
                -1.99004745483398,
                0.99007225036621,
            ],
        ],
        dtype=np.float64,
    )

    weighted = sosfilt(sos, pcm, axis=0)
    energy = np.einsum("ij,ij->i", weighted, weighted)
    del weighted

    cumulative = np.empty(len(energy) + 1, dtype=np.float64)
    cumulative[0] = 0
    np.cumsum(energy, out=cumulative[1:])

    window = round(0.400 * SR)
    hi = np.clip(sample_ends, 0, len(pcm))
    lo = np.clip(sample_ends - window, 0, len(pcm))
    mean_energy = (cumulative[hi] - cumulative[lo]) / window

    return (
        -0.691 + 10 * np.log10(np.maximum(mean_energy, 1e-12))
    ).astype(np.float32)


def analyze_audio(pcm, fps):
    """Precompute frame-aligned features; FFT work is batched on the GPU."""
    hop = SR // fps  # Exact for 24, 30 and 60 FPS.
    frame_count = (len(pcm) + hop - 1) // hop
    ends_cpu = np.arange(frame_count, dtype=np.int64) * hop + hop // 2

    lufs = momentary_lufs(pcm, ends_cpu)
    audio = torch.from_numpy(pcm).to("cuda")
    ends = torch.from_numpy(ends_cpu).to("cuda")

    # Sample peaks for each video-frame interval.
    padded = F.pad(audio, (0, 0, 0, frame_count * hop - len(pcm)))
    peaks = padded.reshape(frame_count, hop, 2).abs().amax(dim=1)

    # Sliding one-second peak-hold markers.
    held = F.max_pool1d(
        F.pad(peaks.T[None], (fps - 1, 0)),
        kernel_size=fps,
        stride=1,
    )[0].T

    # 5 ms peak envelope preserves short transients in the scrolling timeline.
    envelope_hop = 240
    envelope = audio.abs().amax(dim=1)
    envelope = F.pad(envelope, (0, (-len(pcm)) % envelope_hop))
    envelope = envelope.reshape(-1, envelope_hop).amax(dim=1)

    # Causal, stereo-power spectrum: anti-phase material does not cancel.
    nfft = 4096
    window = torch.hann_window(nfft, device="cuda")
    offsets = torch.arange(-nfft, 0, device="cuda")
    frequencies = torch.logspace(
        math.log10(20), math.log10(20_000), 1024, device="cuda"
    )
    bin_positions = frequencies * nfft / SR
    bin_lo = bin_positions.floor().long()
    bin_fraction = bin_positions - bin_lo

    spectrum = torch.empty((frame_count, 1024), device="cuda")
    normalization = (2 / window.sum()) ** 2

    for first in range(0, frame_count, 512):
        last = min(first + 512, frame_count)
        positions = ends[first:last, None] + offsets[None]
        samples = gather_audio(audio, positions)
        fft = torch.fft.rfft(samples * window[None, :, None], dim=1)
        power = (
            fft.real.square() + fft.imag.square()
        ).mean(dim=-1) * normalization

        log_power = (
            power[:, bin_lo] * (1 - bin_fraction)
            + power[:, bin_lo + 1] * bin_fraction
        )
        spectrum[first:last] = 10 * log_power.clamp_min(1e-12).log10()

    return {
        "audio": audio,
        "ends": ends,
        "frames": frame_count,
        "peak_db": 20 * peaks.clamp_min(1e-5).log10(),
        "hold_db": 20 * held.clamp_min(1e-5).log10(),
        "lufs": torch.from_numpy(lufs).to("cuda"),
        "envelope": envelope,
        "spectrum": spectrum,
    }


# ---------------------------------------------------------------------------
# GPU scene renderer
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, features, options, width, height, artwork, accents):
        self.f = features
        self.options = options
        self.width, self.height = width, height
        self.device = features["audio"].device
        portrait = options["orientation"] == "portrait"
        self.scale = width / (1080 if portrait else 1920)
        self.px = lambda value: round(value * self.scale)

        self.primary = torch.tensor(
            accents[0], dtype=torch.float32, device=self.device
        )
        self.secondary = torch.tensor(
            accents[1], dtype=torch.float32, device=self.device
        )
        self.ink = torch.tensor(
            [237, 243, 251], dtype=torch.float32, device=self.device
        )
        self.ink8 = self.ink.to(torch.uint8)
        self.glyphs = {}

        background_color = tuple(
            int(v) for v in np.array(accents[0]) * 0.025 + [4, 7, 13]
        )
        panel_color = tuple(
            int(v) for v in np.array(accents[0]) * 0.025 + [10, 14, 22]
        )
        grid_color = tuple(
            int(v) for v in np.array(accents[0]) * 0.08 + [18, 23, 32]
        )

        base = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(base)
        S = self.px
        white = (237, 243, 251)
        muted = (143, 158, 180)

        # Header and optional cover.
        cover_box = (S(40), S(32), S(184), S(176))
        if artwork is not None:
            cover = ImageOps.fit(artwork, (S(144), S(144)))
            mask = Image.new("L", cover.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, cover.width - 1, cover.height - 1),
                radius=S(14),
                fill=255,
            )
            base.paste(cover, (S(40), S(32)), mask)
        else:
            draw.rounded_rectangle(
                cover_box, radius=S(14), fill=panel_color
            )
            for radius in (58, 44, 28, 8):
                draw.ellipse(
                    (
                        S(112 - radius), S(104 - radius),
                        S(112 + radius), S(104 + radius),
                    ),
                    outline=accents[0],
                    width=max(1, S(2)),
                )

        title_font = self.font(38 if portrait else 52, bold=True)
        title = options["title"]
        available = width - S(248)
        while title and draw.textlength(title, font=title_font) > available:
            title = title[:-2].rstrip() + "…"

        draw.text((S(208), S(36)), title, font=title_font, fill=white)
        draw.text(
            (S(208), S(96 if portrait else 108)),
            options["artist"],
            font=self.font(26 if portrait else 28),
            fill=accents[0],
        )
        draw.text(
            (S(208), S(146)),
            (
                f"{width} × {height} · {options['fps']} FPS · "
                f"{options['codec'].upper()} · 48 kHz stereo"
            ),
            font=self.font(12 if portrait else 14),
            fill=muted,
        )

        if portrait:
            layout = {
                "scope": (40, 200, 1000, 210),
                "envelope": (40, 430, 1000, 148),
                "trace": (40, 598, 1000, 130),
                "spectrum": (40, 748, 1000, 206),
                "fan": (40, 974, 490, 340),
                "meters": (550, 974, 490, 340),
                "spectrogram": (40, 1334, 1000, 530),
            }
        else:
            layout = {
                "scope": (40, 200, 800, 240),
                "envelope": (40, 460, 800, 180),
                "trace": (40, 660, 800, 140),
                "spectrum": (40, 820, 800, 220),
                "fan": (860, 200, 480, 400),
                "meters": (860, 620, 480, 420),
                "spectrogram": (1360, 200, 520, 840),
            }

        labels = {
            "scope": ("OSCILLOSCOPE", "L / R · 40 ms"),
            "envelope": ("AMPLITUDE TIMELINE", "−3 s / NOW / +3 s"),
            "trace": ("ZOOM TRACE", "L · triggered · 5 ms"),
            "spectrum": ("SPECTRUM", "−90…0 dBFS · log Hz"),
            "fan": ("STEREO FAN", "200 ms · auto gain"),
            "meters": ("PEAK + LOUDNESS", "dBFS / LUFS-M"),
            "spectrogram": ("SPECTROGRAM ↑", "8 s · −90…0 dBFS"),
        }

        self.panels = {}
        for name, box in layout.items():
            x, y, w, h = (S(value) for value in box)
            draw.rounded_rectangle(
                (x, y, x + w, y + h),
                radius=S(14),
                fill=panel_color,
                outline=grid_color,
                width=max(1, S(1)),
            )
            title, caption = labels[name]
            draw.text(
                (x + S(18), y + S(11)),
                title, font=self.font(16, bold=True), fill=white,
            )
            caption_font = self.font(10)
            caption_width = draw.textlength(caption, font=caption_font)
            draw.text(
                (x + w - S(18) - caption_width, y + S(15)),
                caption, font=caption_font, fill=muted,
            )

            bottom = 26 if name in ("spectrum", "spectrogram") else 18
            self.panels[name] = (
                x + S(18), y + S(42),
                w - S(36), h - S(42 + bottom),
            )

        for name in ("scope", "envelope", "trace", "spectrum"):
            x, y, w, h = self.panels[name]
            for fraction in (0.25, 0.5, 0.75):
                yy = y + round(fraction * (h - 1))
                draw.line((x, yy, x + w - 1, yy), fill=grid_color)
            for fraction in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875):
                xx = x + round(fraction * (w - 1))
                draw.line((xx, y, xx, y + h - 1), fill=grid_color)

        for name in ("spectrum", "spectrogram"):
            x, y, w, h = self.panels[name]
            for frequency, text in (
                (20, "20"), (100, "100"), (1000, "1k"),
                (10_000, "10k"), (20_000, "20k"),
            ):
                position = math.log(frequency / 20) / math.log(1000)
                xx = x + position * (w - 1)
                draw.line((xx, y, xx, y + h - 1), fill=grid_color)
                draw.text(
                    (xx, y + h + S(3)), text,
                    anchor="mt", font=self.font(10), fill=muted,
                )

        self.make_fan(draw, accents, grid_color, muted)
        self.make_meters(draw, grid_color, muted)

        draw.text(
            (S(40), height - S(25)),
            "PCM scopes · 5 ms peak envelope · sample peak / 1 s hold · "
            "LUFS-M / 400 ms · stereo fan width = angular spread",
            font=self.font(10),
            fill=muted,
        )

        self.base = torch.from_numpy(np.array(base)).to(self.device)
        self.rows = {
            h: torch.arange(h, device=self.device)[None, :, None].float()
            for _, _, _, h in self.panels.values()
        }

        # Palette-derived spectrogram color lookup table.
        p, q = np.array(accents, dtype=np.float32)
        stops = np.array(
            [
                panel_color,
                p * 0.30,
                p,
                q,
                0.85 * np.full(3, 255) + 0.15 * (p + q) / 2,
            ],
            dtype=np.float32,
        )
        positions = np.linspace(0, len(stops) - 1, 256)
        lut = np.stack(
            [
                np.interp(positions, np.arange(len(stops)), stops[:, channel])
                for channel in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
        self.lut = torch.from_numpy(lut).to(self.device)

    def font(self, size, bold=False, mono=False):
        name = (
            "DejaVuSansMono.ttf" if mono
            else "DejaVuSans-Bold.ttf" if bold
            else "DejaVuSans.ttf"
        )
        return ImageFont.truetype(
            f"{FONT_DIR}/{name}", max(1, self.px(size))
        )

    def make_fan(self, draw, accents, grid_color, muted):
        x, y, w, h = self.panels["fan"]
        S = self.px
        cx, cy = w / 2, h - S(76)
        radius = min(w * 0.46, cy - S(18))
        self.fan_geometry = (cx, cy, radius)

        # A genuinely semi-circular, palette-colored background fan.
        for angle in range(-90, 90, 10):
            fraction = (angle + 95) / 180
            color = (
                np.array(accents[0]) * (1 - fraction)
                + np.array(accents[1]) * fraction
            )
            color = tuple(int(v) for v in color * 0.10 + [9, 12, 18])
            points = [(x + cx, y + cy)]
            for a in (angle, angle + 10):
                radians = math.radians(a)
                points.append(
                    (
                        x + cx + radius * math.sin(radians),
                        y + cy - radius * math.cos(radians),
                    )
                )
            draw.polygon(points, fill=color)

        for fraction in (0.33, 0.66, 1):
            r = radius * fraction
            draw.arc(
                (x + cx - r, y + cy - r, x + cx + r, y + cy + r),
                180, 360, fill=grid_color, width=max(1, S(1)),
            )

        for angle, label in ((-45, "L"), (0, "MONO"), (45, "R")):
            radians = math.radians(angle)
            xx = cx + radius * math.sin(radians)
            yy = cy - radius * math.cos(radians)
            draw.line((x + cx, y + cy, x + xx, y + yy), fill=grid_color)
            draw.text(
                (x + xx, y + yy - S(11)), label,
                anchor="mm", font=self.font(10), fill=muted,
            )

        self.stereo_bars = []
        for label, yy, left, right in (
            ("BAL", h - S(45), "L", "R"),
            ("PHASE", h - S(18), "−1", "+1"),
        ):
            bx, bw, bh = round(w * 0.26), round(w * 0.56), max(2, S(5))
            rect = (x + bx, y + yy, bw, bh)
            self.stereo_bars.append(rect)
            draw.rectangle(
                (rect[0], rect[1], rect[0] + bw, rect[1] + bh),
                fill=grid_color,
            )
            draw.text(
                (x, y + yy - S(5)), label,
                font=self.font(10), fill=muted,
            )
            draw.text(
                (x + bx - S(20), y + yy - S(5)), left,
                font=self.font(10), fill=muted,
            )
            draw.text(
                (x + bx + bw + S(5), y + yy - S(5)), right,
                font=self.font(10), fill=muted,
            )

        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        angle = torch.atan2(xx.float() - cx, cy - yy.float())
        fraction = (0.5 + angle / math.pi).clamp(0, 1)[..., None]
        self.fan_color = (
            self.primary * (1 - fraction) + self.secondary * fraction
        )

    def make_meters(self, draw, grid_color, muted):
        x, y, w, h = self.panels["meters"]
        S = self.px
        bar_height = h - S(60)
        self.meter_bars = []
        self.peak_number_centers = []

        for fraction, width_fraction, label in (
            (0.05, 0.10, "L / dBFS"),
            (0.25, 0.10, "R / dBFS"),
            (0.47, 0.08, "M"),
        ):
            bx, bw = x + round(w * fraction), round(w * width_fraction)
            rect = (bx, y, bw, bar_height)
            self.meter_bars.append(rect)
            draw.rectangle(
                (bx, y, bx + bw, y + bar_height), fill=(7, 11, 18)
            )
            for fraction_y in (0, 0.25, 0.5, 0.75, 1):
                yy = y + round((bar_height - 1) * fraction_y)
                draw.line((bx, yy, bx + bw, yy), fill=grid_color)
            draw.text(
                (bx + bw / 2, y + bar_height + S(5)),
                label, anchor="mt", font=self.font(10), fill=muted,
            )
            self.peak_number_centers.append((bx + bw // 2, y + h - S(23)))

        self.lufs_position = (x + round(w * 0.62), y + round(h * 0.23))
        lx, ly = self.lufs_position
        for offset, text in ((42, "LUFS (M)"), (62, "400 ms"), (94, "REF −14")):
            draw.text(
                (lx, ly + S(offset)), text,
                font=self.font(12), fill=muted,
            )

        # A reference marker, not a normalization operation.
        bx, by, bw, bh = self.meter_bars[2]
        reference_y = by + round((14 / 48) * (bh - 1))
        draw.line(
            (bx - S(3), reference_y, bx + bw + S(3), reference_y),
            fill=(212, 220, 232),
            width=max(1, S(1)),
        )

    def overlay(self, frames, rect, color, alpha):
        x, y, w, h = rect
        destination = frames[:, y:y + h, x:x + w]
        a = alpha if isinstance(alpha, (int, float)) else alpha[..., None]
        destination.copy_(
            (destination.float() * (1 - a) + color * a)
            .round().clamp(0, 255).to(torch.uint8)
        )

    def curve(self, frames, rect, positions_y, color, fill=False):
        _, _, w, h = rect
        rows = self.rows[h]

        previous = torch.cat((positions_y[:, :1], positions_y[:, :-1]), dim=1)
        following = torch.cat((positions_y[:, 1:], positions_y[:, -1:]), dim=1)

        # Column-wise rasterization connects steep adjacent segments.
        span = torch.maximum(
            (positions_y - previous).abs(),
            (following - positions_y).abs(),
        ) * 0.5
        distance = (
            (rows - positions_y[:, None]).abs() - span[:, None]
        ).clamp_min(0)

        alpha = (
            0.85 * torch.exp(-(distance / max(0.7, self.scale)) ** 2)
            + 0.20 * torch.exp(
                -(distance / max(1.5, 2.8 * self.scale)) ** 2
            )
        )
        if fill:
            alpha = alpha + 0.12 * (rows >= positions_y[:, None])
        self.overlay(frames, rect, color, alpha.clamp(0, 0.98))

    def atlas(self, size):
        if size not in self.glyphs:
            font = self.font(size, mono=True)
            cell_width = math.ceil(font.getlength("0"))
            ascent, descent = font.getmetrics()
            cell_height = ascent + descent

            cells = []
            for character in "0123456789-. ":
                cell = Image.new("L", (cell_width, cell_height))
                ImageDraw.Draw(cell).text(
                    (0, ascent), character,
                    font=font, fill=255, anchor="ls",
                )
                cells.append(np.array(cell))

            atlas = torch.from_numpy(np.stack(cells)).to(self.device).float() / 255
            self.glyphs[size] = (atlas, cell_width, cell_height)

        return self.glyphs[size]

    def number(self, frames, x, y, values, size=32, centered=False):
        """Five GPU-selected glyphs, e.g. '-14.1'; no CPU string formatting."""
        atlas, cw, ch = self.atlas(size)
        values = values.clamp(-99.9, 99.9)
        integer = (values.abs() * 10).round().long().clamp_max(999)
        leading = integer // 100
        leading = torch.where(leading == 0, 12, leading)

        indices = torch.stack(
            (
                torch.where(values < -0.05, 10, 12),
                leading,
                (integer // 10) % 10,
                torch.full_like(integer, 11),
                integer % 10,
            ),
            dim=1,
        )
        alpha = atlas[indices].permute(0, 2, 1, 3).reshape(-1, ch, 5 * cw)
        if centered:
            x -= (5 * cw) // 2
        self.overlay(
            frames, (int(x), int(y), 5 * cw, ch), self.ink, alpha
        )

    def render_fan(self, frames, frame_indices):
        rect = self.panels["fan"]
        _, _, w, h = rect
        cx, cy, radius = self.fan_geometry
        batch = len(frame_indices)

        offsets = torch.arange(-9600, 0, 4, device=self.device)
        positions = self.f["ends"][frame_indices, None] + offsets[None]
        stereo = gather_audio(self.f["audio"], positions)
        left, right = stereo.unbind(dim=-1)

        mid = (left + right) * 0.5
        side = (right - left) * 0.5
        magnitude = torch.sqrt(mid.square() + side.square())
        gain = magnitude.amax(dim=1, keepdim=True).clamp_min(0.025)

        # Fold the lower half-plane upward WITHOUT swapping L/R on negative
        # waveform half-cycles. Mono is vertical; anti-phase is horizontal.
        polarity = torch.where(mid >= 0, 1.0, -1.0)
        px = cx + side * polarity / gain * radius * 0.95
        py = cy - mid.abs() / gain * radius * 0.95
        px = px.clamp(0, w - 1.001)
        py = py.clamp(0, h - 1.001)

        ix, iy = px.floor().long(), py.floor().long()
        dx, dy = px - ix, py - iy
        persistence = torch.linspace(-3, 0, len(offsets), device=self.device).exp()
        weights = magnitude / gain * persistence[None]

        density = torch.zeros((batch, h * w), device=self.device)
        for ox, wx in ((0, 1 - dx), (1, dx)):
            for oy, wy in ((0, 1 - dy), (1, dy)):
                indices = (iy + oy) * w + ix + ox
                density.scatter_add_(1, indices, weights * wx * wy)

        density = density.reshape(batch, 1, h, w)
        core = F.avg_pool2d(density, 3, stride=1, padding=1)[:, 0] * 9
        halo = F.avg_pool2d(density, 9, stride=1, padding=4)[:, 0] * 81
        alpha = 1 - torch.exp(-(0.8 * core + 0.035 * halo))
        self.overlay(frames, rect, self.fan_color, alpha.clamp_max(0.97))

        power = stereo.square().sum(dim=1)
        rms = power.sqrt()
        balance = (rms[:, 1] - rms[:, 0]) / (rms.sum(dim=1) + 1e-12)
        correlation = (left * right).sum(dim=1) / (
            (power[:, 0] * power[:, 1]).sqrt() + 1e-12
        )
        audible = power.sum(dim=1) > 1e-10

        for bar, value in zip(self.stereo_bars, (balance, correlation)):
            _, _, bw, bh = bar
            xx = torch.arange(bw, device=self.device)[None, None]
            location = (value.clamp(-1, 1) + 1) * 0.5 * (bw - 1)
            marker = (
                (xx - location[:, None, None]).abs()
                <= max(1, 1.6 * self.scale)
            ).float()
            marker = marker.expand(-1, bh, -1) * audible[:, None, None]
            self.overlay(frames, bar, self.ink, marker)

    def render_meters(self, frames, frame_indices):
        peaks = self.f["peak_db"][frame_indices]
        holds = self.f["hold_db"][frame_indices]
        lufs = self.f["lufs"][frame_indices]

        for channel, rect in enumerate(self.meter_bars):
            _, _, w, h = rect
            value = peaks[:, channel] if channel < 2 else lufs
            floor = -60 if channel < 2 else -48
            normalized = ((value - floor) / -floor).clamp(0, 1)

            level = (
                1 - torch.arange(h, device=self.device).float() / max(1, h - 1)
            )[None, :, None]
            alpha = (
                (level <= normalized[:, None, None])
                & (normalized[:, None, None] > 0)
            ).float() * 0.93

            color = (
                self.primary * (1 - level[..., None])
                + self.secondary * level[..., None]
            )
            self.overlay(frames, rect, color, alpha)

            if channel < 2:
                held = ((holds[:, channel] + 60) / 60).clamp(0, 1)
                marker = (
                    (level - held[:, None, None]).abs() * h
                    <= max(1, self.scale)
                ).float()
                marker *= (holds[:, channel] > -60)[:, None, None]
                self.overlay(frames, rect, self.ink, marker)

                x, y, bw, _ = rect
                clipping = (value >= 0)[:, None, None].float()
                red = torch.tensor(
                    [255, 83, 91], device=self.device, dtype=torch.float32
                )
                self.overlay(
                    frames, (x, y, bw, max(1, self.px(3))), red, clipping
                )

                nx, ny = self.peak_number_centers[channel]
                self.number(
                    frames, nx, ny, value, size=15, centered=True
                )

        self.number(frames, *self.lufs_position, lufs, size=32)

    def render(self, frame_indices):
        batch = len(frame_indices)
        frames = self.base[None].expand(batch, -1, -1, -1).clone()
        ends = self.f["ends"][frame_indices].float()
        audio = self.f["audio"]

        # 1. Raw, dual-channel oscilloscope.
        rect = self.panels["scope"]
        _, _, w, h = rect
        positions = ends[:, None] + torch.linspace(
            -0.040 * SR, -1, w, device=self.device
        )[None]
        raw = sample_audio(audio, positions)
        for channel, color in ((0, self.primary), (1, self.secondary)):
            curve_y = (1 - raw[:, :, channel].clamp(-1, 1) * 0.90) * (h - 1) / 2
            self.curve(frames, rect, curve_y, color)

        # 2. Scrolling peak-amplitude timeline with a fixed central playhead.
        rect = self.panels["envelope"]
        x, y, w, h = rect
        positions = ends[:, None] / 240 + torch.linspace(
            -3 * SR / 240, 3 * SR / 240, w, device=self.device
        )[None]
        envelope = sample_audio(self.f["envelope"], positions).clamp(0, 1)
        distance_from_center = (self.rows[h] - (h - 1) / 2).abs()
        amplitude = envelope[:, None] * h * 0.46
        distance = (distance_from_center - amplitude).abs()
        alpha = (
            0.22 * (distance_from_center <= amplitude)
            + 0.75 * torch.exp(-(distance / max(0.8, self.scale)) ** 2)
        )
        future_fade = torch.where(
            torch.arange(w, device=self.device) < w // 2, 1.0, 0.35
        )
        alpha *= future_fade[None, None]
        alpha *= (envelope[:, None] > 1e-5)
        self.overlay(frames, rect, self.primary, alpha.clamp_max(0.95))
        frames[:, y:y + h, x + w // 2:x + w // 2 + max(1, self.px(1))] = self.ink8

        # 3. Separate 5 ms zoom trace, stabilized by a rising zero crossing.
        rect = self.panels["trace"]
        _, _, w, h = rect
        search_end = self.f["ends"][frame_indices] - round(0.005 * SR)
        offsets = torch.arange(-480, 0, device=self.device)
        search = gather_audio(
            audio, search_end[:, None] + offsets[None]
        )[:, :, 0]
        crossing = (search[:, :-1] <= 0) & (search[:, 1:] > 0)
        last_crossing = torch.where(
            crossing,
            torch.arange(1, 480, device=self.device)[None],
            -1,
        ).amax(dim=1)
        start = torch.where(
            last_crossing >= 0,
            search_end - 480 + last_crossing,
            search_end,
        )
        positions = start[:, None].float() + torch.linspace(
            0, 0.005 * SR - 1, w, device=self.device
        )[None]
        zoom = sample_audio(audio, positions)[:, :, 0]
        self.curve(
            frames, rect,
            (1 - zoom.clamp(-1, 1) * 0.90) * (h - 1) / 2,
            self.secondary,
        )

        # 4. Log-frequency spectrum, 20 Hz–20 kHz.
        rect = self.panels["spectrum"]
        _, _, w, h = rect
        spectrum = F.interpolate(
            self.f["spectrum"][frame_indices, None],
            size=w, mode="linear", align_corners=True,
        )[:, 0]
        level = ((spectrum + 90) / 90).clamp(0, 1)
        fraction = torch.linspace(0, 1, w, device=self.device)[None, None, :, None]
        color = self.primary * (1 - fraction) + self.secondary * fraction
        self.curve(
            frames, rect, (1 - level) * (h - 1), color, fill=True
        )

        # 5. Vertical waterfall: frequency across X, time down Y, newest below.
        rect = self.panels["spectrogram"]
        x, y, w, h = rect
        history = torch.linspace(
            -8 * self.options["fps"], 0, h, device=self.device
        )
        history_indices = frame_indices[:, None].float() + history[None]
        grid_y = (
            2 * history_indices / max(1, self.f["frames"] - 1) - 1
        )
        grid_x = torch.linspace(-1, 1, w, device=self.device)
        grid = torch.stack(
            (
                grid_x[None, None].expand(batch, h, w),
                grid_y[:, :, None].expand(batch, h, w),
            ),
            dim=-1,
        )
        history_db = F.grid_sample(
            self.f["spectrum"][None, None].expand(batch, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0]
        intensity = ((history_db + 90) / 90).clamp(0, 1).pow(0.85)
        heat = self.lut[(intensity * 255).long()]
        valid = (history_indices >= 0)[:, :, None].float()
        self.overlay(frames, rect, heat, valid * 0.96)
        frames[:, y + h - 1:y + h, x:x + w] = self.ink8

        # 6–7. Stereo fan and numerical peak/LUFS meters.
        self.render_fan(frames, frame_indices)
        self.render_meters(frames, frame_indices)

        # Thin track-progress line.
        line_height = max(1, self.px(3))
        xx = torch.linspace(0, 1, self.width, device=self.device)[None, None]
        progress = (frame_indices.float() + 1) / self.f["frames"]
        alpha = (xx <= progress[:, None, None]).float() * 0.85
        self.overlay(
            frames,
            (0, self.height - line_height, self.width, line_height),
            self.primary,
            alpha,
        )
        return frames


# ---------------------------------------------------------------------------
# GPU pixel conversion and encoding
# ---------------------------------------------------------------------------

def rgb_to_i420(rgb):
    """
    Batched GPU RGB -> limited-range BT.709 planar YUV420.

    This halves transfer/pipe bandwidth relative to RGB24 and avoids FFmpeg
    doing the full-resolution RGB-to-YUV conversion on the CPU.
    """
    batch = rgb.shape[0]
    full = rgb.float()
    r, g, b = full.unbind(dim=-1)

    y = (
        16 + (219 / 255) * (0.2126 * r + 0.7152 * g + 0.0722 * b)
    ).round().clamp(16, 235).to(torch.uint8)

    small = (
        full[:, 0::2, 0::2]
        + full[:, 1::2, 0::2]
        + full[:, 0::2, 1::2]
        + full[:, 1::2, 1::2]
    ) * 0.25
    r, g, b = small.unbind(dim=-1)

    u = (
        128 + (224 / 255) * (-0.114572 * r - 0.385428 * g + 0.5 * b)
    ).round().clamp(16, 240).to(torch.uint8)

    v = (
        128 + (224 / 255) * (0.5 * r - 0.454153 * g - 0.045847 * b)
    ).round().clamp(16, 240).to(torch.uint8)

    return torch.cat(
        (y.reshape(batch, -1), u.reshape(batch, -1), v.reshape(batch, -1)),
        dim=1,
    )


def choose_encoder(codec, width, height, fps, require_nvenc):
    name = "av1_nvenc" if codec == "av1" else "h264_nvenc"
    quality = "28" if codec == "av1" else "20"
    arguments = [
        "-c:v", name,
        "-preset", "p4",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", quality,
        "-b:v", "0",
    ]

    probe = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
            "-frames:v", "1",
            *arguments,
            "-pix_fmt", "yuv420p",
            "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=90,
    )

    if probe.returncode == 0:
        return name, arguments, None

    reason = probe.stderr.decode(errors="replace").strip()[-1500:]
    if require_nvenc:
        raise RuntimeError(f"NVENC is required but unavailable:\n{reason}")

    if codec == "h264":
        name = "libx264"
        arguments = [
            "-c:v", name, "-preset", "veryfast", "-crf", "19",
        ]
    else:
        name = "libsvtav1"
        arguments = [
            "-c:v", name,
            "-preset", "10",
            "-crf", "28",
            "-svtav1-params", "lp=8",
        ]

    encoders = subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if not re.search(rf"\b{re.escape(name)}\b", encoders):
        raise RuntimeError(
            f"Neither {codec} NVENC nor software encoder {name} is available.\n"
            f"NVENC error: {reason}"
        )

    return name, arguments, reason


def encode_video(scene, pcm_path, output_path, encoder_args, batch_size, report):
    width, height = scene.width, scene.height
    fps = scene.options["fps"]
    total_frames = scene.f["frames"]
    video_duration = total_frames / fps

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
        "-f", "rawvideo",
        "-pixel_format", "yuv420p",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-f", "f32le", "-ar", str(SR), "-ac", "2",
        "-i", str(pcm_path),
        "-map", "0:v:0", "-map", "1:a:0",
        *encoder_args,
        "-threads", "8",
        "-g", str(fps * 2),
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-chroma_sample_location", "center",
        "-c:a", "aac", "-b:a", "320k",
        "-af", "apad",
        "-t", f"{video_duration:.9f}",
        "-movflags", "+faststart",
        str(output_path),
    ]

    # Two pinned batches overlap GPU work, device-to-host copy, and pipe I/O.
    bytes_per_frame = width * height * 3 // 2
    buffers = [
        torch.empty(
            (batch_size, bytes_per_frame),
            dtype=torch.uint8,
            pin_memory=True,
        )
        for _ in range(2)
    ]

    free_slots = queue.Queue()
    free_slots.put(0)
    free_slots.put(1)
    ready = queue.Queue(maxsize=2)
    writer_errors = []
    copy_stream = torch.cuda.Stream()

    log_path = Path(output_path).with_suffix(".ffmpeg.log")
    with open(log_path, "w+b") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            bufsize=0,
        )

        def writer():
            while True:
                item = ready.get()
                if item is None:
                    return
                slot, count, event = item
                try:
                    event.synchronize()
                    if not writer_errors:
                        data = memoryview(buffers[slot][:count].numpy()).cast("B")
                        while len(data):
                            written = process.stdin.write(data)
                            if not written:
                                raise BrokenPipeError("FFmpeg closed its input.")
                            data = data[written:]
                except Exception as exc:
                    writer_errors.append(exc)
                finally:
                    # Continue draining after an error so the producer can exit.
                    free_slots.put(slot)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        problem = None
        last_report = 0.0

        try:
            for first in range(0, total_frames, batch_size):
                slot = free_slots.get()
                if writer_errors:
                    raise RuntimeError(str(writer_errors[0]))

                last = min(first + batch_size, total_frames)
                indices = torch.arange(first, last, device="cuda")
                packed = rgb_to_i420(scene.render(indices))

                compute_stream = torch.cuda.current_stream()
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_stream(compute_stream)
                    buffers[slot][:last - first].copy_(
                        packed, non_blocking=True
                    )
                    event = torch.cuda.Event()
                    event.record(copy_stream)
                packed.record_stream(copy_stream)
                ready.put((slot, last - first, event))

                if time.monotonic() - last_report >= 2:
                    report("rendering", 0.10 + 0.88 * last / total_frames)
                    last_report = time.monotonic()

            report("finalizing", 0.98)
        except BaseException as exc:
            problem = exc
        finally:
            ready.put(None)
            thread.join()
            try:
                process.stdin.close()
            except BrokenPipeError as exc:
                writer_errors.append(exc)

            try:
                return_code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return_code = -1
                writer_errors.append(RuntimeError("FFmpeg finalization timed out."))

            log.seek(0)
            log_text = log.read().decode(errors="replace")[-5000:]

        if writer_errors or return_code:
            raise RuntimeError(
                f"Encoding failed: {writer_errors[:1]}\n{log_text}"
            )
        if problem is not None:
            raise problem


# ---------------------------------------------------------------------------
# GPU job
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    max_containers=2,
    scaledown_window=60,
    volumes={ROOT: volume},
)
def render_video(job_id: str, options: dict):
    started = time.monotonic()
    metadata = {"job_id": job_id, **options}

    def report(stage, progress):
        states.put(
            job_id,
            {
                **metadata,
                "state": "running",
                "stage": stage,
                "progress": round(float(progress), 4),
            },
        )

    try:
        volume.reload()
        folder = Path(ROOT) / job_id
        report("decoding", 0)

        height = {"720p": 720, "1080p": 1080, "4k": 2160}[options["resolution"]]
        width = height * 16 // 9
        if options["orientation"] == "portrait":
            width, height = height, width

        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            pcm_path = temporary / "audio.f32"
            output_path = temporary / "video.mp4"

            decode = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-nostdin", "-y",
                    "-protocol_whitelist", "file,pipe",
                    "-i", str(folder / "audio"),
                    "-map", "0:a:0", "-vn", "-sn", "-dn",
                    "-t", str(MAX_SECONDS + 1),
                    "-ar", str(SR), "-ac", "2",
                    "-c:a", "pcm_f32le", "-f", "f32le",
                    str(pcm_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=600,
            )
            if decode.returncode:
                raise ValueError(
                    "Audio decoding failed:\n"
                    + decode.stderr.decode(errors="replace")[-2000:]
                )

            pcm = np.fromfile(pcm_path, dtype=np.float32).reshape(-1, 2)
            if not len(pcm):
                raise ValueError("The audio file contains no decoded samples.")
            if len(pcm) > SR * MAX_SECONDS:
                raise ValueError(f"Audio exceeds the {MAX_SECONDS // 60}-minute limit.")
            if not np.isfinite(pcm).all():
                raise ValueError("Audio contains non-finite PCM samples.")

            cover_path = folder / "cover"
            artwork, accents = load_artwork(
                cover_path if cover_path.exists() else None
            )

            encoder_name, encoder_args, nvenc_warning = choose_encoder(
                options["codec"], width, height, options["fps"],
                options["require_nvenc"],
            )

            batch_size = (
                2 if width * height >= 8_000_000
                else 6 if width * height >= 2_000_000
                else 12
            )

            metadata.update(
                {
                    "width": width,
                    "height": height,
                    "duration_seconds": len(pcm) / SR,
                    "encoder": encoder_name,
                    "hardware_encoding": encoder_name.endswith("_nvenc"),
                    "nvenc_warning": nvenc_warning,
                    "batch_size": batch_size,
                    "palette": [list(color) for color in accents],
                    "audio_codec": "aac",
                }
            )
            report("analyzing", 0.03)

            torch.set_num_threads(8)
            torch.cuda.set_device(0)

            with torch.inference_mode():
                features = analyze_audio(pcm, options["fps"])
                del pcm

                metadata["frames"] = features["frames"]
                metadata["video_duration_seconds"] = (
                    features["frames"] / options["fps"]
                )
                scene = Scene(
                    features, options, width, height, artwork, accents
                )
                report("rendering", 0.10)
                encode_video(
                    scene, pcm_path, output_path,
                    encoder_args, batch_size, report,
                )

            report("publishing", 0.99)
            destination = folder / "video.mp4"
            shutil.copyfile(output_path, destination)
            metadata["size_bytes"] = destination.stat().st_size
            volume.commit()

        metadata["elapsed_seconds"] = round(time.monotonic() - started, 2)
        states.put(
            job_id,
            {**metadata, "state": "succeeded", "progress": 1.0},
        )
        return metadata

    except Exception as exc:
        states.put(
            job_id,
            {
                "job_id": job_id,
                "state": "failed",
                "error": str(exc)[:5000],
            },
        )
        raise


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=3600,
    max_containers=8,
    volumes={ROOT: volume},
    secrets=[modal.Secret.from_name("visualizer-api")],
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse

    api = FastAPI(title="GPU Music Visualizer", version="1.0")
    expected_key = os.environ["VISUALIZER_API_KEY"].encode()

    # Authenticate before multipart parsing. Do not expose expensive GPU jobs
    # anonymously. For a public service, add user accounts and per-user quotas.
    @api.middleware("http")
    async def authenticate(request: Request, call_next):
        supplied = request.headers.get("x-api-key", "").encode()
        if not hmac.compare_digest(supplied, expected_key):
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)

        length = request.headers.get("content-length")
        if length:
            try:
                too_large = int(length) > (
                    MAX_AUDIO_BYTES + MAX_COVER_BYTES + 1024 * 1024
                )
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length"}, status_code=400
                )
            if too_large:
                return JSONResponse(
                    {"detail": "Upload too large"}, status_code=413
                )

        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def validate_id(job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Unknown job")

    async def save_upload(upload, destination, limit):
        count = 0
        try:
            with open(destination, "wb") as file:
                while chunk := await upload.read(1024 * 1024):
                    count += len(chunk)
                    if count > limit:
                        raise HTTPException(413, "Uploaded file exceeds its limit")
                    file.write(chunk)
        finally:
            await upload.close()

        if count == 0:
            raise HTTPException(422, "Uploaded file is empty")

    @api.post("/jobs", status_code=202)
    async def create_job(
        audio: UploadFile = File(...),
        title: str = Form(...),
        artist: str = Form(...),
        cover: UploadFile | None = File(None),
        resolution: str = Form("1080p"),
        fps: int = Form(30),
        codec: str = Form("h264"),
        orientation: str = Form("landscape"),
        require_nvenc: bool = Form(False),
    ):
        resolution = resolution.strip().lower()
        codec = codec.strip().lower().replace(".", "")
        orientation = orientation.strip().lower()
        orientation = {
            "vertical": "portrait",
            "mobile": "portrait",
        }.get(orientation, orientation)

        title = " ".join(title.split())
        artist = " ".join(artist.split())

        if resolution not in ("720p", "1080p", "4k"):
            raise HTTPException(422, "resolution must be 720p, 1080p, or 4K")
        if fps not in (24, 30, 60):
            raise HTTPException(422, "fps must be 24, 30, or 60")
        if codec not in ("av1", "h264"):
            raise HTTPException(422, "codec must be av1 or h264")
        if orientation not in ("landscape", "portrait"):
            raise HTTPException(422, "orientation must be landscape or portrait")
        if not 1 <= len(title) <= 120 or not 1 <= len(artist) <= 120:
            raise HTTPException(422, "title and artist must contain 1–120 characters")

        options = {
            "title": title,
            "artist": artist,
            "resolution": resolution,
            "fps": fps,
            "codec": codec,
            "orientation": orientation,
            "require_nvenc": require_nvenc,
        }

        job_id = uuid.uuid4().hex

        # max_inputs=1 deliberately prevents reload() from overlapping a
        # FileResponse reading this same mounted Volume in this container.
        await volume.reload.aio()
        folder = Path(ROOT) / job_id
        folder.mkdir(parents=True)

        try:
            await save_upload(audio, folder / "audio", MAX_AUDIO_BYTES)
            if cover is not None and cover.filename:
                await save_upload(cover, folder / "cover", MAX_COVER_BYTES)
            await volume.commit.aio()
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            await volume.commit.aio()
            raise

        await states.put.aio(
            job_id,
            {"job_id": job_id, "state": "queued", "progress": 0.0},
        )

        try:
            call = await render_video.spawn.aio(job_id, options)
            # Separate key avoids racing the renderer's progress updates.
            await states.put.aio(f"call:{job_id}", call.object_id)
        except Exception as exc:
            await states.put.aio(
                job_id,
                {"job_id": job_id, "state": "failed", "error": str(exc)},
            )
            raise HTTPException(503, "Unable to schedule rendering")

        return {
            "job_id": job_id,
            "call_id": call.object_id,
            "status_url": f"/jobs/{job_id}",
            "video_url": f"/jobs/{job_id}/video",
        }

    @api.get("/jobs/{job_id}")
    async def job_status(job_id: str):
        validate_id(job_id)
        record = await states.get.aio(job_id)
        if record is None:
            raise HTTPException(404, "Unknown job")

        # Also detect hard worker failures/timeouts that bypass Python's
        # exception handler.
        if record["state"] in ("queued", "running"):
            call_id = await states.get.aio(f"call:{job_id}")
            if call_id:
                try:
                    result = await modal.FunctionCall.from_id(call_id).get.aio(
                        timeout=0
                    )
                except modal.exception.TimeoutError:
                    pass
                except Exception as exc:
                    record = {
                        "job_id": job_id,
                        "state": "failed",
                        "error": str(exc)[:3000],
                    }
                    await states.put.aio(job_id, record)
                else:
                    record = {
                        **result, "state": "succeeded", "progress": 1.0
                    }
                    await states.put.aio(job_id, record)

        return {**record, "video_url": f"/jobs/{job_id}/video"}

    @api.get("/jobs/{job_id}/video")
    async def download_video(job_id: str):
        record = await job_status(job_id)
        if record["state"] != "succeeded":
            raise HTTPException(409, f"Job is {record['state']}")

        await volume.reload.aio()
        path = Path(ROOT) / job_id / "video.mp4"
        if not path.exists():
            raise HTTPException(404, "Rendered file is unavailable")

        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"visualizer-{job_id}.mp4",
        )

    return api