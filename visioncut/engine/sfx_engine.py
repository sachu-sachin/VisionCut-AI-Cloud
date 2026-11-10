"""
Sound Effect (SFX) Engine for VisionCut AI Cloud.
Programmatically synthesizes crisp, original, royalty-free audio effects
(click, soft pop, notification, whoosh, success chime, error alert, emphasis)
using NumPy and SoundFile, with separate timeline track assignment.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import soundfile as sf

from visioncut.core.models import SFXType, SFXItem


class SFXEngine:
    """
    Synthesizes and exports broadcast-ready 44.1kHz 16-bit PCM WAV sound effects.
    Zero external licensing liabilities; works immediately out-of-the-box.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def apply_envelopes(self, audio: np.ndarray, fade_in_ms: float = 5.0, fade_out_ms: float = 20.0) -> np.ndarray:
        """Applies smooth linear fade-in and exponential fade-out to prevent clicks."""
        n_samples = len(audio)
        in_len = int((fade_in_ms / 1000.0) * self.sample_rate)
        out_len = int((fade_out_ms / 1000.0) * self.sample_rate)

        if in_len > 0 and in_len < n_samples:
            audio[:in_len] *= np.linspace(0.0, 1.0, in_len)
        if out_len > 0 and out_len < n_samples:
            audio[-out_len:] *= np.linspace(1.0, 0.0, out_len)

        # Normalize to prevent digital clipping
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * (0.95 / peak)
        return audio

    def synthesize_mouse_click(self) -> np.ndarray:
        """Crisp mechanical mouse switch click (~45ms)."""
        duration = 0.045
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        decay = np.exp(-t * 120.0)
        # Transient click + resonance
        wave = 0.6 * np.sin(2 * np.pi * 1400 * t) + 0.3 * np.sin(2 * np.pi * 2800 * t)
        noise = (np.random.rand(len(t)) - 0.5) * 0.2
        audio = (wave + noise) * decay
        return self.apply_envelopes(audio, fade_in_ms=2.0, fade_out_ms=15.0)

    def synthesize_soft_pop(self) -> np.ndarray:
        """Pleasant subtle pop sound (~80ms)."""
        duration = 0.08
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        freq = np.linspace(220, 680, len(t))
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        decay = np.exp(-t * 50.0)
        audio = np.sin(phase) * decay
        return self.apply_envelopes(audio, fade_in_ms=4.0, fade_out_ms=25.0)

    def synthesize_notification(self) -> np.ndarray:
        """Friendly two-tone bell notification (~350ms)."""
        duration = 0.35
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        half = len(t) // 2
        tone1 = np.sin(2 * np.pi * 523.25 * t[:half]) * np.exp(-t[:half] * 8.0)  # C5
        tone2 = np.sin(2 * np.pi * 659.25 * t[half:]) * np.exp(-t[:len(t)-half] * 7.0)  # E5
        audio = np.concatenate([tone1, tone2])
        return self.apply_envelopes(audio, fade_in_ms=5.0, fade_out_ms=60.0)

    def synthesize_whoosh(self) -> np.ndarray:
        """Airy transition whoosh for zoom and pan effects (~420ms)."""
        duration = 0.42
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        noise = np.random.randn(len(t))
        # Bell-shaped envelope
        center = duration / 2.0
        envelope = np.exp(-((t - center) ** 2) / (2 * (0.09 ** 2)))
        audio = noise * envelope * 0.45
        return self.apply_envelopes(audio, fade_in_ms=20.0, fade_out_ms=50.0)

    def synthesize_success_chime(self) -> np.ndarray:
        """Uplifting 4-note major arpeggio (~550ms)."""
        duration = 0.55
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        step = len(t) // 4
        # C5, E5, G5, C6
        notes = [523.25, 659.25, 783.99, 1046.50]
        chunks = []
        for i, freq in enumerate(notes):
            t_chunk = np.linspace(0, duration / 4, step, endpoint=False)
            decay = np.exp(-t_chunk * (6.0 - i * 0.5))
            chunk = np.sin(2 * np.pi * freq * t_chunk) * decay
            chunks.append(chunk)
        audio = np.concatenate(chunks)
        return self.apply_envelopes(audio, fade_in_ms=5.0, fade_out_ms=80.0)

    def synthesize_error_alert(self) -> np.ndarray:
        """Subtle warning dissonance buzzer (~300ms)."""
        duration = 0.30
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 311.13 * t)
        decay = np.exp(-t * 9.0)
        audio = tone * decay
        return self.apply_envelopes(audio, fade_in_ms=4.0, fade_out_ms=40.0)

    def synthesize_subtle_emphasis(self) -> np.ndarray:
        """High-pitched gentle ping for UI focus (~220ms)."""
        duration = 0.22
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 880.0 * t) * np.exp(-t * 18.0)
        return self.apply_envelopes(audio, fade_in_ms=3.0, fade_out_ms=30.0)

    def generate_sfx(self, sfx_type: SFXType, volume: float = 0.8) -> np.ndarray:
        generators = {
            SFXType.MOUSE_CLICK: self.synthesize_mouse_click,
            SFXType.SOFT_POP: self.synthesize_soft_pop,
            SFXType.NOTIFICATION: self.synthesize_notification,
            SFXType.WHOOSH: self.synthesize_whoosh,
            SFXType.SUCCESS_CHIME: self.synthesize_success_chime,
            SFXType.ERROR_ALERT: self.synthesize_error_alert,
            SFXType.SUBTLE_EMPHASIS: self.synthesize_subtle_emphasis,
        }
        gen = generators.get(sfx_type, self.synthesize_soft_pop)
        raw_audio = gen()
        # Scale by volume (0.0 to 1.0)
        return (raw_audio * max(0.0, min(1.0, volume))).astype(np.float32)

    def export_sfx_library(self, output_dir: str | Path) -> Dict[SFXType, str]:
        """Synthesizes and exports all default sound effects to WAV files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        library_paths = {}

        for sfx_type in SFXType:
            filename = f"{sfx_type.value}.wav"
            file_path = out / filename
            audio_data = self.generate_sfx(sfx_type)
            sf.write(str(file_path), audio_data, self.sample_rate, subtype="PCM_16")
            library_paths[sfx_type] = str(file_path)

        return library_paths

