"""
Audio analysis module for VisionCut AI Cloud.
Analyzes audio tracks for prolonged silence (accidental gaps),
waveform energy, and clipping detection.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import soundfile as sf

from visioncut.core.config import PreprocessingSettings


class AudioSegmentAnalysis:
    def __init__(
        self,
        duration_sec: float,
        sample_rate: int,
        rms_db_per_second: List[float],
        silence_spans: List[Tuple[float, float]],
        clipping_spans: List[Tuple[float, float]],
    ):
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.rms_db_per_second = rms_db_per_second
        self.silence_spans = silence_spans
        self.clipping_spans = clipping_spans

    def is_silent_at(self, timestamp_sec: float) -> bool:
        for start, end in self.silence_spans:
            if start <= timestamp_sec <= end:
                return True
        return False

    def is_clipped_at(self, timestamp_sec: float) -> bool:
        for start, end in self.clipping_spans:
            if start <= timestamp_sec <= end:
                return True
        return False

    def get_rms_at(self, timestamp_sec: float) -> float:
        sec_idx = int(timestamp_sec)
        if 0 <= sec_idx < len(self.rms_db_per_second):
            return self.rms_db_per_second[sec_idx]
        return -100.0


class AudioAnalyzer:
    """Performs lightweight, memory-efficient audio waveform analysis."""

    def __init__(self, settings: PreprocessingSettings):
        self.settings = settings

    def analyze_wav(self, wav_path: str) -> AudioSegmentAnalysis:
        """Reads audio file in streaming blocks and computes RMS envelope and silence periods."""
        path_obj = Path(wav_path)
        if not path_obj.exists():
            return AudioSegmentAnalysis(0.0, 44100, [], [], [])

        with sf.SoundFile(str(path_obj)) as f:
            sr = f.samplerate
            channels = f.channels
            total_samples = len(f)
            duration_sec = total_samples / sr

            block_size = sr  # 1 second blocks
            rms_per_sec: List[float] = []
            silence_spans: List[Tuple[float, float]] = []
            clipping_spans: List[Tuple[float, float]] = []

            current_silence_start: float | None = None
            current_clip_start: float | None = None

            sec_cursor = 0.0
            clip_threshold_val = 10.0 ** (self.settings.clipping_db_threshold / 20.0)

            while True:
                data = f.read(block_size, dtype="float32")
                if len(data) == 0:
                    break

                # If multi-channel, average to mono
                if channels > 1:
                    data = np.mean(data, axis=1)

                # Check clipping
                max_amp = float(np.max(np.abs(data)))
                is_clipped = max_amp >= clip_threshold_val

                # Compute RMS
                mean_sq = float(np.mean(data ** 2))
                rms = math.sqrt(mean_sq)
                if rms > 1e-7:
                    db = 20.0 * math.log10(rms)
                else:
                    db = -100.0
                rms_per_sec.append(round(db, 2))

                is_silent = db <= self.settings.silence_db_threshold

                # Track silence spans
                if is_silent:
                    if current_silence_start is None:
                        current_silence_start = sec_cursor
                else:
                    if current_silence_start is not None:
                        sil_dur = sec_cursor - current_silence_start
                        if sil_dur >= self.settings.silence_min_duration_sec:
                            silence_spans.append((round(current_silence_start, 2), round(sec_cursor, 2)))
                        current_silence_start = None

                # Track clipping spans
                if is_clipped:
                    if current_clip_start is None:
                        current_clip_start = sec_cursor
                else:
                    if current_clip_start is not None:
                        clipping_spans.append((round(current_clip_start, 2), round(sec_cursor, 2)))
                        current_clip_start = None

                sec_cursor += len(data) / sr

            # Handle trailing silence
            if current_silence_start is not None:
                sil_dur = sec_cursor - current_silence_start
                if sil_dur >= self.settings.silence_min_duration_sec:
                    silence_spans.append((round(current_silence_start, 2), round(sec_cursor, 2)))

            if current_clip_start is not None:
                clipping_spans.append((round(current_clip_start, 2), round(sec_cursor, 2)))

        return AudioSegmentAnalysis(
            duration_sec=duration_sec,
            sample_rate=sr,
            rms_db_per_second=rms_per_sec,
            silence_spans=silence_spans,
            clipping_spans=clipping_spans
        )

