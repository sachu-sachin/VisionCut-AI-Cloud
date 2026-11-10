"""
Local Video Preprocessing and Adaptive Frame Sampler for VisionCut AI Cloud.
Extracts representative frames, detects scene changes, static periods,
and visual activity without overloading system RAM.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from visioncut.core.models import VideoMetadata, FrameSample
from visioncut.core.config import PreprocessingSettings
from visioncut.core.ffmpeg_utils import FFmpegRunner
from visioncut.preprocessing.audio_analyzer import AudioAnalyzer, AudioSegmentAnalysis


class VideoAnalyzer:
    """
    Lightweight, streaming video analyzer.
    Processes frames in small iterative batches to guarantee low RAM usage (< 300MB)
    even on 10-minute 1080p 30fps screen recordings.
    """

    def __init__(
        self,
        settings: PreprocessingSettings,
        ffmpeg_runner: Optional[FFmpegRunner] = None
    ):
        self.settings = settings
        self.ffmpeg = ffmpeg_runner or FFmpegRunner()
        self.audio_analyzer = AudioAnalyzer(settings)

    def preprocess_video(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[VideoMetadata, List[FrameSample], Optional[AudioSegmentAnalysis], str]:
        """
        Executes full local preprocessing:
        1. Reads video metadata.
        2. Demuxes audio and performs silence/waveform analysis.
        3. Iterates through video at adaptive intervals.
        4. Detects scene changes, static periods, and repetitive patterns.
        5. Saves compressed thumbnails for cloud analysis.
        6. Generates contact sheet.

        Returns:
            (metadata, sampled_frames, audio_analysis, contact_sheet_path)
        """
        out_dir = Path(output_dir)
        frames_dir = out_dir / "frames"
        audio_dir = out_dir / "audio"
        preview_dir = out_dir / "preview"

        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(0.05, "Reading video metadata...")

        metadata = self.ffmpeg.get_video_metadata(video_path)

        # 2. Audio preprocessing
        audio_analysis = None
        if metadata.has_audio:
            if progress_callback:
                progress_callback(0.12, "Extracting and analyzing audio track...")
            temp_wav = str(audio_dir / "extracted_audio.wav")
            if self.ffmpeg.extract_audio_wav(video_path, temp_wav):
                audio_analysis = self.audio_analyzer.analyze_wav(temp_wav)

        # 3. Open Video with OpenCV
        if progress_callback:
            progress_callback(0.20, "Beginning streaming visual frame sampling...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {video_path}")

        fps = metadata.fps if metadata.fps > 0 else 30.0
        step_frames = max(1, int(round(fps / self.settings.initial_sample_rate_fps)))

        sampled_frames: List[FrameSample] = []
        frame_file_paths: List[str] = []
        timestamps: List[float] = []

        prev_phash: Optional[imagehash.ImageHash] = None
        prev_hsv_hist: Optional[np.ndarray] = None
        consecutive_static_count = 0

        total_est_samples = max(1, int(metadata.total_frames / step_frames))
        sample_idx = 0
        current_frame_pos = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_sec = round(current_frame_pos / fps, 3)

            # Resize frame for efficient comparison and cloud upload
            thumb = cv2.resize(
                frame,
                (self.settings.thumbnail_width, self.settings.thumbnail_height),
                interpolation=cv2.INTER_AREA
            )

            # Convert to PIL Image for perceptual hashing
            rgb_thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_thumb)
            curr_phash = imagehash.phash(pil_img)

            # Compute HSV histogram for color shift / major scene change detection
            hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            is_scene_change = False
            motion_score = 0.0
            is_static = False

            if prev_phash is not None and prev_hsv_hist is not None:
                # Perceptual hash distance (0 - 64)
                hash_dist = curr_phash - prev_phash
                # Histogram correlation (-1 to +1, 1 is identical)
                hist_corr = cv2.compareHist(prev_hsv_hist, hist, cv2.HISTCMP_CORREL)
                motion_score = round(float(hash_dist) / 64.0, 3)

                # Scene change threshold: large hash distance or sudden color histogram shift
                if hash_dist > 18 or hist_corr < (1.0 - self.settings.scene_change_threshold):
                    is_scene_change = True

                # Static check: very small perceptual hash distance
                if hash_dist <= self.settings.phash_similarity_threshold:
                    consecutive_static_count += 1
                else:
                    consecutive_static_count = 0

                static_duration = consecutive_static_count * (step_frames / fps)
                if static_duration >= self.settings.static_period_min_seconds:
                    is_static = True
            else:
                # First frame is always a key anchor
                is_scene_change = True

            prev_phash = curr_phash
            prev_hsv_hist = hist

            # Audio state at this timestamp
            is_silent = False
            is_clipped = False
            audio_rms = -60.0
            if audio_analysis:
                is_silent = audio_analysis.is_silent_at(timestamp_sec)
                is_clipped = audio_analysis.is_clipped_at(timestamp_sec)
                audio_rms = audio_analysis.get_rms_at(timestamp_sec)

            # Adaptive sampling logic:
            # If screen is static and nothing is changing, we don't need to waste Colab VLM calls
            # on middle redundant frames, but we preserve first and last of static period
            needs_cloud_analysis = True
            if is_static and consecutive_static_count > 2:
                needs_cloud_analysis = False

            # Save JPEG thumbnail
            frame_filename = f"frame_{sample_idx:05d}_{timestamp_sec:.2f}s.jpg"
            frame_abs_path = frames_dir / frame_filename
            pil_img.save(str(frame_abs_path), "JPEG", quality=self.settings.jpeg_quality)

            frame_rel_path = f"frames/{frame_filename}"
            frame_sample = FrameSample(
                frame_idx=current_frame_pos,
                timestamp_sec=timestamp_sec,
                image_rel_path=frame_rel_path,
                phash=str(curr_phash),
                motion_score=motion_score,
                is_scene_change=is_scene_change,
                is_static_period=is_static,
                audio_rms=audio_rms,
                is_silence=is_silent,
                is_clipped=is_clipped,
                needs_cloud_analysis=needs_cloud_analysis
            )

            sampled_frames.append(frame_sample)
            frame_file_paths.append(str(frame_abs_path))
            timestamps.append(timestamp_sec)

            sample_idx += 1
            current_frame_pos += step_frames

            if progress_callback and sample_idx % 10 == 0:
                pct = 0.20 + 0.65 * min(1.0, sample_idx / total_est_samples)
                progress_callback(
                    pct,
                    f"Sampled {sample_idx} frames ({timestamp_sec:.1f}s / {metadata.duration_sec:.1f}s)..."
                )

        cap.release()

        # 4. Generate Review Contact Sheet
        if progress_callback:
            progress_callback(0.88, "Generating review contact sheet...")

        contact_sheet_path = preview_dir / "contact_sheet.jpg"
        # Pick representative subset for contact sheet (e.g. up to 30 key visual moments)
        key_indices = [
            i for i, f in enumerate(sampled_frames)
            if f.is_scene_change or (i % max(1, len(sampled_frames) // 25) == 0)
        ]
        if not key_indices:
            key_indices = list(range(min(24, len(sampled_frames))))

        cs_frame_paths = [frame_file_paths[i] for i in key_indices]
        cs_timestamps = [timestamps[i] for i in key_indices]

        generated_cs = self.ffmpeg.generate_contact_sheet(
            cs_frame_paths,
            cs_timestamps,
            str(contact_sheet_path),
            columns=self.settings.contact_sheet_cols
        )

        if progress_callback:
            progress_callback(1.0, "Local video preprocessing complete.")

        return metadata, sampled_frames, audio_analysis, generated_cs

