"""
FFmpeg and Media Processing Utilities for VisionCut AI Cloud.
Provides robust wrappers around FFmpeg for metadata extraction,
lightweight audio demuxing, frame extraction, and contact sheet generation.
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visioncut.core.models import VideoMetadata


def get_ffmpeg_path() -> str:
    """Finds FFmpeg executable via imageio_ffmpeg, system PATH, or common locations."""
    # 1. Try imageio_ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    # 2. Try shutil.which
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    # 3. Try common Windows paths
    candidates = [
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        str(Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return "ffmpeg"


class FFmpegRunner:
    """Helper for executing ffmpeg commands safely."""

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()

    def run(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.ffmpeg_path] + args
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            startupinfo=startupinfo,
            check=False
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"FFmpeg command failed with code {result.returncode}:\n{result.stderr}")
        return result

    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Reads detailed video and audio stream metadata using OpenCV + FFmpeg probe."""
        path_obj = Path(video_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        file_size = path_obj.stat().st_size
        filename = path_obj.name

        # 1. Open with OpenCV to get basic video parameters
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(total_frames / fps) if fps > 0 else 0.0
        cap.release()

        # 2. Check audio stream using ffmpeg -i
        has_audio = False
        sample_rate = 44100
        channels = 2
        v_codec = "h264"
        a_codec = "aac"

        res = self.run(["-i", str(video_path)], check=False)
        output = res.stderr

        for line in output.splitlines():
            line_str = line.strip()
            if "Audio:" in line_str:
                has_audio = True
                # Parse sample rate
                for part in line_str.split(","):
                    part = part.strip()
                    if "Hz" in part:
                        try:
                            sample_rate = int(part.replace("Hz", "").strip())
                        except ValueError:
                            pass
                    if "stereo" in part:
                        channels = 2
                    elif "mono" in part:
                        channels = 1
                    if "Audio:" in part:
                        a_codec = part.split("Audio:")[1].strip().split()[0]
            if "Video:" in line_str:
                for part in line_str.split(","):
                    part = part.strip()
                    if "Video:" in part:
                        v_codec = part.split("Video:")[1].strip().split()[0]

        return VideoMetadata(
            file_path=str(path_obj.resolve()),
            filename=filename,
            duration_sec=round(duration, 3),
            width=width,
            height=height,
            fps=round(fps, 3),
            total_frames=total_frames,
            has_audio=has_audio,
            audio_sample_rate=sample_rate,
            audio_channels=channels,
            video_codec=v_codec,
            audio_codec=a_codec,
            file_size_bytes=file_size
        )

    def extract_audio_wav(self, video_path: str, output_wav_path: str, sample_rate: int = 44100) -> bool:
        """Extracts mono/stereo 16-bit PCM WAV for audio analysis."""
        out_p = Path(output_wav_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res = self.run([
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono is ideal for waveform & silence analysis
            str(output_wav_path)
        ], check=False)
        return res.returncode == 0 and out_p.exists()

    def generate_contact_sheet(
        self,
        frame_paths: List[str],
        timestamps: List[float],
        output_image_path: str,
        columns: int = 5,
        thumb_width: int = 320,
        thumb_height: int = 180
    ) -> str:
        """
        Creates a contact sheet grid image from sampled frames with timestamp badges.
        Useful for quick human inspection of the entire recording.
        """
        if not frame_paths:
            return ""

        num_frames = len(frame_paths)
        rows = int(np.ceil(num_frames / columns))
        padding = 10
        label_height = 24

        cell_w = thumb_width
        cell_h = thumb_height + label_height

        sheet_w = columns * cell_w + (columns + 1) * padding
        sheet_h = rows * cell_h + (rows + 1) * padding

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(26, 28, 36))
        draw = ImageDraw.Draw(sheet)

        for idx, (fpath, ts) in enumerate(zip(frame_paths, timestamps)):
            if not os.path.exists(fpath):
                continue
            r = idx // columns
            c = idx % columns

            x = padding + c * (cell_w + padding)
            y = padding + r * (cell_h + padding)

            try:
                with Image.open(fpath) as img:
                    img_thumb = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
                    sheet.paste(img_thumb, (x, y))
            except Exception:
                continue

            # Draw timestamp badge
            ts_m = int(ts // 60)
            ts_s = int(ts % 60)
            ts_str = f"#{idx+1:02d} - {ts_m:02d}:{ts_s:02d}"

            draw.rectangle([x, y + thumb_height, x + thumb_width, y + cell_h], fill=(18, 19, 24))
            draw.text((x + 8, y + thumb_height + 4), ts_str, fill=(220, 225, 235))

        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(str(out_path), "JPEG", quality=88)
        return str(out_path)

