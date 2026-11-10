"""
Visual Preview Video Renderer for VisionCut AI Cloud.
Renders an annotated preview MP4 video combining smart cuts,
motion graphic overlays (rectangles, arrows, circles, callouts),
camera zooms, and synchronized sound effects for direct visual inspection.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import soundfile as sf

from visioncut.core.models import EditingPlan, EditingActionType, SFXType
from visioncut.core.ffmpeg_utils import FFmpegRunner


class PreviewRenderer:
    """Renders a composite preview MP4 video reflecting the full editing plan."""

    def __init__(self, ffmpeg_runner: Optional[FFmpegRunner] = None):
        self.ffmpeg = ffmpeg_runner or FFmpegRunner()

    def render_preview_video(
        self,
        source_video_path: str,
        plan: EditingPlan,
        output_mp4_path: str,
        sfx_dir: str,
        preview_width: int = 1280,
        preview_height: int = 720
    ) -> str:
        """
        Renders an annotated preview video with:
        - Accidental gaps cut out
        - Glowing rectangle highlight
        - Directional arrow
        - Circular highlight
        - Digital camera zoom
        - Callout text badge
        - Synchronized SFX audio track
        """
        out_p = Path(output_mp4_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        temp_video_silent = str(out_p.parent / "temp_preview_visuals.mp4")
        temp_audio_mixed = str(out_p.parent / "temp_preview_mixed.wav")

        cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {source_video_path}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video_silent, fourcc, src_fps, (preview_width, preview_height))

        # Build list of active source frame ranges from cuts
        cuts = plan.cuts or [{"in": 0.0, "out": plan.original_duration}]
        frame_ranges = []
        for c in cuts:
            f_in = int(round(c["in"] * src_fps))
            f_out = int(round(c["out"] * src_fps))
            if f_out > f_in:
                frame_ranges.append((f_in, f_out))

        # Precompute decisions active at each original timestamp
        decisions = [d for d in plan.decisions if d.approved]

        # Audio setup
        sample_rate = 44100
        total_edited_duration = sum(c["out"] - c["in"] for c in cuts)
        total_audio_samples = int(total_edited_duration * sample_rate)
        mixed_audio = np.zeros(total_audio_samples, dtype=np.float32)

        # Extract source audio into numpy
        temp_src_wav = str(out_p.parent / "temp_source_audio.wav")
        has_src_audio = self.ffmpeg.extract_audio_wav(source_video_path, temp_src_wav, sample_rate)
        src_audio_data = None
        if has_src_audio and Path(temp_src_wav).exists():
            try:
                src_audio_data, _ = sf.read(temp_src_wav, dtype="float32")
            except Exception:
                pass

        edited_time_cursor = 0.0
        current_out_frame = 0

        for f_start, f_end in frame_ranges:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

            # Copy slice of source narration audio into edited timeline
            seg_dur = (f_end - f_start) / src_fps
            audio_start_sample = int(edited_time_cursor * sample_rate)
            audio_end_sample = audio_start_sample + int(seg_dur * sample_rate)

            if src_audio_data is not None:
                src_a_start = int((f_start / src_fps) * sample_rate)
                src_a_end = src_a_start + (audio_end_sample - audio_start_sample)
                if src_a_end <= len(src_audio_data):
                    mixed_audio[audio_start_sample:audio_end_sample] = src_audio_data[src_a_start:src_a_end]

            for src_f_idx in range(f_start, f_end):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                orig_t = src_f_idx / src_fps
                timeline_t = edited_time_cursor + (src_f_idx - f_start) / src_fps

                # Check if camera zoom is active
                active_zoom = None
                for d in decisions:
                    if d.action_type == EditingActionType.ZOOM_PAN and d.start_time <= orig_t <= d.end_time:
                        active_zoom = d
                        break

                if active_zoom and active_zoom.target_bbox:
                    # Apply digital camera crop & zoom
                    box = active_zoom.target_bbox
                    ymin = int(box.ymin * src_h)
                    xmin = int(box.xmin * src_w)
                    ymax = int(box.ymax * src_h)
                    xmax = int(box.xmax * src_w)
                    crop = frame[max(0, ymin):min(src_h, ymax), max(0, xmin):min(src_w, xmax)]
                    if crop.size > 0:
                        frame = cv2.resize(crop, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

                # Resize to preview resolution
                out_frame = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

                # Overlay active visual annotations
                for d in decisions:
                    if not (d.start_time <= orig_t <= d.end_time):
                        continue

                    # 1. Rectangle
                    if d.action_type == EditingActionType.HIGHLIGHT_BOX and d.target_bbox:
                        rx, ry, rw, rh = d.target_bbox.to_pixel_rect(preview_width, preview_height)
                        # Pulse effect
                        pulse = int(128 + 127 * math.sin(current_out_frame * 0.15))
                        cv2.rectangle(out_frame, (rx, ry), (rx + rw, ry + rh), (0, pulse, 255), 4)
                        cv2.rectangle(out_frame, (rx - 2, ry - 2), (rx + rw + 2, ry + rh + 2), (50, 50, 220), 1)

                    # 2. Circle
                    elif d.action_type == EditingActionType.DRAW_CIRCLE and d.target_bbox:
                        rx, ry, rw, rh = d.target_bbox.to_pixel_rect(preview_width, preview_height)
                        cx = rx + rw // 2
                        cy = ry + rh // 2
                        rad = max(24, int(max(rw, rh) * 0.65))
                        cv2.circle(out_frame, (cx, cy), rad, (255, 220, 0), 4)

                    # 3. Arrow
                    elif d.action_type == EditingActionType.DRAW_ARROW and d.target_bbox:
                        bx = int(d.target_bbox.center_x * preview_width)
                        by = int(d.target_bbox.center_y * preview_height)
                        start_pt = (max(50, bx - 140), max(50, by - 100))
                        cv2.arrowedLine(out_frame, start_pt, (bx, by), (80, 50, 255), 5, tipLength=0.25)

                    # 4. Text Callout Badge
                    elif d.action_type == EditingActionType.TEXT_CALLOUT:
                        badge_x = 40
                        badge_y = 60
                        text = d.text_content or "IMPORTANT NOTE"
                        cv2.rectangle(out_frame, (badge_x, badge_y - 35), (badge_x + 460, badge_y + 15), (20, 24, 35), -1)
                        cv2.rectangle(out_frame, (badge_x, badge_y - 35), (badge_x + 460, badge_y + 15), (0, 220, 255), 2)
                        cv2.putText(out_frame, text, (badge_x + 15, badge_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Add VisionCut watermark badge in top-right
                cv2.rectangle(out_frame, (preview_width - 240, 15), (preview_width - 15, 45), (15, 18, 25), -1)
                cv2.putText(out_frame, "VisionCut AI Preview", (preview_width - 230, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 1, cv2.LINE_AA)

                writer.write(out_frame)
                current_out_frame += 1

            edited_time_cursor += seg_dur

        cap.release()
        writer.release()

        # Mix SFX Audio into the timeline
        sfx_path_map = {
            SFXType.MOUSE_CLICK: Path(sfx_dir) / "mouse_click.wav",
            SFXType.WHOOSH: Path(sfx_dir) / "whoosh.wav",
            SFXType.SUCCESS_CHIME: Path(sfx_dir) / "success_chime.wav",
            SFXType.ERROR_ALERT: Path(sfx_dir) / "error_alert.wav"
        }

        # Place SFX at decisions
        for d in decisions:
            if not d.sfx_type or d.sfx_type not in sfx_path_map:
                continue
            sfx_file = sfx_path_map[d.sfx_type]
            if not sfx_file.exists():
                continue

            try:
                sfx_wave, sfx_sr = sf.read(str(sfx_file), dtype="float32")
                # Map original decision time to edited timeline time
                # Find which cut segment contains d.start_time
                target_t = None
                acc_t = 0.0
                for c in cuts:
                    if c["in"] <= d.start_time <= c["out"]:
                        target_t = acc_t + (d.start_time - c["in"])
                        break
                    acc_t += (c["out"] - c["in"])

                if target_t is not None:
                    start_samp = int(target_t * sample_rate)
                    end_samp = min(len(mixed_audio), start_samp + len(sfx_wave))
                    dur_samp = end_samp - start_samp
                    if dur_samp > 0:
                        mixed_audio[start_samp:end_samp] += sfx_wave[:dur_samp] * 0.75
            except Exception:
                pass

        # Write mixed audio WAV
        sf.write(temp_audio_mixed, mixed_audio, sample_rate, subtype="PCM_16")

        # Final mux with FFmpeg (H.264 + AAC)
        self.ffmpeg.run([
            "-y",
            "-i", temp_video_silent,
            "-i", temp_audio_mixed,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(out_p)
        ], check=True)

        # Cleanup intermediates
        for tmp in [temp_video_silent, temp_audio_mixed, temp_src_wav]:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass

        return str(out_p)

