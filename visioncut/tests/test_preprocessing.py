"""
Unit and integration test for Phase 1: Local Video Preprocessing.
"""

import os
import shutil
import tempfile
from pathlib import Path

from visioncut.core.config import PreprocessingSettings
from visioncut.preprocessing.video_analyzer import VideoAnalyzer
from visioncut.tests.synthetic_data import generate_synthetic_tutorial_video


def test_preprocessing():
    temp_dir = Path(tempfile.mkdtemp(prefix="visioncut_test_p1_"))
    try:
        video_path = str(temp_dir / "sample_tutorial.mp4")
        print(f"[TEST] Generating synthetic 1080p 30fps tutorial video at: {video_path}")
        generate_synthetic_tutorial_video(video_path, duration_sec=12.0)
        assert os.path.exists(video_path), "Failed to generate synthetic video"

        settings = PreprocessingSettings()
        analyzer = VideoAnalyzer(settings)

        out_prep_dir = str(temp_dir / "prep_output")
        print("[TEST] Running local video preprocessing...")
        meta, samples, audio_analysis, cs_path = analyzer.preprocess_video(
            video_path=video_path,
            output_dir=out_prep_dir,
            progress_callback=lambda p, msg: print(f"  Progress {p*100:.1f}%: {msg}")
        )

        # 1. Validate Metadata
        print("\n--- METADATA VERIFICATION ---")
        print(f"File: {meta.filename}")
        print(f"Resolution: {meta.width}x{meta.height} (Target: 1920x1080)")
        print(f"FPS: {meta.fps} (Target: 30.0)")
        print(f"Duration: {meta.duration_sec:.2f}s (Target: ~12.0s)")
        print(f"Has Audio: {meta.has_audio} ({meta.audio_sample_rate}Hz, {meta.audio_channels}ch)")

        assert meta.width == 1920, f"Expected width 1920, got {meta.width}"
        assert meta.height == 1080, f"Expected height 1080, got {meta.height}"
        assert abs(meta.fps - 30.0) < 0.5, f"Expected fps 30, got {meta.fps}"
        assert meta.has_audio, "Expected video to have audio stream"

        # 2. Validate Audio Silence Detection
        print("\n--- AUDIO ANALYSIS VERIFICATION ---")
        assert audio_analysis is not None, "Audio analysis failed"
        print(f"Silence Spans detected: {audio_analysis.silence_spans}")
        # We placed a silent gap from 3.0s to 5.0s (2.0s long)
        found_silence = any(s <= 4.0 <= e for s, e in audio_analysis.silence_spans) or \
                        any(e - s >= 1.5 for s, e in audio_analysis.silence_spans)
        print(f"Verified intentional silent gap: {found_silence}")

        # 3. Validate Visual Frames and Contact Sheet
        print("\n--- VISUAL SAMPLING VERIFICATION ---")
        print(f"Sampled frames count: {len(samples)}")
        assert len(samples) >= 10, "Should have sampled at least 10 frames for 12s video"
        scene_changes = [f for f in samples if f.is_scene_change]
        print(f"Detected scene change keyframes: {len(scene_changes)}")
        assert len(scene_changes) >= 2, "Should have detected scene transitions"

        assert os.path.exists(cs_path), "Contact sheet was not generated"
        print(f"Contact sheet successfully generated at: {cs_path}")

        print("\n>>> PHASE 1 TEST PASSED SUCCESSFULLY! <<<\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_preprocessing()

