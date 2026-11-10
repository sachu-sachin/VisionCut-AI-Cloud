"""
Unit and integration test for Phases 6-9: Motion Graphics, SFX,
Kdenlive Project Generation, and Project Validation.
"""

import os
import shutil
import tempfile
from pathlib import Path

from visioncut.core.models import (
    VideoMetadata, UIEvent, UIEventType, EditingDecision,
    EditingActionType, ConfidenceLevel, SFXType, EditingPlan, BoundingBox
)
from visioncut.kdenlive.project_generator import KdenliveProjectGenerator
from visioncut.kdenlive.validator import ProjectValidator
from visioncut.tests.synthetic_data import generate_synthetic_tutorial_video


def test_kdenlive_pipeline():
    temp_dir = Path(tempfile.mkdtemp(prefix="visioncut_test_kdenlive_"))
    try:
        # 1. Create source video
        video_path = str(temp_dir / "tutorial_sample.mp4")
        generate_synthetic_tutorial_video(video_path, duration_sec=10.0)

        meta = VideoMetadata(
            file_path=video_path,
            filename="tutorial_sample.mp4",
            duration_sec=10.0,
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=300,
            has_audio=True
        )

        # 2. Build mock editing plan with:
        # - Smart cut: remove 3.0s to 5.0s pause
        # - Action button with highlight box + click SFX at 8.0s
        # - Terminal execution with arrow at 5.5s
        # - Result display with callout at 9.0s
        cuts = [
            {"in": 0.0, "out": 3.0},
            {"in": 5.0, "out": 10.0}
        ]

        decisions = [
            EditingDecision(
                decision_id="dec_01",
                start_time=5.5,
                end_time=7.5,
                action_type=EditingActionType.DRAW_ARROW,
                target_bbox=BoundingBox(ymin=0.45, xmin=0.30, ymax=0.85, xmax=0.90),
                confidence=0.92,
                reason="Direct attention to terminal command"
            ),
            EditingDecision(
                decision_id="dec_02",
                start_time=8.0,
                end_time=9.5,
                action_type=EditingActionType.HIGHLIGHT_BOX,
                target_bbox=BoundingBox(ymin=0.37, xmin=0.41, ymax=0.45, xmax=0.58),
                sfx_type=SFXType.MOUSE_CLICK,
                confidence=0.95,
                reason="Highlight primary action button"
            ),
            EditingDecision(
                decision_id="dec_03",
                start_time=9.0,
                end_time=10.0,
                action_type=EditingActionType.TEXT_CALLOUT,
                target_bbox=BoundingBox(ymin=0.27, xmin=0.31, ymax=0.42, xmax=0.68),
                text_content="EXECUTION SUCCESSFUL",
                sfx_type=SFXType.SUCCESS_CHIME,
                confidence=0.96,
                reason="Showcase final success result"
            )
        ]

        plan = EditingPlan(
            plan_id="plan_sample",
            video_metadata=meta,
            original_duration=10.0,
            estimated_edited_duration=8.0,
            events=[],
            decisions=decisions,
            cuts=cuts
        )

        out_proj_dir = str(temp_dir / "VisionCut_Output" / "Tutorial_Sample")

        # 3. Generate Project
        print("[TEST] Generating Kdenlive project...")
        generator = KdenliveProjectGenerator()
        kdenlive_path, summary = generator.generate_project(
            plan=plan,
            source_video_path=video_path,
            output_project_dir=out_proj_dir,
            project_title="Tutorial_Sample"
        )

        print(f"Generated Kdenlive file: {kdenlive_path}")
        assert os.path.exists(kdenlive_path), "Project file not created"

        # 4. Validate Project with Validator
        print("[TEST] Validating generated Kdenlive project...")
        validator = ProjectValidator()
        report = validator.validate_project(kdenlive_path)

        print(f"Validation Result: Passed={report.is_valid}")
        print(f"Summary: {report.summary}")
        for chk in report.checks:
            print(f"  Check '{chk.name}': {chk.passed} ({chk.details})")

        assert report.is_valid, f"Project failed validation: {report.errors}"

        # 5. Verify asset directories
        proj_root = Path(out_proj_dir)
        assert (proj_root / "Graphics" / "boxes").exists()
        assert (proj_root / "Graphics" / "arrows").exists()
        assert (proj_root / "Graphics" / "callouts").exists()
        assert (proj_root / "Audio" / "sfx").exists()
        assert (proj_root / "Preview" / "report.html").exists()
        assert (proj_root / "Preview" / "validation.json").exists()

        # Check SFX WAV files
        click_wav = proj_root / "Audio" / "sfx" / "mouse_click.wav"
        assert click_wav.exists(), "mouse_click.wav missing"
        assert click_wav.stat().st_size > 1000, "WAV file empty"

        print("\n>>> PHASES 6-9 TEST PASSED SUCCESSFULLY! <<<\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_kdenlive_pipeline()

