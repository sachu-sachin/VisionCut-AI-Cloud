"""
Unit and integration test for Phase 2: Google Drive Job Queue.
"""

import os
import shutil
import tempfile
from pathlib import Path

from visioncut.core.models import (
    VideoMetadata, FrameSample, JobStatus, UIEventType, EditingActionType
)
from visioncut.queue.drive_queue import DriveJobQueue


def test_queue():
    temp_dir = Path(tempfile.mkdtemp(prefix="visioncut_test_p2_"))
    try:
        queue_root = temp_dir / "VisionCutCloud"
        queue = DriveJobQueue(queue_root)

        # 1. Create mock metadata and frames
        meta = VideoMetadata(
            file_path="C:/videos/tutorial.mp4",
            filename="tutorial.mp4",
            duration_sec=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=1800,
            has_audio=True
        )

        # Create dummy frame files in a source directory
        src_frames_dir = temp_dir / "src_frames"
        src_frames_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for i in range(5):
            fname = f"frame_{i:05d}_{i*1.0:.2f}s.jpg"
            fpath = src_frames_dir / fname
            with open(fpath, "wb") as f:
                f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB")  # minimal jpeg header
            samples.append(FrameSample(
                frame_idx=i * 30,
                timestamp_sec=float(i),
                image_rel_path=f"frames/{fname}",
                needs_cloud_analysis=(i % 2 == 0)
            ))

        print("[TEST] Submitting job to Google Drive Job Queue...")
        job_meta = queue.submit_job(
            video_metadata=meta,
            sampled_frames=samples,
            frames_source_dir=str(src_frames_dir),
            settings={"quality": "high", "sensitivity": 0.8}
        )

        print(f"Created Job ID: {job_meta.job_id}")
        assert job_meta.status == JobStatus.PENDING

        # 2. Verify job directory layout
        job_dir = queue.jobs_dir / job_meta.job_id
        assert (job_dir / "job.json").exists(), "job.json missing"
        assert (job_dir / "status.json").exists(), "status.json missing"
        assert (job_dir / "metadata" / "video_metadata.json").exists(), "video_metadata.json missing"
        assert (job_dir / "frames").exists(), "frames folder missing"

        # 3. Verify status transitions
        print("[TEST] Updating status to PROCESSING (45%)...")
        queue.update_job_status(job_meta.job_id, JobStatus.PROCESSING, progress=45.0, message="Analyzing frame batch 1")
        status, data = queue.get_job_status(job_meta.job_id)
        assert status == JobStatus.PROCESSING
        assert data["progress_percent"] == 45.0

        # 4. Write Results
        print("[TEST] Writing AI analysis results...")
        mock_events = [
            {
                "event_id": "evt_001",
                "timestamp": 2.0,
                "duration": 1.5,
                "event_type": UIEventType.BUTTON_CLICK.value,
                "description": "User clicked Execute button",
                "ui_target": "Execute Button",
                "bbox": {"ymin": 0.4, "xmin": 0.4, "ymax": 0.5, "xmax": 0.6},
                "confidence": 0.95,
                "recommended_action": EditingActionType.HIGHLIGHT_BOX.value,
                "reason": "Important action trigger"
            }
        ]
        queue.write_results(job_meta.job_id, mock_events, report={"model": "Florence-2-base"})

        # 5. Fetch Results
        fetched = queue.fetch_results(job_meta.job_id)
        assert fetched is not None, "Failed to fetch results"
        assert len(fetched) == 1
        assert fetched[0]["event_id"] == "evt_001"
        print(f"Successfully fetched {len(fetched)} events")

        # 6. Archive Job
        print("[TEST] Archiving completed job...")
        archived = queue.archive_job(job_meta.job_id)
        assert archived, "Failed to archive job"
        assert (queue.archive_dir / job_meta.job_id).exists()
        assert not (queue.jobs_dir / job_meta.job_id).exists()
        # Verify results folder is preserved!
        assert (queue.results_dir / job_meta.job_id / "events.json").exists(), "Results were erroneously deleted!"

        print("\n>>> PHASE 2 TEST PASSED SUCCESSFULLY! <<<\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_queue()

