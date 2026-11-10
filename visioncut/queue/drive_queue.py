"""
Google Drive Job Queue Manager for VisionCut AI Cloud.
Manages asynchronous job dispatching, atomic status updates,
checkpoints, and results synchronization between Desktop App and Cloud Workers.
"""

from __future__ import annotations
import os
import sys
import json
import uuid
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from visioncut.core.models import (
    JobStatus, JobMetadata, VideoMetadata, FrameSample, UIEvent, EditingPlan
)
from visioncut.core.config import AppConfig


class AtomicFileWriter:
    """Provides cross-platform atomic JSON and binary file writes."""

    @staticmethod
    def write_json(path: Path | str, data: Any, indent: int = 2) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.with_suffix(f".tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)
                f.flush()
                os.fsync(f.fileno())
            # Replace target atomically
            temp_file.replace(target)
        except Exception:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise

    @staticmethod
    def read_json(path: Path | str) -> Any:
        target = Path(path)
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)


class DriveJobQueue:
    """
    Manages the job queue directory structure:
    VisionCutCloud/
      jobs/
        <job_id>/
          job.json
          status.json
          frames/
          metadata/
      results/
        <job_id>/
          status.json
          events.json
          checkpoints/
      archive/
        <job_id>/
    """

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        self.archive_dir = self.root / "archive"

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def generate_job_id(self, prefix: str = "job") -> str:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        return f"{prefix}_{timestamp_str}_{short_id}"

    def submit_job(
        self,
        video_metadata: VideoMetadata,
        sampled_frames: List[FrameSample],
        frames_source_dir: str,
        settings: Dict[str, Any],
        model_name: str = "microsoft/Florence-2-base"
    ) -> JobMetadata:
        """
        Creates and stages a new job in the queue.
        Copies only frames flagged as `needs_cloud_analysis` to minimize upload bandwidth.
        """
        job_id = self.generate_job_id()
        job_dir = self.jobs_dir / job_id
        frames_target_dir = job_dir / "frames"
        metadata_target_dir = job_dir / "metadata"

        frames_target_dir.mkdir(parents=True, exist_ok=True)
        metadata_target_dir.mkdir(parents=True, exist_ok=True)

        frames_to_upload = [f for f in sampled_frames if f.needs_cloud_analysis]
        uploaded_frame_indices: List[int] = []

        src_dir = Path(frames_source_dir)
        for sample in frames_to_upload:
            src_file = src_dir / Path(sample.image_rel_path).name
            if src_file.exists():
                dst_file = frames_target_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                uploaded_frame_indices.append(sample.frame_idx)

        # Write metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        job_meta = JobMetadata(
            job_id=job_id,
            created_at=now_iso,
            updated_at=now_iso,
            status=JobStatus.PENDING,
            video_filename=video_metadata.filename,
            total_sampled_frames=len(uploaded_frame_indices),
            settings=settings,
            retry_count=0,
            model_name=model_name,
            progress_percent=0.0
        )

        AtomicFileWriter.write_json(job_dir / "job.json", job_meta.model_dump())
        AtomicFileWriter.write_json(job_dir / "status.json", {
            "status": JobStatus.PENDING.value,
            "updated_at": now_iso,
            "message": "Job queued for cloud processing"
        })
        AtomicFileWriter.write_json(metadata_target_dir / "video_metadata.json", video_metadata.model_dump())
        AtomicFileWriter.write_json(
            metadata_target_dir / "sampled_frames.json",
            [f.model_dump() for f in sampled_frames]
        )

        return job_meta

    def list_jobs(self, status_filter: Optional[JobStatus] = None) -> List[JobMetadata]:
        jobs: List[JobMetadata] = []
        if not self.jobs_dir.exists():
            return jobs

        for entry in self.jobs_dir.iterdir():
            if entry.is_dir():
                job_json = entry / "job.json"
                if job_json.exists():
                    try:
                        data = AtomicFileWriter.read_json(job_json)
                        if data:
                            meta = JobMetadata(**data)
                            # Check status file for real-time updates
                            status_file = entry / "status.json"
                            if status_file.exists():
                                s_data = AtomicFileWriter.read_json(status_file)
                                if s_data and "status" in s_data:
                                    meta.status = JobStatus(s_data["status"])
                                    meta.updated_at = s_data.get("updated_at", meta.updated_at)
                                    meta.progress_percent = float(s_data.get("progress_percent", meta.progress_percent))
                            if status_filter is None or meta.status == status_filter:
                                jobs.append(meta)
                    except Exception:
                        pass
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get_job_status(self, job_id: str) -> Tuple[JobStatus, Dict[str, Any]]:
        """Returns the current live status and status payload of a job."""
        # First check results dir (if completed)
        res_status = self.results_dir / job_id / "status.json"
        if res_status.exists():
            data = AtomicFileWriter.read_json(res_status)
            if data:
                return JobStatus(data.get("status", JobStatus.COMPLETED.value)), data

        # Check job queue dir
        job_status = self.jobs_dir / job_id / "status.json"
        if job_status.exists():
            data = AtomicFileWriter.read_json(job_status)
            if data:
                return JobStatus(data.get("status", JobStatus.PENDING.value)), data

        return JobStatus.FAILED, {"error": "Job not found"}

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: float = 0.0,
        message: str = "",
        error: Optional[str] = None
    ) -> None:
        """Atomically updates job status and progress."""
        now_iso = datetime.now(timezone.utc).isoformat()
        status_payload = {
            "job_id": job_id,
            "status": status.value,
            "progress_percent": round(progress, 1),
            "updated_at": now_iso,
            "message": message,
            "error": error
        }

        # Update in jobs dir if exists
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            AtomicFileWriter.write_json(job_dir / "status.json", status_payload)
            job_meta_file = job_dir / "job.json"
            if job_meta_file.exists():
                meta_data = AtomicFileWriter.read_json(job_meta_file)
                if meta_data:
                    meta_data["status"] = status.value
                    meta_data["updated_at"] = now_iso
                    meta_data["progress_percent"] = round(progress, 1)
                    if error:
                        meta_data["error_message"] = error
                    AtomicFileWriter.write_json(job_meta_file, meta_data)

        # If completed or failed, also write into results dir
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            res_dir = self.results_dir / job_id
            res_dir.mkdir(parents=True, exist_ok=True)
            AtomicFileWriter.write_json(res_dir / "status.json", status_payload)

    def write_results(
        self,
        job_id: str,
        events: List[Dict[str, Any]],
        report: Optional[Dict[str, Any]] = None
    ) -> None:
        """Writes AI visual understanding results to results/<job_id>/events.json atomically."""
        res_dir = self.results_dir / job_id
        res_dir.mkdir(parents=True, exist_ok=True)

        AtomicFileWriter.write_json(res_dir / "events.json", events)
        if report:
            AtomicFileWriter.write_json(res_dir / "report.json", report)

        self.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=100.0,
            message=f"Completed AI analysis with {len(events)} detected events"
        )

    def fetch_results(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        """Reads completed results from Google Drive results directory."""
        events_file = self.results_dir / job_id / "events.json"
        if not events_file.exists():
            return None
        return AtomicFileWriter.read_json(events_file)

    def archive_job(self, job_id: str) -> bool:
        """
        Moves completed job from jobs/ to archive/.
        Crucial requirement: Never deletes results until confirmed.
        """
        job_dir = self.jobs_dir / job_id
        if not job_dir.exists():
            return False

        target_dir = self.archive_dir / job_id
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        shutil.move(str(job_dir), str(target_dir))
        return True

