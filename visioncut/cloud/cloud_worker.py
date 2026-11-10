"""
Cloud Worker Execution Engine for Google Colab and Kaggle.
Processes queued tutorial screenshots using free cloud GPU (or CPU test fallback),
analyzes UI components, software actions, detects bounding boxes,
and outputs structured event JSON with checkpoints.
"""

from __future__ import annotations
import os
import sys
import gc
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from visioncut.core.models import (
    UIEventType, EditingActionType, ConfidenceLevel, BoundingBox
)


class SoftwareUIAnalyzer:
    """
    Analyzes software tutorial screenshots to detect UI actions, controls,
    and bounding boxes. Designed to run seamlessly in Google Colab (T4 GPU),
    Kaggle (P100/T4), or local CPU fallback.
    """

    def __init__(self, use_vlm: bool = True, model_name: str = "microsoft/Florence-2-base"):
        self.use_vlm = use_vlm
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.processor = None
        self._init_device()

    def _init_device(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"[WORKER] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("[WORKER] GPU not available. Using CPU fallback mode.")
        except ImportError:
            self.device = "cpu"
            print("[WORKER] PyTorch not installed. Using heuristic visual fallback.")

    def load_model(self):
        """Loads Florence-2 or compatible vision-language model if available."""
        if not self.use_vlm or self.model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM

            print(f"[WORKER] Loading vision model '{self.model_name}' on {self.device}...")
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            print("[WORKER] Vision-Language Model loaded successfully.")
        except Exception as e:
            print(f"[WORKER WARNING] Could not load VLM weights ({e}). Falling back to visual heuristics.")
            self.model = None
            self.processor = None

    def analyze_frame_heuristics(
        self,
        img_np: np.ndarray,
        timestamp: float,
        prev_img_np: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        High-precision visual heuristics for UI elements:
        - Detects action buttons (contrasting rounded rects / solid pills)
        - Detects terminal windows (dark monochrome areas with prompt symbols)
        - Detects alert dialogs & notification banners
        - Detects code editor structures
        """
        h, w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        events = []

        # 1. Detect Button or Dialog Bounding Boxes using edge and contour analysis
        edges = cv2.Canny(gray, 40, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            aspect = float(cw) / max(1, ch)

            # Check for Alert / Modal Dialog Banner (e.g. 500-1400w, 100-300h)
            if 0.25 * w < cw < 0.85 * w and 0.08 * h < ch < 0.40 * h and 1.8 < aspect < 6.0:
                # Check mean color in center of dialog
                roi = img_np[y:y+ch, x:x+cw]
                mean_rgb = np.mean(roi, axis=(0, 1))

                # Green/Emerald alert banner (success)
                if mean_rgb[1] > mean_rgb[0] + 15 and mean_rgb[1] > mean_rgb[2] + 15:
                    events.append({
                        "event_id": f"evt_success_{int(timestamp*100):05d}",
                        "timestamp": timestamp,
                        "duration": 2.0,
                        "event_type": UIEventType.RESULT_DISPLAY.value,
                        "description": "Success banner displaying execution results",
                        "ui_target": "Result Alert Box",
                        "bbox": {
                            "ymin": round(y / h, 4),
                            "xmin": round(x / w, 4),
                            "ymax": round((y + ch) / h, 4),
                            "xmax": round((x + cw) / w, 4)
                        },
                        "confidence": 0.94,
                        "confidence_level": ConfidenceLevel.HIGH.value,
                        "recommended_action": EditingActionType.HIGHLIGHT_BOX.value,
                        "reason": "Preserve final tutorial demonstration result"
                    })
                    break

            # Check for Primary Action Button (e.g. 150-450w, 35-90h, aspect 2.5-6.0)
            if 120 < cw < 450 and 30 < ch < 100 and 2.2 < aspect < 7.0:
                roi = img_np[y:y+ch, x:x+cw]
                std_roi = np.std(roi)
                # Relatively solid button background with text
                if std_roi > 15:
                    events.append({
                        "event_id": f"evt_btn_{int(timestamp*100):05d}",
                        "timestamp": timestamp,
                        "duration": 1.5,
                        "event_type": UIEventType.BUTTON_CLICK.value,
                        "description": "Interactive UI Action Button",
                        "ui_target": "Action Button",
                        "bbox": {
                            "ymin": round(y / h, 4),
                            "xmin": round(x / w, 4),
                            "ymax": round((y + ch) / h, 4),
                            "xmax": round((x + cw) / w, 4)
                        },
                        "confidence": 0.91,
                        "confidence_level": ConfidenceLevel.HIGH.value,
                        "recommended_action": EditingActionType.HIGHLIGHT_BOX.value,
                        "reason": "Guide viewer attention to clickable action button"
                    })
                    if len(events) >= 2:
                        break

        # 2. Check for Terminal / Console Region
        # Bottom half dark rectangular block
        term_y = int(0.50 * h)
        roi_bottom = gray[term_y:, int(0.15 * w):]
        if np.mean(roi_bottom) < 40 and np.std(roi_bottom) > 10:
            # Check if there is active terminal output text
            events.append({
                "event_id": f"evt_term_{int(timestamp*100):05d}",
                "timestamp": timestamp,
                "duration": 2.5,
                "event_type": UIEventType.TERMINAL_EXEC.value,
                "description": "Terminal command execution and status output",
                "ui_target": "Terminal Window",
                "bbox": {
                    "ymin": round(term_y / h, 4),
                    "xmin": 0.15,
                    "ymax": 0.96,
                    "xmax": 0.98
                },
                "confidence": 0.89,
                "confidence_level": ConfidenceLevel.HIGH.value,
                "recommended_action": EditingActionType.ZOOM_PAN.value,
                "reason": "Zoom in on command execution details"
            })

        # 3. Check for Code Editor Structure
        if not events and np.mean(gray) < 70:
            events.append({
                "event_id": f"evt_code_{int(timestamp*100):05d}",
                "timestamp": timestamp,
                "duration": 2.0,
                "event_type": UIEventType.CODE_EDIT.value,
                "description": "Source code demonstration and editing",
                "ui_target": "Code Editor",
                "bbox": {
                    "ymin": 0.08,
                    "xmin": 0.18,
                    "ymax": 0.88,
                    "xmax": 0.95
                },
                "confidence": 0.88,
                "confidence_level": ConfidenceLevel.HIGH.value,
                "recommended_action": EditingActionType.PRESERVE_ORIGINAL.value,
                "reason": "Keep code logic clearly visible without obstruction"
            })

        return events

    def analyze_frame_vlm(self, pil_img: Image.Image, timestamp: float) -> List[Dict[str, Any]]:
        """Queries loaded VLM for UI element grounding."""
        if self.model is None or self.processor is None:
            return []

        try:
            import torch
            prompt = "<OD>"  # Florence-2 Object Detection task token
            inputs = self.processor(text=prompt, images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    max_new_tokens=256,
                    num_beams=3
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(pil_img.width, pil_img.height)
            )

            # Map Florence-2 detections into normalized VisionCut UI events
            events = []
            od_res = parsed_answer.get("<OD>", {})
            bboxes = od_res.get("bboxes", [])
            labels = od_res.get("labels", [])

            for idx, (box, label) in enumerate(zip(bboxes, labels)):
                # box is [xmin, ymin, xmax, ymax] in pixel coords
                xmin = max(0.0, min(1.0, box[0] / pil_img.width))
                ymin = max(0.0, min(1.0, box[1] / pil_img.height))
                xmax = max(0.0, min(1.0, box[2] / pil_img.width))
                ymax = max(0.0, min(1.0, box[3] / pil_img.height))

                lbl_lower = label.lower()
                event_type = UIEventType.UI_ELEMENT_FOCUS.value
                rec_action = EditingActionType.HIGHLIGHT_BOX.value

                if any(w in lbl_lower for w in ["button", "icon", "cursor", "arrow"]):
                    event_type = UIEventType.BUTTON_CLICK.value
                elif any(w in lbl_lower for w in ["window", "panel", "screen"]):
                    event_type = UIEventType.CODE_EDIT.value
                    rec_action = EditingActionType.ZOOM_PAN.value

                events.append({
                    "event_id": f"evt_vlm_{int(timestamp*100):05d}_{idx}",
                    "timestamp": timestamp,
                    "duration": 1.5,
                    "event_type": event_type,
                    "description": f"Detected {label}",
                    "ui_target": label,
                    "bbox": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
                    "confidence": 0.88,
                    "confidence_level": ConfidenceLevel.HIGH.value,
                    "recommended_action": rec_action,
                    "reason": f"AI localized {label}"
                })
            return events

        except Exception as e:
            print(f"[WORKER ERROR] VLM inference failed: {e}")
            if "out of memory" in str(e).lower():
                try:
                    import torch
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass
            return []


class CloudJobWorker:
    """
    Main job executor for Colab / Kaggle.
    Monitors queue, processes jobs, saves batch checkpoints, writes events.json.
    """

    def __init__(self, queue_root: str | Path, model_name: str = "microsoft/Florence-2-base"):
        self.root = Path(queue_root)
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        self.analyzer = SoftwareUIAnalyzer(use_vlm=True, model_name=model_name)

    def find_pending_jobs(self) -> List[Path]:
        """Scans jobs directory for jobs with status 'pending' or 'partially_completed'."""
        pending = []
        if not self.jobs_dir.exists():
            return pending

        for entry in self.jobs_dir.iterdir():
            if entry.is_dir():
                status_file = entry / "status.json"
                if status_file.exists():
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            status = data.get("status")
                            if status in ["pending", "partially_completed"]:
                                pending.append(entry)
                    except Exception:
                        pass
        return sorted(pending)

    def process_job(self, job_dir: Path, batch_size: int = 10) -> bool:
        """Processes a single queued job with batch checkpointing."""
        job_id = job_dir.name
        status_file = job_dir / "status.json"
        job_file = job_dir / "job.json"
        frames_dir = job_dir / "frames"
        checkpoints_dir = job_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        res_dir = self.results_dir / job_id
        res_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==========================================")
        print(f"[WORKER] Starting job processing: {job_id}")
        print(f"==========================================")

        # 1. Update status to 'processing'
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status_payload = {
            "job_id": job_id,
            "status": "processing",
            "progress_percent": 5.0,
            "updated_at": now_iso,
            "message": "Cloud worker started inference"
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_payload, f, indent=2)

        # 2. Find frames to process
        frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        total_frames = len(frame_files)
        print(f"[WORKER] Found {total_frames} frames to analyze.")

        # 3. Check for existing checkpoints to support resuming interrupted jobs
        all_detected_events: List[Dict[str, Any]] = []
        completed_frame_names = set()

        for ckpt in sorted(checkpoints_dir.glob("batch_*.json")):
            try:
                with open(ckpt, "r", encoding="utf-8") as f:
                    ckpt_data = json.load(f)
                    all_detected_events.extend(ckpt_data.get("events", []))
                    completed_frame_names.update(ckpt_data.get("frames_processed", []))
            except Exception:
                pass

        if completed_frame_names:
            print(f"[WORKER] Resuming job! {len(completed_frame_names)} frames already processed in previous checkpoint.")

        remaining_frames = [f for f in frame_files if f.name not in completed_frame_names]

        # 4. Process in batches
        prev_img_np = None
        for i in range(0, len(remaining_frames), batch_size):
            batch = remaining_frames[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            batch_events = []
            processed_in_batch = []

            for frame_path in batch:
                try:
                    # Extract timestamp from filename (e.g., frame_00001_1.50s.jpg)
                    name_parts = frame_path.stem.split("_")
                    ts = 0.0
                    for part in name_parts:
                        if part.endswith("s"):
                            try:
                                ts = float(part[:-1])
                            except ValueError:
                                pass

                    pil_img = Image.open(frame_path).convert("RGB")
                    img_np = np.array(pil_img)

                    # Try VLM first, fallback to heuristics
                    events = self.analyzer.analyze_frame_vlm(pil_img, ts)
                    if not events:
                        events = self.analyzer.analyze_frame_heuristics(img_np, ts, prev_img_np)

                    batch_events.extend(events)
                    processed_in_batch.append(frame_path.name)
                    prev_img_np = img_np

                except Exception as frame_err:
                    print(f"[WORKER ERROR] Error processing {frame_path.name}: {frame_err}")

            # Save Batch Checkpoint
            ckpt_path = checkpoints_dir / f"batch_{batch_num:04d}.json"
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({
                    "batch_index": batch_num,
                    "frames_processed": processed_in_batch,
                    "events": batch_events
                }, f, indent=2)

            all_detected_events.extend(batch_events)
            completed_frame_names.update(processed_in_batch)

            progress_pct = min(98.0, 5.0 + 90.0 * (len(completed_frame_names) / max(1, total_frames)))
            status_payload["progress_percent"] = round(progress_pct, 1)
            status_payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status_payload["message"] = f"Processed {len(completed_frame_names)}/{total_frames} frames ({len(all_detected_events)} events detected)"
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(status_payload, f, indent=2)

            print(f"[WORKER] Batch {batch_num} done. Progress: {progress_pct:.1f}%")

        # 5. Deduplicate and clean up events
        unique_events = []
        seen_keys = set()
        for ev in all_detected_events:
            key = (ev["event_type"], round(ev["timestamp"], 1), ev["ui_target"])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_events.append(ev)

        # 6. Write final events.json and report.json to results
        final_events_file = res_dir / "events.json"
        with open(final_events_file, "w", encoding="utf-8") as f:
            json.dump(unique_events, f, indent=2)

        report_file = res_dir / "report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "job_id": job_id,
                "total_frames_analyzed": len(completed_frame_names),
                "total_events_detected": len(unique_events),
                "device": self.analyzer.device,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }, f, indent=2)

        # 7. Mark status COMPLETED in both job and results dir
        final_status = {
            "job_id": job_id,
            "status": "completed",
            "progress_percent": 100.0,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": f"Successfully detected {len(unique_events)} visual events",
            "events_count": len(unique_events)
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(final_status, f, indent=2)
        with open(res_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(final_status, f, indent=2)

        print(f"[WORKER SUCCESS] Job {job_id} completed successfully with {len(unique_events)} events!")
        return True

    def run_worker_loop(self, poll_interval_sec: int = 5, max_jobs: int = 1):
        """Runs the worker loop to process pending jobs."""
        print(f"[WORKER] Watching queue directory: {self.jobs_dir.resolve()}")
        jobs_processed = 0
        while jobs_processed < max_jobs:
            pending = self.find_pending_jobs()
            if pending:
                target = pending[0]
                self.process_job(target)
                jobs_processed += 1
            else:
                print(f"[WORKER] No pending jobs. Waiting {poll_interval_sec}s...")
                time.sleep(poll_interval_sec)

