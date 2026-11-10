"""
Application configuration for VisionCut AI Cloud.
Manages persistent settings, paths, defaults, and thresholds.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


DEFAULT_APP_DATA_DIR = Path.home() / ".visioncut"
DEFAULT_WORKSPACE_DIR = Path.home() / "VisionCut_Output"
DEFAULT_DRIVE_FOLDER_NAME = "VisionCutCloud"


@dataclass
class PreprocessingSettings:
    initial_sample_rate_fps: float = 1.0       # 1 frame per second baseline
    scene_change_threshold: float = 0.35        # Histogram / pixel difference ratio
    phash_similarity_threshold: int = 6         # Hamming distance <= 6 is near-identical
    static_period_min_seconds: float = 3.0      # >= 3 sec without change is a static period
    silence_db_threshold: float = -42.0         # dB RMS for silence
    silence_min_duration_sec: float = 1.5       # Accidental gap minimum (1.5s+)
    clipping_db_threshold: float = -0.5         # Audio clipping limit
    thumbnail_width: int = 960                  # Reduced width for cloud upload
    thumbnail_height: int = 540                 # Reduced height for cloud upload
    jpeg_quality: int = 85                      # Preserve sharp text readability for VLM
    contact_sheet_cols: int = 6                 # Columns in review contact sheet
    max_frames_per_cloud_batch: int = 30        # Free Colab T4 friendly batch size


@dataclass
class MotionGraphicsSettings:
    primary_color: str = "#FFCC00"             # Neon Gold/Yellow (high visibility)
    secondary_color: str = "#00E5FF"           # Cyan accent
    alert_color: str = "#FF3366"               # Crimson Red (errors/warnings)
    success_color: str = "#00E676"             # Emerald Green
    default_border_width: int = 4
    default_font_family: str = "Segoe UI, Arial, sans-serif"
    zoom_default_factor: float = 1.35          # Default zoom scale
    zoom_transition_duration: float = 0.45     # Smooth ease in/out seconds


@dataclass
class AppConfig:
    app_data_dir: str = str(DEFAULT_APP_DATA_DIR)
    output_dir: str = str(DEFAULT_WORKSPACE_DIR)
    drive_sync_dir: str = ""                    # e.g. G:\My Drive\VisionCutCloud
    drive_mode: str = "desktop_sync"            # "desktop_sync", "api", "local_sim"
    kdenlive_path: str = ""                     # Path to kdenlive.exe if custom
    vlm_model_id: str = "microsoft/Florence-2-base"
    smart_cut_sensitivity: float = 0.70         # 0.0 to 1.0
    annotation_intensity: float = 0.80          # 0.0 to 1.0
    zoom_intensity: float = 0.75                # 0.0 to 1.0
    sfx_intensity: float = 0.80                 # 0.0 to 1.0
    preprocessing: PreprocessingSettings = field(default_factory=PreprocessingSettings)
    motion_graphics: MotionGraphicsSettings = field(default_factory=MotionGraphicsSettings)

    @classmethod
    def get_config_path(cls) -> Path:
        p = Path(DEFAULT_APP_DATA_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p / "config.json"

    @classmethod
    def load(cls) -> AppConfig:
        config_path = cls.get_config_path()
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prep_data = data.pop("preprocessing", {})
                mg_data = data.pop("motion_graphics", {})
                return cls(
                    preprocessing=PreprocessingSettings(**prep_data),
                    motion_graphics=MotionGraphicsSettings(**mg_data),
                    **data
                )
            except Exception:
                pass
        cfg = cls()
        cfg.detect_environment()
        cfg.save()
        return cfg

    def save(self) -> None:
        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def detect_environment(self) -> None:
        """Auto-detects Google Drive desktop folder and Kdenlive installation."""
        # 1. Check Google Drive for Desktop standard mount paths (G:, H:, or ~/Google Drive)
        potential_drive_roots = [
            Path("G:/My Drive"),
            Path("H:/My Drive"),
            Path.home() / "Google Drive" / "My Drive",
            Path.home() / "Google Drive",
        ]
        for root in potential_drive_roots:
            if root.exists():
                cloud_folder = root / DEFAULT_DRIVE_FOLDER_NAME
                self.drive_sync_dir = str(cloud_folder)
                self.drive_mode = "desktop_sync"
                break
        if not self.drive_sync_dir:
            # Fallback to local simulation path inside app data
            sim_dir = Path(self.app_data_dir) / DEFAULT_DRIVE_FOLDER_NAME
            self.drive_sync_dir = str(sim_dir)
            self.drive_mode = "local_sim"

        # 2. Check common Kdenlive install paths on Windows
        potential_kdenlive = [
            Path("C:/Program Files/kdenlive/bin/kdenlive.exe"),
            Path("C:/Program Files/Kdenlive/bin/kdenlive.exe"),
            Path("C:/Program Files (x86)/kdenlive/bin/kdenlive.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "kdenlive" / "bin" / "kdenlive.exe"
        ]
        for kp in potential_kdenlive:
            if kp.exists():
                self.kdenlive_path = str(kp)
                break
