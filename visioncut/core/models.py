"""
Core data models for VisionCut AI Cloud.
Defines schemas for video metadata, frames, visual events, editing decisions,
annotations, sound effects, job queue states, and project validation reports.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UIEventType(str, Enum):
    BROWSER_NAV = "browser_navigation"
    CODE_EDIT = "code_editor_change"
    TERMINAL_EXEC = "terminal_execution"
    SETTINGS_CHANGE = "settings_change"
    BUTTON_CLICK = "button_click"
    MENU_SELECT = "menu_select"
    DIALOG_POPUP = "dialog_popup"
    ERROR_ALERT = "error_alert"
    RESULT_DISPLAY = "result_display"
    REPETITIVE_ACTION = "repetitive_action"
    IDLE_WAIT = "unnecessary_waiting"
    UNNECESSARY_MOUSE = "unnecessary_mouse_movement"
    UI_ELEMENT_FOCUS = "ui_element_focus"
    SCENE_TRANSITION = "scene_transition"


class EditingActionType(str, Enum):
    CUT_GAP = "cut_accidental_gap"
    SHORTEN_WAIT = "shorten_waiting_period"
    ZOOM_PAN = "zoom_and_pan"
    HIGHLIGHT_BOX = "highlight_box"
    DRAW_ARROW = "draw_arrow"
    DRAW_CIRCLE = "draw_circle"
    TEXT_CALLOUT = "text_callout"
    CURSOR_RIPPLE = "cursor_ripple"
    PLAY_SFX = "play_sound_effect"
    ADD_MARKER = "add_review_marker"
    PRESERVE_ORIGINAL = "preserve_original"


class ConfidenceLevel(str, Enum):
    HIGH = "high"       # >= 0.85: Apply automatically
    MEDIUM = "medium"   # 0.50 - 0.84: Add review marker
    LOW = "low"         # < 0.50: Preserve original footage


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SFXType(str, Enum):
    MOUSE_CLICK = "mouse_click"
    SOFT_POP = "soft_pop"
    NOTIFICATION = "notification"
    WHOOSH = "whoosh"
    SUCCESS_CHIME = "success_chime"
    ERROR_ALERT = "error_alert"
    SUBTLE_EMPHASIS = "subtle_emphasis"


class BoundingBox(BaseModel):
    """
    Normalized bounding box coordinates [ymin, xmin, ymax, xmax],
    where all values are floats in [0.0, 1.0].
    """
    ymin: float = Field(0.0, ge=0.0, le=1.0)
    xmin: float = Field(0.0, ge=0.0, le=1.0)
    ymax: float = Field(1.0, ge=0.0, le=1.0)
    xmax: float = Field(1.0, ge=0.0, le=1.0)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    def to_pixel_rect(self, video_width: int, video_height: int) -> tuple[int, int, int, int]:
        """Returns (x, y, w, h) in integer pixels."""
        x = int(round(self.xmin * video_width))
        y = int(round(self.ymin * video_height))
        w = int(round((self.xmax - self.xmin) * video_width))
        h = int(round((self.ymax - self.ymin) * video_height))
        return (x, y, max(1, w), max(1, h))


class VideoMetadata(BaseModel):
    file_path: str
    filename: str
    duration_sec: float
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    total_frames: int = 0
    has_audio: bool = True
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    video_codec: str = "h264"
    audio_codec: str = "aac"
    file_size_bytes: int = 0


class FrameSample(BaseModel):
    frame_idx: int
    timestamp_sec: float
    image_rel_path: str
    phash: Optional[str] = None
    motion_score: float = 0.0
    is_scene_change: bool = False
    is_static_period: bool = False
    audio_rms: float = 0.0
    is_silence: bool = False
    is_clipped: bool = False
    needs_cloud_analysis: bool = True


class UIEvent(BaseModel):
    event_id: str
    timestamp: float
    duration: float = 1.0
    event_type: UIEventType
    description: str
    ui_target: str
    bbox: BoundingBox = Field(default_factory=BoundingBox)
    confidence: float = 0.90
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    recommended_action: EditingActionType = EditingActionType.PRESERVE_ORIGINAL
    reason: str = ""
    approved: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EditingDecision(BaseModel):
    decision_id: str
    start_time: float
    end_time: float
    action_type: EditingActionType
    speed_multiplier: float = 1.0
    cut_out: bool = False
    target_bbox: Optional[BoundingBox] = None
    text_content: Optional[str] = None
    sfx_type: Optional[SFXType] = None
    confidence: float = 0.90
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    reason: str = ""
    approved: bool = True


class AnnotationItem(BaseModel):
    annotation_id: str
    start_time: float
    end_time: float
    shape_type: str  # "rect", "rounded_rect", "circle", "arrow", "cursor_ripple", "text_callout", "spotlight"
    bbox: BoundingBox
    color: str = "#FFCC00"
    border_color: str = "#FF3366"
    border_width: int = 4
    opacity: float = 0.95
    animation: str = "fade_in_scale"  # "fade_in_scale", "draw_path", "pulse", "static"
    text: Optional[str] = None
    svg_rel_path: Optional[str] = None
    confidence: float = 0.90
    approved: bool = True


class SFXItem(BaseModel):
    sfx_id: str
    timestamp: float
    sfx_type: SFXType
    volume: float = 0.70  # 0.0 to 1.0
    duration: float = 0.35
    audio_rel_path: str = ""
    track: str = "A2"
    approved: bool = True


class EditingPlan(BaseModel):
    plan_id: str
    video_metadata: VideoMetadata
    original_duration: float
    estimated_edited_duration: float
    events: List[UIEvent] = Field(default_factory=list)
    decisions: List[EditingDecision] = Field(default_factory=list)
    annotations: List[AnnotationItem] = Field(default_factory=list)
    sfx_items: List[SFXItem] = Field(default_factory=list)
    cuts: List[Dict[str, float]] = Field(default_factory=list)  # [{"in": start, "out": end}]


class JobMetadata(BaseModel):
    job_id: str
    created_at: str
    updated_at: str
    status: JobStatus = JobStatus.PENDING
    video_filename: str
    total_sampled_frames: int = 0
    settings: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    model_name: str = "microsoft/Florence-2-base"
    error_message: Optional[str] = None
    progress_percent: float = 0.0


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    details: str
    severity: str = "error"  # "error", "warning", "info"


class ProjectValidationReport(BaseModel):
    is_valid: bool = True
    project_path: str
    timestamp: str
    checks: List[ValidationCheck] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary: str = ""

