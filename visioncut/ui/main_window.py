"""
Main Desktop Application Window for VisionCut AI Cloud.
Built with PySide6, providing a polished dark-themed interface across
Dashboard, Video Import, AI Settings, Editing Preview, and Export screens.
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget,
    QPushButton, QLabel, QFrame, QFileDialog, QProgressBar,
    QSlider, QSpinBox, QComboBox, QScrollArea, QMessageBox,
    QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QIcon

from visioncut.core.config import AppConfig
from visioncut.core.models import (
    VideoMetadata, FrameSample, JobStatus, JobMetadata,
    EditingPlan, UIEvent, ConfidenceLevel
)
from visioncut.core.ffmpeg_utils import FFmpegRunner
from visioncut.preprocessing.video_analyzer import VideoAnalyzer
from visioncut.queue.drive_queue import DriveJobQueue
from visioncut.engine.visual_understanding import VisualUnderstandingEngine
from visioncut.engine.smart_editor import SmartVideoEditor
from visioncut.kdenlive.project_generator import KdenliveProjectGenerator
from visioncut.kdenlive.validator import ProjectValidator
from visioncut.ui.styles import DARK_THEME_QSS
from visioncut.ui.widgets.event_table import DecisionTableWidget


class PreprocessWorker(QThread):
    progress = Signal(float, str)
    finished = Signal(object, object, object, str)  # meta, samples, audio_analysis, cs_path
    error = Signal(str)

    def __init__(self, video_path: str, output_dir: str, config: AppConfig):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config

    def run(self):
        try:
            analyzer = VideoAnalyzer(self.config.preprocessing)
            meta, samples, audio_analysis, cs_path = analyzer.preprocess_video(
                video_path=self.video_path,
                output_dir=self.output_dir,
                progress_callback=lambda p, msg: self.progress.emit(p, msg)
            )
            self.finished.emit(meta, samples, audio_analysis, cs_path)
        except Exception as e:
            self.error.emit(str(e))


class CloudPollWorker(QThread):
    progress = Signal(float, str)
    job_completed = Signal(list)  # raw events
    job_failed = Signal(str)

    def __init__(self, queue: DriveJobQueue, job_id: str, poll_interval_sec: float = 2.0):
        super().__init__()
        self.queue = queue
        self.job_id = job_id
        self.poll_interval = poll_interval_sec
        self.running = True

    def run(self):
        while self.running:
            status, data = self.queue.get_job_status(self.job_id)
            pct = data.get("progress_percent", 0.0)
            msg = data.get("message", "Processing in cloud...")
            self.progress.emit(pct, msg)

            if status == JobStatus.COMPLETED:
                results = self.queue.fetch_results(self.job_id)
                self.job_completed.emit(results or [])
                break
            elif status == JobStatus.FAILED:
                self.job_failed.emit(data.get("error", "Cloud processing failed"))
                break

            self.msleep(int(self.poll_interval * 1000))

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    """Main application window for VisionCut AI Cloud."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionCut AI Cloud - Automatic Software Tutorial Editor")
        self.resize(1280, 820)
        self.setMinimumSize(1024, 680)

        self.config = AppConfig.load()
        self.queue = DriveJobQueue(self.config.drive_sync_dir)

        # Runtime Session State
        self.current_video_path: Optional[str] = None
        self.current_metadata: Optional[VideoMetadata] = None
        self.current_samples: List[FrameSample] = []
        self.current_audio_analysis = None
        self.current_contact_sheet: Optional[str] = None
        self.current_job_id: Optional[str] = None
        self.current_plan: Optional[EditingPlan] = None
        self.generated_project_path: Optional[str] = None

        self._setup_ui()
        self.setStyleSheet(DARK_THEME_QSS)

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # Content Pages Stack
        self.pages_stack = QStackedWidget()
        self.pages_stack.addWidget(self._create_dashboard_page())
        self.pages_stack.addWidget(self._create_import_page())
        self.pages_stack.addWidget(self._create_settings_page())
        self.pages_stack.addWidget(self._create_preview_page())
        self.pages_stack.addWidget(self._create_export_page())

        main_layout.addWidget(self.pages_stack, stretch=1)

    def _create_sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(230)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(8)

        # App Brand
        brand_lbl = QLabel("VISIONCUT AI")
        brand_lbl.setObjectName("titleLabel")
        brand_lbl.setStyleSheet("color: #00E5FF; font-size: 18px; font-weight: 800; letter-spacing: 1px;")
        layout.addWidget(brand_lbl)

        sub_lbl = QLabel("CLOUD WORKER EDITION")
        sub_lbl.setObjectName("subtitleLabel")
        sub_lbl.setStyleSheet("font-size: 10px; font-weight: 600; margin-bottom: 20px;")
        layout.addWidget(sub_lbl)

        # Nav Buttons
        self.nav_buttons = []
        tabs = [
            ("Dashboard", 0),
            ("Video Import", 1),
            ("AI Settings", 2),
            ("Editing Preview", 3),
            ("Export Project", 4)
        ]
        for name, idx in tabs:
            btn = QPushButton(f"  {name}")
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setChecked(idx == 0)
            btn.clicked.connect(lambda checked=False, i=idx: self.switch_page(i))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()

        # Cloud Status Indicator in Sidebar
        status_box = QFrame()
        status_box.setStyleSheet("background: #14161F; border-radius: 8px; padding: 10px;")
        st_layout = QVBoxLayout(status_box)
        st_layout.setContentsMargins(8, 8, 8, 8)

        st_title = QLabel("CLOUD STATUS")
        st_title.setStyleSheet("font-size: 10px; color: #72798F; font-weight: 700;")
        st_layout.addWidget(st_title)

        mode_name = "Google Drive" if self.config.drive_mode == "desktop_sync" else "Local Queue"
        self.side_status_lbl = QLabel(f"Connected: {mode_name}")
        self.side_status_lbl.setStyleSheet("color: #00E676; font-size: 11px; font-weight: 600;")
        st_layout.addWidget(self.side_status_lbl)

        layout.addWidget(status_box)
        return frame

    def switch_page(self, index: int):
        self.pages_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    # ==========================================================
    # 1. DASHBOARD PAGE
    # ==========================================================
    def _create_dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        # Header
        h_box = QHBoxLayout()
        v_title = QVBoxLayout()
        title = QLabel("Dashboard")
        title.setObjectName("titleLabel")
        sub = QLabel("Overview of your tutorial editing jobs and cloud GPU status")
        sub.setObjectName("subtitleLabel")
        v_title.addWidget(title)
        v_title.addWidget(sub)
        h_box.addLayout(v_title)
        h_box.addStretch()

        new_proj_btn = QPushButton("+ New Tutorial Project")
        new_proj_btn.setProperty("class", "primaryBtn")
        new_proj_btn.clicked.connect(lambda: self.switch_page(1))
        h_box.addWidget(new_proj_btn)
        layout.addLayout(h_box)

        # Stats Cards Grid
        grid = QGridLayout()
        grid.setSpacing(16)

        cards_data = [
            ("CLOUD WORKER", "Google Colab", "Free T4 GPU Ready"),
            ("VISION MODEL", "Florence-2-base", "Open Weights (MIT)"),
            ("QUEUE MODE", self.config.drive_mode.title(), self.config.drive_sync_dir or "Ready"),
            ("TARGET SPECS", "1080p @ 30 FPS", "Zero-Cost Local Preprocessing")
        ]
        for col, (head, val, desc) in enumerate(cards_data):
            card = QFrame()
            card.setProperty("class", "card")
            c_layout = QVBoxLayout(card)
            t_lbl = QLabel(head)
            t_lbl.setObjectName("statTitle")
            v_lbl = QLabel(val)
            v_lbl.setObjectName("statValue")
            d_lbl = QLabel(desc)
            d_lbl.setObjectName("subtitleLabel")
            c_layout.addWidget(t_lbl)
            c_layout.addWidget(v_lbl)
            c_layout.addWidget(d_lbl)
            grid.addWidget(card, 0, col)

        layout.addLayout(grid)

        # Workflow Banner
        banner = QFrame()
        banner.setProperty("class", "card")
        b_layout = QVBoxLayout(banner)
        b_title = QLabel("Daily Automatic Workflow")
        b_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #FFFFFF;")
        b_desc = QLabel(
            "1. Select a screen recording  ->  2. Local preprocessing & Drive Queue  ->  "
            "3. Run free Colab GPU Worker  ->  4. Review annotations  ->  5. Export Kdenlive project"
        )
        b_desc.setStyleSheet("color: #00E5FF; font-size: 13px; font-weight: 500; margin-top: 6px;")
        b_layout.addWidget(b_title)
        b_layout.addWidget(b_desc)
        layout.addWidget(banner)

        layout.addStretch()
        return page

    # ==========================================================
    # 2. VIDEO IMPORT PAGE
    # ==========================================================
    def _create_import_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        title = QLabel("Video Import & Local Preprocessing")
        title.setObjectName("titleLabel")
        sub = QLabel("Select your software tutorial screen recording. The original video remains strictly on your laptop.")
        sub.setObjectName("subtitleLabel")
        layout.addWidget(title)
        layout.addWidget(sub)

        # File Selector Card
        card = QFrame()
        card.setProperty("class", "card")
        c_layout = QVBoxLayout(card)

        f_box = QHBoxLayout()
        self.file_path_lbl = QLabel("No video selected")
        self.file_path_lbl.setStyleSheet("color: #8E94A8; font-family: monospace;")
        browse_btn = QPushButton("Browse Screen Recording...")
        browse_btn.setProperty("class", "secondaryBtn")
        browse_btn.clicked.connect(self._browse_video)
        f_box.addWidget(self.file_path_lbl, stretch=1)
        f_box.addWidget(browse_btn)
        c_layout.addLayout(f_box)

        # Metadata Card
        self.meta_frame = QFrame()
        self.meta_frame.setVisible(False)
        m_layout = QGridLayout(self.meta_frame)

        self.lbl_res = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_dur = QLabel("-")
        self.lbl_audio = QLabel("-")

        m_layout.addWidget(QLabel("Resolution:"), 0, 0)
        m_layout.addWidget(self.lbl_res, 0, 1)
        m_layout.addWidget(QLabel("Frame Rate:"), 0, 2)
        m_layout.addWidget(self.lbl_fps, 0, 3)
        m_layout.addWidget(QLabel("Duration:"), 1, 0)
        m_layout.addWidget(self.lbl_dur, 1, 1)
        m_layout.addWidget(QLabel("Audio Stream:"), 1, 2)
        m_layout.addWidget(self.lbl_audio, 1, 3)

        c_layout.addWidget(self.meta_frame)
        layout.addWidget(card)

        # Preprocessing Progress
        self.prep_progress_box = QFrame()
        self.prep_progress_box.setProperty("class", "card")
        self.prep_progress_box.setVisible(False)
        p_layout = QVBoxLayout(self.prep_progress_box)

        self.prep_msg_lbl = QLabel("Initializing...")
        self.prep_bar = QProgressBar()
        self.prep_bar.setRange(0, 100)
        p_layout.addWidget(self.prep_msg_lbl)
        p_layout.addWidget(self.prep_bar)
        layout.addWidget(self.prep_progress_box)

        # Action Buttons
        act_box = QHBoxLayout()
        act_box.addStretch()

        self.btn_preprocess = QPushButton("Analyze & Queue to Cloud")
        self.btn_preprocess.setProperty("class", "primaryBtn")
        self.btn_preprocess.setEnabled(False)
        self.btn_preprocess.clicked.connect(self._start_preprocessing)
        act_box.addWidget(self.btn_preprocess)

        layout.addLayout(act_box)
        layout.addStretch()
        return page

    def _browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Screen Recording", "", "Video Files (*.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self._load_video_file(file_path)

    def _load_video_file(self, file_path: str):
        self.current_video_path = file_path
        self.file_path_lbl.setText(file_path)

        runner = FFmpegRunner()
        try:
            meta = runner.get_video_metadata(file_path)
            self.current_metadata = meta

            self.lbl_res.setText(f"{meta.width} x {meta.height}")
            self.lbl_fps.setText(f"{meta.fps:.1f} FPS")
            mins = int(meta.duration_sec // 60)
            secs = int(meta.duration_sec % 60)
            self.lbl_dur.setText(f"{mins:02d}:{secs:02d} ({meta.duration_sec:.1f}s)")
            self.lbl_audio.setText(f"{meta.audio_codec.upper()} {meta.audio_sample_rate}Hz" if meta.has_audio else "No Audio")

            self.meta_frame.setVisible(True)
            self.btn_preprocess.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Metadata Error", f"Could not read video streams:\n{e}")

    def _start_preprocessing(self):
        if not self.current_video_path:
            return

        self.prep_progress_box.setVisible(True)
        self.btn_preprocess.setEnabled(False)

        temp_prep_dir = str(Path(self.config.app_data_dir) / "staging" / Path(self.current_video_path).stem)

        self.prep_thread = PreprocessWorker(self.current_video_path, temp_prep_dir, self.config)
        self.prep_thread.progress.connect(self._on_prep_progress)
        self.prep_thread.finished.connect(self._on_prep_finished)
        self.prep_thread.error.connect(self._on_prep_error)
        self.prep_thread.start()

    def _on_prep_progress(self, progress: float, msg: str):
        self.prep_bar.setValue(int(progress * 100))
        self.prep_msg_lbl.setText(msg)

    def _on_prep_finished(self, meta: VideoMetadata, samples: List[FrameSample], audio_analysis, cs_path: str):
        self.current_metadata = meta
        self.current_samples = samples
        self.current_audio_analysis = audio_analysis
        self.current_contact_sheet = cs_path

        # Submit Job to Google Drive Queue
        self.prep_msg_lbl.setText("Submitting job to Google Drive Queue...")
        staging_dir = Path(self.config.app_data_dir) / "staging" / Path(self.current_video_path).stem / "frames"

        job_meta = self.queue.submit_job(
            video_metadata=meta,
            sampled_frames=samples,
            frames_source_dir=str(staging_dir),
            settings={
                "smart_cut_sensitivity": self.config.smart_cut_sensitivity,
                "annotation_intensity": self.config.annotation_intensity
            }
        )
        self.current_job_id = job_meta.job_id

        # Update Preview UI Contact Sheet
        if os.path.exists(cs_path):
            pix = QPixmap(cs_path).scaledToWidth(800, Qt.TransformationMode.SmoothTransformation)
            self.cs_viewer.setPixmap(pix)

        # Switch to Preview screen & start cloud polling
        self.switch_page(3)
        self._start_cloud_polling(job_meta.job_id)

    def _on_prep_error(self, err_msg: str):
        self.btn_preprocess.setEnabled(True)
        QMessageBox.critical(self, "Preprocessing Error", f"Local preprocessing failed:\n{err_msg}")

    # ==========================================================
    # 3. AI SETTINGS PAGE
    # ==========================================================
    def _create_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        title = QLabel("AI Analysis & Motion Graphics Settings")
        title.setObjectName("titleLabel")
        sub = QLabel("Configure sensitivity thresholds for automatic cuts, vector animations, and sound effects.")
        sub.setObjectName("subtitleLabel")
        layout.addWidget(title)
        layout.addWidget(sub)

        card = QFrame()
        card.setProperty("class", "card")
        c_layout = QGridLayout(card)
        c_layout.setSpacing(16)

        # Smart Cut Sensitivity
        c_layout.addWidget(QLabel("Smart Cut Sensitivity:"), 0, 0)
        self.slider_cut = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut.setRange(10, 100)
        self.slider_cut.setValue(int(self.config.smart_cut_sensitivity * 100))
        c_layout.addWidget(self.slider_cut, 0, 1)

        # Annotation Intensity
        c_layout.addWidget(QLabel("Annotation Intensity:"), 1, 0)
        self.slider_ann = QSlider(Qt.Orientation.Horizontal)
        self.slider_ann.setRange(10, 100)
        self.slider_ann.setValue(int(self.config.annotation_intensity * 100))
        c_layout.addWidget(self.slider_ann, 1, 1)

        # Zoom Intensity
        c_layout.addWidget(QLabel("Zoom Intensity:"), 2, 0)
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(10, 100)
        self.slider_zoom.setValue(int(self.config.zoom_intensity * 100))
        c_layout.addWidget(self.slider_zoom, 2, 1)

        # SFX Intensity
        c_layout.addWidget(QLabel("Sound Effects Volume:"), 3, 0)
        self.slider_sfx = QSlider(Qt.Orientation.Horizontal)
        self.slider_sfx.setRange(10, 100)
        self.slider_sfx.setValue(int(self.config.sfx_intensity * 100))
        c_layout.addWidget(self.slider_sfx, 3, 1)

        layout.addWidget(card)

        btn_save = QPushButton("Save Settings")
        btn_save.setProperty("class", "primaryBtn")
        btn_save.clicked.connect(self._save_settings)
        layout.addWidget(btn_save)

        layout.addStretch()
        return page

    def _save_settings(self):
        self.config.smart_cut_sensitivity = self.slider_cut.value() / 100.0
        self.config.annotation_intensity = self.slider_ann.value() / 100.0
        self.config.zoom_intensity = self.slider_zoom.value() / 100.0
        self.config.sfx_intensity = self.slider_sfx.value() / 100.0
        self.config.save()
        QMessageBox.information(self, "Settings Saved", "Settings successfully saved!")

    # ==========================================================
    # 4. EDITING PREVIEW PAGE
    # ==========================================================
    def _create_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        title = QLabel("AI Editing Decision Review")
        title.setObjectName("titleLabel")
        sub = QLabel("Review detected UI actions, vector annotations, and conservative smart cuts before project export.")
        sub.setObjectName("subtitleLabel")
        layout.addWidget(title)
        layout.addWidget(sub)

        # Cloud Worker Status Banner
        self.worker_banner = QFrame()
        self.worker_banner.setProperty("class", "card")
        w_layout = QVBoxLayout(self.worker_banner)

        self.w_status_lbl = QLabel("Awaiting Cloud Worker analysis...")
        self.w_status_lbl.setStyleSheet("font-weight: 700; color: #00E5FF;")
        self.w_progress = QProgressBar()
        self.w_progress.setRange(0, 100)

        # Direct Colab notebook launcher action
        colab_h = QHBoxLayout()
        colab_btn = QPushButton("Open Colab Worker Notebook")
        colab_btn.setProperty("class", "secondaryBtn")
        colab_btn.clicked.connect(self._open_colab_notebook)
        colab_h.addWidget(self.w_status_lbl, stretch=1)
        colab_h.addWidget(colab_btn)

        w_layout.addLayout(colab_h)
        w_layout.addWidget(self.w_progress)
        layout.addWidget(self.worker_banner)

        # Decision Table
        self.decision_table = DecisionTableWidget()
        layout.addWidget(self.decision_table, stretch=1)

        # Contact Sheet Preview Card (Collapsible scroll area)
        self.cs_scroll = QScrollArea()
        self.cs_scroll.setFixedHeight(180)
        self.cs_scroll.setWidgetResizable(True)
        self.cs_viewer = QLabel("Contact Sheet Preview will appear here.")
        self.cs_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cs_scroll.setWidget(self.cs_viewer)
        layout.addWidget(self.cs_scroll)

        # Bottom Bar: Approve All / Reject All & Proceed to Export
        b_bar = QHBoxLayout()
        btn_sel_all = QPushButton("Approve All")
        btn_sel_all.setProperty("class", "secondaryBtn")
        btn_sel_all.clicked.connect(lambda: self.decision_table.select_all(True))

        btn_rej_all = QPushButton("Reject All")
        btn_rej_all.setProperty("class", "secondaryBtn")
        btn_rej_all.clicked.connect(lambda: self.decision_table.select_all(False))

        b_bar.addWidget(btn_sel_all)
        b_bar.addWidget(btn_rej_all)
        b_bar.addStretch()

        self.btn_goto_export = QPushButton("Proceed to Kdenlive Export ->")
        self.btn_goto_export.setProperty("class", "successBtn")
        self.btn_goto_export.clicked.connect(lambda: self.switch_page(4))
        b_bar.addWidget(self.btn_goto_export)

        layout.addLayout(b_bar)
        return page

    def _open_colab_notebook(self):
        colab_nb = Path(__file__).parent.parent / "cloud" / "visioncut_colab_worker.ipynb"
        QMessageBox.information(
            self,
            "Google Colab Worker",
            f"Colab Notebook located at:\n{colab_nb}\n\n"
            "Open Google Colab (colab.research.google.com), upload this notebook, "
            "and click 'Runtime -> Run all' to process queued jobs with free GPU!"
        )

    def _start_cloud_polling(self, job_id: str):
        self.poll_thread = CloudPollWorker(self.queue, job_id)
        self.poll_thread.progress.connect(self._on_cloud_progress)
        self.poll_thread.job_completed.connect(self._on_cloud_completed)
        self.poll_thread.job_failed.connect(self._on_cloud_failed)
        self.poll_thread.start()

    def _on_cloud_progress(self, pct: float, msg: str):
        self.w_progress.setValue(int(pct))
        self.w_status_lbl.setText(f"Cloud Status: {msg} ({pct:.0f}%)")

    def _on_cloud_completed(self, raw_events: List[Dict[str, Any]]):
        self.w_progress.setValue(100)
        self.w_status_lbl.setText("Cloud AI Analysis Complete! Review suggestions below:")

        # Parse with Visual Understanding Engine
        engine = VisualUnderstandingEngine(self.config.smart_cut_sensitivity)
        parsed_events = engine.parse_raw_events(raw_events)

        # Build Editing Plan with Smart Editor
        editor = SmartVideoEditor(self.config.smart_cut_sensitivity)
        plan = editor.build_editing_plan(
            self.current_metadata,
            parsed_events,
            self.current_samples,
            self.current_audio_analysis
        )
        self.current_plan = plan

        # Load into Decision Table
        self.decision_table.load_decisions(plan.decisions)

    def _on_cloud_failed(self, err_msg: str):
        self.w_status_lbl.setText(f"Cloud Analysis Notice: {err_msg}")

    # ==========================================================
    # 5. EXPORT PAGE
    # ==========================================================
    def _create_export_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        title = QLabel("Kdenlive Project Export & Validation")
        title.setObjectName("titleLabel")
        sub = QLabel("Generate fully editable Kdenlive project files with separate timeline tracks (V1-V5, A1-A2).")
        sub.setObjectName("subtitleLabel")
        layout.addWidget(title)
        layout.addWidget(sub)

        # Action Card
        card = QFrame()
        card.setProperty("class", "card")
        c_layout = QVBoxLayout(card)

        btn_generate = QPushButton("Generate Kdenlive Project")
        btn_generate.setProperty("class", "primaryBtn")
        btn_generate.setFixedHeight(45)
        btn_generate.clicked.connect(self._generate_kdenlive_project)
        c_layout.addWidget(btn_generate)

        # Validation Report Status
        self.val_status_card = QFrame()
        self.val_status_card.setVisible(False)
        v_layout = QVBoxLayout(self.val_status_card)

        self.val_header_lbl = QLabel("Validation Status: -")
        self.val_header_lbl.setStyleSheet("font-weight: 700; font-size: 15px;")
        self.val_summary_lbl = QLabel("-")
        self.val_summary_lbl.setStyleSheet("color: #8E94A8;")

        v_layout.addWidget(self.val_header_lbl)
        v_layout.addWidget(self.val_summary_lbl)
        c_layout.addWidget(self.val_status_card)

        layout.addWidget(card)

        # Open Buttons
        self.export_actions_frame = QFrame()
        self.export_actions_frame.setVisible(False)
        ea_layout = QHBoxLayout(self.export_actions_frame)

        btn_open_kdenlive = QPushButton("Open Project in Kdenlive")
        btn_open_kdenlive.setProperty("class", "successBtn")
        btn_open_kdenlive.clicked.connect(self._open_in_kdenlive)

        btn_open_folder = QPushButton("Open Output Folder")
        btn_open_folder.setProperty("class", "secondaryBtn")
        btn_open_folder.clicked.connect(self._open_output_folder)

        ea_layout.addWidget(btn_open_kdenlive)
        ea_layout.addWidget(btn_open_folder)
        layout.addWidget(self.export_actions_frame)

        layout.addStretch()
        return page

    def _generate_kdenlive_project(self):
        if not self.current_plan or not self.current_video_path:
            QMessageBox.warning(self, "No Active Plan", "Please import and analyze a video recording first.")
            return

        proj_name = Path(self.current_video_path).stem
        out_proj_dir = str(Path(self.config.output_dir) / proj_name)

        generator = KdenliveProjectGenerator()
        kdenlive_file, summary = generator.generate_project(
            plan=self.current_plan,
            source_video_path=self.current_video_path,
            output_project_dir=out_proj_dir,
            project_title=proj_name
        )
        self.generated_project_path = kdenlive_file

        # Validate Project
        validator = ProjectValidator()
        report = validator.validate_project(kdenlive_file)

        self.val_status_card.setVisible(True)
        self.export_actions_frame.setVisible(True)

        if report.is_valid:
            self.val_header_lbl.setText("Validation: PASSED [100% Verified]")
            self.val_header_lbl.setStyleSheet("font-weight: 700; font-size: 15px; color: #00E676;")
            self.val_summary_lbl.setText(
                f"Generated 7 timeline tracks, {summary['cuts_count']} cuts, "
                f"{summary['annotations_count']} vector SVG graphics, {summary['sfx_count']} SFX clips."
            )
        else:
            self.val_header_lbl.setText("Validation: FAILED")
            self.val_header_lbl.setStyleSheet("font-weight: 700; font-size: 15px; color: #FF3366;")
            self.val_summary_lbl.setText(f"Errors: {', '.join(report.errors)}")

    def _open_in_kdenlive(self):
        if not self.generated_project_path:
            return
        kp = self.config.kdenlive_path
        if kp and os.path.exists(kp):
            subprocess.Popen([kp, self.generated_project_path])
        else:
            # Try default association on Windows
            try:
                os.startfile(self.generated_project_path)
            except Exception as e:
                QMessageBox.information(
                    self,
                    "Kdenlive Executable",
                    f"Kdenlive is not associated with .kdenlive files or installed at the default path.\n\n"
                    f"Project file generated at:\n{self.generated_project_path}\n\n"
                    "Please install Kdenlive or set its path in the config."
                )

    def _open_output_folder(self):
        if self.generated_project_path:
            folder = str(Path(self.generated_project_path).parent)
            os.startfile(folder)

