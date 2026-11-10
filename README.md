# VisionCut AI Cloud
### Automatic AI-Powered Software Tutorial Editor for Windows Desktop

**VisionCut AI Cloud** is a complete, professional Windows desktop application designed to automatically analyze software screen recordings using free cloud GPU resources, generate vector visual annotations and motion graphics, and create fully editable **Kdenlive** projects.

---

## 1. Architectural Highlights

```
+---------------------------+
| WINDOWS DESKTOP APP       |  PySide6 Dark UI: Dashboard, Import, AI Settings, Preview, Export
+-------------+-------------+
              | (Lightweight streaming preprocessing, contact sheet, audio silence/clipping)
              v
+-------------+-------------+
| LOCAL PREPROCESSING       |  Preserves original video & audio locally; Never loads 18k frames into RAM
+-------------+-------------+
              | (Compressed 960x540 JPEGs + metadata JSON)
              v
+-------------+-------------+
| GOOGLE DRIVE JOB QUEUE    |  VisionCutCloud/ jobs/, results/, archive/ with atomic state updates
+-------------+-------------+
              | (Free Google Colab T4 GPU or Kaggle Dual-T4 fallback)
              v
+-------------+-------------+
| CLOUD VLM WORKER          |  microsoft/Florence-2-base UI grounding, batch checkpoints, JSON events
+-------------+-------------+
              | (events.json downloaded and verified)
              v
+-------------+-------------+
| SMART EDITING & ASSETS    |  Conservative cuts, SVG vectors (boxes, arrows, callouts), synthesized SFX
+-------------+-------------+
              | (Frame-accurate multi-track project assembly)
              v
+-------------+-------------+
| EDITABLE KDENLIVE PROJECT |  7 Tracks: V5 (Motion Graphics), V4 (Callouts), V3 (Highlights),
+---------------------------+            V2 (Zooms), V1 (Main Video), A2 (SFX), A1 (Original Audio)
```

### Key Engineering Guarantees
- **Zero Cost**: No paid AI APIs, no paid GPU subscriptions, no local neural net inference.
- **Hardware-Friendly**: Runs smoothly on Intel integrated graphics and 16 GB RAM laptops.
- **Privacy & Bandwidth Preservation**: The 1080p source video remains 100% local on your laptop. Only representative sampled frames (~10-30 JPEGs per video) are transferred to the cloud queue.
- **Non-Destructive**: Every editing cut, zoom, arrow, callout, and sound effect is placed on its own timeline track and can be modified or removed in Kdenlive. Original footage is never overwritten or permanently baked.
- **Conservative Cuts**: Accidental gaps are removed only when **both** audio silence (>1.5s) and visual inactivity are confirmed. Code edits, terminal commands, and results are strictly preserved.

---

## 2. Directory Layout

```
Aieditor/
  ├── main.py                          # Main entry point (GUI, --demo, --test)
  ├── README.md                        # Complete system documentation
  └── visioncut/
      ├── core/
      │   ├── config.py                # App configuration & environmental autodetection
      │   ├── models.py                # Pydantic schemas (Events, Decisions, BoundingBox)
      │   └── ffmpeg_utils.py          # FFmpeg wrappers, probes, and contact sheet generators
      ├── preprocessing/
      │   ├── video_analyzer.py        # Streaming adaptive frame sampler & scene detector
      │   └── audio_analyzer.py        # Streaming RMS silence & clipping detector
      ├── queue/
      │   └── drive_queue.py           # Google Drive atomic job queue & checkpoint manager
      ├── cloud/
      │   ├── cloud_worker.py          # Cloud inference engine with Florence-2 & heuristics
      │   ├── visioncut_colab_worker.ipynb   # Ready-to-run Google Colab Notebook (Free T4)
      │   └── visioncut_kaggle_worker.ipynb  # Ready-to-run Kaggle Notebook Fallback
      ├── engine/
      │   ├── visual_understanding.py  # Event classification, normalization & confidence rating
      │   ├── smart_editor.py          # Conservative cut decision & zoom planning
      │   ├── motion_graphics.py       # High-visibility vector SVG generator (Boxes, Arrows, Callouts)
      │   └── sfx_engine.py            # Procedural audio synthesizer (Click, Pop, Whoosh, Chime)
      ├── kdenlive/
      │   ├── project_generator.py     # Multi-track MLT XML Kdenlive project builder
      │   └── validator.py             # Strict production QA validator & HTML report generator
      ├── ui/
      │   ├── main_window.py           # PySide6 multi-screen desktop application
      │   ├── styles.py                # Dark theme stylesheet
      │   └── widgets/
      │       └── event_table.py       # Interactive decision review table with approve/reject toggles
      └── tests/
          ├── synthetic_data.py        # Synthetic 1080p 30fps tutorial video generator
          ├── test_preprocessing.py    # Phase 1 unit test
          ├── test_queue.py            # Phase 2 unit test
          ├── test_kdenlive.py         # Phases 6-9 unit test
          └── test_pipeline.py         # Complete Section 16 1-minute end-to-end verification
```

---

## 3. Daily Automatic Workflow

1. **Launch VisionCut AI**:
   ```bash
   python main.py
   ```
2. **Import Recording**: Select your 1080p 30 FPS tutorial recording.
3. **Analyze & Queue**: VisionCut performs lightweight streaming preprocessing, creates a contact sheet, and pushes a lightweight job to your Google Drive queue (`VisionCutCloud/jobs/`).
4. **Run Free Cloud Worker**:
   - Open `visioncut/cloud/visioncut_colab_worker.ipynb` in Google Colab.
   - Run **Runtime -> Run all** using free Colab GPU.
   - The worker analyzes the interface, predicts bounding boxes, and writes results back to `VisionCutCloud/results/`.
5. **Review Decisions in Desktop App**:
   - The desktop app automatically detects the completed job.
   - Inspect detected events, bounding boxes, and confidence levels.
   - Toggle individual checkboxes to approve or reject suggestions.
6. **Export Kdenlive Project**:
   - Click **Generate Kdenlive Project**.
   - The built-in QA engine automatically validates all media paths, tracks, and timestamps.
   - Click **Open in Kdenlive** to make final fine-tunings and export.

---

## 4. Testing & Verification

VisionCut includes comprehensive automated testing suites:

- **Run Full Section 16 1-Minute Demonstration**:
  ```bash
  python main.py --demo
  ```
  *Verifies: 1 detected action, 1 animated rectangle, 1 animated arrow, 1 circle, 1 zoom, 1 text callout, 1 click sound, 1 whoosh sound, 1 valid 7-track Kdenlive project with synchronized audio.*

- **Run All Unit & Integration Tests**:
  ```bash
  python main.py --test
  ```

