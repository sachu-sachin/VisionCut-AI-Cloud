"""
Synthetic test video and audio generator for VisionCut AI testing.
Creates realistic 1080p 30fps screen tutorial footage with UI elements,
buttons, terminal text, audio narration tone, and silence.
"""

import math
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf

from visioncut.core.ffmpeg_utils import get_ffmpeg_path, FFmpegRunner


def generate_synthetic_tutorial_video(
    output_path: str,
    duration_sec: float = 12.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30
) -> str:
    """
    Creates an MP4 video with:
    - 0s - 3s: Code Editor window showing Python code.
    - 3s - 5s: Accidental static wait period / idle pause with silence.
    - 5s - 8s: Terminal window with execution command and prompt.
    - 8s - 10s: Button click on an 'Execute Pipeline' button with a highlight box.
    - 10s - 12s: Success alert banner displaying final results.
    Includes a synchronized audio track with tone bursts and silence.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    temp_video = str(out_file.parent / "temp_video_nosound.mp4")
    temp_audio = str(out_file.parent / "temp_audio.wav")

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Dark theme IDE background
        img[:] = (30, 30, 34)

        # Title bar
        cv2.rectangle(img, (0, 0), (width, 45), (45, 45, 52), -1)
        cv2.putText(img, "VisionCut Studio - [Tutorial_Sample.py]", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 225), 2, cv2.LINE_AA)

        # Window controls (close, min, max dots)
        cv2.circle(img, (width - 30, 22), 8, (60, 60, 230), -1)
        cv2.circle(img, (width - 60, 22), 8, (60, 200, 230), -1)
        cv2.circle(img, (width - 90, 22), 8, (60, 220, 60), -1)

        # Sidebar
        cv2.rectangle(img, (0, 45), (260, height), (38, 38, 44), -1)
        cv2.putText(img, "EXPLORER", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "> src/", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1, cv2.LINE_AA)
        cv2.putText(img, "  main.py", (45, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "  models.py", (45, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1, cv2.LINE_AA)

        if t < 5.0:
            # Code editor area
            cv2.putText(img, "def build_model():", (320, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 180, 80), 2, cv2.LINE_AA)
            cv2.putText(img, "    model = VisionNetwork()", (360, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "    model.compile(optimizer='adam')", (360, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 255, 180), 2, cv2.LINE_AA)
            cv2.putText(img, "    return model", (360, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 180, 80), 2, cv2.LINE_AA)

            # Cursor blinking
            if int(t * 2) % 2 == 0:
                cv2.rectangle(img, (580, 245), (584, 275), (255, 255, 255), -1)

        elif 5.0 <= t < 8.0:
            # Terminal execution
            cv2.rectangle(img, (280, 500), (width - 40, height - 40), (20, 20, 24), -1)
            cv2.putText(img, "TERMINAL (bash)", (300, 535),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(img, "$ python main.py --train --epochs 5", (300, 580),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(img, "[INFO] Loading dataset... 100% complete", (300, 620),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, "[INFO] Epoch 5/5 loss: 0.042 - accuracy: 0.985", (300, 660),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        elif 8.0 <= t < 10.0:
            # Action button demonstration
            btn_x, btn_y, btn_w, btn_h = 800, 400, 320, 70
            cv2.rectangle(img, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 122, 204), -1)
            cv2.rectangle(img, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 2)
            cv2.putText(img, "EXECUTE PIPELINE", (btn_x + 30, btn_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Simulated mouse cursor moving to button
            mx = int(700 + (t - 8.0) * 120)
            my = int(350 + (t - 8.0) * 45)
            # Draw cursor arrow
            pts = np.array([[mx, my], [mx + 18, my + 14], [mx + 10, my + 16], [mx + 16, my + 26],
                            [mx + 11, my + 28], [mx + 6, my + 18], [mx, my + 22]], np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255))
            cv2.polylines(img, [pts], True, (0, 0, 0), 2)

        else:
            # 10s - 12s: Success Result Banner
            box_x, box_y, box_w, box_h = 600, 300, 720, 160
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (35, 80, 50), -1)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (60, 220, 100), 3)
            cv2.putText(img, "TRAINING COMPLETED SUCCESSFULLY", (box_x + 40, box_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "Final Model Checkpoint Saved: weights_v1.pt", (box_x + 40, box_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 240, 180), 2, cv2.LINE_AA)

        out.write(img)

    out.release()

    # Generate synchronized audio WAV (44.1kHz mono)
    sample_rate = 44100
    total_audio_samples = int(duration_sec * sample_rate)
    audio_data = np.zeros(total_audio_samples, dtype=np.float32)

    t_axis = np.arange(total_audio_samples) / sample_rate

    # Narration voice simulation: 220Hz + harmonics
    # Voice active: 0s to 3s
    idx_0 = 0
    idx_3 = int(3.0 * sample_rate)
    voice_1 = 0.35 * np.sin(2 * math.pi * 220 * t_axis[idx_0:idx_3]) + \
              0.15 * np.sin(2 * math.pi * 440 * t_axis[idx_0:idx_3])
    audio_data[idx_0:idx_3] = voice_1

    # 3s to 5s is SILENCE (simulates accidental gap / thinking pause)
    # audio_data is 0.0 here

    # 5s to 8s is active terminal narration
    idx_5 = int(5.0 * sample_rate)
    idx_8 = int(8.0 * sample_rate)
    voice_2 = 0.40 * np.sin(2 * math.pi * 240 * t_axis[idx_5:idx_8]) + \
              0.18 * np.sin(2 * math.pi * 480 * t_axis[idx_5:idx_8])
    audio_data[idx_5:idx_8] = voice_2

    # 8.5s is a click sound spike
    idx_click = int(8.5 * sample_rate)
    click_len = int(0.05 * sample_rate)
    t_click = np.arange(click_len) / sample_rate
    click_wave = 0.8 * np.sin(2 * math.pi * 1200 * t_click) * np.exp(-t_click * 60)
    audio_data[idx_click:idx_click + click_len] += click_wave.astype(np.float32)

    # 10s to 12s is active result narration
    idx_10 = int(10.0 * sample_rate)
    idx_12 = int(12.0 * sample_rate)
    voice_3 = 0.38 * np.sin(2 * math.pi * 260 * t_axis[idx_10:idx_12])
    audio_data[idx_10:idx_12] = voice_3

    sf.write(temp_audio, audio_data, sample_rate)

    # Mux video and audio into final MP4 with FFmpeg
    runner = FFmpegRunner()
    runner.run([
        "-y",
        "-i", temp_video,
        "-i", temp_audio,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_file)
    ], check=True)

    # Clean up temp intermediates
    try:
        Path(temp_video).unlink(missing_ok=True)
        Path(temp_audio).unlink(missing_ok=True)
    except Exception:
        pass

    return str(out_file)

