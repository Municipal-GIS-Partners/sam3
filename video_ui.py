#!/usr/bin/env python
"""
SAM3 Gradio UI – Video Predictor with Pre-Shrinking
---------------------------------------------------

Key changes vs. your previous version:

- Before SAM3 sees the video, we run it through ffmpeg:
    - Downscale to 720p/480p (or keep original but force even dimensions).
    - Reduce FPS (e.g., 8–10) to cut the frame count.
    - Optionally trim to first N seconds.

- SAM3 then runs on the preprocessed video file, which massively reduces RAM usage
  and avoids "Unable to allocate 102 GiB" errors.

You still get:
- start_session → add_prompt → propagate_in_video
- Streaming overlay and stride control
"""

import os
import time
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr

from sam3.model_builder import (
    build_sam3_video_predictor,
    build_sam3_image_model,  # unused here, but left for future image mode
)
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401 (unused, but kept)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sam3_video = None
_sam3_image = None  # reserved for later


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def load_video_predictor():
    global _sam3_video
    if _sam3_video is None:
        print("[SAM3] Loading video predictor...")
        _sam3_video = build_sam3_video_predictor()
        print("[SAM3] Video predictor ready.")
    return _sam3_video


def downscale_frame(frame, mode: str):
    """
    Downscale a single frame for visualization/output.
    This is separate from the ffmpeg preprocessing step.
    """
    if mode == "Original":
        return frame

    h, w = frame.shape[:2]
    if mode == "720p":
        target_h = 720
    else:  # "480p"
        target_h = 480

    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def color_for_id(obj_id, cmap):
    if obj_id not in cmap:
        np.random.seed(obj_id + 123)
        cmap[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())
    return cmap[obj_id]


def ensure_ffmpeg_available():
    """
    Check that ffmpeg is available on PATH.
    Raise a friendly error if not.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install ffmpeg and make sure the 'ffmpeg' command is available."
        )


def preprocess_video_with_ffmpeg(
    video_path: str,
    downscale_mode: str,
    target_fps: int,
    max_duration_sec: int | None,
    out_dir: Path,
) -> str:
    """
    Use ffmpeg to create a pre-shrunk version of the video so SAM3
    doesn't try to allocate hundreds of GB of RAM.

    - downscale_mode: "Original", "720p", "480p" (affects resolution)
    - target_fps: new FPS for processing (e.g., 8–10)
    - max_duration_sec: trim to first N seconds (None = no trim)
    """
    ensure_ffmpeg_available()

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"sam3_preprocessed_{ts}.mp4"

    # Build the ffmpeg filter graph
    vf_filters = []

    if downscale_mode == "720p":
        # scale to 720p height, keep aspect ratio, enforce even dimensions
        vf_filters.append("scale=-2:720")
    elif downscale_mode == "480p":
        vf_filters.append("scale=-2:480")
    else:
        # keep “original” size but force even dimensions for codec safety
        vf_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

    vf = ",".join(vf_filters)

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", video_path,
        "-vf", vf,
        "-r", str(target_fps),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
    ]

    if max_duration_sec is not None and max_duration_sec > 0:
        cmd.extend(["-t", str(max_duration_sec)])

    cmd.append(str(out_path))

    print("[FFMPEG] Preprocessing video:")
    print(" ", " ".join(cmd))

    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        print("[FFMPEG] stderr:\n", completed.stderr)
        raise RuntimeError(f"ffmpeg failed with code {completed.returncode}")

    print(f"[FFMPEG] Preprocessed video saved to: {out_path}")
    return str(out_path)


# --------------------------------------------------------------------
# MAIN VIDEO TRACKING LOGIC (with pre-shrinking)
# --------------------------------------------------------------------
def run_video_predictor(
    video_file,
    prompt,
    downscale_mode,
    stride,
    process_fps,
    max_duration_sec,
    progress=gr.Progress(),
):
    if video_file is None:
        return None, "No video provided."
    if not prompt:
        return None, "Please enter a prompt (e.g., 'a white truck')."

    original_video_path = getattr(video_file, "name")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------
    # 1. Pre-shrink video via ffmpeg
    # ------------------------------------------------
    try:
        preprocessed_path = preprocess_video_with_ffmpeg(
            video_path=original_video_path,
            downscale_mode=downscale_mode,
            target_fps=int(process_fps),
            max_duration_sec=int(max_duration_sec) if max_duration_sec > 0 else None,
            out_dir=outputs_dir,
        )
    except Exception as e:
        return None, f"Error during ffmpeg preprocessing: {e}"

    predictor = load_video_predictor()

    # ------------------------------------------------
    # 2. Start SAM3 video session on the PREPROCESSED video
    # ------------------------------------------------
    try:
        response = predictor.handle_request(
            dict(
                type="start_session",
                resource_path=preprocessed_path,
            )
        )
    except Exception as e:
        return None, f"Error starting SAM3 session: {e}"

    session_id = response["session_id"]

    # Attach text prompt to frame 0
    predictor.handle_request(
        dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )

    # ------------------------------------------------
    # 3. Prepare output video writer
    # ------------------------------------------------
    cap = cv2.VideoCapture(preprocessed_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or process_fps or 10

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        predictor.handle_request(dict(type="close_session", session_id=session_id))
        return None, "Could not read first frame from preprocessed video."

    first_frame = downscale_frame(first_frame, downscale_mode)
    out_h, out_w = first_frame.shape[:2]

    out_path = outputs_dir / f"sam3_out_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (out_w, out_h),
    )

    cap.release()

    # ------------------------------------------------
    # 4. Streaming propagation loop
    # ------------------------------------------------
    color_map = {}
    frame_counter = 0

    progress(0, desc="Tracking...")

    try:
        for resp in predictor.handle_stream_request(
            dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            idx = resp["frame_index"]
            outputs = resp["outputs"]

            # Stride: only write every Nth frame
            if idx % stride != 0:
                continue

            # Load that frame from the preprocessed video
            cap = cv2.VideoCapture(preprocessed_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()

            if not ok:
                print(f"[WARN] Could not load frame {idx}")
                continue

            frame = downscale_frame(frame, downscale_mode)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            masks = outputs.get("masks")
            ids = outputs.get("ids")

            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()
            if torch.is_tensor(ids):
                ids = ids.cpu().numpy()

            if masks is not None and ids is not None:
                for i in range(len(masks)):
                    obj_id = int(ids[i])
                    color = np.array(color_for_id(obj_id, color_map))
                    m = masks[i] > 0.5
                    rgb[m] = 0.6 * rgb[m] + 0.4 * color

            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            frame_counter += 1
            if total_frames > 0:
                progress(
                    min(frame_counter / total_frames, 1.0),
                    desc=f"Frame {idx}/{total_frames} (processed)",
                )

    finally:
        writer.release()
        # Close SAM3 session to free GPU/CPU memory
        try:
            predictor.handle_request(dict(type="close_session", session_id=session_id))
        except Exception:
            pass

    return str(out_path), (
        f"Finished tracking {frame_counter} frames "
        f"from preprocessed video (fps={fps_in:.2f}, total_frames={total_frames})."
    )


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def create_demo():
    with gr.Blocks(title="SAM3 Video Tracker (Pre-Shrunk)") as demo:
        gr.Markdown(
            """
            # **SAM3 Full Video Tracking – Optimized**

            This version **preprocesses the video with ffmpeg** before passing it to SAM3:

            - Downscale to 720p / 480p (or keep original size but make it codec-safe)
            - Reduce FPS (e.g., 8–10)
            - Optionally trim to first N seconds

            This avoids the huge in-memory `(T, H, W, C)` float32 tensor
            that was causing 100+ GiB allocation errors.
            """
        )

        with gr.Tab("Video Tracking"):
            with gr.Row():
                with gr.Column():
                    video = gr.File(label="Upload Video")
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="e.g., a white truck",
                    )
                    downscale = gr.Radio(
                        ["Original", "720p", "480p"],
                        value="720p",
                        label="Processing & Display Resolution",
                    )
                    stride = gr.Slider(
                        1,
                        10,
                        value=1,
                        step=1,
                        label="Frame Stride (process every Nth frame)",
                    )
                    process_fps = gr.Slider(
                        2,
                        30,
                        value=8,
                        step=1,
                        label="Processing FPS (after ffmpeg downsampling)",
                    )
                    max_duration = gr.Slider(
                        0,
                        600,
                        value=120,
                        step=10,
                        label="Max Duration (seconds, 0 = no limit)",
                    )
                    run = gr.Button("Run Tracker")

                with gr.Column():
                    out_video = gr.Video(label="Tracked Output")
                    out_log = gr.Markdown()

            run.click(
                run_video_predictor,
                inputs=[video, prompt, downscale, stride, process_fps, max_duration],
                outputs=[out_video, out_log],
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
