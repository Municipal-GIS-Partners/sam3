#!/usr/bin/env python
"""
SAM3 Gradio UI – Correct Video Predictor Implementation
-------------------------------------------------------

This version uses the **real SAM3 API**:

- start_session
- add_prompt
- propagate_in_video (streaming)

It is memory-safe and works with long videos.

"""

import os
import cv2
import time
import numpy as np
import torch
import gradio as gr
from pathlib import Path

from sam3.model_builder import (
    build_sam3_video_predictor,
    build_sam3_image_model,
)
from sam3.model.sam3_image_processor import Sam3Processor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sam3_video = None
_sam3_image = None


def load_video_predictor():
    global _sam3_video
    if _sam3_video is None:
        print("[SAM3] Loading video predictor...")
        _sam3_video = build_sam3_video_predictor()
        print("[SAM3] Video predictor ready.")
    return _sam3_video


def downscale_frame(frame, mode):
    if mode == "Original":
        return frame

    h, w = frame.shape[:2]
    if mode == "720p":
        target_h = 720
    else:  # 480p
        target_h = 480

    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def color_for_id(obj_id, cmap):
    if obj_id not in cmap:
        np.random.seed(obj_id + 123)
        cmap[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())
    return cmap[obj_id]


# --------------------------------------------------------------------
# MAIN VIDEO TRACKING LOGIC (Correct API)
# --------------------------------------------------------------------
def run_video_predictor(video_file, prompt, downscale_mode, stride, progress=gr.Progress()):

    if video_file is None or not prompt:
        return None, "No video or no prompt."

    video_path = getattr(video_file, "name")
    predictor = load_video_predictor()

    # ---------------------------------------
    # Start SAM3 video session
    # ---------------------------------------
    response = predictor.handle_request(
        dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    # ---------------------------------------
    # Attach text prompt to frame 0
    # ---------------------------------------
    predictor.handle_request(
        dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        )
    )

    # ---------------------------------------
    # Prepare output video writer
    # ---------------------------------------
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    ok, first_frame = cap.read()
    if not ok:
        return None, "Could not read first frame."

    first_frame = downscale_frame(first_frame, downscale_mode)
    out_h, out_w = first_frame.shape[:2]

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/sam3_out_{int(time.time())}.mp4"
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (out_w, out_h)
    )

    cap.release()

    # ---------------------------------------
    # Streaming propagation
    # ---------------------------------------
    color_map = {}
    frame_counter = 0

    progress(0, desc="Tracking...")

    # *************** THE CORRECT API ***************
    for resp in predictor.handle_stream_request(
        dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        idx = resp["frame_index"]
        outputs = resp["outputs"]

        # Skip frames via stride
        if idx % stride != 0:
            continue

        # Load that frame manually from file
        cap = cv2.VideoCapture(video_path)
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

        if masks is not None:
            for i in range(len(masks)):
                obj_id = int(ids[i])
                color = np.array(color_for_id(obj_id, color_map))
                m = masks[i] > 0.5
                rgb[m] = 0.6 * rgb[m] + 0.4 * color

        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        frame_counter += 1
        progress(frame_counter / total_frames, desc=f"Frame {idx}/{total_frames}")

    writer.release()

    # Close session to free GPU
    predictor.handle_request(dict(type="close_session", session_id=session_id))

    return out_path, f"Finished tracking {frame_counter} frames."


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def create_demo():
    with gr.Blocks(title="SAM3 Video Tracker (Correct API)") as demo:
        gr.Markdown(
            """
            # **SAM3 Full Video Tracking**  
            Uses the *real* API: start_session → add_prompt → propagate_in_video.
            """
        )

        with gr.Tab("Video Tracking"):
            with gr.Row():
                with gr.Column():
                    video = gr.File(label="Upload Video")
                    prompt = gr.Textbox(label="Prompt", placeholder="e.g., a white truck")
                    downscale = gr.Radio(
                        ["Original", "720p", "480p"],
                        value="720p",
                        label="Downscale Mode"
                    )
                    stride = gr.Slider(
                        1, 10, value=1, step=1,
                        label="Frame Stride"
                    )
                    run = gr.Button("Run Tracker")

                with gr.Column():
                    out_video = gr.Video(label="Tracked Output")
                    out_log = gr.Markdown()

            run.click(
                run_video_predictor,
                inputs=[video, prompt, downscale, stride],
                outputs=[out_video, out_log]
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
