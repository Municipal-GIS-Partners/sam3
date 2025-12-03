#!/usr/bin/env python
"""
SAM3 Image Gradio UI – Official Image API Only
----------------------------------------------

Uses the *documented* image API:

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt="...")

Features:
- Upload image
- Text prompt describing the concept
- Score threshold
- Optional limit on number of instances
- Colored instance overlays
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import gradio as gr

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# Global model + processor
# ============================================================================

_sam3_model = None
_sam3_processor = None


def load_sam3_image_model():
    """
    Lazily build the SAM3 image model + processor once, then reuse.
    This follows the official README pattern.
    """
    global _sam3_model, _sam3_processor
    if _sam3_model is None or _sam3_processor is None:
        print("[SAM3] Building image model...")
        _sam3_model = build_sam3_image_model()
        _sam3_processor = Sam3Processor(_sam3_model)
        print("[SAM3] Image model ready.")
    return _sam3_model, _sam3_processor


# ============================================================================
# Mask overlay helpers
# ============================================================================

def _random_color_for_idx(idx: int) -> np.ndarray:
    """
    Deterministic pseudo-random color per instance index.
    Just to visually separate multiple instances.
    """
    rng = np.random.RandomState(idx + 12345)
    return rng.randint(0, 255, size=3, dtype=np.uint8)


def overlay_instance_masks(
    image: Image.Image,
    masks: np.ndarray,
    scores: np.ndarray,
    score_thresh: float = 0.0,
    max_instances: int | None = None,
) -> Image.Image:
    """
    Overlay multiple instance masks on top of an RGB image.

    Args:
        image: PIL RGB image.
        masks: numpy array [N, H, W] of binary/float masks.
        scores: numpy array [N] of confidences.
        score_thresh: discard instances below this.
        max_instances: if set, keep only the top-K by score.

    Returns:
        PIL Image with colored mask overlays.
    """
    if masks is None or len(masks) == 0:
        return image

    # Filter by score
    keep = scores >= score_thresh
    if keep.sum() == 0:
        return image

    masks = masks[keep]
    scores = scores[keep]

    # Sort by score (highest first)
    order = np.argsort(-scores)
    masks = masks[order]
    scores = scores[order]

    if max_instances is not None and max_instances > 0:
        masks = masks[:max_instances]
        scores = scores[:max_instances]

    img_np = np.array(image).astype(np.uint8)
    overlay = img_np.copy()

    for i, m in enumerate(masks):
        # Binary mask; SAM3 may give float masks
        m_bin = m > 0.5
        if not m_bin.any():
            continue

        color = _random_color_for_idx(i)  # (3,)
        alpha = 0.6

        # Blend color where mask is true
        overlay[m_bin] = (
            (1.0 - alpha) * overlay[m_bin].astype(np.float32)
            + alpha * color.astype(np.float32)
        ).astype(np.uint8)

    return Image.fromarray(overlay)


# ============================================================================
# Core SAM3 image call
# ============================================================================

def run_sam3_on_image(
    image: Image.Image,
    prompt: str,
    score_thresh: float = 0.0,
    max_instances: int | None = None,
):
    """
    Run SAM3's image model on a single image with a text prompt.

    This function uses ONLY the documented API:
      - build_sam3_image_model()
      - Sam3Processor(model)
      - processor.set_image(...)
      - processor.set_text_prompt(...)
    """
    if image is None or prompt is None or prompt.strip() == "":
        return None, "Please provide an image and a non-empty text prompt."

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    _, processor = load_sam3_image_model()

    # Initialize state with the image
    state = processor.set_image(image)

    # Text prompt
    output = processor.set_text_prompt(
        state=state,
        prompt=prompt.strip(),
    )

    # Expected keys from SAM3: "masks", "boxes", "scores"
    masks = output.get("masks", None)
    scores = output.get("scores", None)

    if masks is None or scores is None:
        return image, "Model returned no masks or scores."

    # Convert tensors → numpy
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    # Overlay all instances
    overlaid = overlay_instance_masks(
        image,
        masks,
        scores,
        score_thresh=score_thresh,
        max_instances=max_instances,
    )

    # Build a short log string
    num_total = len(scores)
    num_kept = int((scores >= score_thresh).sum())
    log = (
        f"Total instances: **{num_total}**  \n"
        f"Above threshold ({score_thresh:.2f}): **{num_kept}**"
        + (f"  \nMax instances shown: **{max_instances}**" if max_instances else "")
    )

    return overlaid, log


# ============================================================================
# Gradio UI callbacks
# ============================================================================

def gr_segment_image(image, prompt, score_thresh, max_instances):
    """
    Wrapper for Gradio. Returns (image, markdown_log).
    """
    if max_instances is not None:
        try:
            max_instances_int = int(max_instances)
            if max_instances_int <= 0:
                max_instances_int = None
        except Exception:
            max_instances_int = None
    else:
        max_instances_int = None

    result_img, log = run_sam3_on_image(
        image=image,
        prompt=prompt,
        score_thresh=score_thresh,
        max_instances=max_instances_int,
    )
    return result_img, log


# ============================================================================
# Gradio App
# ============================================================================

def create_demo() -> gr.Blocks:
    with gr.Blocks(title="SAM3 Image Segmentation UI") as demo:
        gr.Markdown(
            """
            # SAM 3 – Image Concept Segmentation (Text Prompt Only)

            - Upload an image
            - Enter a text description of what you want segmented  
              (e.g. `"a person"`, `"white car"`, `"yellow excavator"`)
            - Adjust score threshold / max instances as needed
            """
        )

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(
                    label="Upload image",
                    type="pil",   # Gradio will pass a PIL.Image.Image
                )
                prompt_box = gr.Textbox(
                    label="Text prompt",
                    placeholder='e.g. "a person", "a dog", "a white truck"',
                )
                score_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    label="Score threshold (filter low-confidence instances)",
                )
                max_inst_box = gr.Number(
                    value=0,
                    precision=0,
                    label="Max instances to show (0 = no limit)",
                )
                run_button = gr.Button("Run SAM3")

            with gr.Column():
                img_output = gr.Image(
                    label="Segmented image (overlays)",
                )
                log_output = gr.Markdown(
                    label="Run summary",
                )

        run_button.click(
            fn=gr_segment_image,
            inputs=[img_input, prompt_box, score_slider, max_inst_box],
            outputs=[img_output, log_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    # You can set share=True if you want a public link
    demo.launch()
