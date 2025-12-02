#!/usr/bin/env python
"""
SAM3 GeoTIFF Pipeline UI
------------------------

Small Gradio interface to run the GeoTIFF building pipeline:
  - Reads tiles from an input folder (defaults to ArcGIS export path)
  - Writes mask GeoTIFFs and optional vector polygons
  - Uses SAM3 open-vocabulary segmentation with a text concept (default: building)

Run:
    python -m geo_building_ui
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import rasterio
import torch
from PIL import Image

from geo_building_pipeline import (
    DEFAULT_INPUT_DIR,
    _prepare_image_array,
    _render_overlay,
    _normalize_image_for_overlay,
    load_sam3_model,
    run_geo_building_pipeline,
    segment_buildings_for_tile,
)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


_MODEL_CACHE = {}


def _get_cached_model(device: str):
    """Cache SAM3 model per device to avoid reloads between preview runs."""
    if device not in _MODEL_CACHE:
        _MODEL_CACHE[device] = load_sam3_model(device=device)
    return _MODEL_CACHE[device]


def _maybe_preload_default_model():
    """
    Optionally preload the default device model at UI launch so first requests are faster.
    Disable by setting env SAM3_UI_PRELOAD=0.
    """
    if os.getenv("SAM3_UI_PRELOAD", "1") == "0":
        return
    device = _default_device()
    try:
        logging.info("[SAM3 UI] Preloading model on %s", device)
        _get_cached_model(device=device)
        logging.info("[SAM3 UI] Model ready on %s", device)
    except Exception as exc:  # pragma: no cover - best-effort preload
        logging.exception("Model preload failed: %s", exc)


def _run_pipeline_with_logging(
    input_dir: str,
    mask_out_dir: str,
    vector_out_dir: Optional[str],
    write_vectors: bool,
    overlay_out_dir: Optional[str],
    write_overlays: bool,
    concept: str,
    min_area: float,
    confidence_threshold: float,
    device: str,
    max_tiles: Optional[int],
    overwrite: bool,
) -> str:
    """
    Run the pipeline while capturing logs to return in the UI.
    """
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    try:
        run_geo_building_pipeline(
            input_dir=Path(input_dir),
            mask_out_dir=Path(mask_out_dir),
            vector_out_dir=Path(vector_out_dir) if write_vectors and vector_out_dir else None,
            overlay_out_dir=Path(overlay_out_dir) if write_overlays and overlay_out_dir else None,
            concept=concept,
            min_area=min_area,
            confidence_threshold=confidence_threshold,
            device=device,
            max_tiles=max_tiles,
            overwrite=overwrite,
        )
    except Exception as exc:  # pragma: no cover - runtime reporting
        logging.exception("Pipeline failed: %s", exc)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(prev_level)

    return log_buffer.getvalue()


def _colorize_mask(mask: np.ndarray) -> Image.Image:
    """
    Convert a uint mask to a pseudo-colored PIL image for quick viewing.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")
    labels = np.unique(mask)
    base = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    for lbl in labels:
        if lbl == 0:
            continue
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        base[mask == lbl] = color
    return Image.fromarray(base)


def _load_outputs(
    overlay_dir: Optional[str],
    mask_dir: Optional[str],
    sample_limit: int = 12,
) -> tuple[list[tuple[Image.Image, str]], list[tuple[Image.Image, str]], str]:
    """
    Load recently written overlay PNGs and mask GeoTIFFs for viewing in the UI.
    Returns (overlays, masks, log).
    """
    overlays: list[tuple[Image.Image, str]] = []
    masks: list[tuple[Image.Image, str]] = []
    logs: list[str] = []

    if overlay_dir and os.path.isdir(overlay_dir):
        overlay_paths = sorted(Path(overlay_dir).glob("*.png"))
        if overlay_paths:
            for pth in overlay_paths[:sample_limit]:
                try:
                    overlays.append((Image.open(pth), pth.name))
                except Exception as exc:  # pragma: no cover - runtime guardrail
                    logs.append(f"Failed reading overlay {pth.name}: {exc}")
        else:
            logs.append(f"No overlays found in {overlay_dir}")
    elif overlay_dir:
        logs.append(f"Overlay directory not found: {overlay_dir}")

    if mask_dir and os.path.isdir(mask_dir):
        mask_paths = sorted(list(Path(mask_dir).glob("*.tif")) + list(Path(mask_dir).glob("*.tiff")))
        if mask_paths:
            for pth in mask_paths[:sample_limit]:
                try:
                    with rasterio.open(pth) as src:
                        arr = src.read(1)
                        colored = _colorize_mask(arr)
                        masks.append((colored, pth.name))
                except Exception as exc:  # pragma: no cover - runtime guardrail
                    logs.append(f"Failed reading mask {pth.name}: {exc}")
        else:
            logs.append(f"No masks found in {mask_dir}")
    elif mask_dir:
        logs.append(f"Mask directory not found: {mask_dir}")

    log_str = "\n".join(logs) if logs else "Loaded outputs."
    return overlays, masks, log_str


def build_interface():
    with gr.Blocks(title="SAM3 GeoTIFF Pipeline") as demo:
        gr.Markdown(
            "## SAM3 GeoTIFF Building Pipeline\n"
            "Run open-vocabulary segmentation over GeoTIFF tiles and export masks/vectors "
            "ready for ArcGIS or any GIS tool."
        )

        with gr.Row():
            input_dir = gr.Textbox(
                label="Input directory",
                value=str(DEFAULT_INPUT_DIR),
                placeholder="Folder of GeoTIFF tiles exported from ArcGIS",
            )
            mask_out_dir = gr.Textbox(
                label="Mask output directory",
                placeholder="Folder to write mask GeoTIFFs",
            )

        with gr.Row():
            vector_out_dir = gr.Textbox(
                label="Vector output directory (optional)",
                placeholder="Folder to write polygons (GeoPackage/GeoJSON). Leave empty to skip.",
            )
            write_vectors = gr.Checkbox(
                label="Write vectors",
                value=False,
                info="Toggle on to export polygons (same as CLI flag).",
            )
            overlay_out_dir = gr.Textbox(
                label="Overlay output directory (optional)",
                placeholder="Folder to write preview overlays (PNG). Leave empty to skip.",
            )
            write_overlays = gr.Checkbox(
                label="Write overlays",
                value=True,
                info="Toggle on to save PNG overlays.",
            )
            concept = gr.Textbox(
                label="Concept prompt",
                value="building",
                placeholder='e.g., "building"',
            )
            conf_threshold = gr.Slider(
                label="Confidence threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.25,
            )

        with gr.Row():
            min_area = gr.Number(
                label="Min polygon area (CRS units)",
                value=20.0,
                precision=2,
            )
            max_tiles = gr.Number(
                label="Max tiles (optional)",
                value=10,
                precision=0,
            )

        with gr.Row():
            device = gr.Radio(
                choices=["cuda", "cpu"],
                value=_default_device(),
                label="Device",
            )
            overwrite = gr.Checkbox(label="Overwrite existing outputs", value=False)

        run_btn = gr.Button("Run Pipeline", variant="primary")
        logs = gr.Textbox(
            label="Pipeline log",
            lines=15,
            placeholder="Run the pipeline to see progress...",
        )

        def _on_run(
            input_dir,
            mask_out_dir,
            vector_out_dir,
            write_vectors,
            overlay_out_dir,
            write_overlays,
            concept,
            conf_threshold,
            min_area,
            device,
            max_tiles,
            overwrite,
            progress=gr.Progress(track_tqdm=True),
        ):
            max_tiles_int: Optional[int]
            try:
                max_tiles_int = int(max_tiles) if max_tiles not in (None, "", "None") else None
            except (TypeError, ValueError):
                max_tiles_int = None

            try:
                min_area_val = float(min_area)
            except (TypeError, ValueError):
                min_area_val = 20.0

            overlay_dir_eff = overlay_out_dir or (
                os.path.join(mask_out_dir, "overlays") if write_overlays else None
            )
            vector_dir_eff = vector_out_dir if write_vectors else None

            return _run_pipeline_with_logging(
                input_dir=input_dir,
                mask_out_dir=mask_out_dir,
                vector_out_dir=vector_dir_eff or None,
                write_vectors=bool(write_vectors),
                overlay_out_dir=overlay_dir_eff or None,
                write_overlays=bool(write_overlays),
                concept=concept,
                min_area=min_area_val,
                confidence_threshold=float(conf_threshold),
                device=device,
                max_tiles=max_tiles_int,
                overwrite=bool(overwrite),
            )

        run_btn.click(
            _on_run,
            inputs=[
                input_dir,
                mask_out_dir,
                vector_out_dir,
                write_vectors,
                overlay_out_dir,
                write_overlays,
                concept,
                conf_threshold,
                min_area,
                device,
                max_tiles,
                overwrite,
            ],
            outputs=logs,
        )

        gr.Markdown("### Quick preview on a single GeoTIFF (no files written)")
        with gr.Row():
            preview_tile = gr.File(
                label="GeoTIFF tile",
                file_types=[".tif", ".tiff"],
            )
            preview_btn = gr.Button("Preview single tile")

        preview_image = gr.Image(label="Overlay preview", type="pil")
        preview_log = gr.Textbox(
            label="Preview log",
            placeholder="Run preview to see details...",
            lines=4,
        )

        def _preview_single_tile(
            tile_file,
            concept,
            conf_threshold,
            device,
            progress=gr.Progress(track_tqdm=False),
        ):
            if tile_file is None:
                return None, "Please select a GeoTIFF tile."
            tile_path = getattr(tile_file, "name", None)
            if tile_path is None or not os.path.isfile(tile_path):
                return None, "Invalid file."

            try:
                with rasterio.open(tile_path) as src:
                    img_arr = _prepare_image_array(src)
                model = _get_cached_model(device=device)
                mask = segment_buildings_for_tile(
                    model=model,
                    image_array=img_arr,
                    concept=concept,
                    confidence_threshold=float(conf_threshold),
                )
                overlay = _render_overlay(img_arr, mask)
                return overlay, f"Previewed {os.path.basename(tile_path)} on {device}. Labels: {int(mask.max())}"
            except Exception as exc:  # pragma: no cover - runtime guardrail
                logging.exception("Preview failed: %s", exc)
                return None, f"Preview failed: {exc}"

        preview_btn.click(
            _preview_single_tile,
            inputs=[preview_tile, concept, conf_threshold, device],
            outputs=[preview_image, preview_log],
        )

        gr.Markdown("### View recent outputs (overlays & masks)")
        with gr.Row():
            refresh_btn = gr.Button("Load outputs from above folders")
            sample_limit = gr.Slider(
                label="Max items to display",
                minimum=1,
                maximum=30,
                step=1,
                value=12,
            )
        overlays_gallery = gr.Gallery(
            label="Overlays (PNG)",
            show_label=True,
            columns=4,
            height="auto",
        )
        masks_gallery = gr.Gallery(
            label="Masks (colorized labels)",
            show_label=True,
            columns=4,
            height="auto",
        )
        output_log = gr.Textbox(
            label="Output loader log",
            lines=3,
            placeholder="Press refresh to load overlays/masks.",
        )

        def _load_outputs_cb(
            overlay_out_dir,
            mask_out_dir,
            sample_limit,
            progress=gr.Progress(track_tqdm=False),
        ):
            overlay_dir_eff = overlay_out_dir or (
                os.path.join(mask_out_dir, "overlays") if mask_out_dir else None
            )
            overlays, masks, log_msg = _load_outputs(
                overlay_dir=overlay_dir_eff or None,
                mask_dir=mask_out_dir or None,
                sample_limit=int(sample_limit),
            )
            return overlays, masks, log_msg

        refresh_btn.click(
            _load_outputs_cb,
            inputs=[overlay_out_dir, mask_out_dir, sample_limit],
            outputs=[overlays_gallery, masks_gallery, output_log],
        )

    return demo


if __name__ == "__main__":
    _maybe_preload_default_model()
    ui = build_interface()
    ui.launch()
