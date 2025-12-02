import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - optional dependency
    gpd = None


DEFAULT_INPUT_DIR = Path(r"G:\sam3\sam3\assets\geotiff_tiles")
DEFAULT_CONFIDENCE_THRESHOLD = 0.25  # Lowered to catch more instances by default


def load_sam3_model(device: str):
    """
    Load the SAM3 image model using the canonical build function and move it to the
    requested device. We rely on the huggingface_hub integration to fetch weights if
    they are not already cached locally.
    """
    model = build_sam3_image_model(device=device)
    model.eval()
    model._geo_device = device  # Cache for downstream processor reuse
    return model


def _get_processor(model, confidence_threshold: float) -> Sam3Processor:
    processor = getattr(model, "_geo_processor", None)
    cached_thresh = getattr(model, "_geo_conf_threshold", None)
    if processor is None or cached_thresh != confidence_threshold:
        device = getattr(model, "_geo_device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = Sam3Processor(
            model, device=device, confidence_threshold=confidence_threshold
        )
        model._geo_processor = processor
        model._geo_conf_threshold = confidence_threshold
    return processor


def segment_buildings_for_tile(
    model,
    image_array: np.ndarray,
    concept: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """
    Use SAM3 to perform concept segmentation on image_array for the given concept.

    Returns:
        A 2D numpy array (H x W) with 0 as background and positive integers for each
        detected instance. Masks are sorted by score, and higher-scoring instances
        claim pixels first.
    """
    processor = _get_processor(model, confidence_threshold=confidence_threshold)
    # Follow the documented Image API: convert to PIL so height/width are read correctly.
    pil_image = Image.fromarray(image_array)
    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(prompt=concept, state=state)

    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or masks.numel() == 0:
        height, width = image_array.shape[:2]
        return np.zeros((height, width), dtype=np.uint16)

    # masks: [N, 1, H, W] -> [N, H, W]
    masks_np = masks.squeeze(1).cpu().numpy().astype(np.bool_)
    scores_np = scores.cpu().numpy()
    if masks_np.ndim == 2:  # Single mask edge case
        masks_np = np.expand_dims(masks_np, axis=0)
        scores_np = np.expand_dims(scores_np, axis=0)

    order = np.argsort(scores_np)[::-1]  # Highest score first
    labeled_mask = np.zeros(masks_np.shape[1:], dtype=np.uint16)
    for label, idx in enumerate(order, start=1):
        candidate_mask = masks_np[idx]
        labeled_mask[(candidate_mask) & (labeled_mask == 0)] = label

    return labeled_mask


def mask_to_polygons(
    mask: np.ndarray,
    transform,
    crs,
    min_area: float,
):
    """
    Convert a labeled mask into polygons with georeferencing.

    Args:
        mask: 2D array with 0 as background and positive integers as labels.
        transform: Affine transform from the source raster.
        crs: Coordinate reference system from the source raster.
        min_area: Minimum polygon area (in CRS units) to keep.
    """
    if gpd is None:
        raise ImportError("geopandas is required for vector export but is not installed")

    polygons = []
    labels = []
    areas = []
    for geom_mapping, value in shapes(mask.astype(np.int32), transform=transform):
        if value == 0:
            continue
        geometry = shapely_shape(geom_mapping)
        area = geometry.area
        if area < min_area:
            continue
        polygons.append(geometry)
        labels.append(int(value))
        areas.append(float(area))

    return gpd.GeoDataFrame(
        {"label": labels, "area": areas, "geometry": polygons},
        crs=crs,
    )


def _prepare_image_array(dataset: rasterio.io.DatasetReader) -> np.ndarray:
    """
    Read a rasterio dataset and convert it to an RGB-like numpy array (H, W, 3)
    suitable for SAM3.

    Rules:
      - If C > 3: use the first 3 bands (assumed to be RGB / RGB-like).
      - If C == 3: use as-is.
      - If C == 2: keep the first band and treat as single-channel.
      - If C == 1: single-channel.
      - Single-channel is then repeated to 3 channels so SAM3 always sees 3.
    """
    image = dataset.read()  # (C, H, W)
    c, h, w = image.shape

    # If more than 3 bands (e.g., RGBA, RGB+NIR, etc.), keep the first 3
    if c > 3:
        # Heuristic: assume bands 1–3 are the most useful (RGB / main composite)
        image = image[:3, :, :]
        c = 3

    # If exactly 2 bands, just use the first band (e.g., intensity)
    if c == 2:
        image = image[:1, :, :]
        c = 1

    # Move channels to last dimension -> (H, W, C')
    image = np.moveaxis(image, 0, -1)  # (H, W, C')

    # Handle grayscale (H, W) or (H, W, 1)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if image.shape[2] == 1:
        # Repeat single channel to fake RGB
        image = np.repeat(image, 3, axis=2)  # (H, W, 3)

    # Scale each channel to 0-255 uint8 so SAM3 normalization sees normal image ranges.
    img = image.astype(np.float32)
    mins = img.min(axis=(0, 1), keepdims=True)
    maxs = img.max(axis=(0, 1), keepdims=True)
    denom = np.clip(maxs - mins, 1e-6, None)
    img = (img - mins) / denom
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _normalize_image_for_overlay(image_array: np.ndarray) -> np.ndarray:
    """Normalize image for visualization; handles 8/16-bit inputs."""
    img = image_array.astype(np.float32)
    mins = img.min(axis=(0, 1), keepdims=True)
    maxs = img.max(axis=(0, 1), keepdims=True)
    denom = np.clip(maxs - mins, 1e-6, None)
    img = (img - mins) / denom
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _render_overlay(
    image_array: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    """Render a simple color overlay for a labeled mask on top of the source image."""
    base = _normalize_image_for_overlay(image_array)
    overlay = base.copy()

    unique_labels = [lbl for lbl in np.unique(mask) if lbl != 0]
    rng = np.random.default_rng(12345)
    for lbl in unique_labels:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        m = mask == lbl
        overlay[m] = (
            (1.0 - alpha) * overlay[m].astype(np.float32)
            + alpha * color.astype(np.float32)
        ).astype(np.uint8)

    return Image.fromarray(overlay)



def run_geo_building_pipeline(
    input_dir: Path,
    mask_out_dir: Path,
    vector_out_dir: Optional[Path],
    overlay_out_dir: Optional[Path],
    concept: str,
    min_area: float,
    confidence_threshold: float,
    device: str,
    max_tiles: Optional[int],
    overwrite: bool,
) -> None:
    """
    Run SAM3 open-vocabulary segmentation on a folder of GeoTIFF tiles and export
    georeferenced masks (and optional vectors).

    Assumptions:
        - Input GeoTIFFs are RGB or single-channel; single-channel inputs are
          replicated to three channels.
        - CRS and transform are preserved on outputs.
        - min_area is applied in CRS units (e.g., m^2 for projected data).
        - overlay_out_dir: if provided, PNG overlays of source + mask will be saved.
        - confidence_threshold: passed to Sam3Processor to control mask filtering.
    """
    logging.info("Loading SAM3 model on device %s", device)
    model = load_sam3_model(device=device)
    if vector_out_dir and gpd is None:
        raise ImportError(
            "geopandas is required for vector export; install it or omit --vector-out-dir"
        )
    mask_out_dir.mkdir(parents=True, exist_ok=True)
    if vector_out_dir:
        vector_out_dir.mkdir(parents=True, exist_ok=True)
    if overlay_out_dir:
        overlay_out_dir.mkdir(parents=True, exist_ok=True)

    concept_slug = concept.lower().replace(" ", "_")
    tile_paths = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )
    if max_tiles is not None:
        tile_paths = tile_paths[:max_tiles]

    logging.info("Found %d tiles in %s", len(tile_paths), input_dir)
    for idx, tile_path in enumerate(tile_paths, start=1):
        try:
            with rasterio.open(tile_path) as src:
                image_array = _prepare_image_array(src)
                profile = src.profile
                crs = src.crs
                transform = src.transform

            mask_filename = (
                f"{tile_path.stem}_{concept_slug}_mask.tif"
                if concept_slug
                else f"{tile_path.stem}_mask.tif"
            )
            mask_path = mask_out_dir / mask_filename
            vector_path = None
            if vector_out_dir:
                vector_filename = (
                    f"{tile_path.stem}_{concept_slug}_polygons.gpkg"
                    if concept_slug
                    else f"{tile_path.stem}_polygons.gpkg"
                )
                vector_path = vector_out_dir / vector_filename

            if not overwrite and mask_path.exists():
                logging.info(
                    "[%d/%d] Skipping %s; mask exists and overwrite is False",
                    idx,
                    len(tile_paths),
                    tile_path.name,
                )
                continue

            logging.info("[%d/%d] Processing %s", idx, len(tile_paths), tile_path.name)
            labeled_mask = segment_buildings_for_tile(
                model=model,
                image_array=image_array,
                concept=concept,
                confidence_threshold=confidence_threshold,
            )
            logging.info(
                "Mask stats for %s: labels=%d (confidence_threshold=%.3f)",
                tile_path.name,
                int(labeled_mask.max()),
                confidence_threshold,
            )

            mask_profile = profile.copy()
            mask_profile.update(
                dtype="uint16",
                count=1,
                nodata=0,
                compress="lzw",
                tiled=True,
                blockxsize=min(profile.get("blockxsize", labeled_mask.shape[1]), 512),
                blockysize=min(profile.get("blockysize", labeled_mask.shape[0]), 512),
            )
            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(labeled_mask.astype(np.uint16), 1)
            logging.info("Wrote mask to %s", mask_path)

            if overlay_out_dir:
                overlay_img = _render_overlay(image_array, labeled_mask)
                overlay_filename = (
                    f"{tile_path.stem}_{concept_slug}_overlay.png"
                    if concept_slug
                    else f"{tile_path.stem}_overlay.png"
                )
                overlay_path = overlay_out_dir / overlay_filename
                overlay_img.save(overlay_path)
                logging.info("Wrote overlay to %s", overlay_path)

            if vector_path:
                gdf = mask_to_polygons(
                    mask=labeled_mask,
                    transform=transform,
                    crs=crs,
                    min_area=min_area,
                )
                if len(gdf) == 0:
                    logging.info("No polygons above min_area=%s for %s", min_area, tile_path.name)
                else:
                    driver = "GPKG" if vector_path.suffix.lower() == ".gpkg" else "GeoJSON"
                    gdf.to_file(vector_path, driver=driver)
                    logging.info("Wrote vectors to %s", vector_path)
        except Exception as exc:  # pragma: no cover - runtime guardrail
            logging.exception("Failed processing %s: %s", tile_path, exc)
            continue


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 open-vocabulary segmentation over GeoTIFF tiles and export "
            "georeferenced masks (and optional vectors) ready for GIS tools."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder of GeoTIFF tiles exported from ArcGIS (default aligns with README paths).",
    )
    parser.add_argument(
        "--mask-out-dir",
        type=Path,
        required=True,
        help="Folder to write GeoTIFF masks.",
    )
    parser.add_argument(
        "--vector-out-dir",
        type=Path,
        default=None,
        help="Optional folder to write polygon vectors (GeoPackage or GeoJSON).",
    )
    parser.add_argument(
        "--overlay-out-dir",
        type=Path,
        default=None,
        help="Optional folder to save overlay PNGs (source + mask).",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="building",
        help='Text prompt for SAM3 concept segmentation (e.g., "building").',
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=20.0,
        help="Minimum polygon area to keep (uses CRS units).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence to keep a mask (passed to SAM3 processor).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to run inference on ("cuda" or "cpu").',
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Limit processing to the first N tiles for quick testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs when set.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_geo_building_pipeline(
        input_dir=args.input_dir,
        mask_out_dir=args.mask_out_dir,
        vector_out_dir=args.vector_out_dir,
        overlay_out_dir=args.overlay_out_dir,
        concept=args.concept,
        min_area=args.min_area,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        max_tiles=args.max_tiles,
        overwrite=args.overwrite,
    )
