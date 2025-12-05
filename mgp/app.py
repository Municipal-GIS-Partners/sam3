"""
MGP Building Segmentation Pipeline
Municipal GIS Partners • Spark Initiative

A user-friendly application for extracting building footprints from 
orthoimagery using SAM3 (Segment Anything Model 3).

This version uses a side-by-side image preview in the Segmentation step:
- Left: Original imagery
- Right: Segmentation overlay (buildings highlighted)

Requirements:
    pip install gradio segment-geospatial[samgeo3] rasterio geopandas pyproj
    pip install git+https://github.com/huggingface/transformers.git

Usage:
    python mgp_building_segmentation_app.py
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
import numpy as np

print(f"Gradio version: {gr.__version__}")

# Check for required packages
def check_dependencies():
    """Check if all required packages are installed."""
    missing = []
    try:
        import rasterio
    except ImportError:
        missing.append("rasterio")
    try:
        import geopandas
    except ImportError:
        missing.append("geopandas")
    try:
        from samgeo import SamGeo3  # noqa: F401
    except ImportError:
        missing.append("segment-geospatial[samgeo3]")
    try:
        from pyproj import CRS  # noqa: F401
    except ImportError:
        missing.append("pyproj")
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

# Only import if dependencies are available
if check_dependencies():
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.windows import Window
    from rasterio.features import shapes
    import geopandas as gpd
    from shapely.geometry import shape
    from pyproj import CRS
    from samgeo import SamGeo3, download_file


# =============================================================================
# CONFIGURATION
# =============================================================================

CRS_OPTIONS = [
    "EPSG:6455 - Illinois East (ftUS)",
    "EPSG:6456 - Illinois West (ftUS)", 
    "EPSG:3435 - Illinois East (ftUS) [Legacy]",
    "EPSG:3436 - Illinois West (ftUS) [Legacy]",
    "EPSG:4326 - WGS84 (Lat/Long)",
    "Keep Original CRS"
]

CRS_MAP = {
    "EPSG:6455 - Illinois East (ftUS)": "EPSG:6455",
    "EPSG:6456 - Illinois West (ftUS)": "EPSG:6456",
    "EPSG:3435 - Illinois East (ftUS) [Legacy]": "EPSG:3435",
    "EPSG:3436 - Illinois West (ftUS) [Legacy]": "EPSG:3436",
    "EPSG:4326 - WGS84 (Lat/Long)": "EPSG:4326",
    "Keep Original CRS": None
}

TILE_SIZE_OPTIONS = [
    "2048 x 2048 (Recommended)",
    "1024 x 1024 (Small structures)",
    "4096 x 4096 (Large buildings)",
    "No Tiling"
]

TILE_SIZE_MAP = {
    "2048 x 2048 (Recommended)": 2048,
    "1024 x 1024 (Small structures)": 1024,
    "4096 x 4096 (Large buildings)": 4096,
    "No Tiling": None
}

# Predefined prompt suggestions for common use cases
PROMPT_SUGGESTIONS = [
    "building",
    "house", 
    "garage",
    "shed",
    "commercial building",
    "roof",
    "structure",
    "parking lot",
    "pool",
    "solar panel"
]

# Global state
app_state = {
    "sam3": None,
    "current_image": None,
    "current_masks": None,
    "buildings_gdf": None,
    "output_dir": None,
    "is_initialized": False
}


# =============================================================================
# IMAGE PREVIEW HELPERS
# =============================================================================

def stretch_to_ubyte(arr: np.ndarray) -> np.ndarray:
    """Contrast-stretch a single band to 0–255 uint8."""
    arr = arr.astype(np.float32)
    v_min, v_max = np.percentile(arr, (2, 98))
    if v_max <= v_min:
        v_min, v_max = arr.min(), arr.max()
    if v_max <= v_min:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - v_min) / (v_max - v_min)
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def read_raster_as_rgb(image_path: str, max_size: int = 1024) -> np.ndarray:
    """Read a GeoTIFF and return an RGB uint8 array suitable for Gradio."""
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        scale = 1.0
        max_dim = max(width, height)
        if max_dim > max_size:
            scale = max_size / max_dim
        out_height = max(1, int(height * scale))
        out_width = max(1, int(width * scale))

        if src.count >= 3:
            data = src.read(
                [1, 2, 3],
                out_shape=(3, out_height, out_width),
                resampling=Resampling.bilinear
            )
        else:
            # Grayscale → RGB
            data = src.read(
                1,
                out_shape=(1, out_height, out_width),
                resampling=Resampling.bilinear
            )
            data = np.repeat(data, 3, axis=0)

    rgb = np.zeros_like(data, dtype=np.uint8)
    for i in range(3):
        rgb[i] = stretch_to_ubyte(data[i])

    rgb = np.moveaxis(rgb, 0, -1)  # (H, W, 3)
    return rgb


def make_segmentation_overlay(image_path: str, mask_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build:
    - original_rgb: RGB preview of imagery
    - overlay_rgb: imagery with mask overlaid in red
    """
    if image_path is None or not os.path.exists(image_path):
        return None, None

    original_rgb = read_raster_as_rgb(image_path)

    if mask_path is None or not os.path.exists(mask_path):
        return original_rgb, None

    with rasterio.open(mask_path) as msrc:
        mask = msrc.read(1)

    # Resize mask to match preview dimensions if needed
    h_img, w_img, _ = original_rgb.shape
    h_mask, w_mask = mask.shape

    if (h_mask, w_mask) != (h_img, w_img):
        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(
                1,
                out_shape=(h_img, w_img),
                resampling=Resampling.nearest
            )

    overlay = original_rgb.copy()
    mask_binary = mask > 0

    if np.any(mask_binary):
        # Blend red where mask is > 0
        red_color = np.array([255, 0, 0], dtype=np.float32)
        alpha = 0.5
        overlay_float = overlay.astype(np.float32)
        overlay_float[mask_binary] = (
            (1 - alpha) * overlay_float[mask_binary] + alpha * red_color
        )
        overlay = np.clip(overlay_float, 0, 255).astype(np.uint8)

    return original_rgb, overlay


def get_preview_images() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return current preview images based on app_state."""
    return make_segmentation_overlay(
        app_state.get("current_image"),
        app_state.get("current_masks")
    )


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_output_dir():
    """Get or create temporary output directory."""
    if app_state["output_dir"] is None:
        app_state["output_dir"] = Path(tempfile.mkdtemp(prefix="mgp_segmentation_"))
    return app_state["output_dir"]


def initialize_sam3() -> str:
    """Initialize the SAM3 model."""
    if app_state["is_initialized"]:
        return "✓ SAM3 already initialized and ready!"
    
    try:
        app_state["sam3"] = SamGeo3(
            backend="meta",
            device=None,
            checkpoint_path=None,
            load_from_HF=True
        )
        app_state["is_initialized"] = True
        
        try:
            import torch
            device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                device = f"GPU: {gpu_name}"
        except ImportError:
            device = "Unknown"
        
        return f"""✓ SAM3 initialized successfully!
        
Device: {device}

You can now upload imagery in Step 2.
"""
    
    except Exception as e:
        return f"""✗ Failed to initialize SAM3:
{str(e)}

Troubleshooting:
1. Request access at huggingface.co/facebook/sam3
2. Run: huggingface-cli login
3. Install transformers from git:
   pip install git+https://github.com/huggingface/transformers.git"""


def inspect_raster(file_obj) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Inspect uploaded raster and return metadata + preview images."""
    if file_obj is None:
        return (
            "No file uploaded yet.",
            None,
            None,
        )
    
    # Handle different Gradio file object types
    if hasattr(file_obj, 'name'):
        file_path = file_obj.name
    else:
        file_path = file_obj
    
    try:
        with rasterio.open(file_path) as src:
            crs_info = str(src.crs) if src.crs else "No CRS defined!"
            epsg = src.crs.to_epsg() if src.crs else None
            
            if epsg == 6455:
                crs_status = "Already in Illinois State Plane East (EPSG:6455)"
            elif epsg:
                crs_status = f"Current EPSG:{epsg} - May need reprojection"
            else:
                crs_status = "No CRS - Georeferencing required!"
            
            res_x, res_y = src.res
            
            report = f"""RASTER METADATA
================
File: {Path(file_path).name}
Dimensions: {src.width:,} x {src.height:,} pixels
Bands: {src.count}
Data Type: {src.dtypes[0]}
Resolution: {res_x:.4f} x {res_y:.4f}
NoData: {src.nodata}

COORDINATE SYSTEM
=================
CRS: {crs_info}
Status: {crs_status}

BOUNDS
======
Left: {src.bounds.left:,.2f}
Bottom: {src.bounds.bottom:,.2f}
Right: {src.bounds.right:,.2f}
Top: {src.bounds.top:,.2f}
"""
            app_state["current_image"] = file_path

        original, overlay = get_preview_images()
        return report, original, overlay
    
    except Exception as e:
        return f"Error reading file: {str(e)}", None, None


def reproject_raster(target_crs_name: str) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Reproject raster to target CRS."""
    input_path = app_state.get("current_image")
    if input_path is None:
        return "No file uploaded.", None, None
    
    target_crs = CRS_MAP.get(target_crs_name)
    
    if target_crs is None:
        # Keep original
        status = "Keeping original CRS - no reprojection needed."
        original, overlay = get_preview_images()
        return status, original, overlay
    
    try:
        with rasterio.open(input_path) as src:
            if src.crs and src.crs.to_epsg() == int(target_crs.split(":")[1]):
                status = f"Already in {target_crs} - no reprojection needed."
                original, overlay = get_preview_images()
                return status, original, overlay
            
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            output_dir = get_output_dir()
            output_path = output_dir / f"reprojected_{Path(input_path).stem}.tif"
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            
            app_state["current_image"] = str(output_path)
            original, overlay = get_preview_images()
            
            return f"Reprojected to {target_crs}\nOutput: {output_path.name}", original, overlay
    
    except Exception as e:
        return f"Reprojection failed: {str(e)}", None, None


def tile_raster(tile_size_name: str) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split raster into tiles, load first tile for preview/segmentation."""
    input_path = app_state.get("current_image")
    if input_path is None:
        return "No file uploaded.", None, None
    
    tile_size = TILE_SIZE_MAP.get(tile_size_name)
    
    if tile_size is None:
        status = "No tiling selected - using full image."
        original, overlay = get_preview_images()
        return status, original, overlay
    
    try:
        output_dir = get_output_dir() / "tiles"
        output_dir.mkdir(exist_ok=True)
        
        tile_paths = []
        
        with rasterio.open(input_path) as src:
            if src.width <= tile_size and src.height <= tile_size:
                status = f"Image ({src.width}x{src.height}) fits in single tile."
                original, overlay = get_preview_images()
                return status, original, overlay
            
            n_tiles_x = (src.width + tile_size - 1) // tile_size
            n_tiles_y = (src.height + tile_size - 1) // tile_size
            
            for row in range(n_tiles_y):
                for col in range(n_tiles_x):
                    x_off = col * tile_size
                    y_off = row * tile_size
                    width = min(tile_size, src.width - x_off)
                    height = min(tile_size, src.height - y_off)
                    
                    window = Window(x_off, y_off, width, height)
                    tile_data = src.read(window=window)
                    
                    if tile_data.max() == 0:
                        continue
                    
                    tile_transform = src.window_transform(window)
                    tile_name = f"tile_{row:03d}_{col:03d}.tif"
                    tile_path = output_dir / tile_name
                    
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'width': width,
                        'height': height,
                        'transform': tile_transform
                    })
                    
                    with rasterio.open(tile_path, 'w', **kwargs) as dst:
                        dst.write(tile_data)
                    
                    tile_paths.append(str(tile_path))
            
            if tile_paths:
                app_state["current_image"] = tile_paths[0]
                original, overlay = get_preview_images()
            else:
                original, overlay = None, None
            
            status = f"""TILING COMPLETE
===============
Input: {src.width:,} x {src.height:,} pixels
Tile Size: {tile_size} x {tile_size}
Grid: {n_tiles_x} x {n_tiles_y}
Tiles Created: {len(tile_paths)}

First tile loaded for segmentation.
"""
            return status, original, overlay
    
    except Exception as e:
        return f"Tiling failed: {str(e)}", None, None


def run_segmentation(
    text_prompt: str, 
    min_area: float,
    use_confidence_filter: bool,
    confidence_threshold: float,
    use_stability_filter: bool,
    stability_threshold: float
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run SAM3 segmentation on image with confidence filtering options."""
    if not app_state["is_initialized"]:
        status = "⚠️ Please initialize SAM3 first (Step 1)"
        original, overlay = get_preview_images()
        return status, original, overlay
    
    image_path = app_state.get("current_image")
    if image_path is None:
        status = "⚠️ No image loaded. Please upload imagery first."
        original, overlay = get_preview_images()
        return status, original, overlay
    
    if not text_prompt.strip():
        status = "⚠️ Please enter a text prompt (e.g., 'building', 'house', 'garage')"
        original, overlay = get_preview_images()
        return status, original, overlay
    
    try:
        sam3 = app_state["sam3"]
        
        # Set the image
        sam3.set_image(image_path)
        
        # Build generation parameters
        gen_kwargs = {
            "prompt": text_prompt.strip()
        }
        
        # Apply confidence/stability filtering if enabled
        if use_confidence_filter:
            gen_kwargs["pred_iou_thresh"] = confidence_threshold
        
        if use_stability_filter:
            gen_kwargs["stability_score_thresh"] = stability_threshold
        
        # Generate masks
        sam3.generate_masks(**gen_kwargs)
        
        output_dir = get_output_dir()
        mask_path = output_dir / "masks.tif"
        sam3.save_masks(str(mask_path))
        
        app_state["current_masks"] = str(mask_path)
        
        # Count detected objects
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
        
        unique_vals = np.unique(mask_data)
        unique_vals = unique_vals[unique_vals > 0]
        
        # Build filter info string
        filter_info = []
        if use_confidence_filter:
            filter_info.append(f"Confidence ≥ {confidence_threshold:.2f}")
        if use_stability_filter:
            filter_info.append(f"Stability ≥ {stability_threshold:.2f}")
        filter_str = " | ".join(filter_info) if filter_info else "None"
        
        report = f"""SEGMENTATION COMPLETE
=====================
Prompt: "{text_prompt}"
Objects Detected: {len(unique_vals)}
Filters Applied: {filter_str}
Mask File: masks.tif

The right-hand preview now shows the segmentation overlay.
Next: Click "Vectorize & Export" to convert to GIS polygons.
"""
        original, overlay = get_preview_images()
        return report, original, overlay
    
    except Exception as e:
        import traceback
        status = f"Segmentation failed:\n{str(e)}\n\n{traceback.format_exc()}"
        original, overlay = get_preview_images()
        return status, original, overlay


def vectorize_and_export(municipality: str, min_area: float, target_crs_name: str) -> Tuple[str, Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert masks to vector polygons and export."""
    mask_path = app_state.get("current_masks")
    image_path = app_state.get("current_image")
    
    if mask_path is None:
        original, overlay = get_preview_images()
        return "No masks to vectorize. Run segmentation first.", None, original, overlay
    
    target_crs = CRS_MAP.get(target_crs_name, "EPSG:6455")
    if target_crs is None:
        target_crs = "EPSG:6455"
    
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            src_crs = src.crs
            
            results = list(shapes(mask.astype(np.int32), transform=transform))
            
            geometries = []
            values = []
            for geom, value in results:
                if value > 0:
                    geometries.append(shape(geom))
                    values.append(int(value))
            
            if not geometries:
                original, overlay = get_preview_images()
                return "No features extracted from mask.", None, original, overlay
            
            gdf = gpd.GeoDataFrame(
                {'mask_id': values}, 
                geometry=geometries, 
                crs=src_crs
            )
        
        # Reproject if needed
        if gdf.crs and str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        # Calculate area and filter
        gdf['area_sqft'] = gdf.geometry.area
        initial_count = len(gdf)
        gdf = gdf[gdf['area_sqft'] >= min_area].copy()
        filtered_count = initial_count - len(gdf)
        
        # Add attributes
        gdf['building_id'] = range(1, len(gdf) + 1)
        gdf['municipality'] = municipality if municipality else "Unknown"
        gdf['source'] = 'SAM3_Segmentation'
        gdf['proc_date'] = datetime.now().strftime("%Y-%m-%d")
        gdf['area_sqft'] = gdf['area_sqft'].round(2)
        
        gdf = gdf[['building_id', 'municipality', 'area_sqft', 'source', 'proc_date', 'geometry']]
        
        app_state["buildings_gdf"] = gdf
        
        # Export
        output_dir = get_output_dir()
        base_name = f"{municipality or 'buildings'}_{datetime.now().strftime('%Y%m%d')}"
        
        gpkg_path = output_dir / f"{base_name}.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG", layer="building_footprints")
        
        shp_dir = output_dir / "shapefile"
        shp_dir.mkdir(exist_ok=True)
        shp_path = shp_dir / f"{base_name}.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        
        geojson_path = output_dir / f"{base_name}.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        
        # Create ZIP
        zip_path = output_dir / f"{base_name}_all_formats.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(gpkg_path, gpkg_path.name)
            zipf.write(geojson_path, geojson_path.name)
            for f in shp_dir.glob("*"):
                zipf.write(f, f"shapefile/{f.name}")
        
        total_area = gdf['area_sqft'].sum()
        mean_area = gdf['area_sqft'].mean()
        
        report = f"""EXPORT COMPLETE
===============
Municipality: {municipality or 'Not specified'}
Buildings: {len(gdf):,}
Filtered (too small): {filtered_count:,}
Total Area: {total_area:,.0f} sq ft
Average Size: {mean_area:,.0f} sq ft
CRS: {target_crs}

FILES CREATED
=============
- {gpkg_path.name} (GeoPackage)
- {shp_path.name} (Shapefile)  
- {geojson_path.name} (GeoJSON)
- {zip_path.name} (All formats)

AREA STATISTICS
===============
Min: {gdf['area_sqft'].min():,.0f} sq ft
Max: {gdf['area_sqft'].max():,.0f} sq ft
Median: {gdf['area_sqft'].median():,.0f} sq ft

The right-hand preview shows the same segmentation overlay used to generate these polygons.
"""
        original, overlay = get_preview_images()
        return report, str(zip_path), original, overlay
    
    except Exception as e:
        import traceback
        status = f"Export failed:\n{str(e)}\n\n{traceback.format_exc()}"
        original, overlay = get_preview_images()
        return status, None, original, overlay


def load_demo_image() -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load demo imagery for testing."""
    try:
        url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif"
        image_path = download_file(url)
        app_state["current_image"] = image_path
        metadata, _, _ = inspect_raster(image_path)
        original, overlay = get_preview_images()
        return metadata, original, overlay
    except Exception as e:
        return f"Failed to load demo: {str(e)}", None, None


def update_prompt_from_suggestion(suggestion: str) -> str:
    """Update text prompt from dropdown suggestion."""
    return suggestion


def toggle_confidence_visibility(enabled: bool):
    """Toggle visibility of confidence threshold slider."""
    return gr.update(visible=enabled)


def toggle_stability_visibility(enabled: bool):
    """Toggle visibility of stability threshold slider."""
    return gr.update(visible=enabled)


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_app():
    """Create the Gradio application."""
    
    with gr.Blocks() as app:
        
        gr.Markdown("""
# MGP Building Segmentation Pipeline
### Municipal GIS Partners - Spark Initiative

Extract building footprints from orthoimagery using AI-powered segmentation (SAM3).

The side-by-side comparison lives in **Step 4 (Segmentation)**:
- **Left:** Original imagery  
- **Right:** Segmentation overlay (buildings highlighted in red)

---
        """)
        
        # Step 1: Initialize
        gr.Markdown("## Step 1: Initialize SAM3 Model")
        gr.Markdown("""
**First Time Setup:** Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), 
then run `huggingface-cli login` in your terminal.
        """)
        
        with gr.Row():
            init_btn = gr.Button("Initialize SAM3")
            demo_btn = gr.Button("Load Demo Image")
        
        init_status = gr.Textbox(label="Status", lines=4, interactive=False)
        
        gr.Markdown("---")
        
        # Step 2: Upload
        gr.Markdown("## Step 2: Upload Imagery")
        gr.Markdown("Upload orthoimagery in GeoTIFF format. Recommended: Cook County imagery, NAIP, municipal flights.")
        
        file_upload = gr.File(label="Upload GeoTIFF", file_types=[".tif", ".tiff"])
        metadata_display = gr.Textbox(label="Metadata", lines=12, interactive=False)
        
        gr.Markdown("---")
        
        # Step 3: CRS & Tiling
        gr.Markdown("## Step 3: Coordinate System & Tiling")
        gr.Markdown("Reproject to Illinois State Plane and tile large images for better detection.")
        
        with gr.Row():
            crs_dropdown = gr.Dropdown(
                choices=CRS_OPTIONS,
                value="EPSG:6455 - Illinois East (ftUS)",
                label="Target CRS"
            )
            reproject_btn = gr.Button("Reproject")
        
        reproject_status = gr.Textbox(label="Reproject Status", lines=3, interactive=False)
        
        with gr.Row():
            tile_dropdown = gr.Dropdown(
                choices=TILE_SIZE_OPTIONS,
                value="2048 x 2048 (Recommended)",
                label="Tile Size"
            )
            tile_btn = gr.Button("Create Tiles")
        
        tile_status = gr.Textbox(label="Tile Status", lines=6, interactive=False)
        
        gr.Markdown("---")
        
        # Step 4: Segmentation (with side-by-side view)
        gr.Markdown("## Step 4: Run Segmentation")
        gr.Markdown("""
Enter a text prompt describing what to detect: **"building"** (general), **"house"** (residential), **"garage"** (accessory structures)

The panels below show:
- **Left:** Original imagery
- **Right:** Segmentation overlay (buildings highlighted in red) once segmentation has been run.
        """)
        
        with gr.Row():
            original_image_display = gr.Image(label="Original Imagery", interactive=False)
            segmentation_display = gr.Image(label="Segmentation Overlay", interactive=False)
        
        with gr.Row():
            prompt_suggestions = gr.Dropdown(
                choices=PROMPT_SUGGESTIONS,
                label="Quick Select Prompt",
                value="building"
            )
            text_prompt = gr.Textbox(label="Custom Prompt", value="building")
            min_area_slider = gr.Slider(
                minimum=0,
                maximum=1000,
                value=100,
                step=10,
                label="Min Area (sq ft)"
            )
        
        # Confidence filtering toggles
        gr.Markdown("### Confidence Filters (Advanced)")
        gr.Markdown("Enable these to filter out low-confidence detections. Higher thresholds = fewer but more reliable results.")
        
        with gr.Row():
            use_confidence_filter = gr.Checkbox(
                label="Enable Confidence Filter (pred_iou_thresh)",
                value=False
            )
            confidence_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.88,
                step=0.01,
                label="Confidence Threshold",
                visible=False
            )
        
        with gr.Row():
            use_stability_filter = gr.Checkbox(
                label="Enable Stability Filter (stability_score_thresh)",
                value=False
            )
            stability_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.95,
                step=0.01,
                label="Stability Threshold",
                visible=False
            )
        
        segment_btn = gr.Button("Run Segmentation")
        segment_status = gr.Textbox(label="Segmentation Status", lines=8, interactive=False)
        
        gr.Markdown("---")
        
        # Step 5: Export
        gr.Markdown("## Step 5: Export to GIS")
        gr.Markdown("Convert masks to vector polygons. Exports: GeoPackage, Shapefile, GeoJSON.")
        
        with gr.Row():
            municipality_input = gr.Textbox(label="Municipality Name", placeholder="e.g., Lake Forest")
            export_crs = gr.Dropdown(
                choices=CRS_OPTIONS,
                value="EPSG:6455 - Illinois East (ftUS)",
                label="Export CRS"
            )
        
        export_btn = gr.Button("Vectorize & Export")
        
        export_status = gr.Textbox(label="Export Status", lines=15, interactive=False)
        download_file_output = gr.File(label="Download Results")
        
        gr.Markdown("---")
        
        # Help
        gr.Markdown("""
## Help & Resources

**Imagery Sources:**
- Cook County GIS Hub - 6 inch resolution
- NAIP - 1 meter resolution (via Geospatial Data Gateway)

**Coordinate Systems:**
- EPSG:6455 - Illinois East (Cook, Lake, DuPage, etc.)
- EPSG:6456 - Illinois West

**Confidence Filters Explained:**
- **Confidence (pred_iou_thresh)**: SAM3's predicted IoU score. Higher = more confident. Default: 0.88
- **Stability (stability_score_thresh)**: Mask consistency score. Higher = cleaner boundaries. Default: 0.95

**Troubleshooting:**
- "SAM3 not initialized" → Click Initialize in Step 1
- "No CRS defined" → Add georeferencing in ArcGIS/QGIS
- Missing small buildings → Use 1024x1024 tile size
- Too many false positives → Enable confidence filter, increase threshold

---
*MGP Building Segmentation Pipeline v1.2 | Municipal GIS Partners | Spark Initiative*
        """)
        
        # =================================================================
        # Event Handlers
        # =================================================================
        
        # Initialize button
        init_btn.click(fn=initialize_sam3, outputs=init_status)
        
        # Demo button
        demo_btn.click(
            fn=load_demo_image,
            outputs=[metadata_display, original_image_display, segmentation_display]
        )
        
        # File upload
        file_upload.change(
            fn=inspect_raster,
            inputs=file_upload,
            outputs=[metadata_display, original_image_display, segmentation_display]
        )
        
        # Reproject
        reproject_btn.click(
            fn=reproject_raster,
            inputs=[crs_dropdown],
            outputs=[reproject_status, original_image_display, segmentation_display]
        )
        
        # Tile
        tile_btn.click(
            fn=tile_raster,
            inputs=[tile_dropdown],
            outputs=[tile_status, original_image_display, segmentation_display]
        )
        
        # Prompt suggestion updates text prompt
        prompt_suggestions.change(
            fn=update_prompt_from_suggestion,
            inputs=[prompt_suggestions],
            outputs=[text_prompt]
        )
        
        # Toggle confidence slider visibility
        use_confidence_filter.change(
            fn=toggle_confidence_visibility,
            inputs=[use_confidence_filter],
            outputs=[confidence_threshold]
        )
        
        # Toggle stability slider visibility
        use_stability_filter.change(
            fn=toggle_stability_visibility,
            inputs=[use_stability_filter],
            outputs=[stability_threshold]
        )
        
        # Segmentation button
        segment_btn.click(
            fn=run_segmentation,
            inputs=[
                text_prompt, 
                min_area_slider,
                use_confidence_filter,
                confidence_threshold,
                use_stability_filter,
                stability_threshold
            ],
            outputs=[segment_status, original_image_display, segmentation_display]
        )
        
        # Export button
        export_btn.click(
            fn=vectorize_and_export,
            inputs=[municipality_input, min_area_slider, export_crs],
            outputs=[export_status, download_file_output, original_image_display, segmentation_display]
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ============================================================
         MGP Building Segmentation Pipeline v1.2
         Municipal GIS Partners - AI/ML Initiative
                        John V. Kenny
    ============================================================
    """)
    
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
