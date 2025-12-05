"""
MGP Building Segmentation Pipeline
Municipal GIS Partners • Spark Initiative

A user-friendly application for extracting building footprints from 
orthoimagery using SAM3 (Segment Anything Model 3).

Requirements:
    pip install gradio segment-geospatial[samgeo3] rasterio geopandas pyproj leafmap
    pip install git+https://github.com/huggingface/transformers.git

Usage:
    python mgp_building_segmentation_app.py
"""

import os
import sys
import tempfile
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
import numpy as np

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
        from samgeo import SamGeo3
    except ImportError:
        missing.append("segment-geospatial[samgeo3]")
    try:
        import leafmap
    except ImportError:
        missing.append("leafmap")
    
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

# Illinois State Plane coordinate systems
CRS_OPTIONS = {
    "EPSG:6455 - Illinois East (ftUS)": "EPSG:6455",
    "EPSG:6456 - Illinois West (ftUS)": "EPSG:6456",
    "EPSG:3435 - Illinois East (ftUS) [Legacy]": "EPSG:3435",
    "EPSG:3436 - Illinois West (ftUS) [Legacy]": "EPSG:3436",
    "EPSG:4326 - WGS84 (Lat/Long)": "EPSG:4326",
    "Keep Original CRS": None
}

TILE_SIZE_OPTIONS = {
    "2048 × 2048 (Recommended for residential)": 2048,
    "1024 × 1024 (Better for small structures)": 1024,
    "4096 × 4096 (Large commercial/industrial)": 4096,
    "No Tiling (Small images only)": None
}

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
# CORE FUNCTIONS
# =============================================================================

def get_output_dir():
    """Get or create temporary output directory."""
    if app_state["output_dir"] is None:
        app_state["output_dir"] = Path(tempfile.mkdtemp(prefix="mgp_segmentation_"))
    return app_state["output_dir"]


def initialize_sam3(progress=gr.Progress()) -> str:
    """Initialize the SAM3 model."""
    if app_state["is_initialized"]:
        return "✅ SAM3 already initialized and ready!"
    
    try:
        progress(0.2, desc="Loading SAM3 model...")
        app_state["sam3"] = SamGeo3(
            backend="transformers",
            device=None,  # Auto-detect GPU/CPU
            checkpoint_path=None,
            load_from_HF=True
        )
        app_state["is_initialized"] = True
        progress(1.0, desc="Complete!")
        
        # Check if GPU is available
        import torch
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        return f"✅ SAM3 initialized successfully!\n\n**Device:** {device}\n\nYou can now upload imagery and begin segmentation."
    
    except Exception as e:
        return f"❌ Failed to initialize SAM3:\n\n{str(e)}\n\n**Troubleshooting:**\n1. Ensure you have requested access at huggingface.co/facebook/sam3\n2. Run `huggingface-cli login` in terminal\n3. Check that transformers is installed from git"


def inspect_raster(file_path: str) -> Tuple[str, Optional[np.ndarray]]:
    """Inspect uploaded raster and return metadata."""
    if file_path is None:
        return "No file uploaded yet.", None
    
    try:
        with rasterio.open(file_path) as src:
            # Read preview image (first 3 bands, downsampled)
            scale = max(1, max(src.width, src.height) // 1000)
            preview = src.read(
                indexes=[1, 2, 3] if src.count >= 3 else [1],
                out_shape=(
                    min(3, src.count),
                    src.height // scale,
                    src.width // scale
                ),
                resampling=Resampling.bilinear
            )
            
            # Normalize for display
            if preview.shape[0] == 3:
                preview = np.transpose(preview, (1, 2, 0))
                preview = ((preview - preview.min()) / (preview.max() - preview.min()) * 255).astype(np.uint8)
            else:
                preview = preview[0]
                preview = ((preview - preview.min()) / (preview.max() - preview.min()) * 255).astype(np.uint8)
            
            # Build metadata report
            crs_info = str(src.crs) if src.crs else "⚠️ No CRS defined!"
            epsg = src.crs.to_epsg() if src.crs else None
            
            # Check CRS compatibility
            crs_status = ""
            if epsg == 6455:
                crs_status = "✅ Already in Illinois State Plane East (EPSG:6455)"
            elif epsg:
                crs_status = f"ℹ️ Current EPSG:{epsg} - May need reprojection"
            else:
                crs_status = "⚠️ No CRS - Georeferencing required!"
            
            # Resolution in feet (approximate if in degrees)
            res_x, res_y = src.res
            if src.crs and src.crs.is_geographic:
                res_note = f"{res_x:.8f}° × {res_y:.8f}° (geographic)"
            else:
                units = src.crs.linear_units if src.crs else "units"
                res_note = f"{res_x:.4f} × {res_y:.4f} {units}"
            
            # Tiling recommendation
            total_pixels = src.width * src.height
            if total_pixels > 4096 * 4096:
                tile_rec = "🔶 **Large image** - Tiling strongly recommended (2048×2048)"
            elif total_pixels > 2048 * 2048:
                tile_rec = "ℹ️ Medium image - Tiling recommended for best results"
            else:
                tile_rec = "✅ Small image - Can process without tiling"
            
            report = f"""## 📊 Raster Metadata

| Property | Value |
|----------|-------|
| **File** | {Path(file_path).name} |
| **Dimensions** | {src.width:,} × {src.height:,} pixels |
| **Bands** | {src.count} |
| **Data Type** | {src.dtypes[0]} |
| **Resolution** | {res_note} |
| **NoData** | {src.nodata} |

### 🗺️ Coordinate Reference System
**CRS:** {crs_info}

{crs_status}

### 📐 Bounds
- **Left:** {src.bounds.left:,.2f}
- **Bottom:** {src.bounds.bottom:,.2f}
- **Right:** {src.bounds.right:,.2f}
- **Top:** {src.bounds.top:,.2f}

### 💡 Recommendation
{tile_rec}
"""
            app_state["current_image"] = file_path
            return report, preview
    
    except Exception as e:
        return f"❌ Error reading file:\n\n{str(e)}", None


def reproject_raster(
    input_path: str,
    target_crs_name: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Reproject raster to target CRS."""
    if input_path is None:
        return "No file uploaded.", ""
    
    target_crs = CRS_OPTIONS.get(target_crs_name)
    
    if target_crs is None:
        return "Keeping original CRS - no reprojection needed.", input_path
    
    try:
        with rasterio.open(input_path) as src:
            # Check if already in target CRS
            if src.crs and src.crs.to_epsg() == int(target_crs.split(":")[1]):
                return f"✅ Already in {target_crs} - no reprojection needed.", input_path
            
            progress(0.2, desc="Calculating transform...")
            
            # Calculate new transform
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
            
            # Output path
            output_dir = get_output_dir()
            output_path = output_dir / f"reprojected_{Path(input_path).stem}.tif"
            
            progress(0.4, desc="Reprojecting bands...")
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    progress(0.4 + (0.5 * i / src.count), desc=f"Reprojecting band {i}/{src.count}...")
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            
            progress(1.0, desc="Complete!")
            app_state["current_image"] = str(output_path)
            
            return f"✅ Reprojected to {target_crs}\n\n**Output:** {output_path.name}", str(output_path)
    
    except Exception as e:
        return f"❌ Reprojection failed:\n\n{str(e)}", ""


def tile_raster(
    input_path: str,
    tile_size_name: str,
    progress=gr.Progress()
) -> Tuple[str, List[str]]:
    """Split raster into tiles."""
    if input_path is None:
        return "No file uploaded.", []
    
    tile_size = TILE_SIZE_OPTIONS.get(tile_size_name)
    
    if tile_size is None:
        return "No tiling selected - using full image.", [input_path]
    
    try:
        output_dir = get_output_dir() / "tiles"
        output_dir.mkdir(exist_ok=True)
        
        tile_paths = []
        
        with rasterio.open(input_path) as src:
            # Check if tiling needed
            if src.width <= tile_size and src.height <= tile_size:
                return f"✅ Image ({src.width}×{src.height}) fits in single tile.", [input_path]
            
            n_tiles_x = (src.width + tile_size - 1) // tile_size
            n_tiles_y = (src.height + tile_size - 1) // tile_size
            total_tiles = n_tiles_x * n_tiles_y
            
            progress(0.1, desc=f"Creating {total_tiles} tiles...")
            
            tile_count = 0
            for row in range(n_tiles_y):
                for col in range(n_tiles_x):
                    tile_count += 1
                    progress(0.1 + (0.8 * tile_count / total_tiles), 
                            desc=f"Tile {tile_count}/{total_tiles}")
                    
                    x_off = col * tile_size
                    y_off = row * tile_size
                    width = min(tile_size, src.width - x_off)
                    height = min(tile_size, src.height - y_off)
                    
                    window = Window(x_off, y_off, width, height)
                    tile_data = src.read(window=window)
                    
                    # Skip empty tiles
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
            
            progress(1.0, desc="Complete!")
            
            report = f"""✅ **Tiling Complete**

| Property | Value |
|----------|-------|
| **Input Size** | {src.width:,} × {src.height:,} pixels |
| **Tile Size** | {tile_size} × {tile_size} pixels |
| **Grid** | {n_tiles_x} × {n_tiles_y} |
| **Total Tiles** | {len(tile_paths)} (excluding empty) |

Tiles saved to: `{output_dir}`
"""
            return report, tile_paths
    
    except Exception as e:
        return f"❌ Tiling failed:\n\n{str(e)}", []


def run_segmentation(
    image_path: str,
    text_prompt: str,
    min_area: float,
    progress=gr.Progress()
) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
    """Run SAM3 segmentation on image."""
    if not app_state["is_initialized"]:
        return "❌ Please initialize SAM3 first (Step 1)", None, None
    
    if image_path is None or image_path == "":
        return "❌ No image loaded. Please upload imagery first.", None, None
    
    try:
        sam3 = app_state["sam3"]
        
        progress(0.2, desc="Loading image into SAM3...")
        sam3.set_image(image_path)
        
        progress(0.5, desc=f"Generating masks with prompt: '{text_prompt}'...")
        sam3.generate_masks(prompt=text_prompt)
        
        progress(0.7, desc="Saving masks...")
        output_dir = get_output_dir()
        mask_path = output_dir / "masks.tif"
        sam3.save_masks(str(mask_path))
        
        app_state["current_masks"] = str(mask_path)
        
        # Create visualization
        progress(0.85, desc="Creating visualization...")
        
        # Read mask for display
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
        
        # Color the masks
        unique_vals = np.unique(mask_data)
        unique_vals = unique_vals[unique_vals > 0]  # Exclude background
        
        # Create colored visualization
        colored = np.zeros((*mask_data.shape, 3), dtype=np.uint8)
        np.random.seed(42)  # Consistent colors
        for val in unique_vals:
            color = np.random.randint(50, 255, 3)
            colored[mask_data == val] = color
        
        progress(1.0, desc="Complete!")
        
        report = f"""✅ **Segmentation Complete**

| Property | Value |
|----------|-------|
| **Prompt** | "{text_prompt}" |
| **Objects Detected** | {len(unique_vals)} |
| **Mask File** | masks.tif |

**Next Step:** Click "Vectorize & Export" to convert masks to GIS polygons.
"""
        return report, colored, str(mask_path)
    
    except Exception as e:
        return f"❌ Segmentation failed:\n\n{str(e)}", None, None


def vectorize_and_export(
    mask_path: str,
    municipality: str,
    min_area: float,
    target_crs_name: str,
    progress=gr.Progress()
) -> Tuple[str, Optional[str]]:
    """Convert masks to vector polygons and export."""
    if mask_path is None or mask_path == "":
        return "❌ No masks to vectorize. Run segmentation first.", None
    
    target_crs = CRS_OPTIONS.get(target_crs_name, "EPSG:6455")
    if target_crs is None:
        target_crs = "EPSG:6455"  # Default
    
    try:
        progress(0.2, desc="Reading mask raster...")
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            src_crs = src.crs
            
            progress(0.4, desc="Extracting polygons...")
            
            results = list(shapes(mask.astype(np.int32), transform=transform))
            
            geometries = []
            values = []
            for geom, value in results:
                if value > 0:
                    geometries.append(shape(geom))
                    values.append(int(value))
            
            if not geometries:
                return "⚠️ No features extracted from mask.", None
            
            gdf = gpd.GeoDataFrame(
                {'mask_id': values}, 
                geometry=geometries, 
                crs=src_crs
            )
        
        progress(0.6, desc="Reprojecting vectors...")
        
        # Reproject if needed
        if gdf.crs and str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        progress(0.7, desc="Calculating areas and filtering...")
        
        # Calculate area
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
        
        # Reorder columns
        gdf = gdf[['building_id', 'municipality', 'area_sqft', 'source', 'proc_date', 'geometry']]
        
        app_state["buildings_gdf"] = gdf
        
        progress(0.8, desc="Exporting to GIS formats...")
        
        # Export
        output_dir = get_output_dir()
        base_name = f"{municipality or 'buildings'}_{datetime.now().strftime('%Y%m%d')}"
        
        # GeoPackage
        gpkg_path = output_dir / f"{base_name}.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG", layer="building_footprints")
        
        # Shapefile
        shp_dir = output_dir / "shapefile"
        shp_dir.mkdir(exist_ok=True)
        shp_path = shp_dir / f"{base_name}.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        
        # GeoJSON
        geojson_path = output_dir / f"{base_name}.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        
        # Create ZIP of all outputs
        progress(0.95, desc="Creating download package...")
        zip_path = output_dir / f"{base_name}_all_formats.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(gpkg_path, gpkg_path.name)
            zipf.write(geojson_path, geojson_path.name)
            for f in shp_dir.glob("*"):
                zipf.write(f, f"shapefile/{f.name}")
        
        progress(1.0, desc="Complete!")
        
        # Statistics
        total_area = gdf['area_sqft'].sum()
        mean_area = gdf['area_sqft'].mean()
        
        report = f"""✅ **Export Complete**

## Summary
| Property | Value |
|----------|-------|
| **Municipality** | {municipality or 'Not specified'} |
| **Buildings Extracted** | {len(gdf):,} |
| **Filtered (too small)** | {filtered_count:,} |
| **Total Area** | {total_area:,.0f} sq ft |
| **Average Size** | {mean_area:,.0f} sq ft |
| **CRS** | {target_crs} |

## Exported Files
- 📦 **GeoPackage:** {gpkg_path.name}
- 📁 **Shapefile:** {shp_path.name}
- 🌐 **GeoJSON:** {geojson_path.name}
- 🗜️ **All Formats (ZIP):** {zip_path.name}

## Area Statistics
| Metric | Value (sq ft) |
|--------|---------------|
| Minimum | {gdf['area_sqft'].min():,.0f} |
| Maximum | {gdf['area_sqft'].max():,.0f} |
| Median | {gdf['area_sqft'].median():,.0f} |

**Download the ZIP file below to get all formats.**
"""
        return report, str(zip_path)
    
    except Exception as e:
        import traceback
        return f"❌ Export failed:\n\n{str(e)}\n\n{traceback.format_exc()}", None


def load_demo_image(progress=gr.Progress()) -> Tuple[str, str, Optional[np.ndarray]]:
    """Load demo imagery for testing."""
    try:
        progress(0.3, desc="Downloading demo imagery...")
        url = "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/uc_berkeley.tif"
        image_path = download_file(url)
        
        progress(0.7, desc="Inspecting image...")
        report, preview = inspect_raster(image_path)
        
        progress(1.0, desc="Complete!")
        return image_path, report, preview
    
    except Exception as e:
        return "", f"❌ Failed to load demo: {str(e)}", None


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_app():
    """Create the Gradio application."""
    
    with gr.Blocks(
        title="MGP Building Segmentation Pipeline",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .header-text { text-align: center; }
        .step-header { 
            background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .info-box {
            background: #f0f7ff;
            border-left: 4px solid #2d5a87;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        """
    ) as app:
        
        # Header
        gr.Markdown("""
        # 🏢 MGP Building Segmentation Pipeline
        ### Municipal GIS Partners • Spark Initiative
        
        Extract building footprints from orthoimagery using AI-powered segmentation.
        All outputs are projected to Illinois State Plane for direct GIS integration.
        
        ---
        """, elem_classes="header-text")
        
        # State variables
        current_image_path = gr.State(value="")
        current_mask_path = gr.State(value="")
        tile_paths_state = gr.State(value=[])
        
        # =================================================================
        # STEP 1: Initialize
        # =================================================================
        with gr.Accordion("📌 Step 1: Initialize SAM3 Model", open=True):
            gr.Markdown("""
            <div class="info-box">
            <strong>First Time Setup:</strong> You need Hugging Face access to SAM3.
            <ol>
            <li>Request access at <a href="https://huggingface.co/facebook/sam3" target="_blank">huggingface.co/facebook/sam3</a></li>
            <li>Run <code>huggingface-cli login</code> in your terminal</li>
            <li>Click Initialize below</li>
            </ol>
            </div>
            """)
            
            with gr.Row():
                init_btn = gr.Button("🚀 Initialize SAM3", variant="primary", scale=2)
                demo_btn = gr.Button("📥 Load Demo Image", variant="secondary", scale=1)
            
            init_status = gr.Markdown("*Click Initialize to load the AI model...*")
        
        # =================================================================
        # STEP 2: Upload & Inspect
        # =================================================================
        with gr.Accordion("📁 Step 2: Upload & Inspect Imagery", open=True):
            gr.Markdown("""
            Upload your orthoimagery (GeoTIFF format). The tool will display metadata
            and check coordinate system compatibility.
            
            **Recommended sources:** Cook County orthoimagery, NAIP, municipal aerial flights
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload GeoTIFF",
                        file_types=[".tif", ".tiff"],
                        type="filepath"
                    )
                    
                with gr.Column(scale=1):
                    image_preview = gr.Image(
                        label="Preview",
                        type="numpy",
                        interactive=False
                    )
            
            metadata_display = gr.Markdown("*Upload an image to see metadata...*")
        
        # =================================================================
        # STEP 3: CRS & Tiling
        # =================================================================
        with gr.Accordion("🗺️ Step 3: Coordinate System & Tiling", open=False):
            gr.Markdown("""
            **Coordinate Reference System:** All MGP deliverables use Illinois State Plane East (EPSG:6455).
            
            **Tiling:** Large images should be split into tiles for better detection of small buildings.
            SAM3 works best with 2048×2048 pixel tiles.
            """)
            
            with gr.Row():
                with gr.Column():
                    crs_dropdown = gr.Dropdown(
                        choices=list(CRS_OPTIONS.keys()),
                        value="EPSG:6455 - Illinois East (ftUS)",
                        label="Target Coordinate System"
                    )
                    reproject_btn = gr.Button("🔄 Reproject Image", variant="secondary")
                    reproject_status = gr.Markdown("")
                
                with gr.Column():
                    tile_dropdown = gr.Dropdown(
                        choices=list(TILE_SIZE_OPTIONS.keys()),
                        value="2048 × 2048 (Recommended for residential)",
                        label="Tile Size"
                    )
                    tile_btn = gr.Button("✂️ Create Tiles", variant="secondary")
                    tile_status = gr.Markdown("")
        
        # =================================================================
        # STEP 4: Segmentation
        # =================================================================
        with gr.Accordion("🎯 Step 4: Run Segmentation", open=False):
            gr.Markdown("""
            Enter a text prompt to guide the AI. Common prompts:
            - **"building"** - General building detection
            - **"house"** - Residential structures
            - **"commercial building"** - Larger structures
            - **"garage"** - Accessory structures
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_prompt = gr.Textbox(
                        label="Segmentation Prompt",
                        value="building",
                        placeholder="Enter prompt (e.g., 'building', 'house')"
                    )
                    
                    min_area_slider = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        value=100,
                        step=10,
                        label="Minimum Building Area (sq ft)",
                        info="Filter out small artifacts"
                    )
                    
                    segment_btn = gr.Button("🎯 Run Segmentation", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    segment_preview = gr.Image(
                        label="Segmentation Result",
                        type="numpy",
                        interactive=False
                    )
            
            segment_status = gr.Markdown("*Configure options and click Run Segmentation...*")
        
        # =================================================================
        # STEP 5: Export
        # =================================================================
        with gr.Accordion("📤 Step 5: Export to GIS Formats", open=False):
            gr.Markdown("""
            Convert segmentation masks to vector polygons and export to standard GIS formats.
            
            **Output formats:**
            - GeoPackage (.gpkg) - Recommended for most GIS software
            - Shapefile (.shp) - Legacy format compatibility  
            - GeoJSON (.geojson) - Web mapping applications
            """)
            
            with gr.Row():
                municipality_input = gr.Textbox(
                    label="Municipality Name",
                    placeholder="e.g., Lake Forest, Northbrook",
                    info="Added to output attributes"
                )
                
                export_crs = gr.Dropdown(
                    choices=list(CRS_OPTIONS.keys()),
                    value="EPSG:6455 - Illinois East (ftUS)",
                    label="Export CRS"
                )
            
            export_btn = gr.Button("📦 Vectorize & Export", variant="primary", size="lg")
            
            export_status = gr.Markdown("*Run segmentation first, then export...*")
            
            download_file_output = gr.File(
                label="📥 Download Results",
                visible=True
            )
        
        # =================================================================
        # Help & Reference
        # =================================================================
        with gr.Accordion("❓ Help & Reference", open=False):
            gr.Markdown("""
            ## Quick Start Guide
            
            1. **Initialize** the SAM3 model (one-time setup)
            2. **Upload** your orthoimagery (GeoTIFF with georeferencing)
            3. **Check CRS** - Reproject to Illinois State Plane if needed
            4. **Tile** large images for better small building detection
            5. **Segment** using text prompts
            6. **Export** to your preferred GIS format
            
            ## Imagery Sources
            
            | Source | Resolution | Coverage |
            |--------|------------|----------|
            | [Cook County](https://hub-cookcountyil.opendata.arcgis.com/) | 6 inch | Cook County |
            | NAIP | 1 meter | Statewide |
            | Municipal flights | Varies | Individual communities |
            
            ## Coordinate Systems
            
            | EPSG | Name | Use Case |
            |------|------|----------|
            | 6455 | Illinois East (ftUS) | Cook County, Lake County, etc. |
            | 6456 | Illinois West (ftUS) | Western Illinois |
            | 3435 | Illinois East (Legacy) | Older datasets |
            
            ## Troubleshooting
            
            **"SAM3 not initialized"** - Click Initialize in Step 1
            
            **"No CRS defined"** - Your image lacks georeferencing. Use ArcGIS/QGIS to define CRS.
            
            **Missing small buildings** - Use smaller tile size (1024×1024)
            
            **Memory errors** - Reduce tile size or process on machine with more RAM/GPU
            
            ## Resources
            
            - [segment-geospatial docs](https://samgeo.gishub.org/)
            - [Cook County GIS Portal](https://hub-cookcountyil.opendata.arcgis.com/)
            - [EPSG:6455 Reference](https://epsg.io/6455)
            """)
        
        # =================================================================
        # Footer
        # =================================================================
        gr.Markdown("""
        ---
        *MGP Building Segmentation Pipeline v1.0 • Municipal GIS Partners • Spark Initiative*
        """)
        
        # =================================================================
        # Event Handlers
        # =================================================================
        
        init_btn.click(
            fn=initialize_sam3,
            outputs=init_status
        )
        
        demo_btn.click(
            fn=load_demo_image,
            outputs=[current_image_path, metadata_display, image_preview]
        )
        
        file_upload.change(
            fn=inspect_raster,
            inputs=file_upload,
            outputs=[metadata_display, image_preview]
        ).then(
            fn=lambda x: x,
            inputs=file_upload,
            outputs=current_image_path
        )
        
        reproject_btn.click(
            fn=reproject_raster,
            inputs=[current_image_path, crs_dropdown],
            outputs=[reproject_status, current_image_path]
        )
        
        tile_btn.click(
            fn=tile_raster,
            inputs=[current_image_path, tile_dropdown],
            outputs=[tile_status, tile_paths_state]
        )
        
        segment_btn.click(
            fn=run_segmentation,
            inputs=[current_image_path, text_prompt, min_area_slider],
            outputs=[segment_status, segment_preview, current_mask_path]
        )
        
        export_btn.click(
            fn=vectorize_and_export,
            inputs=[current_mask_path, municipality_input, min_area_slider, export_crs],
            outputs=[export_status, download_file_output]
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         MGP Building Segmentation Pipeline                     ║
    ║         Municipal GIS Partners • Spark Initiative              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set True to create public link
        inbrowser=True
    )
