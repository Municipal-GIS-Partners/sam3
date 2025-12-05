# MGP Building Segmentation Pipeline

**Municipal GIS Partners • Spark Initiative**

A user-friendly application for extracting building footprints from orthoimagery using AI-powered segmentation (SAM3). Designed for GIS analysts and non-technical staff alike.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv mgp-segmentation
source mgp-segmentation/bin/activate  # On Windows: mgp-segmentation\Scripts\activate

# Install packages
pip install gradio rasterio geopandas pyproj leafmap
pip install "segment-geospatial[samgeo3]"
pip install git+https://github.com/huggingface/transformers.git
```

### 2. Get SAM3 Access (One-Time Setup)

1. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Click **"Request Access"** and fill out the form
3. Wait for approval (usually < 24 hours)
4. Login via terminal:
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted.

### 3. Run the App

```bash
python app.py
```

The app will open in your browser at `http://localhost:7860`

---

## 📖 User Guide

### Step 1: Initialize SAM3

Click the **Initialize SAM3** button. This loads the AI model and may take a minute on first run (downloads ~2GB of model weights).

> 💡 **Tip:** Click "Load Demo Image" to test with sample imagery before using your own data.

### Step 2: Upload Imagery

Upload your orthoimagery in **GeoTIFF format**. The app will display:
- Image dimensions and resolution
- Coordinate reference system (CRS)
- Recommendations for tiling

**Supported imagery:**
- Cook County orthophotos (6-inch resolution)
- NAIP imagery (1-meter resolution)
- Any georeferenced GeoTIFF

### Step 3: Configure CRS & Tiling

**Coordinate System:**
- Select **EPSG:6455 - Illinois East** for Cook County and northeastern Illinois
- Select **EPSG:6456 - Illinois West** for western Illinois communities

**Tiling:**
- **2048×2048** - Best for residential areas (recommended)
- **1024×1024** - Better for detecting small structures (garages, sheds)
- **4096×4096** - Large commercial/industrial buildings

### Step 4: Run Segmentation

Enter a text prompt to guide detection:

| Prompt | Best For |
|--------|----------|
| `building` | General building detection |
| `house` | Residential structures |
| `commercial building` | Larger structures |
| `garage` | Accessory structures |

Click **Run Segmentation** and wait for results.

### Step 5: Export to GIS

1. Enter the **municipality name** (added to output attributes)
2. Confirm **export CRS** (Illinois State Plane recommended)
3. Click **Vectorize & Export**
4. **Download** the ZIP file containing all formats

**Output formats:**
- **GeoPackage (.gpkg)** - Recommended, works in ArcGIS Pro and QGIS
- **Shapefile (.shp)** - Legacy format for older systems
- **GeoJSON (.geojson)** - Web mapping applications

---

## 📊 Output Attributes

| Field | Description |
|-------|-------------|
| `building_id` | Unique identifier |
| `municipality` | Community name |
| `area_sqft` | Building footprint area in square feet |
| `source` | Processing source (SAM3_Segmentation) |
| `proc_date` | Processing date |

---

## 🗺️ Coordinate Reference Systems

For MGP communities, use these standard CRS options:

| Communities | CRS | EPSG |
|-------------|-----|------|
| Cook, Lake, DuPage, Kane, McHenry, Will, Kendall, Grundy, Kankakee | Illinois East | 6455 |
| Western Illinois communities | Illinois West | 6456 |

---

## ⚠️ Troubleshooting

### "SAM3 not initialized"
Click the Initialize button in Step 1.

### "No CRS defined"
Your image lacks georeferencing. Open in ArcGIS Pro or QGIS to define the coordinate system before processing.

### Small buildings not detected
- Use smaller tile size (1024×1024)
- Try more specific prompts like "small house" or "garage"

### Out of memory
- Reduce tile size
- Close other applications
- Process on a machine with more RAM or a GPU

### Hugging Face authentication failed
1. Verify you have SAM3 access approved
2. Re-run `huggingface-cli login`
3. Check your token hasn't expired

---

## 💻 System Requirements

**Minimum:**
- Python 3.9+
- 16 GB RAM
- 10 GB disk space

**Recommended:**
- NVIDIA GPU with 8+ GB VRAM (CUDA)
- 32 GB RAM
- SSD storage

---

## 📁 File Structure

```
mgp-segmentation/
├── mgp_building_segmentation_app.py   # Main application
├── mgp_building_segmentation.ipynb    # Jupyter notebook version
├── README.md                          # This file
└── outputs/                           # Generated outputs (auto-created)
    ├── tiles/                         # Tiled imagery
    ├── masks.tif                      # Segmentation masks
    └── [municipality]_buildings_*.gpkg # Vector outputs
```

---

## 🔗 Resources

- [Cook County GIS Portal](https://hub-cookcountyil.opendata.arcgis.com/)
- [segment-geospatial Documentation](https://samgeo.gishub.org/)
- [SAM3 on Hugging Face](https://huggingface.co/facebook/sam3)
- [EPSG:6455 Reference](https://epsg.io/6455)

---

## 📧 Support

For questions or issues, contact the Spark team or open a ticket in the internal helpdesk.

---

*MGP Building Segmentation Pipeline v1.0*  
*Municipal GIS Partners • Spark Initiative*