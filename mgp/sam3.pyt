# =============================================================================
# ⚙️  ENVIRONMENT SETUP – READ BEFORE RUNNING SAM3 IN ARCGIS PRO
# =============================================================================
#
# This toolbox requires a custom ArcGIS Pro Python environment with SAM3
# (segment-geospatial) and HuggingFace support installed.
#
# ⚠️ IMPORTANT:
# Do NOT install these packages into the default "arcgispro-py3" environment.
# Always CLONE the environment first.
#
# ---------------------------------------------------------------------------
# STEP 1 — Identify the active ArcGIS Pro environment
# ---------------------------------------------------------------------------
# In ArcGIS Pro:
#   Project → Settings → Python
#   Note the path shown under "Current Environment"
#
# Example:
#   C:\Users\<you>\AppData\Local\ESRI\conda\envs\arcgispro-py3
#
# ---------------------------------------------------------------------------
# STEP 2 — Clone the environment (strongly recommended)
# ---------------------------------------------------------------------------
# In the same Python settings page:
#   1. Click "Manage Environments"
#   2. Select "arcgispro-py3"
#   3. Click "Clone"
#   4. Name the clone:
#        arcgispro-py3-sam3
#   5. Set the cloned env as the ACTIVE environment
#
# ArcGIS Pro will restart using the cloned environment.
#
# ---------------------------------------------------------------------------
# STEP 3 — Open the correct Python Command Prompt from Pro
# ---------------------------------------------------------------------------
# In ArcGIS Pro:
#   Project → Settings → Python → Open Python Command Prompt
#
# ⚠️ This is the ONLY safe way to ensure you install packages into the same
# environment ArcGIS Pro is using. Do NOT use Anaconda Prompt or system CMD.
#
# Verify with:
#   where python
#
# It should point to:
#   ...\ESRI\conda\envs\arcgispro-py3-sam3\python.exe
#
# ---------------------------------------------------------------------------
# STEP 4 — Install required packages into the Pro environment
# ---------------------------------------------------------------------------
#
# Run the following EXACTLY from the Python prompt opened from Pro:
#
#   pip install segment-geospatial[samgeo3]
#   pip install huggingface_hub
#   pip install rasterio geopandas pyproj
#   pip install git+https://github.com/huggingface/transformers.git
#
# ---------------------------------------------------------------------------
# STEP 5 — Verify installation inside ArcGIS Pro
# ---------------------------------------------------------------------------
# In ArcGIS Pro:
#   View → Python
#
# Run:
#   from samgeo import SamGeo3
#   from huggingface_hub import login
#
# If no errors occur, the environment is configured correctly.
#
# ---------------------------------------------------------------------------
# STEP 6 — Prepare your HuggingFace access token
# ---------------------------------------------------------------------------
# 1. Go to: https://huggingface.co/settings/tokens
# 2. Create a new Access Token with "Read" permissions
# 3. Paste this token into the "HuggingFace Access Token" parameter
#    when running the SAM3 tool in ArcGIS Pro
#
# The token is:
#   • Used only for this session
#   • Not written to disk
#   • Not stored in the Pro project
#
# ---------------------------------------------------------------------------
# COMMON FAILURE CAUSES
# ---------------------------------------------------------------------------
# • "ModuleNotFoundError: samgeo"  → Wrong Python environment
# • SAM3 initializes but fails     → HuggingFace token missing or invalid
# • Pro crashes on run             → Torch/GPU mismatch (use CPU first)
#
# ---------------------------------------------------------------------------
# FIRST-RUN NOTE
# ---------------------------------------------------------------------------
# The first time SAM3 runs it will download model weights (~GBs).
# This may take several minutes depending on network speed.
#
# =============================================================================

import os
import tempfile
from pathlib import Path

import arcpy

# Hugging Face login helper
try:
    from huggingface_hub import login as hf_login
except ImportError:
    hf_login = None

# SamGeo3 (segment-geospatial)
try:
    from samgeo import SamGeo3
except ImportError:
    SamGeo3 = None


# ----------------------------------------------------------------------
# Toolbox definition
# ----------------------------------------------------------------------
class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the .pyt is the toolbox name)."""
        self.label = "SAM3 Tools"
        self.alias = "sam3tools"
        self.tools = [Sam3BuildingSegmentation]


# ----------------------------------------------------------------------
# Tool definition
# ----------------------------------------------------------------------
class Sam3BuildingSegmentation(object):
    """
    SAM3 Building Segmentation

    Uses Segment Anything Model 3 (SamGeo3) to segment features from aerial imagery
    based on a natural-language text prompt (e.g., "building", "house", "garage"),
    then converts the mask to polygons, filters by minimum area, and writes a
    feature class.

    Typical use:
        - Input raster: projected orthoimagery (State Plane, UTM, etc.)
        - Prompt: "building"
        - Min area: 100 sq ft (or as needed)
        - Output: feature class in a file geodatabase or shapefile
    """

    def __init__(self):
        self.label = "SAM3 Building Segmentation"
        self.description = (
            "Extract features (e.g., buildings) from orthoimagery using SAM3 text prompts, "
            "then convert to polygons and filter by minimum area."
        )
        self.canRunInBackground = True

    # ------------------------------------------------------------------
    def getParameterInfo(self):
        """Define parameter definitions."""

        # 0 - Input raster
        in_raster = arcpy.Parameter(
            displayName="Input Raster",
            name="in_raster",
            datatype="DERasterDataset",
            parameterType="Required",
            direction="Input"
        )

        # 1 - Text prompt
        prompt = arcpy.Parameter(
            displayName="Text Prompt",
            name="prompt",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        prompt.value = "building"

        # 2 - Minimum area (sq ft)
        min_area = arcpy.Parameter(
            displayName="Minimum Area (square feet)",
            name="min_area_sqft",
            datatype="GPDouble",
            parameterType="Required",
            direction="Input"
        )
        min_area.value = 100.0

        # 3 - Enable confidence filter
        use_conf = arcpy.Parameter(
            displayName="Enable Confidence Filter (pred_iou_thresh)",
            name="use_confidence_filter",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        use_conf.value = False

        # 4 - Confidence threshold
        conf_thresh = arcpy.Parameter(
            displayName="Confidence Threshold (0–1, higher = stricter)",
            name="confidence_threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        conf_thresh.value = 0.88

        # 5 - Enable stability filter
        use_stab = arcpy.Parameter(
            displayName="Enable Stability Filter (stability_score_thresh)",
            name="use_stability_filter",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        use_stab.value = False

        # 6 - Stability threshold
        stab_thresh = arcpy.Parameter(
            displayName="Stability Threshold (0–1, higher = stricter)",
            name="stability_threshold",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )
        stab_thresh.value = 0.95

        # 7 - Dissolve polygons
        dissolve_polys = arcpy.Parameter(
            displayName="Dissolve Output Polygons",
            name="dissolve_polygons",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input"
        )
        dissolve_polys.value = True

        # 8 - HuggingFace token
        hf_token = arcpy.Parameter(
            displayName="HuggingFace Access Token",
            name="hf_token",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        hf_token.value = ""
        hf_token.category = "Authentication"

        # 9 - Output feature class
        out_fc = arcpy.Parameter(
            displayName="Output Feature Class",
            name="out_feature_class",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

        return [
            in_raster,
            prompt,
            min_area,
            use_conf,
            conf_thresh,
            use_stab,
            stab_thresh,
            dissolve_polys,
            hf_token,
            out_fc
        ]

    # ------------------------------------------------------------------
    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    # ------------------------------------------------------------------
    def updateParameters(self, parameters):
        """Modify parameter defaults/visibility before validation."""
        use_conf = parameters[3]
        conf_thresh = parameters[4]
        use_stab = parameters[5]
        stab_thresh = parameters[6]

        # Toggle confidence threshold enabled/disabled
        conf_thresh.enabled = bool(use_conf.value)

        # Toggle stability threshold enabled/disabled
        stab_thresh.enabled = bool(use_stab.value)

        return

    # ------------------------------------------------------------------
    def updateMessages(self, parameters):
        """Modify messages created by internal validation."""
        if SamGeo3 is None:
            msg = (
                "The 'segment-geospatial' package with SAM3 support is not installed in this "
                "Python environment. Install with:\n"
                "    pip install segment-geospatial[samgeo3]\n"
                "and ensure ArcGIS Pro is using that environment."
            )
            for p in parameters:
                if p:
                    p.setErrorMessage(msg)

        if hf_login is None:
            msg = (
                "The 'huggingface_hub' package is not installed. Install with:\n"
                "    pip install huggingface_hub\n"
                "in the same environment ArcGIS Pro is using."
            )
            for p in parameters:
                if p:
                    p.setErrorMessage(msg)

        return

    # ------------------------------------------------------------------
    def execute(self, parameters, messages):
        """Main execution: run SAM3, raster→polygon, area filter, output FC."""

        # ------------------------------------------------------------------
        # 0. Sanity checks and parameter unpacking
        # ------------------------------------------------------------------
        if SamGeo3 is None:
            raise arcpy.ExecuteError(
                "SamGeo3 is not available. Install 'segment-geospatial[samgeo3]' "
                "in the ArcGIS Pro Python environment."
            )

        if hf_login is None:
            raise arcpy.ExecuteError(
                "huggingface_hub is not available. Install 'huggingface_hub' "
                "in the ArcGIS Pro Python environment."
            )

        in_raster = parameters[0].valueAsText
        prompt = parameters[1].valueAsText
        min_area_sqft = float(parameters[2].value)

        use_conf = bool(parameters[3].value)
        conf_thresh = float(parameters[4].value) if parameters[4].value is not None else 0.88

        use_stab = bool(parameters[5].value)
        stab_thresh = float(parameters[6].value) if parameters[6].value is not None else 0.95

        dissolve_polys = bool(parameters[7].value)

        hf_token = parameters[8].valueAsText
        out_fc = parameters[9].valueAsText

        if not prompt or not prompt.strip():
            raise arcpy.ExecuteError("Text prompt is required (e.g., 'building', 'house').")

        prompt = prompt.strip()

        if not hf_token or not hf_token.strip():
            raise arcpy.ExecuteError(
                "HuggingFace access token is required to download/use SAM3 checkpoints."
            )

        arcpy.AddMessage("============================================")
        arcpy.AddMessage("   SAM3 Building Segmentation (SAM3Tools)   ")
        arcpy.AddMessage("============================================")
        arcpy.AddMessage(f"Input raster: {in_raster}")
        arcpy.AddMessage(f"Prompt: '{prompt}'")
        arcpy.AddMessage(f"Minimum area: {min_area_sqft:.2f} sq ft")
        arcpy.AddMessage(f"Use confidence filter: {use_conf} (threshold={conf_thresh:.2f})")
        arcpy.AddMessage(f"Use stability filter: {use_stab} (threshold={stab_thresh:.2f})")
        arcpy.AddMessage(f"Dissolve polygons: {dissolve_polys}")
        arcpy.AddMessage(f"Output FC: {out_fc}")

        scratch_gdb = arcpy.env.scratchGDB
        scratch_folder = arcpy.env.scratchFolder or tempfile.gettempdir()

        arcpy.AddMessage(f"Scratch GDB: {scratch_gdb}")
        arcpy.AddMessage(f"Scratch folder: {scratch_folder}")

        # ------------------------------------------------------------------
        # 1. Authenticate with HuggingFace using provided token
        # ------------------------------------------------------------------
        try:
            arcpy.AddMessage("Authenticating with HuggingFace...")
            hf_login(token=hf_token.strip(), add_to_git_credential=False)
            os.environ["HF_TOKEN"] = hf_token.strip()
            arcpy.AddMessage("HuggingFace authentication successful.")
        except Exception as e:
            raise arcpy.ExecuteError(
                f"HuggingFace login failed. Verify your token.\n\n{e}"
            )

        # ------------------------------------------------------------------
        # 2. Run SAM3 segmentation and save mask raster
        # ------------------------------------------------------------------
        try:
            arcpy.AddMessage("Initializing SAM3 (SamGeo3)...")
            sam3 = SamGeo3(
                backend="meta",
                device=None,           # Let SamGeo3 decide CPU/GPU
                checkpoint_path=None,  # Use default HF checkpoint
                load_from_HF=True
            )
        except Exception as e:
            raise arcpy.ExecuteError(
                "Failed to initialize SAM3. Ensure you have model access on Hugging Face.\n\n{}".format(e)
            )

        try:
            arcpy.AddMessage("Setting SAM3 input image...")
            sam3.set_image(in_raster)
        except Exception as e:
            raise arcpy.ExecuteError(
                "Failed to load input raster into SAM3. Check the raster path and format.\n\n{}".format(e)
            )

        arcpy.AddMessage("Running SAM3 mask generation...")
        gen_kwargs = {"prompt": prompt}
        if use_conf:
            gen_kwargs["pred_iou_thresh"] = conf_thresh
        if use_stab:
            gen_kwargs["stability_score_thresh"] = stab_thresh

        try:
            sam3.generate_masks(**gen_kwargs)
        except Exception as e:
            raise arcpy.ExecuteError(
                "SAM3 mask generation failed. Check prompt and model configuration.\n\n{}".format(e)
            )

        mask_name = "sam3_mask.tif"
        mask_path = os.path.join(scratch_folder, mask_name)

        try:
            arcpy.AddMessage(f"Saving SAM3 mask raster to: {mask_path}")
            sam3.save_masks(mask_path)
        except Exception as e:
            raise arcpy.ExecuteError(
                "Failed to save SAM3 mask raster.\n\n{}".format(e)
            )

        # ------------------------------------------------------------------
        # 3. Convert mask to polygons (RasterToPolygon)
        # ------------------------------------------------------------------
        arcpy.AddMessage("Converting mask raster to polygons...")

        mask_polygon_fc = os.path.join(scratch_gdb, "sam3_mask_polygons")
        if arcpy.Exists(mask_polygon_fc):
            arcpy.management.Delete(mask_polygon_fc)

        try:
            arcpy.conversion.RasterToPolygon(
                in_raster=mask_path,
                out_polygon_features=mask_polygon_fc,
                simplify="NO_SIMPLIFY",
                raster_field="VALUE"
            )
        except Exception as e:
            raise arcpy.ExecuteError(
                "RasterToPolygon failed. Ensure the mask raster is valid.\n\n{}".format(e)
            )

        # ------------------------------------------------------------------
        # 4. Filter polygons where mask value > 0 (actual objects)
        # ------------------------------------------------------------------
        arcpy.AddMessage("Filtering out background polygons (mask value = 0)...")

        fields = [f.name for f in arcpy.ListFields(mask_polygon_fc)]
        value_field = None
        for candidate in ("gridcode", "VALUE", "value"):
            if candidate in fields:
                value_field = candidate
                break

        if value_field is None:
            raise arcpy.ExecuteError(
                "Could not find a suitable value field on the polygon feature class "
                "(e.g., 'gridcode' or 'VALUE')."
            )

        lyr = "sam3_mask_layer"
        if arcpy.Exists(lyr):
            arcpy.management.Delete(lyr)

        arcpy.management.MakeFeatureLayer(mask_polygon_fc, lyr)
        arcpy.management.SelectLayerByAttribute(
            lyr,
            "NEW_SELECTION",
            f"{arcpy.AddFieldDelimiters(lyr, value_field)} > 0"
        )

        filtered_fc = os.path.join(scratch_gdb, "sam3_mask_filtered")
        if arcpy.Exists(filtered_fc):
            arcpy.management.Delete(filtered_fc)

        arcpy.management.CopyFeatures(lyr, filtered_fc)
        arcpy.AddMessage("Background polygons removed.")

        # ------------------------------------------------------------------
        # 5. Add area field (sq ft) and filter by minimum area
        # ------------------------------------------------------------------
        arcpy.AddMessage("Adding area field (square feet) and filtering by minimum area...")

        area_field = "area_sqft"
        if area_field in [f.name for f in arcpy.ListFields(filtered_fc)]:
            arcpy.management.DeleteField(filtered_fc, area_field)

        arcpy.management.AddField(
            in_table=filtered_fc,
            field_name=area_field,
            field_type="DOUBLE"
        )

        # Calculate area in US square feet
        arcpy.management.CalculateGeometryAttributes(
            in_features=filtered_fc,
            geometry_property=[[area_field, "AREA"]],
            length_unit="",
            area_unit="SQUARE_FEET_US"
        )

        small_removed_fc = os.path.join(scratch_gdb, "sam3_mask_area_filtered")
        if arcpy.Exists(small_removed_fc):
            arcpy.management.Delete(small_removed_fc)

        lyr_area = "sam3_area_layer"
        if arcpy.Exists(lyr_area):
            arcpy.management.Delete(lyr_area)

        arcpy.management.MakeFeatureLayer(filtered_fc, lyr_area)
        arcpy.management.SelectLayerByAttribute(
            lyr_area,
            "NEW_SELECTION",
            f"{arcpy.AddFieldDelimiters(lyr_area, area_field)} >= {min_area_sqft}"
        )

        arcpy.management.CopyFeatures(lyr_area, small_removed_fc)

        # ------------------------------------------------------------------
        # 6. Optional dissolve
        # ------------------------------------------------------------------
        final_fc = small_removed_fc

        if dissolve_polys:
            arcpy.AddMessage("Dissolving polygons into multipart features...")
            dissolved_fc = os.path.join(scratch_gdb, "sam3_mask_dissolved")
            if arcpy.Exists(dissolved_fc):
                arcpy.management.Delete(dissolved_fc)

            arcpy.management.Dissolve(
                in_features=small_removed_fc,
                out_feature_class=dissolved_fc,
                dissolve_field="",
                multi_part="MULTI_PART",
                unsplit_lines="DISSOLVE_LINES"
            )
            final_fc = dissolved_fc

        # ------------------------------------------------------------------
        # 7. Copy final result to user-specified output feature class
        # ------------------------------------------------------------------
        arcpy.AddMessage(f"Writing final output to: {out_fc}")

        out_ws, out_name = os.path.split(out_fc)
        if not out_ws:
            raise arcpy.ExecuteError(
                "Output feature class must include a workspace path (e.g., a file geodatabase)."
            )

        if not arcpy.Exists(out_ws):
            raise arcpy.ExecuteError(
                f"Output workspace does not exist: {out_ws}"
            )

        if arcpy.Exists(out_fc):
            arcpy.management.Delete(out_fc)

        arcpy.conversion.FeatureClassToFeatureClass(
            in_features=final_fc,
            out_path=out_ws,
            out_name=out_name
        )

        # Set output parameter so Pro knows the result
        parameters[9].value = out_fc

        # ------------------------------------------------------------------
        # 8. Summary
        # ------------------------------------------------------------------
        count_result = int(arcpy.management.GetCount(out_fc).getOutput(0))
        arcpy.AddMessage("--------------------------------------------")
        arcpy.AddMessage("SAM3 Building Segmentation completed.")
        arcpy.AddMessage(f"Output features: {count_result}")
        arcpy.AddMessage(f"Output: {out_fc}")
        arcpy.AddMessage("--------------------------------------------")
