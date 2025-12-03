"""
As-Built Autopilot - Interactive utility as-built extraction using SAM3
Author: John Kenny, Municipal GIS Partners

Features:
- Interactive georeferencing in Jupyter notebook
- AI-powered segmentation with SAM3
- Vector extraction (points, lines, polygons)
- Direct export to geodatabase/shapefile
"""

import os
import geopandas as gpd
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point, LineString, Polygon
import numpy as np
from pathlib import Path
import leafmap
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import tempfile
import warnings
warnings.filterwarnings('ignore')


class AsBuiltAutopilot:
    """
    Automated extraction of utility features from as-built documents.
    
    Complete workflow:
    1. Interactive georeferencing of raw as-built scans
    2. AI-powered feature segmentation with SAM3
    3. Vector conversion (points, lines, polygons)
    4. Export to geodatabase/shapefile for ArcGIS Pro
    """
    
    def __init__(self, sam3_model, target_crs="EPSG:3435"):
        """
        Initialize the As-Built Autopilot.
        
        Parameters:
        -----------
        sam3_model : SamGeo3
            Initialized SAM3 model
        target_crs : str
            Target coordinate reference system 
            Default: Illinois State Plane East NAD83 (feet) - EPSG:3435
        """
        self.sam3 = sam3_model
        self.target_crs = target_crs
        
        # Image state
        self.raw_image_path = None
        self.georef_image_path = None
        self.image_crs = None
        self.image_bounds = None
        self.transform = None
        
        # Georeferencing state
        self.gcps = []
        self.temp_dir = tempfile.mkdtemp()
        
        # Segmentation state
        self.masks = None
        self.features_gdf = None
        
        # UI state
        self.map = None
        self.georef_map = None
        
        print(f"✓ As-Built Autopilot initialized")
        print(f"  Target CRS: {self.target_crs}")
    
    def start_georeferencing(self, image_path, initial_center=None, initial_zoom=16):
        """
        Launch interactive georeferencing interface in Jupyter notebook.
        
        Workflow:
        1. Display raw as-built image on left
        2. Display basemap on right for selecting GCP locations
        3. User clicks matching points on both maps
        4. Calculate transformation from GCPs
        5. Generate georeferenced GeoTIFF
        
        Parameters:
        -----------
        image_path : str or Path
            Path to raw (non-georeferenced) as-built image (PNG, JPG, TIFF)
        initial_center : tuple, optional
            (lat, lon) for initial basemap center
        initial_zoom : int
            Initial zoom level for basemap
        """
        self.raw_image_path = Path(image_path)
        
        if not self.raw_image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Display instructions
        print("🗺️ Interactive Georeferencing Interface")
        print("=" * 70)
        print("Instructions:")
        print("1. Study both the as-built image and the basemap")
        print("2. Click 'Add GCP' button")
        print("3. Click a recognizable point on the as-built (left)")
        print("4. Click the same point on the basemap (right)")
        print("5. Repeat for at least 4 control points (more is better!)")
        print("6. Click 'Generate Georeferenced Image' when done")
        print("=" * 70)
        print("\nTips:")
        print("- Use building corners, curb intersections, manhole covers")
        print("- Distribute GCPs across the entire document")
        print("- 6-8 GCPs gives better accuracy than 4")
        print("=" * 70)
        
        # Initialize GCP tracking
        self.gcps = []
        self.current_image_point = None
        
        # Create dual map interface
        self._create_georeferencing_ui(initial_center, initial_zoom)
    
    def _create_georeferencing_ui(self, center, zoom):
        """Create the dual-map georeferencing interface."""
        from PIL import Image
        import io
        
        # Load and display image
        img = Image.open(self.raw_image_path)
        
        # Convert to bytes for widget
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Store image dimensions
        self.image_width, self.image_height = img.size
        
        # Left side: Raw image viewer with click tracking
        image_html = f'''
        <div style="border: 2px solid #4CAF50; padding: 10px; background: #f5f5f5;">
            <h3 style="margin-top: 0;">As-Built Document</h3>
            <p><i>Click control points on this image</i></p>
            <img id="asbuilt_img" src="data:image/png;base64,{self._image_to_base64(self.raw_image_path)}" 
                 style="max-width: 100%; cursor: crosshair; border: 1px solid #ddd;"
                 onclick="handleImageClick(event)">
            <div id="img_coords" style="margin-top: 10px; font-family: monospace;">
                Click on the image to set image coordinates
            </div>
        </div>
        <script>
        function handleImageClick(event) {{
            var img = document.getElementById('asbuilt_img');
            var rect = img.getBoundingClientRect();
            var x = event.clientX - rect.left;
            var y = event.clientY - rect.top;
            
            // Scale to actual image dimensions
            var scaleX = {self.image_width} / img.width;
            var scaleY = {self.image_height} / img.height;
            var actualX = Math.round(x * scaleX);
            var actualY = Math.round(y * scaleY);
            
            document.getElementById('img_coords').innerHTML = 
                'Image coordinates: X=' + actualX + ', Y=' + actualY + ' (click map next)';
            
            // Store in Python via Jupyter comms
            var kernel = IPython.notebook.kernel;
            kernel.execute('autopilot.current_image_point = (' + actualX + ', ' + actualY + ')');
        }}
        </script>
        '''
        
        image_widget = widgets.HTML(value=image_html)
        
        # Right side: Interactive basemap
        if center:
            self.georef_map = leafmap.Map(center=center, zoom=zoom, height="500px")
        else:
            # Default to Chicago area
            self.georef_map = leafmap.Map(center=[41.8781, -87.6298], zoom=zoom, height="500px")
        
        # Add basemap options
        self.georef_map.add_basemap("Esri.WorldImagery")
        
        # GCP controls
        add_gcp_btn = widgets.Button(
            description='📍 Add GCP',
            button_style='success',
            tooltip='Click to start adding a Ground Control Point',
            layout=widgets.Layout(width='150px')
        )
        
        generate_btn = widgets.Button(
            description='🌍 Generate Georeferenced Image',
            button_style='primary',
            disabled=True,
            layout=widgets.Layout(width='250px')
        )
        
        clear_gcps_btn = widgets.Button(
            description='🗑️ Clear All GCPs',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # GCP list display
        gcp_output = widgets.Output(layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            min_height='100px'
        ))
        
        with gcp_output:
            print("Ground Control Points: 0")
            print("Add at least 4 GCPs to continue")
        
        # Status output
        status_output = widgets.Output(layout=widgets.Layout(
            border='2px solid #2196F3',
            padding='10px',
            background_color='#E3F2FD'
        ))
        
        with status_output:
            print("💡 Ready to add ground control points")
        
        def on_add_gcp_click(b):
            """Initiate GCP collection."""
            with status_output:
                clear_output()
                print("👆 Step 1: Click on the as-built image (left)")
                print("👆 Step 2: Then click on the map (right) at the same location")
        
        def on_map_click(**kwargs):
            """Handle map clicks for GCP creation."""
            if 'coordinates' not in kwargs:
                return
            
            coords = kwargs['coordinates']
            lat, lon = coords
            
            # Check if image point was set
            if self.current_image_point is None:
                with status_output:
                    clear_output()
                    print("⚠️ Please click on the image first to set image coordinates")
                return
            
            # Create GCP
            img_x, img_y = self.current_image_point
            gcp = GroundControlPoint(
                row=float(img_y), 
                col=float(img_x), 
                x=lon,  # Longitude (X in geographic coords)
                y=lat,  # Latitude (Y in geographic coords)
                crs=rasterio.crs.CRS.from_epsg(4326)
            )
            
            self.gcps.append(gcp)
            self.current_image_point = None
            
            # Update GCP display
            with gcp_output:
                clear_output()
                print(f"Ground Control Points: {len(self.gcps)}")
                print("-" * 60)
                for i, g in enumerate(self.gcps, 1):
                    print(f"  {i}. Image({g.col:.0f}, {g.row:.0f}) → Map({g.y:.6f}, {g.x:.6f})")
                print("-" * 60)
                if len(self.gcps) >= 4:
                    print("✓ Minimum GCPs collected!")
                else:
                    print(f"Need {4 - len(self.gcps)} more GCP(s)")
            
            with status_output:
                clear_output()
                print(f"✓ GCP #{len(self.gcps)} added successfully!")
                if len(self.gcps) >= 4:
                    print("✓ Ready to generate georeferenced image!")
                    generate_btn.disabled = False
                else:
                    print(f"Add {4 - len(self.gcps)} more GCP(s) (minimum 4 required)")
            
            # Add marker to map
            self.georef_map.add_marker(
                location=[lat, lon], 
                popup=f"GCP {len(self.gcps)}<br>Image: ({img_x}, {img_y})"
            )
        
        def on_generate_click(b):
            """Generate the georeferenced GeoTIFF."""
            with status_output:
                clear_output()
                print("🔄 Calculating transformation from GCPs...")
                print("🔄 Generating georeferenced GeoTIFF...")
            
            try:
                self._generate_georeferenced_image()
                
                with status_output:
                    clear_output()
                    print("=" * 60)
                    print("✓ SUCCESS! Georeferenced image created")
                    print("=" * 60)
                    print(f"📄 Output: {self.georef_image_path}")
                    print(f"📐 CRS: {self.image_crs}")
                    print(f"📍 Bounds: {self.image_bounds}")
                    print("\n✓ Ready for segmentation!")
                    print("   Run: autopilot.create_segmentation_interface()")
                
            except Exception as e:
                with status_output:
                    clear_output()
                    print("❌ Error generating georeferenced image:")
                    print(f"   {str(e)}")
                    import traceback
                    print("\nFull traceback:")
                    print(traceback.format_exc())
        
        def on_clear_gcps_click(b):
            """Clear all GCPs and reset."""
            self.gcps = []
            self.current_image_point = None
            generate_btn.disabled = True
            
            with gcp_output:
                clear_output()
                print("Ground Control Points: 0")
                print("Add at least 4 GCPs to continue")
            
            with status_output:
                clear_output()
                print("🗑️ All GCPs cleared")
                print("💡 Ready to add new ground control points")
            
            # Note: Cannot easily remove markers from leafmap
            # Would need to track marker references
        
        # Connect event handlers
        add_gcp_btn.on_click(on_add_gcp_click)
        generate_btn.on_click(on_generate_click)
        clear_gcps_btn.on_click(on_clear_gcps_click)
        self.georef_map.on_interaction(on_map_click)
        
        # Layout
        buttons = widgets.HBox([
            add_gcp_btn, 
            generate_btn, 
            clear_gcps_btn
        ], layout=widgets.Layout(justify_content='space-around'))
        
        left_panel = widgets.VBox([
            image_widget
        ], layout=widgets.Layout(width='48%'))
        
        right_panel = widgets.VBox([
            widgets.HTML("<div style='border: 2px solid #2196F3; padding: 10px; background: #f5f5f5;'>"
                        "<h3 style='margin-top: 0;'>Basemap (Select GCP Locations)</h3>"
                        "<p><i>Click corresponding points on this map</i></p></div>"),
            self.georef_map
        ], layout=widgets.Layout(width='48%'))
        
        main_layout = widgets.HBox([
            left_panel, 
            right_panel
        ], layout=widgets.Layout(justify_content='space-between'))
        
        full_interface = widgets.VBox([
            buttons,
            main_layout,
            widgets.HTML("<h4>Ground Control Points:</h4>"),
            gcp_output,
            widgets.HTML("<h4>Status:</h4>"),
            status_output
        ])
        
        display(full_interface)
    
    def _image_to_base64(self, image_path):
        """Convert image to base64 for HTML embedding."""
        import base64
        from PIL import Image
        import io
        
        img = Image.open(image_path)
        # Resize if too large for browser
        max_size = 800
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode()
    
    def _generate_georeferenced_image(self):
        """Generate georeferenced GeoTIFF from GCPs."""
        if len(self.gcps) < 4:
            raise ValueError(f"Need at least 4 Ground Control Points (have {len(self.gcps)})")
        
        print(f"Using {len(self.gcps)} ground control points...")
        
        # Read raw image
        from PIL import Image
        img = Image.open(self.raw_image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Calculate transform from GCPs
        print("Calculating transformation...")
        transform = from_gcps(self.gcps)
        
        # Create output path
        self.georef_image_path = Path(self.temp_dir) / f"georef_{self.raw_image_path.stem}.tif"
        
        # Write georeferenced GeoTIFF
        print("Writing GeoTIFF...")
        
        # Transpose array from HWC to CHW format for rasterio
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, (2, 0, 1))
            count = img_array.shape[0]
        else:
            img_array = img_array[np.newaxis, :, :]
            count = 1
        
        with rasterio.open(
            self.georef_image_path,
            'w',
            driver='GTiff',
            height=img_array.shape[1],
            width=img_array.shape[2],
            count=count,
            dtype=img_array.dtype,
            crs=self.target_crs,
            transform=transform
        ) as dst:
            dst.write(img_array)
        
        # Update state
        self.image_crs = self.target_crs
        self.transform = transform
        
        with rasterio.open(self.georef_image_path) as src:
            self.image_bounds = src.bounds
        
        print(f"✓ Georeferenced image created: {self.georef_image_path}")
    
    def load_georeferenced_asbuilt(self, image_path, reproject_if_needed=True):
        """
        Load a pre-georeferenced as-built image.
        
        Use this if you already have a georeferenced GeoTIFF.
        
        Parameters:
        -----------
        image_path : str or Path
            Path to georeferenced GeoTIFF
        reproject_if_needed : bool
            If True, warn if CRS differs from target
            
        Returns:
        --------
        dict : Image metadata including CRS, bounds, transform
        """
        image_path = Path(image_path)
        self.georef_image_path = image_path
        
        with rasterio.open(image_path) as src:
            self.image_crs = src.crs.to_string() if src.crs else None
            self.image_bounds = src.bounds
            self.transform = src.transform
            
            # Get image metadata
            metadata = {
                'crs': self.image_crs,
                'bounds': self.image_bounds,
                'width': src.width,
                'height': src.height,
                'transform': self.transform,
                'resolution': (src.transform[0], abs(src.transform[4]))
            }
            
            print(f"✓ Loaded as-built: {image_path.name}")
            print(f"  CRS: {self.image_crs}")
            print(f"  Bounds: {self.image_bounds}")
            print(f"  Size: {src.width} x {src.height}")
            
            # Check CRS
            if reproject_if_needed and self.image_crs and self.image_crs != self.target_crs:
                print(f"\n⚠️  Warning: Image CRS differs from target CRS")
                print(f"   Image CRS: {self.image_crs}")
                print(f"   Target CRS: {self.target_crs}")
                print(f"   Consider reprojecting for accurate measurements")
                
        return metadata
    
    def set_image_for_segmentation(self):
        """
        Set the loaded image in SAM3 for segmentation.
        """
        if self.georef_image_path is None:
            raise ValueError("No image loaded. Call load_georeferenced_asbuilt() or start_georeferencing() first.")
            
        print("Loading image into SAM3...")
        self.sam3.set_image(str(self.georef_image_path))
        print(f"✓ Image ready for segmentation")
    
    def segment_features(self, prompt=None, bbox=None, output_path=None):
        """
        Segment features from the as-built using text prompts or bounding boxes.
        
        Parameters:
        -----------
        prompt : str, optional
            Text prompt (e.g., "water valve", "manhole", "pipe")
        bbox : list, optional
            Bounding box [xmin, ymin, xmax, ymax] in pixel coordinates
        output_path : str, optional
            Path to save segmentation masks as GeoTIFF
            
        Returns:
        --------
        numpy.ndarray : Segmentation masks
        """
        if prompt:
            print(f"🔍 Segmenting features with prompt: '{prompt}'")
            self.sam3.generate_masks(prompt=prompt)
        elif bbox:
            print(f"🔍 Segmenting features in bounding box: {bbox}")
            self.sam3.generate_masks(bbox=bbox)
        else:
            raise ValueError("Must provide either 'prompt' or 'bbox'")
        
        # Get masks
        self.masks = self.sam3.masks
        
        if output_path:
            self.sam3.save_masks(output_path)
            print(f"✓ Masks saved to: {output_path}")
            
        num_features = len(np.unique(self.masks)) - 1  # Exclude background
        print(f"✓ Segmented {num_features} features")
        
        return self.masks
    
    def masks_to_points(self, mask_ids=None, attribute_template=None):
        """
        Convert segmentation masks to point features (centroids).
        
        Perfect for: valves, manholes, hydrants, utility poles
        
        Parameters:
        -----------
        mask_ids : list, optional
            Specific mask IDs to convert. If None, converts all masks.
        attribute_template : dict, optional
            Template for feature attributes
            Example: {'feature_type': 'water_valve', 'material': 'cast_iron'}
            
        Returns:
        --------
        GeoDataFrame : Point features with geometry and attributes
        """
        if self.masks is None:
            raise ValueError("No masks available. Run segment_features() first.")
            
        features = []
        unique_masks = np.unique(self.masks)
        unique_masks = unique_masks[unique_masks != 0]  # Remove background
        
        if mask_ids:
            unique_masks = [m for m in unique_masks if m in mask_ids]
        
        print(f"Converting {len(unique_masks)} masks to points...")
        
        for mask_id in unique_masks:
            # Get mask pixels
            mask_pixels = np.where(self.masks == mask_id)
            
            if len(mask_pixels[0]) == 0:
                continue
            
            # Calculate centroid in pixel coordinates
            centroid_y = np.mean(mask_pixels[0])
            centroid_x = np.mean(mask_pixels[1])
            
            # Convert pixel coordinates to geographic coordinates
            geo_x, geo_y = rasterio.transform.xy(self.transform, centroid_y, centroid_x)
            
            # Create point geometry
            point = Point(geo_x, geo_y)
            
            # Build attributes
            attrs = {
                'mask_id': int(mask_id), 
                'area_pixels': len(mask_pixels[0])
            }
            if attribute_template:
                attrs.update(attribute_template)
            
            features.append({**attrs, 'geometry': point})
        
        # Create GeoDataFrame
        self.features_gdf = gpd.GeoDataFrame(features, crs=self.image_crs)
        
        print(f"✓ Created {len(self.features_gdf)} point features")
        
        return self.features_gdf
    
    def masks_to_polygons(self, mask_ids=None, attribute_template=None, simplify_tolerance=0.5):
        """
        Convert segmentation masks to polygon features.
        
        Perfect for: buildings, ponds, large structures, tanks
        
        Parameters:
        -----------
        mask_ids : list, optional
            Specific mask IDs to convert
        attribute_template : dict, optional
            Template for feature attributes
        simplify_tolerance : float
            Tolerance for simplifying polygon vertices (in map units)
            
        Returns:
        --------
        GeoDataFrame : Polygon features
        """
        if self.masks is None:
            raise ValueError("No masks available. Run segment_features() first.")
        
        from rasterio import features as rio_features
        
        features_list = []
        unique_masks = np.unique(self.masks)
        unique_masks = unique_masks[unique_masks != 0]
        
        if mask_ids:
            unique_masks = [m for m in unique_masks if m in mask_ids]
        
        print(f"Converting {len(unique_masks)} masks to polygons...")
        
        for mask_id in unique_masks:
            # Create binary mask for this feature
            binary_mask = (self.masks == mask_id).astype(np.uint8)
            
            # Convert raster to vector polygons
            shapes = rio_features.shapes(binary_mask, transform=self.transform)
            
            for geom, value in shapes:
                if value == 1:  # Only process feature pixels
                    # Create polygon from coordinates
                    if geom['type'] == 'Polygon':
                        polygon = Polygon(geom['coordinates'][0])
                    else:
                        continue
                    
                    # Simplify if requested
                    if simplify_tolerance > 0:
                        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)
                    
                    # Build attributes
                    attrs = {
                        'mask_id': int(mask_id),
                        'area_sqft': polygon.area,
                        'perimeter_ft': polygon.length
                    }
                    if attribute_template:
                        attrs.update(attribute_template)
                    
                    features_list.append({**attrs, 'geometry': polygon})
        
        self.features_gdf = gpd.GeoDataFrame(features_list, crs=self.image_crs)
        
        print(f"✓ Created {len(self.features_gdf)} polygon features")
        
        return self.features_gdf
    
    def masks_to_lines(self, mask_ids=None, attribute_template=None, 
                       simplify_tolerance=0.5, min_length=5.0):
        """
        Convert segmentation masks to line features (skeletonized centerlines).
        
        Perfect for: pipes, cables, conduits, trenches
        
        Parameters:
        -----------
        mask_ids : list, optional
            Specific mask IDs to convert
        attribute_template : dict, optional
            Template for feature attributes
        simplify_tolerance : float
            Tolerance for simplifying line vertices
        min_length : float
            Minimum line length to keep (filters out noise)
            
        Returns:
        --------
        GeoDataFrame : Line features
        """
        if self.masks is None:
            raise ValueError("No masks available. Run segment_features() first.")
        
        from skimage.morphology import skeletonize
        from rasterio import features as rio_features
        
        features_list = []
        unique_masks = np.unique(self.masks)
        unique_masks = unique_masks[unique_masks != 0]
        
        if mask_ids:
            unique_masks = [m for m in unique_masks if m in mask_ids]
        
        print(f"Converting {len(unique_masks)} masks to lines (skeletonizing)...")
        
        for mask_id in unique_masks:
            # Create binary mask
            binary_mask = (self.masks == mask_id).astype(bool)
            
            # Skeletonize to get centerline
            skeleton = skeletonize(binary_mask)
            skeleton = skeleton.astype(np.uint8)
            
            # Convert skeleton to vector lines
            shapes = rio_features.shapes(skeleton, transform=self.transform)
            
            for geom, value in shapes:
                if value == 1:
                    # Convert to LineString
                    coords = geom['coordinates'][0]
                    if len(coords) < 2:
                        continue
                    
                    line = LineString(coords)
                    
                    # Filter by minimum length
                    if line.length < min_length:
                        continue
                    
                    # Simplify if requested
                    if simplify_tolerance > 0:
                        line = line.simplify(simplify_tolerance, preserve_topology=True)
                    
                    # Build attributes
                    attrs = {
                        'mask_id': int(mask_id),
                        'length_ft': line.length
                    }
                    if attribute_template:
                        attrs.update(attribute_template)
                    
                    features_list.append({**attrs, 'geometry': line})
        
        self.features_gdf = gpd.GeoDataFrame(features_list, crs=self.image_crs)
        
        print(f"✓ Created {len(self.features_gdf)} line features")
        
        return self.features_gdf
    
    def reproject_features(self, target_crs=None):
        """
        Reproject features to target CRS.
        
        Parameters:
        -----------
        target_crs : str, optional
            Target CRS (defaults to self.target_crs)
        """
        if self.features_gdf is None:
            raise ValueError("No features available. Convert masks first.")
        
        target = target_crs or self.target_crs
        
        if str(self.features_gdf.crs) != str(target):
            print(f"Reprojecting from {self.features_gdf.crs} to {target}...")
            self.features_gdf = self.features_gdf.to_crs(target)
            print(f"✓ Features reprojected")
        else:
            print(f"✓ Features already in target CRS: {target}")
        
        return self.features_gdf
    
    def create_segmentation_interface(self, height="800px"):
        """
        Create interactive segmentation interface with SAM3.
        
        Features:
        - Text prompt input for feature types
        - Real-time segmentation preview
        - Feature type selection (points/lines/polygons)
        - Attribute entry
        - Export options
        
        Parameters:
        -----------
        height : str
            Map height for display
        """
        if self.georef_image_path is None:
            raise ValueError("No georeferenced image loaded! "
                           "Run start_georeferencing() or load_georeferenced_asbuilt() first.")
        
        print("🎯 Interactive Segmentation Interface")
        print("=" * 70)
        
        # Set image in SAM3
        self.set_image_for_segmentation()
        
        # Create interactive map
        self.map = leafmap.Map(height=height)
        self.map.add_raster(str(self.georef_image_path), layer_name="As-Built")
        
        # Create controls
        prompt_input = widgets.Text(
            value='',
            placeholder='Enter feature type (e.g., "valve", "manhole", "pipe")',
            description='Text Prompt:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        geometry_type = widgets.Dropdown(
            options=['Points', 'Lines', 'Polygons'],
            value='Points',
            description='Output Type:',
            style={'description_width': '100px'}
        )
        
        segment_btn = widgets.Button(
            description='🔍 Segment Features',
            button_style='success',
            tooltip='Run segmentation with current prompt',
            layout=widgets.Layout(width='180px')
        )
        
        convert_btn = widgets.Button(
            description='📐 Convert to Vector',
            button_style='primary',
            disabled=True,
            layout=widgets.Layout(width='180px')
        )
        
        export_type = widgets.Dropdown(
            options=['Geodatabase', 'Shapefile', 'GeoJSON'],
            value='Shapefile',
            description='Export Format:',
            style={'description_width': '100px'}
        )
        
        export_path = widgets.Text(
            value='',
            placeholder='/path/to/output.shp',
            description='Export Path:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        export_btn = widgets.Button(
            description='💾 Export Features',
            button_style='info',
            disabled=True,
            layout=widgets.Layout(width='180px')
        )
        
        # Attribute fields
        feature_type_input = widgets.Text(
            value='',
            placeholder='e.g., water_valve, storm_manhole',
            description='Feature Type:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        material_input = widgets.Text(
            value='',
            placeholder='e.g., cast_iron, PVC, concrete',
            description='Material:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        size_input = widgets.Text(
            value='',
            placeholder='e.g., 8", 12", 48"',
            description='Size:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='400px')
        )
        
        # Output display
        output = widgets.Output(layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            min_height='150px',
            max_height='400px',
            overflow_y='auto'
        ))
        
        with output:
            print("💡 Enter a text prompt and click 'Segment Features' to begin")
        
        def on_segment_click(b):
            """Run segmentation."""
            if not prompt_input.value:
                with output:
                    clear_output()
                    print("❌ Please enter a text prompt")
                return
            
            with output:
                clear_output()
                print(f"🔍 Segmenting features: '{prompt_input.value}'...")
                print("⏳ This may take a moment...")
            
            try:
                self.segment_features(prompt=prompt_input.value)
                
                with output:
                    clear_output()
                    num_features = len(np.unique(self.masks)) - 1
                    print("=" * 60)
                    print(f"✓ Segmentation complete!")
                    print(f"✓ Found {num_features} features")
                    print("=" * 60)
                    print("\nNext steps:")
                    print("1. Select output geometry type (Points/Lines/Polygons)")
                    print("2. Optionally fill in attribute fields")
                    print("3. Click 'Convert to Vector'")
                
                convert_btn.disabled = False
                
                # Show masks on map (optional - can be slow)
                # self.sam3.show_anns()
                
            except Exception as e:
                with output:
                    clear_output()
                    print("❌ Segmentation error:")
                    print(f"   {str(e)}")
                    import traceback
                    print("\nFull traceback:")
                    print(traceback.format_exc())
        
        def on_convert_click(b):
            """Convert masks to vector features."""
            with output:
                clear_output()
                print(f"📐 Converting {len(np.unique(self.masks)) - 1} masks to {geometry_type.value.lower()}...")
            
            try:
                # Build attribute template
                attrs = {}
                if feature_type_input.value:
                    attrs['feature_type'] = feature_type_input.value
                if material_input.value:
                    attrs['material'] = material_input.value
                if size_input.value:
                    attrs['size'] = size_input.value
                
                # Convert based on selected geometry type
                if geometry_type.value == 'Points':
                    self.masks_to_points(attribute_template=attrs)
                elif geometry_type.value == 'Lines':
                    self.masks_to_lines(attribute_template=attrs)
                else:  # Polygons
                    self.masks_to_polygons(attribute_template=attrs)
                
                # Reproject to target CRS
                self.reproject_features()
                
                with output:
                    clear_output()
                    print("=" * 60)
                    print(f"✓ Vector conversion complete!")
                    print("=" * 60)
                    print(f"Created {len(self.features_gdf)} {geometry_type.value.lower()}")
                    print(f"CRS: {self.features_gdf.crs}")
                    print("\nAttribute Preview:")
                    print(self.features_gdf.head())
                    print("\n" + "=" * 60)
                    print("Ready to export!")
                    print("1. Select export format")
                    print("2. Enter export path")
                    print("3. Click 'Export Features'")
                
                export_btn.disabled = False
                
            except Exception as e:
                with output:
                    clear_output()
                    print("❌ Conversion error:")
                    print(f"   {str(e)}")
                    import traceback
                    print("\nFull traceback:")
                    print(traceback.format_exc())
        
        def on_export_click(b):
            """Export features to file."""
            if not export_path.value:
                with output:
                    clear_output()
                    print("❌ Please specify an export path")
                return
            
            with output:
                clear_output()
                print(f"💾 Exporting to {export_type.value}...")
            
            try:
                export_file = Path(export_path.value)
                
                if export_type.value == 'Geodatabase':
                    # Extract GDB path and feature class name
                    if '.gdb' in str(export_file):
                        parts = str(export_file).split('.gdb')
                        gdb_path = Path(parts[0] + '.gdb')
                        fc_name = Path(parts[1]).stem if len(parts) > 1 else 'features'
                    else:
                        gdb_path = export_file.parent / f"{export_file.stem}.gdb"
                        fc_name = export_file.stem
                    
                    self.export_to_geodatabase(gdb_path, fc_name)
                    export_location = gdb_path / fc_name
                    
                elif export_type.value == 'Shapefile':
                    self.export_to_shapefile(export_path.value)
                    export_location = export_path.value
                    
                else:  # GeoJSON
                    self.features_gdf.to_file(export_path.value, driver='GeoJSON')
                    export_location = export_path.value
                    print(f"✓ Exported to GeoJSON: {export_path.value}")
                
                with output:
                    clear_output()
                    print("=" * 60)
                    print("🎉 SUCCESS!")
                    print("=" * 60)
                    print(f"✓ Exported {len(self.features_gdf)} features")
                    print(f"✓ Location: {export_location}")
                    print(f"✓ CRS: {self.features_gdf.crs}")
                    print("\n" + "=" * 60)
                    print("As-Built processing complete!")
                    print("\nNext steps:")
                    print("1. Open in ArcGIS Pro")
                    print("2. Review and edit features as needed")
                    print("3. Integrate into your GIS database")
                    
            except Exception as e:
                with output:
                    clear_output()
                    print("❌ Export error:")
                    print(f"   {str(e)}")
                    import traceback
                    print("\nFull traceback:")
                    print(traceback.format_exc())
        
        # Connect handlers
        segment_btn.on_click(on_segment_click)
        convert_btn.on_click(on_convert_click)
        export_btn.on_click(on_export_click)
        
        # Layout
        segmentation_controls = widgets.VBox([
            widgets.HTML("<h3 style='background: #4CAF50; color: white; padding: 10px;'>"
                        "1️⃣ Segment Features</h3>"),
            prompt_input,
            segment_btn
        ], layout=widgets.Layout(margin='10px 0'))
        
        conversion_controls = widgets.VBox([
            widgets.HTML("<h3 style='background: #2196F3; color: white; padding: 10px;'>"
                        "2️⃣ Convert to Vector</h3>"),
            geometry_type,
            widgets.HTML("<p style='margin: 10px 0; font-weight: bold;'>Attributes (optional):</p>"),
            feature_type_input,
            material_input,
            size_input,
            convert_btn
        ], layout=widgets.Layout(margin='10px 0'))
        
        export_controls = widgets.VBox([
            widgets.HTML("<h3 style='background: #FF9800; color: white; padding: 10px;'>"
                        "3️⃣ Export</h3>"),
            export_type,
            export_path,
            export_btn
        ], layout=widgets.Layout(margin='10px 0'))
        
        controls_panel = widgets.VBox([
            segmentation_controls,
            widgets.HTML("<hr style='border: 1px solid #ddd;'>"),
            conversion_controls,
            widgets.HTML("<hr style='border: 1px solid #ddd;'>"),
            export_controls,
            widgets.HTML("<hr style='border: 1px solid #ddd;'>"),
            widgets.HTML("<h3>Output:</h3>"),
            output
        ], layout=widgets.Layout(
            padding='15px',
            border='2px solid #ddd',
            border_radius='5px'
        ))
        
        map_panel = widgets.VBox([
            widgets.HTML("<h3>As-Built Map</h3>"),
            self.map
        ])
        
        interface = widgets.HBox([
            map_panel,
            controls_panel
        ], layout=widgets.Layout(width='100%'))
        
        display(interface)
    
    def export_to_geodatabase(self, gdb_path, feature_class_name, overwrite=True):
        """
        Export features to an Esri File Geodatabase.
        
        Parameters:
        -----------
        gdb_path : str or Path
            Path to .gdb geodatabase
        feature_class_name : str
            Name for the feature class
        overwrite : bool
            Overwrite if exists
        """
        if self.features_gdf is None:
            raise ValueError("No features to export. Convert masks first.")
        
        gdb_path = Path(gdb_path)
        
        # Ensure GDB exists
        if not gdb_path.exists():
            print(f"Creating geodatabase: {gdb_path}")
            gdb_path.mkdir(parents=True, exist_ok=True)
        
        # Export using OpenFileGDB driver
        try:
            self.features_gdf.to_file(
                str(gdb_path), 
                layer=feature_class_name, 
                driver="OpenFileGDB"
            )
            print(f"✓ Exported to {gdb_path}/{feature_class_name}")
        except Exception as e:
            # Fallback to shapefile if GDB fails
            print(f"⚠️  Geodatabase export failed, trying shapefile...")
            shp_path = gdb_path.parent / f"{feature_class_name}.shp"
            self.export_to_shapefile(shp_path)
        
        return gdb_path / feature_class_name
    
    def export_to_shapefile(self, output_path, overwrite=True):
        """
        Export features to a shapefile.
        
        Parameters:
        -----------
        output_path : str or Path
            Path to output shapefile (.shp)
        overwrite : bool
            Overwrite if exists
        """
        if self.features_gdf is None:
            raise ValueError("No features to export. Convert masks first.")
        
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing files if overwrite
        if overwrite and output_path.exists():
            import glob
            for f in glob.glob(str(output_path.with_suffix('.*'))):
                os.remove(f)
        
        self.features_gdf.to_file(str(output_path))
        
        print(f"✓ Exported {len(self.features_gdf)} features to {output_path}")
        
        return output_path


# Convenience function for quick setup
def create_autopilot(target_crs="EPSG:3435", use_gpu=True):
    """
    Quick setup function to create AsBuiltAutopilot with SAM3.
    
    Parameters:
    -----------
    target_crs : str
        Target coordinate reference system
    use_gpu : bool
        Use GPU acceleration if available
        
    Returns:
    --------
    AsBuiltAutopilot instance
    """
    from samgeo import SamGeo3
    import torch
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    print(f"Initializing SAM3 on {device}...")
    
    sam3 = SamGeo3(
        backend="transformers",
        device=device,
        checkpoint_path=None,
        load_from_HF=True
    )
    
    autopilot = AsBuiltAutopilot(sam3_model=sam3, target_crs=target_crs)
    
    return autopilot
