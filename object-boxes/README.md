# SAM3 Bounding Box Segmentation UI

A modern web-based UI for drawing bounding boxes over images and using SAM3 to segment objects within those boxes.

## Features

- **Bounding Box Mode**: Draw bounding boxes on your image with click-and-drag interaction
- **Text Prompt Mode**: Use natural language prompts to segment objects (e.g., "a person", "yellow excavator")
- **Interactive Canvas**: Visual feedback while drawing boxes
- **Real-time Results**: See segmentation masks overlaid on your image
- **Download Results**: Save the segmented image to your computer
- **Multi-page Docs**: Upload TIFF/PDF files, draw boxes once, and process every page

## Installation

1. Make sure you have SAM3 installed (from the parent directory):
```bash
cd ../..
pip install -e .
```

2. Install additional dependencies for the UI:
```bash
cd sam3/object-boxes
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the UI:
   - Upload an image
   - Choose between Bounding Box or Text Prompt mode
   - **Bounding Box Mode**: Click and drag on the image to draw boxes around objects
   - **Text Prompt Mode**: Enter a text description of what you want to segment
   - Click "Run SAM3" to process
   - Download your results!

### PDFs and TIFFs
- Upload the PDF/TIFF; the server converts it and shows the first page
- Draw boxes once (or enter a text prompt) on that preview page
- Run SAM3 to process **every page**; you can preview each page in the results dropdown

## How It Works

This UI uses:
- **Backend**: Flask web server that interfaces with SAM3
- **Frontend**: Vanilla JavaScript with HTML5 Canvas for drawing
- **SAM3 API**: Uses the official `build_sam3_image_model()` and `Sam3Processor` API

### Bounding Box Mode
When you draw boxes, they are sent to SAM3 via `processor.set_boxes()` to segment objects within those regions.

### Text Prompt Mode
Text prompts are sent to SAM3 via `processor.set_text_prompt()` to find and segment objects matching the description.

## Technical Details

- Maximum upload size: 64MB
- Canvas automatically scales large images for display
- Bounding boxes are scaled back to original image dimensions for processing
- Results show confidence scores for each detected mask
- Each mask is overlaid with a unique color for visualization

## Architecture

```
object-boxes/
├── app.py                 # Flask backend
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── js/
│       └── app.js        # Canvas drawing & API calls
├── templates/
│   └── index.html        # Main UI
├── requirements.txt
└── README.md
```

## Troubleshooting

**Model loading is slow**: The first request will download the SAM3 checkpoint from HuggingFace. Subsequent requests will be faster.

**Out of memory**: Reduce image size or use a machine with more VRAM/RAM.

**Canvas not responding**: Make sure you've uploaded an image first and are in Bounding Box mode.

## License

This UI inherits the same license as SAM3.
