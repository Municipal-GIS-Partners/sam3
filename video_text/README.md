# SAM3 Video Text-Prompt Segmentation

A web-based interface for segmenting objects in videos using text prompts with Meta's SAM3 (Segment Anything Model 3).

## Features

- **Video Upload**: Support for common video formats (MP4, AVI, MOV, etc.)
- **Configurable Frame Extraction**: Extract frames at uniform intervals (0.5-10 seconds)
- **Text-Prompt Segmentation**: Segment objects using natural language prompts
- **Confidence Threshold Control**: Adjust minimum confidence for mask filtering
- **Side-by-Side Viewer**: Compare original and segmented frames
- **Interactive Navigation**: Browse results with previous/next buttons, thumbnails, or keyboard arrows
- **Download Options**: Download individual frames or all results as a ZIP file

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, falls back to CPU)
- Disk space for temporary video storage and results

## Installation

### 1. Create Conda Environment

```bash
conda create -n sam3 python=3.11
conda activate sam3
```

### 2. Install PyTorch

For CUDA 12.6 (RTX 5070 and newer):
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For CUDA 12.4 (Tesla T4 and compatible):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Dependencies

```bash
cd video_text
pip install -r requirements.txt
```

### 4. Install SAM3

From the parent directory:
```bash
cd ..
pip install -e .
```

## Usage

### 1. Start the Server

```bash
cd video_text
python app.py
```

The server will start on http://localhost:5000

### 2. Upload and Process Video

1. Open your browser to http://localhost:5000
2. Click "Choose Video File" and upload a video
3. Enter a text prompt (e.g., "person", "car", "dog")
4. Adjust the frame interval slider (default: 1.0 seconds)
5. Adjust the confidence threshold slider (default: 0.50)
6. Click "Process Video"

### 3. View Results

- Use Previous/Next buttons to navigate frames
- Click thumbnails to jump to specific frames
- Use arrow keys (← →) for keyboard navigation
- Download individual frames or all results as ZIP

## How It Works

1. **Video Upload**: Video is uploaded and metadata is extracted using OpenCV
2. **Frame Extraction**: Frames are extracted at uniform intervals using frame seeking
3. **Segmentation**: Each frame is processed individually with SAM3's text-prompt segmentation
4. **Visualization**: Masks are overlaid on original frames with colored transparency
5. **Results Storage**: Processed frames are saved to disk for viewing and download

## Technical Details

### Frame Extraction

Frames are extracted at uniform intervals calculated as:
```
frame_indices = range(0, total_frames, fps * interval_seconds)
```

This ensures consistent temporal sampling across the video.

### SAM3 Integration

The application uses SAM3's image processor on individual frames:

```python
# Load model once (singleton)
model, processor = load_sam3_model()

# Process each frame
state = processor.set_image(image)
processor.set_confidence_threshold(threshold)
results = processor.set_text_prompt(prompt, state=state)

# Extract masks and scores
masks = results['masks'].detach().cpu().numpy()
scores = results['scores'].detach().cpu().numpy()
```

### Session Management

- Each upload creates a unique session ID
- Files are stored in `outputs/{session_id}/`
- Sessions are automatically cleaned up after 24 hours (optional)
- Manual cleanup via DELETE endpoint

## Directory Structure

```
video_text/
├── app.py                  # Flask backend
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── static/
│   ├── css/
│   │   └── style.css      # UI styles
│   └── js/
│       └── app.js         # Frontend logic
├── templates/
│   └── index.html         # Main UI
├── uploads/               # Temporary video storage (gitignored)
└── outputs/               # Results storage (gitignored)
    └── {session_id}/
        ├── video.{ext}
        ├── frames/
        └── results/
```

## Troubleshooting

### Video Upload Fails
- Check file size (max 512MB)
- Ensure video format is supported
- Verify disk space available

### Segmentation Produces No Results
- Try lowering the confidence threshold
- Check if the text prompt matches objects in the video
- Verify SAM3 model loaded successfully

### Slow Processing
- GPU recommended for faster processing
- Reduce frame interval to process fewer frames
- Consider using a shorter video for testing

## Performance Notes

- **Frame Extraction**: ~1-5 seconds for typical videos
- **Segmentation**: ~2-5 seconds per frame on GPU, ~10-30 seconds on CPU
- **Memory Usage**: ~4-8GB GPU memory recommended

## Credits

Built with:
- [SAM3](https://github.com/facebookresearch/sam3) by Meta AI
- Flask for web framework
- OpenCV for video processing

## License

See parent SAM3 project for license information.
