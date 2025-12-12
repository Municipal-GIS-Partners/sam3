# Quick Start Guide

## Installation (5 minutes)

### 1. Install Dependencies

Make sure you're in your SAM3 conda environment:

```bash
conda activate sam3
cd c:\dev\sam3\video_text
pip install -r requirements.txt
```

### 2. Verify SAM3 Installation

```bash
python -c "import sys; sys.path.insert(0, '../sam3'); from sam3.model_builder import build_sam3_image_model; print('SAM3 OK')"
```

## Running the Application

### Start the Server

```bash
python app.py
```

You should see output like:
```
======================================================================
SAM3 Video Text-Prompt Segmentation - Starting up...
======================================================================

[INFO] PyTorch version: 2.7.0
[INFO] CUDA available: True
[INFO] GPU device: NVIDIA GeForce RTX 5070

Server starting on http://localhost:5000
======================================================================
```

### Open in Browser

Navigate to: **http://localhost:5000**

## First Test Run

### 1. Upload a Video
- Click "Choose Video File"
- Select a short video (10-30 seconds recommended for testing)
- Wait for upload to complete

### 2. Configure Settings
- **Text Prompt**: Enter what you want to segment (e.g., "person", "car", "bottle")
- **Frame Interval**: Set to 1.0 seconds (process 1 frame per second)
- **Confidence Threshold**: Keep at 0.50 for first test

### 3. Process
- Click "Process Video"
- Wait for processing to complete (progress bar will show status)

### 4. View Results
- Use Previous/Next buttons to navigate
- Click thumbnails to jump to frames
- Use arrow keys (← →) for quick navigation

### 5. Download
- Click "Download Current Frame" for single image
- Click "Download All Results (ZIP)" for all processed frames

## Tips for Best Results

### Text Prompts
- Use simple, specific terms: "person", "car", "dog"
- Avoid complex phrases: "person wearing red shirt" → just use "person"
- Try singular form: "car" instead of "cars"

### Frame Interval
- **Fast action**: 0.5 seconds (more frames, slower processing)
- **Moderate**: 1.0 seconds (balanced)
- **Static scenes**: 2.0-5.0 seconds (fewer frames, faster processing)

### Confidence Threshold
- **Default**: 0.50 (balanced)
- **More masks**: Lower to 0.30-0.40
- **Fewer false positives**: Raise to 0.60-0.70

## Troubleshooting

### SAM3 Model Not Loading
```bash
# Verify SAM3 is installed correctly
cd c:\dev\sam3\sam3
pip install -e .
```

### Port Already in Use
If port 5000 is busy, edit `app.py` line 668:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

### Out of Memory (GPU)
- Process shorter videos
- Increase frame interval (fewer frames)
- Reduce video resolution before upload

### No Masks Found
- Lower confidence threshold
- Try different text prompts
- Verify object is visible in frames

## Example Prompts

Good prompts for common scenarios:

- **People**: "person"
- **Vehicles**: "car", "truck", "bicycle"
- **Animals**: "dog", "cat", "bird"
- **Objects**: "bottle", "cup", "phone", "laptop"
- **Sports**: "ball", "racket", "goal"
- **Nature**: "tree", "flower", "rock"

## Keyboard Shortcuts

- **←** Previous frame
- **→** Next frame
- **Home** First frame (via browser scroll)
- **End** Last frame (via browser scroll)

## Performance Expectations

### Frame Extraction
- **10-second video**: ~1-2 seconds
- **60-second video**: ~3-5 seconds

### Segmentation (GPU)
- **Per frame**: ~2-5 seconds
- **10 frames**: ~30-50 seconds
- **60 frames**: ~3-5 minutes

### Segmentation (CPU)
- **Per frame**: ~15-30 seconds
- **10 frames**: ~3-5 minutes
- **60 frames**: ~15-30 minutes

## What's Next?

### Try Different Videos
- Test with various content types
- Experiment with different objects
- Try indoor vs outdoor scenes

### Optimize Settings
- Find best confidence threshold for your use case
- Adjust frame interval based on video content
- Test different text prompts

### Production Use
- Process longer videos in batches
- Save important results before closing browser
- Consider automating with the API endpoints

## Need Help?

Check the main [README.md](README.md) for:
- Detailed technical documentation
- API endpoint specifications
- Session management details
- Advanced configuration options

Enjoy segmenting! 🎥✨
