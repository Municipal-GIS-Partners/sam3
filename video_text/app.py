#!/usr/bin/env python
"""
SAM3 Video Text-Prompt Segmentation - Flask Backend
---------------------------------------------------
A web-based UI for uploading videos, extracting frames, and using SAM3
to segment objects with text prompts.
"""

import os
import sys
import uuid
import time
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import numpy as np
from PIL import Image
import torch
import cv2
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Add parent directory to path for SAM3 imports
sys.path.insert(0, str(Path(__file__).parent.parent / "sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max file size for videos
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'outputs'

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# Global model and processor (singleton pattern)
_sam3_model = None
_sam3_processor = None


def load_sam3_model():
    """Lazily load SAM3 model once."""
    global _sam3_model, _sam3_processor
    if _sam3_model is None or _sam3_processor is None:
        print("\n" + "=" * 70)
        print("[SAM3] Building image model (first-time setup)...")
        print("[SAM3] This may take a few minutes to download checkpoint...")
        print("=" * 70)

        start_time = time.time()

        _sam3_model = build_sam3_image_model()
        _sam3_processor = Sam3Processor(_sam3_model)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"[SAM3] Model loaded successfully in {elapsed:.1f} seconds")
        print("[SAM3] Ready to process images!")
        print("=" * 70 + "\n")
    return _sam3_model, _sam3_processor


class SessionManager:
    """Manage session storage and cleanup."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.sessions: Dict[str, Dict] = {}

    def create_session(self, video_filename: str) -> str:
        """Create new session with unique ID."""
        session_id = uuid.uuid4().hex
        session_dir = self.base_dir / session_id

        # Create directory structure
        (session_dir / 'frames').mkdir(parents=True, exist_ok=True)
        (session_dir / 'results').mkdir(parents=True, exist_ok=True)

        self.sessions[session_id] = {
            'id': session_id,
            'created_at': time.time(),
            'video_filename': video_filename,
            'dir': session_dir
        }

        print(f"[SESSION] Created session {session_id}")
        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        """Get session directory path."""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        return self.sessions[session_id]['dir']

    def cleanup_session(self, session_id: str):
        """Delete all files for a session."""
        if session_id in self.sessions:
            session_dir = self.sessions[session_id]['dir']
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
                print(f"[SESSION] Cleaned up session {session_id}")
            del self.sessions[session_id]

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours."""
        current_time = time.time()
        to_remove = []

        for session_id, info in self.sessions.items():
            age_hours = (current_time - info['created_at']) / 3600
            if age_hours > max_age_hours:
                to_remove.append(session_id)

        for session_id in to_remove:
            self.cleanup_session(session_id)

        if to_remove:
            print(f"[SESSION] Cleaned up {len(to_remove)} old sessions")


# Global session manager
session_manager = SessionManager(app.config['OUTPUT_FOLDER'])


class FrameExtractor:
    """Extract frames from video at uniform intervals."""

    def __init__(self, video_path: str, output_dir: Path):
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def get_metadata(self):
        """Extract video metadata."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': total_frames / fps if fps > 0 else 0
        }

    def extract_uniform_frames(self, interval_seconds: float):
        """
        Extract frames at uniform intervals.

        Args:
            interval_seconds: Time between extracted frames

        Returns:
            List of frame info dicts
        """
        metadata = self.get_metadata()
        fps = metadata['fps']
        total_frames = metadata['total_frames']

        # Calculate frame indices to extract
        frame_interval = int(fps * interval_seconds)
        if frame_interval < 1:
            frame_interval = 1

        frame_indices = list(range(0, total_frames, frame_interval))

        print(f"[EXTRACT] Extracting {len(frame_indices)} frames at {interval_seconds}s intervals")

        extracted = []
        for frame_idx in frame_indices:
            # Seek to frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Save frame
            frame_filename = f"frame_{frame_idx:06d}.png"
            frame_path = self.output_dir / frame_filename
            pil_image.save(frame_path)

            extracted.append({
                'idx': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else 0,
                'path': str(frame_path),
                'filename': frame_filename
            })

        print(f"[EXTRACT] Successfully extracted {len(extracted)} frames")
        return extracted


def overlay_masks_on_image(image, masks, scores, alpha=0.6):
    """
    Overlay multiple masks on an image with different colors.

    Args:
        image: PIL Image
        masks: numpy array [N, H, W] of binary masks
        scores: numpy array [N] of confidence scores
        alpha: transparency of overlay

    Returns:
        PIL Image with overlays
    """
    if masks is None or len(masks) == 0:
        return image

    img_np = np.array(image).astype(np.uint8)
    overlay = img_np.copy()
    target_h, target_w = img_np.shape[:2]

    # Generate colors
    colors = []
    for i in range(len(masks)):
        rng = np.random.RandomState(i + 42)
        colors.append(tuple(rng.randint(50, 255, size=3).tolist()))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        while mask.ndim > 2:
            mask = mask.squeeze()

        mask_bin = mask > 0.5

        # Resize if needed
        if mask_bin.shape != (target_h, target_w):
            mask_img = Image.fromarray((mask_bin * 255).astype(np.uint8), mode='L')
            mask_img = mask_img.resize((target_w, target_h), Image.BILINEAR)
            mask_bin = np.array(mask_img) > 127

        if not mask_bin.any():
            continue

        color = np.array(colors[i % len(colors)], dtype=np.float32)

        # Blend
        overlay[mask_bin] = (
            (1.0 - alpha) * overlay[mask_bin].astype(np.float32)
            + alpha * color
        ).astype(np.uint8)

    return Image.fromarray(overlay)


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    print(f"[ERROR] Upload too large: {error}")
    return jsonify({'error': 'File too large. Maximum upload size is 512MB.'}), 413


@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Upload video and extract metadata.

    Returns:
        {session_id, video_info: {fps, total_frames, duration, width, height}}
    """
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        filename = secure_filename(file.filename or 'video.mp4')
        print(f"[UPLOAD] Received video: {filename}")

        # Create session
        session_id = session_manager.create_session(filename)
        session_dir = session_manager.get_session_dir(session_id)

        # Save video
        video_path = session_dir / filename
        file.save(str(video_path))
        print(f"[UPLOAD] Saved video to {video_path}")

        # Extract metadata
        with FrameExtractor(str(video_path), session_dir / 'frames') as extractor:
            metadata = extractor.get_metadata()

        print(f"[UPLOAD] Video metadata: {metadata}")

        return jsonify({
            'session_id': session_id,
            'video_info': metadata
        })

    except Exception as e:
        print(f"[UPLOAD][ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/extract_frames', methods=['POST'])
def extract_frames():
    """
    Extract frames at specified interval.

    Input: {session_id, frame_interval_seconds}
    Returns: {session_id, frames: [{idx, timestamp, filename}]}
    """
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        interval_seconds = float(data.get('frame_interval_seconds', 1.0))

        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400

        session_dir = session_manager.get_session_dir(session_id)
        session_info = session_manager.sessions[session_id]
        video_path = session_dir / session_info['video_filename']

        if not video_path.exists():
            return jsonify({'error': 'Video file not found'}), 404

        print(f"[EXTRACT] Session {session_id}, interval={interval_seconds}s")

        # Extract frames
        with FrameExtractor(str(video_path), session_dir / 'frames') as extractor:
            frames = extractor.extract_uniform_frames(interval_seconds)

        # Store frame info in session
        session_manager.sessions[session_id]['frames'] = frames

        return jsonify({
            'session_id': session_id,
            'frames': frames
        })

    except Exception as e:
        print(f"[EXTRACT][ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/process', methods=['POST'])
def process_frames():
    """
    Process frames with SAM3 text-prompt segmentation.

    Input: {session_id, text_prompt, confidence_threshold}
    Returns: {results: [{frame_idx, timestamp, num_masks, scores}]}
    """
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        text_prompt = data.get('text_prompt', '').strip()
        confidence_threshold = float(data.get('confidence_threshold', 0.5))

        if not session_id:
            return jsonify({'error': 'No session_id provided'}), 400

        if not text_prompt:
            return jsonify({'error': 'No text prompt provided'}), 400

        session_dir = session_manager.get_session_dir(session_id)
        frames = session_manager.sessions[session_id].get('frames', [])

        if not frames:
            return jsonify({'error': 'No frames found. Please extract frames first.'}), 400

        print(f"[PROCESS] Session {session_id}, prompt='{text_prompt}', threshold={confidence_threshold}")
        print(f"[PROCESS] Processing {len(frames)} frames...")

        # Load SAM3 model
        _, processor = load_sam3_model()
        processor.confidence_threshold = confidence_threshold

        results = []
        for i, frame_info in enumerate(frames):
            frame_path = Path(frame_info['path'])

            if not frame_path.exists():
                print(f"[PROCESS] Warning: Frame not found: {frame_path}")
                continue

            # Load frame
            image = Image.open(frame_path)

            # Run SAM3
            state = processor.set_image(image)
            output = processor.set_text_prompt(text_prompt, state=state)

            # Extract results
            masks = output.get('masks')
            scores = output.get('scores')

            # Convert tensors to numpy
            if masks is not None and torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()
            if scores is not None and torch.is_tensor(scores):
                scores = scores.detach().cpu().numpy()

            num_masks = len(masks) if masks is not None else 0

            # Create overlay visualization
            if num_masks > 0:
                result_image = overlay_masks_on_image(image, masks, scores, alpha=0.6)
            else:
                result_image = image

            # Save result
            result_filename = f"result_{frame_info['idx']:06d}.png"
            result_path = session_dir / 'results' / result_filename
            result_image.save(result_path)

            results.append({
                'frame_idx': frame_info['idx'],
                'timestamp': frame_info['timestamp'],
                'filename': frame_info['filename'],
                'result_filename': result_filename,
                'num_masks': num_masks,
                'scores': scores.tolist() if scores is not None else []
            })

            if (i + 1) % 10 == 0:
                print(f"[PROCESS] Processed {i + 1}/{len(frames)} frames")

        print(f"[PROCESS] Completed processing {len(results)} frames")

        # Store results in session
        session_manager.sessions[session_id]['results'] = results

        return jsonify({
            'results': results
        })

    except Exception as e:
        print(f"[PROCESS][ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/image/<session_id>/<int:frame_idx>/<image_type>')
def get_image(session_id, frame_idx, image_type):
    """
    Serve individual image.

    Args:
        session_id: Session ID
        frame_idx: Frame index
        image_type: 'original' or 'result'
    """
    try:
        session_dir = session_manager.get_session_dir(session_id)

        if image_type == 'original':
            image_path = session_dir / 'frames' / f"frame_{frame_idx:06d}.png"
        elif image_type == 'result':
            image_path = session_dir / 'results' / f"result_{frame_idx:06d}.png"
        else:
            return jsonify({'error': 'Invalid image type'}), 400

        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404

        return send_file(image_path, mimetype='image/png')

    except Exception as e:
        print(f"[IMAGE][ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<session_id>/<int:frame_idx>')
def download_frame(session_id, frame_idx):
    """Download single result frame."""
    try:
        session_dir = session_manager.get_session_dir(session_id)
        image_path = session_dir / 'results' / f"result_{frame_idx:06d}.png"

        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404

        return send_file(
            image_path,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"result_{frame_idx:06d}.png"
        )

    except Exception as e:
        print(f"[DOWNLOAD][ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download_all/<session_id>')
def download_all(session_id):
    """Download all results as ZIP."""
    try:
        session_dir = session_manager.get_session_dir(session_id)
        results_dir = session_dir / 'results'

        if not results_dir.exists():
            return jsonify({'error': 'Results not found'}), 404

        # Create in-memory ZIP
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for result_file in sorted(results_dir.glob('result_*.png')):
                zf.write(result_file, result_file.name)

        memory_file.seek(0)

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'sam3_results_{session_id[:8]}.zip'
        )

    except Exception as e:
        print(f"[DOWNLOAD_ALL][ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Clean up session data."""
    try:
        session_manager.cleanup_session(session_id)
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"[DELETE][ERROR] {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("SAM3 Video Text-Prompt Segmentation - Starting up...")
    print("=" * 70)

    # Check Python and dependencies
    print(f"\n[INFO] Python version: {sys.version}")
    print(f"[INFO] Working directory: {os.getcwd()}")

    # Check PyTorch
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU device: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("[WARN] CUDA not available - running on CPU (will be slow)")

    # Check Flask config
    print(f"\n[INFO] Flask max upload size: {app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024:.0f} MB")
    print(f"[INFO] Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"[INFO] Output folder: {app.config['OUTPUT_FOLDER']}")

    print("\n[INFO] SAM3 model will be loaded on first segmentation request")
    print("[INFO] First request may take longer due to checkpoint download")

    print("\n" + "=" * 70)
    print("Server starting on http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
