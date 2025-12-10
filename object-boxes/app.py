#!/usr/bin/env python
"""
SAM3 Bounding Box UI - Flask Backend
-------------------------------------
A web-based UI for drawing bounding boxes over images and using SAM3
to segment objects within those boxes.
"""

import os
import base64
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size for uploads

# Global model and processor
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

        import time
        start_time = time.time()

        _sam3_model = build_sam3_image_model()
        _sam3_processor = Sam3Processor(_sam3_model)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"[SAM3] Model loaded successfully in {elapsed:.1f} seconds")
        print("[SAM3] Ready to process images!")
        print("=" * 70 + "\n")
    return _sam3_model, _sam3_processor


def decode_base64_image(data_url):
    """Decode base64 image data URL to PIL Image."""
    header, encoded = data_url.split(',', 1)
    decoded = base64.b64decode(encoded)
    image = Image.open(BytesIO(decoded))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def encode_image_to_base64(image):
    """Encode PIL Image to base64 data URL."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


class Document:
    """
    Holds multi-page documents (PDF/TIFF) so we can render pages on demand.
    """
    def __init__(self, kind: str, page_count: int, reference_size: Tuple[int, int],
                 renderer: Callable[[int], Image.Image], page_sizes: Optional[List[Tuple[int, int]]] = None):
        self.kind = kind
        self.page_count = page_count
        self.reference_size = reference_size
        self._renderer = renderer
        self.page_sizes = page_sizes or [reference_size for _ in range(page_count)]

    def render_page(self, idx: int) -> Image.Image:
        return self._renderer(idx)

    def get_page_size(self, idx: int) -> Tuple[int, int]:
        if self.page_sizes and 0 <= idx < len(self.page_sizes):
            return self.page_sizes[idx]
        return self.reference_size


class DocumentStore:
    def __init__(self):
        self._store: Dict[str, Document] = {}

    def add(self, doc: Document) -> str:
        doc_id = uuid.uuid4().hex
        self._store[doc_id] = doc
        return doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        return self._store.get(doc_id)


doc_store = DocumentStore()


def _render_pdf_pages(pdf_bytes: bytes, max_side: int = 8000, target_dpi: float = 300.0) -> Tuple[Callable[[int], Image.Image], int, List[Tuple[int, int]]]:
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF (fitz) is required to process PDF files. Please install with 'pip install pymupdf'.") from e

    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = pdf_doc.page_count

    scale_from_dpi = target_dpi / 72.0  # PDF default is 72 DPI

    def render(idx: int) -> Image.Image:
        page = pdf_doc.load_page(idx)
        rect = page.rect
        width, height = rect.width, rect.height
        max_dim = max(width, height)
        scale_limit = max_side / max_dim if max_dim > 0 else scale_from_dpi
        scale = min(scale_from_dpi, scale_limit) if max_dim > 0 else scale_from_dpi
        scale = max(1.0, scale)
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    page_sizes = []
    for i in range(page_count):
        rect = pdf_doc.load_page(i).rect
        max_dim = max(rect.width, rect.height)
        scale_limit = max_side / max_dim if max_dim > 0 else scale_from_dpi
        scale = min(scale_from_dpi, scale_limit) if max_dim > 0 else scale_from_dpi
        scale = max(1.0, scale)
        page_sizes.append((int(rect.width * scale), int(rect.height * scale)))

    return render, page_count, page_sizes


def _render_tiff_frames(file: Image.Image, max_side: int = 8000) -> Tuple[Callable[[int], Image.Image], int, List[Tuple[int, int]]]:
    frames: List[bytes] = []
    sizes: List[Tuple[int, int]] = []

    try:
        frame_count = getattr(file, "n_frames", 1)
    except Exception:
        frame_count = 1

    for i in range(frame_count):
        file.seek(i)
        frame = file.convert("RGB")
        w, h = frame.size
        sizes.append((w, h))

        # Downscale if huge to prevent memory blowups
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            frame = frame.resize(new_size, Image.BILINEAR)
            w, h = new_size
            sizes[-1] = (w, h)

        buf = BytesIO()
        frame.save(buf, format="PNG")
        frames.append(buf.getvalue())

    def render(idx: int) -> Image.Image:
        buf = BytesIO(frames[idx])
        return Image.open(buf).convert("RGB")

    return render, frame_count, sizes


def overlay_masks_on_image(image, masks, scores, colors=None, alpha=0.5):
    """
    Overlay multiple masks on an image with different colors.

    Args:
        image: PIL Image
        masks: numpy array [N, H, W] of binary masks
        scores: numpy array [N] of confidence scores
        colors: list of RGB tuples, one per mask
        alpha: transparency of overlay

    Returns:
        PIL Image with overlays
    """
    if masks is None or len(masks) == 0:
        return image

    img_np = np.array(image).astype(np.uint8)
    overlay = img_np.copy()
    target_h, target_w = img_np.shape[:2]

    # Default colors if not provided
    if colors is None:
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


def process_page_with_tiling(processor, image, boxes_xyxy, prompt, tile_size=1200, overlap=128):
    """
    Run tiled inference to preserve high resolution and catch small objects.

    Args:
        processor: Sam3Processor
        image: PIL Image
        boxes_xyxy: list of [x1, y1, x2, y2] in pixel coords
        prompt: text prompt or empty string
        tile_size: max tile dimension
        overlap: number of pixels to overlap tiles for seamless stitching
    """
    img_w, img_h = image.size
    base_overlay = np.array(image).astype(np.uint8)
    all_scores = []
    all_colors = []
    rng = np.random.RandomState(1234)
    total_masks = 0

    # Normalize boxes once for quick filtering
    boxes_array = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)

    step = tile_size - overlap
    y0 = 0
    while y0 < img_h:
        x0 = 0
        y1 = min(y0 + tile_size, img_h)
        while x0 < img_w:
            x1 = min(x0 + tile_size, img_w)
            tile = image.crop((x0, y0, x1, y1))

            # Select boxes that intersect this tile
            tile_boxes = []
            if boxes_array.size > 0:
                x1s = boxes_array[:, 0]
                y1s = boxes_array[:, 1]
                x2s = boxes_array[:, 2]
                y2s = boxes_array[:, 3]
                intersects = (x2s > x0) & (x1s < x1) & (y2s > y0) & (y1s < y1)
                selected = boxes_array[intersects]
                for bx in selected:
                    adj = [
                        max(bx[0] - x0, 0),
                        max(bx[1] - y0, 0),
                        min(bx[2] - x0, x1 - x0),
                        min(bx[3] - y0, y1 - y0),
                    ]
                    if adj[2] - adj[0] > 1 and adj[3] - adj[1] > 1:
                        tile_boxes.append(adj)

            # Skip tiles without prompts/boxes
            if not prompt and len(tile_boxes) == 0 and boxes_array.size > 0:
                x0 += step
                continue

            state_tile = processor.set_image(tile)
            if prompt:
                output_tile = processor.set_text_prompt(state=state_tile, prompt=prompt)
            else:
                output_tile = processor.set_boxes(state=state_tile, boxes=tile_boxes)

            masks_tile = output_tile.get('masks', None)
            scores_tile = output_tile.get('scores', None)
            if masks_tile is None or scores_tile is None:
                x0 += step
                continue

            if torch.is_tensor(masks_tile):
                masks_tile = masks_tile.detach().cpu().numpy()
            if torch.is_tensor(scores_tile):
                scores_tile = scores_tile.detach().cpu().numpy()

            # Colors for this tile
            colors_tile = []
            for _ in range(len(masks_tile)):
                colors_tile.append(tuple(rng.randint(50, 255, size=3).tolist()))

            # Overlay tile results
            tile_overlay = overlay_masks_on_image(tile, masks_tile, scores_tile, colors=colors_tile)
            tile_np = np.array(tile_overlay).astype(np.uint8)
            base_overlay[y0:y1, x0:x1] = tile_np

            all_scores.extend(scores_tile.tolist())
            all_colors.extend(colors_tile)
            total_masks += len(masks_tile)

            x0 += step
        y0 += step

    result_image = Image.fromarray(base_overlay)
    return {
        'result_image': encode_image_to_base64(result_image),
        'num_masks': total_masks,
        'scores': all_scores,
        'colors': all_colors,
        'width': img_w,
        'height': img_h,
    }


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    print(f"[ERROR] Upload too large: {error}")
    return jsonify({'error': 'File too large. Maximum upload size is 64MB.'}), 413

@app.route('/convert', methods=['POST'])
def convert_image():
    """
    Convert uploaded image/PDF to PNG data URL for browser display.
    - Standard images: returned as PNG data URL.
    - TIFF/PDF: stored server-side with doc_id; first page returned for preview.
    """
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        filename = secure_filename(file.filename or 'upload')
        file_bytes = file.read()
        print(f"[CONVERT] Received file: {filename} ({len(file_bytes)} bytes)")

        ext = Path(filename).suffix.lower()

        # PDF handling
        if ext == ".pdf":
            print("[CONVERT] PDF detected; rendering pages")
            renderer, page_count, page_sizes = _render_pdf_pages(file_bytes)
            preview = renderer(0)
            doc = Document(
                kind="pdf",
                page_count=page_count,
                reference_size=page_sizes[0],
                renderer=renderer,
                page_sizes=page_sizes,
            )
            doc_id = doc_store.add(doc)
            result_b64 = encode_image_to_base64(preview)
            return jsonify({
                'doc_id': doc_id,
                'doc_type': 'pdf',
                'page_count': page_count,
                'image_data': result_b64,
                'width': preview.width,
                'height': preview.height,
                'reference_width': page_sizes[0][0],
                'reference_height': page_sizes[0][1]
            })

        # TIFF handling (including multi-frame)
        if ext in {".tif", ".tiff"}:
            print("[CONVERT] TIFF detected; processing frames")
            image = Image.open(BytesIO(file_bytes))
            renderer, frame_count, frame_sizes = _render_tiff_frames(image)
            doc = Document(
                kind="tiff",
                page_count=frame_count,
                reference_size=frame_sizes[0],
                renderer=renderer,
                page_sizes=frame_sizes,
            )
            doc_id = doc_store.add(doc)
            preview = doc.render_page(0)
            result_b64 = encode_image_to_base64(preview)
            return jsonify({
                'doc_id': doc_id,
                'doc_type': 'tiff',
                'page_count': frame_count,
                'image_data': result_b64,
                'width': preview.width,
                'height': preview.height,
                'reference_width': frame_sizes[0][0],
                'reference_height': frame_sizes[0][1]
            })

        # Standard image
        image = Image.open(BytesIO(file_bytes))
        print(f"[CONVERT] Opened image mode={image.mode} size={image.size}")

        if image.mode != 'RGB':
            image = image.convert('RGB')
            print(f"[CONVERT] Converted image to RGB")

        result_b64 = encode_image_to_base64(image)
        return jsonify({
            'image_data': result_b64,
            'width': image.width,
            'height': image.height,
            'doc_id': None,
            'doc_type': 'image',
            'page_count': 1,
            'reference_width': image.width,
            'reference_height': image.height
        })
    except Exception as e:
        print(f"[CONVERT][ERROR] Failed to convert image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/page/<doc_id>/<int:page_index>', methods=['GET'])
def get_page(doc_id, page_index):
    """Return a specific page from a cached document."""
    doc = doc_store.get(doc_id)
    if doc is None:
        return jsonify({'error': 'Document not found or expired'}), 404
    if page_index < 0 or page_index >= doc.page_count:
        return jsonify({'error': 'Page index out of range'}), 400

    page_image = doc.render_page(page_index)
    page_b64 = encode_image_to_base64(page_image)
    width, height = page_image.size
    return jsonify({
        'doc_id': doc_id,
        'page_index': page_index,
        'image_data': page_b64,
        'width': width,
        'height': height,
        'reference_width': width,
        'reference_height': height,
        'page_count': doc.page_count
    })


@app.route('/segment', methods=['POST'])
def segment():
    """
    Process bounding boxes and return segmentation masks.

    Expects JSON:
    {
        "image": "data:image/png;base64,...",
        "doc_id": "optional doc id if processing multi-page document",
        "boxes": [[x1, y1, x2, y2], ...],
        "prompt": "optional text prompt"
    }
    """
    try:
        import time
        request_start = time.time()

        print("\n[REQUEST] New segmentation request received")

        data = request.json or {}

        doc_id = data.get('doc_id')
        boxes = data.get('boxes', [])
        prompt = data.get('prompt', '').strip()
        image_data_url = data.get('image')
        doc_page_count = None
        drawing_page_index = int(data.get('drawing_page_index') or 0)
        drawing_page_width = data.get('drawing_page_width')
        drawing_page_height = data.get('drawing_page_height')

        if not doc_id and not image_data_url:
            return jsonify({'error': 'No image or document provided'}), 400

        if len(boxes) == 0 and not prompt:
            print("[ERROR] No boxes or prompt provided")
            return jsonify({'error': 'Please provide bounding boxes or a text prompt'}), 400

        # Load model
        print("[REQUEST] Loading SAM3 model...")
        _, processor = load_sam3_model()

        def process_single_image(image, boxes_xyxy):
            # Use tiling to preserve high resolution and small objects.
            return process_page_with_tiling(processor, image, boxes_xyxy, prompt)

        # Multi-page document processing
        if doc_id:
            print(f"[REQUEST] Document mode enabled (doc_id={doc_id})")
            doc = doc_store.get(doc_id)
            if doc is None:
                return jsonify({'error': 'Document not found or expired'}), 400

            ref_w, ref_h = doc.reference_size
            base_w = float(drawing_page_width or ref_w)
            base_h = float(drawing_page_height or ref_h)
            if len(boxes) > 0:
                norm_boxes = np.array(boxes, dtype=np.float32) / np.array([base_w, base_h, base_w, base_h], dtype=np.float32)
            else:
                norm_boxes = np.zeros((0, 4), dtype=np.float32)
            page_results = []

            for page_idx in range(doc.page_count):
                print(f"[REQUEST] Processing page {page_idx + 1}/{doc.page_count}")
                page_image = doc.render_page(page_idx)
                page_w, page_h = page_image.size
                if not prompt:
                    page_boxes = (norm_boxes * np.array([page_w, page_h, page_w, page_h], dtype=np.float32)).tolist()
                else:
                    page_boxes = boxes  # unused when prompt is present

                page_payload = process_single_image(page_image, page_boxes)
                page_payload['page_index'] = page_idx
                page_payload['width'] = page_w
                page_payload['height'] = page_h
                page_results.append(page_payload)

            total_time = time.time() - request_start
            print(f"[REQUEST] Document request completed in {total_time:.2f}s\n")

            return jsonify({
                'page_count': doc.page_count,
                'pages': page_results
            })

        # Single image processing (default)
        print("[REQUEST] Decoding image...")
        image = decode_base64_image(image_data_url)
        print(f"[REQUEST] Image size: {image.size[0]}x{image.size[1]}")

        response_payload = process_single_image(image, boxes)

        total_time = time.time() - request_start
        print(f"[REQUEST] Request completed in {total_time:.2f}s\n")

        return jsonify(response_payload)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("SAM3 Bounding Box UI - Starting up...")
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
    print(f"[INFO] Flask debug mode: {True}")

    print("\n[INFO] SAM3 model will be loaded on first segmentation request")
    print("[INFO] First request may take longer due to checkpoint download")

    print("\n" + "=" * 70)
    print("Server starting on http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
