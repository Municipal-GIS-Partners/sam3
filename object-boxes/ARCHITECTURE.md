# SAM3 Object Boxes UI – Architecture & Implementation Notes

This document explains what the app does, how it works end‑to‑end, and highlights key mechanisms and tradeoffs. It reads like a code review walkthrough so you can quickly reason about changes or fixes.

## What It Does
- Web UI to upload an image/PDF/TIFF, draw boxes (or provide a text prompt), and run SAM3 segmentation.
- Supports multi‑page docs: draw once on any page; applies across all pages. Includes page navigation and zoom.
- Preserves detail by tiling each page for inference and stitching overlays back together.
- Shows your drawn boxes (dashed) plus model overlays per page; results selectable via dropdown.

## High-Level Flow
1) **Upload**  
   - Images go directly to the canvas.  
   - PDF/TIFF are converted server‑side; first page preview is returned along with a `doc_id` and per‑page sizes.
2) **Draw / Prompt**  
   - Bounding boxes normalized to drawing page dimensions; optional text prompt.
3) **Segment**  
   - Frontend POSTs `/segment` with boxes/prompt and `doc_id` (if doc).  
   - Backend loads model lazily, then processes each page via tiled inference to keep resolution and catch small objects.
4) **Results**  
   - Per‑page overlays returned; UI lets you pick pages to view masks.  
   - Drawn boxes are re‑rendered in dashed strokes on every result page.

## Backend Components (app.py)
- **Endpoints**
  - `GET /` renders UI.
  - `POST /convert` converts uploads. For PDF/TIFF it caches a `Document` (pages rendered via PyMuPDF or PIL frames) and returns preview + `doc_id`.
  - `GET /page/<doc_id>/<int:page_index>` fetches a specific page PNG (used for navigation).
  - `POST /segment` runs segmentation on single image or all pages of a cached doc.
- **Document Cache**
  - `DocumentStore` holds `Document` objects with renderers and per‑page sizes.
- **Image helpers**
  - `decode_base64_image`, `encode_image_to_base64`, `overlay_masks_on_image`.
  - `process_page_with_tiling`: splits page into overlapping tiles (default 1200 px tiles, 128 px overlap), filters boxes per tile, runs SAM3 on each tile, stitches overlays, accumulates scores/colors.
- **Segmentation flow**
  - Normalizes user boxes from drawing page size to page size.
  - For each page: render page → select per‑page boxes → tiled inference → overlay masks → return per‑page payload.
  - Text prompt path uses same tiling (runs prompt on each tile).
- **Error handling**
  - 64 MB upload cap with 413 handler.
  - Safe guards for missing doc/page and missing inputs.
- **Dependencies**
  - `pymupdf` for PDFs; `Pillow`, `numpy`, `torch`, `flask`.

## Model Processor Notes (sam3/model/sam3_image_processor.py)
- Added `set_boxes` helper:
  - Requires `set_image` first.
  - Accepts `boxes` or `boxes_xyxy` (pixel xyxy), converts to normalized cxcywh.
  - Injects dummy text prompt (“visual”) if no language features set.
  - Shapes match SAM3 geometry prompt expectations: boxes `(N, B, 4)`, labels `(N, B)`.
- Resets prompts per call to avoid state leakage between requests.

## Frontend Components (templates/static)
- **UI (index.html + style.css)**
  - Upload, mode toggle (box/text), run, download.
  - Zoom controls; page controls (prev/next + label); loading overlay; results section with page dropdown.
- **JS (static/js/app.js)**
  - State: canvas, boxes, doc meta (`docId`, `pageCount`), current page indices (drawing/result), zoom, saved normalized boxes for overlays.
  - Upload handler:
    - Routes PDF/TIFF to `/convert`; stores `docMeta`, first page preview.
    - Images use FileReader path.
  - Page navigation:
    - `changePage` → `loadDocPage` (fetches `/page/<doc_id>/<page_index>`), resets boxes/results, redraws.
  - Segmentation:
    - Scales boxes to drawing page size; stores normalized boxes; sends `doc_id`, drawing page index/size, boxes or prompt to `/segment`.
  - Results rendering:
    - Multi‑page: populates dropdown, resizes canvas per result page, draws model overlay + dashed saved boxes, shows mask list.
    - Single‑page: same overlay logic.
  - Zoom: CSS scale with scrollable container; reset on load/page change.

## Tiling Details
- Purpose: keep pixel fidelity and detect small objects by avoiding global downscale.
- Defaults: 1200 px tiles, 128 px overlap to reduce seam artifacts.
- Box filtering per tile: only boxes intersecting a tile are run on that tile; coords adjusted to tile space.
- Stitching: tile overlays written back into a copy of the original image array.
- Tradeoff: more tiles = slower but higher recall for small objects; adjust `tile_size`/`overlap` in `process_page_with_tiling` if needed.

## Behavior/Assumptions
- Boxes are interpreted in xyxy pixel coords on the page where you drew them; normalized to apply across all pages.
- Results are per page; UI dropdown shows all pages processed, not just the first.
- Downloads save the currently displayed canvas (including masks and your dashed boxes).
- Upload size limit 64 MB; PDFs rendered ~300 DPI capped at 8k max side to balance quality/memory.

## Quick How-To
1) Start server: `python app.py`.
2) Upload PDF/TIFF (or image). Use Prev/Next to pick a page, zoom as needed.
3) Draw boxes (or enter text prompt) and run segmentation.
4) Use results dropdown to view each page with model masks; dashed boxes show your inputs.
5) Download saves the displayed page result.

## Risks / Next Steps (if you iterate)
- Tiling cost: large docs with many tiles can be slow; consider batching or adaptive tiling.
- Memory: high DPI + many tiles may grow RAM; monitor if bumping `tile_size` or DPI limits.
- Cache: `DocumentStore` is in‑memory; add eviction for long‑running servers.
- Coordinate integrity: we normalize on drawing page; if future UI changes rescale before drawing, revalidate normalization.
