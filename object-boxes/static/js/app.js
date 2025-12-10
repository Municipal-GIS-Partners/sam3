// SAM3 Bounding Box UI - Frontend JavaScript

class BoundingBoxApp {
    constructor() {
        this.canvas = document.getElementById('imageCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.image = null;
        this.originalImageSize = null;
        this.docMeta = null; // { docId, docType, pageCount }
        this.displayImageSize = null;
        this.boxes = [];
        this.currentBox = null;
        this.isDrawing = false;
        this.mode = 'box';
        this.resultImage = null;
        this.resultPages = null;
        this.currentResultPageIndex = 0;
        this.maskDetailsContainer = null;
        this.zoom = 1;
        this.minZoom = 0.5;
        this.maxZoom = 3;
        this.currentDocPageIndex = 0;
        this.pageCount = 1;
        this.savedBoxesNormalized = null; // normalized to drawing page dims
        this.savedBoxesMode = 'box';

        this.initializeEventListeners();
        this.updatePageControls();
        console.log('[INIT] BoundingBoxApp initialized', {
            canvasFound: !!this.canvas,
            contextFound: !!this.ctx
        });
    }

    initializeEventListeners() {
        // File upload
        document.getElementById('imageUpload').addEventListener('change', (e) => this.handleImageUpload(e));
        document.getElementById('clearImage').addEventListener('click', () => this.clearImage());

        // Mode toggle
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => this.handleModeChange(e.target.value));
        });

        // Box controls
        document.getElementById('clearBoxes').addEventListener('click', () => this.clearAllBoxes());
        document.getElementById('undoBox').addEventListener('click', () => this.undoLastBox());

        // Canvas drawing
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));

        // Run segmentation
        document.getElementById('runSegmentation').addEventListener('click', () => this.runSegmentation());

        // Download result
        document.getElementById('downloadResult').addEventListener('click', () => this.downloadResult());

        // Zoom controls
        document.getElementById('zoomIn').addEventListener('click', () => this.changeZoom(0.2));
        document.getElementById('zoomOut').addEventListener('click', () => this.changeZoom(-0.2));
        document.getElementById('zoomReset').addEventListener('click', () => this.setZoom(1));
        this.applyZoomTransform();
        this.updateZoomLabel();

        // Page controls
        document.getElementById('prevPage').addEventListener('click', () => this.changePage(-1));
        document.getElementById('nextPage').addEventListener('click', () => this.changePage(1));
    }

    async handleImageUpload(e) {
        console.log('[UPLOAD] Image upload triggered');
        const file = e.target.files[0];
        if (!file) {
            console.log('[UPLOAD] No file selected');
            return;
        }

        console.log('[UPLOAD] File selected', {
            name: file.name,
            sizeKb: +(file.size / 1024).toFixed(1),
            type: file.type
        });

        const lowerName = file.name.toLowerCase();
        const isTiff = (file.type && file.type.toLowerCase().includes('tif')) ||
            lowerName.endsWith('.tif') ||
            lowerName.endsWith('.tiff');
        const isPdf = (file.type && file.type.toLowerCase().includes('pdf')) ||
            lowerName.endsWith('.pdf');

        if (isTiff || isPdf) {
            console.warn('[UPLOAD] Document detected, sending to backend for conversion', {
                isTiff,
                isPdf
            });
            this.docMeta = null;
            this.resultPages = null;
            this.currentDocPageIndex = 0;
            await this.loadImageViaConvertEndpoint(file, isPdf ? 'pdf' : 'tiff');
            return;
        }

        this.docMeta = null;
        this.resultPages = null;

        const reader = new FileReader();
        let objectUrl = null;
        let triedObjectUrl = false;

        reader.onload = (event) => {
            console.log('[UPLOAD] File read successfully, creating image object');
            const img = new Image();
            img.onload = () => {
                this.applyLoadedImage(img);
                if (objectUrl) {
                    URL.revokeObjectURL(objectUrl);
                }
            };
            img.onerror = (error) => {
                console.error('[UPLOAD] Error loading image', {
                    error,
                    srcPreview: (img.src || '').slice(0, 80),
                    fileType: file.type,
                    fileSizeKb: +(file.size / 1024).toFixed(1),
                    triedObjectUrl
                });
                if (!triedObjectUrl) {
                    triedObjectUrl = true;
                    objectUrl = URL.createObjectURL(file);
                    console.warn('[UPLOAD] Retrying with object URL fallback');
                    img.src = objectUrl;
                    return;
                }
                alert('Error loading image. Please try a different file.');
            };
            const result = event.target?.result;
            if (!result || typeof result !== 'string') {
                console.error('[UPLOAD] FileReader result missing or not a string', { resultType: typeof result });
                alert('Could not read file. Please try again.');
                return;
            }
            console.log('[UPLOAD] Setting image src from data URL', {
                dataUrlPrefix: result.slice(0, 30),
                dataUrlLength: result.length
            });
            img.src = result;
        };
        reader.onerror = (error) => {
            console.error('[UPLOAD] Error reading file:', error);
            alert('Error reading file. Please try again.');
        };
        reader.readAsDataURL(file);
    }

    setupCanvas() {
        if (!this.image) return;

        // Set canvas size to match image
        // Allow large canvases; scrolling + zoom handles navigation.
        const maxWidth = 3000;
        const maxHeight = 3000;
        const baseWidth = this.displayImageSize?.width || this.image.width;
        const baseHeight = this.displayImageSize?.height || this.image.height;
        let width = baseWidth;
        let height = baseHeight;

        // Scale down if too large
        if (width > maxWidth || height > maxHeight) {
            const ratio = Math.min(maxWidth / width, maxHeight / height);
            width = width * ratio;
            height = height * ratio;
        }

        this.canvas.width = width;
        this.canvas.height = height;

        console.log('[CANVAS] Setup with image', {
            original: { width: baseWidth, height: baseHeight },
            scaled: { width, height }
        });
        this.drawImage();
        this.applyZoomTransform();
    }

    applyLoadedImage(img, providedWidth, providedHeight, referenceWidth, referenceHeight) {
        console.log('[UPLOAD] Image loaded', {
            naturalWidth: img.naturalWidth,
            naturalHeight: img.naturalHeight
        });
        this.image = img;
        const displayWidth = providedWidth || img.naturalWidth || img.width;
        const displayHeight = providedHeight || img.naturalHeight || img.height;
        this.displayImageSize = { width: displayWidth, height: displayHeight };
        this.originalImageSize = {
            width: referenceWidth || displayWidth,
            height: referenceHeight || displayHeight
        };
        this.setupCanvas();
        this.clearAllBoxes();
        this.hideResults();
        this.setZoom(1);
        console.log('[UPLOAD] Canvas setup complete', {
            canvasWidth: this.canvas.width,
            canvasHeight: this.canvas.height
        });
    }

    drawImage() {
        if (!this.image) {
            console.warn('[CANVAS] drawImage called with no image loaded');
            return;
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);

        // Draw all boxes
        if (this.mode === 'box') {
            this.boxes.forEach((box, idx) => {
                this.drawBox(box, this.getBoxColor(idx));
            });

            // Draw current box being drawn
            if (this.currentBox) {
                this.drawBox(this.currentBox, '#00ff00', true);
            }
        }
    }

    drawBox(box, color, isDashed = false) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;

        if (isDashed) {
            this.ctx.setLineDash([5, 5]);
        } else {
            this.ctx.setLineDash([]);
        }

        const width = box.x2 - box.x1;
        const height = box.y2 - box.y1;
        this.ctx.strokeRect(box.x1, box.y1, width, height);

        // Draw semi-transparent fill
        this.ctx.fillStyle = color + '20';
        this.ctx.fillRect(box.x1, box.y1, width, height);
    }

    getBoxColor(idx) {
        const colors = [
            '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff',
            '#00ffff', '#ff8800', '#8800ff', '#00ff88', '#ff0088'
        ];
        return colors[idx % colors.length];
    }

    getCanvasCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    handleMouseDown(e) {
        if (!this.image || this.mode !== 'box') return;

        const coords = this.getCanvasCoordinates(e);
        this.isDrawing = true;
        this.currentBox = {
            x1: coords.x,
            y1: coords.y,
            x2: coords.x,
            y2: coords.y
        };
    }

    handleMouseMove(e) {
        if (!this.isDrawing || !this.currentBox) return;

        const coords = this.getCanvasCoordinates(e);
        this.currentBox.x2 = coords.x;
        this.currentBox.y2 = coords.y;

        this.drawImage();
    }

    handleMouseUp(e) {
        if (!this.isDrawing || !this.currentBox) return;

        this.isDrawing = false;

        // Normalize box coordinates (ensure x1 < x2 and y1 < y2)
        const box = {
            x1: Math.min(this.currentBox.x1, this.currentBox.x2),
            y1: Math.min(this.currentBox.y1, this.currentBox.y2),
            x2: Math.max(this.currentBox.x1, this.currentBox.x2),
            y2: Math.max(this.currentBox.y1, this.currentBox.y2)
        };

        // Only add if box has some area
        const width = box.x2 - box.x1;
        const height = box.y2 - box.y1;
        if (width > 5 && height > 5) {
            this.boxes.push(box);
            this.updateBoxList();
        }

        this.currentBox = null;
        this.drawImage();
    }

    updateBoxList() {
        const boxList = document.getElementById('boxList');
        boxList.innerHTML = '';

        this.boxes.forEach((box, idx) => {
            const item = document.createElement('div');
            item.className = 'box-item';

            const colorDiv = document.createElement('div');
            colorDiv.className = 'box-color';
            colorDiv.style.backgroundColor = this.getBoxColor(idx);

            const coords = document.createElement('div');
            coords.className = 'box-coords';
            coords.textContent = `Box ${idx + 1}: (${Math.round(box.x1)}, ${Math.round(box.y1)}) - (${Math.round(box.x2)}, ${Math.round(box.y2)})`;

            const remove = document.createElement('div');
            remove.className = 'box-remove';
            remove.textContent = '×';
            remove.onclick = () => this.removeBox(idx);

            item.appendChild(colorDiv);
            item.appendChild(coords);
            item.appendChild(remove);
            boxList.appendChild(item);
        });
    }

    removeBox(idx) {
        this.boxes.splice(idx, 1);
        this.updateBoxList();
        this.drawImage();
    }

    clearAllBoxes() {
        this.boxes = [];
        this.updateBoxList();
        this.drawImage();
    }

    undoLastBox() {
        if (this.boxes.length > 0) {
            this.boxes.pop();
            this.updateBoxList();
            this.drawImage();
        }
    }

    clearImage() {
        this.image = null;
        this.docMeta = null;
        this.resultPages = null;
        this.displayImageSize = null;
        this.currentDocPageIndex = 0;
        this.clearAllBoxes();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        document.getElementById('imageUpload').value = '';
        this.hideResults();
        this.setZoom(1);
        this.updatePageControls();
    }

    handleModeChange(mode) {
        this.mode = mode;

        if (mode === 'box') {
            document.getElementById('boxControls').style.display = 'block';
            document.getElementById('textControls').style.display = 'none';
        } else {
            document.getElementById('boxControls').style.display = 'none';
            document.getElementById('textControls').style.display = 'block';
        }

        this.drawImage();
    }

    async runSegmentation() {
        if (!this.image) {
            alert('Please upload an image first');
            return;
        }

        if (this.mode === 'box' && this.boxes.length === 0) {
            alert('Please draw at least one bounding box');
            return;
        }

        if (this.mode === 'text' && !document.getElementById('textPrompt').value.trim()) {
            alert('Please enter a text prompt');
            return;
        }

        // Show loading
        document.getElementById('loadingOverlay').style.display = 'flex';

        try {
            // Get image as base64 (only needed for single-image mode)
            const imageData = this.docMeta?.docId ? null : this.canvas.toDataURL('image/png');

            // Scale boxes back to original image dimensions
            const baseWidth = this.originalImageSize?.width || this.image.width;
            const baseHeight = this.originalImageSize?.height || this.image.height;
            const scaleX = baseWidth / this.canvas.width;
            const scaleY = baseHeight / this.canvas.height;

            const scaledBoxes = this.boxes.map(box => [
                box.x1 * scaleX,
                box.y1 * scaleY,
                box.x2 * scaleX,
                box.y2 * scaleY
            ]);
            this.savedBoxesNormalized = scaledBoxes.map(b => [
                b[0] / baseWidth,
                b[1] / baseHeight,
                b[2] / baseWidth,
                b[3] / baseHeight
            ]);
            this.savedBoxesMode = this.mode;
            const drawingPageIndex = this.docMeta?.docId ? this.currentDocPageIndex : 0;

            const requestData = {
                image: imageData,
                boxes: this.mode === 'box' ? scaledBoxes : [],
                prompt: this.mode === 'text' ? document.getElementById('textPrompt').value : '',
                doc_id: this.docMeta?.docId || null,
                drawing_page_index: drawingPageIndex,
                drawing_page_width: this.originalImageSize?.width,
                drawing_page_height: this.originalImageSize?.height
            };

            const response = await fetch('/segment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Segmentation failed');
            }

            const result = await response.json();
            this.displayResults(result);

        } catch (error) {
            alert('Error: ' + error.message);
            console.error(error);
        } finally {
            document.getElementById('loadingOverlay').style.display = 'none';
        }
    }

    displayResults(result) {
        const resultsInfo = document.getElementById('resultsInfo');
        resultsInfo.innerHTML = '';

        const isMultiPage = Array.isArray(result.pages);
        if (isMultiPage) {
            this.resultPages = result.pages;
            this.currentResultPageIndex = 0;

            const summary = document.createElement('div');
            summary.innerHTML = `
                <p><strong>Segmentation Complete!</strong></p>
                <p>Pages processed: ${result.page_count || result.pages.length}</p>
            `;
            resultsInfo.appendChild(summary);

            const selector = document.createElement('select');
            selector.id = 'resultPageSelect';
            this.resultPages.forEach((page, idx) => {
                const option = document.createElement('option');
                option.value = idx;
                option.textContent = `Page ${idx + 1} (${page.num_masks} masks)`;
                selector.appendChild(option);
            });
            selector.addEventListener('change', (e) => {
                const idx = parseInt(e.target.value, 10) || 0;
                this.currentResultPageIndex = idx;
                this.showResultPage(idx);
            });

            const selectorContainer = document.createElement('div');
            selectorContainer.style.marginBottom = '10px';
            selectorContainer.appendChild(selector);
            resultsInfo.appendChild(selectorContainer);

            const maskContainer = document.createElement('div');
            maskContainer.id = 'maskDetails';
            resultsInfo.appendChild(maskContainer);
            this.maskDetailsContainer = maskContainer;

            this.showResultPage(this.currentResultPageIndex);
        } else {
            // Single image path (legacy)
            this.resultPages = null;
            const img = new Image();
            img.onload = () => {
                this.resultImage = img;
                this.canvas.width = img.width;
                this.canvas.height = img.height;
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
                this.drawSavedBoxesOverlay(img.width, img.height);
            };
            img.src = result.result_image;

            resultsInfo.innerHTML = `
                <p><strong>Segmentation Complete!</strong></p>
                <p>Number of masks: ${result.num_masks}</p>
            `;

            this.appendMaskDetails(resultsInfo, result);
        }

        document.getElementById('resultsSection').style.display = 'block';
    }

    showResultPage(idx) {
        if (!this.resultPages || !this.resultPages[idx]) return;
        const page = this.resultPages[idx];
        const selector = document.getElementById('resultPageSelect');
        if (selector && selector.value !== String(idx)) {
            selector.value = String(idx);
        }
        const img = new Image();
        img.onload = () => {
            this.resultImage = img;
            this.canvas.width = page.width || img.width;
            this.canvas.height = page.height || img.height;
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
            this.drawSavedBoxesOverlay(this.canvas.width, this.canvas.height);
            if (this.maskDetailsContainer) {
                this.maskDetailsContainer.innerHTML = `
                    <p><strong>Page ${idx + 1} of ${this.resultPages.length}</strong></p>
                    <p>Masks on this page: ${page.num_masks}</p>
                `;
                this.appendMaskDetails(this.maskDetailsContainer, page);
            }
        };
        img.src = page.result_image;
    }

    appendMaskDetails(container, result) {
        if (!container) return;
        // Clear any existing mask items except the header paragraphs already set.
        const existingItems = container.querySelectorAll('.mask-item');
        existingItems.forEach(item => item.remove());

        if (result.colors && result.scores) {
            result.colors.forEach((color, idx) => {
                const maskItem = document.createElement('div');
                maskItem.className = 'mask-item';

                const colorDiv = document.createElement('div');
                colorDiv.className = 'mask-color';
                colorDiv.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

                const info = document.createElement('div');
                info.className = 'mask-info';
                info.textContent = `Mask ${idx + 1}: Score ${result.scores[idx].toFixed(3)}`;

                maskItem.appendChild(colorDiv);
                maskItem.appendChild(info);
                container.appendChild(maskItem);
            });
        }
    }

    hideResults() {
        this.resultPages = null;
        this.maskDetailsContainer = null;
        const resultsInfo = document.getElementById('resultsInfo');
        resultsInfo.innerHTML = '';
        document.getElementById('resultsSection').style.display = 'none';
    }

    changePage(delta) {
        if (!this.docMeta || !this.docMeta.pageCount) return;
        const nextIndex = this.currentDocPageIndex + delta;
        if (nextIndex < 0 || nextIndex >= this.docMeta.pageCount) return;
        this.loadDocPage(nextIndex);
    }

    updatePageControls() {
        const controls = document.getElementById('pageControls');
        const label = document.getElementById('pageLabel');
        if (!controls || !label) return;

        if (this.docMeta && this.docMeta.pageCount > 1) {
            controls.style.display = 'flex';
            label.textContent = `Page ${this.currentDocPageIndex + 1}/${this.docMeta.pageCount}`;
        } else {
            controls.style.display = 'none';
        }
    }

    async loadDocPage(pageIndex) {
        if (!this.docMeta || !this.docMeta.docId) return;
        try {
            document.getElementById('loadingOverlay').style.display = 'flex';
            console.log('[DOC] Fetching page', pageIndex + 1);
            const resp = await fetch(`/page/${this.docMeta.docId}/${pageIndex}`);
            if (!resp.ok) {
                const error = await resp.json().catch(() => ({}));
                throw new Error(error.error || 'Failed to load page');
            }
            const data = await resp.json();
            this.currentDocPageIndex = pageIndex;
            this.pageCount = data.page_count || this.docMeta.pageCount;
            this.clearAllBoxes();
            this.hideResults();
            await this.setImageFromDataUrl(
                data.image_data,
                data.width,
                data.height,
                data.reference_width,
                data.reference_height
            );
            this.setZoom(1);
            this.updatePageControls();
        } catch (e) {
            console.error('[DOC] Page load failed', e);
            alert(e.message || 'Failed to load page');
        } finally {
            document.getElementById('loadingOverlay').style.display = 'none';
        }
    }

    drawSavedBoxesOverlay(targetWidth, targetHeight) {
        if (!this.savedBoxesNormalized || this.savedBoxesMode !== 'box') return;
        const ctx = this.ctx;
        ctx.save();
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        this.savedBoxesNormalized.forEach((b, idx) => {
            const x1 = b[0] * targetWidth;
            const y1 = b[1] * targetHeight;
            const x2 = b[2] * targetWidth;
            const y2 = b[3] * targetHeight;
            ctx.strokeStyle = this.getBoxColor(idx);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        });
        ctx.restore();
    }

    changeZoom(delta) {
        const next = this.zoom + delta;
        this.setZoom(next);
    }

    setZoom(value) {
        const clamped = Math.min(this.maxZoom, Math.max(this.minZoom, value));
        this.zoom = clamped;
        this.applyZoomTransform();
        this.updateZoomLabel();
    }

    applyZoomTransform() {
        if (!this.canvas) return;
        this.canvas.style.transformOrigin = 'top left';
        this.canvas.style.transform = `scale(${this.zoom})`;
    }

    updateZoomLabel() {
        const label = document.getElementById('zoomLabel');
        if (label) {
            label.textContent = `${Math.round(this.zoom * 100)}%`;
        }
    }

    downloadResult() {
        if (!this.resultImage) return;

        const link = document.createElement('a');
        link.download = 'sam3_segmentation_result.png';
        link.href = this.canvas.toDataURL('image/png');
        link.click();
    }

    async loadImageViaConvertEndpoint(file, docTypeHint = null) {
        try {
            this.docMeta = null;
            this.resultPages = null;
            document.getElementById('loadingOverlay').style.display = 'flex';
            const formData = new FormData();
            formData.append('file', file);

            console.log('[UPLOAD] Sending file to backend for conversion', {
                name: file.name,
                type: file.type,
                sizeKb: +(file.size / 1024).toFixed(1),
                docTypeHint
            });

            const response = await fetch('/convert', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                if (response.status === 413) {
                    throw new Error(error.error || 'File too large (max 64MB). Please upload a smaller document.');
                }
                throw new Error(error.error || `Convert failed with status ${response.status}`);
            }

            const data = await response.json();
            console.log('[UPLOAD] Conversion succeeded', {
                width: data.width,
                height: data.height,
                imageLength: data.image_data?.length,
                docId: data.doc_id,
                docType: data.doc_type,
                pageCount: data.page_count
            });

            this.docMeta = {
                docId: data.doc_id,
                docType: data.doc_type,
                pageCount: data.page_count
            };
            this.resultPages = null;
            this.currentDocPageIndex = data.page_index ?? 0;
            this.pageCount = data.page_count || 1;
            await this.setImageFromDataUrl(
                data.image_data,
                data.width,
                data.height,
                data.reference_width,
                data.reference_height
            );
            this.setZoom(1);
            this.updatePageControls();
        } catch (error) {
            console.error('[UPLOAD] Document conversion failed', error);
            alert(error.message || 'Could not load document. Please try another file.');
        } finally {
            document.getElementById('loadingOverlay').style.display = 'none';
        }
    }

    async setImageFromDataUrl(dataUrl, displayWidth, displayHeight, referenceWidth, referenceHeight) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.applyLoadedImage(img, displayWidth, displayHeight, referenceWidth, referenceHeight);
                resolve();
            };
            img.onerror = (error) => {
                console.error('[UPLOAD] Failed to load image from data URL', error);
                reject(error);
            };
            img.src = dataUrl;
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BoundingBoxApp();
});
