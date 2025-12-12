// SAM3 Video Text-Prompt Segmentation - Frontend JavaScript

// Application state
const appState = {
    sessionId: null,
    videoInfo: null,
    frames: [],
    results: [],
    currentFrameIndex: 0
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('[APP] Initializing...');
    initializeEventListeners();
});

function initializeEventListeners() {
    // Upload button
    document.getElementById('uploadButton').addEventListener('click', () => {
        document.getElementById('videoUpload').click();
    });

    document.getElementById('videoUpload').addEventListener('change', handleVideoUpload);

    // Slider updates
    document.getElementById('frameInterval').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value).toFixed(1);
        document.getElementById('intervalValue').textContent = value;
        document.getElementById('intervalValueText').textContent = value;
    });

    document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value).toFixed(2);
        document.getElementById('thresholdValue').textContent = value;
    });

    // Process button
    document.getElementById('processButton').addEventListener('click', handleProcessVideo);

    // Navigation buttons
    document.getElementById('prevFrame').addEventListener('click', prevFrame);
    document.getElementById('nextFrame').addEventListener('click', nextFrame);

    // Download buttons
    document.getElementById('downloadCurrent').addEventListener('click', downloadCurrentFrame);
    document.getElementById('downloadAll').addEventListener('click', downloadAllFrames);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (appState.results.length === 0) return;

        if (e.key === 'ArrowLeft') {
            prevFrame();
        } else if (e.key === 'ArrowRight') {
            nextFrame();
        }
    });
}

// ============================================================================
// Video Upload
// ============================================================================

async function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    console.log('[UPLOAD] Selected file:', file.name);

    try {
        showProgress('Uploading video...', 10);

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        await handleError(response);
        const data = await response.json();

        console.log('[UPLOAD] Upload successful:', data);

        // Update state
        appState.sessionId = data.session_id;
        appState.videoInfo = data.video_info;

        // Display video info
        displayVideoInfo(data.video_info);

        // Show controls section
        document.getElementById('controlsSection').classList.remove('hidden');
        hideProgress();

    } catch (error) {
        console.error('[UPLOAD] Error:', error);
        alert('Upload failed: ' + error.message);
        hideProgress();
    }
}

function displayVideoInfo(info) {
    const duration = formatTime(info.duration);
    document.getElementById('videoDuration').textContent = duration;
    document.getElementById('videoFps').textContent = info.fps.toFixed(2);
    document.getElementById('videoResolution').textContent = `${info.width}x${info.height}`;
    document.getElementById('totalFrames').textContent = info.total_frames;

    document.getElementById('videoInfo').classList.remove('hidden');
}

// ============================================================================
// Video Processing
// ============================================================================

async function handleProcessVideo() {
    const prompt = document.getElementById('textPrompt').value.trim();
    const interval = parseFloat(document.getElementById('frameInterval').value);
    const threshold = parseFloat(document.getElementById('confidenceThreshold').value);

    if (!prompt) {
        alert('Please enter a text prompt');
        return;
    }

    if (!appState.sessionId) {
        alert('Please upload a video first');
        return;
    }

    console.log('[PROCESS] Starting with prompt:', prompt);

    try {
        // Step 1: Extract frames
        showProgress('Extracting frames...', 20);

        const extractResponse = await fetch('/extract_frames', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: appState.sessionId,
                frame_interval_seconds: interval
            })
        });

        await handleError(extractResponse);
        const extractData = await extractResponse.json();
        appState.frames = extractData.frames;

        console.log('[PROCESS] Extracted', appState.frames.length, 'frames');

        // Step 2: Process with SAM3
        showProgress(`Segmenting ${appState.frames.length} frames...`, 50);

        const processResponse = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: appState.sessionId,
                text_prompt: prompt,
                confidence_threshold: threshold
            })
        });

        await handleError(processResponse);
        const processData = await processResponse.json();
        appState.results = processData.results;

        console.log('[PROCESS] Processed', appState.results.length, 'frames');

        showProgress('Complete!', 100);

        // Show results
        setTimeout(() => {
            displayResults();
        }, 500);

    } catch (error) {
        console.error('[PROCESS] Error:', error);
        alert('Processing failed: ' + error.message);
        hideProgress();
    }
}

// ============================================================================
// Results Display
// ============================================================================

function displayResults() {
    if (appState.results.length === 0) {
        alert('No results to display');
        return;
    }

    // Hide progress, show results section
    hideProgress();
    document.getElementById('resultsSection').classList.remove('hidden');

    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

    // Generate thumbnail grid
    generateThumbnailGrid();

    // Show first frame
    appState.currentFrameIndex = 0;
    showFrame(0);
}

function showFrame(index) {
    if (index < 0 || index >= appState.results.length) return;

    appState.currentFrameIndex = index;
    const result = appState.results[index];

    // Update images
    document.getElementById('originalImage').src =
        `/image/${appState.sessionId}/${result.frame_idx}/original`;
    document.getElementById('resultImage').src =
        `/image/${appState.sessionId}/${result.frame_idx}/result`;

    // Update info
    document.getElementById('frameCounter').textContent =
        `Frame ${index + 1} / ${appState.results.length}`;
    document.getElementById('timestamp').textContent = result.timestamp.toFixed(2);
    document.getElementById('maskCount').textContent = result.num_masks;

    // Update navigation buttons
    document.getElementById('prevFrame').disabled = (index === 0);
    document.getElementById('nextFrame').disabled = (index === appState.results.length - 1);

    // Highlight active thumbnail
    updateThumbnailHighlight(index);
}

function nextFrame() {
    if (appState.currentFrameIndex < appState.results.length - 1) {
        showFrame(appState.currentFrameIndex + 1);
    }
}

function prevFrame() {
    if (appState.currentFrameIndex > 0) {
        showFrame(appState.currentFrameIndex - 1);
    }
}

function generateThumbnailGrid() {
    const grid = document.getElementById('thumbnailGrid');
    grid.innerHTML = '';

    appState.results.forEach((result, index) => {
        const thumbnail = document.createElement('div');
        thumbnail.className = 'thumbnail';
        thumbnail.dataset.index = index;

        const img = document.createElement('img');
        img.src = `/image/${appState.sessionId}/${result.frame_idx}/result`;
        img.alt = `Frame ${index + 1}`;

        const label = document.createElement('div');
        label.className = 'thumbnail-label';
        label.textContent = `${result.timestamp.toFixed(1)}s`;

        thumbnail.appendChild(img);
        thumbnail.appendChild(label);

        thumbnail.addEventListener('click', () => {
            showFrame(index);
            // Scroll results into view
            document.getElementById('resultsSection').querySelector('.comparison-view')
                .scrollIntoView({ behavior: 'smooth', block: 'center' });
        });

        grid.appendChild(thumbnail);
    });
}

function updateThumbnailHighlight(activeIndex) {
    const thumbnails = document.querySelectorAll('.thumbnail');
    thumbnails.forEach((thumb, index) => {
        if (index === activeIndex) {
            thumb.classList.add('active');
        } else {
            thumb.classList.remove('active');
        }
    });
}

// ============================================================================
// Downloads
// ============================================================================

function downloadCurrentFrame() {
    if (!appState.sessionId || appState.currentFrameIndex < 0) return;

    const result = appState.results[appState.currentFrameIndex];
    const url = `/download/${appState.sessionId}/${result.frame_idx}`;

    window.open(url, '_blank');
}

function downloadAllFrames() {
    if (!appState.sessionId) return;

    const url = `/download_all/${appState.sessionId}`;
    window.open(url, '_blank');
}

// ============================================================================
// Utility Functions
// ============================================================================

function showProgress(message, percent) {
    const progressSection = document.getElementById('progressSection');
    progressSection.classList.remove('hidden');

    document.getElementById('progressText').textContent = message;
    document.getElementById('progressFill').style.width = percent + '%';

    // Scroll to progress
    progressSection.scrollIntoView({ behavior: 'smooth' });
}

function hideProgress() {
    document.getElementById('progressSection').classList.add('hidden');
}

async function handleError(response) {
    if (!response.ok) {
        let errorMessage = 'Unknown error';
        try {
            const error = await response.json();
            errorMessage = error.error || errorMessage;
        } catch (e) {
            errorMessage = response.statusText || errorMessage;
        }
        throw new Error(errorMessage);
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins}:${secs.padStart(4, '0')}`;
}

// ============================================================================
// Cleanup on page unload (optional)
// ============================================================================

window.addEventListener('beforeunload', () => {
    if (appState.sessionId) {
        // Note: This is a fire-and-forget request
        // The session will be cleaned up automatically after 24 hours anyway
        fetch(`/session/${appState.sessionId}`, {
            method: 'DELETE',
            keepalive: true
        }).catch(() => {
            // Ignore errors during cleanup
        });
    }
});
