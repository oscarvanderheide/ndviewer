import argparse
import asyncio
import io
import socket
import sys
import time
import threading
import webbrowser
from collections import OrderedDict

import numpy as np
import nibabel as nib
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import HTMLResponse
from PIL import Image
from matplotlib import colormaps as mpl_colormaps
import qmricolors  # registers lipari, navia colormaps with matplotlib  # noqa: F401

DATA = None
SHAPE = None
GLOBAL_STATS = {}  # {dr_idx: (vmin, vmax)} sampled once at startup

COLORMAPS = ["gray", "lipari", "navia", "viridis", "plasma", "RdBu_r"]
DR_PERCENTILES = [(0, 100), (1, 99), (5, 95), (10, 90)]
DR_LABELS = ["0-100%", "1-99%", "5-95%", "10-90%"]

# RGBA lookup tables (256 x 4)
LUTS = {
    name: np.concatenate(
        [
            (mpl_colormaps[name](np.arange(256) / 255.0) * 255).astype(np.uint8)[:, :3],
            np.full((256, 1), 255, dtype=np.uint8),
        ],
        axis=1,
    )
    for name in COLORMAPS
}

app = FastAPI()


def load_data(filepath):
    if filepath.endswith(".npy"):
        return np.load(filepath, mmap_mode="r")
    elif filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        img = nib.load(filepath)
        return img.dataobj
    elif filepath.endswith(".zarr") or filepath.endswith(".zarr.zip"):
        import zarr

        return zarr.open(filepath, mode="r")
    else:
        raise ValueError(
            "Unsupported format. Please provide a .npy, .nii/.nii.gz, or .zarr file"
        )


def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]
    while mshape[0] * mshape[1] < batch:
        mshape[1] += 1
    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1
    return tuple(mshape)


def _sample_for_stats(max_samples=200_000):
    """Sample data without loading the full array (mmap-friendly)."""
    total = int(np.prod(SHAPE))
    if total <= max_samples:
        sample = np.array(DATA).ravel()
    else:
        # Walk along dim 0 (sequential = mmap-friendly for C-contiguous arrays)
        n_take = min(10, SHAPE[0])
        step = max(1, SHAPE[0] // n_take)
        chunks = []
        for i in range(0, SHAPE[0], step):
            chunks.append(np.array(DATA[i]).ravel())
            if sum(c.size for c in chunks) >= max_samples:
                break
        sample = np.concatenate(chunks)
    if np.iscomplexobj(sample):
        sample = np.abs(sample)
    return np.nan_to_num(sample).astype(np.float32)


def compute_global_stats():
    global GLOBAL_STATS
    try:
        print("Computing global contrast statistics...", end="", flush=True)
        sample = _sample_for_stats()
        GLOBAL_STATS = {
            i: (float(np.percentile(sample, lo)), float(np.percentile(sample, hi)))
            for i, (lo, hi) in enumerate(DR_PERCENTILES)
        }
        print(f" done. ({len(sample):,} samples)")
    except Exception as e:
        print(f" failed ({e}), using per-slice stats.")
        GLOBAL_STATS = {}


# ---------------------------------------------------------------------------
# Two-level cache:
#  1. Raw float32 slice (LRU ~200 slices — avoids re-reading disk on colormap/DR change)
#  2. Rendered RGBA array (LRU, grown dynamically to hold the full active dim)
# ---------------------------------------------------------------------------
_raw_cache = OrderedDict()
_RAW_CACHE_MAX = 200

_rgba_cache = OrderedDict()
_RGBA_CACHE_MAX = 512

_mosaic_cache = OrderedDict()
_MOSAIC_CACHE_MAX = 64  # mosaics are large; 64 is plenty

# ---------------------------------------------------------------------------
# Background preload state
# ---------------------------------------------------------------------------
_preload_gen = 0  # increment to cancel the running preload thread
_preload_done = 0  # slices rendered so far
_preload_total = 0  # total slices in current preload
_preload_skipped = False  # True when array is too large to preload
_preload_lock = threading.Lock()


def extract_slice(dim_x, dim_y, idx_list):
    key = (dim_x, dim_y, tuple(idx_list))
    if key in _raw_cache:
        _raw_cache.move_to_end(key)
        return _raw_cache[key]

    slicer = [
        slice(None) if i in (dim_x, dim_y) else idx_list[i] for i in range(len(SHAPE))
    ]
    extracted = np.array(DATA[tuple(slicer)])
    if np.iscomplexobj(extracted):
        extracted = np.abs(extracted)
    if dim_x < dim_y:
        extracted = extracted.T
    result = np.nan_to_num(extracted).astype(np.float32)

    _raw_cache[key] = result
    if len(_raw_cache) > _RAW_CACHE_MAX:
        _raw_cache.popitem(last=False)
    return result


def apply_colormap_rgba(extracted, colormap, dr):
    """Map float32 array → RGBA uint8 array using precomputed global stats."""
    if dr in GLOBAL_STATS:
        vmin, vmax = GLOBAL_STATS[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(extracted, pct_lo))
        vmax = float(np.percentile(extracted, pct_hi))
    if vmax > vmin:
        normalized = np.clip((extracted - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(extracted)
    lut = LUTS.get(colormap, LUTS["gray"])
    return lut[(normalized * 255).astype(np.uint8)]  # (H, W, 4)


def render_rgba(dim_x, dim_y, idx_tuple, colormap, dr):
    """Return cached RGBA (H, W, 4) uint8 array."""
    key = (dim_x, dim_y, idx_tuple, colormap, dr)
    if key in _rgba_cache:
        _rgba_cache.move_to_end(key)
        return _rgba_cache[key]
    extracted = extract_slice(dim_x, dim_y, list(idx_tuple))
    rgba = apply_colormap_rgba(extracted, colormap, dr)
    _rgba_cache[key] = rgba
    if len(_rgba_cache) > _RGBA_CACHE_MAX:
        _rgba_cache.popitem(last=False)
    return rgba


def render_mosaic(dim_x, dim_y, dim_z, idx_tuple, colormap, dr):
    """Return cached RGBA mosaic of all dim_z slices."""
    idx_norm = list(idx_tuple)
    idx_norm[dim_z] = 0  # dim_z position in idx doesn't affect the mosaic
    key = (dim_x, dim_y, dim_z, tuple(idx_norm), colormap, dr)
    if key in _mosaic_cache:
        _mosaic_cache.move_to_end(key)
        return _mosaic_cache[key]

    n = SHAPE[dim_z]
    frames = [
        extract_slice(
            dim_x, dim_y, [i if j == dim_z else idx_tuple[j] for j in range(len(SHAPE))]
        )
        for i in range(n)
    ]
    all_data = np.stack(frames)  # (n, H, W)

    if dr in GLOBAL_STATS:
        vmin, vmax = GLOBAL_STATS[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    padded = np.zeros((rows * cols, H, W), dtype=np.float32)
    padded[:n] = all_data
    grid = (
        padded.reshape(rows, cols, H, W)
        .transpose(0, 2, 1, 3)
        .reshape(rows * H, cols * W)
    )

    if vmax > vmin:
        normalized = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(grid)

    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    _mosaic_cache[key] = rgba
    if len(_mosaic_cache) > _MOSAIC_CACHE_MAX:
        _mosaic_cache.popitem(last=False)
    return rgba


def _run_preload(gen, dim_x, dim_y, idx_list, colormap, dr, slice_dim, dim_z=-1):
    """Background thread: pre-render every slice of slice_dim into cache."""
    global _preload_done, _preload_total, _preload_skipped, _RGBA_CACHE_MAX

    n = SHAPE[slice_dim]
    H = SHAPE[dim_y]
    W = SHAPE[dim_x]
    if dim_z >= 0:
        nz = SHAPE[dim_z]
        mrows, mcols = mosaic_shape(nz)
        size_bytes = n * (mrows * H) * (mcols * W) * 4
    else:
        size_bytes = n * H * W * 4

    with _preload_lock:
        _preload_total = n
        _preload_done = 0
        if size_bytes > 500 * 1024 * 1024:
            _preload_skipped = True
            return
        _preload_skipped = False
        if dim_z < 0:
            _RGBA_CACHE_MAX = max(512, n * 4)

    for i in range(n):
        if _preload_gen != gen:
            return
        idx = list(idx_list)
        idx[slice_dim] = i
        if dim_z >= 0:
            render_mosaic(dim_x, dim_y, dim_z, tuple(idx), colormap, dr)
        else:
            render_rgba(dim_x, dim_y, tuple(idx), colormap, dr)
        with _preload_lock:
            _preload_done = i + 1


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # The receiver task continuously drains the WebSocket buffer and keeps only
    # the LATEST request. This is the key to dropping stale frames: by the time
    # the renderer finishes one slice, many keypresses may have arrived; we skip
    # all intermediate ones and jump straight to the most recent position.
    latest_msg: dict | None = None
    latest_seq: int = 0
    new_request = asyncio.Event()

    async def receiver():
        nonlocal latest_msg, latest_seq
        try:
            while True:
                msg = await ws.receive_json()
                seq = int(msg.get("seq", 0))
                if seq > latest_seq:
                    latest_seq = seq
                    latest_msg = msg
                    new_request.set()
        except Exception:
            new_request.set()  # wake processor so it can exit

    receiver_task = asyncio.create_task(receiver())
    try:
        while True:
            await new_request.wait()
            new_request.clear()
            if latest_msg is None:
                break

            msg = latest_msg
            seq = latest_seq
            dim_x = int(msg["dim_x"])
            dim_y = int(msg["dim_y"])
            idx_tuple = tuple(int(x) for x in msg["indices"])
            colormap = str(msg.get("colormap", "gray"))
            dr = int(msg.get("dr", 1))
            slice_dim = int(msg.get("slice_dim", -1))
            direction = int(msg.get("direction", 1))
            dim_z = int(msg.get("dim_z", -1))

            # Run blocking numpy work in a thread so the receiver stays live
            if dim_z >= 0:
                rgba = await loop.run_in_executor(
                    None, render_mosaic, dim_x, dim_y, dim_z, idx_tuple, colormap, dr
                )
            else:
                rgba = await loop.run_in_executor(
                    None, render_rgba, dim_x, dim_y, idx_tuple, colormap, dr
                )

            # Another request may have arrived while we were rendering — send
            # only if this is still the one the client is waiting for
            if seq == latest_seq:
                h, w = rgba.shape[:2]
                header = np.array([seq, w, h], dtype=np.uint32).tobytes()
                await ws.send_bytes(header + rgba.tobytes())

                # Warm the cache for the next few slices in the scroll direction
                # so the following keypresses are instant cache hits
                if 0 <= slice_dim < len(SHAPE):

                    def _prefetch(
                        dim_x=dim_x,
                        dim_y=dim_y,
                        idx_tuple=idx_tuple,
                        colormap=colormap,
                        dr=dr,
                        slice_dim=slice_dim,
                        direction=direction,
                        dim_z=dim_z,
                    ):
                        for i in range(1, 5):
                            nxt = idx_tuple[slice_dim] + direction * i
                            if 0 <= nxt < SHAPE[slice_dim]:
                                idx = list(idx_tuple)
                                idx[slice_dim] = nxt
                                if dim_z >= 0:
                                    render_mosaic(
                                        dim_x, dim_y, dim_z, tuple(idx), colormap, dr
                                    )
                                else:
                                    render_rgba(dim_x, dim_y, tuple(idx), colormap, dr)

                    loop.run_in_executor(None, _prefetch)

    except Exception:
        pass
    finally:
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass


@app.get("/clearcache")
def clear_cache():
    _raw_cache.clear()
    _rgba_cache.clear()
    _mosaic_cache.clear()
    return {"status": "ok"}


@app.post("/preload")
async def start_preload(request: Request):
    global _preload_gen
    body = await request.json()
    dim_x = int(body["dim_x"])
    dim_y = int(body["dim_y"])
    idx_list = [int(x) for x in body["indices"]]
    colormap = str(body.get("colormap", "gray"))
    dr = int(body.get("dr", 1))
    slice_dim = int(body["slice_dim"])
    dim_z = int(body.get("dim_z", -1))

    # Cancel any running preload and start a new one
    _preload_gen += 1
    gen = _preload_gen
    threading.Thread(
        target=_run_preload,
        args=(gen, dim_x, dim_y, idx_list, colormap, dr, slice_dim, dim_z),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.get("/preload_status")
def get_preload_status():
    with _preload_lock:
        return {
            "done": _preload_done,
            "total": _preload_total,
            "skipped": _preload_skipped,
        }


@app.get("/metadata")
def get_metadata():
    return {"shape": list(SHAPE)}


@app.get("/slice")
def get_slice(
    dim_x: int,
    dim_y: int,
    indices: str,
    colormap: str = "gray",
    dr: int = 1,
    slice_dim: int = -1,
):
    """HTTP fallback (used by nothing in the UI, kept for debugging)."""
    idx_tuple = tuple(int(x) for x in indices.split(","))
    rgba = render_rgba(dim_x, dim_y, idx_tuple, colormap, dr)
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=300"},
    )


@app.get("/grid")
def get_grid(
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    idx_list = [int(x) for x in indices.split(",")]
    n = SHAPE[slice_dim]
    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    if dr in GLOBAL_STATS:
        vmin, vmax = GLOBAL_STATS[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    padded = np.zeros((rows * cols, H, W), dtype=np.float32)
    padded[:n] = all_data
    mosaic = (
        padded.reshape(rows, cols, H, W)
        .transpose(0, 2, 1, 3)
        .reshape(rows * H, cols * W)
    )

    if vmax > vmin:
        normalized = np.clip((mosaic - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(mosaic)

    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    img = Image.fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/gif")
def get_gif(
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    idx_list = [int(x) for x in indices.split(",")]
    n = SHAPE[slice_dim]
    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    if dr in GLOBAL_STATS:
        vmin, vmax = GLOBAL_STATS[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    gif_frames = []
    for frame in frames:
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(frame)
        rgba = lut[(normalized * 255).astype(np.uint8)]
        gif_frames.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))

    buf = io.BytesIO()
    gif_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=100,
    )
    return Response(content=buf.getvalue(), media_type="image/gif")


@app.get("/")
def get_ui():
    html_content = (
        """<!DOCTYPE html>
<html>
<head>
    <title>NDViewer</title>
    <style>
        :root {
            --bg: #111; --surface: #1e1e1e; --border: #444;
            --text: #ccc; --muted: #777; --subtle: #444;
            --highlight: #fff; --canvas-border: #555;
        }
        html, body { background: transparent; margin: 0; padding: 0; }
        #wrapper {
            background: var(--bg); color: var(--text); font-family: monospace;
            display: inline-flex; flex-direction: column; align-items: center;
            padding: 16px 20px 20px; min-width: fit-content;
        }
        #wrapper.light {
            --bg: #f0f0f0; --surface: #e0e0e0; --border: #bbb;
            --text: #333; --muted: #888; --subtle: #bbb;
            --highlight: #000; --canvas-border: #999;
        }
        #info { margin-bottom: 20px; font-size: 16px; white-space: pre-wrap; text-align: left; background: var(--surface); padding: 15px; border-radius: 8px; border: 1px solid var(--border); width: fit-content; }
        canvas { border: 1px solid var(--canvas-border); image-rendering: pixelated; outline: none; cursor: crosshair; }
        .highlight { color: var(--highlight); font-weight: bold; }
        .muted { color: var(--muted); }
        #status { margin-top: 8px; font-size: 13px; color: var(--muted); min-height: 1.2em; }
        #preload-status { margin-top: 4px; font-size: 12px; color: var(--subtle); min-height: 1em; }
        #toast {
            margin-top: 8px; font-size: 13px; color: var(--text);
            min-height: 1.2em; opacity: 0; transition: opacity 0.8s ease;
        }
        #help-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.75); z-index: 10; justify-content: center; align-items: center;
        }
        #help-overlay.visible { display: flex; }
        #help-box {
            background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
            padding: 30px 40px; font-size: 15px; line-height: 2; color: var(--text);
            white-space: pre;
        }
        #help-box .key { color: var(--highlight); font-weight: bold; display: inline-block; min-width: 140px; }
    </style>
</head>
<body>
<div id="wrapper">
    <div id="info">Connecting...</div>
    <canvas id="viewer" tabindex="0"></canvas>
    <!-- Hidden textarea: VS Code passes all keys (including arrows) to focused text inputs,
         unlike other focusable elements where it intercepts navigation keys. -->
    <textarea id="keyboard-sink" autocomplete="off" autocorrect="off" spellcheck="false"
              style="position:fixed;top:-1px;left:-1px;width:1px;height:1px;opacity:0;border:none;padding:0;margin:0;resize:none;overflow:hidden;"></textarea>
    <div id="status"></div>
    <div id="toast"></div>
    <div id="preload-status"></div>
    <div id="help-overlay">
        <div id="help-box"><span class="key">j / ↓</span>  previous slice
<span class="key">k / ↑</span>  next slice
<span class="key">h / ←</span>  previous slice dimension
<span class="key">l / →</span>  next slice dimension
<span class="key">x</span>  swap horizontal dim with slice dim
<span class="key">y</span>  swap vertical dim with slice dim
<span class="key">Space</span>  toggle auto-play
<span class="key">z</span>  claim dim as z (grid), scroll through next dim
<span class="key">c</span>  cycle colormap
<span class="key">d</span>  cycle dynamic range
<span class="key">t</span>  toggle dark / light theme
<span class="key">s</span>  save screenshot (PNG)
<span class="key">g</span>  save GIF of current slice dim
<span class="key">+ / -</span>  zoom in / out
<span class="key">scroll</span>  scroll through slices
<span class="key">?</span>  toggle this help</div>
    </div>
</div>
    <script>
        const COLORMAPS = """
        + str(COLORMAPS)
        + """;
        const DR_LABELS = """
        + str(DR_LABELS)
        + """;

        let shape = [];
        let dim_x = 0, dim_y = 1, current_slice_dim = 2;
        let indices = [];
        let colormap_idx = 0, dr_idx = 1;
        let isPlaying = false, playInterval = null;
        let dim_z = -1;
        let lastDirection = 1;
        let isDark = true;

        // WebSocket state
        let ws = null, wsReady = false, wsSentSeq = 0;

        // Preload polling state
        let preloadPolling = null;
        let preloadActiveDim = -1;

        // Toast state
        let toastTimer = null;

        // Zoom state
        let userZoom = 0.6;

        const canvas = document.getElementById('viewer');
        const ctx = canvas.getContext('2d');
        const sink = document.getElementById('keyboard-sink');

        function scaleCanvas(w, h) {
            const maxW = window.innerWidth * 0.95;
            const maxH = window.innerHeight * 0.70;
            const scale = Math.min(maxW / w, maxH / h) * userZoom;
            canvas.style.width  = Math.round(w * scale) + 'px';
            canvas.style.height = Math.round(h * scale) + 'px';
        }

        function showToast(msg) {
            const el = document.getElementById('toast');
            el.textContent = msg;
            el.style.transition = 'none';
            el.style.opacity = '1';
            if (toastTimer) clearTimeout(toastTimer);
            toastTimer = setTimeout(() => {
                el.style.transition = 'opacity 0.8s ease';
                el.style.opacity = '0';
            }, 1500);
        }

        function initWebSocket() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${proto}//${location.host}/ws`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                wsReady = true;
                setStatus('');
                sink.focus();
                updateView();
            };

            ws.onmessage = (event) => {
                const buf = event.data;
                const header = new Uint32Array(buf, 0, 3);
                const seq    = header[0];
                const width  = header[1];
                const height = header[2];

                // Discard stale frames — only render the most recently requested seq
                if (seq !== wsSentSeq) return;

                const rgba = new Uint8ClampedArray(buf, 12);
                canvas.width  = width;
                canvas.height = height;
                ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
                scaleCanvas(width, height);
            };

            ws.onclose = () => {
                wsReady = false;
                setStatus('WebSocket closed — reconnecting...');
                setTimeout(initWebSocket, 1000);
            };

            ws.onerror = () => ws.close();
        }

        function triggerPreload() {
            if (shape.length < 3) return;  // nothing to scroll for 1D/2D arrays
            preloadActiveDim = current_slice_dim;
            fetch('/preload', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    dim_x, dim_y, dim_z,
                    indices: [...indices],
                    colormap: COLORMAPS[colormap_idx],
                    dr: dr_idx,
                    slice_dim: current_slice_dim,
                })
            });
            if (preloadPolling) clearInterval(preloadPolling);
            preloadPolling = setInterval(pollPreloadStatus, 500);
        }

        function pollPreloadStatus() {
            fetch('/preload_status').then(r => r.json()).then(data => {
                const el = document.getElementById('preload-status');
                if (data.skipped) {
                    el.textContent = 'Array too large for full preload (>500 MB) — using prefetch.';
                    clearInterval(preloadPolling); preloadPolling = null;
                } else if (data.total > 0 && data.done >= data.total) {
                    el.textContent = '';
                    clearInterval(preloadPolling); preloadPolling = null;
                } else if (data.total > 0) {
                    const pct = Math.round(data.done / data.total * 100);
                    el.textContent = `Preloading dim ${preloadActiveDim}: ${data.done}/${data.total} (${pct}%)`;
                }
            });
        }

        async function init() {
            const res = await fetch('/metadata');
            const data = await res.json();
            shape = data.shape;
            indices = shape.map(s => Math.floor(s / 2));
            dim_x = 0; dim_y = 1;
            current_slice_dim = shape.length > 2 ? 2 : 0;
            initWebSocket();  // calls updateView() on open
            triggerPreload();
        }

        function renderInfo() {
            const idxStr = indices.map((v, i) => {
                if (i === dim_x) return 'x';
                if (i === dim_y) return 'y';
                if (i === dim_z) return 'z';
                if (i === current_slice_dim) return `<span class="highlight">[${v}]</span>`;
                return v;
            }).join(', ');
            let text = `Shape: [${shape.join(', ')}]\\n`;
            text += `Index: [${idxStr}]`;
            text += `  <span class="muted">(? for help)</span>`;
            document.getElementById('info').innerHTML = text;
        }

        function setStatus(msg) { document.getElementById('status').textContent = msg; }

        function updateView() {
            renderInfo();
            if (!wsReady) return;
            wsSentSeq++;
            ws.send(JSON.stringify({
                seq: wsSentSeq,
                dim_x, dim_y, dim_z,
                indices: [...indices],
                colormap: COLORMAPS[colormap_idx],
                dr: dr_idx,
                slice_dim: current_slice_dim,
                direction: lastDirection,
            }));
        }

        function stopPlay() {
            clearInterval(playInterval); playInterval = null;
            isPlaying = false; setStatus('');
        }

        function togglePlay() {
            if (isPlaying) { stopPlay(); return; }
            isPlaying = true;
            setStatus('▶ playing  (Space to stop)');
            playInterval = setInterval(() => {
                indices[current_slice_dim] = (indices[current_slice_dim] + 1) % shape[current_slice_dim];
                updateView();
            }, 80);
        }

        function saveScreenshot() {
            const link = document.createElement('a');
            link.download = `slice_${indices.join('-')}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
            setStatus('Screenshot saved.');
            setTimeout(() => setStatus(''), 2000);
        }

        async function saveGif() {
            setStatus('Generating GIF...');
            const url = `/gif?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&colormap=${COLORMAPS[colormap_idx]}&dr=${dr_idx}&slice_dim=${current_slice_dim}`;
            const res = await fetch(url);
            const blob = await res.blob();
            const link = document.createElement('a');
            link.download = `dim${current_slice_dim}.gif`;
            link.href = URL.createObjectURL(blob);
            link.click();
            setStatus('GIF saved.');
            setTimeout(() => setStatus(''), 2000);
        }

        const helpOverlay = document.getElementById('help-overlay');

        // Re-focus the sink whenever the user clicks anywhere in the viewer.
        // The hidden textarea trick: VS Code passes all key events (including arrow keys
        // and h/l which it normally swallows for notebook navigation) to focused text
        // inputs, because it assumes those need full keyboard access for cursor movement.
        canvas.addEventListener('click', () => sink.focus());
        document.addEventListener('click', () => sink.focus());

        sink.addEventListener('keydown', (e) => {
            e.preventDefault();            // stop browser default (e.g. textarea scrolling)
            e.stopImmediatePropagation();  // stop other handlers
            if (e.key === '?') { helpOverlay.classList.toggle('visible'); return; }
            if (e.key === 'Escape') { helpOverlay.classList.remove('visible'); return; }
            if (e.key === '+' || e.key === '=') {
                userZoom = Math.min(userZoom * 1.02, 8.0);
                scaleCanvas(canvas.width, canvas.height);
                showToast(`zoom: ${Math.round(userZoom * 100)}%`);
            } else if (e.key === '-') {
                userZoom = Math.max(userZoom / 1.02, 0.1);
                scaleCanvas(canvas.width, canvas.height);
                showToast(`zoom: ${Math.round(userZoom * 100)}%`);
            } else if (e.key === 'z') {
                if (dim_z >= 0) {
                    // Exit z-mode: restore dim_z back as the scroll dim
                    current_slice_dim = dim_z;
                    dim_z = -1;
                } else {
                    // Enter z-mode: claim current scroll dim as z, advance scroll
                    if (shape.length < 4) return;  // need a free dim to scroll through
                    dim_z = current_slice_dim;
                    do { current_slice_dim = (current_slice_dim + 1) % shape.length; }
                    while (current_slice_dim === dim_x || current_slice_dim === dim_y || current_slice_dim === dim_z);
                }
                updateView(); triggerPreload();
            } else if (e.key === ' ') {
                e.preventDefault();
                togglePlay();
            } else if (e.key === 't') {
                isDark = !isDark;
                document.getElementById('wrapper').classList.toggle('light', !isDark);
            } else if (e.key === 's') {
                saveScreenshot();
            } else if (e.key === 'g') {
                saveGif();
            } else if (e.key === 'c') {
                colormap_idx = (colormap_idx + 1) % COLORMAPS.length;
                fetch('/clearcache'); updateView(); triggerPreload();
                showToast(`colormap: ${COLORMAPS[colormap_idx]}`);
            } else if (e.key === 'd') {
                dr_idx = (dr_idx + 1) % DR_LABELS.length;
                fetch('/clearcache'); updateView(); triggerPreload();
                showToast(`range: ${DR_LABELS[dr_idx]}`);
            } else if (e.key === 'j' || e.key === 'ArrowDown') {
                e.preventDefault();
                lastDirection = -1; indices[current_slice_dim] = Math.max(0, indices[current_slice_dim] - 1); updateView();
            } else if (e.key === 'k' || e.key === 'ArrowUp') {
                e.preventDefault();
                lastDirection = 1; indices[current_slice_dim] = Math.min(shape[current_slice_dim] - 1, indices[current_slice_dim] + 1); updateView();
            } else if (e.key === 'h' || e.key === 'ArrowLeft') {
                e.preventDefault();
                const minFree = dim_z >= 0 ? 3 : 2;
                do { current_slice_dim = (current_slice_dim - 1 + shape.length) % shape.length; }
                while ((current_slice_dim === dim_x || current_slice_dim === dim_y || current_slice_dim === dim_z) && shape.length > minFree);
                updateView(); triggerPreload();
            } else if (e.key === 'l' || e.key === 'ArrowRight') {
                e.preventDefault();
                const minFree = dim_z >= 0 ? 3 : 2;
                do { current_slice_dim = (current_slice_dim + 1) % shape.length; }
                while ((current_slice_dim === dim_x || current_slice_dim === dim_y || current_slice_dim === dim_z) && shape.length > minFree);
                updateView(); triggerPreload();
            } else if (e.key === 'x') {
                if (shape.length < 3) return;
                [dim_x, current_slice_dim] = [current_slice_dim, dim_x];
                dim_z = -1;
                updateView(); triggerPreload();
            } else if (e.key === 'y') {
                if (shape.length < 3) return;
                [dim_y, current_slice_dim] = [current_slice_dim, dim_y];
                dim_z = -1;
                updateView(); triggerPreload();
            }
        });

        helpOverlay.addEventListener('click', () => { helpOverlay.classList.remove('visible'); sink.focus(); });

        window.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY > 0) {
                lastDirection = -1;
                indices[current_slice_dim] = Math.max(0, indices[current_slice_dim] - 1);
            } else {
                lastDirection = 1;
                indices[current_slice_dim] = Math.min(shape[current_slice_dim] - 1, indices[current_slice_dim] + 1);
            }
            updateView();
        }, {passive: false});

        init();
    </script>
</body>
</html>"""
    )
    return HTMLResponse(content=html_content)


# ---------------------------------------------------------------------------
# Jupyter / in-process API
# ---------------------------------------------------------------------------
_jupyter_server_port: int | None = None  # port of the running background server


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        # ipykernel is used by Jupyter notebook, JupyterLab, and VS Code
        # interactive window — all kernel-based environments.
        return "ipykernel" in type(shell).__module__
    except ImportError:
        return False


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until the local TCP port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)


async def _serve_background(port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()


def _set_data(data):
    """Update the global DATA/SHAPE and flush all caches."""
    global DATA, SHAPE
    if isinstance(data, str):
        data = load_data(data)
    DATA = data
    SHAPE = data.shape
    _raw_cache.clear()
    _rgba_cache.clear()
    _mosaic_cache.clear()
    compute_global_stats()


def view(data, port: int = 8123, inline: bool | None = None, height: int = 500):
    """View an ND array inline in Jupyter or in a browser window.

    Parameters
    ----------
    data:
        NumPy array (or anything with ``.shape``) to view, or a file path string.
    port:
        Local port for the FastAPI server (default 8123).
    inline:
        ``True``  – embed an IFrame in the Jupyter cell output.
        ``False`` – open a browser window (blocking, like the CLI).
        ``None``  – auto-detect: inline when inside Jupyter, browser otherwise.
    height:
        IFrame height in pixels (inline mode only).

    Examples
    --------
    >>> import numpy as np
    >>> from ndviewer import view
    >>> view(np.random.rand(64, 64, 30))          # auto-detects Jupyter
    >>> view("scan.nii.gz", port=8124)            # new port for a second array
    """
    global _jupyter_server_port

    _set_data(data)

    if inline is None:
        inline = _in_jupyter()

    url = f"http://127.0.0.1:{port}"

    if inline:
        # Start a background server the first time (or when the port changes).
        if _jupyter_server_port != port:
            # Always use a daemon thread with its own event loop — never schedule
            # on Jupyter's event loop via ensure_future, which only runs after the
            # current cell finishes and would return the IFrame before the server
            # is ready.  The thread starts immediately and _wait_for_port blocks
            # here until the TCP socket is actually accepting connections, so the
            # IFrame is only returned once the server is guaranteed to be up.
            threading.Thread(
                target=lambda: asyncio.run(_serve_background(port)),
                daemon=True,
            ).start()
            _wait_for_port(port)
            _jupyter_server_port = port

        from IPython.display import IFrame  # only needed in Jupyter

        return IFrame(src=url, width="100%", height=height)
    else:
        # Blocking browser mode (same as CLI).
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    global DATA, SHAPE

    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument("file", help="Path to .npy, .nii/.nii.gz, or .zarr file")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    try:
        DATA = load_data(args.file)
        SHAPE = DATA.shape
        try:
            size_str = f" ({DATA.nbytes // 1024**2} MB)"
        except AttributeError:
            size_str = ""
        print(f"Loaded {args.file} with shape {SHAPE}{size_str}")
        compute_global_stats()
        print(f"Open http://127.0.0.1:{args.port} in your browser")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    url = f"http://127.0.0.1:{args.port}"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
