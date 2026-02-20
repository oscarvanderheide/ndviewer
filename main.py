
import argparse
import io
import sys
import numpy as np
import nibabel as nib
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from PIL import Image
from matplotlib import colormaps as mpl_colormaps

DATA = None
SHAPE = None

COLORMAPS = ['gray', 'viridis', 'plasma', 'RdBu_r']
DR_PERCENTILES = [(0, 100), (1, 99), (5, 95), (10, 90)]
DR_LABELS = ['0-100%', '1-99%', '5-95%', '10-90%']

app = FastAPI()

def load_data(filepath):
    if filepath.endswith('.npy'):
        return np.load(filepath, mmap_mode='r')
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        img = nib.load(filepath)
        return img.dataobj
    else:
        raise ValueError("Unsupported format. Please provide a .npy or .nii/.nii.gz file")

def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]
    while mshape[0] * mshape[1] < batch:
        mshape[1] += 1
    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1
    return tuple(mshape)

def extract_slice(dim_x, dim_y, idx_list):
    slicer = []
    for i in range(len(SHAPE)):
        if i == dim_x or i == dim_y:
            slicer.append(slice(None))
        else:
            slicer.append(idx_list[i])
    extracted = np.array(DATA[tuple(slicer)])
    if np.iscomplexobj(extracted):
        extracted = np.abs(extracted)
    if dim_x < dim_y:
        extracted = extracted.T
    return np.nan_to_num(extracted).astype(np.float64)

def apply_colormap(extracted, colormap, dr):
    pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
    vmin = np.percentile(extracted, pct_lo)
    vmax = np.percentile(extracted, pct_hi)
    if vmax > vmin:
        normalized = np.clip((extracted - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(extracted)
    cmap = mpl_colormaps[colormap if colormap in COLORMAPS else 'gray']
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    return rgba[:, :, :3]  # drop alpha

@app.get("/")
def get_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NDViewer</title>
        <style>
            body { background: #111; color: #00ffcc; font-family: monospace; display: flex; flex-direction: column; align-items: center; padding-top: 20px; margin:0;}
            #info { margin-bottom: 20px; font-size: 16px; white-space: pre-wrap; text-align: left; background: #222; padding: 15px; border-radius: 8px; border: 1px solid #444; width: 600px;}
            canvas { border: 1px solid #555; image-rendering: pixelated; }
            .highlight { color: #fff; font-weight: bold; }
            #status { margin-top: 8px; font-size: 13px; color: #888; min-height: 1.2em; }
            #help-overlay {
                display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.75); z-index: 10; justify-content: center; align-items: center;
            }
            #help-overlay.visible { display: flex; }
            #help-box {
                background: #222; border: 1px solid #555; border-radius: 10px;
                padding: 30px 40px; font-size: 15px; line-height: 2; color: #00ffcc;
                white-space: pre;
            }
            #help-box .key { color: #fff; font-weight: bold; display: inline-block; min-width: 140px; }
        </style>
    </head>
    <body>
        <div id="info">Loading...</div>
        <canvas id="viewer"></canvas>
        <div id="status"></div>
        <div id="help-overlay">
            <div id="help-box"><span class="key">j / ↓</span>  previous slice
<span class="key">k / ↑</span>  next slice
<span class="key">h / ←</span>  previous slice dimension
<span class="key">l / →</span>  next slice dimension
<span class="key">x</span>  swap horizontal dim with slice dim
<span class="key">y</span>  swap vertical dim with slice dim
<span class="key">Space</span>  toggle auto-play
<span class="key">z</span>  toggle grid (all slices mosaic)
<span class="key">c</span>  cycle colormap
<span class="key">d</span>  cycle dynamic range
<span class="key">s</span>  save screenshot (PNG)
<span class="key">g</span>  save GIF of current slice dim
<span class="key">scroll</span>  scroll through slices
<span class="key">?</span>  toggle this help</div>
        </div>
        <script>
            const COLORMAPS = """ + str(COLORMAPS) + """;
            const DR_LABELS = """ + str(DR_LABELS) + """;

            let shape = [];
            let dim_x = 0;
            let dim_y = 1;
            let current_slice_dim = 2;
            let indices = [];
            let colormap_idx = 0;
            let dr_idx = 1;  // default 1-99%
            let isPlaying = false;
            let playInterval = null;
            let gridMode = false;

            async function init() {
                const res = await fetch('/metadata');
                const data = await res.json();
                shape = data.shape;
                indices = shape.map(s => Math.floor(s / 2));

                const ndim = shape.length;
                dim_x = 0;
                dim_y = 1;
                current_slice_dim = ndim > 2 ? 2 : 0;

                updateView();
            }

            function renderInfo() {
                const idxStr = indices.map((v, i) => {
                    if (i === dim_x) return 'x';
                    if (i === dim_y) return 'y';
                    if (i === current_slice_dim) return `<span class="highlight">[${v}]</span>`;
                    return v;
                }).join(', ');
                let text = `Shape: [${shape.join(', ')}]\n`;
                text += `Index: [${idxStr}]\n`;
                text += `Colormap: ${COLORMAPS[colormap_idx]}   DR: ${DR_LABELS[dr_idx]}`;
                if (gridMode) text += `   <span style="color:#fff">  [GRID]</span>`;
                text += `   <span style="color:#888">(? for help)</span>`;
                document.getElementById('info').innerHTML = text;
            }

            function setStatus(msg) {
                document.getElementById('status').textContent = msg;
            }

            const canvas = document.getElementById('viewer');
            const ctx = canvas.getContext('2d');
            let isFetching = false;
            let pendingUpdate = false;

            async function updateView() {
                renderInfo();
                if (isFetching) {
                    pendingUpdate = true;
                    return;
                }
                isFetching = true;
                pendingUpdate = false;

                const url = gridMode
                    ? `/grid?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&colormap=${COLORMAPS[colormap_idx]}&dr=${dr_idx}&slice_dim=${current_slice_dim}`
                    : `/slice?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}&colormap=${COLORMAPS[colormap_idx]}&dr=${dr_idx}`;
                try {
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        const maxW = window.innerWidth * 0.95;
                        const maxH = window.innerHeight * 0.70;
                        const scale = Math.min(maxW / img.width, maxH / img.height);
                        canvas.style.width = Math.round(img.width * scale) + 'px';
                        canvas.style.height = Math.round(img.height * scale) + 'px';
                        isFetching = false;
                        if (pendingUpdate) updateView();
                    };
                    img.src = url;
                } catch (e) {
                    isFetching = false;
                }
            }

            function togglePlay() {
                if (isPlaying) {
                    clearInterval(playInterval);
                    playInterval = null;
                    isPlaying = false;
                    setStatus('');
                } else {
                    isPlaying = true;
                    setStatus('▶ playing  (Space to stop)');
                    playInterval = setInterval(() => {
                        indices[current_slice_dim] = (indices[current_slice_dim] + 1) % shape[current_slice_dim];
                        updateView();
                    }, 100);
                }
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
            window.addEventListener('keydown', (e) => {
                if (e.key === '?') {
                    helpOverlay.classList.toggle('visible');
                    return;
                }
                if (e.key === 'Escape') {
                    helpOverlay.classList.remove('visible');
                    return;
                }
                if (e.key === 'z') {
                    gridMode = !gridMode;
                    if (gridMode && isPlaying) togglePlay();
                    updateView();
                } else if (e.key === ' ') {
                    e.preventDefault();
                    if (!gridMode) togglePlay();
                } else if (e.key === 's') {
                    saveScreenshot();
                } else if (e.key === 'g') {
                    saveGif();
                } else if (e.key === 'c') {
                    colormap_idx = (colormap_idx + 1) % COLORMAPS.length;
                    updateView();
                } else if (e.key === 'd') {
                    dr_idx = (dr_idx + 1) % DR_LABELS.length;
                    updateView();
                } else if (e.key === 'j' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    if (!gridMode) { indices[current_slice_dim] = Math.max(0, indices[current_slice_dim] - 1); updateView(); }
                } else if (e.key === 'k' || e.key === 'ArrowUp') {
                    e.preventDefault();
                    if (!gridMode) { indices[current_slice_dim] = Math.min(shape[current_slice_dim] - 1, indices[current_slice_dim] + 1); updateView(); }
                } else if (e.key === 'h' || e.key === 'ArrowLeft') {
                    e.preventDefault();
                    do { current_slice_dim = (current_slice_dim - 1 + shape.length) % shape.length; }
                    while ((current_slice_dim === dim_x || current_slice_dim === dim_y) && shape.length > 2);
                    updateView();
                } else if (e.key === 'l' || e.key === 'ArrowRight') {
                    e.preventDefault();
                    do { current_slice_dim = (current_slice_dim + 1) % shape.length; }
                    while ((current_slice_dim === dim_x || current_slice_dim === dim_y) && shape.length > 2);
                    updateView();
                } else if (e.key === 'x') {
                    if (shape.length < 3) return;
                    let temp = dim_x;
                    dim_x = current_slice_dim;
                    current_slice_dim = temp;
                    updateView();
                } else if (e.key === 'y') {
                    if (shape.length < 3) return;
                    let temp = dim_y;
                    dim_y = current_slice_dim;
                    current_slice_dim = temp;
                    updateView();
                }
            });
            helpOverlay.addEventListener('click', () => helpOverlay.classList.remove('visible'));

            window.addEventListener('wheel', (e) => {
                e.preventDefault();
                if (gridMode) return;
                if (e.deltaY > 0) {
                    indices[current_slice_dim] = Math.max(0, indices[current_slice_dim] - 1);
                } else {
                    indices[current_slice_dim] = Math.min(shape[current_slice_dim] - 1, indices[current_slice_dim] + 1);
                }
                updateView();
            }, {passive: false});

            init();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/metadata")
def get_metadata():
    return {"shape": list(SHAPE)}

@app.get("/slice")
def get_slice(dim_x: int, dim_y: int, indices: str, colormap: str = 'gray', dr: int = 1):
    idx_list = [int(x) for x in indices.split(',')]
    extracted = extract_slice(dim_x, dim_y, idx_list)
    rgb = apply_colormap(extracted, colormap, dr)
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/grid")
def get_grid(dim_x: int, dim_y: int, indices: str, slice_dim: int, colormap: str = 'gray', dr: int = 1):
    idx_list = [int(x) for x in indices.split(',')]
    n = SHAPE[slice_dim]

    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(dim_x, dim_y, idx_list))

    all_data = np.stack(frames)  # (n, H, W)
    pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
    vmin = np.percentile(all_data, pct_lo)
    vmax = np.percentile(all_data, pct_hi)

    cmap = mpl_colormaps[colormap if colormap in COLORMAPS else 'gray']
    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape

    # Pad to fill grid
    padded = np.zeros((rows * cols, H, W), dtype=np.float64)
    padded[:n] = all_data

    # Tile into mosaic: (rows, cols, H, W) -> (rows*H, cols*W)
    mosaic = padded.reshape(rows, cols, H, W).transpose(0, 2, 1, 3).reshape(rows * H, cols * W)

    if vmax > vmin:
        normalized = np.clip((mosaic - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(mosaic)

    rgb = (cmap(normalized) * 255).astype(np.uint8)[:, :, :3]
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/gif")
def get_gif(dim_x: int, dim_y: int, indices: str, slice_dim: int, colormap: str = 'gray', dr: int = 1):
    idx_list = [int(x) for x in indices.split(',')]
    n = SHAPE[slice_dim]

    # Collect all frames and compute global normalization
    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
    vmin = np.percentile(all_data, pct_lo)
    vmax = np.percentile(all_data, pct_hi)

    cmap = mpl_colormaps[colormap if colormap in COLORMAPS else 'gray']

    gif_frames = []
    for frame in frames:
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(frame)
        rgb = (cmap(normalized) * 255).astype(np.uint8)[:, :, :3]
        gif_frames.append(Image.fromarray(rgb, mode='RGB'))

    buf = io.BytesIO()
    gif_frames[0].save(buf, format='GIF', save_all=True,
                       append_images=gif_frames[1:], loop=0, duration=100)
    return Response(content=buf.getvalue(), media_type="image/gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument("file", help="Path to .npy or .nii.gz file")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    try:
        DATA = load_data(args.file)
        SHAPE = DATA.shape
        print(f"Loaded {args.file} mapped to shape {SHAPE}")
        print(f"Open http://127.0.0.1:{args.port} in your browser")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    import threading, webbrowser
    url = f"http://127.0.0.1:{args.port}"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
