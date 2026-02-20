
import argparse
import io
import sys
import numpy as np
import nibabel as nib
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from PIL import Image

DATA = None
SHAPE = None

app = FastAPI()

def load_data(filepath):
    if filepath.endswith('.npy'):
        return np.load(filepath, mmap_mode='r')
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        img = nib.load(filepath)
        return img.dataobj
    else:
        raise ValueError("Unsupported format. Please provide a .npy or .nii/.nii.gz file")

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
            #help-box .key { color: #fff; font-weight: bold; display: inline-block; min-width: 120px; }
        </style>
    </head>
    <body>
        <div id="info">Loading...</div>
        <canvas id="viewer"></canvas>
        <div id="help-overlay">
            <div id="help-box"><span class="key">j / ↓</span>  previous slice
<span class="key">k / ↑</span>  next slice
<span class="key">h / ←</span>  previous slice dimension
<span class="key">l / →</span>  next slice dimension
<span class="key">x</span>  swap horizontal dim with slice dim
<span class="key">y</span>  swap vertical dim with slice dim
<span class="key">scroll</span>  scroll through slices
<span class="key">?</span>  toggle this help</div>
        </div>
        <script>
            let shape = [];
            let dim_x = 0;
            let dim_y = 1;
            let current_slice_dim = 2;
            let indices = [];

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
                text += `Index: [${idxStr}]  <span style="color:#888">(?  for help)</span>`;
                document.getElementById('info').innerHTML = text;
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

                const url = `/slice?dim_x=${dim_x}&dim_y=${dim_y}&indices=${indices.join(',')}`;
                try {
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                        // Scale canvas display size to fill viewport
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
                if (e.key === 'j' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    indices[current_slice_dim] = Math.max(0, indices[current_slice_dim] - 1);
                    updateView();
                } else if (e.key === 'k' || e.key === 'ArrowUp') {
                    e.preventDefault();
                    indices[current_slice_dim] = Math.min(shape[current_slice_dim] - 1, indices[current_slice_dim] + 1);
                    updateView();
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
def get_slice(dim_x: int, dim_y: int, indices: str):
    idx_list = [int(x) for x in indices.split(',')]

    # Build slicing tuple
    slicer = []
    for i in range(len(SHAPE)):
        if i == dim_x or i == dim_y:
            slicer.append(slice(None))
        else:
            slicer.append(idx_list[i])

    # Extract the 2D plane into memory
    extracted = np.array(DATA[tuple(slicer)])

    if np.iscomplexobj(extracted):
        extracted = np.abs(extracted)

    # After slicing, numpy preserves axis order from the original array.
    # The two remaining axes are in ascending dimension order.
    # We want shape (n_dimy, n_dimx) for PIL (rows x cols).
    # When dim_x < dim_y, extracted is (n_dimx, n_dimy) → transpose needed.
    if dim_x < dim_y:
        extracted = extracted.T

    extracted = np.nan_to_num(extracted)
    vmin, vmax = np.min(extracted), np.max(extracted)

    if vmax > vmin:
        img_data = ((extracted - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        img_data = np.zeros_like(extracted, dtype=np.uint8)

    img = Image.fromarray(img_data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return Response(content=buf.getvalue(), media_type="image/png")

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

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
