// Virtual Tripwire line editor — a draggable two-point line drawn on a
// <canvas> positioned over one viewer window's <img> MJPEG stream, shown
// only while that window is in edit mode. In normal viewing the line is
// NOT drawn by this script at all — it's already baked into the video by
// the server (core/vision.py's _draw_tripwire, fed by
// web/booth_manager.py's crossing detection), so this file only ever
// touches the DOM while a window is actively being edited.

const tripwireState = {}; // viewerId -> { camId, x1,y1,x2,y2, insideSide, dragging }
const TRIPWIRE_COLOR = '#ff9500';
const TRIPWIRE_HANDLE_RADIUS = 8;

// Split in two on purpose: the canvas alone covers the full video area
// (inside .cam-stage-wrap, on top of the <img>) so every pixel of it is
// draggable, while the side-select/save/delete/close controls live in a
// *separate* block placed below the video entirely (see
// tripwireToolbarHtml, appended as a sibling of .cam-stage-wrap in
// booth.js's viewerPanelHtml) — an earlier version overlaid the toolbar on
// the bottom of the canvas, which silently ate drag clicks on any line
// handle that happened to land underneath it.
function tripwireCanvasHtml(viewerId) {
    return `<canvas class="tripwire-canvas hidden" id="tripwire-canvas-${viewerId}"></canvas>`;
}

function tripwireToolbarHtml(viewerId) {
    return `
    <div class="tripwire-toolbar hidden" id="tripwire-toolbar-${viewerId}">
      <span>ด้านภายในบูธ:</span>
      <label><input type="radio" name="tw-side-${viewerId}" value="A" checked> ด้าน A</label>
      <label><input type="radio" name="tw-side-${viewerId}" value="B"> ด้าน B</label>
      <button class="btn-small tripwire-save-btn" data-viewer="${viewerId}">${iconHtml('check', { size: 13, className: 'icon-inline' })}บันทึก</button>
      <button class="btn-small btn-danger tripwire-delete-btn" data-viewer="${viewerId}">${iconHtml('trash', { size: 13, className: 'icon-inline' })}ลบเส้น</button>
      <button class="btn-small tripwire-cancel-btn" data-viewer="${viewerId}">ปิด</button>
    </div>`;
}

function tripwireEditorElements(viewerId) {
    return [
        document.getElementById(`tripwire-canvas-${viewerId}`),
        document.getElementById(`tripwire-toolbar-${viewerId}`),
    ];
}

async function onToggleTripwireEditor(viewerId, camId) {
    const [canvas, toolbar] = tripwireEditorElements(viewerId);
    if (!canvas || !toolbar) return;
    if (!canvas.classList.contains('hidden')) {
        closeTripwireEditor(viewerId);
        return;
    }
    // Only one line editor open at a time — dragging two at once serves no
    // purpose and doubles up on window-level mousemove listeners for no
    // reason.
    document.querySelectorAll('.tripwire-canvas, .tripwire-toolbar').forEach((el) => {
        if (!el.id.endsWith(`-${viewerId}`)) el.classList.add('hidden');
    });

    let line = null;
    try {
        const res = await fetch(`/api/booth/cameras/${camId}/tripwire`);
        if (res.ok) line = await res.json();
    } catch (e) { /* fall through to the default line below */ }

    tripwireState[viewerId] = line ? {
        camId, x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2, insideSide: line.inside_side, dragging: null,
    } : {
        // A sensible default: a vertical line through the middle, "inside"
        // on the left — the user drags both ends into place from here.
        camId, x1: 0.5, y1: 0.15, x2: 0.5, y2: 0.85, insideSide: 'A', dragging: null,
    };

    toolbar.querySelectorAll(`input[name="tw-side-${viewerId}"]`).forEach((r) => {
        r.checked = r.value === tripwireState[viewerId].insideSide;
    });

    canvas.classList.remove('hidden');
    toolbar.classList.remove('hidden');
    setupTripwireCanvas(viewerId);
}
window.onToggleTripwireEditor = onToggleTripwireEditor;

function closeTripwireEditor(viewerId) {
    const [canvas, toolbar] = tripwireEditorElements(viewerId);
    canvas?.classList.add('hidden');
    toolbar?.classList.add('hidden');
    delete tripwireState[viewerId];
}

function tripwirePointFromEvent(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
    return {
        nx: Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)),
        ny: Math.min(1, Math.max(0, (clientY - rect.top) / rect.height)),
    };
}

function tripwireHandleDistPx(canvas, nx, ny, px, py) {
    const rect = canvas.getBoundingClientRect();
    return Math.hypot((nx - px) * rect.width, (ny - py) * rect.height);
}

function setupTripwireCanvas(viewerId) {
    const canvas = document.getElementById(`tripwire-canvas-${viewerId}`);
    const img = document.getElementById(`img-viewer-${viewerId}`);
    if (!canvas || !img) return;

    function resize() {
        canvas.width = img.clientWidth || canvas.parentElement.clientWidth;
        canvas.height = img.clientHeight || canvas.parentElement.clientHeight;
        drawTripwireCanvas(viewerId);
    }
    resize();
    if (window.ResizeObserver && !canvas._tripwireResizeObserver) {
        const ro = new ResizeObserver(resize);
        ro.observe(img);
        canvas._tripwireResizeObserver = ro; // kept alive with the canvas element itself
    }

    const pick = (nx, ny, st) => {
        if (tripwireHandleDistPx(canvas, nx, ny, st.x1, st.y1) <= TRIPWIRE_HANDLE_RADIUS + 6) return 'p1';
        if (tripwireHandleDistPx(canvas, nx, ny, st.x2, st.y2) <= TRIPWIRE_HANDLE_RADIUS + 6) return 'p2';
        return null;
    };

    if (!canvas._tripwireBound) {
        canvas._tripwireBound = true;

        canvas.addEventListener('mousedown', (e) => {
            const st = tripwireState[viewerId];
            if (!st) return;
            const { nx, ny } = tripwirePointFromEvent(canvas, e);
            st.dragging = pick(nx, ny, st);
        });
        window.addEventListener('mousemove', (e) => {
            const st = tripwireState[viewerId];
            if (!st || !st.dragging) return;
            const { nx, ny } = tripwirePointFromEvent(canvas, e);
            if (st.dragging === 'p1') { st.x1 = nx; st.y1 = ny; } else { st.x2 = nx; st.y2 = ny; }
            drawTripwireCanvas(viewerId);
        });
        window.addEventListener('mouseup', () => {
            const st = tripwireState[viewerId];
            if (st) st.dragging = null;
        });

        canvas.addEventListener('touchstart', (e) => {
            const st = tripwireState[viewerId];
            if (!st) return;
            const { nx, ny } = tripwirePointFromEvent(canvas, e);
            st.dragging = pick(nx, ny, st);
            if (st.dragging) e.preventDefault();
        }, { passive: false });
        canvas.addEventListener('touchmove', (e) => {
            const st = tripwireState[viewerId];
            if (!st || !st.dragging) return;
            e.preventDefault();
            const { nx, ny } = tripwirePointFromEvent(canvas, e);
            if (st.dragging === 'p1') { st.x1 = nx; st.y1 = ny; } else { st.x2 = nx; st.y2 = ny; }
            drawTripwireCanvas(viewerId);
        }, { passive: false });
        canvas.addEventListener('touchend', () => {
            const st = tripwireState[viewerId];
            if (st) st.dragging = null;
        });
    }

    document.getElementById(`tripwire-toolbar-${viewerId}`)
        ?.querySelectorAll(`input[name="tw-side-${viewerId}"]`)
        .forEach((r) => r.addEventListener('change', () => {
            const st = tripwireState[viewerId];
            if (!st) return;
            st.insideSide = r.value;
            drawTripwireCanvas(viewerId);
        }));

    drawTripwireCanvas(viewerId);
}

function drawTripwireCanvas(viewerId) {
    const st = tripwireState[viewerId];
    const canvas = document.getElementById(`tripwire-canvas-${viewerId}`);
    if (!st || !canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (w === 0 || h === 0) return;

    const x1 = st.x1 * w, y1 = st.y1 * h, x2 = st.x2 * w, y2 = st.y2 * h;

    ctx.strokeStyle = TRIPWIRE_COLOR;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // A short perpendicular arrow toward "inside" — same cross-product
    // convention as core/tripwire.py's TripwireLine.side_of, so what the
    // user drags here is exactly what the server will later count against.
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;
    let px = -dy / len, py = dx / len;
    if (st.insideSide === 'B') { px = -px; py = -py; }
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const tipX = mx + px * 32, tipY = my + py * 32;

    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();
    const angle = Math.atan2(tipY - my, tipX - mx);
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - 8 * Math.cos(angle - Math.PI / 6), tipY - 8 * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - 8 * Math.cos(angle + Math.PI / 6), tipY - 8 * Math.sin(angle + Math.PI / 6));
    ctx.stroke();

    ctx.fillStyle = TRIPWIRE_COLOR;
    ctx.font = 'bold 13px sans-serif';
    ctx.fillText('IN', tipX + px * 16, tipY + py * 16);
    ctx.fillText('OUT', mx - px * 24, my - py * 24);

    [[x1, y1], [x2, y2]].forEach(([hx, hy]) => {
        ctx.beginPath();
        ctx.arc(hx, hy, TRIPWIRE_HANDLE_RADIUS, 0, Math.PI * 2);
        ctx.fillStyle = TRIPWIRE_COLOR;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
    });
}

document.addEventListener('click', async (e) => {
    const saveBtn = e.target.closest('.tripwire-save-btn');
    const deleteBtn = e.target.closest('.tripwire-delete-btn');
    const cancelBtn = e.target.closest('.tripwire-cancel-btn');

    if (saveBtn) {
        const viewerId = saveBtn.dataset.viewer;
        const st = tripwireState[viewerId];
        if (!st) return;
        saveBtn.disabled = true;
        try {
            const res = await fetch(`/api/booth/cameras/${st.camId}/tripwire`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    x1: st.x1, y1: st.y1, x2: st.x2, y2: st.y2,
                    inside_side: st.insideSide, enabled: true,
                }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
                alert(`บันทึกเส้นนับคนไม่สำเร็จ: ${err.detail || res.status}`);
                return;
            }
        } catch (err) {
            alert('เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ — บันทึกเส้นนับคนไม่สำเร็จ');
            return;
        } finally {
            saveBtn.disabled = false;
        }
        closeTripwireEditor(viewerId);
    } else if (deleteBtn) {
        const viewerId = deleteBtn.dataset.viewer;
        const st = tripwireState[viewerId];
        if (!st) return;
        if (!confirm('ลบเส้นนับคนของกล้องนี้หรือไม่? ตัวนับ IN/OUT ที่บันทึกไว้แล้วจะยังอยู่ใน Dashboard')) return;
        try {
            await fetch(`/api/booth/cameras/${st.camId}/tripwire`, { method: 'DELETE' });
        } catch (err) { /* best-effort — closing the editor either way is fine */ }
        closeTripwireEditor(viewerId);
    } else if (cancelBtn) {
        closeTripwireEditor(cancelBtn.dataset.viewer);
    }
});
