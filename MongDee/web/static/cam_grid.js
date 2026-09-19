// Shared #cam-grid interactions — used by both /booth and /product-view,
// since both pages show a grid of camera panels with the same markup
// (.cam-panel, .cam-icon-btn[data-action=popout|fullscreen]).
//
// "popout" opens the camera in its own window so it can be dragged onto a
// different physical monitor and full-screened there — one browser window
// can only be full-screen on one screen at a time, so showing several
// cameras full-screen across several monitors needs one window per camera.
// "fullscreen" full-screens the panel in place, using whichever screen this
// window currently sits on.
//
// Panels are looked up via btn.closest('.cam-panel') rather than a
// `panel-${camId}` id — /booth's viewer windows (see booth.js) let more
// than one panel show the same camera at once, so camId alone is no longer
// a unique DOM key there.
document.getElementById('cam-grid')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.cam-icon-btn');
    if (!btn) return;
    const camId = btn.dataset.cam;

    if (btn.dataset.action === 'popout') {
        window.open(`/booth/camera/${camId}`, `mongdee-cam-${camId}`,
            'width=900,height=680,noopener');
    } else if (btn.dataset.action === 'fullscreen') {
        const panel = btn.closest('.cam-panel');
        if (!panel) return;
        if (!document.fullscreenElement) {
            (panel.requestFullscreen || panel.webkitRequestFullscreen)?.call(panel);
        } else {
            document.exitFullscreen?.();
        }
    } else if (btn.dataset.action === 'toggle') {
        toggleCamera(camId, btn);
    } else if (btn.dataset.action === 'remove-viewer') {
        // Only present on pages that manage viewer windows (see booth.js) —
        // closes this viewing pane without touching the camera itself.
        window.removeCamViewer?.(btn.dataset.viewer);
    } else if (btn.dataset.action === 'tripwire') {
        // Only present where the Virtual Tripwire editor is wired up (see
        // web/static/tripwire.js).
        window.onToggleTripwireEditor?.(btn.dataset.viewer, camId);
    }
});

// Per-window camera picker (see booth.js's viewer windows) — delegated the
// same way so it works for panels created after this script ran.
document.getElementById('cam-grid')?.addEventListener('change', (e) => {
    const select = e.target.closest('.cam-select');
    if (!select) return;
    window.onViewerCameraChange?.(select.dataset.viewer, select.value || null);
});

// Turn a camera on/off in place — for a camera that's plugged in but not
// currently needed at the booth. Distinct from removing it in /settings,
// which forgets the camera entirely.
async function toggleCamera(camId, btn) {
    const enabled = btn.dataset.enabled === 'true';
    const nextEnabled = !enabled;
    btn.disabled = true;
    try {
        const res = await fetch(`/api/booth/cameras/${camId}/enabled`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: nextEnabled }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        btn.dataset.enabled = String(nextEnabled);
        btn.title = nextEnabled ? 'เปิด/ปิดกล้องนี้ (สำหรับกล้องที่ไม่ได้ใช้งาน)' : 'กล้องนี้ปิดใช้งานอยู่ — คลิกเพื่อเปิด';
        btn.closest('.cam-panel')?.classList.toggle('cam-panel-off', !nextEnabled);
        window.onCameraEnabledChange?.(camId, nextEnabled);
    } catch (e) {
        // leave dataset.enabled unchanged so a retry click sends the right intent
    } finally {
        btn.disabled = false;
    }
}

document.querySelectorAll('.cam-icon-btn[data-action="toggle"]').forEach((btn) => {
    if (btn.dataset.enabled === 'false') {
        btn.closest('.cam-panel')?.classList.add('cam-panel-off');
    }
});

function updateCamFullscreenIcons() {
    document.querySelectorAll('.cam-icon-btn[data-action="fullscreen"]').forEach((btn) => {
        const isFs = document.fullscreenElement && document.fullscreenElement === btn.closest('.cam-panel');
        btn.innerHTML = iconHtml(isFs ? 'minimize' : 'maximize', { size: 14, className: 'icon-inline' });
        btn.title = isFs ? 'ออกจากโหมดเต็มจอ' : 'ดูกล้องนี้แบบเต็มจอ';
    });
}

document.addEventListener('fullscreenchange', updateCamFullscreenIcons);
document.addEventListener('webkitfullscreenchange', updateCamFullscreenIcons);
