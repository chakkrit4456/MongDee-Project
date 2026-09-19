const stage = document.getElementById('cam-stage');
const fsBtn = document.getElementById('fullscreen-btn');

function updateFullscreenBtn() {
    const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
    fsBtn.innerHTML = iconHtml(isFs ? 'minimize' : 'maximize', { size: 16 }) +
        (isFs ? 'ออกจากเต็มจอ' : 'เต็มจอ');
}

fsBtn.addEventListener('click', () => {
    if (!(document.fullscreenElement || document.webkitFullscreenElement)) {
        (stage.requestFullscreen || stage.webkitRequestFullscreen)?.call(stage);
    } else {
        (document.exitFullscreen || document.webkitExitFullscreen)?.call(document);
    }
});

document.addEventListener('fullscreenchange', updateFullscreenBtn);
document.addEventListener('webkitfullscreenchange', updateFullscreenBtn);
updateFullscreenBtn();

const streamImg = document.querySelector('#cam-stage img');
// A dropped connection (network blip, server restart) otherwise leaves this
// <img> broken with nothing to retry it — reload the stream a beat later so
// a popped-out window (often left running unattended on its own monitor)
// recovers on its own instead of staying frozen/broken indefinitely.
streamImg?.addEventListener('error', () => {
    setTimeout(() => {
        if (streamImg.isConnected) streamImg.src = `/stream/${CAMERA_ID}?t=${Date.now()}`;
    }, 1500);
});

let lastStatus = null;

async function pollStatus() {
    try {
        const res = await fetch('/api/state');
        const state = await res.json();
        const info = state.cameras[CAMERA_ID];
        if (info) {
            const dot = document.getElementById(`dot-${CAMERA_ID}`);
            const statusText = document.getElementById(`status-${CAMERA_ID}`);
            if (dot) dot.className = 'status-dot status-' + info.status;
            if (statusText) statusText.textContent = info.message || info.status;
            // Force a fresh stream connection on reconnect (see booth.js's
            // updateViewerStatuses for the full rationale) rather than
            // trusting a possibly-stalled one to resume by itself.
            if (lastStatus && lastStatus !== 'online' && info.status === 'online' && streamImg) {
                streamImg.src = `/stream/${CAMERA_ID}?t=${Date.now()}`;
            }
            lastStatus = info.status;
        }
    } catch (e) { /* transient network hiccup — next poll will retry */ }
    setTimeout(pollStatus, 2000);
}

pollStatus();

// Right-side detail panel — live people (by category) / product counts for
// just this camera (see BoothManager.get_camera_snapshot). Polled on its
// own short interval, independent of pollStatus's connection-status loop
// above, so a slow/failed snapshot fetch never delays reconnect handling.
const detailEls = {
    total: document.getElementById('cam-detail-people-total'),
    male: document.getElementById('cam-detail-male'),
    female: document.getElementById('cam-detail-female'),
    unknown: document.getElementById('cam-detail-unknown'),
    products: document.getElementById('cam-detail-products'),
};

async function pollCameraSnapshot() {
    try {
        const res = await fetch(`/api/booth/cameras/${CAMERA_ID}/snapshot`);
        if (res.ok) {
            const s = await res.json();
            if (detailEls.total) detailEls.total.textContent = s.people_total;
            if (detailEls.male) detailEls.male.textContent = s.male;
            if (detailEls.female) detailEls.female.textContent = s.female;
            if (detailEls.unknown) detailEls.unknown.textContent = s.unknown;
            if (detailEls.products) detailEls.products.textContent = s.products_detected;
        }
    } catch (e) { /* transient network hiccup — next poll will retry */ }
    setTimeout(pollCameraSnapshot, 2000);
}

pollCameraSnapshot();
