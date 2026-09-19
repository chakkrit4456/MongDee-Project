const ALERT_ICONS = {
    camera_offline: 'alertTriangle',
    camera_online: 'check',
    camera_connected: 'plug',
    camera_reconnecting: 'refresh',
    camera_disabled: 'plug',
    camera_degraded: 'activity',
    product_found: 'box',
    ai_paused: 'alertTriangle',
    ai_resumed: 'check',
};

function renderState(state) {
    const closedBanner = document.getElementById('closed-banner');
    if (state.running === false) {
        let text = 'บูธปิดอยู่ขณะนี้ — กล้องหยุดทำงานตามตารางเวลา';
        if (state.open_time && state.close_time) {
            text += ` (เปิดให้บริการ ${state.open_time} - ${state.close_time} น.)`;
        }
        document.getElementById('closed-banner-text').textContent = text;
        closedBanner.classList.remove('hidden');
    } else {
        closedBanner.classList.add('hidden');
    }

    updateViewerStatuses(state.cameras);

    const peoplePill = document.getElementById('people-now-pill');
    peoplePill.innerHTML = iconHtml('users', { size: 14, className: 'icon-inline' }) + `คนในกล้อง: ${state.people_now}`;

    // CPU-pressure AI Pause (see core/performance.py's AdaptiveController) —
    // camera video keeps streaming live throughout this; only detection/
    // product-recognition output pauses, so this is informational, not an
    // error state.
    const aiPausedPill = document.getElementById('ai-paused-pill');
    if (state.ai_paused) {
        aiPausedPill.innerHTML = iconHtml('alertTriangle', { size: 14, className: 'icon-inline' }) + 'AI พักชั่วคราว (CPU สูง)';
        aiPausedPill.classList.remove('hidden');
    } else {
        aiPausedPill.classList.add('hidden');
    }

    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = '';
    for (const alert of state.recent_alerts) {
        const div = document.createElement('div');
        const t = new Date(alert.ts * 1000).toLocaleTimeString('th-TH', { hour12: false });
        const icon = ALERT_ICONS[alert.type] || 'info';
        div.innerHTML = `${iconHtml(icon, { size: 14, className: 'icon-inline' })}[${t}] ${alert.text}`;
        alertsList.appendChild(div);
    }
}

async function pollState() {
    try {
        const res = await fetch('/api/state');
        const state = await res.json();
        renderState(state);
    } catch (e) { /* transient network hiccup — next poll will retry */ }
    setTimeout(pollState, 1500);
}

// Dashboard side drawer — opens in-place next to the booth view instead of a
// separate tab, so a live demo can show analytics without alt-tabbing. The
// iframe is same-origin, so it shares localStorage with this page and picks
// up the dark/light theme (see theme.js) and auto-refreshes itself on its
// own 5s interval — nothing extra needed here to keep it in sync.
const dashboardDrawer = document.getElementById('dashboard-drawer');
const dashboardIframe = document.getElementById('dashboard-drawer-iframe');
let dashboardDrawerLoaded = false;

function setDashboardDrawerOpen(open) {
    if (open && !dashboardDrawerLoaded) {
        dashboardIframe.src = '/dashboard';
        dashboardDrawerLoaded = true;
    }
    dashboardDrawer.classList.toggle('open', open);
}

document.getElementById('open-dashboard-btn').addEventListener('click', () => {
    setDashboardDrawerOpen(!dashboardDrawer.classList.contains('open'));
});
document.getElementById('dashboard-drawer-close').addEventListener('click', () => setDashboardDrawerOpen(false));
document.getElementById('dashboard-drawer-popout').addEventListener('click', () => {
    window.open('/dashboard', '_blank', 'noopener');
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && dashboardDrawer.classList.contains('open')) setDashboardDrawerOpen(false);
});

document.getElementById('open-product-btn').addEventListener('click', () => {
    window.open('/product-view', '_blank', 'noopener');
});

document.getElementById('readiness-btn').addEventListener('click', async () => {
    const res = await fetch('/api/readiness', { method: 'POST' });
    const report = await res.json();

    const pill = document.getElementById('readiness-pill');
    pill.innerHTML = report.overall_ok
        ? iconHtml('check', { size: 14, className: 'icon-inline' }) + 'READY'
        : iconHtml('x', { size: 14, className: 'icon-inline' }) + 'NOT READY';
    pill.className = 'pill ' + (report.overall_ok ? 'ready' : 'not-ready');

    const headline = document.getElementById('readiness-headline');
    headline.textContent = report.overall_ok ? 'บูธพร้อมใช้งาน (READY)' : 'บูธยังไม่พร้อม (NOT READY)';
    headline.style.color = report.overall_ok ? 'var(--success)' : 'var(--danger)';

    const list = document.getElementById('readiness-list');
    list.innerHTML = '';
    for (const c of report.components) {
        const icon = c.ok ? 'check' : (c.critical ? 'x' : 'alertTriangle');
        const block = document.createElement('div');
        block.className = 'readiness-component';
        let html = `<div class="readiness-summary">${iconHtml(icon, { size: 14, className: 'icon-inline' })}${c.component} — ${c.detail}</div>`;
        const advanced = c.advanced || {};
        const keys = Object.keys(advanced);
        if (keys.length) {
            html += '<dl class="readiness-advanced">';
            for (const key of keys) {
                html += `<dt>${key}</dt><dd>${advanced[key]}</dd>`;
            }
            html += '</dl>';
        }
        block.innerHTML = html;
        list.appendChild(block);
    }
    document.getElementById('readiness-modal').classList.remove('hidden');
});

// ---------------------------------------------------------------- viewer windows
// A "viewer window" (one .cam-panel tile in #cam-grid) is just a pane
// pointed at a camera_id via its own dropdown — decoupled from how many
// physical cameras exist, so windows can be added/removed and repointed at
// any camera independently of that camera's own lifecycle. This also fixes
// two real bugs in the old fixed-one-panel-per-camera grid: a camera
// plugged in mid-session (or added via /settings from another tab) had no
// panel until the page was reloaded, and a camera that dropped offline kept
// showing its last frame frozen forever instead of an honest "reconnecting"
// state (the frozen-frame half of that is now fixed server-side in
// web/booth_manager.py; the img-refresh-on-recovery below covers the
// browser-side half — resuming a stream that never technically errored but
// also never gained enough re-established buffering to display new bytes
// on some browsers).
//
// Deliberately NOT persisted to localStorage: an earlier version remembered
// the window layout across reloads, but that meant a page reload restored
// whatever was saved *instead of* matching the cameras actually connected
// right now — directly contradicting "opening the page should show one
// window per currently-connected camera". So every page load starts fresh
// from INITIAL_CAMERA_IDS, and dismissedCameraIds (below) is in-memory only
// — it just stops *this session's* periodic poll from instantly re-adding a
// window the user just closed; a reload always re-evaluates from scratch.
let availableCameraIds = INITIAL_CAMERA_IDS.slice();
let cameraEnabled = {};
let lastCameraStatus = {};
let dismissedCameraIds = new Set();
let viewers = INITIAL_CAMERA_IDS.map((camId) => ({ id: newViewerId(), camId }));

function newViewerId() {
    return 'v' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function removeCamViewer(viewerId) {
    const viewer = viewers.find((v) => v.id === viewerId);
    if (viewer && viewer.camId) dismissedCameraIds.add(viewer.camId);
    viewers = viewers.filter((v) => v.id !== viewerId);
    renderViewers();
}
window.removeCamViewer = removeCamViewer;

function onViewerCameraChange(viewerId, camId) {
    const viewer = viewers.find((v) => v.id === viewerId);
    if (!viewer) return;
    viewer.camId = camId || null;
    if (camId) dismissedCameraIds.delete(camId);
    renderViewers();
}
window.onViewerCameraChange = onViewerCameraChange;

function onCameraEnabledChange(camId, enabled) {
    cameraEnabled[camId] = enabled;
}
window.onCameraEnabledChange = onCameraEnabledChange;

// "Connected" here means actually streaming right now (online, or degraded
// — degraded is just AI falling behind under CPU load, the video feed
// itself is unaffected), not merely "registered at some point". A camera
// that's offline/reconnecting/disabled/never-yet-confirmed must not be
// pickable from the dropdown as if it were a live source.
function isCameraConnected(camId) {
    const status = lastCameraStatus[camId];
    return status === 'online' || status === 'degraded';
}

function cameraOptionsHtml(selected) {
    let html = '<option value="">— เลือกกล้อง —</option>';
    for (const camId of availableCameraIds) {
        // Always keep the panel's own current selection listed, even if it
        // just dropped — otherwise the moment a connected camera goes
        // offline, its own dropdown would silently lose track of what it's
        // pointed at instead of showing the "reconnecting" state on it.
        if (camId !== selected && !isCameraConnected(camId)) continue;
        html += `<option value="${camId}"${camId === selected ? ' selected' : ''}>${camId}</option>`;
    }
    return html;
}

function viewerPanelHtml(viewer) {
    const camId = viewer.camId && availableCameraIds.includes(viewer.camId) ? viewer.camId : null;
    const enabled = camId ? cameraEnabled[camId] !== false : true;
    const actions = camId ? `
        <button class="cam-icon-btn" data-action="toggle" data-cam="${camId}" data-enabled="${enabled}" title="เปิด/ปิดกล้องนี้ (สำหรับกล้องที่ไม่ได้ใช้งาน)">${iconHtml('plug', { size: 14, className: 'icon-inline' })}</button>
        <button class="cam-icon-btn" data-action="tripwire" data-cam="${camId}" data-viewer="${viewer.id}" title="ตั้งค่าเส้นนับคน (Virtual Tripwire)">${iconHtml('tripwire', { size: 14, className: 'icon-inline' })}</button>
        <button class="cam-icon-btn" data-action="popout" data-cam="${camId}" title="เปิดกล้องนี้ในหน้าต่างใหม่ (ย้ายไปแสดงอีกจอได้)">${iconHtml('openWindow', { size: 14, className: 'icon-inline' })}</button>
        <button class="cam-icon-btn" data-action="fullscreen" data-cam="${camId}" title="ดูกล้องนี้แบบเต็มจอ">${iconHtml('maximize', { size: 14, className: 'icon-inline' })}</button>` : '';
    return `
    <div class="panel cam-panel${camId && !enabled ? ' cam-panel-off' : ''}" id="viewer-${viewer.id}">
      <div class="cam-title">
        <span class="status-dot status-unknown" id="dot-viewer-${viewer.id}"></span>
        <select class="cam-select" data-viewer="${viewer.id}">${cameraOptionsHtml(camId)}</select>
        <div class="cam-actions">
          ${actions}
          <button class="cam-icon-btn" data-action="remove-viewer" data-viewer="${viewer.id}" title="ปิดหน้าต่างนี้"><svg class="icon-inline" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="5" x2="19" y2="19"/><line x1="19" y1="5" x2="5" y2="19"/></svg></button>
        </div>
      </div>
      ${camId ? `
      <div class="cam-stage-wrap" id="stage-wrap-${viewer.id}">
        <img alt="${camId}" id="img-viewer-${viewer.id}" data-cam="${camId}">
        ${tripwireCanvasHtml(viewer.id)}
      </div>
      ${tripwireToolbarHtml(viewer.id)}` : '<div class="status-text" style="padding:40px 0; text-align:center;">ยังไม่ได้เลือกกล้องสำหรับหน้าต่างนี้</div>'}
      <div class="status-text" id="status-viewer-${viewer.id}">${camId ? 'กำลังเชื่อมต่อ...' : ''}</div>
    </div>`;
}

function attachImgRecovery(img) {
    // A genuinely dropped connection (network blip, server restart) leaves
    // an MJPEG <img> broken with nothing to retry it on its own. Reload the
    // stream a beat later so the window recovers by itself instead of
    // staying broken until the user refreshes the whole page.
    img.addEventListener('error', () => {
        setTimeout(() => {
            if (img.isConnected) img.src = `/stream/${img.dataset.cam}?t=${Date.now()}`;
        }, 1500);
    });
}

function renderViewers() {
    const grid = document.getElementById('cam-grid');
    if (!grid) return;
    if (!viewers.length) {
        grid.innerHTML = '<div class="panel cam-panel"><div class="status-text">ยังไม่มีหน้าต่างแสดงภาพ — กด "เพิ่มหน้าต่างกล้อง" ด้านบน</div></div>';
        return;
    }
    grid.innerHTML = viewers.map(viewerPanelHtml).join('');
    const imgs = grid.querySelectorAll('img[id^="img-viewer-"]');
    imgs.forEach(attachImgRecovery);
    // Multiple <img src> assignments in the very same tick each open their
    // own multipart/x-mixed-replace connection at once — browsers can
    // corrupt the first frame of whichever one loses that race (seen as a
    // broken-image icon on a random one of the panels, not always the same
    // camera). The <img> tags are created with no `src` above specifically
    // so this can stagger the actual connection starts a beat apart instead
    // of firing them all in the same JS tick.
    imgs.forEach((img, i) => {
        setTimeout(() => {
            if (img.isConnected) img.src = `/stream/${img.dataset.cam}`;
        }, i * 250);
    });
    updateCamFullscreenIcons();
}

// Reflects live per-camera status (from /api/state, polled every 1.5s via
// pollState/renderState) onto every viewer window currently showing that
// camera — more than one window can point at the same camera at once. On a
// transition into "online" from anything else (a real reconnect after an
// unplug, or the very first connect), the <img> src is reloaded with a
// cache-busting query param to force a brand-new stream connection rather
// than trust a possibly-stalled one to resume on its own.
function updateViewerStatuses(cameras) {
    for (const [camId, info] of Object.entries(cameras)) {
        const prevStatus = lastCameraStatus[camId];
        const justRecovered = prevStatus && prevStatus !== 'online' && info.status === 'online';
        lastCameraStatus[camId] = info.status;

        for (const viewer of viewers) {
            if (viewer.camId !== camId) continue;
            const dot = document.getElementById(`dot-viewer-${viewer.id}`);
            const statusText = document.getElementById(`status-viewer-${viewer.id}`);
            if (dot) dot.className = 'status-dot status-' + info.status;
            if (statusText) statusText.textContent = info.message || info.status;
            if (justRecovered) {
                const img = document.getElementById(`img-viewer-${viewer.id}`);
                if (img) img.src = `/stream/${camId}?t=${Date.now()}`;
            }
        }
    }
}

document.getElementById('add-viewer-btn')?.addEventListener('click', () => {
    // Always adds a blank "choose a camera" window rather than guessing —
    // auto-picking an already-shown camera made the click look like it did
    // nothing (a second tile with identical video is easy to miss). This
    // way every click is unambiguously visible: a new tile with an empty
    // dropdown appears immediately.
    viewers.push({ id: newViewerId(), camId: null });
    renderViewers();
});

// Keeps every dropdown's option list current without touching anything
// else — in particular, never replaces an <img>'s DOM node, which would
// otherwise tear down and reopen its MJPEG connection every poll tick for
// no reason (defeating the "never stutter" goal just as badly as the bug
// this whole feature fixes).
function refreshSelectOptions() {
    document.querySelectorAll('#cam-grid .cam-select').forEach((select) => {
        const viewer = viewers.find((v) => v.id === select.dataset.viewer);
        select.innerHTML = cameraOptionsHtml(viewer ? viewer.camId : select.value);
    });
}

// Same idea for the enabled/disabled look of the toggle button + panel
// dimming, kept current in case a *different* tab/device toggled the
// camera — again without rebuilding the panel itself.
function refreshToggleButtons() {
    document.querySelectorAll('#cam-grid .cam-icon-btn[data-action="toggle"]').forEach((btn) => {
        const enabled = cameraEnabled[btn.dataset.cam] !== false;
        if (String(enabled) !== btn.dataset.enabled) {
            btn.dataset.enabled = String(enabled);
            btn.closest('.cam-panel')?.classList.toggle('cam-panel-off', !enabled);
        }
    });
}

async function pollCameraList() {
    try {
        const res = await fetch('/api/booth/settings');
        const settings = await res.json();
        const ids = settings.cameras.map((c) => c.camera_id);
        const enabledMap = {};
        for (const c of settings.cameras) enabledMap[c.camera_id] = c.enabled;
        cameraEnabled = enabledMap;
        availableCameraIds = ids;
        const idSet = new Set(ids);

        // A camera with no window showing it yet (freshly plugged in and
        // picked up by the Hot-Plug Scan, or added via /settings from
        // another tab) gets its own viewer window automatically — the whole
        // point of the fix is the user doesn't have to do anything to see
        // it. Deliberately-closed cameras (see removeCamViewer) are the one
        // exception, so closing a window sticks.
        let structuralChange = false;
        for (const camId of ids) {
            if (!dismissedCameraIds.has(camId) && !viewers.some((v) => v.camId === camId)) {
                viewers.push({ id: newViewerId(), camId });
                structuralChange = true;
            }
        }
        // A camera removed via /settings: any window pointed at it falls
        // back to an empty "choose a camera" state instead of vanishing, so
        // the rest of the operator's window layout stays put.
        for (const viewer of viewers) {
            if (viewer.camId && !idSet.has(viewer.camId)) {
                viewer.camId = null;
                structuralChange = true;
            }
        }
        if (structuralChange) {
            renderViewers();  // panel set actually changed — a full rebuild is unavoidable here
        } else {
            refreshSelectOptions();
            refreshToggleButtons();
        }
    } catch (e) { /* transient network hiccup — next poll will retry */ }
    setTimeout(pollCameraList, 4000);
}

renderViewers();
pollCameraList();
pollState();
