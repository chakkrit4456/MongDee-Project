// Performance Monitor panel on /booth — polls /api/performance (backed by
// core/performance.py's PerformanceMonitor + AdaptiveController) and shows
// real, measured per-camera FPS/latency/drop-rate plus system CPU/RAM/VRAM.
// See MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md section 29.

const STATUS_LABELS_TH = {
    connecting: 'กำลังเชื่อมต่อ', online: 'ออนไลน์', degraded: 'ประสิทธิภาพลดลง',
    reconnecting: 'กำลังเชื่อมต่อใหม่', offline: 'ออฟไลน์', error: 'ผิดพลาด',
    disabled: 'ปิดใช้งาน', unknown: 'ไม่ทราบสถานะ',
};

// Same "overloaded" thresholds core/performance.py's AdaptiveConfig itself
// backs off AI settings at (ai_latency_high_ms, drop_rate_high) — reused
// here purely for the card colors, so "this looks bad" on screen matches
// what the system already considers bad enough to act on. The "danger"
// tier (2x the AdaptiveConfig threshold) and the display-latency/FPS-ratio
// thresholds below have no such system-enforced equivalent — they're a
// judgment call for the color cue only, not a claim the system enforces them.
const AI_LATENCY_WARN_MS = 250, AI_LATENCY_BAD_MS = 500;
const DROP_RATE_WARN = 0.05, DROP_RATE_BAD = 0.15;
const DISPLAY_LATENCY_WARN_MS = 150, DISPLAY_LATENCY_BAD_MS = 300;
const FPS_RATIO_WARN = 0.8, FPS_RATIO_BAD = 0.5;

function metricBadge(valueText, tier) {
    const cls = tier === 'bad' ? 'mbad' : tier === 'warn' ? 'mmid' : 'mgood';
    return `<span class="metric-badge ${cls}">${valueText}</span>`;
}

function tierFor(value, warn, bad, higherIsWorse = true) {
    const overWarn = higherIsWorse ? value >= warn : value <= warn;
    const overBad = higherIsWorse ? value >= bad : value <= bad;
    return overBad ? 'bad' : overWarn ? 'warn' : 'good';
}

function renderPerformance(perf) {
    const sys = perf.system || {};
    const hw = perf.hardware || {};
    const cpuPct = sys.cpu_percent;
    const ramPct = sys.ram_percent;
    const cpuBadge = cpuPct != null
        ? metricBadge(`CPU: ${cpuPct.toFixed(0)}%`, tierFor(cpuPct, 70, 85))
        : '<span class="pill">CPU: n/a</span>';
    const ramBadge = ramPct != null
        ? metricBadge(`RAM: ${ramPct.toFixed(0)}% (${sys.ram_used_gb.toFixed(1)} GB)`, tierFor(ramPct, 70, 85))
        : '<span class="pill">RAM: n/a</span>';
    const gpu = hw.gpu_available
        ? `${hw.gpu_name || 'GPU'}${sys.gpu_vram_used_gb != null ? ` — ${sys.gpu_vram_used_gb.toFixed(2)} GB VRAM ใช้งาน` : ''}`
        : 'ไม่มี (CPU only)';
    const systemRow = document.getElementById('perf-system-row');
    if (systemRow) {
        systemRow.innerHTML =
            cpuBadge + ramBadge +
            `<span class="pill">GPU: ${gpu}</span>` +
            `<span class="pill">Hardware Tier: ${hw.tier || 'n/a'} (${hw.cpu_logical_cores || '?'} cores)</span>`;
    }

    const cameras = perf.cameras || [];

    // Primary view: one glanceable card per camera, key numbers colored by
    // whether they're in a range the system itself would call fine/strained/bad.
    const grid = document.getElementById('perf-camera-grid');
    if (grid) {
        grid.innerHTML = cameras.length ? '' : '<div class="list-empty">ยังไม่มีข้อมูลประสิทธิภาพกล้อง</div>';
        for (const cam of cameras) {
            const card = document.createElement('div');
            card.className = 'perf-card';
            const statusLabel = STATUS_LABELS_TH[cam.status] || cam.status;
            const fpsRatio = cam.requested_fps > 0 ? cam.actual_fps / cam.requested_fps : 1;
            const dropPct = cam.drop_rate * 100;
            const res = cam.resolution ? `${cam.resolution[0]}x${cam.resolution[1]}` : '-';
            card.innerHTML =
                `<div class="perf-card-header">` +
                `<span class="status-dot status-${cam.status}"></span>${cam.camera_id}` +
                `<span class="perf-status-text">${statusLabel}</span></div>` +
                `<div class="perf-metrics">` +
                `<div class="perf-metric"><div class="metric-value">` +
                metricBadge(`${cam.actual_fps.toFixed(1)} / ${cam.requested_fps.toFixed(1)}`, tierFor(fpsRatio, FPS_RATIO_WARN, FPS_RATIO_BAD, false)) +
                `</div><div class="metric-label">FPS จริง / ที่ขอ</div></div>` +
                `<div class="perf-metric"><div class="metric-value">${cam.ai_fps.toFixed(1)}</div><div class="metric-label">AI FPS</div></div>` +
                `<div class="perf-metric"><div class="metric-value">` +
                metricBadge(`${cam.avg_display_latency_ms.toFixed(0)} ms`, tierFor(cam.avg_display_latency_ms, DISPLAY_LATENCY_WARN_MS, DISPLAY_LATENCY_BAD_MS)) +
                `</div><div class="metric-label">Display Latency</div></div>` +
                `<div class="perf-metric"><div class="metric-value">` +
                metricBadge(`${dropPct.toFixed(1)}%`, tierFor(cam.drop_rate, DROP_RATE_WARN, DROP_RATE_BAD)) +
                `</div><div class="metric-label">เฟรมที่ทิ้ง (${cam.dropped_frames})</div></div>` +
                `</div>` +
                `<div class="perf-secondary">ความละเอียด ${res} · AI Latency ` +
                metricBadge(`${cam.avg_ai_latency_ms.toFixed(0)} ms`, tierFor(cam.avg_ai_latency_ms, AI_LATENCY_WARN_MS, AI_LATENCY_BAD_MS)) +
                ` · AI ทุก ${cam.detect_every_n_frames ?? '-'} เฟรม · AI Resolution ${cam.ai_imgsz ?? '-'}px</div>`;
            grid.appendChild(card);
        }
    }

    // Full raw numbers stay reachable in the collapsed <details> table (see
    // booth.html) for anyone who wants every column at once.
    const body = document.getElementById('perf-cameras-body');
    if (!body) return;
    body.innerHTML = '';
    for (const cam of cameras) {
        const tr = document.createElement('tr');
        const res = cam.resolution ? `${cam.resolution[0]}x${cam.resolution[1]}` : '-';
        const statusLabel = STATUS_LABELS_TH[cam.status] || cam.status;
        tr.innerHTML =
            `<td>${cam.camera_id}</td>` +
            `<td>${statusLabel}</td>` +
            `<td>${res}</td>` +
            `<td>${cam.requested_fps.toFixed(1)}</td>` +
            `<td>${cam.actual_fps.toFixed(1)}</td>` +
            `<td>${cam.ai_fps.toFixed(1)}</td>` +
            `<td>${cam.dropped_frames} (${(cam.drop_rate * 100).toFixed(1)}%)</td>` +
            `<td>${cam.avg_display_latency_ms.toFixed(0)} ms</td>` +
            `<td>${cam.avg_ai_latency_ms.toFixed(0)} ms</td>` +
            `<td>${cam.detect_every_n_frames ?? '-'}</td>` +
            `<td>${cam.ai_imgsz ?? '-'}px</td>`;
        body.appendChild(tr);
    }
}

async function pollPerformance() {
    try {
        const res = await fetch('/api/performance');
        renderPerformance(await res.json());
    } catch (e) { /* transient network hiccup — next poll will retry */ }
    setTimeout(pollPerformance, 2000);
}

pollPerformance();
