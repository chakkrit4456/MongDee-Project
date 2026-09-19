let activeBoothId = null;

function showError(msg) {
    let banner = document.getElementById('settings-error-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'settings-error-banner';
        banner.className = 'error-banner';
        document.body.insertBefore(banner, document.body.firstChild);
    }
    banner.textContent = msg;
    banner.classList.remove('hidden');
}

function clearError() {
    const banner = document.getElementById('settings-error-banner');
    if (banner) banner.classList.add('hidden');
}

// Every fetch on this page goes through here so a stale server process
// (page refreshed, but `python web_server.py` itself never restarted, so a
// newly-added API route doesn't exist yet) shows a clear message instead of
// the button just silently doing nothing.
async function apiFetch(url, options) {
    let res;
    try {
        res = await fetch(url, options);
    } catch (e) {
        showError(`เชื่อมต่อเซิร์ฟเวอร์ไม่ได้: ${e.message}`);
        throw e;
    }
    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
            const body = await res.clone().json();
            if (body && body.detail) detail = body.detail;
        } catch (_) { /* body wasn't JSON, keep the HTTP status */ }
        // FastAPI/Starlette's own catch-all for "no route matched at all" always
        // has this exact literal body — a legitimate app-level 404 (e.g. "ไม่พบ
        // บูธนี้" from a real HTTPException) always carries its own specific
        // Thai detail instead, so this reliably means the endpoint itself
        // doesn't exist in the server that's currently running.
        if (res.status === 404 && detail === 'Not Found') {
            showError(`ไม่พบฟีเจอร์นี้ในเซิร์ฟเวอร์ที่กำลังรันอยู่ (${url}) — โค้ดฝั่งเซิร์ฟเวอร์อาจยังไม่ได้อัปเดต ` +
                `กรุณาปิดหน้าต่าง/เทอร์มินัลที่รัน python web_server.py แล้วรันคำสั่งนี้ใหม่ (แค่รีเฟรชหน้าเว็บไม่พอ)`);
        } else {
            showError(`เกิดข้อผิดพลาด: ${detail}`);
        }
        throw new Error(detail);
    }
    clearError();
    return res;
}

async function loadSettings() {
    const res = await apiFetch('/api/booth/settings');
    const settings = await res.json();
    activeBoothId = settings.booth_id;
    renderCameraList(settings.cameras);
    await Promise.all([loadEvents(), loadBooths()]);
}

// -------------------------------------------------------------- events
async function loadEvents() {
    const res = await apiFetch('/api/registry/events');
    const events = await res.json();
    renderEventsTable(events);
    renderEventOptions(events);
}

function renderEventsTable(events) {
    const body = document.getElementById('events-body');
    body.innerHTML = '';
    if (!events.length) {
        body.innerHTML = '<tr><td colspan="4" style="color:var(--text-faint);">ยังไม่มี Event</td></tr>';
        return;
    }
    for (const ev of events) {
        const tr = document.createElement('tr');
        tr.innerHTML =
            `<td>${ev.id}</td><td>${ev.name}</td><td>${ev.booth_count}</td>` +
            `<td><button class="btn-danger btn-small event-delete-btn">${iconHtml('trash', { size: 13 })}</button></td>`;
        tr.querySelector('.event-delete-btn').addEventListener('click', () => deleteEvent(ev.id, ev.name));
        body.appendChild(tr);
    }
}

function renderEventOptions(events) {
    const select = document.getElementById('new-booth-event');
    select.innerHTML = '<option value="">ไม่มี</option>';
    for (const ev of events) {
        const opt = document.createElement('option');
        opt.value = ev.id; opt.textContent = ev.name;
        select.appendChild(opt);
    }
}

document.getElementById('add-event-btn').addEventListener('click', async () => {
    const input = document.getElementById('new-event-name');
    const name = input.value.trim();
    if (!name) return;
    try {
        await apiFetch('/api/registry/events', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
    } catch (e) { return; }
    input.value = '';
    await loadEvents();
    await loadBooths();
});

async function deleteEvent(eventId, name) {
    if (!confirm(`ลบ Event "${name}" หรือไม่? บูธที่สังกัดอยู่จะกลายเป็นไม่มี Event (ไม่ถูกลบ) การกระทำนี้ย้อนกลับไม่ได้`)) return;
    try {
        await apiFetch(`/api/registry/events/${eventId}`, { method: 'DELETE' });
    } catch (e) { return; }
    await loadSettings();
}

// -------------------------------------------------------------- booths
async function loadBooths() {
    const res = await apiFetch('/api/registry/booths');
    const booths = await res.json();
    renderBoothsTable(booths);
}

function renderBoothsTable(booths) {
    const body = document.getElementById('booths-body');
    body.innerHTML = '';
    if (!booths.length) {
        body.innerHTML = '<tr><td colspan="6" style="color:var(--text-faint);">ยังไม่มีบูธ</td></tr>';
        return;
    }
    for (const b of booths) {
        const tr = document.createElement('tr');
        const statusBadge = b.active
            ? '<span class="pill ready">กำลังใช้งาน</span>'
            : '<button class="btn-small activate-btn">ตั้งเป็นบูธที่ใช้งานอยู่</button>';
        const deleteBtn = b.active
            ? `<button class="btn-danger btn-small" disabled title="ลบบูธที่กำลังใช้งานอยู่ไม่ได้">${iconHtml('trash', { size: 13 })}</button>`
            : `<button class="btn-danger btn-small booth-delete-btn">${iconHtml('trash', { size: 13 })}</button>`;
        const scheduleCell = document.createElement('td');
        const openInput = document.createElement('input');
        openInput.type = 'time'; openInput.value = b.open_time || ''; openInput.style.width = '110px';
        const closeInput = document.createElement('input');
        closeInput.type = 'time'; closeInput.value = b.close_time || ''; closeInput.style.width = '110px';
        const saveBtn = document.createElement('button');
        saveBtn.className = 'btn-small'; saveBtn.title = 'บันทึกเวลาเปิด-ปิดบูธนี้';
        saveBtn.innerHTML = iconHtml('check', { size: 13 });
        saveBtn.addEventListener('click', () => saveBoothSchedule(b.id, openInput.value, closeInput.value));
        scheduleCell.style.cssText = 'display:flex; gap:4px; align-items:center;';
        scheduleCell.append(openInput, document.createTextNode('–'), closeInput, saveBtn);

        tr.innerHTML =
            `<td>${b.id}</td><td>${b.name}</td><td>${b.event_name || 'ไม่มี'}</td>` +
            `<td></td><td>${statusBadge}</td><td>${deleteBtn}</td>`;
        tr.children[3].replaceWith(scheduleCell);
        if (!b.active) {
            tr.querySelector('.activate-btn').addEventListener('click', () => activateBooth(b.id));
            tr.querySelector('.booth-delete-btn').addEventListener('click', () => deleteBooth(b.id, b.name));
        }
        body.appendChild(tr);
    }
}

async function saveBoothSchedule(boothId, openTime, closeTime) {
    const status = document.getElementById('registry-status');
    try {
        await apiFetch(`/api/registry/booths/${boothId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ open_time: openTime || '', close_time: closeTime || '' }),
        });
    } catch (e) { return; }
    status.textContent = 'บันทึกเวลาเปิด-ปิดบูธแล้ว';
    setTimeout(() => { status.textContent = ''; }, 2000);
    await loadBooths();
}

document.getElementById('add-booth-btn').addEventListener('click', async () => {
    const name = document.getElementById('new-booth-name').value.trim();
    if (!name) return;
    const eventId = document.getElementById('new-booth-event').value;
    try {
        await apiFetch('/api/registry/booths', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, event_id: eventId || null }),
        });
    } catch (e) { return; }
    document.getElementById('new-booth-name').value = '';
    await loadBooths();
    await loadEvents();
});

async function activateBooth(boothId) {
    const status = document.getElementById('registry-status');
    status.textContent = 'กำลังเปลี่ยนบูธที่ใช้งาน...';
    try {
        await apiFetch(`/api/registry/booths/${boothId}/activate`, { method: 'POST' });
    } catch (e) { status.textContent = ''; return; }
    status.textContent = 'เปลี่ยนบูธที่ใช้งานแล้ว';
    setTimeout(() => { status.textContent = ''; }, 2000);
    await loadSettings();
}

async function deleteBooth(boothId, name) {
    if (!confirm(`ลบบูธ "${name}" และข้อมูลที่บันทึกไว้ทั้งหมดของบูธนี้หรือไม่? การกระทำนี้ย้อนกลับไม่ได้`)) return;
    try {
        await apiFetch(`/api/registry/booths/${boothId}`, { method: 'DELETE' });
    } catch (e) { return; }
    await loadBooths();
    await loadEvents();
}

// -------------------------------------------------------------- cameras
function renderCameraList(cameras) {
    const list = document.getElementById('camera-list');
    list.innerHTML = '';
    if (!cameras.length) {
        list.innerHTML = '<div style="color:var(--text-faint);">ยังไม่มีกล้อง</div>';
        return;
    }
    for (const cam of cameras) {
        const enabled = cam.enabled !== false;
        const row = document.createElement('div');
        row.className = 'camera-row' + (enabled ? '' : ' camera-row-off');
        row.innerHTML =
            `<span>${cam.camera_id} <span style="color:var(--text-faint);">(${cam.device})</span>` +
            (enabled ? '' : ' <span style="color:var(--text-faint);">— ปิดใช้งานอยู่</span>') + `</span>` +
            `<div class="camera-row-actions">` +
            `<button class="btn-small toggle-camera-btn">` + iconHtml('plug', { size: 14 }) +
            (enabled ? 'ปิดกล้อง' : 'เปิดกล้อง') + `</button>` +
            `<button class="btn-danger btn-small" data-camera-id="${cam.camera_id}">` +
            iconHtml('trash', { size: 14 }) + 'ลบ</button>' +
            `</div>`;
        row.querySelector('.toggle-camera-btn').addEventListener('click', () => setCameraEnabled(cam.camera_id, !enabled));
        row.querySelector('.btn-danger').addEventListener('click', () => removeCamera(cam.camera_id));
        list.appendChild(row);
    }
}

async function setCameraEnabled(cameraId, enabled) {
    try {
        await apiFetch(`/api/booth/cameras/${cameraId}/enabled`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
    } catch (e) { return; }
    await loadSettings();
}

document.getElementById('add-camera-btn').addEventListener('click', async () => {
    const input = document.getElementById('new-camera-device');
    const device = input.value.trim();
    if (!device) return;
    const status = document.getElementById('camera-status');
    status.textContent = 'กำลังเพิ่มกล้อง...';
    try {
        await apiFetch('/api/booth/cameras', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device }),
        });
    } catch (e) {
        status.textContent = `ล้มเหลว: ${e.message}`;
        return;
    }
    input.value = '';
    status.textContent = 'เพิ่มกล้องแล้ว — ระบบกำลังเชื่อมต่อ';
    await loadSettings();
});

async function removeCamera(cameraId) {
    if (!confirm(`ลบกล้อง ${cameraId} ออกจากบูธนี้หรือไม่?`)) return;
    try {
        await apiFetch(`/api/booth/cameras/${cameraId}`, { method: 'DELETE' });
    } catch (e) { return; }
    await loadSettings();
}

// -------------------------------------------------------------- GPU
// Acceleration is now installed automatically (see requirements.txt /
// install.bat/.sh) — this panel is read-only status, nothing to trigger here.
async function loadGpuStatus() {
    try {
        const res = await apiFetch('/api/system/gpu_status');
        const status = await res.json();
        const el = document.getElementById('gpu-status');
        el.innerHTML = status.using_gpu
            ? `<span class="pill ready">${iconHtml('check', { size: 13 })}กำลังใช้ GPU: ${status.device}</span>`
            : `<span class="pill not-ready">${iconHtml('chip', { size: 13 })}ยังไม่ได้ใช้ GPU (${status.device})</span>`;
    } catch (e) { /* apiFetch already showed the error banner */ }
}

loadGpuStatus();

document.getElementById('reset-data-btn').addEventListener('click', async () => {
    if (!confirm('ลบข้อมูลที่บันทึกไว้ทั้งหมดของบูธนี้ (ย้อนกลับไม่ได้) ยืนยันหรือไม่?')) return;
    if (!confirm('ยืนยันอีกครั้ง: ข้อมูลทั้งหมดของบูธนี้จะถูกลบถาวร ต้องการดำเนินการต่อหรือไม่?')) return;
    try {
        await apiFetch('/api/booth/reset_data', { method: 'POST' });
    } catch (e) { return; }
    alert('ลบข้อมูลบูธนี้เรียบร้อยแล้ว');
});

loadSettings();
