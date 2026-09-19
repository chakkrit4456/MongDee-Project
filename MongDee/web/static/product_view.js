let lastProductSeq = -1;

function speak(text) {
    try {
        if (!('speechSynthesis' in window)) return;
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = 'th-TH';
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utter);
    } catch (e) { /* TTS is a nice-to-have; ignore if the browser can't do it */ }
}

function addHistoryLine(iconName, text) {
    const list = document.getElementById('history-list');
    const line = document.createElement('div');
    const now = new Date().toLocaleTimeString('th-TH', { hour12: false });
    line.innerHTML = `${iconHtml(iconName, { size: 14, className: 'icon-inline' })}[${now}] ${text}`;
    list.insertBefore(line, list.firstChild);
    while (list.children.length > 100) list.removeChild(list.lastChild);
}

function renderProductInfo(p) {
    document.getElementById('product-name').textContent = p.name;
    document.getElementById('product-tagline').textContent = p.tagline || '';
    document.getElementById('product-price').textContent = p.price ? `ราคา: ${p.price}` : '';
    document.getElementById('product-desc').textContent = p.description || '';
    document.getElementById('product-source').textContent =
        `ตรวจพบโดยกล้อง: ${p.cameras} · ความมั่นใจ ${(p.confidence * 100).toFixed(0)}%`;

    const faqList = document.getElementById('faq-list');
    faqList.innerHTML = '';
    for (const entry of (p.faq || [])) {
        const div = document.createElement('div');
        div.className = 'faq-item';
        div.innerHTML = `<div class="faq-q">${entry.q}</div><div class="faq-a">${entry.a}</div>`;
        faqList.appendChild(div);
    }
}

// Camera panels here are rebuilt from the live status list on every poll
// instead of once from the server-rendered template — a fixed one-panel-
// per-camera-at-page-load grid meant a camera plugged in (or added via
// /settings) after this page opened had no panel until a manual reload,
// and one that dropped offline had no way to recover its <img> if the
// underlying connection actually died (see attachImgRecovery). Rebuilding
// is keyed off the sorted id list, so it's a no-op on every poll where
// nothing changed.
let renderedCameraIds = null;
let lastCameraStatus = {};

function panelHtml(camId) {
    return `
    <div class="panel cam-panel" id="panel-${camId}">
      <div class="cam-title">
        <span class="status-dot status-unknown" id="dot-${camId}"></span>
        ${camId}
        <div class="cam-actions">
          <button class="cam-icon-btn" data-action="popout" data-cam="${camId}" title="เปิดกล้องนี้ในหน้าต่างใหม่ (ย้ายไปแสดงอีกจอได้)">${iconHtml('openWindow', { size: 14, className: 'icon-inline' })}</button>
          <button class="cam-icon-btn" data-action="fullscreen" data-cam="${camId}" title="ดูกล้องนี้แบบเต็มจอ">${iconHtml('maximize', { size: 14, className: 'icon-inline' })}</button>
        </div>
      </div>
      <img src="/stream/${camId}" alt="${camId}" id="img-${camId}" data-cam="${camId}">
      <div class="status-text" id="status-${camId}">กำลังเชื่อมต่อ...</div>
    </div>`;
}

function attachImgRecovery(img) {
    img.addEventListener('error', () => {
        setTimeout(() => {
            if (img.isConnected) img.src = `/stream/${img.dataset.cam}?t=${Date.now()}`;
        }, 1500);
    });
}

function syncCameraGrid(cameraIds) {
    const key = cameraIds.slice().sort().join(',');
    if (key === renderedCameraIds) return;
    renderedCameraIds = key;
    const grid = document.getElementById('cam-grid');
    grid.innerHTML = cameraIds.length
        ? cameraIds.map(panelHtml).join('')
        : '<div class="panel cam-panel"><div class="status-text">ยังไม่พบกล้อง — เชื่อมต่อเว็บแคมแล้วรอสักครู่ ระบบจะตรวจพบอัตโนมัติ</div></div>';
    grid.querySelectorAll('img[id^="img-"]').forEach(attachImgRecovery);
}

function renderCameraStatus(cameras) {
    syncCameraGrid(Object.keys(cameras));
    for (const [camId, info] of Object.entries(cameras)) {
        const dot = document.getElementById(`dot-${camId}`);
        const statusText = document.getElementById(`status-${camId}`);
        if (dot) dot.className = 'status-dot status-' + info.status;
        if (statusText) statusText.textContent = info.message || info.status;

        const prevStatus = lastCameraStatus[camId];
        if (prevStatus && prevStatus !== 'online' && info.status === 'online') {
            const img = document.getElementById(`img-${camId}`);
            if (img) img.src = `/stream/${camId}?t=${Date.now()}`;
        }
        lastCameraStatus[camId] = info.status;
    }
}

async function pollState() {
    try {
        const res = await fetch('/api/state');
        const state = await res.json();
        renderCameraStatus(state.cameras);

        if (state.current_product) {
            const p = state.current_product;
            renderProductInfo(p);

            if (state.product_seq !== lastProductSeq) {
                lastProductSeq = state.product_seq;
                addHistoryLine('box', `พบสินค้า: ${p.name} (${p.cameras})`);
                speak(p.speak_text);
            }
        }
    } catch (e) { /* transient network hiccup — next poll will retry */ }
    setTimeout(pollState, 1500);
}

pollState();
