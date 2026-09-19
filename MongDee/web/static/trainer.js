let selectedKey = null;
let products = [];
let pollingProgress = false;
let recording = false;

const GUIDE_RECT = { x: 0.25, y: 0.20, w: 0.50, h: 0.60 };  // fraction of natural image size
const RECORD_TICK_MS = 500;
const RECORD_TICKS = 30;  // 30 x 0.5s = 15s — enough time for one slow, full 360° turn
const RECORD_COUNTDOWN_SEC = 3;  // pause before capture starts, so there's time to place the product
const LAST_CAMERA_KEY = 'mongdee_trainer_last_camera';

async function fetchProducts() {
    const res = await fetch('/api/products');
    products = await res.json();
    renderProductList();
}

function renderProductList() {
    const list = document.getElementById('product-list');
    list.innerHTML = '';
    for (const p of products) {
        const dotClass = p.sample_count >= RECOMMENDED_SAMPLES ? 'ready' : (p.sample_count > 0 ? 'partial' : 'empty');
        const div = document.createElement('div');
        div.className = 'product-list-item' + (p.key === selectedKey ? ' selected' : '');
        div.innerHTML =
            `<span><span class="sample-dot sample-dot-${dotClass}"></span>${p.name} (${p.sample_count} ภาพ)</span>` +
            `<button type="button" class="btn-danger btn-small product-delete-btn">${iconHtml('trash', { size: 13 })}</button>`;
        div.addEventListener('click', () => selectProduct(p.key));
        div.querySelector('.product-delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteProduct(p.key, p.name);
        });
        list.appendChild(div);
    }
    if (!selectedKey && products.length > 0) selectProduct(products[0].key);
}

function selectProduct(key) {
    selectedKey = key;
    renderProductList();
    const product = products.find(p => p.key === key);
    if (!product) return;

    document.getElementById('detail-empty').classList.add('hidden');
    document.getElementById('detail-content').classList.remove('hidden');
    document.getElementById('detail-title').textContent = `${product.name} · key: ${product.key}`;

    const n = product.sample_count;
    const pct = Math.min(100, Math.round(100 * n / RECOMMENDED_SAMPLES));
    document.getElementById('sample-progress').style.width = pct + '%';

    let status;
    if (n === 0) status = 'ยังไม่มีข้อมูลเทรน — อัปโหลดรูปภาพ วิดีโอ หรือบันทึกจากกล้องเพื่อเริ่มต้น';
    else if (n < RECOMMENDED_SAMPLES) status = `มีข้อมูล ${n} ภาพ — แนะนำให้อัปโหลดเพิ่มอีกอย่างน้อย ${RECOMMENDED_SAMPLES - n} ภาพ เพื่อความแม่นยำ`;
    else status = `มีข้อมูล ${n} ภาพ — พร้อมใช้งานแล้ว (อัปโหลดเพิ่มได้เสมอเพื่อความแม่นยำที่สูงขึ้น)`;
    document.getElementById('detail-status').textContent = status;

    document.getElementById('import-log').textContent = '';
    document.getElementById('import-progress-wrap').classList.add('hidden');
    document.getElementById('record-status').textContent = '';

    document.getElementById('edit-name').value = product.name || '';
    document.getElementById('edit-tagline').value = product.tagline || '';
    document.getElementById('edit-price').value = product.price || '';
    document.getElementById('edit-description').value = product.description || '';
    renderFaqRows('edit-faq-rows', product.faq || []);
}

// ---------------------------------------------------------------- FAQ rows
function renderFaqRows(containerId, faq) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    for (const entry of faq) addFaqRow(containerId, entry.q, entry.a);
}

function addFaqRow(containerId, q, a) {
    const container = document.getElementById(containerId);
    const row = document.createElement('div');
    row.className = 'faq-row';
    row.innerHTML =
        '<input type="text" class="faq-q-input" placeholder="คำถาม">' +
        '<input type="text" class="faq-a-input" placeholder="คำตอบ">' +
        '<button type="button" class="faq-remove-btn">' + iconHtml('x', { size: 14 }) + '</button>';
    row.querySelector('.faq-q-input').value = q || '';
    row.querySelector('.faq-a-input').value = a || '';
    row.querySelector('.faq-remove-btn').addEventListener('click', () => row.remove());
    container.appendChild(row);
}

function collectFaqRows(containerId) {
    const rows = document.querySelectorAll(`#${containerId} .faq-row`);
    const faq = [];
    for (const row of rows) {
        const q = row.querySelector('.faq-q-input').value.trim();
        const a = row.querySelector('.faq-a-input').value.trim();
        if (q && a) faq.push({ q, a });
    }
    return faq;
}

document.getElementById('new-add-faq-btn').addEventListener('click', () => addFaqRow('new-faq-rows'));
document.getElementById('edit-add-faq-btn').addEventListener('click', () => addFaqRow('edit-faq-rows'));

// ------------------------------------------------------- product JSON import
// Shared by both the "add new product" modal and the per-product "import
// files" control below -- one JSON shape understood everywhere a product's
// details can be filled in from a file instead of typed by hand.
function fillProductForm(prefix, product) {
    if (typeof product.name === 'string') document.getElementById(`${prefix}-name`).value = product.name;
    if (typeof product.tagline === 'string') document.getElementById(`${prefix}-tagline`).value = product.tagline;
    if (typeof product.price === 'string' || typeof product.price === 'number') document.getElementById(`${prefix}-price`).value = String(product.price);
    if (typeof product.description === 'string') document.getElementById(`${prefix}-description`).value = product.description;
    if (Array.isArray(product.faq)) renderFaqRows(`${prefix}-faq-rows`, product.faq.filter(x => x && x.q && x.a));
}

// A single product: either a flat {name, tagline, ...} object, or that same
// shape nested one level under an arbitrary wrapper key (e.g. {"product": {...}}).
function extractSingleProduct(data) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) return null;
    if (typeof data.name === 'string') return data;
    const keys = Object.keys(data);
    if (keys.length === 1 && data[keys[0]] && typeof data[keys[0]] === 'object' && typeof data[keys[0]].name === 'string') {
        return data[keys[0]];
    }
    return null;
}

// Multiple products in one file: either a JSON array of product objects, or
// a dict keyed by product key/name whose values are product objects.
function extractMultipleProducts(data) {
    if (Array.isArray(data)) {
        const items = data.filter(p => p && typeof p === 'object' && typeof p.name === 'string');
        return items.length ? items : null;
    }
    if (data && typeof data === 'object') {
        const items = Object.values(data).filter(p => p && typeof p === 'object' && typeof p.name === 'string');
        if (items.length > 1) return items;
    }
    return null;
}

function classifyImportFile(file) {
    const name = file.name.toLowerCase();
    if (file.type === 'application/json' || name.endsWith('.json')) return 'json';
    if (file.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|gif|heic|heif)$/.test(name)) return 'image';
    if (file.type.startsWith('video/') || /\.(mp4|webm|mov|avi|mkv|m4v)$/.test(name)) return 'video';
    return 'other';
}

// ------------------------------------------------------------- add product
document.getElementById('add-product-btn').addEventListener('click', () => {
    document.getElementById('new-name').value = '';
    document.getElementById('new-tagline').value = '';
    document.getElementById('new-price').value = '';
    document.getElementById('new-description').value = '';
    renderFaqRows('new-faq-rows', []);
    document.getElementById('new-import-json-log').textContent = '';
    document.getElementById('add-modal').classList.remove('hidden');
});

document.getElementById('new-import-json-input').addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    e.target.value = '';
    const logEl = document.getElementById('new-import-json-log');
    if (!files.length) return;

    const singleProducts = [];
    const parseErrors = [];
    for (const f of files) {
        let data;
        try {
            data = JSON.parse(await f.text());
        } catch (err) {
            parseErrors.push(`${f.name}: อ่านไฟล์ไม่สำเร็จ (${err.message})`);
            continue;
        }
        const many = extractMultipleProducts(data);
        if (many) {
            singleProducts.push(...many);
            continue;
        }
        const single = extractSingleProduct(data);
        if (single) singleProducts.push(single);
        else parseErrors.push(`${f.name}: รูปแบบ JSON ไม่ถูกต้อง (ต้องมีฟิลด์ "name")`);
    }

    if (!singleProducts.length) {
        logEl.textContent = parseErrors.join(' | ') || 'ไม่พบข้อมูลสินค้าในไฟล์';
        return;
    }

    if (singleProducts.length === 1 && files.length === 1) {
        // One file, one product -- prefill the form so the user can review
        // before confirming, same as typing it in by hand.
        fillProductForm('new', singleProducts[0]);
        logEl.textContent = `เติมฟอร์มจากไฟล์ ${files[0].name} แล้ว — ตรวจสอบแล้วกด "เพิ่มสินค้า" เพื่อยืนยัน` +
            (parseErrors.length ? ` | ${parseErrors.join(' | ')}` : '');
        return;
    }

    // Multiple products -- creating each through the modal one at a time
    // would be tedious, so import them directly via the same endpoint
    // "เพิ่มสินค้าใหม่" itself uses.
    if (!confirm(`พบข้อมูลสินค้า ${singleProducts.length} รายการ ต้องการนำเข้าทั้งหมดเลยหรือไม่?`)) return;
    let ok = 0, fail = 0;
    for (const product of singleProducts) {
        try {
            const res = await fetch('/api/products', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: product.name,
                    tagline: product.tagline || '',
                    price: product.price != null ? String(product.price) : '',
                    description: product.description || '',
                    faq: Array.isArray(product.faq) ? product.faq.filter(x => x && x.q && x.a) : [],
                }),
            });
            if (res.ok) ok++; else fail++;
        } catch (err) { fail++; }
    }
    logEl.textContent = `นำเข้าสำเร็จ ${ok} รายการ` + (fail ? `, ล้มเหลว ${fail} รายการ` : '') +
        (parseErrors.length ? ` | ข้ามไฟล์ที่อ่านไม่ได้: ${parseErrors.join(' | ')}` : '');
    if (ok) {
        document.getElementById('add-modal').classList.add('hidden');
        await fetchProducts();
    }
});

document.getElementById('add-confirm-btn').addEventListener('click', async () => {
    const name = document.getElementById('new-name').value.trim();
    if (!name) { alert('กรุณากรอกชื่อสินค้า'); return; }
    const res = await fetch('/api/products', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name,
            tagline: document.getElementById('new-tagline').value.trim(),
            price: document.getElementById('new-price').value.trim(),
            description: document.getElementById('new-description').value.trim(),
            faq: collectFaqRows('new-faq-rows'),
        }),
    });
    const data = await res.json();
    document.getElementById('add-modal').classList.add('hidden');
    selectedKey = data.key;
    await fetchProducts();
});

// ------------------------------------------------------------ edit product
document.getElementById('edit-save-btn').addEventListener('click', async () => {
    if (!selectedKey) return;
    const name = document.getElementById('edit-name').value.trim();
    if (!name) { alert('กรุณากรอกชื่อสินค้า'); return; }
    await fetch(`/api/products/${selectedKey}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name,
            tagline: document.getElementById('edit-tagline').value.trim(),
            price: document.getElementById('edit-price').value.trim(),
            description: document.getElementById('edit-description').value.trim(),
            faq: collectFaqRows('edit-faq-rows'),
        }),
    });
    await fetchProducts();
    selectProduct(selectedKey);
});

// -------------------------------------------------------------- upload
document.getElementById('import-files-input').addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    e.target.value = '';
    if (!files.length || !selectedKey) return;

    const images = [], videos = [], jsonFiles = [], others = [];
    for (const f of files) {
        const kind = classifyImportFile(f);
        if (kind === 'image') images.push(f);
        else if (kind === 'video') videos.push(f);
        else if (kind === 'json') jsonFiles.push(f);
        else others.push(f);
    }

    const notes = [];
    if (others.length) notes.push(`ข้ามไฟล์ที่ไม่รองรับ ${others.length} ไฟล์ (${others.map(f => f.name).join(', ')})`);

    if (jsonFiles.length) {
        if (jsonFiles.length > 1) notes.push(`พบไฟล์ .json หลายไฟล์ — ใช้เฉพาะไฟล์แรก (${jsonFiles[0].name})`);
        try {
            const data = JSON.parse(await jsonFiles[0].text());
            const product = extractSingleProduct(data);
            if (!product) throw new Error('ไม่พบฟิลด์ "name" ในไฟล์');
            fillProductForm('edit', product);
            notes.push(`เติมฟอร์มรายละเอียดสินค้าจาก ${jsonFiles[0].name} แล้ว — กด "บันทึกรายละเอียด" เพื่อยืนยัน`);
        } catch (err) {
            notes.push(`อ่านไฟล์ ${jsonFiles[0].name} ไม่สำเร็จ: ${err.message}`);
        }
    }
    if (images.length || videos.length) notes.push('กำลังประมวลผลรูปภาพ/วิดีโอ...');
    if (notes.length) document.getElementById('import-log').textContent = notes.join(' | ');

    // Images/videos must be sent one import job at a time -- the backend
    // rejects a second import for the same product while one is still
    // running (see web/booth_manager.py's start_image_import/start_video_import),
    // so each call below is fully awaited before the next one starts.
    if (images.length) {
        const form = new FormData();
        for (const f of images) form.append('files', f);
        await startImport(`/api/products/${selectedKey}/upload_images`, form);
    }
    for (const v of videos) {
        const form = new FormData();
        form.append('file', v);
        await startImport(`/api/products/${selectedKey}/upload_video`, form);
    }
});

async function startImport(url, form) {
    document.getElementById('import-progress-wrap').classList.remove('hidden');
    document.getElementById('import-progress').style.width = '0%';

    const res = await fetch(url, { method: 'POST', body: form });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'ล้มเหลว' }));
        document.getElementById('import-log').textContent = `ล้มเหลว: ${err.detail}`;
        document.getElementById('import-progress-wrap').classList.add('hidden');
        return;
    }
    await pollImportProgress();
}

// Polls until the current product's import job reaches a terminal state
// (done/error), updating the progress UI along the way, then resolves --
// callers that need to run imports one after another (see above) can
// `await` this instead of firing it and moving on.
async function pollImportProgress() {
    if (!selectedKey) return;
    if (pollingProgress) {
        while (pollingProgress) await sleep(200);
        return;
    }
    pollingProgress = true;
    try {
        while (true) {
            const res = await fetch(`/api/products/${selectedKey}/import_progress`);
            const progress = await res.json();

            if (progress.status === 'running') {
                const pct = progress.total > 0 ? Math.round(100 * progress.done / progress.total) : 0;
                document.getElementById('import-progress').style.width = pct + '%';
                document.getElementById('import-log').textContent = `ประมวลผลแล้ว ${progress.done}/${progress.total}`;
                await sleep(500);
                continue;
            }
            if (progress.status === 'done') {
                document.getElementById('import-log').textContent = `เพิ่มข้อมูลเทรนสำเร็จ ${progress.done} ภาพ`;
                document.getElementById('import-progress-wrap').classList.add('hidden');
                await fetchProducts();
                selectProduct(selectedKey);
            } else if (progress.status === 'error') {
                document.getElementById('import-log').textContent = `ล้มเหลว: ${progress.message || ''}`;
                document.getElementById('import-progress-wrap').classList.add('hidden');
            }
            return;
        }
    } finally {
        pollingProgress = false;
    }
}

async function deleteProduct(key, name) {
    if (!confirm(`ลบสินค้า "${name}" ออกจากระบบทั้งหมด (รวมข้อมูลเทรน) หรือไม่? การกระทำนี้ย้อนกลับไม่ได้`)) return;
    await fetch(`/api/products/${key}`, { method: 'DELETE' });
    if (selectedKey === key) {
        selectedKey = null;
        document.getElementById('detail-empty').classList.remove('hidden');
        document.getElementById('detail-content').classList.add('hidden');
    }
    await fetchProducts();
}

document.getElementById('clear-samples-btn').addEventListener('click', async () => {
    if (!selectedKey) return;
    const product = products.find(p => p.key === selectedKey);
    if (!confirm(`ลบข้อมูลเทรนทั้งหมดของ "${product.name}" หรือไม่?`)) return;
    await fetch(`/api/products/${selectedKey}/clear_samples`, { method: 'POST' });
    await fetchProducts();
    selectProduct(selectedKey);
});

// -------------------------------------------------------- record from cam
function initRecordCamera() {
    const select = document.getElementById('record-camera-select');
    for (const camId of CAMERA_IDS) {
        const opt = document.createElement('option');
        opt.value = camId; opt.textContent = camId;
        select.appendChild(opt);
    }
    const updateImg = () => {
        document.getElementById('record-camera-img').src = '/stream/' + select.value + '?t=' + Date.now();
        try { localStorage.setItem(LAST_CAMERA_KEY, select.value); } catch (e) { /* private mode etc. */ }
    };
    select.addEventListener('change', updateImg);
    if (CAMERA_IDS.length > 0) {
        // If multiple cameras are configured, restore whichever one was used
        // last time instead of always defaulting back to the first.
        let lastCamera = null;
        try { lastCamera = localStorage.getItem(LAST_CAMERA_KEY); } catch (e) { /* ignore */ }
        if (lastCamera && CAMERA_IDS.includes(lastCamera)) select.value = lastCamera;
        updateImg();
    }
}

document.getElementById('record-start-btn').addEventListener('click', startRecording);

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function startRecording() {
    if (recording || !selectedKey) return;
    const img = document.getElementById('record-camera-img');
    if (!img.naturalWidth) {
        document.getElementById('record-status').textContent = 'ยังไม่พบภาพจากกล้อง กรุณารอสักครู่';
        return;
    }
    recording = true;
    const btn = document.getElementById('record-start-btn');
    btn.disabled = true;

    const guide = document.getElementById('record-guide');
    const guideLabel = document.getElementById('record-guide-label');
    const ring = document.getElementById('record-ring');
    const countdownEl = document.getElementById('record-countdown');
    const angleStatus = document.getElementById('record-angle-status');
    const statusEl = document.getElementById('record-status');

    let blobs = [];
    try {
        // Pre-roll countdown: gives the user time to center the product in
        // the guide frame before any frame is actually captured.
        guideLabel.textContent = 'เตรียมสินค้าให้อยู่กึ่งกลางกรอบ...';
        countdownEl.classList.remove('hidden');
        for (let s = RECORD_COUNTDOWN_SEC; s > 0; s--) {
            countdownEl.textContent = String(s);
            statusEl.textContent = `เริ่มบันทึกใน ${s}...`;
            await sleep(1000);
        }
        countdownEl.classList.add('hidden');

        guide.classList.add('recording');
        guideLabel.textContent = 'หมุนสินค้าช้าๆ ให้อยู่ในกรอบนี้ตลอด';
        ring.classList.add('active');
        ring.style.setProperty('--progress', '0');

        const canvas = document.getElementById('record-canvas');
        const sx = img.naturalWidth * GUIDE_RECT.x;
        const sy = img.naturalHeight * GUIDE_RECT.y;
        const sw = img.naturalWidth * GUIDE_RECT.w;
        const sh = img.naturalHeight * GUIDE_RECT.h;
        canvas.width = sw;
        canvas.height = sh;
        const ctx = canvas.getContext('2d');

        for (let tick = 0; tick < RECORD_TICKS; tick++) {
            ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
            const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
            if (blob) blobs.push(blob);

            // Assume an even rotation speed across the whole take, so the ring
            // sweep and angle readout double as a pacing guide: reaching ~360°
            // right as the last frame is captured means "about right" speed.
            const fraction = (tick + 1) / RECORD_TICKS;
            const angle = Math.round(fraction * 360);
            ring.style.setProperty('--progress', String(angle));
            angleStatus.textContent = angle >= 355
                ? `หมุนครบรอบแล้ว (~360°) — เก็บภาพเกือบครบแล้ว`
                : `หมุนสินค้าต่อไปเรื่อยๆ — ประมาณ ${angle}° / 360°`;
            statusEl.textContent =
                `กำลังบันทึก... ${tick + 1}/${RECORD_TICKS} ภาพ (${((RECORD_TICKS - tick - 1) * RECORD_TICK_MS / 1000).toFixed(1)} วินาทีที่เหลือ)`;
            if (tick < RECORD_TICKS - 1) await sleep(RECORD_TICK_MS);
        }
    } finally {
        countdownEl.classList.add('hidden');
        guide.classList.remove('recording');
        guideLabel.textContent = 'วางสินค้าในกรอบนี้';
        ring.classList.remove('active');
        angleStatus.textContent = '';
        btn.disabled = false;
        recording = false;
    }

    if (!blobs.length) {
        statusEl.textContent = 'บันทึกภาพไม่สำเร็จ กรุณาลองใหม่';
        return;
    }
    statusEl.textContent = `บันทึกครบ ${blobs.length} ภาพ (รอบตัว 360°) กำลังส่งประมวลผล...`;
    const form = new FormData();
    blobs.forEach((blob, i) => form.append('files', blob, `record-${i}.jpg`));
    await startImport(`/api/products/${selectedKey}/upload_images`, form);
}

initRecordCamera();
fetchProducts();
setInterval(fetchProducts, 8000);
