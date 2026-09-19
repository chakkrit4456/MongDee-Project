const captureBtn = document.getElementById("captureBtn");
const moreBtn = document.getElementById("moreBtn");
const captureStatus = document.getElementById("captureStatus");
const frameGrid = document.getElementById("frameGrid");
const selectionCount = document.getElementById("selectionCount");
const minCountEl = document.getElementById("minCount");
const targetCountEl = document.getElementById("targetCount");
const formCard = document.getElementById("formCard");
const placeholderCard = document.getElementById("placeholderCard");
const placeholderText = document.getElementById("placeholderText");
const productForm = document.getElementById("productForm");
const cancelBtn = document.getElementById("cancelBtn");
const toast = document.getElementById("toast");
const savedResult = document.getElementById("savedResult");
const savedMessage = document.getElementById("savedMessage");
const catalogList = document.getElementById("enrolledProducts");
const catalogStatus = document.getElementById("catalogStatus");
const catalogCount = document.getElementById("catalogCount");
let busy = false;

function setBusy(value) {
  busy = value;
  captureBtn.disabled = moreBtn.disabled = cancelBtn.disabled = value;
  document.getElementById("cameraSelectBtn").disabled = value;
  document.getElementById("cameraSelect").disabled = value;
  productForm.querySelector('button[type="submit"]').disabled = value;
  document.getElementById("nextProductBtn").disabled = value;
}

function errorMessage(data, fallback) {
  if (typeof data.detail === "string") return data.detail;
  return fallback;
}

const MIN_IMAGES = 3;
const TARGET_IMAGES = 5;
minCountEl.textContent = MIN_IMAGES;
targetCountEl.textContent = TARGET_IMAGES;

let currentSessionId = null;
let frames = []; // [{frame_id, thumbnail_data_url}]
let selected = new Set();

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function renderGrid() {
  frameGrid.replaceChildren();
  for (const f of frames) {
    const tile = el("div", "frame-tile" + (selected.has(f.frame_id) ? " selected" : ""));
    const img = el("img");
    img.src = f.thumbnail_data_url;
    const check = el("div", "check");
    check.textContent = "✓";
    tile.append(img, check);
    tile.onclick = () => {
      if (busy) return;
      if (selected.has(f.frame_id)) selected.delete(f.frame_id);
      else selected.add(f.frame_id);
      renderGrid();
    };
    frameGrid.append(tile);
  }

  if (frames.length) {
    selectionCount.textContent = `เลือกแล้ว ${selected.size} / ${frames.length} รูป (ต้องการอย่างน้อย ${MIN_IMAGES} รูป)`;
  } else {
    selectionCount.textContent = "";
  }
  updateFormVisibility();
}

function updateFormVisibility() {
  if (selected.size >= MIN_IMAGES) {
    formCard.style.display = "block";
    placeholderCard.style.display = "none";
  } else {
    formCard.style.display = "none";
    placeholderCard.style.display = "block";
    placeholderText.textContent = frames.length
      ? `เลือกภาพที่ใช้งานได้อีกอย่างน้อย ${MIN_IMAGES - selected.size} รูปก่อนกรอกข้อมูลสินค้า`
      : 'วางสินค้าแล้วกด “สแกนสินค้าเพื่อเพิ่มเข้ารายการ” จากนั้นเลือกภาพและกรอกข้อมูล';
  }
}

function applyBatch(sessionId, newFrames) {
  currentSessionId = sessionId;
  frames = frames.concat(newFrames);
  for (const f of newFrames) selected.add(f.frame_id); // reviewed by deselecting, not selecting
  renderGrid();
  moreBtn.style.display = "inline-block";
}

async function runCapture(url, replaceBatch = false) {
  if (busy) return;
  setBusy(true);
  captureStatus.textContent = "กำลังเก็บภาพ... วางสินค้าไว้บนแท่นและอย่าขยับกล้อง (ประมาณ 7 วินาที)";
  try {
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(errorMessage(err, "เก็บภาพไม่สำเร็จ"));
    }
    const data = await res.json();
    if (replaceBatch) {
      const previousSessionId = currentSessionId;
      frames = [];
      selected = new Set();
      if (previousSessionId && previousSessionId !== data.session_id) {
        fetch(`/api/enroll/${previousSessionId}`, { method: "DELETE" }).catch(() => {});
      }
    }
    savedResult.style.display = "none";
    applyBatch(data.session_id, data.frames);
    captureStatus.textContent = `เก็บภาพได้เพิ่ม ${data.frames.length} รูป — คลิกรูปที่ไม่ชัด/ไม่ต้องการเพื่อเอาออก`;
  } catch (err) {
    captureStatus.textContent = "เกิดข้อผิดพลาด: " + err.message;
  } finally {
    setBusy(false);
  }
}

captureBtn.addEventListener("click", () => {
  runCapture("/api/enroll/start", true);
});

moreBtn.addEventListener("click", () => {
  if (!currentSessionId) return;
  runCapture(`/api/enroll/${currentSessionId}/more`);
});

cancelBtn.addEventListener("click", async () => {
  if (busy) return;
  setBusy(true);
  try {
    if (currentSessionId) await fetch(`/api/enroll/${currentSessionId}`, { method: "DELETE" });
    resetAll();
  } catch (err) {
    toast.textContent = "ยกเลิกไม่สำเร็จ กรุณาลองใหม่";
  } finally { setBusy(false); }
});

productForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy || !currentSessionId || selected.size < MIN_IMAGES) return;

  const formData = new FormData(productForm);
  const body = Object.fromEntries(formData.entries());
  body.price = body.price === "" ? null : parseFloat(body.price);
  if (body.production_date === "") body.production_date = null;
  if (body.expiry_date === "") body.expiry_date = null;
  body.selected_frame_ids = Array.from(selected);
  body.name = body.name.trim();
  if (!body.name) {
    toast.textContent = "กรุณากรอกชื่อสินค้า";
    return;
  }

  setBusy(true);
  try {
    const res = await fetch(`/api/enroll/${currentSessionId}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(errorMessage(err, "บันทึกไม่สำเร็จ"));
    }
    const product = await res.json();
    resetAll();
    savedMessage.textContent = `เพิ่ม “${product.name}” เข้ารายการสินค้าแล้ว พร้อมใช้เป็นตัวอย่างสำหรับการจดจำสินค้า`;
    savedResult.style.display = "block";
    await loadCatalog();
  } catch (err) {
    toast.className = "toast error";
    toast.textContent = "เกิดข้อผิดพลาด: " + err.message;
  } finally {
    setBusy(false);
  }
});

function resetAll() {
  currentSessionId = null;
  frames = [];
  selected = new Set();
  productForm.reset();
  moreBtn.style.display = "none";
  captureStatus.textContent = "";
  toast.textContent = "";
  renderGrid();
}

async function loadFromQuerySession() {
  const params = new URLSearchParams(location.search);
  const sid = params.get("session");
  if (!sid) return;

  captureStatus.textContent = "กำลังโหลดภาพจากการสแกนล่าสุด...";
  try {
    const res = await fetch(`/api/enroll/${sid}`);
    if (!res.ok) throw new Error("ไม่พบข้อมูลภาพจากการสแกนนั้นแล้ว กรุณาเริ่มบันทึกใหม่");
    const data = await res.json();
    applyBatch(data.session_id, data.frames);
    captureStatus.textContent = "โหลดภาพจากการสแกนล่าสุดแล้ว เลือกภาพที่ใช้งานได้แล้วกรอกข้อมูลด้านขวา";
  } catch (err) {
    captureStatus.textContent = err.message;
  }
}

renderGrid();
loadFromQuerySession();

// -------- Camera selection --------

const cameraSelect = document.getElementById("cameraSelect");
const cameraSelectBtn = document.getElementById("cameraSelectBtn");
const cameraSelectStatus = document.getElementById("cameraSelectStatus");
const camFeed = document.getElementById("camFeed");

async function loadCameraDevices() {
  try {
    const res = await fetch("/api/camera/devices");
    const data = await res.json();
    cameraSelect.replaceChildren();
    for (const d of data.devices) {
      const opt = document.createElement("option");
      opt.value = d.index;
      const resText = d.width ? ` ${d.width}x${d.height}` : "";
      opt.textContent = `กล้อง ${d.index}${resText}${d.index === data.current_index ? " (กำลังใช้งาน)" : ""}`;
      if (d.index === data.current_index) opt.selected = true;
      cameraSelect.append(opt);
    }
    if (!data.devices.length) {
      cameraSelectStatus.textContent = "ไม่พบกล้องในระบบ";
    }
  } catch (err) {
    cameraSelectStatus.textContent = "โหลดรายชื่อกล้องไม่สำเร็จ";
  }
}

cameraSelectBtn.addEventListener("click", async () => {
  const index = parseInt(cameraSelect.value, 10);
  if (Number.isNaN(index)) return;
  cameraSelectBtn.disabled = true;
  cameraSelectStatus.textContent = "กำลังเปลี่ยนกล้อง...";
  try {
    const res = await fetch("/api/camera/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "เปลี่ยนกล้องไม่สำเร็จ");
    }
    cameraSelectStatus.textContent = "เปลี่ยนกล้องแล้ว";
    camFeed.src = "/api/camera/stream?t=" + Date.now();
    await loadCameraDevices();
  } catch (err) {
    cameraSelectStatus.textContent = "เกิดข้อผิดพลาด: " + err.message;
  } finally {
    cameraSelectBtn.disabled = false;
  }
});

loadCameraDevices();

// -------- Live product-detection status + bounding box --------

const detectBadge = document.getElementById("detectBadge");
const detectBox = document.getElementById("detectBox");
const detectLabel = document.getElementById("detectLabel");

async function pollDetectStatus() {
  try {
    const res = await fetch("/api/camera/status");
    const status = await res.json();
    if (!status.camera_open) {
      detectBadge.textContent = "ไม่พบภาพสดจากกล้อง";
      detectBadge.className = "pill warn";
    } else if (!status.has_reference) {
      detectBadge.textContent = "ยังไม่ได้ตั้งค่าพื้นเปล่า — ไปที่หน้าปรับตั้งค่าก่อนสแกน";
      detectBadge.className = "pill warn";
    } else if (status.state === "present") {
      detectBadge.textContent = "ตรวจพบสินค้าในกรอบแล้ว พร้อมสแกน";
      detectBadge.className = "pill ok";
    } else {
      detectBadge.textContent = "ยังไม่พบสินค้า — วางสินค้าให้อยู่กึ่งกลางกรอบ";
      detectBadge.className = "pill";
    }

    applyDetectBoxStatus(detectBox, detectLabel, status);
  } catch (err) {
    detectBadge.textContent = "เชื่อมต่อสถานะกล้องไม่ได้";
    detectBadge.className = "pill warn";
    detectBox.className = "detect-box";
    detectLabel.textContent = "";
  }
}

positionDetectBox(detectBox);
pollDetectStatus();
setInterval(pollDetectStatus, 400);

async function loadCatalog() {
  const refresh = document.getElementById("refreshCatalogBtn");
  refresh.disabled = true;
  catalogStatus.textContent = "กำลังโหลดรายการสินค้า...";
  try {
    const res = await fetch("/api/products", { cache: "no-store" });
    if (!res.ok) throw new Error("โหลดรายการสินค้าไม่สำเร็จ กดรีเฟรชเพื่อลองใหม่");
    const products = await res.json();
    catalogList.replaceChildren();
    catalogCount.textContent = `(${products.length})`;
    for (const product of products) {
      const card = el("div", "product-card");
      const row = el("div", "row");
      const path = product.cover_image_path || product.thumbnail_path;
      if (path) {
        const img = el("img", "thumb");
        img.src = "/" + path;
        img.alt = product.name;
        img.loading = "lazy";
        row.append(img);
      }
      const info = el("div", "info");
      const name = el("div", "name");
      name.textContent = product.name;
      if (product.needs_reembed) {
        const flag = el("span", "pill warn", "ต้องสแกนใหม่");
        flag.style.marginLeft = "0.5rem";
        name.append(flag);
      }
      const meta = el("div", "meta");
      if (product.category) meta.append(product.category);
      if (product.category && product.price != null) meta.append(" · ");
      if (product.price != null) meta.append(buildPrice(product.price));
      info.append(name, meta);
      row.append(info);
      card.append(row);
      catalogList.append(card);
    }
    catalogStatus.textContent = products.length ? "รายการนี้ใช้ร่วมกับหน้าจอสแกนและหน้าจัดการสินค้า" : "ยังไม่มีสินค้า สแกนและบันทึกสินค้าชิ้นแรกได้จากด้านบน";
  } catch (err) {
    catalogStatus.textContent = err.message;
  } finally { refresh.disabled = false; }
}

document.getElementById("refreshCatalogBtn").addEventListener("click", loadCatalog);
document.getElementById("nextProductBtn").addEventListener("click", () => {
  if (busy) return;
  savedResult.style.display = "none";
  captureBtn.scrollIntoView({ behavior: "smooth", block: "center" });
  captureBtn.focus();
  captureStatus.textContent = "เปลี่ยนเป็นสินค้าชิ้นถัดไป แล้วกดสแกนเพื่อเพิ่มเข้ารายการ";
});
loadCatalog();
