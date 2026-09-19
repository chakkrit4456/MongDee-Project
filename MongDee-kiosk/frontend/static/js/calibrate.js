const rawBox = document.getElementById("rawBox");
const roiBox = document.getElementById("roiBox");
const previewFrame = document.getElementById("previewFrame");
const previewCrop = document.getElementById("previewCrop");
const previewFeed = document.getElementById("previewFeed");
const stateBadge = document.getElementById("stateBadge");
const ratioNum = document.getElementById("ratioNum");
const gaugeFill = document.getElementById("gaugeFill");
const markOff = document.getElementById("markOff");
const markOn = document.getElementById("markOn");
const toast = document.getElementById("toast");
const resetBtn = document.getElementById("resetBtn");
const captureRefBtn = document.getElementById("captureRefBtn");
const refWarning = document.getElementById("refWarning");

const sliders = {
  onRatio: document.getElementById("onRatio"),
  offRatio: document.getElementById("offRatio"),
};
const labels = {
  onRatio: document.getElementById("onVal"),
  offRatio: document.getElementById("offVal"),
};

// -------- ROI: drag-to-move, drag-corner-to-resize --------

const ROI_MIN_DIM = 0.05;
let roi = { x: 0.2, y: 0.15, w: 0.6, h: 0.75 };

function updateRoiOverlay() {
  roiBox.style.left = `${roi.x * 100}%`;
  roiBox.style.top = `${roi.y * 100}%`;
  roiBox.style.width = `${roi.w * 100}%`;
  roiBox.style.height = `${roi.h * 100}%`;
}

function updatePreview() {
  const nw = previewFeed.naturalWidth || 1280;
  const nh = previewFeed.naturalHeight || 720;
  fitRoiContain(previewCrop, previewFeed, roi, nw, nh);
}

function setRoi(next, { save = true } = {}) {
  const w = Math.min(1, Math.max(ROI_MIN_DIM, next.w));
  const h = Math.min(1, Math.max(ROI_MIN_DIM, next.h));
  roi = {
    x: Math.min(1 - w, Math.max(0, next.x)),
    y: Math.min(1 - h, Math.max(0, next.y)),
    w,
    h,
  };
  updateRoiOverlay();
  updatePreview();
  if (save) scheduleSave();
}

let dragMode = null; // "move" | "nw" | "ne" | "sw" | "se"
let dragStart = null;

function fracFromEvent(e) {
  const rect = rawBox.getBoundingClientRect();
  return {
    fx: (e.clientX - rect.left) / rect.width,
    fy: (e.clientY - rect.top) / rect.height,
  };
}

function onDrag(e) {
  const { fx, fy } = fracFromEvent(e);
  const r = dragStart.roi;
  if (dragMode === "move") {
    setRoi({ x: r.x + (fx - dragStart.fx), y: r.y + (fy - dragStart.fy), w: r.w, h: r.h });
  } else if (dragMode === "nw") {
    setRoi({ x: fx, y: fy, w: r.x + r.w - fx, h: r.y + r.h - fy });
  } else if (dragMode === "ne") {
    setRoi({ x: r.x, y: fy, w: fx - r.x, h: r.y + r.h - fy });
  } else if (dragMode === "sw") {
    setRoi({ x: fx, y: r.y, w: r.x + r.w - fx, h: fy - r.y });
  } else if (dragMode === "se") {
    setRoi({ x: r.x, y: r.y, w: fx - r.x, h: fy - r.y });
  }
}

function endDrag() {
  document.removeEventListener("pointermove", onDrag);
  dragMode = null;
  dragStart = null;
}

function beginDrag(mode, e) {
  dragMode = mode;
  dragStart = { roi: { ...roi }, ...fracFromEvent(e) };
  document.addEventListener("pointermove", onDrag);
  document.addEventListener("pointerup", endDrag, { once: true });
  e.preventDefault();
}

roiBox.addEventListener("pointerdown", (e) => {
  if (e.target.classList.contains("roi-handle")) return;
  beginDrag("move", e);
});
for (const handle of roiBox.querySelectorAll(".roi-handle")) {
  handle.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    beginDrag(handle.dataset.corner, e);
  });
}

// -------- Sensitivity + save/reset --------

// The gauge represents 0..GAUGE_MAX (not 0..1) since real foreground
// ratios usually land under 0.3 — scaling to 1.0 would squeeze everything
// into a sliver on the left and make the gauge useless for calibration.
const GAUGE_MAX = 0.5;
const toGaugePct = (v) => Math.min(100, (v / GAUGE_MAX) * 100);

function updateThresholdLabels() {
  labels.onRatio.textContent = parseFloat(sliders.onRatio.value).toFixed(2);
  labels.offRatio.textContent = parseFloat(sliders.offRatio.value).toFixed(2);
  markOn.style.left = `${toGaugePct(parseFloat(sliders.onRatio.value))}%`;
  markOff.style.left = `${toGaugePct(parseFloat(sliders.offRatio.value))}%`;
}

function currentPayload() {
  return {
    roi,
    presence_on_ratio: parseFloat(sliders.onRatio.value),
    presence_off_ratio: parseFloat(sliders.offRatio.value),
  };
}

function applyInitial(data) {
  roi = { ...data.roi };
  updateRoiOverlay();
  updatePreview();
  sliders.onRatio.value = data.presence_on_ratio;
  sliders.offRatio.value = data.presence_off_ratio;
  updateThresholdLabels();
}

let saveTimer = null;
function scheduleSave() {
  toast.textContent = "กำลังบันทึก...";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      const res = await fetch("/api/calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentPayload()),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(typeof err.detail === "string" ? err.detail : "ค่าที่ตั้งไม่ถูกต้อง กรุณาตรวจสอบกรอบและความไว");
      }
      toast.textContent = "ใช้ค่านี้แล้ว (บันทึกอัตโนมัติ)";
    } catch (err) {
      toast.textContent = "เกิดข้อผิดพลาด: " + err.message;
    }
  }, 350);
}

for (const key of ["onRatio", "offRatio"]) {
  sliders[key].addEventListener("input", () => {
    if (+sliders.offRatio.value >= +sliders.onRatio.value) {
      if (key === "onRatio") sliders.offRatio.value = Math.max(0, +sliders.onRatio.value - 0.01).toFixed(2);
      else sliders.onRatio.value = (+sliders.offRatio.value + 0.01).toFixed(2);
    }
    updateThresholdLabels();
    scheduleSave();
  });
}

captureRefBtn.addEventListener("click", async () => {
  captureRefBtn.disabled = true;
  toast.textContent = "กำลังบันทึกภาพพื้นเปล่า...";
  try {
    const res = await fetch("/api/calibration/reference", { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(typeof err.detail === "string" ? err.detail : "บันทึกพื้นเปล่าไม่สำเร็จ");
    }
    toast.textContent = "บันทึกภาพพื้นเปล่าแล้ว — ตอนนี้สถานะควรกลับเป็น \"ว่าง\" ทันที";
    refWarning.style.display = "none";
  } catch (err) {
    toast.textContent = "เกิดข้อผิดพลาด: " + err.message;
  } finally {
    captureRefBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", async () => {
  toast.textContent = "กำลังรีเซ็ต...";
  try {
    const res = await fetch("/api/calibration/reset", { method: "POST" });
    const data = await res.json();
    applyInitial(data);
    toast.textContent = "รีเซ็ตเป็นค่าเริ่มต้นแล้ว";
  } catch (err) {
    toast.textContent = "รีเซ็ตไม่สำเร็จ";
  }
});

async function loadInitial() {
  const res = await fetch("/api/calibration");
  const data = await res.json();
  applyInitial(data);
}

async function pollStatus() {
  try {
    const res = await fetch("/api/camera/status");
    const status = await res.json();
    stateBadge.textContent = !status.camera_open ? "ไม่พบภาพสดจากกล้อง" : !status.has_reference ? "รอบันทึกพื้นเปล่า" : status.state === "present" ? "ตรวจพบสินค้า" : "ว่าง";
    stateBadge.className = "pill" + (status.state === "present" ? " ok" : "");
    const ratio = status.foreground_ratio || 0;
    ratioNum.textContent = ratio.toFixed(3);
    ratioNum.className = "ratio-num" + (status.state === "present" ? " hot" : "");
    gaugeFill.style.width = `${toGaugePct(ratio)}%`;
    refWarning.style.display = status.has_reference ? "none" : "block";
  } catch (err) {
    stateBadge.textContent = "เชื่อมต่อไม่ได้";
  }
}

loadInitial();
pollStatus();
setInterval(pollStatus, 400);

if (previewFeed.complete && previewFeed.naturalWidth) {
  updatePreview();
} else {
  previewFeed.addEventListener("load", updatePreview, { once: true });
}
window.addEventListener("resize", updatePreview);

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
    previewFeed.src = "/api/camera/stream?t=" + Date.now();
    await loadCameraDevices();
  } catch (err) {
    cameraSelectStatus.textContent = "เกิดข้อผิดพลาด: " + err.message;
  } finally {
    cameraSelectBtn.disabled = false;
  }
});

loadCameraDevices();
