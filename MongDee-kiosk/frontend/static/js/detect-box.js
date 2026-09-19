// Shared by enroll.js and kiosk.js: positions the ROI overlay box (from
// /api/calibration) and colors/labels it from /api/camera/status' box_status
// (none/blurry/ok) — kept in one place so both pages stay in sync.
const DETECT_BOX_LABELS = { none: "ไม่พบสินค้า", blurry: "สินค้าไม่ชัด", ok: "ปกติ" };

async function positionDetectBox(boxEl) {
  try {
    const res = await fetch("/api/calibration");
    const data = await res.json();
    const { x, y, w, h } = data.roi;
    boxEl.style.left = `${x * 100}%`;
    boxEl.style.top = `${y * 100}%`;
    boxEl.style.width = `${w * 100}%`;
    boxEl.style.height = `${h * 100}%`;
  } catch (err) {
    // Leave the box hidden (no status class applied yet) if this fails.
  }
}

// Scales/positions imgEl inside a cropEl of size (cw, ch) so the calibrated
// ROI is uniformly "contain"-fit — the whole ROI is always fully visible
// (never distorted, never cropped off), letterboxed on whichever axis
// doesn't match cropEl's aspect ratio. Shared by the kiosk's customer-facing
// 9:16 display and the settings page's live preview, so both always show
// exactly the same thing. Pure/synchronous — callers own fetching the ROI.
function fitRoiContain(cropEl, imgEl, roi, nw, nh) {
  const { x, y, w, h } = roi;
  const cw = cropEl.clientWidth || 1;
  const ch = cropEl.clientHeight || 1;
  const roiWpx = w * nw;
  const roiHpx = h * nh;
  const scale = Math.min(cw / roiWpx, ch / roiHpx);
  imgEl.style.width = `${((nw * scale) / cw) * 100}%`;
  imgEl.style.height = `${((nh * scale) / ch) * 100}%`;
  const centerXpx = (x + w / 2) * nw;
  const centerYpx = (y + h / 2) * nh;
  imgEl.style.left = `${50 - ((centerXpx * scale) / cw) * 100}%`;
  imgEl.style.top = `${50 - ((centerYpx * scale) / ch) * 100}%`;
}

// Kiosk-only: crops the live feed down to the calibrated ROI (fetched fresh)
// so the customer-facing view never shows the booth's walls/floor around
// the platform.
async function applyCameraCrop(cropEl, imgEl) {
  try {
    const res = await fetch("/api/calibration");
    const data = await res.json();
    const nw = imgEl.naturalWidth || 1280;
    const nh = imgEl.naturalHeight || 720;
    fitRoiContain(cropEl, imgEl, data.roi, nw, nh);
  } catch (err) {
    // Leave the feed uncropped if this fails.
  }
}

function applyDetectBoxStatus(boxEl, labelEl, status) {
  const boxStatus = status && status.camera_open && status.has_reference ? status.box_status : null;
  boxEl.className = "detect-box" + (boxStatus ? ` status-${boxStatus}` : "");
  labelEl.textContent = boxStatus ? DETECT_BOX_LABELS[boxStatus] || "" : "";
}
