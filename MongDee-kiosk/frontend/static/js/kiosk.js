const infoPane = document.getElementById("infoPane");
const statusBadge = document.getElementById("statusBadge");
const muteBtn = document.getElementById("muteBtn");
const popup = document.getElementById("productPopup");
const popupContent = document.getElementById("popupContent");

// Line icons (not emoji) so the button renders in the theme's own color
// instead of the OS's colorful emoji glyphs.
const ICON_VOLUME_ON = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`;
const ICON_VOLUME_OFF = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>`;

// Muted by default (browsers block audible autoplay without a user gesture
// anyway) — this is a persistent system-wide toggle so a customer doesn't
// have to notice/hover the video's own native controls to find sound. It
// stays in sync both ways with the video's native volume control (see the
// "volumechange" listener in buildProductVideo) and persists across scans
// and page reloads on this kiosk.
let systemMuted = localStorage.getItem("mongdee_muted") !== "false";

function updateMuteButtonUI() {
  if (!muteBtn) return;
  muteBtn.innerHTML = systemMuted ? ICON_VOLUME_OFF : ICON_VOLUME_ON;
  muteBtn.classList.toggle("unmuted", !systemMuted);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatMoney(n) {
  return new Intl.NumberFormat("th-TH", { minimumFractionDigits: 0, maximumFractionDigits: 2 }).format(n);
}

const THAI_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."];

function formatThaiDate(isoDate) {
  // isoDate is "YYYY-MM-DD" from a <input type="date">; parse manually so
  // no timezone shift can roll the date to the previous/next day.
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(isoDate || "");
  if (!m) return isoDate;
  const [, y, mo, d] = m;
  return `${parseInt(d, 10)} ${THAI_MONTHS[parseInt(mo, 10) - 1]} ${parseInt(y, 10) + 543}`;
}

// --- Idle / transitional states -----------------------------------------

function renderIdle() {
  infoPane.replaceChildren();
  const wrap = el("div", "idle-msg");
  const logo = el("img", "idle-logo");
  logo.src = "/static/img/logo.png";
  logo.alt = "MongDee";
  wrap.append(logo);
  wrap.append(el("div", "big", "หยิบสินค้าขึ้นมาดูได้เลย"), el("div", "", "ยกสินค้าให้กล้องเห็น ระบบจะแสดงรายละเอียดของชิ้นนั้น"));
  const steps = el("ol", "idle-steps");
  for (const text of ["หยิบสินค้าที่สนใจ", "ยกให้กล้องเห็นในกรอบ", "ดูรายละเอียดและวิดีโอบนจอ"]) steps.append(el("li", "", text));
  wrap.append(steps);
  const browse = el("a", "video-cta", "ดูสินค้าทั้งหมด");
  browse.href = "/products";
  wrap.append(browse);
  infoPane.append(wrap);
}

function renderScanning() {
  infoPane.replaceChildren();
  const wrap = el("div", "idle-msg");
  wrap.append(el("div", "spinner"), el("div", "big", "กำลังดูสินค้าในมือ…"));
  infoPane.append(wrap);
}

function renderNotice(result) {
  infoPane.replaceChildren();
  const wrap = el("div", "idle-msg");
  wrap.append(el("div", "big", result.message || "กำลังเตรียมระบบ"));
  if (result.status === "needs_reference") {
    const link = el("a", "video-cta", "ไปหน้าปรับตั้งค่า");
    link.href = "/calibrate";
    wrap.append(link);
  } else if (result.status === "unstable") {
    wrap.append(el("div", "", "ระบบจะลองใหม่ ถือสินค้าให้นิ่งขึ้นอีกนิด"));
  }
  infoPane.append(wrap);
}

// --- Product detail (shared by side pane summary + popup) ---------------

function productImage(p, className) {
  const src = p.cover_image_path || p.thumbnail_path;
  if (!src) return null;
  const img = el("img", className);
  img.src = "/" + src;
  img.alt = p.name;
  return img;
}

function buildProductVideo(p, className) {
  if (!p.video_path) return null;
  const video = el("video", className);
  video.src = "/" + p.video_path;
  video.autoplay = true;
  video.loop = true;
  video.playsInline = true;
  video.muted = systemMuted;
  video.controls = true;
  // Keep the system-wide mute button in sync if sound gets toggled from
  // the video's own native volume control instead.
  video.addEventListener("volumechange", () => {
    systemMuted = video.muted;
    localStorage.setItem("mongdee_muted", String(systemMuted));
    updateMuteButtonUI();
  });
  return video;
}

// Turns a YouTube/Vimeo/Facebook watch link into its autoplay embed URL, so
// the popup can play the attached video inline instead of just linking out.
// Returns null for URLs from providers we don't know how to embed (or that
// fail to parse) — callers fall back to a plain external link for those.
function videoEmbedUrl(rawUrl) {
  let u;
  try {
    u = new URL(rawUrl);
  } catch (err) {
    return null;
  }
  const host = u.hostname.replace(/^www\.|^m\./, "");
  if (host === "youtube.com") {
    const id = u.searchParams.get("v") || u.pathname.match(/^\/shorts\/([\w-]+)/)?.[1];
    if (id) return `https://www.youtube.com/embed/${id}?autoplay=1&mute=1&playsinline=1`;
  } else if (host === "youtu.be") {
    const id = u.pathname.slice(1);
    if (id) return `https://www.youtube.com/embed/${id}?autoplay=1&mute=1&playsinline=1`;
  } else if (host === "vimeo.com") {
    const id = u.pathname.split("/").filter(Boolean)[0];
    if (id) return `https://player.vimeo.com/video/${id}?autoplay=1&muted=1`;
  } else if (host === "facebook.com" || host === "fb.watch") {
    return `https://www.facebook.com/plugins/video.php?href=${encodeURIComponent(rawUrl)}&autoplay=true&mute=1`;
  }
  return null;
}

function buildVideoEmbed(embedUrl, title) {
  const iframe = el("iframe");
  iframe.src = embedUrl;
  iframe.title = title;
  iframe.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
  iframe.allowFullscreen = true;
  return iframe;
}

function buildDetailFields(container, p) {
  const fields = [
    ["แหล่งที่มา", p.origin],
    ["วัสดุ", p.material],
    ["วิธีการผลิต", p.process],
    ["วันผลิต", p.production_date ? formatThaiDate(p.production_date) : null],
    ["วันหมดอายุ", p.expiry_date ? formatThaiDate(p.expiry_date) : null],
  ];
  for (const [label, value] of fields) {
    if (!value) continue;
    const row = el("div", "spec-row");
    row.append(el("span", "spec-label", label), el("span", "spec-value", value));
    container.append(row);
  }
  if (p.story) {
    container.append(el("div", "popup-section-title", "รายละเอียดสินค้า"));
    container.append(el("div", "story", p.story));
  }
}

// Compact summary in the side pane — the popup carries the full detail + video.
function renderMatched(result) {
  infoPane.replaceChildren();
  const p = result.product;

  const img = productImage(p, "product-thumb");
  if (img) infoPane.append(img);

  infoPane.append(el("div", "product-name", p.name));
  if (p.price != null) {
    const priceTag = el("div", "price-tag");
    priceTag.append(buildPrice(p.price));
    infoPane.append(priceTag);
  }

  const metaRow = el("div", "meta-row");
  if (p.category) metaRow.append(el("span", "pill", p.category));
  if (result.manually_confirmed) {
    metaRow.append(el("span", "pill ok", "ยืนยันโดยผู้ใช้"));
  } else if (typeof location !== "undefined" && /[?&]debug\b/.test(location.search || "")) {
    // Similarity is an engineering number, not something customers should see; add ?debug to the URL to show it.
    metaRow.append(el("span", "pill ok", `ความคล้าย ${Math.round((result.confidence || 0) * 100)}%`));
  }
  if (metaRow.childElementCount) infoPane.append(metaRow);

  const openBtn = el("button", "match-detail-btn", (p.video_path || p.video_link) ? "ดูรายละเอียด · วิดีโอ" : "ดูรายละเอียด");
  openBtn.type = "button";
  openBtn.addEventListener("click", () => openPopup(result, true));
  infoPane.append(openBtn);
}

function candidateButtons(result, onPick) {
  const options = el("div", "options");
  for (const c of result.candidates) {
    const btn = el("button", "secondary", `${c.name} (${Math.round(c.score * 100)}%)`);
    btn.onclick = () => onPick(c.product_id);
    options.append(btn);
  }
  return options;
}

function addEnrollButton(result, label) {
  if (!result.pending_enroll_session_id) return;
  const addBtn = el("button", "", label);
  addBtn.style.marginTop = "1.5rem";
  addBtn.onclick = () => {
    window.location.href = `/enroll?session=${result.pending_enroll_session_id}`;
  };
  infoPane.append(addBtn);
}

function renderAmbiguous(result) {
  infoPane.replaceChildren();
  const wrap = el("div", "idle-msg");
  wrap.append(el("div", "big", "ไม่แน่ใจว่าเป็นสินค้าชิ้นไหน"));
  wrap.append(el("div", "", "ระบบพบสินค้าที่คล้ายกันหลายชิ้น กรุณาเลือกให้ถูกต้อง"));
  infoPane.append(wrap);

  const box = el("div", "candidates");
  box.append(el("div", "field-label", "เลือกสินค้าที่ถูกต้อง"));
  box.append(candidateButtons(result, (productId) => correctMatch(result.scan_event_id, productId)));
  infoPane.append(box);

  addEnrollButton(result, "ไม่ใช่ตัวเลือกด้านบน — เพิ่มเป็นสินค้าใหม่");
}

function renderUnknown(result) {
  infoPane.replaceChildren();
  const wrap = el("div", "idle-msg");
  wrap.append(el("div", "big", "ยังระบุสินค้าไม่ได้แน่ชัด"));
  wrap.append(el("div", "", "ลองหมุนสินค้าช้า ๆ ให้กล้องเห็นหลายด้าน หรือเลือกจากรายการด้านล่าง"));
  infoPane.append(wrap);

  if (result.candidates && result.candidates.length) {
    const box = el("div", "candidates");
    box.append(el("div", "field-label", "สินค้าที่ใกล้เคียง (ความมั่นใจต่ำ)"));
    box.append(candidateButtons(result, (productId) => correctMatch(result.scan_event_id, productId)));
    infoPane.append(box);
  }

  addEnrollButton(result, "เพิ่มสินค้านี้เข้าระบบ");
}

// --- Pop-up shown when a held product is recognized --------------------

const dialogSupported = popup && typeof popup.showModal === "function";
let popupProductId = null;      // product currently in the popup
let popupDismissedId = null;    // product the customer closed the popup on

function buildPopupContent(result) {
  popupContent.replaceChildren();
  const p = result.product;

  const head = el("div", "popup-head");
  if (p.category) head.append(el("span", "pill brand", p.category));
  head.append(el("h2", "popup-title", p.name));
  popupContent.append(head);

  const media = el("div", "popup-media");
  const thumbs = el("div", "popup-thumbs");
  const video = buildProductVideo(p, "");
  const embedUrl = !video && p.video_link ? videoEmbedUrl(p.video_link) : null;
  const defaultNode = video || (embedUrl ? buildVideoEmbed(embedUrl, p.name) : productImage(p, ""));
  mountMediaGallery(media, thumbs, { defaultNode, frameUrls: p.spin_frames, galleryUrls: (p.gallery || []).map((g) => g.path), altText: p.name });
  if (media.childElementCount) popupContent.append(media);
  if (thumbs.childElementCount) popupContent.append(thumbs);

  const body = el("div", "popup-body");
  if (p.price != null) {
    const priceBox = el("div", "popup-price");
    priceBox.append(buildPrice(p.price));
    body.append(priceBox);
  }

  const actions = el("div", "popup-actions");
  if (p.video_link && !embedUrl) {
    // Dedicated video link, but from a provider we don't know how to embed
    // (see videoEmbedUrl) — fall back to an external link.
    const videoLink = el("a", "video-cta");
    videoLink.href = p.video_link;
    videoLink.target = "_blank";
    videoLink.rel = "noopener";
    videoLink.append(el("span", "video-cta-icon", "▶"), el("span", "", "ดูวิดีโอ"));
    actions.append(videoLink);
  }
  if (p.video_url) {
    const contactLink = el("a", "video-cta outline");
    contactLink.href = p.video_url;
    contactLink.target = "_blank";
    contactLink.rel = "noopener";
    contactLink.append(el("span", "video-cta-icon", "↗"), el("span", "", "ช่องทางการติดต่อ"));
    actions.append(contactLink);
  }
  if (actions.childElementCount) body.append(actions);
  buildDetailFields(body, p);

  popupContent.append(body);

  const footer = el("div", "popup-footer");
  const done = el("button", "popup-done", "กลับไปหน้าสแกน");
  done.type = "button";
  done.onclick = () => closePopup(true);
  footer.append(done);
  popupContent.append(footer);
}

function openPopup(result, force) {
  if (!dialogSupported || !result.product) return;
  const id = result.product.id;
  if (!force && (id === popupProductId || id === popupDismissedId)) return;
  popupProductId = id;
  popupDismissedId = null;
  buildPopupContent(result);
  if (!popup.open) popup.showModal();
  // No auto-close timer: stays open however long the customer keeps holding
  // the product, however long that is — it only closes when they set the
  // product down (pollRecognition below), or close it themselves.
}

function closePopup(userDismissed) {
  if (!dialogSupported) return;
  if (userDismissed && popupProductId != null) popupDismissedId = popupProductId;
  popupProductId = null;
  const video = popupContent.querySelector("video");
  if (video) video.pause();
  // Closing the <dialog> does not stop an embedded YouTube/Vimeo/Facebook
  // iframe — it keeps playing (and its audio keeps going) in the background
  // until the page reloads unless its src is cleared here.
  for (const iframe of popupContent.querySelectorAll("iframe")) iframe.src = "about:blank";
  if (popup.open) popup.close();
}

if (dialogSupported) {
  popup.addEventListener("cancel", (e) => { e.preventDefault(); closePopup(true); });
  // Click on the backdrop (outside the dialog box) closes it.
  popup.addEventListener("click", (e) => {
    const box = popup.getBoundingClientRect();
    if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) {
      closePopup(true);
    }
  });
}

// --- Correction / polling ---------------------------------------------

async function correctMatch(scanEventId, productId) {
  if (!scanEventId) return;
  try {
    const res = await fetch("/api/recognition/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scan_event_id: scanEventId, product_id: productId }),
    });
    if (!res.ok) throw new Error("ผลสแกนเปลี่ยนแล้ว กรุณาลองเลือกใหม่");
    await pollRecognition();
  } catch (err) {
    statusBadge.textContent = err.message;
  }
}

let lastResultKey = null;
let recognitionPolling = false;

// The hint under the camera follows what is actually happening instead of
// always repeating "pick a product up".
const camHint = document.getElementById("camHint");
const guideHint = document.getElementById("guideHint");
const guideSteps = document.getElementById("guideSteps");
let cameraState = null;          // "empty" | "present" | "offline"
let recognitionStatus = "idle";

// Step list beside the camera (portrait layout): the current step follows
// what the system is really doing rather than cycling on a timer.
const guideItems = ["หยิบสินค้าที่สนใจ", "ยกให้กล้องเห็นในกรอบ", "ดูรายละเอียดและวิดีโอบนจอ"]
  .map((label) => el("li", "guide-step", label));
if (guideSteps) guideSteps.append(...guideItems);

function updateCamHint() {
  let text = "";
  if (cameraState === "empty") text = "หยิบสินค้าขึ้นมาให้กล้องเห็นในกรอบ";
  else if (cameraState === "present" && recognitionStatus === "scanning") text = "กำลังสแกน… ถือสินค้าให้นิ่ง ๆ";
  for (const node of [camHint, guideHint]) {
    if (!node) continue;
    node.textContent = text;
    node.style.display = text ? "" : "none";
  }

  let active = 0;
  if (recognitionStatus === "matched") active = 2;
  else if (cameraState === "present" || recognitionStatus === "scanning") active = 1;
  guideItems.forEach((item, i) => {
    item.className = "guide-step" + (i < active ? " done" : i === active ? " active" : "");
  });
}
updateCamHint();

async function pollRecognition() {
  if (recognitionPolling) return;
  recognitionPolling = true;
  try {
    const res = await fetch("/api/recognition/current", { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error("connection failed");
    const result = await res.json();
    recognitionStatus = result.status;
    updateCamHint();
    const key = JSON.stringify([result.status, result.scan_event_id, result.product?.id, result.updated_at]);
    if (key !== lastResultKey) {
      lastResultKey = key;
      infoPane.classList.toggle("is-matched", result.status === "matched");
      if (result.status === "matched") {
        renderMatched(result);
        openPopup(result, false);
      } else {
        // Lifting the product away closes the popup on its own (no need to
        // hunt for a close button); it just never times out on its own
        // while the product is still in view — see openPopup.
        closePopup(false);
        popupDismissedId = null;
        if (result.status === "idle") renderIdle();
        else if (result.status === "scanning") renderScanning();
        else if (result.status === "ambiguous") renderAmbiguous(result);
        else if (result.status === "unknown") renderUnknown(result);
        else renderNotice(result);
      }
    }
  } catch (err) {
    recognitionStatus = "offline";
    updateCamHint();
    if (lastResultKey !== "offline") {
      lastResultKey = "offline";
      closePopup(false);
      renderNotice({ message: "เชื่อมต่อระบบไม่ได้ กำลังลองใหม่" });
    }
  } finally {
    recognitionPolling = false;
  }
}

const detectBox = document.getElementById("detectBox");
const detectLabel = document.getElementById("detectLabel");

async function pollCameraStatus() {
  try {
    const res = await fetch("/api/camera/status");
    const status = await res.json();
    cameraState = (!status.camera_open || !status.has_reference) ? "offline" : status.state;
    updateCamHint();
    if (!status.camera_open) {
      statusBadge.textContent = "ไม่พบกล้อง";
    } else if (!status.has_reference) {
      statusBadge.textContent = "กรุณาบันทึกภาพพื้นเปล่า";
    } else {
      statusBadge.textContent = status.state === "present" ? "เห็นสินค้าแล้ว" : "พร้อมสแกน";
    }

    applyDetectBoxStatus(detectBox, detectLabel, status);
  } catch (err) {
    cameraState = "offline";
    updateCamHint();
    statusBadge.textContent = "เชื่อมต่อไม่ได้";
    detectBox.className = "detect-box";
    detectLabel.textContent = "";
  }
}

const kioskCamera = document.getElementById("kioskCamera");
const camCrop = document.getElementById("camCrop");

// Sizes cam-frame to the largest 9:16 box that fits inside cam-pane. Done in
// JS (not CSS aspect-ratio) because no CSS-only rule reliably constrains a
// fixed-ratio box on both axes inside a flexible sibling layout. Must run
// before applyCameraCrop below, which measures camCrop's now-definite size.
const camFrame = document.getElementById("camFrame");
function fitCamFrame() {
  const pane = camFrame.parentElement;
  const pw = pane.clientWidth;
  const ph = pane.clientHeight;
  const targetAR = 9 / 16;
  let w = ph * targetAR;
  let h = ph;
  if (w > pw) {
    w = pw;
    h = pw / targetAR;
  }
  camFrame.style.width = `${w}px`;
  camFrame.style.height = `${h}px`;
}
fitCamFrame();

function refreshCameraCrop() {
  if (kioskCamera.naturalWidth) applyCameraCrop(camCrop, kioskCamera);
}
if (kioskCamera.complete && kioskCamera.naturalWidth) {
  refreshCameraCrop();
} else {
  kioskCamera.addEventListener("load", refreshCameraCrop, { once: true });
}
window.addEventListener("resize", () => {
  fitCamFrame();
  refreshCameraCrop();
});

renderIdle();
pollRecognition();
pollCameraStatus();
setInterval(pollRecognition, 1000);
setInterval(pollCameraStatus, 2000);

updateMuteButtonUI();
if (muteBtn) {
  muteBtn.addEventListener("click", () => {
    systemMuted = !systemMuted;
    localStorage.setItem("mongdee_muted", String(systemMuted));
    updateMuteButtonUI();
    for (const video of document.querySelectorAll("video")) video.muted = systemMuted;
  });
}

const adminMenuBtn = document.getElementById("adminMenuBtn");
const adminMenu = document.getElementById("adminMenu");
if (adminMenuBtn && adminMenu) {
  adminMenuBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    adminMenu.classList.toggle("open");
  });
  document.addEventListener("click", (e) => {
    if (adminMenu.classList.contains("open") && !adminMenu.contains(e.target) && e.target !== adminMenuBtn) {
      adminMenu.classList.remove("open");
    }
  });
}
