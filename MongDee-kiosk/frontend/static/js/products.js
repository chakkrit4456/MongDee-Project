// Customer-facing product showcase: a browsable grid; tapping a card opens
// the same detail popup the kiosk shows when it recognizes a held product.
const grid = document.getElementById("showcaseGrid");
const showcaseStatus = document.getElementById("showcaseStatus");
const popup = document.getElementById("productPopup");
const popupContent = document.getElementById("popupContent");
const dialogSupported = popup && typeof popup.showModal === "function";

let systemMuted = true;
try { systemMuted = localStorage.getItem("mongdee_muted") !== "false"; } catch (e) { /* private mode */ }

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
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(isoDate || "");
  if (!m) return isoDate;
  const [, y, mo, d] = m;
  return `${parseInt(d, 10)} ${THAI_MONTHS[parseInt(mo, 10) - 1]} ${parseInt(y, 10) + 543}`;
}

function productImage(p) {
  const src = p.cover_image_path || p.thumbnail_path;
  if (!src) return null;
  const img = el("img");
  img.src = "/" + src;
  img.alt = p.name;
  img.loading = "lazy";
  return img;
}

// Turns a YouTube/Vimeo/Facebook watch link into its autoplay embed URL —
// see the identical helper in kiosk.js for the customer-facing scan flow.
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

function buildDetail(p) {
  popupContent.replaceChildren();

  const head = el("div", "popup-head");
  if (p.category) head.append(el("span", "pill brand", p.category));
  head.append(el("h2", "popup-title", p.name));
  popupContent.append(head);

  const media = el("div", "popup-media");
  const thumbs = el("div", "popup-thumbs");
  const embedUrl = !p.video_path && p.video_link ? videoEmbedUrl(p.video_link) : null;
  let defaultNode;
  if (p.video_path) {
    const video = el("video");
    video.src = "/" + p.video_path;
    video.autoplay = true;
    video.loop = true;
    video.playsInline = true;
    video.muted = systemMuted;
    video.controls = true;
    defaultNode = video;
  } else if (embedUrl) {
    const iframe = el("iframe");
    iframe.src = embedUrl;
    iframe.title = p.name;
    iframe.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
    iframe.allowFullscreen = true;
    defaultNode = iframe;
  } else {
    defaultNode = productImage(p);
  }
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
    body.append(row);
  }
  if (p.story) {
    body.append(el("div", "popup-section-title", "รายละเอียดสินค้า"));
    body.append(el("div", "story", p.story));
  }
  popupContent.append(body);

  const footer = el("div", "popup-footer");
  const done = el("button", "popup-done", "ปิด");
  done.type = "button";
  done.onclick = closeDetail;
  footer.append(done);
  popupContent.append(footer);
}

function openDetail(p) {
  if (!dialogSupported) return;
  buildDetail(p);
  if (!popup.open) popup.showModal();
}

function closeDetail() {
  if (!dialogSupported) return;
  const video = popupContent.querySelector("video");
  if (video) video.pause();
  // Closing the <dialog> does not stop an embedded YouTube/Vimeo/Facebook
  // iframe — it keeps playing (and its audio keeps going) in the background
  // until the page reloads unless its src is cleared here.
  for (const iframe of popupContent.querySelectorAll("iframe")) iframe.src = "about:blank";
  if (popup.open) popup.close();
}

if (dialogSupported) {
  popup.addEventListener("cancel", (e) => { e.preventDefault(); closeDetail(); });
  popup.addEventListener("click", (e) => {
    const box = popup.getBoundingClientRect();
    if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) closeDetail();
  });
}

function card(p) {
  const item = el("button", "showcase-card");
  item.type = "button";
  const media = el("div", "showcase-media");
  const img = productImage(p);
  if (img) media.append(img);
  else media.append(el("div", "showcase-media-empty", "ไม่มีภาพ"));
  if (p.video_path) media.append(el("span", "showcase-badge", "▶ วิดีโอ"));
  item.append(media);

  const body = el("div", "showcase-body");
  body.append(el("div", "showcase-name", p.name));
  const meta = el("div", "showcase-meta");
  if (p.category) meta.append(el("span", "", p.category));
  if (p.category && p.price != null) meta.append(el("span", "meta-sep", " · "));
  if (p.price != null) meta.append(buildPrice(p.price));
  if (!meta.childElementCount) meta.textContent = " ";
  body.append(meta);
  item.append(body);

  item.addEventListener("click", () => openDetail(p));
  return item;
}

const searchInput = document.getElementById("searchInput");
const categoryChips = document.getElementById("categoryChips");
const NO_CATEGORY = "__none__";
const NO_CATEGORY_LABEL = "ไม่ระบุหมวด";
let allProducts = [];
let activeCategory = null;  // null = every category
const cardCache = new Map();

function matchesQuery(p) {
  const words = ((searchInput && searchInput.value) || "").trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const haystack = [p.name, p.category, p.origin, p.material].filter(Boolean).join(" ").toLowerCase();
  return words.every((word) => haystack.includes(word));
}

function categoryKey(p) {
  return p.category || NO_CATEGORY;
}

// Category buttons with counts that follow the current search text (so a chip
// never promises products the search has already filtered out).
function renderChips() {
  if (!categoryChips) return;
  const counts = new Map();
  for (const p of allProducts) {
    if (matchesQuery(p)) counts.set(categoryKey(p), (counts.get(categoryKey(p)) || 0) + 1);
  }
  if (activeCategory !== null && !counts.has(activeCategory)) counts.set(activeCategory, 0);
  const keys = [...counts.keys()].sort((a, b) => {
    if (a === NO_CATEGORY) return 1;
    if (b === NO_CATEGORY) return -1;
    return counts.get(b) - counts.get(a) || a.localeCompare(b, "th");
  });
  const total = [...counts.values()].reduce((sum, n) => sum + n, 0);

  const chip = (key, label, count) => {
    const button = el("button", key === activeCategory ? "chip active" : "chip");
    button.type = "button";
    button.append(el("span", "", label), el("span", "chip-count", String(count)));
    button.onclick = () => { activeCategory = key; render(); };
    return button;
  };
  categoryChips.replaceChildren(
    chip(null, "ทั้งหมด", total),
    ...keys.map((key) => chip(key, key === NO_CATEGORY ? NO_CATEGORY_LABEL : key, counts.get(key))),
  );
}

function clearFilters() {
  if (searchInput) searchInput.value = "";
  activeCategory = null;
  render();
  if (searchInput && searchInput.focus) searchInput.focus();
}

function emptyState() {
  const box = el("div", "showcase-empty");
  box.append(el("div", "showcase-empty-title", "ไม่พบสินค้าที่ตรงกับที่ค้นหา"));
  box.append(el("div", "hint", "ลองใช้คำอื่น หรือเลือกหมวดหมู่ “ทั้งหมด”"));
  const reset = el("button", "secondary", "ล้างตัวกรอง");
  reset.type = "button";
  reset.onclick = clearFilters;
  box.append(reset);
  return box;
}

function render() {
  const visible = allProducts.filter((p) => matchesQuery(p) && (activeCategory === null || categoryKey(p) === activeCategory));
  renderChips();
  if (visible.length) {
    grid.replaceChildren(...visible.map((p) => {
      if (!cardCache.has(p.id)) cardCache.set(p.id, card(p));
      return cardCache.get(p.id);
    }));
  } else if (allProducts.length) {
    grid.replaceChildren(emptyState());
  } else {
    grid.replaceChildren();
  }
  showcaseStatus.textContent = allProducts.length
    ? `แสดง ${visible.length} จาก ${allProducts.length} รายการ · คลิกที่สินค้าเพื่อดูรายละเอียดและวิดีโอ`
    : "ยังไม่มีสินค้าในระบบ";
}

if (searchInput) searchInput.addEventListener("input", render);

async function load() {
  try {
    const res = await fetch("/api/products", { cache: "no-store" });
    if (!res.ok) throw new Error("โหลดรายการสินค้าไม่สำเร็จ");
    allProducts = await res.json();
    render();
  } catch (err) {
    showcaseStatus.textContent = err.message;
  }
}

load();
