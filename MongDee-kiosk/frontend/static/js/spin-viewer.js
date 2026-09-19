// 360-style spin viewer for the product detail popup: plays back the
// angle photos saved during enrollment (product.spin_frames, in capture
// order) as a drag-to-rotate sequence, the same interaction pattern as a
// typical e-commerce 360 product view. Shared by kiosk.js and products.js.
function buildSpinViewer(frameUrls, altText) {
  const wrap = document.createElement("div");
  wrap.className = "spin-viewer";

  const img = document.createElement("img");
  img.alt = altText || "";
  img.draggable = false;
  img.src = "/" + frameUrls[0];
  wrap.append(img);

  const hint = document.createElement("div");
  hint.className = "spin-hint";
  hint.textContent = "ลากเพื่อหมุนดูรอบสินค้า ↔";
  wrap.append(hint);

  const PX_PER_FRAME = 10;
  let index = 0;
  let dragging = false;
  let startX = 0;
  let startIndex = 0;

  function setIndex(i) {
    index = ((i % frameUrls.length) + frameUrls.length) % frameUrls.length;
    img.src = "/" + frameUrls[index];
  }

  function onDown(e) {
    dragging = true;
    startX = e.clientX;
    startIndex = index;
    wrap.classList.add("dragging");
    wrap.setPointerCapture?.(e.pointerId);
    hint.classList.add("hidden");
  }
  function onMove(e) {
    if (!dragging) return;
    const steps = Math.trunc((e.clientX - startX) / PX_PER_FRAME);
    setIndex(startIndex - steps);
  }
  function onUp() {
    dragging = false;
    wrap.classList.remove("dragging");
  }

  wrap.addEventListener("pointerdown", onDown);
  wrap.addEventListener("pointermove", onMove);
  wrap.addEventListener("pointerup", onUp);
  wrap.addEventListener("pointerleave", onUp);
  wrap.addEventListener("pointercancel", onUp);

  // Preload the rest so dragging through the full turn doesn't stutter.
  for (let i = 1; i < frameUrls.length; i++) {
    const pre = new Image();
    pre.src = "/" + frameUrls[i];
  }

  return wrap;
}

// Minimum saved angles before a spin viewer is worth showing over a plain
// static photo — a couple of frames wouldn't read as a rotation.
const SPIN_MIN_FRAMES = 4;

// Angle thumbnails are evenly sampled down to this many so the whole gallery
// (360° + video + angles = 10 slots) stays on a single row; the 360° viewer
// still covers every saved frame.
const MAX_FRAME_THUMBS = 8;

// Fills `stage` with the selected media and `strip` with a row of thumbnails
// (360° spin viewer, the product's video/photo, then its saved angle photos),
// the way an e-commerce product gallery works. Opens on the video/photo when
// there is one, otherwise on the first angle photo. The strip stays empty
// when there is only one thing to show.
function mountMediaGallery(stage, strip, { defaultNode, frameUrls, galleryUrls, altText }) {
  const frames = frameUrls || [];
  // Photos added by hand in the manage page win over the auto-captured angles.
  const curated = (galleryUrls || []).slice(0, MAX_FRAME_THUMBS);
  const items = [];
  if (frames.length >= SPIN_MIN_FRAMES) items.push({ kind: "spin" });
  if (defaultNode) items.push({ kind: "main" });
  if (curated.length) {
    for (const url of curated) items.push({ kind: "frame", url });
  } else {
    const thumbCount = Math.min(frames.length, MAX_FRAME_THUMBS);
    for (let i = 0; i < thumbCount; i++) {
      const index = thumbCount === 1 ? 0 : Math.round((i * (frames.length - 1)) / (thumbCount - 1));
      items.push({ kind: "frame", url: frames[index] });
    }
  }
  if (!items.length) return;

  const isVideo = defaultNode && (defaultNode.tagName === "VIDEO" || defaultNode.tagName === "IFRAME");
  const buttons = [];
  let spinNode = null;
  let current = -1;

  function thumbClass(index) {
    const item = items[index];
    return "gallery-thumb" + (item.kind === "spin" ? " is-spin" : "") + (index === current ? " active" : "");
  }

  function select(index) {
    current = index;
    const item = items[index];
    stage.replaceChildren();
    if (item.kind === "spin") {
      if (!spinNode) spinNode = buildSpinViewer(frames, altText);
      stage.append(spinNode);
    } else if (item.kind === "main") {
      stage.append(defaultNode);
      if (defaultNode.tagName === "VIDEO" && defaultNode.play) {
        const started = defaultNode.play();
        if (started && started.catch) started.catch(() => {});
      }
    } else {
      const img = document.createElement("img");
      img.alt = altText || "";
      img.src = "/" + item.url;
      stage.append(img);
    }
    if (item.kind !== "main" && defaultNode && defaultNode.tagName === "VIDEO" && defaultNode.pause) defaultNode.pause();
    buttons.forEach((button, i) => { button.className = thumbClass(i); });
  }

  items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    if (item.kind === "spin") {
      button.textContent = "360°";
    } else if (item.kind === "main" && isVideo) {
      button.textContent = "▶ วิดีโอ";
    } else {
      const src = item.kind === "frame" ? "/" + item.url : (typeof defaultNode.src === "string" ? defaultNode.src : "");
      if (src) {
        const thumb = document.createElement("img");
        thumb.alt = altText || "";
        thumb.src = src;
        button.append(thumb);
      } else {
        button.textContent = "ภาพ";
      }
    }
    button.className = thumbClass(index);
    button.onclick = () => select(index);
    buttons.push(button);
  });

  let start = items.findIndex((item) => item.kind === "main");
  if (start < 0) start = items.findIndex((item) => item.kind === "frame");
  select(start < 0 ? 0 : start);

  // A missing/broken video or photo shouldn't leave an empty stage when angle photos exist.
  if (defaultNode && defaultNode.addEventListener) {
    defaultNode.addEventListener("error", () => {
      const fallback = items.findIndex((item) => item.kind === "frame");
      if (current >= 0 && items[current].kind === "main" && fallback >= 0) select(fallback);
    });
  }

  if (items.length > 1) strip.append(...buttons);
}
