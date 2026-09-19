// Product media lives on disk and can go missing (copied DB without files,
// deleted upload). Swap broken images/videos for a neutral placeholder instead
// of the browser's broken-image icon or a black player.
(function () {
  const PLACEHOLDER = "data:image/svg+xml;utf8," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">' +
    '<rect width="320" height="240" fill="#e9eff8"/>' +
    '<g fill="none" stroke="#8a97ab" stroke-width="3" stroke-linejoin="round" opacity="0.7">' +
    '<path d="M160 88l44 22v44l-44 22-44-22v-44z"/><path d="M116 110l44 22 44-22M160 132v44"/></g></svg>'
  );

  document.addEventListener("error", (event) => {
    const target = event.target;
    if (target instanceof HTMLImageElement) {
      if (target.dataset.mediaMissing || target.src.includes("/api/camera/")) return;
      target.dataset.mediaMissing = "1";
      target.classList.add("img-missing");
      target.src = PLACEHOLDER;
    } else if (target instanceof HTMLVideoElement) {
      const note = document.createElement("div");
      note.className = "media-missing";
      note.textContent = "เล่นวิดีโอนี้ไม่ได้ (ไม่พบไฟล์)";
      target.replaceWith(note);
    }
  }, true);
})();
