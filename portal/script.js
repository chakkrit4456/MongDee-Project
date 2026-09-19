const PROJECTS = ["mongdee", "kiosk"];
const STORAGE_PREFIX = "mongdee_portal_url_";
const CHECK_INTERVAL_MS = 20000;
const CHECK_TIMEOUT_MS = 4000;

function storageKey(project) {
  return STORAGE_PREFIX + project;
}

function getUrl(project) {
  try {
    return localStorage.getItem(storageKey(project)) || "";
  } catch {
    return "";
  }
}

function setUrl(project, url) {
  try {
    localStorage.setItem(storageKey(project), url);
  } catch {
    // Private browsing / blocked storage — the field still works this visit.
  }
}

function setStatus(project, state, label) {
  const el = document.querySelector(`[data-status-for="${project}"]`);
  if (!el) return;
  el.dataset.state = state;
  el.querySelector(".status-label").textContent = label;
}

async function checkStatus(project, url) {
  if (!url) {
    setStatus(project, "offline", "ยังไม่ตั้งค่า");
    return;
  }
  setStatus(project, "checking", "กำลังตรวจสอบ...");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);
  try {
    // no-cors: we can't read the response, only whether the request
    // reached a live server at all — enough for an online/offline hint.
    await fetch(url, { mode: "no-cors", cache: "no-store", signal: controller.signal });
    setStatus(project, "online", "ออนไลน์");
  } catch {
    setStatus(project, "offline", "เข้าถึงไม่ได้");
  } finally {
    clearTimeout(timer);
  }
}

function refreshAll() {
  for (const project of PROJECTS) {
    checkStatus(project, getUrl(project));
  }
}

function init() {
  for (const project of PROJECTS) {
    const input = document.querySelector(`[data-url-input="${project}"]`);
    const saveBtn = document.querySelector(`[data-save="${project}"]`);
    const openBtn = document.querySelector(`[data-open="${project}"]`);

    input.value = getUrl(project);

    saveBtn.addEventListener("click", () => {
      const url = input.value.trim();
      setUrl(project, url);
      checkStatus(project, url);
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") saveBtn.click();
    });

    openBtn.addEventListener("click", () => {
      const url = getUrl(project) || input.value.trim();
      if (!url) {
        input.focus();
        return;
      }
      window.open(url, "_blank", "noopener");
    });
  }

  refreshAll();
  setInterval(refreshAll, CHECK_INTERVAL_MS);
}

document.addEventListener("DOMContentLoaded", init);
