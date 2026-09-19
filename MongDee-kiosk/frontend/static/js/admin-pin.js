// Admin PIN prompts. The server enforces the PIN (X-Admin-Pin header) on every
// request that deletes data; this file collects it from staff and sets/changes it.
const PIN_HEADER = "X-Admin-Pin";

// Modal with one or more password fields. Resolves with {fieldName: value}, or null if cancelled.
function pinDialog({ title, message, fields, confirmLabel = "ตกลง", danger = false, error = "" }) {
  return new Promise((resolve) => {
    const dialog = document.createElement("dialog");
    dialog.className = "pin-dialog";
    const form = document.createElement("form");

    const heading = document.createElement("h3");
    heading.textContent = title;
    form.append(heading);
    if (message) {
      const text = document.createElement("p");
      text.className = "pin-message";
      text.textContent = message;
      form.append(text);
    }

    const inputs = {};
    for (const field of fields) {
      const wrap = document.createElement("div");
      wrap.className = "field";
      const label = document.createElement("label");
      label.textContent = field.label;
      const input = document.createElement("input");
      input.type = "password";
      input.inputMode = "numeric";
      input.autocomplete = "off";
      input.maxLength = 12;
      input.required = true;
      wrap.append(label, input);
      form.append(wrap);
      inputs[field.name] = input;
    }

    const problem = document.createElement("div");
    problem.className = "pin-error";
    problem.setAttribute("role", "alert");
    problem.textContent = error;
    form.append(problem);

    const row = document.createElement("div");
    row.className = "button-row";
    const ok = document.createElement("button");
    ok.type = "submit";
    ok.className = danger ? "danger-solid" : "";
    ok.textContent = confirmLabel;
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "secondary";
    cancel.textContent = "ยกเลิก";
    row.append(ok, cancel);
    form.append(row);
    dialog.append(form);
    document.body.append(dialog);

    let answered = false;
    const finish = (value) => {
      if (answered) return;
      answered = true;
      dialog.close();
      dialog.remove();
      resolve(value);
    };
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      finish(Object.fromEntries(Object.entries(inputs).map(([name, input]) => [name, input.value])));
    });
    cancel.onclick = () => finish(null);
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); finish(null); });

    dialog.showModal();
    Object.values(inputs)[0].focus();
  });
}

async function errorMessage(res) {
  const body = await res.json().catch(() => ({}));
  const detail = body.detail;
  if (detail && typeof detail === "object") return detail.message || "เกิดข้อผิดพลาด";
  return detail || "เกิดข้อผิดพลาด";
}

async function pinIsSet() {
  const res = await fetch("/api/admin/pin", { cache: "no-store" });
  return (await res.json()).is_set;
}

// Makes sure an admin PIN exists, asking staff to choose one the first time.
async function ensurePinIsSet() {
  if (await pinIsSet()) return true;
  let error = "";
  for (;;) {
    const values = await pinDialog({
      title: "ตั้งรหัสผ่านแอดมิน",
      message: "ยังไม่ได้ตั้งรหัสผ่าน กรุณาตั้งรหัสตัวเลข 4–12 หลัก ใช้ยืนยันก่อนลบข้อมูลทุกครั้ง",
      fields: [{ name: "pin", label: "รหัสใหม่" }, { name: "again", label: "ยืนยันรหัสอีกครั้ง" }],
      confirmLabel: "ตั้งรหัส",
      error,
    });
    if (!values) return false;
    if (values.pin !== values.again) { error = "รหัสสองครั้งไม่ตรงกัน"; continue; }
    const res = await fetch("/api/admin/pin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_pin: values.pin }),
    });
    if (res.ok) return true;
    error = await errorMessage(res);
  }
}

// Sends a DELETE after asking for the PIN. Resolves with the Response, or null when staff cancel.
async function protectedDelete(url, { title, message, confirmLabel = "ลบ" }) {
  if (!(await ensurePinIsSet())) return null;
  let error = "";
  for (;;) {
    const values = await pinDialog({
      title, message, confirmLabel, error, danger: true,
      fields: [{ name: "pin", label: "รหัสผ่านแอดมิน" }],
    });
    if (!values) return null;
    const res = await fetch(url, { method: "DELETE", headers: { [PIN_HEADER]: values.pin } });
    if (res.status === 401 || res.status === 429) {  // wrong PIN, or temporarily locked: ask again
      error = await errorMessage(res);
      continue;
    }
    return res;
  }
}

// Change the PIN (asks for the current one). Resolves true when changed.
async function changeAdminPin() {
  if (!(await pinIsSet())) return ensurePinIsSet();
  let error = "";
  for (;;) {
    const values = await pinDialog({
      title: "เปลี่ยนรหัสผ่านแอดมิน",
      fields: [
        { name: "current", label: "รหัสปัจจุบัน" },
        { name: "pin", label: "รหัสใหม่ (ตัวเลข 4–12 หลัก)" },
        { name: "again", label: "ยืนยันรหัสใหม่" },
      ],
      confirmLabel: "เปลี่ยนรหัส",
      error,
    });
    if (!values) return false;
    if (values.pin !== values.again) { error = "รหัสใหม่สองครั้งไม่ตรงกัน"; continue; }
    const res = await fetch("/api/admin/pin", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_pin: values.pin, current_pin: values.current }),
    });
    if (res.ok) return true;
    error = await errorMessage(res);
  }
}
