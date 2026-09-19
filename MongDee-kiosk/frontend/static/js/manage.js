const productList = document.getElementById("productList");
const emptyState = document.getElementById("emptyState");

let products = [];
const GALLERY_MAX = 8;
const openGalleries = new Set();  // product ids whose photo manager is open; survives re-render

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function loadProducts() {
  const res = await fetch("/api/products");
  products = await res.json();
  render();
}

function render() {
  productList.replaceChildren();
  emptyState.style.display = products.length ? "none" : "block";
  for (const p of products) {
    productList.append(buildCard(p));
  }
}

const EDIT_FIELDS = [
  ["name", "ชื่อสินค้า", "input", "text"],
  ["price", "ราคา (บาท)", "input", "number"],
  ["category", "ประเภทสินค้า", "input", "text"],
  ["origin", "แหล่งที่มา", "input", "text"],
  ["material", "วัสดุ", "input", "text"],
  ["process", "วิธีการผลิต", "input", "text"],
  ["production_date", "วันผลิต", "input", "date"],
  ["expiry_date", "วันหมดอายุ", "input", "date"],
  ["story", "รายละเอียดสินค้า", "textarea", null],
  ["video_link", "ลิงก์วิดีโอนำเสนอ (YouTube ฯลฯ)", "input", "url"],
  ["video_url", "ลิงก์ช่องทางติดต่อ / ข้อมูลเพิ่มเติม", "input", "url"],
];

function formatMoney(n) {
  return new Intl.NumberFormat("th-TH", { minimumFractionDigits: 0, maximumFractionDigits: 2 }).format(n);
}

function buildCard(p) {
  const card = el("div", "card product-card");
  const row = el("div", "row");

  const displayImage = p.cover_image_path || p.thumbnail_path;
  if (displayImage) {
    const img = el("img", "thumb");
    img.src = "/" + displayImage;
    row.append(img);
  } else {
    row.append(el("div", "thumb placeholder", "ไม่มีภาพ"));
  }

  const info = el("div", "info");
  const nameRow = el("div", "name");
  nameRow.textContent = p.name;
  if (p.needs_reembed) {
    const flag = el("span", "pill warn", "ต้องสแกนใหม่");
    flag.style.marginLeft = "0.5rem";
    nameRow.append(flag);
  }
  info.append(nameRow);
  const metaParts = [];
  if (p.price != null) metaParts.push(buildPrice(p.price));
  if (p.category) metaParts.push(p.category);
  if (p.origin) metaParts.push(p.origin);
  if (p.video_path) metaParts.push("มีวิดีโอสินค้า");
  const metaLine = el("div", "meta");
  metaParts.forEach((part, i) => {
    if (i) metaLine.append(" · ");
    metaLine.append(part);
  });
  if (!metaParts.length) metaLine.textContent = "-";
  info.append(metaLine);
  row.append(info);

  const actions = el("div", "actions");
  const coverBtn = el("button", "secondary small", "รูปหน้าปก");
  const galleryBtn = el("button", "secondary small", `รูปภาพ (${(p.gallery || []).length})`);
  galleryBtn.type = "button";
  const videoBtn = el("button", "secondary small", p.video_path ? "เปลี่ยนวิดีโอ" : "อัปโหลดวิดีโอ");
  const exportBtn = el("button", "secondary small", "ส่งออก");
  exportBtn.type = "button";
  exportBtn.onclick = () => { window.location.href = `/api/products/${p.id}/export`; };
  const editBtn = el("button", "secondary small", "แก้ไข");
  const delBtn = el("button", "danger small", "ลบ");
  actions.append(coverBtn, galleryBtn, videoBtn, exportBtn, editBtn, delBtn);
  if (p.video_path) {
    const removeVideoBtn = el("button", "danger small", "ลบวิดีโอ");
    actions.append(removeVideoBtn);
    removeVideoBtn.onclick = async () => {
      removeVideoBtn.disabled = true;
      try {
        const res = await protectedDelete(`/api/products/${p.id}/video`, {
          title: "ลบวิดีโอสินค้า",
          message: "ลบวิดีโอสินค้านี้? หน้าจอแสดงผลจะกลับไปใช้รูปหน้าปกแทน",
        });
        if (!res) { removeVideoBtn.disabled = false; return; }  // cancelled
        if (!res.ok) throw new Error(await errorMessage(res));
        const updated = await res.json();
        const idx = products.findIndex((x) => x.id === updated.id);
        products[idx] = updated;
        render();
      } catch (err) {
        alert(err.message);
        removeVideoBtn.disabled = false;
      }
    };
  }
  row.append(actions);
  card.append(row);

  const coverFormHolder = el("div");
  card.append(coverFormHolder);

  const galleryHolder = el("div");
  card.append(galleryHolder);
  const toggleGallery = (open) => {
    if (open) {
      openGalleries.add(p.id);
      galleryHolder.replaceChildren(buildGalleryManager(p));
    } else {
      openGalleries.delete(p.id);
      galleryHolder.replaceChildren();
    }
    galleryBtn.className = open ? "small" : "secondary small";
  };
  galleryBtn.onclick = () => toggleGallery(!openGalleries.has(p.id));
  if (openGalleries.has(p.id)) toggleGallery(true);

  const formHolder = el("div");
  card.append(formHolder);

  const coverInput = document.createElement("input");
  coverInput.type = "file";
  coverInput.accept = "image/jpeg,image/png,image/webp";
  coverInput.style.display = "none";
  card.append(coverInput);

  const videoInput = document.createElement("input");
  videoInput.type = "file";
  videoInput.accept = "video/mp4,video/webm,video/ogg";
  videoInput.style.display = "none";
  card.append(videoInput);

  const coverStatus = el("div", "hint");
  coverFormHolder.append(coverStatus);

  coverBtn.onclick = () => coverInput.click();
  videoBtn.onclick = () => videoInput.click();

  coverInput.onchange = async () => {
    const file = coverInput.files[0];
    if (!file) return;
    coverStatus.textContent = "กำลังอัปโหลด...";
    coverBtn.disabled = true;
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`/api/products/${p.id}/cover`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "อัปโหลดไม่สำเร็จ");
      }
      const updated = await res.json();
      const idx = products.findIndex((x) => x.id === updated.id);
      products[idx] = updated;
      render();
    } catch (err) {
      coverStatus.textContent = "เกิดข้อผิดพลาด: " + err.message;
    } finally {
      coverBtn.disabled = false;
      coverInput.value = "";
    }
  };

  videoInput.onchange = async () => {
    const file = videoInput.files[0];
    if (!file) return;
    coverStatus.textContent = "กำลังอัปโหลดวิดีโอ...";
    videoBtn.disabled = true;
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`/api/products/${p.id}/video`, { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "อัปโหลดไม่สำเร็จ");
      }
      const updated = await res.json();
      const idx = products.findIndex((x) => x.id === updated.id);
      products[idx] = updated;
      render();
    } catch (err) {
      coverStatus.textContent = "เกิดข้อผิดพลาด: " + err.message;
    } finally {
      videoBtn.disabled = false;
      videoInput.value = "";
    }
  };

  editBtn.onclick = () => {
    if (formHolder.childNodes.length) {
      formHolder.replaceChildren();
      editBtn.textContent = "แก้ไข";
    } else {
      formHolder.append(
        buildEditForm(p, () => {
          formHolder.replaceChildren();
          editBtn.textContent = "แก้ไข";
        })
      );
      editBtn.textContent = "ปิดฟอร์ม";
    }
  };

  delBtn.onclick = async () => {
    delBtn.disabled = true;
    try {
      const res = await protectedDelete(`/api/products/${p.id}`, {
        title: "ลบสินค้า",
        message: `ต้องการลบสินค้า "${p.name}" ใช่หรือไม่? การลบไม่สามารถย้อนกลับได้`,
      });
      if (!res) { delBtn.disabled = false; return; }  // cancelled
      if (!res.ok) throw new Error(await errorMessage(res));
      products = products.filter((x) => x.id !== p.id);
      render();
    } catch (err) {
      alert(err.message);
      delBtn.disabled = false;
    }
  };

  return card;
}

// Photos shown as the popup's thumbnail row. Only these hand-added photos are
// editable; the auto-captured angle frames stay tied to recognition data.
function buildGalleryManager(p) {
  const panel = el("div", "gallery-manager");
  const images = p.gallery || [];

  panel.append(el("div", "gallery-manager-title", `รูปภาพสินค้าในป๊อปอัป (${images.length}/${GALLERY_MAX})`));
  panel.append(el("p", "hint", "เมื่อมีรูปที่เพิ่มเอง ป๊อปอัปจะแสดงเฉพาะรูปเหล่านี้แทนรูปมุมมองอัตโนมัติ ส่วนโหมด 360° ยังใช้รูปจากการสแกนเหมือนเดิม"));

  const status = el("div", "hint");
  const apply = async (request, failure) => {
    status.textContent = "กำลังดำเนินการ...";
    try {
      const res = await request();
      if (!res) { status.textContent = ""; return; }  // cancelled at the PIN prompt
      if (!res.ok) throw new Error(await errorMessage(res));
      const updated = await res.json();
      const idx = products.findIndex((x) => x.id === updated.id);
      products[idx] = updated;
      render();
    } catch (err) {
      status.textContent = "เกิดข้อผิดพลาด: " + (err.message || failure);
    }
  };

  const grid = el("div", "gallery-manager-grid");
  images.forEach((image, index) => {
    const tile = el("div", "gallery-manager-tile");
    const img = el("img");
    img.src = "/" + image.path;
    img.alt = `รูปที่ ${index + 1}`;
    tile.append(img);

    const controls = el("div", "gallery-manager-controls");
    const move = (from, to) => {
      const ids = images.map((x) => x.id);
      [ids[from], ids[to]] = [ids[to], ids[from]];
      apply(() => fetch(`/api/products/${p.id}/images/order`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids }),
      }), "จัดลำดับไม่สำเร็จ");
    };
    const left = el("button", "secondary small", "◀");
    left.type = "button";
    left.title = "ย้ายไปก่อนหน้า";
    left.disabled = index === 0;
    left.onclick = () => move(index, index - 1);
    const right = el("button", "secondary small", "▶");
    right.type = "button";
    right.title = "ย้ายไปถัดไป";
    right.disabled = index === images.length - 1;
    right.onclick = () => move(index, index + 1);
    const del = el("button", "danger small", "ลบ");
    del.type = "button";
    del.onclick = () => apply(() => protectedDelete(`/api/products/${p.id}/images/${image.id}`, {
      title: "ลบรูปภาพ",
      message: "ต้องการลบรูปนี้ออกจากสินค้าใช่หรือไม่? การลบไม่สามารถย้อนกลับได้",
    }), "ลบรูปไม่สำเร็จ");
    controls.append(left, right, del);
    tile.append(controls);
    grid.append(tile);
  });
  if (images.length) panel.append(grid);

  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/jpeg,image/png,image/webp";
  input.multiple = true;
  input.style.display = "none";
  input.onchange = () => {
    const files = [...input.files];
    input.value = "";
    if (!files.length) return;
    const formData = new FormData();
    for (const file of files) formData.append("files", file);
    apply(() => fetch(`/api/products/${p.id}/images`, { method: "POST", body: formData }), "อัปโหลดไม่สำเร็จ");
  };
  const add = el("button", "", "+ เพิ่มรูป");
  add.type = "button";
  add.disabled = images.length >= GALLERY_MAX;
  add.onclick = () => input.click();
  const row = el("div", "button-row");
  row.append(add, input);
  if (images.length >= GALLERY_MAX) row.append(el("span", "hint", `ครบ ${GALLERY_MAX} รูปแล้ว ลบบางรูปก่อนจึงจะเพิ่มได้`));
  panel.append(row, status);
  return panel;
}

function buildEditForm(p, onClose) {
  const form = el("form", "edit-form");
  const inputs = {};

  for (const [key, label, tag, type] of EDIT_FIELDS) {
    const wrap = el("div", "field");
    wrap.append(el("label", "", label));
    const input = document.createElement(tag);
    if (tag === "textarea") input.rows = 4;
    else if (type) input.type = type;
    if (type === "number") input.step = "0.01";
    input.value = p[key] ?? "";
    wrap.append(input);
    form.append(wrap);
    inputs[key] = input;
  }

  const toast = el("div", "hint");

  const btnRow = el("div", "button-row");
  const saveBtn = el("button", "", "บันทึก");
  saveBtn.type = "submit";
  const cancelBtn = el("button", "secondary", "ยกเลิก");
  cancelBtn.type = "button";
  cancelBtn.onclick = onClose;
  btnRow.append(saveBtn, cancelBtn);
  form.append(btnRow, toast);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    saveBtn.disabled = true;
    toast.textContent = "";
    const body = {};
    for (const key in inputs) body[key] = inputs[key].value;
    body.price = body.price === "" ? null : parseFloat(body.price);
    if (body.production_date === "") body.production_date = null;
    if (body.expiry_date === "") body.expiry_date = null;

    try {
      const res = await fetch(`/api/products/${p.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "บันทึกไม่สำเร็จ");
      }
      const updated = await res.json();
      const idx = products.findIndex((x) => x.id === updated.id);
      products[idx] = updated;
      render();
    } catch (err) {
      toast.textContent = "เกิดข้อผิดพลาด: " + err.message;
      saveBtn.disabled = false;
    }
  });

  return form;
}

const changePinBtn = document.getElementById("changePinBtn");
if (changePinBtn) {
  changePinBtn.onclick = async () => {
    const changed = await changeAdminPin();
    document.getElementById("pinStatus").textContent = changed ? "ตั้งรหัสผ่านแอดมินเรียบร้อยแล้ว" : "";
  };
}

loadProducts();
