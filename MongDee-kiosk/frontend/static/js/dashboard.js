function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const RANGES = [["today", "วันนี้"], ["7d", "7 วันล่าสุด"], ["30d", "30 วันล่าสุด"], ["all", "ทั้งหมด"]];
const STORAGE_KEY = "mongdee_report_range";

let range = "all";
try {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (RANGES.some(([key]) => key === saved)) range = saved;
} catch (e) { /* private mode */ }

const $ = (id) => document.getElementById(id);

function updateExportLink() {
  const link = $("exportLink");
  if (link) link.href = `/api/analytics/export?range=${range}`;
}

function renderTabs() {
  updateExportLink();
  $("rangeTabs").replaceChildren(...RANGES.map(([key, label]) => {
    const button = el("button", key === range ? "chip active" : "chip", label);
    button.type = "button";
    button.onclick = () => {
      range = key;
      try { localStorage.setItem(STORAGE_KEY, key); } catch (e) { /* private mode */ }
      renderTabs();
      return refresh();
    };
    return button;
  }));
}

async function getJson(path) {
  const res = await fetch(`${path}?range=${range}`, { cache: "no-store" });
  if (!res.ok) throw new Error("โหลดรายงานไม่สำเร็จ");
  return res.json();
}

function renderOverview(overview, summary) {
  $("statTotalScans").textContent = overview.successful;
  $("statUnmatched").textContent = overview.unmatched;
  $("statRate").textContent = overview.success_rate == null ? "-" : `${Math.round(overview.success_rate * 100)}%`;
  $("statProducts").textContent = `${overview.products_scanned}/${overview.products_total}`;
  $("statCorrected").textContent = overview.corrected;
  $("statTopProduct").textContent = summary.length && summary[0].scan_count > 0 ? summary[0].name : "-";
  renderDayNote(overview);
  $("dashNote").textContent = overview.orphaned
    ? `ไม่นับบันทึกสแกน ${overview.orphaned} รายการของสินค้าที่ถูกลบไปแล้ว`
    : "";
}

// "Start a new day" only moves where today's counters begin; nothing is deleted.
function renderDayNote(overview) {
  const note = $("dayNote");
  if (!overview.day_reset) {
    note.replaceChildren();
    return;
  }
  const time = new Date(overview.day_start * 1000).toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit" });
  const undo = el("button", "link-button", "ยกเลิกการรีเซ็ต");
  undo.type = "button";
  undo.onclick = async () => {
    await fetch("/api/analytics/reset-day", { method: "DELETE" });
    return refresh();
  };
  note.replaceChildren(el("span", "", `ตัวเลขของ “วันนี้” เริ่มนับตั้งแต่ ${time} น. (รีเซ็ตเอง) `), undo);
}

async function startNewDay() {
  if (!confirm("เริ่มนับวันใหม่ตอนนี้?\nตัวเลขของ “วันนี้” จะเริ่มจากศูนย์ ข้อมูลเดิมไม่ถูกลบ ยังดูได้ในช่วงเวลาอื่นและส่งออกเป็นรายงานได้")) return;
  try {
    const res = await fetch("/api/analytics/reset-day", { method: "POST" });
    if (!res.ok) throw new Error("รีเซ็ตไม่สำเร็จ");
    range = "today";
    try { localStorage.setItem(STORAGE_KEY, range); } catch (e) { /* private mode */ }
    renderTabs();
    await refresh();
  } catch (err) {
    $("dashUpdated").textContent = err.message;
  }
}

function renderBars(summary) {
  const chart = $("barChart");
  chart.replaceChildren();
  if (!summary.length) {
    chart.append(el("div", "hint", "ยังไม่มีข้อมูลสินค้า"));
    return;
  }
  const maxCount = Math.max(1, ...summary.map((r) => r.scan_count));
  for (const r of summary) {
    const row = el("div", "bar-row");
    row.append(el("div", "name", r.name));
    const track = el("div", "bar-track");
    const fill = el("div", "bar-fill");
    fill.style.width = `${(r.scan_count / maxCount) * 100}%`;
    track.append(fill);
    row.append(track);
    row.append(el("div", "count", String(r.scan_count)));
    chart.append(row);
  }
}

// Vertical bar chart with the count printed above each bar and the tallest bar highlighted.
function drawColumns(chart, labels, values, labelFor, tooltipFor) {
  chart.replaceChildren();
  labels.replaceChildren();
  const maxCount = Math.max(1, ...values);
  values.forEach((count, i) => {
    const bar = el("div", count > 0 && count === maxCount ? "hourly-bar peak" : "hourly-bar");
    bar.style.height = `${Math.max(2, (count / maxCount) * 100)}%`;
    bar.title = tooltipFor(i, count);
    if (count > 0) bar.append(el("span", "bar-value", String(count)));
    chart.append(bar);
    labels.append(el("span", "", labelFor(i)));
  });
}

function renderHourly(rows) {
  const byHour = new Array(24).fill(0);
  for (const r of rows) byHour[r.hour] = r.count;
  const total = byHour.reduce((sum, n) => sum + n, 0);
  const chart = $("hourlyChart");
  const labels = $("hourlyLabels");
  if (!total) {
    chart.replaceChildren(el("div", "hint", "ยังไม่มีข้อมูลในช่วงเวลานี้"));
    labels.replaceChildren();
    $("peakNote").textContent = "";
    return;
  }
  drawColumns(chart, labels, byHour, (h) => (h % 3 === 0 ? String(h) : ""), (h, n) => `${h}:00 - ${n} ครั้ง`);
  const peak = byHour.indexOf(Math.max(...byHour));
  $("peakNote").textContent = `ช่วงที่มีผู้สแกนมากที่สุด: ${peak}:00–${peak}:59 น. (${byHour[peak]} ครั้ง)`;
}

function renderDaily(rows) {
  const chart = $("dailyChart");
  const labels = $("dailyLabels");
  if (!rows.some((r) => r.count > 0)) {
    chart.replaceChildren(el("div", "hint", "ยังไม่มีข้อมูลในช่วงเวลานี้"));
    labels.replaceChildren();
    return;
  }
  const every = rows.length <= 8 ? 1 : rows.length <= 31 ? 5 : 10;
  const short = (day) => `${Number(day.slice(8))}/${Number(day.slice(5, 7))}`;
  drawColumns(chart, labels, rows.map((r) => r.count),
    (i) => (i % every === 0 || i === rows.length - 1 ? short(rows[i].day) : ""),
    (i, n) => `${short(rows[i].day)} - ${n} ครั้ง`);
}

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  const requested = range;
  try {
    const [overview, summary, hourly, daily] = await Promise.all([
      getJson("/api/analytics/overview"), getJson("/api/analytics/summary"),
      getJson("/api/analytics/hourly"), getJson("/api/analytics/daily"),
    ]);
    if (requested !== range) return;  // the range changed while loading
    renderOverview(overview, summary);
    renderBars(summary);
    renderHourly(hourly);
    renderDaily(daily);
    $("dashUpdated").textContent = `อัปเดตล่าสุด ${new Date().toLocaleTimeString("th-TH")}`;
  } catch (err) {
    $("dashUpdated").textContent = err.message;
  } finally {
    refreshing = false;
    if (requested !== range) refresh();
  }
}

$("resetDayBtn").onclick = startNewDay;
renderTabs();
refresh();
setInterval(refresh, 5000);
