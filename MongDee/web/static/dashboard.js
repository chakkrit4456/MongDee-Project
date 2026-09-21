let externalInteractions = [];
let externalHealth = [];

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
}

function chartTooltip() {
    let tt = document.getElementById('chart-tooltip');
    if (!tt) {
        tt = document.createElement('div');
        tt.id = 'chart-tooltip';
        tt.className = 'chart-tooltip';
        document.body.appendChild(tt);
    }
    return tt;
}
function showTooltip(x, y, valueText, labelText) {
    const tt = chartTooltip();
    tt.innerHTML = '';
    const v = document.createElement('div');
    v.className = 'tt-value'; v.textContent = valueText;
    const l = document.createElement('div');
    l.className = 'tt-label'; l.textContent = labelText;
    tt.appendChild(v); tt.appendChild(l);
    tt.style.left = x + 'px';
    tt.style.top = y + 'px';
    tt.classList.add('visible');
}
function hideTooltip() {
    chartTooltip().classList.remove('visible');
}

const HOUR_MS = 3600;

// Shared by every "N events per hour, last 24h or one picked calendar day"
// view on this dashboard (tripwire entries, product interest, ...).
// selectedDate ("YYYY-MM-DD", local) ถ้ามี = แบ่งตามชั่วโมง 0:00-23:00 ของวันนั้น
// (ตรงกับ day picker) ถ้าไม่มี (เลือก "ทุกวัน") = 24 ชม.ล่าสุดนับจากตอนนี้
function _bucketByHour(rows, selectedDate) {
    const startHour = selectedDate
        ? new Date(selectedDate + 'T00:00:00').getTime() / 1000
        : Math.floor(Date.now() / 1000 / HOUR_MS) * HOUR_MS - 23 * HOUR_MS;
    const buckets = new Array(24).fill(0);
    for (const r of rows) {
        if (!r.ts) continue;
        const idx = Math.floor((r.ts - startHour) / HOUR_MS);
        if (idx >= 0 && idx < 24) buckets[idx]++;
    }
    return { startHour, buckets };
}

function _hourRangeLabel(startHour, i) {
    const from = new Date((startHour + i * HOUR_MS) * 1000);
    const to = new Date((startHour + (i + 1) * HOUR_MS) * 1000);
    const pad = n => String(n).padStart(2, '0');
    return `${pad(from.getHours())}:00 - ${pad(to.getHours())}:00`;
}

// สินค้าที่ยังไม่มีคนสนใจ Events by hour — spec: MongDee person-gender/age
// master continuation prompt section 41. Reuses holdHistory (product_hold_
// events rows, already fetched for the "ประวัติกิจกรรม" tab) — each row is
// exactly one confirmed interaction session (see core/interest_tracker.py:
// one row per confirm->finalize pair, never one per live-state poll), so a
// plain per-hour COUNT here can never double-count a still-CONFIRMING
// interaction just because the dashboard polled it several times.
function renderHourlyInterest(holdHistory, selectedDate) {
    document.getElementById('hourly-interest-caption').textContent =
        selectedDate ? fmtDateLabel(selectedDate) : '24 ชม.ล่าสุด';
    const body = document.getElementById('hourly-interest-body');
    if (!holdHistory || holdHistory.length === 0) {
        body.innerHTML = emptyRow(2, 'ยังไม่มีข้อมูลความสนใจสินค้า');
        return;
    }
    const { startHour, buckets } = _bucketByHour(holdHistory, selectedDate);
    body.innerHTML = '';
    for (let i = 23; i >= 0; i--) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${_hourRangeLabel(startHour, i)}</td><td>${buckets[i]}</td>`;
        body.appendChild(tr);
    }
}

// คนเข้าบูธรายชั่วโมง — เฉพาะ Tripwire IN crossings จริง (ไม่ใช่ presence_sessions
// ซึ่งเป็นแค่ "กล้องเห็นคนอยู่ในเฟรม" — คนเดียวเดินผ่าน 2 กล้อง หรือถูกกล้องเดียวเห็นซ้ำ
// หลายครั้ง จะทำให้จำนวนเกินจริง). ต้องข้ามเส้น Virtual Tripwire ที่ตั้งค่าไว้จริงถึงจะนับ
function renderTrafficChart(crossings, selectedDate) {
    const wrap = document.getElementById('traffic-chart-wrap');
    wrap.innerHTML = '';
    document.getElementById('traffic-chart-caption').textContent =
        selectedDate ? fmtDateLabel(selectedDate) : '24 ชม.ล่าสุด';
    const visitsBody = document.getElementById('visits-history-body');
    if (!crossings || crossings.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'chart-empty';
        empty.textContent = 'ยังไม่มีข้อมูลคนเข้าบูธ (ยังไม่มีการข้ามเส้น Tripwire)';
        wrap.appendChild(empty);
        visitsBody.innerHTML = emptyRow(2, 'ยังไม่มีข้อมูลคนเข้าบูธ (ยังไม่มีการข้ามเส้น Tripwire)');
        return;
    }

    const { startHour, buckets } = _bucketByHour(crossings, selectedDate);

    // ตารางประวัติคู่กับกราฟ (เหมือน "สินค้ายอดนิยม") — เรียงล่าสุดขึ้นก่อน ให้ตรงกับ
    // ตาราง "ประวัติ..." อื่นๆ ในหน้านี้
    visitsBody.innerHTML = '';
    for (let i = 23; i >= 0; i--) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${_hourRangeLabel(startHour, i)}</td><td>${buckets[i]}</td>`;
        visitsBody.appendChild(tr);
    }

    const W = 420, H = 110, padL = 28, padR = 8, padT = 8, padB = 18;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const maxVal = Math.max(1, ...buckets);
    const niceMax = maxVal <= 5 ? 5 : Math.ceil(maxVal / 5) * 5;

    const xAt = i => padL + (plotW * i) / 23;
    const yAt = v => padT + plotH - (plotH * v) / niceMax;

    const svg = svgEl('svg', { class: 'chart-svg', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'คนเข้าบูธรายชั่วโมง' });

    // gridlines + y ticks (0 / mid / max)
    [0, 0.5, 1].forEach(f => {
        const v = niceMax * f;
        const y = yAt(v);
        svg.appendChild(svgEl('line', { class: 'chart-grid-line', x1: padL, x2: W - padR, y1: y, y2: y }));
        const t = svgEl('text', { class: 'chart-axis-text', x: padL - 6, y: y + 3, 'text-anchor': 'end' });
        t.textContent = Math.round(v);
        svg.appendChild(t);
    });

    // x-axis hour labels (every 4th hour)
    for (let i = 0; i < 24; i++) {
        if (i % 4 !== 0) continue;
        const hourLabel = new Date((startHour + i * HOUR_MS) * 1000).getHours() + ':00';
        const t = svgEl('text', { class: 'chart-axis-text', x: xAt(i), y: H - 6, 'text-anchor': 'middle' });
        t.textContent = hourLabel;
        svg.appendChild(t);
    }

    // area
    let areaD = `M ${xAt(0)} ${yAt(0)} `;
    buckets.forEach((v, i) => { areaD += `L ${xAt(i)} ${yAt(v)} `; });
    areaD += `L ${xAt(23)} ${yAt(0)} Z`;
    svg.appendChild(svgEl('path', { class: 'chart-area-path', d: areaD }));

    // line
    let lineD = '';
    buckets.forEach((v, i) => { lineD += (i === 0 ? 'M ' : 'L ') + xAt(i) + ' ' + yAt(v) + ' '; });
    svg.appendChild(svgEl('path', { class: 'chart-line-path', d: lineD }));

    // end dot on last bucket
    const lastI = 23;
    svg.appendChild(svgEl('circle', { class: 'chart-end-dot', cx: xAt(lastI), cy: yAt(buckets[lastI]), r: 4 }));

    // crosshair (hidden by default)
    const crosshair = svgEl('line', { class: 'chart-crosshair', x1: padL, x2: padL, y1: padT, y2: H - padB });
    svg.appendChild(crosshair);

    // hit columns for hover
    const colW = plotW / 24;
    for (let i = 0; i < 24; i++) {
        const hit = svgEl('rect', {
            class: 'chart-hit-col', x: padL + colW * i, y: padT, width: colW, height: plotH,
        });
        hit.addEventListener('pointermove', (ev) => {
            crosshair.setAttribute('x1', xAt(i)); crosshair.setAttribute('x2', xAt(i));
            crosshair.style.opacity = 1;
            const hourLabel = new Date((startHour + i * HOUR_MS) * 1000).getHours() + ':00 น.';
            showTooltip(ev.clientX, ev.clientY, `${buckets[i]} คน`, hourLabel);
        });
        hit.addEventListener('pointerleave', () => { crosshair.style.opacity = 0; hideTooltip(); });
        svg.appendChild(hit);
    }

    wrap.appendChild(svg);
}

// สินค้ายอดนิยม — แท่งแนวนอน จำนวนครั้งที่หยิบ/ถือ (top 8)
function renderTopProductsChart(movers) {
    const wrap = document.getElementById('top-products-chart-wrap');
    wrap.innerHTML = '';
    const top = [...(movers || [])].sort((a, b) => (b.mover_count || 0) - (a.mover_count || 0)).slice(0, 8);
    if (top.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'chart-empty';
        empty.textContent = 'ยังไม่มีข้อมูลสินค้ายอดนิยม';
        wrap.appendChild(empty);
        return;
    }

    const rowH = 22, barH = 13;
    const padL = 100, padR = 34, padT = 4, padB = 4;
    const W = 420, H = padT + padB + rowH * top.length;
    const plotW = W - padL - padR;
    const maxVal = Math.max(1, ...top.map(m => m.mover_count || 0));

    const svg = svgEl('svg', { class: 'chart-svg', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'สินค้ายอดนิยม' });

    top.forEach((m, i) => {
        const y = padT + rowH * i;
        const barW = Math.max(2, (plotW * (m.mover_count || 0)) / maxVal);

        const nameLabel = svgEl('text', {
            class: 'chart-bar-label', x: padL - 10, y: y + barH / 2 + 4, 'text-anchor': 'end',
        });
        nameLabel.textContent = m.product_name && m.product_name.length > 16
            ? m.product_name.slice(0, 15) + '…' : (m.product_name || '');
        const titleEl = svgEl('title', {});
        titleEl.textContent = m.product_name || '';
        nameLabel.appendChild(titleEl);
        svg.appendChild(nameLabel);

        const bar = svgEl('rect', {
            class: 'chart-bar', x: padL, y: y + (rowH - barH) / 2, width: barW, height: barH, rx: 4,
        });
        bar.addEventListener('pointermove', (ev) => {
            showTooltip(ev.clientX, ev.clientY, `${m.mover_count || 0} ครั้ง`, m.product_name || '');
        });
        bar.addEventListener('pointerleave', hideTooltip);
        svg.appendChild(bar);

        const valueLabel = svgEl('text', {
            class: 'chart-value-text', x: padL + barW + 8, y: y + barH / 2 + 4,
        });
        valueLabel.textContent = m.mover_count || 0;
        svg.appendChild(valueLabel);
    });

    wrap.appendChild(svg);
}

const STATE_LABELS = { idle: 'นิ่ง (ไม่มีการขยับ)', held: 'มีคนถือ', unsettled: 'กำลังตั้งค่าเริ่มต้น' };

// Live interest state (spec: MongDee person-gender/age master continuation
// prompt sections 32-34) — backend (core/interest_tracker.py's live_status)
// is the source of truth; this dashboard only ever renders whatever it
// says, polled via /api/dashboard/live same as every other "live" number
// on this page (no WebSocket in this app — see that endpoint's own notes).
const LIVE_STATUS_LABELS = {
    NOT_INTERACTING: 'ไม่มีการโต้ตอบ',
    POSSIBLE_INTERACTION: '⏳ กำลังตรวจสอบความสนใจ',
    INTEREST_CONFIRMED: '❤️ สนใจสินค้า',
};

function fmtTs(ts) {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleString('th-TH', { hour12: false });
}

function emptyRow(colspan, message) {
    return `<tr><td colspan="${colspan}" class="table-empty">${message}</td></tr>`;
}

// "สรุปทั้งงาน (Event)" tab — only meaningful once an Event ID is actually
// selected (a per-day breakdown only makes sense *within* one event; mixing
// events would be exactly the "ห้ามนำข้อมูลจากงานอื่นมาปะปน" the feature
// exists to prevent), so this is the one section of the dashboard that
// stays hidden behind a hint until the Event filter is set — unlike every
// other card/table here, which already work with no event selected at all.
let lastEventSummaryKey = null;

async function refreshEventSummary(eventId, boothId) {
    const hint = document.getElementById('event-summary-hint');
    const content = document.getElementById('event-summary-content');
    if (!eventId) {
        hint.classList.remove('hidden');
        content.classList.add('hidden');
        lastEventSummaryKey = null;
        return;
    }
    // Avoid refetching on every 5s poll if the selection hasn't changed —
    // this data only changes as fast as new crossings/interactions land,
    // and the existing 5s poll for everything else already covers that;
    // this just skips redundant identical requests when nothing selected
    // has changed since the last successful fetch.
    const key = `${eventId}|${boothId || ''}`;
    if (key === lastEventSummaryKey) return;

    let sessions, breakdown;
    try {
        [sessions, breakdown] = await Promise.all([
            fetchLocal(`/api/dashboard/event_sessions?event_id=${encodeURIComponent(eventId)}`),
            fetchLocal(`/api/dashboard/event_daily_breakdown?event_id=${encodeURIComponent(eventId)}` +
                (boothId ? `&booth_id=${encodeURIComponent(boothId)}` : '')),
        ]);
    } catch (e) {
        return; // fetchLocal already surfaces the error banner; keep showing stale data
    }
    lastEventSummaryKey = key;

    hint.classList.add('hidden');
    content.classList.remove('hidden');

    const sessionsBody = document.getElementById('event-sessions-body');
    sessionsBody.innerHTML = sessions.length ? '' : emptyRow(3, 'ยังไม่มีข้อมูลสำหรับงานนี้');
    for (const s of sessions) {
        const schedule = (s.open_time && s.close_time) ? `${s.open_time}–${s.close_time}` : 'ไม่ได้ตั้งตาราง';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${s.date}</td><td>${s.booth_name}</td><td>${schedule}</td>`;
        sessionsBody.appendChild(tr);
    }

    const breakdownBody = document.getElementById('event-breakdown-body');
    breakdownBody.innerHTML = breakdown.length ? '' : emptyRow(7, 'ยังไม่มีข้อมูลสำหรับงานนี้');
    for (const d of breakdown) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${d.date}</td><td>${d.visitors_in}</td><td>${d.visitors_out}</td>` +
            `<td>${d.peak_inside}</td><td>${d.total_interactions}</td><td>${d.unique_products}</td>` +
            `<td>${d.presence_count ? d.avg_dwell_sec.toFixed(1) : 'N/A'}</td>`;
        breakdownBody.appendChild(tr);
    }
}

function confidenceBadge(conf) {
    if (typeof conf !== 'number') return '';
    const pct = Math.round(conf * 100);
    const cls = conf >= 0.75 ? 'conf-good' : conf >= 0.5 ? 'conf-mid' : 'conf-low';
    return `<span class="confidence-badge ${cls}">${pct}%</span>`;
}

// สถานะที่ต้องคงไว้ระหว่างรอบ refresh() เพื่อให้ช่องค้นหาในแท็บ "ประวัติกิจกรรม"
// กรองข้อมูลที่ดึงมาแล้วได้ทันทีโดยไม่ต้องยิง fetch ใหม่ทุกครั้งที่พิมพ์
let lastFilteredInteractions = [];
let lastHoldHistory = [];
let lastPresenceHistory = [];
let lastBoothLabels = {};

const CATEGORY_LABELS = { female: 'หญิง', male: 'ชาย', child: 'เด็ก', unknown: 'ไม่ทราบ' };

function matchesSearch(productName, query) {
    return !query || (productName || '').toLowerCase().includes(query);
}

function matchesAny(query, ...fields) {
    return !query || fields.some(f => (f || '').toLowerCase().includes(query));
}

function renderHistoryTables() {
    const query = document.getElementById('history-search').value.trim().toLowerCase();

    const interactionsBody = document.getElementById('interactions-body');
    const sortedInteractions = [...lastFilteredInteractions]
        .filter(r => matchesSearch(r.product_name, query))
        .sort((a, b) => (b.ts || 0) - (a.ts || 0)).slice(0, 200);
    interactionsBody.innerHTML = sortedInteractions.length ? '' : emptyRow(5,
        query ? `ไม่พบปฏิสัมพันธ์ที่ตรงกับ "${query}"` : 'ยังไม่มีปฏิสัมพันธ์กับสินค้า — ระบบพร้อมทำงาน รอคนเดินผ่านหน้ากล้อง');
    for (const r of sortedInteractions) {
        const tr = document.createElement('tr');
        const boothName = lastBoothLabels[r.booth_id] || r.booth_id || '';
        tr.innerHTML = `<td>${fmtTs(r.ts)}</td><td>${boothName}</td><td>${r.camera_id || ''}</td>` +
            `<td>${r.product_name || ''}</td><td>${confidenceBadge(r.confidence)}</td>`;
        interactionsBody.appendChild(tr);
    }

    const holdBody = document.getElementById('hold-history-body');
    const filteredHold = lastHoldHistory.filter(h => matchesSearch(h.product_name, query));
    holdBody.innerHTML = filteredHold.length ? '' : emptyRow(4,
        query ? `ไม่พบประวัติการหยิบที่ตรงกับ "${query}"` : 'ยังไม่มีประวัติการหยิบจับสินค้า');
    for (const h of filteredHold) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${fmtTs(h.ts)}</td><td>${h.camera_id}</td><td>${h.product_name}</td><td>${h.duration_sec.toFixed(1)}</td>`;
        holdBody.appendChild(tr);
    }

    const presenceBody = document.getElementById('presence-history-body');
    const filteredPresence = lastPresenceHistory.filter(p => {
        const boothName = lastBoothLabels[p.booth_id] || p.booth_id || '';
        const catLabel = CATEGORY_LABELS[p.category] || '';
        return matchesAny(query, boothName, p.camera_id, catLabel);
    });
    presenceBody.innerHTML = filteredPresence.length ? '' : emptyRow(5,
        query ? `ไม่พบประวัติที่ตรงกับ "${query}"` : 'ยังไม่มีประวัติช่วงเวลาที่คนอยู่ในกล้อง');
    for (const p of filteredPresence) {
        const tr = document.createElement('tr');
        const boothName = lastBoothLabels[p.booth_id] || p.booth_id || '';
        const catLabel = CATEGORY_LABELS[p.category] || CATEGORY_LABELS.unknown;
        tr.innerHTML = `<td>${fmtTs(p.ts)}</td><td>${boothName}</td><td>${p.camera_id}</td>` +
            `<td>${catLabel}</td><td>${p.duration_sec.toFixed(1)}</td>`;
        presenceBody.appendChild(tr);
    }

    renderTimeline(query, sortedInteractions, filteredHold, filteredPresence);
}

const TIMELINE_LIMIT = 300;

// รวม 3 แหล่งข้อมูล (interactions / hold events / presence sessions) เป็น
// feed เดียวเรียงเวลา แทนที่จะให้ต้องเปิดดู 3 ตารางแยกกันเพื่อปะติดปะต่อว่า
// ช่วงนั้นเกิดอะไรขึ้นบ้าง — ตารางแบบละเอียดเดิมยังอยู่ครบใน <details> ด้านล่าง
function renderTimeline(query, interactions, holdEvents, presenceSessions) {
    const list = document.getElementById('timeline-list');
    const entries = [];

    for (const r of interactions) {
        const boothName = lastBoothLabels[r.booth_id] || r.booth_id || '';
        const conf = typeof r.confidence === 'number' ? ` (มั่นใจ ${(r.confidence * 100).toFixed(0)}%)` : '';
        entries.push({
            ts: r.ts, type: 't-interaction', icon: 'box',
            text: `พบสินค้า “${r.product_name || ''}” ที่ ${r.camera_id || ''}${boothName ? ' · ' + boothName : ''}${conf}`,
        });
    }
    for (const h of holdEvents) {
        entries.push({
            ts: h.ts, type: 't-hold', icon: 'activity',
            text: `มีคนหยิบ “${h.product_name}” ที่ ${h.camera_id} เป็นเวลา ${h.duration_sec.toFixed(1)} วินาที`,
        });
    }
    for (const p of presenceSessions) {
        const boothName = lastBoothLabels[p.booth_id] || p.booth_id || '';
        const catLabel = CATEGORY_LABELS[p.category] || CATEGORY_LABELS.unknown;
        entries.push({
            ts: p.ts, type: 't-presence', icon: 'users',
            text: `มีคนอยู่หน้ากล้อง ${p.camera_id}${boothName ? ' · ' + boothName : ''} ${p.duration_sec.toFixed(1)} วินาที (${catLabel})`,
        });
    }

    entries.sort((a, b) => (b.ts || 0) - (a.ts || 0));
    const shown = entries.slice(0, TIMELINE_LIMIT);

    if (shown.length === 0) {
        list.innerHTML = `<div class="table-empty">${query ? `ไม่พบกิจกรรมที่ตรงกับ "${query}"` : 'ยังไม่มีกิจกรรม — ระบบพร้อมทำงาน รอคนเดินผ่านหน้ากล้อง'}</div>`;
        return;
    }
    list.innerHTML = '';
    for (const e of shown) {
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.innerHTML =
            `<span class="timeline-icon ${e.type}">${iconHtml(e.icon, { size: 14 })}</span>` +
            `<div class="timeline-body"><div></div><div class="timeline-time">${fmtTs(e.ts)}</div></div>`;
        item.querySelector('.timeline-body > div').textContent = e.text;
        list.appendChild(item);
    }
}

async function fetchLocal(path) {
    const res = await fetch(path);
    return res.json();
}

// "YYYY-MM-DD" in the browser's local timezone — matches SQLite's
// date(ts,'unixepoch','localtime') on the server (see core/database.py's
// query_available_dates), so a day picked here lines up with the day the
// server used to bucket that same row.
function localDateStr(ts) {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    const pad = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function applyFilters(rows) {
    const eventId = document.getElementById('event-filter').value;
    const boothId = document.getElementById('booth-filter').value;
    const date = document.getElementById('date-filter').value;
    return rows.filter(r =>
        (!eventId || r.event_id === eventId) &&
        (!boothId || r.booth_id === boothId) &&
        (!date || localDateStr(r.ts) === date)
    );
}

function syncSelect(select, values, labels) {
    labels = labels || {};
    const current = select.value;
    select.innerHTML = '<option value="">ทั้งหมด</option>';
    for (const v of values) {
        const opt = document.createElement('option');
        opt.value = v; opt.textContent = labels[v] ? `${labels[v]} (${v})` : v;
        select.appendChild(opt);
    }
    select.value = values.includes(current) ? current : '';
}

function fmtDateLabel(dateStr) {
    const d = new Date(dateStr + 'T00:00:00');
    const label = d.toLocaleDateString('th-TH', { day: 'numeric', month: 'short', year: 'numeric' });
    return dateStr === localDateStr(Date.now() / 1000) ? `${label} (วันนี้)` : label;
}

function syncDateSelect(select, dates) {
    const current = select.value;
    select.innerHTML = '<option value="">ทุกวัน</option>';
    for (const d of dates) {
        const opt = document.createElement('option');
        opt.value = d; opt.textContent = fmtDateLabel(d);
        select.appendChild(opt);
    }
    select.value = dates.includes(current) ? current : '';
}

// event_id/booth_id only — used to scope the day-picker's own option list,
// which obviously can't itself be filtered by the day being picked.
function eventBoothQuery() {
    const eventId = document.getElementById('event-filter').value;
    const boothId = document.getElementById('booth-filter').value;
    const params = new URLSearchParams();
    if (eventId) params.set('event_id', eventId);
    if (boothId) params.set('booth_id', boothId);
    return params.toString() ? `?${params.toString()}` : '';
}

function currentQuery() {
    const params = new URLSearchParams(eventBoothQuery().slice(1));
    const date = document.getElementById('date-filter').value;
    if (date) params.set('date', date);
    return params.toString() ? `?${params.toString()}` : '';
}

const BREAKDOWN_ITEMS = [
    { key: 'female', dot: 'dot-female', label: 'หญิง' },
    { key: 'male', dot: 'dot-male', label: 'ชาย' },
    { key: 'unknown', dot: 'dot-unknown', label: 'ไม่ทราบ' },
];

function renderGenderBreakdown(breakdown) {
    const row = document.getElementById('gender-breakdown-row');
    const total = BREAKDOWN_ITEMS.reduce((sum, it) => sum + (breakdown[it.key] || 0), 0);
    if (total === 0) {
        row.innerHTML = '<div class="list-empty">ยังไม่มีข้อมูลคนเข้าบูธ</div>';
        return;
    }
    row.innerHTML = '';
    for (const it of BREAKDOWN_ITEMS) {
        const n = breakdown[it.key] || 0;
        const pct = Math.round((n / total) * 100);
        const div = document.createElement('div');
        div.className = 'breakdown-item';
        div.innerHTML = `<span class="dot ${it.dot}"></span>` +
            `<span class="breakdown-count">${n}</span>` +
            `<span class="breakdown-label">${it.label} (${pct}%)</span>`;
        row.appendChild(div);
    }
}

const ATTRIBUTE_GENDER_ITEMS = [
    { key: 'FEMALE', dot: 'dot-female', label: 'หญิง (โมเดลคาดการณ์)' },
    { key: 'MALE', dot: 'dot-male', label: 'ชาย (โมเดลคาดการณ์)' },
    { key: 'UNKNOWN', dot: 'dot-unknown', label: 'ไม่ทราบ' },
];

function _renderBreakdownRow(rowId, items, counts, emptyText) {
    const row = document.getElementById(rowId);
    const total = items.reduce((sum, it) => sum + (counts[it.key] || 0), 0);
    if (total === 0) {
        row.innerHTML = `<div class="list-empty">${emptyText}</div>`;
        return;
    }
    row.innerHTML = '';
    for (const it of items) {
        const n = counts[it.key] || 0;
        const pct = Math.round((n / total) * 100);
        const div = document.createElement('div');
        div.className = 'breakdown-item';
        div.innerHTML = `<span class="dot ${it.dot}"></span>` +
            `<span class="breakdown-count">${n}</span>` +
            `<span class="breakdown-label">${it.label} (${pct}%)</span>`;
        row.appendChild(div);
    }
}

// Spec: MongDee_Master_Prompt_FairFace_Age_Gender.md section 16/17 — a
// separate, clearly-labeled "model-predicted" breakdown keyed by distinct
// Global Person (core/reid.py + core/attributes.py), never mixed with the
// existing per-visit gender-breakdown card above.
function renderAttributeBreakdown(breakdown) {
    _renderBreakdownRow('attribute-gender-breakdown-row', ATTRIBUTE_GENDER_ITEMS,
        breakdown.gender || {}, 'ยังไม่มีข้อมูล — ตั้งค่า FairFace ก่อน');
}

// Product Interest Overview KPI cards — spec: MongDee person-gender/age
// master prompt section 39. total/unique come straight from
// /api/dashboard/interest_overview; the with/without-interest product
// counts are filled in by renderNoInterestProducts below (it already has to
// cross-reference the full catalog against productInterestSummary anyway).
function renderInterestOverview(overview) {
    document.getElementById('stat-interest-total').textContent = overview.total_interest_events;
    document.getElementById('stat-interest-unique-persons').textContent = overview.unique_interested_persons;
}

// Gender × Product / Age × Product matrices (spec sections 28/29/43/44) —
// each row already comes pre-aggregated per product from the server; this
// just renders it, newest/most-interest-first for parity with the other
// product tables on this tab.
function renderProductGenderMatrix(rows) {
    const body = document.getElementById('product-gender-matrix-body');
    body.innerHTML = rows.length ? '' : emptyRow(4, 'ยังไม่มีข้อมูลความสนใจสินค้า');
    for (const r of [...rows].sort((a, b) => (b.MALE + b.FEMALE + b.UNKNOWN) - (a.MALE + a.FEMALE + a.UNKNOWN))) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${r.product_name}</td><td>${r.MALE}</td><td>${r.FEMALE}</td><td>${r.UNKNOWN}</td>`;
        body.appendChild(tr);
    }
}

// Spec section 37: "Product With No Interest" must still show up, so this
// combines the *full* catalog (products, from /api/products — the source
// of truth for what products exist, since product_hold_events only ever
// has rows for products someone actually touched) against
// productInterestSummary's per-product interest_count, the client-side
// equivalent of the spec's suggested LEFT JOIN (there's no SQL products
// table to join against — see query_product_interest_summary's docstring).
// Also fills in the "products with/without interest" KPI cards.
function renderNoInterestProducts(products, interestSummary) {
    const interestedKeys = new Set(interestSummary.map(r => r.product_key));
    const withoutInterest = products.filter(p => !interestedKeys.has(p.key));

    document.getElementById('stat-products-with-interest').textContent = interestedKeys.size;
    document.getElementById('stat-products-without-interest').textContent = withoutInterest.length;

    const body = document.getElementById('no-interest-products-body');
    body.innerHTML = withoutInterest.length
        ? '' : emptyRow(2, products.length ? 'ทุกสินค้าในระบบมีคนสนใจแล้วอย่างน้อย 1 ครั้ง' : 'ยังไม่มีสินค้าในระบบ');
    for (const p of withoutInterest) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${p.name}</td><td>${p.tagline || ''}</td>`;
        body.appendChild(tr);
    }
}

async function _refreshImpl() {
    const query = currentQuery();
    const [interactions, health, products, live, movers, presence, holdHistory, presenceHistory,
           knownIds, events, booths, availableDates, genderBreakdown, tripwireStats, uniquePeople,
           attributeBreakdown, tripwireCrossings, interestOverview, productGenderMatrix,
           productAgeMatrix, genderInterestTotals, ageInterestTotals, productInterestSummary] = await Promise.all([
        fetchLocal('/api/dashboard/interactions'),
        fetchLocal('/api/dashboard/health'),
        fetchLocal('/api/products'),
        fetchLocal('/api/dashboard/live'),
        fetchLocal(`/api/dashboard/product_movers${query}`),
        fetchLocal(`/api/dashboard/presence_stats${query}`),
        fetchLocal(`/api/dashboard/product_hold_history${query}`),
        fetchLocal(`/api/dashboard/presence_sessions${query}`),
        fetchLocal('/api/dashboard/known_ids'),
        fetchLocal('/api/registry/events'),
        fetchLocal('/api/registry/booths'),
        fetchLocal(`/api/dashboard/available_dates${eventBoothQuery()}`),
        fetchLocal(`/api/dashboard/presence_breakdown${query}`),
        fetchLocal(`/api/dashboard/tripwire_stats${query}`),
        fetchLocal(`/api/dashboard/unique_people${query}`),
        fetchLocal(`/api/dashboard/attribute_breakdown${query}`),
        fetchLocal(`/api/dashboard/tripwire_crossings${query}`),
        fetchLocal(`/api/dashboard/interest_overview${query}`),
        fetchLocal(`/api/dashboard/product_gender_matrix${query}`),
        fetchLocal(`/api/dashboard/product_age_matrix${query}`),
        fetchLocal(`/api/dashboard/gender_interest_totals${query}`),
        fetchLocal(`/api/dashboard/age_interest_totals${query}`),
        fetchLocal(`/api/dashboard/product_interest_summary${query}`),
    ]);
    syncDateSelect(document.getElementById('date-filter'), availableDates);
    renderGenderBreakdown(genderBreakdown);
    const allInteractions = interactions.concat(externalInteractions);
    const allHealth = health.concat(externalHealth);

    // The registry (Events/Booths created via /settings) is the primary,
    // curated source — known_ids (scans every log table) is unioned in as a
    // fallback so pre-registry/orphaned log data is still selectable, just
    // without a friendly name.
    const eventLabels = Object.fromEntries(events.map(e => [e.id, e.name]));
    const boothLabels = Object.fromEntries(booths.map(b => [b.id, b.name]));
    const eventIds = [...new Set([
        ...events.map(e => e.id),
        ...knownIds.event_ids,
        ...allInteractions.concat(allHealth).map(r => r.event_id).filter(Boolean),
    ])].sort();
    const boothIds = [...new Set([
        ...booths.map(b => b.id),
        ...knownIds.booth_ids,
        ...allInteractions.concat(allHealth).map(r => r.booth_id).filter(Boolean),
    ])].sort();
    syncSelect(document.getElementById('event-filter'), eventIds, eventLabels);
    syncSelect(document.getElementById('booth-filter'), boothIds, boothLabels);

    const exportLink = document.getElementById('export-link');
    exportLink.href = `/api/dashboard/export.xlsx${query}`;

    const filteredInteractions = applyFilters(allInteractions);
    const filteredHealth = applyFilters(allHealth);

    const activeBoothIds = new Set([...filteredInteractions, ...filteredHealth].map(r => r.booth_id));

    // ชื่อบูธ/Event ที่กำลังดูอยู่ — ใช้ตัวที่เลือกในฟิลเตอร์ ถ้าไม่ได้เลือกใช้บูธที่เปิดหน้านี้อยู่
    const selectedBoothId = document.getElementById('booth-filter').value || document.body.dataset.boothId;
    const selectedEventId = document.getElementById('event-filter').value || document.body.dataset.eventId;
    document.getElementById('context-booth-name').textContent =
        (selectedBoothId && (boothLabels[selectedBoothId] || selectedBoothId)) || 'ทุกบูธ';
    document.getElementById('context-event-name').textContent =
        selectedEventId ? `· Event: ${eventLabels[selectedEventId] || selectedEventId}` : '';

    // Alerts (Open Alerts stat + the critical-alert banner) were removed from this dashboard --
    // they're live on the booth screen itself now (web/templates/booth.html's "การแจ้งเตือนล่าสุด"
    // panel, driven by state.recent_alerts in booth.js), so a duplicate summary here was redundant.
    // Device/health HISTORY (the "อุปกรณ์ & แจ้งเตือน" tab below) is unchanged -- that's a
    // filterable audit record across booths/events the live booth panel can't replace.

    // Spec: "จำนวนคนทั้งหมด" must only ever come from actual Virtual
    // Tripwire line crossings (core/tripwire.py) — never presence_sessions
    // (a raw "camera saw someone in frame" count, which over-counts anyone
    // seen by more than one camera or re-detected after a brief gap).
    document.getElementById('stat-visits').textContent = tripwireStats.total_in;
    document.getElementById('stat-total').textContent = filteredInteractions.length;
    document.getElementById('stat-products').textContent = new Set(filteredInteractions.map(r => r.product_name)).size;
    document.getElementById('stat-booths').textContent = activeBoothIds.size;
    document.getElementById('stat-people-now').textContent = live.people_now;
    document.getElementById('stat-avg-presence').textContent = presence.avg_sec.toFixed(1);
    document.getElementById('stat-tripwire-in').textContent = tripwireStats.total_in;
    document.getElementById('stat-tripwire-out').textContent = tripwireStats.total_out;
    document.getElementById('stat-tripwire-peak').textContent = tripwireStats.peak_inside;
    // Spec section 28: must never equal a raw per-camera visible-detection
    // sum — this is COUNT(DISTINCT global Re-ID identity), see
    // core/reid.py + /api/dashboard/unique_people.
    document.getElementById('stat-unique-people').textContent = uniquePeople.unique_people;
    renderAttributeBreakdown(attributeBreakdown);

    refreshEventSummary(selectedEventId, selectedBoothId);

    const catalogList = document.getElementById('catalog-list');
    document.getElementById('catalog-count').textContent = products.length;
    catalogList.innerHTML = products.length
        ? '' : '<div class="list-empty">ยังไม่มีสินค้าในระบบ — ไปเพิ่มที่หน้า AI Trainer</div>';
    const sortedProducts = [...products].sort((a, b) => a.name.localeCompare(b.name, 'th'));
    for (const p of sortedProducts) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'catalog-chip';
        btn.title = `ดูประวัติของ "${p.name}"`;
        const nameEl = document.createElement('span');
        nameEl.className = 'catalog-chip-name';
        nameEl.textContent = p.name;
        const priceEl = document.createElement('span');
        priceEl.className = 'catalog-chip-price';
        priceEl.textContent = p.price ? '฿' + p.price : '';
        btn.append(nameEl, priceEl);
        btn.addEventListener('click', () => goToProductLog(p.name));
        catalogList.appendChild(btn);
    }

    const liveBody = document.getElementById('live-states-body');
    liveBody.innerHTML = live.products.length ? '' : emptyRow(4, 'ยังไม่มีสินค้าที่กำลังมีคนสนใจอยู่ขณะนี้');
    for (const p of live.products) {
        const tr = document.createElement('tr');
        const statusText = LIVE_STATUS_LABELS[p.live_status] || STATE_LABELS[p.state] || p.state;
        if (p.live_status === 'INTEREST_CONFIRMED') tr.className = 'interest-confirmed-row';
        tr.innerHTML = `<td>${p.camera_id}</td><td>${p.product_name}</td>` +
            `<td>${statusText}</td><td>${p.interest_seconds.toFixed(1)}</td>`;
        liveBody.appendChild(tr);
    }

    renderTopProductsChart(movers);
    const topBody = document.getElementById('top-products-body');
    topBody.innerHTML = movers.length ? '' : emptyRow(4, 'ยังไม่มีข้อมูลสินค้ายอดนิยม — รอคนหยิบ/ถือสินค้าหน้ากล้อง');
    for (const m of movers) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${m.product_name}</td><td>${m.mover_count}</td>` +
            `<td>${m.unique_persons || 0}</td><td>${(m.total_interest_sec || 0).toFixed(1)}</td>`;
        topBody.appendChild(tr);
    }

    const inCrossings = tripwireCrossings.filter(c => c.direction === 'in');
    renderTrafficChart(inCrossings, document.getElementById('date-filter').value);

    renderInterestOverview(interestOverview);
    renderProductGenderMatrix(productGenderMatrix);
    _renderBreakdownRow('interest-gender-breakdown-row', ATTRIBUTE_GENDER_ITEMS,
        genderInterestTotals, 'ยังไม่มีข้อมูลความสนใจ — ตั้งค่า Re-ID/FairFace ก่อนถึงจะแยกเพศได้');
    renderNoInterestProducts(products, productInterestSummary);
    renderHourlyInterest(holdHistory, document.getElementById('date-filter').value);

    lastFilteredInteractions = filteredInteractions;
    lastHoldHistory = holdHistory;
    lastPresenceHistory = presenceHistory;
    lastBoothLabels = boothLabels;
    renderHistoryTables();

    const healthBody = document.getElementById('health-body');
    const sortedHealth = [...filteredHealth].sort((a, b) => (b.ts || 0) - (a.ts || 0)).slice(0, 100);
    healthBody.innerHTML = sortedHealth.length ? '' : emptyRow(5, 'ยังไม่มีบันทึกสถานะอุปกรณ์');
    for (const r of sortedHealth) {
        const device = r.component ? `${r.camera_id || ''} (${r.component})` : (r.camera_id || '');
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${fmtTs(r.ts)}</td><td>${r.booth_id || ''}</td><td>${device}</td>` +
            `<td>${r.status || ''}</td><td>${r.message || ''}</td>`;
        healthBody.appendChild(tr);
    }
}

// "อัปเดตล่าสุด" — เพราะหน้านี้รีเฟรชอัตโนมัติทุก 5 วิแบบเงียบๆ ผู้ใช้ไม่มีทาง
// รู้เองว่าตัวเลขที่เห็นสดจริงหรือค้างอยู่ (เช่น เซิร์ฟเวอร์รีสตาร์ท/เน็ตหลุดชั่วคราว)
// จนกว่าจะมีจุดบอกเวลาชัดเจน — อัปเดตเฉพาะตอน fetch สำเร็จเท่านั้น ถ้าล้มเหลว
// เวลาจะค้างที่ครั้งล่าสุดที่ยังสำเร็จ พร้อมข้อความเตือนแยกต่างหาก
let lastUpdateAt = null;
let lastRefreshFailed = false;

async function refresh() {
    try {
        await _refreshImpl();
        lastUpdateAt = Date.now();
        lastRefreshFailed = false;
    } catch (e) {
        lastRefreshFailed = true;
        console.error('refresh() failed:', e);
    }
    renderLastUpdated();
}

function renderLastUpdated() {
    const el = document.getElementById('last-updated-text');
    if (!el) return;
    if (lastUpdateAt === null) {
        el.innerHTML = lastRefreshFailed
            ? '<span class="refresh-dot"></span>เชื่อมต่อเซิร์ฟเวอร์ไม่ได้'
            : '<span class="refresh-dot"></span>กำลังโหลดข้อมูล...';
        el.classList.toggle('stale', lastRefreshFailed);
        return;
    }
    const secsAgo = Math.max(0, Math.round((Date.now() - lastUpdateAt) / 1000));
    const ago = secsAgo < 3 ? 'เมื่อสักครู่' : `${secsAgo} วินาทีที่แล้ว`;
    const failedNote = lastRefreshFailed ? ' (รีเฟรชล่าสุดล้มเหลว — เชื่อมต่อมีปัญหา)' : '';
    // เกิน 30 วิยังไม่ได้ข้อมูลใหม่ (รีเฟรชควรทำงานทุก 5 วิ) = สัญญาณว่าข้อมูลอาจค้าง
    const stale = secsAgo > 30 || lastRefreshFailed;
    el.innerHTML = `<span class="refresh-dot"></span>อัปเดตล่าสุด: ${ago}${failedNote}`;
    el.classList.toggle('stale', stale);
}

setInterval(renderLastUpdated, 1000);

// คลิกสินค้าในการ์ด "สินค้าในระบบ" (แท็บภาพรวม) -> เด้งไปแท็บประวัติกิจกรรม
// พร้อมกรองด้วยชื่อสินค้านั้นทันที โดยใช้ช่องค้นหา/เส้นเวลาที่มีอยู่แล้ว
function goToProductLog(productName) {
    document.querySelector('.tab-btn[data-tab="tab-history"]').click();
    const searchInput = document.getElementById('history-search');
    searchInput.value = productName;
    renderHistoryTables();
    searchInput.focus();
}

document.getElementById('refresh-btn').addEventListener('click', refresh);
document.getElementById('event-filter').addEventListener('change', refresh);
document.getElementById('booth-filter').addEventListener('change', refresh);
document.getElementById('date-filter').addEventListener('change', refresh);
document.getElementById('history-search').addEventListener('input', renderHistoryTables);

// ปุ่ม "ล้างข้อมูล" เดียวบนแดชบอร์ด — ล้างเฉพาะข้อมูลที่บันทึกไว้ (interactions,
// presence sessions, tripwire crossings ฯลฯ) ตามตัวกรอง Event ID / Booth ID /
// วันที่ ที่เลือกอยู่ (เลือกกี่อย่างก็ได้ รวมกันได้) — ไม่ลบตัว Booth/Event ที่
// ลงทะเบียนไว้เอง (นั่นเป็นหน้าที่ของหน้า /settings) ดังนั้นปลอดภัยกว่าการลบ
// registry entry ทั้งอัน.
async function clearDashboardData() {
    const eventId = document.getElementById('event-filter').value;
    const boothId = document.getElementById('booth-filter').value;
    const date = document.getElementById('date-filter').value;

    const scopeParts = [];
    if (eventId) scopeParts.push(`Event ID "${eventId}"`);
    if (boothId) scopeParts.push(`Booth ID "${boothId}"`);
    if (date) scopeParts.push(`วันที่ ${date}`);

    const payload = { event_id: eventId || null, booth_id: boothId || null, date: date || null };
    let confirmMsg;
    if (scopeParts.length > 0) {
        confirmMsg = `ล้างข้อมูลที่บันทึกไว้ทั้งหมดของ ${scopeParts.join(' + ')} หรือไม่?\n\n` +
            'การกระทำนี้ย้อนกลับไม่ได้ (ไม่กระทบการตั้งค่า Booth/Event/เส้นนับคน — ลบเฉพาะประวัติที่บันทึกไว้)';
    } else {
        confirmMsg = 'ไม่ได้เลือกตัวกรองใดเลย — นี่จะ "ล้างข้อมูลที่บันทึกไว้ทั้งหมดของทุกบูธ ทุก Event ทุกวัน"\n\n' +
            'ยืนยันหรือไม่? การกระทำนี้ย้อนกลับไม่ได้';
        payload.confirm_all = true;
    }
    if (!confirm(confirmMsg)) return;

    const res = await fetch('/api/dashboard/delete_scope', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'ล้มเหลว' }));
        alert(`ล้างข้อมูลไม่สำเร็จ: ${err.detail}`);
        return;
    }
    await refresh();
}

document.getElementById('clear-data-btn').addEventListener('click', clearDashboardData);

document.getElementById('import-btn').addEventListener('click', () => {
    document.getElementById('import-file').click();
});
document.getElementById('import-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const data = JSON.parse(reader.result);
            externalInteractions = externalInteractions.concat(data.interactions || []);
            externalHealth = externalHealth.concat(data.health_events || []);
            alert(`นำเข้าข้อมูลจาก ${file.name} สำเร็จ`);
            refresh();
        } catch (err) {
            alert('นำเข้าล้มเหลว: ไฟล์ไม่ใช่ JSON ที่ถูกต้อง');
        }
    };
    reader.readAsText(file);
});

// โหมดพรีเซนต์ — ซ่อน nav/ตัวกรอง/ปุ่มจัดการข้อมูล เหลือแต่ตัวเลข+กราฟตัวใหญ่
// สำหรับเปิดโชว์กรรมการ ใช้ทั้ง CSS (body.presentation-mode) และ Fullscreen
// API จริง (ถ้าเบราว์เซอร์อนุญาต — บาง context เช่น iframe อาจปฏิเสธ ซึ่งไม่เป็นไร
// เพราะ CSS ส่วนลดความรกยังทำงานได้แม้ fullscreen ขอไม่ผ่าน) ปุ่ม refresh/รีเฟรช
// อัตโนมัติทุก 5s (ด้านล่าง) ยังทำงานต่อระหว่างโหมดพรีเซนต์ตามปกติ
function setPresentationMode(on) {
    document.body.classList.toggle('presentation-mode', on);
    if (on) {
        document.documentElement.requestFullscreen?.().catch(() => { /* ok, CSS-only mode still applies */ });
    } else if (document.fullscreenElement) {
        document.exitFullscreen?.().catch(() => {});
    }
}

document.getElementById('presentation-toggle-btn').addEventListener('click', () => setPresentationMode(true));
document.getElementById('presentation-exit-btn').addEventListener('click', () => setPresentationMode(false));
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('presentation-mode')) setPresentationMode(false);
});
// เผื่อผู้ใช้กด Esc ให้เบราว์เซอร์ออกจาก fullscreen เองโดยตรง (ไม่ผ่านปุ่ม/keydown
// ด้านบน) — sync CSS class ให้ตรงกับสถานะ fullscreen จริงเสมอ
document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) document.body.classList.remove('presentation-mode');
});

refresh();
setInterval(refresh, 5000);

for (const btn of document.querySelectorAll('.tab-btn')) {
    btn.addEventListener('click', () => {
        document.querySelector('.tab-btn.active').classList.remove('active');
        document.querySelector('.tab-panel:not(.hidden)').classList.add('hidden');
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.remove('hidden');
    });
}
