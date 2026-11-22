/* Crowd Monitor: Client Controller Logic (Clean, High-Performance) */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 0) => Number(v).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });

  let STATUS = { ok: false, levels: [], thresholds: {}, models: {} };

  // ------------------------------------------------------------ Cấu hình (lưu trên trình duyệt)
  const MAX_SIDES = [384, 512, 640, 768, 1024];   // 512 = mặc định chuẩn của CSRNet
  const DEFAULTS = {
    model: "csrnet",
    area: 100,
    crowd_type: "di chuyển",
    blur: true,
    max_side: 512,
    yolo_max_side: 1024,
    fps: 15,
    stride: 0.5,
    surge_ratio: 50,
    surge_window: 10,
    min_persist: 3
  };

  let S = { ...DEFAULTS };
  try {
    Object.assign(S, JSON.parse(localStorage.getItem("crowd-settings") || "{}"));
  } catch (_) { /* bỏ qua lỗi */ }

  const save = () => {
    try {
      localStorage.setItem("crowd-settings", JSON.stringify(S));
    } catch (_) { /* bỏ qua */ }
  };

  const modelSelect = $("#model-select");
  const resolutionKey = () => S.model === "yolo" ? "yolo_max_side" : "max_side";

  function syncResolution() {
    const value = Number(S[resolutionKey()]);
    $("#s-maxside").value = MAX_SIDES.indexOf(MAX_SIDES.includes(value) ? value : DEFAULTS[resolutionKey()]);
    $("#o-maxside").textContent = `${MAX_SIDES[Number($("#s-maxside").value)]}px`;
  }

  function syncModel() {
    const model = STATUS.models[S.model];
    $("#status-badge").classList.toggle("off", !model?.ok);
    $("#status-text").textContent = model?.ok ? `${model.label} Sẵn sàng` : "Chưa nạp mô hình";
    $("#meta").textContent = model?.ok ? `${model.device.toUpperCase()} · ${model.visualization}` : "Chưa kết nối";
    syncResolution();
  }

  modelSelect.addEventListener("change", () => {
    S.model = modelSelect.value;
    save();
    syncModel();
  });

  function bindSettings() {
    const area = $("#s-area"), blur = $("#s-blur");
    const ranges = [
      ["#s-fps", "fps", "#o-fps", (v) => `${v} FPS`],
      ["#s-stride", "stride", "#o-stride", (v) => v === 0 ? "liền tay" : `${fmt(v, 1)}s`],
      ["#s-surge", "surge_ratio", "#o-surge", (v) => `${v}%`],
      ["#s-window", "surge_window", "#o-window", (v) => `${v}s`],
      ["#s-persist", "min_persist", "#o-persist", (v) => `${v} lần`],
    ];

    area.value = S.area;
    area.addEventListener("change", () => {
      const v = Math.min(100000, Math.max(1, parseFloat(area.value) || DEFAULTS.area));
      area.value = S.area = v;
      save();
    });

    blur.checked = !!S.blur;
    blur.addEventListener("change", () => {
      S.blur = blur.checked;
      save();
    });

    for (const [sel, key, out, f] of ranges) {
      const el = $(sel);
      el.value = S[key];
      const show = () => { $(out).textContent = f(Number(el.value)); };
      el.addEventListener("input", () => { S[key] = Number(el.value); show(); save(); });
      show();
    }

    const ms = $("#s-maxside");
    ms.max = String(MAX_SIDES.length - 1);
    ms.addEventListener("input", () => {
      S[resolutionKey()] = MAX_SIDES[Number(ms.value)];
      syncResolution();
      save();
    });

    syncResolution();

    $$("#s-type button").forEach((b) => b.addEventListener("click", () => {
      S.crowd_type = b.dataset.type;
      save();
      syncType();
    }));

    syncType();
  }

  function syncType() {
    $$("#s-type button").forEach((b) => b.classList.toggle("active", b.dataset.type === S.crowd_type));
    const t = STATUS.thresholds[S.crowd_type];
    $("#s-thr").textContent = t ? `Ngưỡng quy định: Vàng ≥ ${t[0]} · Cam ≥ ${t[1]} · Đỏ ≥ ${t[2]} người/m²` : "";
  }

  function formData() {
    const fd = new FormData();
    const { yolo_max_side, ...params } = S;
    params.max_side = S.model === "yolo" ? yolo_max_side : S.max_side;
    for (const [k, v] of Object.entries(params)) {
      fd.append(k, typeof v === "boolean" ? (v ? "1" : "0") : v);
    }
    return fd;
  }

  // ------------------------------------------------------------ Điều hướng View
  const drawer = $("#settings");
  function go(view) {
    $$(".view-panel").forEach((v) => { v.hidden = v.dataset.view !== view; });
    $$(".nav-tab[data-go]").forEach((b) => {
      const active = b.dataset.go === view;
      b.classList.toggle("active", active);
      b.setAttribute("aria-pressed", String(active));
    });
    setSettings(false);
    if (view === "video" && rows.length) scheduleChartDraw();
    history.replaceState(null, "", `#${view}`);
  }

  function setSettings(open) {
    drawer.hidden = !open;
    if (open) $("#s-area").focus({ preventScroll: true });
  }

  $$("[data-go]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); go(b.dataset.go); }));
  $$("[data-open-settings]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); setSettings(drawer.hidden); }));
  $("#settings-close").addEventListener("click", () => setSettings(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setSettings(false); });

  // ------------------------------------------------------------ Toast
  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2500);
  }

  // ------------------------------------------------------------ Thẻ mức cảnh báo (Trực quan, rõ ràng)
  function levelCard(level, extra = "") {
    const L = STATUS.levels;
    const lv = L[level];
    if (!lv) return "";

    const scale = L.map((x, i) =>
      `<span class="alert-bar-segment" style="${i <= level ? `background-color:${L[i].color}` : ''}"></span>`
    ).join("");

    return `
      <div class="alert-banner" style="--lv-color:${lv.color}">
        <div class="alert-banner-top">
          <span class="alert-level-badge">MỨC ${esc(lv.name)}</span>
          <span class="alert-level-title">${esc(lv.label)}</span>
        </div>
        <div class="alert-action-msg">Hành động: <b>${esc(lv.action)}</b></div>
        ${extra ? `<div class="alert-surge-tag">⚠ ${esc(extra)}</div>` : ""}
        <div class="alert-bar-scale">${scale}</div>
      </div>
    `;
  }

  // ------------------------------------------------------------ Vùng kéo thả tệp
  function dropzone(zone, input, onFile) {
    input.addEventListener("change", () => {
      if (input.files[0]) onFile(input.files[0]);
      input.value = "";
    });
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); zone.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); zone.classList.remove("drag");
    }));
    zone.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files[0];
      if (f) onFile(f);
    });
  }

  async function readError(res) {
    try { return (await res.json()).error || `Lỗi ${res.status}`; } catch (_) { return `Lỗi ${res.status}`; }
  }

  // ------------------------------------------------------------ 01. Phân tích ảnh tĩnh
  let imgData = null, imageRequest = 0;
  async function analyseImage(file) {
    const requestId = ++imageRequest;
    const err = $("#img-error");
    err.hidden = true;
    $("#img-loading").hidden = false;
    const fd = formData();
    fd.append("file", file);

    try {
      const res = await fetch("/api/image", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await readError(res));
      const data = await res.json();
      if (requestId !== imageRequest) return;
      imgData = data;
      $("#img-empty").hidden = true;
      $("#img-result").hidden = false;
      $("#img-count").textContent = fmt(imgData.count);
      $("#img-density").textContent = fmt(imgData.density, 2);
      $("#img-level").innerHTML = levelCard(imgData.level);
      $("#img-time").textContent = `${imgData.model_label} · ${file.name} · ${fmt(imgData.ms)} ms`;
      $("#img-seg").hidden = !imgData.original;
      $("#img-seg [data-show=overlay]").textContent = imgData.visualization;
      showImg("overlay");
    } catch (e) {
      if (requestId !== imageRequest) return;
      err.textContent = e.message;
      err.hidden = false;
    } finally {
      if (requestId === imageRequest) $("#img-loading").hidden = true;
    }
  }

  function showImg(which) {
    if (!imgData) return;
    $("#img-out").src = imgData[which] || imgData.overlay;
    $("#img-caption").textContent = which === "original" ? "Ảnh chụp gốc" :
      `${imgData.visualization} · ${imgData.model_label}${imgData.original ? "" : " · Đã làm mờ bảo mật"}`;
    $$("#img-seg button").forEach((b) => b.classList.toggle("active", b.dataset.show === which));
  }
  $$("#img-seg button").forEach((b) => b.addEventListener("click", () => showImg(b.dataset.show)));
  dropzone($("#img-drop"), $("#img-input"), analyseImage);

  // ------------------------------------------------------------ 02. Video & Camera
  let source = "file", videoFile = null, job = null, rows = [], logRows = [], videoModel = "", videoSettings = null;

  $$("#src-seg button").forEach((b) => b.addEventListener("click", () => {
    if ($("#run-btn").disabled) return;
    source = b.dataset.src;
    $$("#src-seg button").forEach((x) => x.classList.toggle("active", x === b));
    $("#src-file").hidden = source !== "file";
    $("#src-camera").hidden = source !== "camera";
    $("#run-btn").querySelector("span").textContent = source === "file" ? "Bắt đầu phân tích video" : "Bắt đầu theo dõi camera";
  }));

  dropzone($("#vid-drop"), $("#vid-input"), (f) => {
    videoFile = f;
    $("#vid-name").textContent = f.name;
  });

  function setRunning(on, text = "") {
    $("#run-btn").disabled = on;
    $("#stop-btn").hidden = !on;
    $$("#src-seg button").forEach((button) => { button.disabled = on; });
    modelSelect.disabled = on || !STATUS.ok;
    $("#vid-loading").hidden = !on;
    $("#vid-loading-text").textContent = text;
  }

  function fail(msg) {
    const err = $("#vid-error");
    err.textContent = msg;
    err.hidden = false;
  }

  function formatTime(secs) {
    const s = Math.floor(secs);
    const ms = Math.floor((secs - s) * 10);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${ms}s`;
  }

  $("#run-btn").addEventListener("click", async () => {
    $("#vid-error").hidden = true;
    if (!STATUS.models[S.model]?.ok) return fail("Chưa nạp được mô hình AI.");

    const fd = formData();
    fd.append("source", source);
    if (source === "file") {
      if (!videoFile) return fail("Hãy chọn một tệp video trước.");
      fd.append("file", videoFile);
    } else {
      fd.append("url", $("#cam-url").value.trim() || "0");
      fd.append("max_secs", $("#cam-secs").value || 60);
    }

    rows = [];
    logRows = [];
    videoModel = STATUS.models[S.model].label;
    videoSettings = { crowd_type: S.crowd_type, area: S.area };

    $("#live-count").textContent = "–";
    $("#live-density").textContent = "–";
    $("#live-level").innerHTML = `<div class="level-placeholder">Đang khởi tạo luồng phân tích...</div>`;

    for (const selector of ["#live-raw-frame", "#live-frame"]) {
      const image = $(selector);
      image.hidden = true;
      image.removeAttribute("src");
    }

    $("#live-source-empty").hidden = false;
    $("#live-ai-empty").hidden = false;
    $("#live-source-caption").textContent = "00:00:00";
    $("#live-caption").textContent = "Chờ tín hiệu...";
    $("#live-stats").hidden = true;
    $("#live-chart").hidden = true;
    $("#live-progress").hidden = true;
    $("#live-progress .progress-fill").style.width = "0%";
    $("#summary").textContent = "";
    $("#log-wrap").hidden = true;

    setRunning(true, source === "file" ? "Đang nạp video lên máy chủ..." : "Đang mở kết nối camera...");

    let id;
    try {
      const res = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await readError(res));
      id = (await res.json()).id;
    } catch (e) {
      setRunning(false);
      return fail(e.message);
    }

    $("#vid-loading-text").textContent = "Đang xử lý luồng nhận diện...";
    const es = new EventSource(`/api/jobs/${id}/events`);
    job = { id, es, t0: performance.now(), frames: 0, info: null };

    es.addEventListener("start", (e) => {
      job.info = JSON.parse(e.data);
      const i = job.info;
      $("#live-caption").textContent = `${i.model_label} · ${i.visualization}`;
      $("#live-stats").textContent = `Mục tiêu ${fmt(i.target_fps)} FPS · Độ phân giải ${i.max_side}px · Thiết bị: ${i.backend}`;
      $("#live-stats").hidden = false;
    });

    es.addEventListener("tick", (e) => onTick(JSON.parse(e.data)));
    es.addEventListener("done", (e) => finish(JSON.parse(e.data)));
    es.addEventListener("fail", (e) => { fail(JSON.parse(e.data).error); finish(null); });
    es.onerror = () => { if (job && job.es === es) { fail("Mất kết nối tới máy chủ."); finish(null); } };
  });

  $("#stop-btn").addEventListener("click", () => {
    if (!job) return;
    fetch(`/api/jobs/${job.id}/stop`, { method: "POST" });
    $("#vid-loading-text").textContent = "Đang dừng...";
  });

  function onTick(d) {
    const L = STATUS.levels;

    // Cập nhật Khung 1: Camera nguồn trực tiếp
    if (d.raw_frame) {
      const imgRaw = $("#live-raw-frame");
      imgRaw.src = d.raw_frame;
      imgRaw.hidden = false;
      $("#live-source-empty").hidden = true;
      $("#live-source-caption").textContent = formatTime(d.source_t);
      job.frames += 1;
    }

    // Cập nhật Khung 2: Camera nhận diện AI
    if (d.frame) {
      const imgAi = $("#live-frame");
      imgAi.src = d.frame;
      imgAi.hidden = false;
      $("#live-ai-empty").hidden = true;
      $("#live-caption").textContent = `${videoModel} · ${job.info?.visualization ?? "Nhận diện"} · ${fmt(d.t, 1)}s`;
    }

    // Cập nhật số liệu KPI & Mức cảnh báo (TO, NỔI BẬT)
    if (d.fresh && Number.isFinite(d.t) && Number.isFinite(d.count) && Number.isFinite(d.smoothed)) {
      const { t, count, smoothed, density, level, surge } = d;
      rows.push({ t, count, smoothed, density, level, surge });
      for (const ev of d.events) logRows.push({ t, level, text: ev });

      $("#live-count").textContent = fmt(Math.round(d.count));
      $("#live-density").textContent = fmt(d.density, 2);
      $("#live-level").innerHTML = levelCard(d.level, d.surge ? "CẢNH BÁO: TĂNG ĐỘT BIẾN!" : "");
      $("#live-chart").hidden = false;
      scheduleChartDraw();
    }

    // Tiến trình
    if (d.progress != null) {
      $("#live-progress").hidden = false;
      $("#live-progress .progress-fill").style.width = `${(d.progress * 100).toFixed(1)}%`;
    }

    // Thống kê hiệu năng OSD nhỏ bên dưới
    const el = performance.now() - job.t0;
    const shownFps = el > 500 ? (job.frames * 1000) / el : 0;
    $("#live-stats").textContent =
      `Camera ${fmt(shownFps, 1)} FPS` +
      (d.infer_ms ? ` · Suy luận AI: ${fmt(d.infer_ms)} ms (${fmt(d.infer_fps, 1)} FPS)` : "") +
      (d.max_side ? ` · Độ phân giải: ${d.max_side}px` : "");

    if (d.events.length && L[d.level]) {
      toast(`${d.events[d.events.length - 1]}`);
    }
  }

  function finish(summary) {
    if (!job) return;
    job.es.close();
    job = null;
    setRunning(false);

    if (!rows.length && !summary) return;
    $("#log-wrap").hidden = false;
    const L = STATUS.levels;

    $("#log-body").innerHTML = logRows.map((r) => `
      <tr>
        <td style="font-family:var(--font-mono); font-weight:600">${fmt(r.t, 1)}s</td>
        <td><span style="background-color:${L[r.level].color}; color:#fff; font-size:10px; font-weight:800; padding:2px 8px; border-radius:4px">MỨC ${esc(L[r.level].name)}</span></td>
        <td>${esc(r.text)}</td>
      </tr>
    `).join("");

    $("#log-empty").hidden = logRows.length > 0;
    $(".data-table").hidden = logRows.length === 0;

    if (summary) {
      const fps = summary.display_fps ? ` · ${fmt(summary.display_fps, 1)} FPS phát` : "";
      const inf = summary.infer_ms ? ` · Suy luận ${fmt(summary.infer_ms)} ms/lần` : "";
      $("#summary").textContent = `Tổng kết: ${summary.model_label} · ${summary.samples} lần suy luận · Đỉnh điểm: ${fmt(summary.max_count)} người · Mức cao nhất: ${L[summary.max_level].name}${fps}${inf}${summary.stopped ? " (Đã dừng)" : ""}`;
    }

    $("#csv-btn").disabled = rows.length === 0;
  }

  $("#csv-btn").addEventListener("click", () => {
    const L = STATUS.levels;
    const head = ["Mo hinh", "Thoi gian (s)", "So nguoi", "So nguoi (lam tron)", "Mat do (/m2)", "Muc canh bao", "Tang dot bien"];
    const lines = [head.join(",")].concat(rows.map((r) => [
      videoModel,
      r.t.toFixed(1),
      r.count.toFixed(1),
      r.smoothed.toFixed(1),
      r.density.toFixed(3),
      L[r.level].name,
      r.surge ? "True" : "False"
    ].join(",")));

    const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `crowd_monitor_${Date.now()}.csv` });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ------------------------------------------------------------ Biểu đồ thời gian (Canvas)
  const chart = $("#chart");
  let chartFrame = 0;
  function scheduleChartDraw() {
    if (chartFrame) return;
    chartFrame = requestAnimationFrame(() => {
      chartFrame = 0;
      drawChart();
    });
  }

  function chartTimeLabel(seconds, duration) {
    if (duration < 60) return `${fmt(seconds, duration < 10 ? 1 : 0)} s`;
    const wholeSeconds = Math.round(seconds);
    const minutes = Math.floor(wholeSeconds / 60);
    return `${minutes}:${String(wholeSeconds % 60).padStart(2, "0")}`;
  }

  function drawChart() {
    const ctx = chart.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = chart.clientWidth, H = chart.clientHeight;
    if (!ctx || W < 90 || H < 70 || !rows.length) return;

    const pixelWidth = Math.round(W * dpr), pixelHeight = Math.round(H * dpr);
    if (chart.width !== pixelWidth || chart.height !== pixelHeight) {
      chart.width = pixelWidth;
      chart.height = pixelHeight;
    }
    ctx.setTransform(pixelWidth / W, 0, 0, pixelHeight / H, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const last = rows[rows.length - 1];
    const tMax = Math.max(last.t, 1);
    const thr = (STATUS.thresholds[videoSettings?.crowd_type] || []).map((x) => x * videoSettings.area);
    const peak = rows.reduce((max, row) => Math.max(max, row.count, row.smoothed), 0);
    let yMax = Math.max(peak * 1.15, 5);
    const visibleThr = thr.filter((v) => v <= yMax * 1.6);
    if (visibleThr.length) yMax = Math.max(yMax, Math.max(...visibleThr) * 1.08);
    yMax = Math.ceil(yMax / 4) * 4;

    ctx.font = "11px Plus Jakarta Sans, sans-serif";
    const pad = { l: Math.max(36, Math.ceil(ctx.measureText(fmt(yMax)).width) + 12), r: 12, t: 10, b: 23 };
    const plotWidth = W - pad.l - pad.r;
    if (plotWidth < 24) return;
    const x = (t) => pad.l + (t / tMax) * plotWidth;
    const y = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
    ctx.fillStyle = "rgba(56,35,20,0.6)";

    // Đường lưới ngang
    ctx.strokeStyle = "rgba(56,35,20,0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const v = (yMax * i) / 4;
      const yy = Math.round(y(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(W - pad.r, yy);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(fmt(v), pad.l - 6, yy + 4);
    }

    const xSteps = Math.min(6, Math.max(2, Math.floor(plotWidth / 95)));
    for (let i = 0; i <= xSteps; i++) {
      const t = (tMax * i) / xSteps;
      ctx.textAlign = i === 0 ? "left" : i === xSteps ? "right" : "center";
      ctx.fillText(chartTimeLabel(t, tMax), x(t), H - 6);
    }

    // Đường ngưỡng
    ctx.setLineDash([4, 4]);
    let lastLabelY = Infinity;
    thr.forEach((v, i) => {
      const level = STATUS.levels[i + 1];
      if (!level || !Number.isFinite(v) || v > yMax) return;
      const c = level.color;
      const yy = Math.round(y(v)) + 0.5;
      ctx.strokeStyle = c;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(W - pad.r, yy);
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (lastLabelY - yy >= 14) {
        ctx.fillStyle = c;
        ctx.textAlign = "right";
        ctx.fillText(level.name, W - pad.r - 2, yy - 4);
        lastLabelY = yy;
      }
    });
    ctx.setLineDash([]);

    // Vùng & đường số liệu
    const line = (key) => {
      ctx.beginPath();
      rows.forEach((r, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x(r.t), y(r[key])));
    };

    if (rows.length > 1) {
      line("smoothed");
      ctx.lineTo(x(last.t), y(0));
      ctx.lineTo(x(rows[0].t), y(0));
      ctx.closePath();
      ctx.fillStyle = "rgba(56, 35, 20, 0.06)";
      ctx.fill();

      line("count");
      ctx.strokeStyle = "rgba(56, 35, 20, 0.55)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      line("smoothed");
      ctx.strokeStyle = "#382314";
      ctx.lineWidth = 2.2;
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(x(last.t), y(last.smoothed), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = STATUS.levels[last.level]?.color || "#382314";
    ctx.fill();
  }

  new ResizeObserver(() => {
    if (rows.length) scheduleChartDraw();
  }).observe(chart);
  window.addEventListener("resize", () => {
    if (rows.length) scheduleChartDraw();
  });

  // ------------------------------------------------------------ Khởi tạo hệ thống
  async function init() {
    try {
      STATUS = await (await fetch("/api/status")).json();
    } catch (_) { /* Máy chủ có thể chưa sẵn sàng */ }

    for (const option of modelSelect.options) {
      const model = STATUS.models[option.value];
      option.disabled = !model?.ok;
      if (model && !model.ok) option.textContent = `${model.label} (chưa nạp)`;
    }

    if (!STATUS.models[S.model]?.ok) S.model = STATUS.default_model || DEFAULTS.model;
    modelSelect.value = S.model;
    modelSelect.disabled = !STATUS.ok;

    if (!STATUS.ok) {
      const n = $("#model-missing");
      n.textContent = Object.values(STATUS.models).map((model) => model.error).filter(Boolean).join(" ") ||
        "Chưa kết nối được với máy chủ xử lý AI.";
      n.hidden = false;
    }

    bindSettings();
    syncModel();

    const start = location.hash.slice(1);
    go(start === "image" ? "image" : "video");
  }

  init();
})();
