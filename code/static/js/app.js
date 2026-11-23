/* Crowd Monitor: Client Controller Logic (Clean, High-Performance) */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 0) => Number(v).toLocaleString("vi-VN", { minimumFractionDigits: d, maximumFractionDigits: d });
  const firstNum = (...vals) => {
    for (const v of vals) {
      if (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) return Number(v);
    }
    return null;
  };

  let STATUS = { ok: false, levels: [], thresholds: {}, models: {}, database: { ok: false } };

  const ALERT_TYPE_LABELS = {
    density: "Quá tải mật độ",
    overload: "Quá tải mật độ",
    surge: "Tăng đột biến",
    motion_panic: "Chạy tán loạn",
    panic: "Chạy tán loạn",
    counter_flow: "Ngược chiều",
    roi_intrusion: "Vi phạm vùng cấm",
    intrusion: "Vi phạm vùng cấm"
  };

  // ------------------------------------------------------------ Cấu hình (lưu trên trình duyệt)
  const MAX_SIDES = [384, 512, 640, 768, 1024];   // 512 = mặc định chuẩn của CSRNet
  const DEFAULTS = {
    model: "csrnet",
    area: 100,
    crowd_type: "di chuyển",
    blur: true,
    max_side: 512,
    yolo_max_side: 768,
    fps: 15,
    stride: 0,
    surge_ratio: 50,
    surge_window: 10,
    min_persist: 3,
    roi_mode: "full",
    roi_points: [],
    roi_area: 50,
    restricted_thr: 5,
    enable_flow: true,
    show_vectors: true,
    speed_thr: 14,
    entropy_thr: 0.72,
    counter_thr: 30
  };

  let S = { ...DEFAULTS };
  try {
    Object.assign(S, JSON.parse(localStorage.getItem("crowd-settings") || "{}"));
  } catch (_) { /* bỏ qua lỗi */ }
  if (!S._realtime_v2) {
    if (S.stride === 0.5) S.stride = 0;
    if (S.yolo_max_side === 1024) S.yolo_max_side = 768;
    S._realtime_v2 = 1;
    try { localStorage.setItem("crowd-settings", JSON.stringify(S)); } catch (_) { /* bỏ qua */ }
  }

  if (!["full", "roi", "forbidden"].includes(S.roi_mode)) S.roi_mode = "full";
  if (!Array.isArray(S.roi_points)) {
    S.roi_points = [];
  } else {
    S.roi_points = S.roi_points
      .filter((p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(Number(p[0])) && Number.isFinite(Number(p[1])))
      .map((p) => [Math.min(1, Math.max(0, Number(p[0]))), Math.min(1, Math.max(0, Number(p[1])))]);
  }
  S.roi_area = Math.min(100000, Math.max(1, Number(S.roi_area) || DEFAULTS.roi_area));
  S.restricted_thr = Math.min(500, Math.max(1, Math.round(Number(S.restricted_thr) || DEFAULTS.restricted_thr)));
  S.enable_flow = S.enable_flow !== false;
  S.show_vectors = S.show_vectors !== false;
  S.speed_thr = Math.min(40, Math.max(4, Number(S.speed_thr) || DEFAULTS.speed_thr));
  S.entropy_thr = Math.min(0.95, Math.max(0.40, Number(S.entropy_thr) || DEFAULTS.entropy_thr));
  S.counter_thr = Math.min(70, Math.max(15, Number(S.counter_thr) || DEFAULTS.counter_thr));

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
    const model = STATUS.models?.[S.model];
    $("#status-badge").classList.toggle("off", !model?.ok);
    $("#status-text").textContent = model?.ok ? `${model.label} Sẵn sàng` : "Chưa nạp mô hình";
    $("#meta").textContent = model?.ok ? `${model.device.toUpperCase()} · ${model.visualization}` : "Chưa kết nối";
    syncResolution();
  }

  function syncDatabase() {
    const db = STATUS.database;
    const ok = !!db?.ok;
    const badge = $("#db-badge");
    if (badge) {
      badge.textContent = ok ? "PostgreSQL · Sẵn sàng" : "PG · Ngoại tuyến";
      badge.classList.toggle("off", !ok);
      badge.title = db?.error ? `PostgreSQL: ${db.error}` : (ok ? "CSDL PostgreSQL đã kết nối sẵn sàng" : "Trạng thái CSDL PostgreSQL: Ngoại tuyến");
    }
    const histStatus = $("#hist-db-status");
    if (histStatus) {
      histStatus.textContent = ok ? "PostgreSQL · Đang hoạt động" : (db?.error ? `Ngoại tuyến · ${db.error}` : "PG · Ngoại tuyến");
      histStatus.classList.toggle("off", !ok);
    }
  }

  modelSelect.addEventListener("change", () => {
    S.model = modelSelect.value;
    save();
    syncModel();
    if (currentView === "image" && lastImageFile) analyseImage(lastImageFile);
  });

  function bindSettings() {
    const area = $("#s-area"), blur = $("#s-blur"), flow = $("#s-flow"), vectors = $("#s-vectors");
    const ranges = [
      ["#s-fps", "fps", "#o-fps", (v) => `${v} FPS`],
      ["#s-stride", "stride", "#o-stride", (v) => v === 0 ? "liền tay" : `${fmt(v, 1)}s`],
      ["#s-surge", "surge_ratio", "#o-surge", (v) => `${v}%`],
      ["#s-window", "surge_window", "#o-window", (v) => `${v}s`],
      ["#s-persist", "min_persist", "#o-persist", (v) => `${v} lần`],
      ["#s-speed", "speed_thr", "#o-speed", (v) => `${v} %/s`],
      ["#s-entropy", "entropy_thr", "#o-entropy", (v) => fmt(v, 2)],
      ["#s-counter", "counter_thr", "#o-counter", (v) => `${v}%`],
    ];

    area.value = S.area;
    area.addEventListener("change", () => {
      const v = Math.min(100000, Math.max(1, parseFloat(area.value) || DEFAULTS.area));
      area.value = S.area = v;
      save();
      syncAreaFootnotes();
      if (currentView === "image" && lastImageFile && S.roi_mode !== "roi") analyseImage(lastImageFile);
    });

    blur.checked = !!S.blur;
    blur.addEventListener("change", () => {
      S.blur = blur.checked;
      save();
      if (currentView === "image" && lastImageFile) analyseImage(lastImageFile);
    });

    flow.checked = !!S.enable_flow;
    vectors.checked = !!S.show_vectors;
    vectors.disabled = !flow.checked;

    flow.addEventListener("change", () => {
      S.enable_flow = flow.checked;
      vectors.disabled = !flow.checked;
      save();
      if (!job) resetMotionBar();
    });

    vectors.addEventListener("change", () => {
      S.show_vectors = vectors.checked;
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
      if (currentView === "image" && lastImageFile) analyseImage(lastImageFile);
    }));

    syncType();
  }

  function syncType() {
    $$("#s-type button").forEach((b) => b.classList.toggle("active", b.dataset.type === S.crowd_type));
    const t = STATUS.thresholds?.[S.crowd_type];
    $("#s-thr").textContent = t ? `Ngưỡng quy định: Vàng ≥ ${t[0]} · Cam ≥ ${t[1]} · Đỏ ≥ ${t[2]} người/m²` : "";
  }

  function hasValidRoi() {
    return S.roi_mode !== "full" && Array.isArray(S.roi_points) && S.roi_points.length >= 3;
  }

  function effectiveArea() {
    return (S.roi_mode === "roi" && S.roi_points.length >= 3) ? S.roi_area : S.area;
  }

  function syncAreaFootnotes() {
    const foot = $("#live-density-foot");
    const countFoot = $("#live-count-foot");
    if (!foot || !countFoot) return;
    if (S.roi_mode === "roi" && S.roi_points.length >= 3) {
      foot.textContent = `Tính trong vùng ROI (${fmt(S.roi_area)} m²)`;
      countFoot.textContent = "Đếm trong đa giác ROI đã chọn";
    } else if (S.roi_mode === "forbidden" && S.roi_points.length >= 3) {
      foot.textContent = `Toàn khung (${fmt(S.area)} m²) · Giám sát vùng cấm`;
      countFoot.textContent = `Cảnh báo nếu vùng cấm ≥ ${fmt(S.restricted_thr)} người`;
    } else {
      foot.textContent = `Dựa trên diện tích cấu hình (${fmt(S.area)} m²)`;
      countFoot.textContent = "Đếm theo thời gian thực";
    }
  }

  function formData() {
    const fd = new FormData();
    const maxSide = S.model === "yolo" ? S.yolo_max_side : S.max_side;
    const validPoly = hasValidRoi();
    fd.append("model", S.model);
    fd.append("area", S.area);
    fd.append("crowd_type", S.crowd_type);
    fd.append("blur", S.blur ? "1" : "0");
    fd.append("max_side", maxSide);
    fd.append("fps", S.fps);
    fd.append("stride", S.stride);
    fd.append("surge_ratio", S.surge_ratio);
    fd.append("surge_window", S.surge_window);
    fd.append("min_persist", S.min_persist);
    fd.append("roi_mode", validPoly ? S.roi_mode : "full");
    fd.append("roi_points", JSON.stringify(validPoly ? S.roi_points : []));
    fd.append("roi_area", S.roi_area);
    fd.append("restricted_thr", S.restricted_thr);
    fd.append("enable_flow", S.enable_flow ? "1" : "0");
    fd.append("show_vectors", S.show_vectors ? "1" : "0");
    fd.append("speed_thr", S.speed_thr);
    fd.append("entropy_thr", S.entropy_thr);
    fd.append("counter_thr", S.counter_thr);
    return fd;
  }

  // ------------------------------------------------------------ Điều hướng View & Lightbox
  const drawer = $("#settings");
  let currentView = "video";

  function go(view) {
    if (!["image", "video", "history"].includes(view)) view = "video";
    currentView = view;
    $$(".view-panel").forEach((v) => { v.hidden = v.dataset.view !== view; });
    $$(".nav-tab[data-go]").forEach((b) => {
      const active = b.dataset.go === view;
      b.classList.toggle("active", active);
      b.setAttribute("aria-pressed", String(active));
    });
    setSettings(false);
    if (view === "video") {
      if (rows.length) scheduleChartDraw();
      scheduleRoiDraw();
    } else if (view === "image") {
      scheduleRoiDraw();
    } else if (view === "history") {
      loadHistory(true);
    }
    history.replaceState(null, "", `#${view}`);
  }

  function setSettings(open) {
    drawer.hidden = !open;
    if (open) $("#s-area").focus({ preventScroll: true });
  }

  $$("[data-go]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); go(b.dataset.go); }));
  $$("[data-open-settings]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); setSettings(drawer.hidden); }));
  $("#settings-close").addEventListener("click", () => setSettings(false));
  $("#open-history-btn").addEventListener("click", () => go("history"));

  // Lightbox điều khiển xem ảnh bằng chứng
  const lightbox = $("#lightbox");
  let lightboxStore = new Map();
  let lightboxSeq = 0;

  function registerEvidence(meta) {
    const key = String(++lightboxSeq);
    lightboxStore.set(key, meta);
    if (lightboxStore.size > 600) {
      const oldest = lightboxStore.keys().next().value;
      lightboxStore.delete(oldest);
    }
    return key;
  }

  function openLightbox(meta) {
    if (!meta || !meta.url) return;
    $("#lightbox-img").src = meta.url;
    $("#lightbox-title").textContent = meta.title || "Ảnh bằng chứng sự kiện an ninh";

    const lv = STATUS.levels?.[meta.level];
    const badge = $("#lightbox-badge");
    if (lv) {
      badge.textContent = `MỨC ${lv.name}`;
      badge.style.backgroundColor = lv.color;
    } else {
      badge.textContent = meta.typeLabel || "BẰNG CHỨNG";
      badge.style.backgroundColor = "var(--color-alert-red)";
    }

    const chips = [];
    if (meta.timeText) chips.push(`<span><b>Thời điểm:</b> ${esc(meta.timeText)}</span>`);
    if (meta.source) chips.push(`<span><b>Nguồn:</b> ${esc(meta.source)}</span>`);
    if (meta.model) chips.push(`<span><b>Mô hình:</b> ${esc(meta.model)}</span>`);
    if (meta.count != null && Number.isFinite(Number(meta.count))) {
      chips.push(`<span><b>Số người:</b> ${fmt(Math.round(Number(meta.count)))} người</span>`);
    }
    if (meta.density != null && Number.isFinite(Number(meta.density))) {
      chips.push(`<span><b>Mật độ:</b> ${fmt(Number(meta.density), 2)} người/m²</span>`);
    }
    $("#lightbox-meta").innerHTML = chips.join("");

    const dl = $("#lightbox-download");
    dl.href = meta.url;
    const cleanName = meta.url.split("/").pop().split("?")[0] || `evidence_${Date.now()}.jpg`;
    dl.setAttribute("download", cleanName);

    lightbox.hidden = false;
  }

  function closeLightbox() {
    if (lightbox.hidden) return;
    lightbox.hidden = true;
    $("#lightbox-img").removeAttribute("src");
  }

  $("#lightbox-close").addEventListener("click", closeLightbox);
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
  });

  document.addEventListener("click", (e) => {
    const trigger = e.target.closest("[data-evidence-key]");
    if (!trigger) return;
    e.preventDefault();
    const meta = lightboxStore.get(trigger.dataset.evidenceKey);
    if (meta) openLightbox(meta);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!lightbox.hidden) {
        closeLightbox();
        return;
      }
      if (roiDrawing) {
        roiDrawing = false;
        if (S.roi_points.length < 3) S.roi_points = [];
        roiClosed = S.roi_points.length >= 3;
        hoverState = null;
        save();
        syncRoiUI();
        return;
      }
      setSettings(false);
    }
  });

  // ------------------------------------------------------------ Toast
  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2800);
  }

  // ------------------------------------------------------------ Thẻ mức cảnh báo (Đa nhãn bất thường)
  function levelCard(level, badges = []) {
    const L = STATUS.levels || [];
    const lv = L[level] || L[0];
    if (!lv) return "";

    const list = Array.isArray(badges)
      ? badges.filter(Boolean)
      : (badges ? [{ text: String(badges), kind: "danger" }] : []);

    const scale = L.map((x, i) =>
      `<span class="alert-bar-segment" style="${i <= level ? `background-color:${L[i].color}` : ""}"></span>`
    ).join("");

    const badgesHtml = list.length ? `
      <div class="alert-badges-wrap">
        ${list.map((b) => {
          const obj = typeof b === "string" ? { text: b, kind: "danger" } : b;
          const cls = obj.kind === "eta" ? "alert-surge-tag eta-tag" : (obj.kind === "warn" ? "alert-surge-tag warn-tag" : "alert-surge-tag");
          const icon = obj.kind === "eta" ? "⏱" : "⚠";
          return `<span class="${cls}">${icon} ${esc(obj.text)}</span>`;
        }).join("")}
      </div>
    ` : "";

    return `
      <div class="alert-banner" style="--lv-color:${lv.color}">
        <div class="alert-banner-top">
          <span class="alert-level-badge">MỨC ${esc(lv.name)}</span>
          <span class="alert-level-title">${esc(lv.label)}</span>
        </div>
        <div class="alert-action-msg">Hành động: <b>${esc(lv.action)}</b></div>
        ${badgesHtml}
        <div class="alert-bar-scale">${scale}</div>
      </div>
    `;
  }

  function buildAlertBadges(d) {
    const badges = [];
    const m = d.motion || d.flow || {};
    if (d.surge) badges.push({ text: "TĂNG ĐỘT BIẾN MẬT ĐỘ", kind: "danger" });
    if (d.panic || d.motion_panic || m.panic) badges.push({ text: "CHẠY TÁN LOẠN / HỖN LOẠN", kind: "danger" });
    if (d.counter_flow || m.counter_flow) badges.push({ text: "DI CHUYỂN NGƯỢC CHIỀU", kind: "warn" });
    if (d.roi_intrusion || d.intrusion) {
      const rc = d.restricted_count ?? d.roi_count ?? Math.round(d.count || 0);
      badges.push({ text: `XÂM NHẬP VÙNG CẤM (${fmt(rc)} người)`, kind: "danger" });
    }
    const eta = d.overload_eta_s ?? d.eta_overload_s;
    if (eta != null && Number.isFinite(Number(eta)) && Number(eta) > 0) {
      badges.push({ text: `DỰ BÁO CHẠM MỨC ĐỎ SAU ~${fmt(eta, 0)}s`, kind: "eta" });
    }
    return badges;
  }

  // ------------------------------------------------------------ Interactive Polygon ROI Canvas (Video + Image)
  const vidCanvas = $("#vid-roi-canvas");
  const vidImg = $("#live-raw-frame");
  const imgCanvas = $("#img-roi-canvas");
  const imgOut = $("#img-out");

  let roiDrawing = false;
  let roiClosed = S.roi_points.length >= 3;
  let hoverState = null; // { canvas, nx, ny, nearFirst }
  let roiAnimFrame = 0;
  let lastImageFile = null;

  function getBitmapRect(canvas, img) {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (cw <= 0 || ch <= 0) return { rx: 0, ry: 0, rw: 0, rh: 0 };
    if (!img || img.hidden || !img.naturalWidth || !img.naturalHeight) {
      return { rx: 0, ry: 0, rw: cw, rh: ch };
    }
    const fit = getComputedStyle(img).objectFit;
    if (fit === "contain") {
      const iw = img.naturalWidth, ih = img.naturalHeight;
      const scale = Math.min(cw / iw, ch / ih);
      const rw = iw * scale, rh = ih * scale;
      return { rx: (cw - rw) / 2, ry: (ch - rh) / 2, rw, rh };
    }
    return { rx: 0, ry: 0, rw: cw, rh: ch };
  }

  function eventToRoiPoint(e, canvas, img) {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { rx, ry, rw, rh } = getBitmapRect(canvas, img);
    if (rw <= 2 || rh <= 2) return null;
    const nx = Math.min(1, Math.max(0, (cx - rx) / rw));
    const ny = Math.min(1, Math.max(0, (cy - ry) / rh));
    return {
      nx: Number(nx.toFixed(4)),
      ny: Number(ny.toFixed(4)),
      px: rx + nx * rw,
      py: ry + ny * rh,
      rx, ry, rw, rh
    };
  }

  function maybeReanalyseImage() {
    if (currentView === "image" && lastImageFile && !roiDrawing) {
      analyseImage(lastImageFile);
    }
  }

  function finishRoiPolygon() {
    if (S.roi_points.length < 3) {
      toast("Cần ít nhất 3 đỉnh để khép kín vùng đa giác.");
      return;
    }
    roiDrawing = false;
    roiClosed = true;
    hoverState = null;
    save();
    syncRoiUI();
    toast(`Đã chốt vùng ${S.roi_mode === "forbidden" ? "cấm tụ tập" : "ROI"} (${S.roi_points.length} đỉnh).`);
    maybeReanalyseImage();
  }

  function syncRoiUI() {
    const isFull = S.roi_mode === "full";
    const isRoi = S.roi_mode === "roi";
    const isForbidden = S.roi_mode === "forbidden";
    const n = S.roi_points.length;

    for (const prefix of ["vid", "img"]) {
      $$(`#${prefix}-roi-mode button`).forEach((b) => {
        b.classList.toggle("active", b.dataset.roi === S.roi_mode);
      });
      $(`#${prefix}-roi-actions`).hidden = isFull;
      $(`#${prefix}-roi-area-wrap`).hidden = !isRoi;
      $(`#${prefix}-restricted-wrap`).hidden = !isForbidden;
      $(`#${prefix}-roi-area`).value = S.roi_area;
      $(`#${prefix}-restricted-thr`).value = S.restricted_thr;

      const drawBtn = $(`#${prefix}-roi-draw`);
      drawBtn.classList.toggle("active", roiDrawing);
      if (roiDrawing) {
        drawBtn.textContent = n >= 3 ? "✓ Chốt vùng" : "✕ Hủy vẽ";
      } else {
        drawBtn.textContent = n >= 3 ? "✏ Vẽ lại vùng" : "✏ Vẽ đa giác";
      }

      $(`#${prefix}-roi-undo`).disabled = n === 0;
      $(`#${prefix}-roi-clear`).disabled = n === 0 && !roiDrawing;

      const info = $(`#${prefix}-roi-info`);
      info.classList.remove("ready", "forbidden", "drawing");
      if (roiDrawing) {
        info.textContent = n >= 3 ? `Đang vẽ: ${n} đỉnh (bấm Chốt vùng)` : `Đang vẽ: ${n}/3 đỉnh tối thiểu`;
        info.classList.add("drawing");
      } else if (n >= 3 && roiClosed) {
        info.textContent = `Đã chốt: ${n} đỉnh`;
        info.classList.add(isForbidden ? "forbidden" : "ready");
      } else {
        info.textContent = "Chưa vẽ vùng";
      }
    }

    const canDraw = roiDrawing && !isFull;
    vidCanvas.classList.toggle("drawing", canDraw);
    imgCanvas.classList.toggle("drawing", canDraw);
    syncAreaFootnotes();
    scheduleRoiDraw();
  }

  function bindRoiToolbar(prefix) {
    $$(`#${prefix}-roi-mode button`).forEach((btn) => {
      btn.addEventListener("click", () => {
        const nextMode = btn.dataset.roi;
        if (nextMode === S.roi_mode) return;
        S.roi_mode = nextMode;
        if (nextMode === "full") {
          roiDrawing = false;
          hoverState = null;
        } else if (S.roi_points.length < 3) {
          roiClosed = false;
        }
        save();
        syncRoiUI();
        if (nextMode === "full" || (S.roi_points.length >= 3 && roiClosed)) {
          maybeReanalyseImage();
        }
      });
    });

    $(`#${prefix}-roi-draw`).addEventListener("click", () => {
      if (S.roi_mode === "full") return;
      if (prefix === "img" && !imgData && !lastImageFile) {
        toast("Hãy tải lên một bức ảnh trước để vẽ vùng đa giác trực tiếp trên ảnh.");
        return;
      }
      if (roiDrawing) {
        if (S.roi_points.length >= 3) {
          finishRoiPolygon();
        } else {
          roiDrawing = false;
          S.roi_points = [];
          roiClosed = false;
          hoverState = null;
          save();
          syncRoiUI();
        }
      } else {
        S.roi_points = [];
        roiClosed = false;
        roiDrawing = true;
        hoverState = null;
        syncRoiUI();
        toast(prefix === "vid"
          ? "Bấm chuột lên Khung Nguồn (CH-01) để chấm các đỉnh đa giác."
          : "Bấm chuột trực tiếp lên ảnh kết quả để chấm các đỉnh đa giác.");
      }
    });

    $(`#${prefix}-roi-undo`).addEventListener("click", () => {
      if (!S.roi_points.length) return;
      S.roi_points.pop();
      if (S.roi_points.length < 3) {
        roiClosed = false;
        if (S.roi_points.length > 0) roiDrawing = true;
      }
      save();
      syncRoiUI();
      if (!roiDrawing && roiClosed && S.roi_points.length >= 3) {
        maybeReanalyseImage();
      }
    });

    $(`#${prefix}-roi-clear`).addEventListener("click", () => {
      const hadValid = S.roi_points.length >= 3;
      S.roi_points = [];
      roiClosed = false;
      roiDrawing = false;
      hoverState = null;
      save();
      syncRoiUI();
      if (hadValid) maybeReanalyseImage();
    });

    const areaInput = $(`#${prefix}-roi-area`);
    areaInput.addEventListener("change", () => {
      const v = Math.min(100000, Math.max(1, parseFloat(areaInput.value) || DEFAULTS.roi_area));
      S.roi_area = v;
      save();
      syncRoiUI();
      if (S.roi_mode === "roi" && S.roi_points.length >= 3) maybeReanalyseImage();
    });

    const thrInput = $(`#${prefix}-restricted-thr`);
    thrInput.addEventListener("change", () => {
      const v = Math.min(500, Math.max(1, Math.round(parseFloat(thrInput.value) || DEFAULTS.restricted_thr)));
      S.restricted_thr = v;
      save();
      syncRoiUI();
      if (S.roi_mode === "forbidden" && S.roi_points.length >= 3) maybeReanalyseImage();
    });
  }

  function bindRoiCanvas(canvas, img) {
    canvas.addEventListener("mousemove", (e) => {
      if (!roiDrawing || S.roi_mode === "full") return;
      const pt = eventToRoiPoint(e, canvas, img);
      if (!pt) return;
      let nearFirst = false;
      if (S.roi_points.length >= 3) {
        const fx = pt.rx + S.roi_points[0][0] * pt.rw;
        const fy = pt.ry + S.roi_points[0][1] * pt.rh;
        nearFirst = Math.hypot(pt.px - fx, pt.py - fy) <= 15;
      }
      hoverState = { canvas, nx: pt.nx, ny: pt.ny, nearFirst };
      scheduleRoiDraw();
    });

    canvas.addEventListener("mouseleave", () => {
      if (hoverState?.canvas === canvas) {
        hoverState = null;
        scheduleRoiDraw();
      }
    });

    canvas.addEventListener("click", (e) => {
      if (!roiDrawing || S.roi_mode === "full") return;
      e.preventDefault();
      const pt = eventToRoiPoint(e, canvas, img);
      if (!pt) return;

      if (S.roi_points.length >= 3) {
        const fx = pt.rx + S.roi_points[0][0] * pt.rw;
        const fy = pt.ry + S.roi_points[0][1] * pt.rh;
        if (Math.hypot(pt.px - fx, pt.py - fy) <= 15) {
          finishRoiPolygon();
          return;
        }
      }

      // Chống trùng đỉnh khi double-click nhanh tại cùng 1 vị trí
      if (S.roi_points.length > 0) {
        const last = S.roi_points[S.roi_points.length - 1];
        const lx = pt.rx + last[0] * pt.rw;
        const ly = pt.ry + last[1] * pt.rh;
        if (Math.hypot(pt.px - lx, pt.py - ly) < 4) return;
      }

      S.roi_points.push([pt.nx, pt.ny]);
      roiClosed = false;
      save();
      syncRoiUI();
    });

    canvas.addEventListener("dblclick", (e) => {
      if (!roiDrawing || S.roi_mode === "full") return;
      e.preventDefault();
      if (S.roi_points.length >= 3) finishRoiPolygon();
    });
  }

  function scheduleRoiDraw() {
    if (roiAnimFrame) return;
    roiAnimFrame = requestAnimationFrame(() => {
      roiAnimFrame = 0;
      drawRoiOverlay(vidCanvas, vidImg);
      drawRoiOverlay(imgCanvas, imgOut);
    });
  }

  function drawRoiOverlay(canvas, img) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (!ctx || cw <= 0 || ch <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(cw * dpr), ph = Math.round(ch * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    ctx.setTransform(pw / cw, 0, 0, ph / ch, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    if (S.roi_mode === "full") return;
    const { rx, ry, rw, rh } = getBitmapRect(canvas, img);
    if (rw <= 2 || rh <= 2) return;

    const isForbidden = S.roi_mode === "forbidden";
    const stroke = isForbidden ? "#FF4D42" : "#4ADE80";
    const fill = isForbidden ? "rgba(215, 42, 30, 0.22)" : "rgba(56, 176, 72, 0.20)";
    const pillBg = isForbidden ? "rgba(185, 37, 24, 0.92)" : "rgba(35, 71, 29, 0.92)";

    if (roiDrawing) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.42)";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);
      ctx.restore();
    }

    const pts = S.roi_points.map(([nx, ny]) => [rx + nx * rw, ry + ny * rh]);
    if (!pts.length) return;

    const activeHover = roiDrawing && hoverState?.canvas === canvas ? hoverState : null;
    const hx = activeHover ? rx + activeHover.nx * rw : 0;
    const hy = activeHover ? ry + activeHover.ny * rh : 0;

    // Tô nền đa giác
    if (roiClosed && pts.length >= 3) {
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2.4;
      ctx.lineJoin = "round";
      ctx.stroke();
    } else {
      if (activeHover && pts.length >= 2) {
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.lineTo(hx, hy);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (pts.length >= 2) {
        ctx.beginPath();
        pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 2.2;
        ctx.lineJoin = "round";
        ctx.stroke();
      }
      if (activeHover) {
        const [lx, ly] = pts[pts.length - 1];
        ctx.save();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(lx, ly);
        ctx.lineTo(hx, hy);
        ctx.stroke();
        if (pts.length >= 2) {
          ctx.globalAlpha = 0.55;
          ctx.beginPath();
          ctx.moveTo(hx, hy);
          ctx.lineTo(pts[0][0], pts[0][1]);
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // Vẽ các đỉnh đa giác
    pts.forEach(([x, y], i) => {
      const isStartClosable = i === 0 && roiDrawing && pts.length >= 3;
      const r = isStartClosable ? (activeHover?.nearFirst ? 8 : 6.2) : 4.5;
      if (isStartClosable) {
        ctx.beginPath();
        ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.28)";
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = isStartClosable && activeHover?.nearFirst ? "#FFE066" : "#FFFFFF";
      ctx.fill();
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = stroke;
      ctx.stroke();
    });

    // Nhãn hiển thị chế độ vùng
    if (pts.length >= 3 && roiClosed) {
      const label = isForbidden
        ? `VÙNG CẤM (≥ ${S.restricted_thr} người)`
        : `VÙNG ROI (${fmt(S.roi_area)} m²)`;
      ctx.font = "700 11px Plus Jakarta Sans, sans-serif";
      const tw = ctx.measureText(label).width;
      const boxW = tw + 14, boxH = 22;
      const topPt = pts.reduce((best, p) => (p[1] < best[1] ? p : best), pts[0]);
      const bx = Math.min(rx + rw - boxW - 4, Math.max(rx + 4, topPt[0] - boxW / 2));
      const by = topPt[1] - boxH - 8 >= ry + 4 ? topPt[1] - boxH - 8 : Math.min(ry + rh - boxH - 4, topPt[1] + 8);

      ctx.fillStyle = pillBg;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(bx, by, boxW, boxH, 5);
      } else {
        ctx.rect(bx, by, boxW, boxH);
      }
      ctx.fill();
      ctx.fillStyle = "#FFFFFF";
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.fillText(label, bx + 7, by + boxH / 2 + 0.5);
    }
  }

  bindRoiToolbar("vid");
  bindRoiToolbar("img");
  bindRoiCanvas(vidCanvas, vidImg);
  bindRoiCanvas(imgCanvas, imgOut);

  vidImg.addEventListener("load", scheduleRoiDraw);
  imgOut.addEventListener("load", scheduleRoiDraw);
  new ResizeObserver(scheduleRoiDraw).observe(vidCanvas);
  new ResizeObserver(scheduleRoiDraw).observe(imgCanvas);

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

  function normalizeEvidenceUrl(rec) {
    if (!rec) return "";
    if (rec.evidence_url) return rec.evidence_url;
    if (rec.snapshot_url) return rec.snapshot_url;
    if (rec.image_url) return rec.image_url;
    const rawFile = rec.evidence_file || (rec.evidence_path ? String(rec.evidence_path).split(/[\\/]/).pop() : "");
    return rawFile ? `/api/evidence/${encodeURIComponent(rawFile)}` : "";
  }

  // ------------------------------------------------------------ 01. Phân tích ảnh tĩnh
  let imgData = null, imageRequest = 0;

  async function analyseImage(file) {
    lastImageFile = file;
    if (roiDrawing && S.roi_points.length >= 3) {
      roiDrawing = false;
      roiClosed = true;
      syncRoiUI();
    }
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
      $("#img-count").textContent = fmt(Math.round(imgData.count));
      $("#img-density").textContent = fmt(imgData.density, 2);

      const imgBadges = buildAlertBadges(imgData);
      $("#img-level").innerHTML = levelCard(imgData.level, imgBadges);

      // Kiểm tra ảnh bằng chứng nếu có cảnh báo lưu xuống CSDL
      const alertList = Array.isArray(imgData.alert_records) ? imgData.alert_records
        : (Array.isArray(imgData.alerts) ? imgData.alerts : []);
      const firstAlertWithEv = alertList.find((a) => normalizeEvidenceUrl(a));
      const evUrl = normalizeEvidenceUrl(imgData) || normalizeEvidenceUrl(firstAlertWithEv);
      const evBox = $("#img-evidence");
      if (evUrl) {
        const evKey = registerEvidence({
          url: evUrl,
          title: firstAlertWithEv?.message || firstAlertWithEv?.description || `Bằng chứng cảnh báo · ${file.name}`,
          level: firstAlertWithEv?.level ?? imgData.level,
          timeText: new Date().toLocaleString("vi-VN"),
          source: `Ảnh tĩnh (${file.name})`,
          model: imgData.model_label,
          count: imgData.count,
          density: imgData.density
        });
        evBox.innerHTML = `
          <img class="evidence-thumb" src="${esc(evUrl)}" alt="Ảnh bằng chứng" data-evidence-key="${evKey}">
          <div class="img-evidence-info">
            <span class="img-evidence-lbl">Đã lưu bằng chứng cảnh báo</span>
            <button type="button" class="btn-link-evidence" data-evidence-key="${evKey}">🔍 Phóng to ảnh bằng chứng</button>
          </div>
        `;
        evBox.hidden = false;
      } else {
        evBox.hidden = true;
        evBox.innerHTML = "";
      }

      const areaNote = hasValidRoi() && S.roi_mode === "roi" ? ` · ROI ${fmt(S.roi_area)}m²` : "";
      $("#img-time").textContent = `${imgData.model_label} · ${file.name}${areaNote} · ${fmt(imgData.ms)} ms`;
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
    scheduleRoiDraw();
  }
  $$("#img-seg button").forEach((b) => b.addEventListener("click", () => showImg(b.dataset.show)));
  dropzone($("#img-drop"), $("#img-input"), (file) => analyseImage(file));

  // ------------------------------------------------------------ 02. Video & Camera + First-Frame Preview + Optical Flow
  let source = "file", videoFile = null, videoPreviewUrl = null, previewSeq = 0;
  let job = null, rows = [], logRows = [], videoModel = "", videoSettings = null;
  const sourceLabels = {
    file: { run: "Bắt đầu phân tích video", opening: "Đang nạp video lên máy chủ..." },
    camera: { run: "Bắt đầu theo dõi camera", opening: "Đang mở kết nối camera..." },
    youtube: { run: "Bắt đầu nhận diện trực tiếp", opening: "Đang kết nối luồng YouTube..." },
  };

  function extractVideoPreview(file) {
    const token = ++previewSeq;
    videoPreviewUrl = null;
    const blobUrl = URL.createObjectURL(file);
    const vid = document.createElement("video");
    vid.preload = "auto";
    vid.muted = true;
    vid.playsInline = true;

    let cleaned = false;
    const cleanup = () => {
      if (cleaned) return;
      cleaned = true;
      URL.revokeObjectURL(blobUrl);
      vid.removeAttribute("src");
      try { vid.load(); } catch (_) { /* bỏ qua */ }
    };

    const captureFrame = () => {
      if (cleaned || token !== previewSeq) { cleanup(); return; }
      const vw = vid.videoWidth, vh = vid.videoHeight;
      if (!vw || !vh) { cleanup(); return; }
      const maxDim = 960;
      const scale = Math.min(1, maxDim / Math.max(vw, vh));
      const cw = Math.max(2, Math.round(vw * scale));
      const ch = Math.max(2, Math.round(vh * scale));
      const off = document.createElement("canvas");
      off.width = cw;
      off.height = ch;
      const ctx = off.getContext("2d");
      ctx.drawImage(vid, 0, 0, cw, ch);
      const dataUrl = off.toDataURL("image/jpeg", 0.78);
      cleanup();
      if (token !== previewSeq || job) return;
      videoPreviewUrl = dataUrl;
      vidImg.src = dataUrl;
      vidImg.hidden = false;
      $("#live-source-empty").hidden = true;
      $("#live-source-caption").textContent = "XEM TRƯỚC · 00:00.1s";
      scheduleRoiDraw();
    };

    vid.addEventListener("loadedmetadata", () => {
      if (token !== previewSeq) { cleanup(); return; }
      const seekTo = Number.isFinite(vid.duration) && vid.duration > 0
        ? Math.min(0.1, vid.duration * 0.5)
        : 0.1;
      try {
        vid.currentTime = seekTo;
      } catch (_) {
        captureFrame();
      }
    }, { once: true });

    vid.addEventListener("seeked", captureFrame, { once: true });
    vid.addEventListener("error", cleanup, { once: true });
    setTimeout(cleanup, 6000);
    vid.src = blobUrl;
  }

  $$("#src-seg button").forEach((b) => b.addEventListener("click", () => {
    if ($("#run-btn").disabled) return;
    source = b.dataset.src;
    $$("#src-seg button").forEach((x) => x.classList.toggle("active", x === b));
    for (const key of Object.keys(sourceLabels)) $("#src-" + key).hidden = key !== source;
    $("#run-btn").querySelector("span").textContent = sourceLabels[source].run;
    $("#vid-error").hidden = true;
    if (source === "file" && videoPreviewUrl && !job) {
      vidImg.src = videoPreviewUrl;
      vidImg.hidden = false;
      $("#live-source-empty").hidden = true;
      $("#live-source-caption").textContent = "XEM TRƯỚC · 00:00.1s";
      scheduleRoiDraw();
    } else if (source !== "file" && videoPreviewUrl && !job && vidImg.src === videoPreviewUrl) {
      vidImg.hidden = true;
      vidImg.removeAttribute("src");
      $("#live-source-empty").hidden = false;
      $("#live-source-caption").textContent = "00:00:00";
      scheduleRoiDraw();
    }
  }));

  dropzone($("#vid-drop"), $("#vid-input"), (f) => {
    videoFile = f;
    $("#vid-name").textContent = f.name;
    extractVideoPreview(f);
  });

  function resetMotionBar() {
    $("#flow-speed").textContent = "–";
    const speedTag = $("#flow-speed-tag");
    speedTag.textContent = S.enable_flow ? "Chờ dữ liệu" : "Đã tắt";
    speedTag.className = "motion-tag";

    $("#flow-entropy").textContent = "–";
    const bar = $("#flow-entropy-bar");
    bar.style.width = "0%";
    bar.style.backgroundColor = "var(--color-moss-mid)";

    $("#flow-counter").textContent = "–";
    $("#flow-dir").textContent = "–";
    const stateTag = $("#flow-state");
    stateTag.textContent = S.enable_flow ? "Trật tự" : "Tắt Flow";
    stateTag.className = S.enable_flow ? "motion-tag ok" : "motion-tag";
  }

  function updateMotionBar(d) {
    const m = d.motion || d.flow || {};
    const enabled = d.flow_enabled ?? m.enabled ?? S.enable_flow;
    if (!enabled) {
      resetMotionBar();
      return;
    }

    const speed = firstNum(d.flow_speed, d.motion_speed, d.speed_pct, d.speed, m.speed, m.flow_speed, m.speed_pct);
    const entropy = firstNum(d.flow_entropy, d.motion_entropy, d.direction_entropy, d.entropy, m.entropy, m.flow_entropy);

    let counterPct = firstNum(d.counter_pct, d.counter_flow_pct, m.counter_pct, m.counter_flow_pct);
    if (counterPct === null) {
      const rawRatio = firstNum(d.counter_flow_ratio, d.counter_ratio, d.flow_counter, m.counter_ratio, m.counter_flow_ratio);
      if (rawRatio !== null) {
        const isFractionKey = d.counter_flow_ratio != null || d.counter_ratio != null || m.counter_ratio != null || m.counter_flow_ratio != null;
        counterPct = isFractionKey && rawRatio <= 1.0 ? rawRatio * 100 : rawRatio;
      }
    }

    const angle = firstNum(d.dominant_angle, d.flow_angle, d.dominant_dir, d.flow_dir, d.angle_deg, m.dominant_angle, m.angle_deg, m.angle, m.dir_deg);
    const isPanic = !!(d.panic || d.motion_panic || m.panic);
    const isCounter = !!(d.counter_flow || m.counter_flow);

    if (speed !== null) {
      $("#flow-speed").textContent = `${fmt(speed, 1)} %/s`;
      const speedTag = $("#flow-speed-tag");
      if (isPanic || speed >= S.speed_thr) {
        speedTag.textContent = "Chạy nhanh";
        speedTag.className = "motion-tag danger";
      } else if (speed < 1.0) {
        speedTag.textContent = "Tĩnh";
        speedTag.className = "motion-tag";
      } else {
        speedTag.textContent = "Di chuyển ổn định";
        speedTag.className = "motion-tag ok";
      }
    }

    if (entropy !== null) {
      const ec = Math.min(1, Math.max(0, entropy));
      $("#flow-entropy").textContent = fmt(ec, 2);
      const bar = $("#flow-entropy-bar");
      bar.style.width = `${(ec * 100).toFixed(1)}%`;
      bar.style.backgroundColor = ec >= S.entropy_thr
        ? "var(--color-alert-red)"
        : (ec >= S.entropy_thr * 0.72 ? "var(--color-alert-yellow)" : "var(--color-moss-mid)");
    }

    if (counterPct !== null) {
      const cc = Math.min(100, Math.max(0, counterPct));
      $("#flow-counter").textContent = `${fmt(cc, 0)}%`;
    }

    if (angle !== null && (speed === null || speed >= 0.4)) {
      const deg = ((Math.round(angle) % 360) + 360) % 360;
      const arrows = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"];
      $("#flow-dir").textContent = `${arrows[Math.round(deg / 45) % 8]} ${deg}°`;
    } else if (speed !== null && speed < 0.4) {
      $("#flow-dir").textContent = "· Tĩnh";
    }

    const stateTag = $("#flow-state");
    if (isPanic) {
      stateTag.textContent = "🚨 Hỗn loạn";
      stateTag.className = "motion-tag danger";
    } else if (isCounter || (counterPct !== null && counterPct >= S.counter_thr)) {
      stateTag.textContent = "⚠ Ngược chiều";
      stateTag.className = "motion-tag warn";
    } else {
      stateTag.textContent = "Trật tự";
      stateTag.className = "motion-tag ok";
    }
  }

  function renderSessionLogTable() {
    const L = STATUS.levels || [];
    $("#log-body").innerHTML = logRows.map((r) => {
      const lv = L[r.level] || { name: String(r.level ?? ""), color: "#7A5335" };
      let evCell = `<span class="muted-dash">–</span>`;
      if (r.evidence_url) {
        const key = r.evKey || (r.evKey = registerEvidence({
          url: r.evidence_url,
          title: r.text,
          level: r.level,
          timeText: `${fmt(r.t, 1)}s`,
          source: source === "file"
            ? (videoFile?.name || "Tệp video")
            : (source === "youtube"
              ? `YouTube Live (${$("#youtube-url").value.trim() || "trực tiếp"})`
              : `Camera (${$("#cam-url").value.trim() || "0"})`),
          model: videoModel,
          count: r.count,
          density: r.density
        }));
        evCell = `<img class="evidence-thumb" src="${esc(r.evidence_url)}" alt="Bằng chứng" title="Bấm để phóng to ảnh bằng chứng" data-evidence-key="${key}">`;
      }
      return `
        <tr>
          <td style="font-family:var(--font-mono); font-weight:600">${fmt(r.t, 1)}s</td>
          <td><span class="lv-pill" style="background-color:${lv.color}">MỨC ${esc(lv.name)}</span></td>
          <td>${esc(r.text)}</td>
          <td>${evCell}</td>
        </tr>
      `;
    }).join("");

    $("#log-empty").hidden = logRows.length > 0;
    $("#log-table").hidden = logRows.length === 0;
  }

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
    const s = Math.max(0, Math.floor(secs || 0));
    const ms = Math.floor(((secs || 0) - s) * 10);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${Math.max(0, ms)}s`;
  }

  $("#run-btn").addEventListener("click", async () => {
    $("#vid-error").hidden = true;
    if (!STATUS.models?.[S.model]?.ok) return fail("Chưa nạp được mô hình AI.");

    // Tự chốt đa giác nếu người dùng đã chấm >= 3 đỉnh rồi bấm chạy luôn
    if (roiDrawing) {
      if (S.roi_points.length >= 3) {
        roiDrawing = false;
        roiClosed = true;
      } else {
        roiDrawing = false;
        S.roi_points = [];
        roiClosed = false;
      }
      save();
      syncRoiUI();
    }

    const fd = formData();
    fd.append("source", source);
    if (source === "file") {
      if (!videoFile) return fail("Hãy chọn một tệp video trước.");
      fd.append("file", videoFile);
    } else if (source === "youtube") {
      const url = $("#youtube-url").value.trim();
      if (!url) return fail("Hãy nhập liên kết hoặc mã nhúng YouTube.");
      fd.append("url", url);
      fd.append("max_secs", $("#youtube-secs").value || 60);
    } else {
      fd.append("url", $("#cam-url").value.trim() || "0");
      fd.append("max_secs", $("#cam-secs").value || 60);
    }

    rows = [];
    logRows = [];
    videoModel = STATUS.models[S.model].label;
    videoSettings = { crowd_type: S.crowd_type, area: effectiveArea() };

    $("#live-count").textContent = "–";
    $("#live-density").textContent = "–";
    $("#live-level").innerHTML = `<div class="level-placeholder">Đang khởi tạo luồng phân tích...</div>`;
    resetMotionBar();

    const imgAi = $("#live-frame");
    imgAi.hidden = true;
    imgAi.removeAttribute("src");
    $("#live-ai-empty").hidden = false;

    if (source !== "file" || !videoPreviewUrl) {
      vidImg.hidden = true;
      vidImg.removeAttribute("src");
      $("#live-source-empty").hidden = false;
      $("#live-source-caption").textContent = "00:00:00";
    }

    $("#live-caption").textContent = "Chờ tín hiệu...";
    $("#live-stats").hidden = true;
    $("#live-chart").hidden = true;
    $("#live-progress").hidden = true;
    $("#live-progress .progress-fill").style.width = "0%";
    $("#summary").textContent = "";
    $("#log-wrap").hidden = true;
    $("#log-accordion").open = false;
    $("#log-body").innerHTML = "";

    setRunning(true, sourceLabels[source].opening);

    let id;
    try {
      const res = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await readError(res));
      id = (await res.json()).id;
    } catch (e) {
      setRunning(false);
      return fail(e.message);
    }

    const es = new EventSource(`/api/jobs/${id}/events`);
    job = { id, es, t0: performance.now(), frames: 0, aiFrames: 0, info: null };

    es.addEventListener("start", (e) => {
      job.info = JSON.parse(e.data);
      const i = job.info;
      $("#vid-loading-text").textContent = "Đang xử lý luồng nhận diện...";
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
    const L = STATUS.levels || [];

    // Cập nhật Khung 1: Camera nguồn trực tiếp
    if (d.raw_frame) {
      $("#vid-loading").hidden = true;
      vidImg.src = d.raw_frame;
      vidImg.hidden = false;
      $("#live-source-empty").hidden = true;
      $("#live-source-caption").textContent = formatTime(d.source_t ?? d.t);
      job.frames += 1;
    }

    // Cập nhật Khung 2: Camera nhận diện AI (đồng bộ trực tiếp theo FPS hiển thị, 0 trễ khung)
    if (d.frame) {
      $("#vid-loading").hidden = true;
      const imgAi = $("#live-frame");
      imgAi.src = d.frame;
      imgAi.hidden = false;
      $("#live-ai-empty").hidden = true;
      const liveSec = d.source_t ?? d.t;
      $("#live-caption").textContent = d.count != null
        ? `${videoModel} · ${job.info?.visualization ?? "Nhận diện"} · ${formatTime(liveSec)}`
        : `${videoModel} · Đang khởi chạy suy luận · ${formatTime(liveSec)}`;
      job.aiFrames = (job.aiFrames || 0) + 1;
    }

    // Cập nhật thanh Optical Flow Telemetry
    updateMotionBar(d);

    // Cập nhật số liệu KPI & Mức cảnh báo (TO, NỔI BẬT)
    if (d.fresh && Number.isFinite(d.t) && Number.isFinite(d.count) && Number.isFinite(d.smoothed)) {
      const { t, count, smoothed, density, level, surge } = d;
      rows.push({ t, count, smoothed, density, level, surge });

      const wasLogEmpty = logRows.length === 0;
      const incomingRecords = Array.isArray(d.alert_records) && d.alert_records.length
        ? d.alert_records
        : (Array.isArray(d.alerts) && d.alerts.length ? d.alerts : null);

      let latestToastMsg = "";
      if (incomingRecords) {
        for (const rec of incomingRecords) {
          const text = rec.message || rec.text || rec.description || "Phát hiện sự kiện cảnh báo";
          logRows.push({
            t: Number.isFinite(Number(rec.t ?? rec.video_t)) ? Number(rec.t ?? rec.video_t) : t,
            level: rec.level ?? level,
            text,
            evidence_url: normalizeEvidenceUrl(rec),
            count: rec.count ?? count,
            density: rec.density ?? density
          });
          latestToastMsg = text;
        }
      } else if (Array.isArray(d.events) && d.events.length) {
        for (const ev of d.events) {
          if (ev && typeof ev === "object") {
            const text = ev.message || ev.text || ev.description || "Phát hiện sự kiện cảnh báo";
            logRows.push({
              t: Number.isFinite(Number(ev.t)) ? Number(ev.t) : t,
              level: ev.level ?? level,
              text,
              evidence_url: normalizeEvidenceUrl(ev) || normalizeEvidenceUrl(d),
              count: ev.count ?? count,
              density: ev.density ?? density
            });
            latestToastMsg = text;
          } else {
            const text = String(ev);
            logRows.push({
              t,
              level,
              text,
              evidence_url: normalizeEvidenceUrl(d),
              count,
              density
            });
            latestToastMsg = text;
          }
        }
      }

      if (logRows.length > 0) {
        $("#log-wrap").hidden = false;
        if (wasLogEmpty) $("#log-accordion").open = true;
        $("#summary").textContent = `Đang giám sát trực tiếp · Đã ghi nhận ${fmt(logRows.length)} sự kiện cảnh báo`;
        $("#csv-btn").disabled = false;
        renderSessionLogTable();
      }

      $("#live-count").textContent = fmt(Math.round(d.count));
      $("#live-density").textContent = fmt(d.density, 2);
      $("#live-level").innerHTML = levelCard(d.level, buildAlertBadges(d));
      $("#live-chart").hidden = false;
      scheduleChartDraw();

      if (latestToastMsg && L[d.level]) {
        toast(latestToastMsg);
      }
    }

    // Tiến trình
    if (d.progress != null) {
      $("#live-progress").hidden = false;
      $("#live-progress .progress-fill").style.width = `${(d.progress * 100).toFixed(1)}%`;
    }

    // Thống kê hiệu năng OSD nhỏ bên dưới
    if (job) {
      const el = performance.now() - job.t0;
      const shownFps = el > 500 ? (job.frames * 1000) / el : 0;
      const aiShownFps = el > 500 ? ((job.aiFrames || 0) * 1000) / el : 0;
      $("#live-stats").textContent =
        `Luồng nguồn ${fmt(shownFps, 1)} FPS · Khung AI ${fmt(aiShownFps, 1)} FPS` +
        (d.infer_ms ? ` · Cập nhật mô hình: ${fmt(d.infer_ms)} ms (${fmt(d.infer_fps, 1)} lần/s)` : "") +
        (d.max_side ? ` · Độ phân giải AI: ${d.max_side}px` : "");
    }
  }

  function finish(summary) {
    if (!job) return;
    job.es.close();
    job = null;
    setRunning(false);

    if (!rows.length && !summary) return;
    $("#log-wrap").hidden = false;
    const L = STATUS.levels || [];

    renderSessionLogTable();

    if (summary) {
      const fps = summary.display_fps ? ` · ${fmt(summary.display_fps, 1)} FPS phát` : "";
      const inf = summary.infer_ms ? ` · Suy luận ${fmt(summary.infer_ms)} ms/lần` : "";
      const maxLvName = L[summary.max_level]?.name ?? summary.max_level;
      $("#summary").textContent = `Tổng kết: ${summary.model_label} · ${summary.samples} lần suy luận · Đỉnh điểm: ${fmt(summary.max_count)} người · Mức cao nhất: ${maxLvName}${fps}${inf}${summary.stopped ? " (Đã dừng)" : ""}`;
    }

    $("#csv-btn").disabled = rows.length === 0;
  }

  $("#csv-btn").addEventListener("click", () => {
    const L = STATUS.levels || [];
    const head = ["Mo hinh", "Thoi gian (s)", "So nguoi", "So nguoi (lam tron)", "Mat do (/m2)", "Muc canh bao", "Tang dot bien"];
    const lines = [head.join(",")].concat(rows.map((r) => [
      videoModel,
      r.t.toFixed(1),
      r.count.toFixed(1),
      r.smoothed.toFixed(1),
      r.density.toFixed(3),
      L[r.level]?.name ?? r.level,
      r.surge ? "True" : "False"
    ].join(",")));

    const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `crowd_monitor_${Date.now()}.csv` });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ------------------------------------------------------------ 03. Kho Lịch sử Cảnh báo PostgreSQL
  let historyAlerts = [];
  let historySessions = [];
  let histReqSeq = 0;

  function formatDateTime(raw, videoSec) {
    if (!raw && videoSec == null) return "–";
    let main = "";
    if (raw) {
      const dt = new Date(raw);
      main = Number.isNaN(dt.getTime()) ? String(raw) : dt.toLocaleString("vi-VN", { hour12: false });
    }
    const sub = videoSec != null && Number.isFinite(Number(videoSec)) ? ` (t=${fmt(Number(videoSec), 1)}s)` : "";
    return `${main}${sub}`.trim();
  }

  async function loadHistory(refreshSessions = false) {
    const seq = ++histReqSeq;
    const loading = $("#hist-loading");
    const errBox = $("#hist-error");
    errBox.hidden = true;
    loading.hidden = false;

    try {
      // Làm mới trạng thái DB và danh sách phiên nếu cần
      if (refreshSessions) {
        try {
          const stRes = await fetch("/api/status");
          if (stRes.ok) {
            STATUS = await stRes.json();
            syncDatabase();
          }
        } catch (_) { /* bỏ qua */ }

        const sessRes = await fetch("/api/history/sessions");
        if (sessRes.ok) {
          const sessData = await sessRes.json();
          if (seq !== histReqSeq) return;
          historySessions = Array.isArray(sessData) ? sessData : (sessData.sessions || sessData.items || []);
          const totalSessions = sessData.total ?? historySessions.length;
          $("#hist-total-sessions").textContent = fmt(totalSessions);
          populateSessionSelect(historySessions);
          if (sessData.db_ok !== undefined) {
            STATUS.database = { ...(STATUS.database || {}), ok: !!sessData.db_ok };
            syncDatabase();
          }
        }
      }

      const qs = new URLSearchParams();
      const lv = $("#hist-level").value;
      const tp = $("#hist-type").value;
      const sid = $("#hist-session").value;
      if (lv !== "") qs.set("level", lv);
      if (tp !== "") {
        qs.set("alert_type", tp);
        qs.set("type", tp);
      }
      if (sid !== "") {
        qs.set("session_id", sid);
        qs.set("session", sid);
      }
      qs.set("limit", "200");

      const alertRes = await fetch(`/api/history/alerts?${qs.toString()}`);
      if (!alertRes.ok) throw new Error(await readError(alertRes));
      const alertData = await alertRes.json();
      if (seq !== histReqSeq) return;

      historyAlerts = Array.isArray(alertData) ? alertData : (alertData.alerts || alertData.items || []);
      const totalAlerts = alertData.total ?? historyAlerts.length;
      $("#hist-total-alerts").textContent = fmt(totalAlerts);
      if (alertData.total_sessions != null) {
        $("#hist-total-sessions").textContent = fmt(alertData.total_sessions);
      }
      renderHistoryTable();
    } catch (e) {
      if (seq !== histReqSeq) return;
      errBox.textContent = e.message || "Không truy vấn được lịch sử từ PostgreSQL.";
      errBox.hidden = false;
      historyAlerts = [];
      renderHistoryTable();
    } finally {
      if (seq === histReqSeq) loading.hidden = true;
    }
  }

  function populateSessionSelect(sessions) {
    const sel = $("#hist-session");
    const prev = sel.value;
    const opts = [`<option value="">Tất cả phiên giám sát (${sessions.length})</option>`];
    for (const s of sessions) {
      const id = s.id ?? s.session_id ?? "";
      if (!id) continue;
      const src = s.source_name || s.source || s.kind || "Phiên";
      const mdl = s.model_label || s.model || "";
      const when = formatDateTime(s.started_at || s.created_at);
      const cnt = s.alert_count ?? s.alerts_count;
      const cntTxt = cnt != null ? ` · ${cnt} cảnh báo` : "";
      opts.push(`<option value="${esc(id)}">${esc(when)} — ${esc(src)}${mdl ? ` (${esc(mdl)})` : ""}${esc(cntTxt)}</option>`);
    }
    sel.innerHTML = opts.join("");
    if (prev && [...sel.options].some((o) => o.value === prev)) {
      sel.value = prev;
    }
  }

  function renderHistoryTable() {
    const L = STATUS.levels || [];
    const body = $("#hist-body");
    const empty = $("#hist-empty");
    const table = $("#hist-table");

    if (!historyAlerts.length) {
      body.innerHTML = "";
      table.hidden = true;
      empty.hidden = false;
      $("#hist-csv").disabled = true;
      return;
    }

    table.hidden = false;
    empty.hidden = true;
    $("#hist-csv").disabled = false;

    body.innerHTML = historyAlerts.map((a) => {
      const lvIdx = Number.isFinite(Number(a.level)) ? Number(a.level) : 2;
      const lv = L[lvIdx] || { name: a.level_name || String(lvIdx), color: "#C2531F" };
      const rawType = a.alert_type || a.type || "density";
      const typeLabel = ALERT_TYPE_LABELS[rawType] || rawType;
      const timeStr = formatDateTime(a.created_at || a.timestamp || a.ts, a.video_t ?? a.t_sec ?? a.t);
      const srcName = a.source_name || a.source || a.kind || "–";
      const modelName = (a.model_label || a.model || "").toUpperCase();
      const countVal = firstNum(a.count, a.people_count);
      const densityVal = firstNum(a.density);
      const msg = a.message || a.description || a.text || "Cảnh báo đám đông";
      const evUrl = normalizeEvidenceUrl(a);

      let evHtml = `<span class="muted-dash">–</span>`;
      if (evUrl) {
        const key = registerEvidence({
          url: evUrl,
          title: `${typeLabel} · ${msg}`,
          typeLabel,
          level: lvIdx,
          timeText: timeStr,
          source: srcName,
          model: modelName,
          count: countVal,
          density: densityVal
        });
        evHtml = `<img class="evidence-thumb" src="${esc(evUrl)}" alt="Bằng chứng" title="Bấm để xem ảnh gốc" data-evidence-key="${key}">`;
      }

      return `
        <tr>
          <td class="hist-col-time">${esc(timeStr)}</td>
          <td>
            <div class="hist-src-main">${esc(srcName)}</div>
            ${modelName ? `<span class="hist-model-sub">${esc(modelName)}</span>` : ""}
          </td>
          <td>
            <div class="hist-badges-cell">
              <span class="lv-pill" style="background-color:${lv.color}">MỨC ${esc(lv.name)}</span>
              <span class="hist-type-tag">${esc(typeLabel)}</span>
            </div>
          </td>
          <td style="font-family:var(--font-mono)">
            <b>${countVal !== null ? fmt(Math.round(countVal)) : "–"}</b> người
            <div class="hist-density-sub">${densityVal !== null ? `${fmt(densityVal, 2)} ng/m²` : ""}</div>
          </td>
          <td>${esc(msg)}</td>
          <td>${evHtml}</td>
        </tr>
      `;
    }).join("");
  }

  $("#hist-level").addEventListener("change", () => loadHistory(false));
  $("#hist-type").addEventListener("change", () => loadHistory(false));
  $("#hist-session").addEventListener("change", () => loadHistory(false));
  $("#hist-refresh").addEventListener("click", () => loadHistory(true));

  $("#hist-csv").addEventListener("click", () => {
    if (!historyAlerts.length) return;
    const L = STATUS.levels || [];
    const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const head = ["ID", "Thoi gian", "Phien", "Nguon", "Mo hinh", "Loai canh bao", "Muc", "So nguoi", "Mat do (/m2)", "Noi dung", "Anh bang chung"];
    const lines = [head.join(",")].concat(historyAlerts.map((a) => {
      const lvIdx = Number(a.level);
      const lvName = L[lvIdx]?.name || a.level_name || a.level || "";
      const rawType = a.alert_type || a.type || "";
      return [
        csvCell(a.id ?? ""),
        csvCell(formatDateTime(a.created_at || a.timestamp, a.video_t ?? a.t)),
        csvCell(a.session_id ?? ""),
        csvCell(a.source_name || a.source || ""),
        csvCell(a.model_label || a.model || ""),
        csvCell(ALERT_TYPE_LABELS[rawType] || rawType),
        csvCell(lvName),
        csvCell(a.count != null ? Number(a.count).toFixed(1) : ""),
        csvCell(a.density != null ? Number(a.density).toFixed(3) : ""),
        csvCell(a.message || a.description || a.text || ""),
        csvCell(normalizeEvidenceUrl(a))
      ].join(",");
    }));

    const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `crowd_history_${Date.now()}.csv`
    });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  $("#hist-clear").addEventListener("click", async () => {
    if (!confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử phiên giám sát và cảnh báo trong PostgreSQL?")) return;
    try {
      const res = await fetch("/api/history", { method: "DELETE" });
      if (!res.ok) throw new Error(await readError(res));
      $("#hist-session").value = "";
      toast("Đã xóa toàn bộ lịch sử cảnh báo.");
      loadHistory(true);
    } catch (e) {
      toast(e.message || "Không xóa được lịch sử.");
    }
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
    const thr = (STATUS.thresholds?.[videoSettings?.crowd_type] || []).map((x) => x * videoSettings.area);
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
      const level = STATUS.levels?.[i + 1];
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
    ctx.fillStyle = STATUS.levels?.[last.level]?.color || "#382314";
    ctx.fill();
  }

  new ResizeObserver(() => {
    if (rows.length) scheduleChartDraw();
  }).observe(chart);
  window.addEventListener("resize", () => {
    if (rows.length) scheduleChartDraw();
    scheduleRoiDraw();
  });

  // ------------------------------------------------------------ Khởi tạo hệ thống
  async function init() {
    try {
      STATUS = await (await fetch("/api/status")).json();
    } catch (_) { /* Máy chủ có thể chưa sẵn sàng */ }

    for (const option of modelSelect.options) {
      const model = STATUS.models?.[option.value];
      option.disabled = !model?.ok;
      if (model && !model.ok) option.textContent = `${model.label} (chưa nạp)`;
    }

    if (!STATUS.models?.[S.model]?.ok) S.model = STATUS.default_model || DEFAULTS.model;
    modelSelect.value = S.model;
    modelSelect.disabled = !STATUS.ok;

    if (!STATUS.ok) {
      const n = $("#model-missing");
      n.textContent = Object.values(STATUS.models || {}).map((model) => model.error).filter(Boolean).join(" ") ||
        "Chưa kết nối được với máy chủ xử lý AI.";
      n.hidden = false;
    }

    bindSettings();
    syncModel();
    syncDatabase();
    syncRoiUI();
    resetMotionBar();

    const start = location.hash.slice(1);
    go(["image", "video", "history"].includes(start) ? start : "video");
  }

  init();
})();
