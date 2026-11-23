"""Web app (Flask) đếm người, giám sát vùng đa giác (ROI/Vùng cấm), phân tích chuyển động, YouTube Live & lưu trữ PostgreSQL.

Chạy:   python server.py            → mở http://127.0.0.1:5000
Biến môi trường (tùy chọn):
    CROWD_WEIGHTS        đường dẫn trọng số CSRNet (mặc định weights/csrnet_partB.pth)
    CROWD_YOLO_WEIGHTS   đường dẫn trọng số YOLO11s đầu người
    CROWD_DATABASE_URL   chuỗi kết nối PostgreSQL 16 (mặc định đọc từ .env, database crowd_monitor)
    CROWD_NO_DB          đặt "1" để tắt lưu trữ PostgreSQL
    CROWD_HOST           địa chỉ lắng nghe (mặc định 127.0.0.1 — chỉ máy này truy cập được)
    CROWD_PORT           cổng (mặc định 5000)
    CROWD_THREADS        số luồng CPU cho suy luận (mặc định min(12, số nhân logic))
    CROWD_NO_ONNX        đặt "1" để tắt ONNX Runtime, chỉ dùng torch/TorchScript
    CONTACT_EMAIL        email hiện ở thanh liên hệ (bấm để sao chép); để trống thì ẩn

API:
    GET    /api/status                          trạng thái mô hình, backend suy luận, PostgreSQL, ngưỡng
    POST   /api/image                           phân tích một ảnh (multipart: file + tham số + ROI)
    POST   /api/jobs                            tạo phiên phân tích video/camera/YouTube Live → {id}
    GET    /api/jobs/<id>/events                luồng kết quả theo thời gian (Server-Sent Events)
    POST   /api/jobs/<id>/stop                  dừng phiên
    GET    /api/history/alerts                  truy vấn lịch sử cảnh báo từ PostgreSQL
    GET    /api/history/sessions                danh sách các phiên phân tích đã lưu
    GET    /api/history/sessions/<sid>/timeseries  chuỗi thời gian chi tiết của một phiên
    DELETE /api/history                         xóa toàn bộ lịch sử cảnh báo, phiên và ảnh bằng chứng
    GET    /api/evidence/<filename>             tải ảnh bằng chứng JPEG an toàn

Luồng video/camera/YouTube Live chạy ba vòng độc lập (xem VideoSession): đọc khung → suy luận → hiển thị.
Nhờ vậy khung hình vẫn phát đủ FPS mục tiêu trong khi CSRNet/YOLO + Optical Flow chỉ cập nhật
lớp phủ khi tính xong; việc lưu trữ PostgreSQL chạy bất đồng bộ trong luồng nền không gây trễ khung.
"""
import base64
import json
import os
import tempfile
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from crowd_core import (
    LEVELS,
    THRESHOLDS,
    CrowdMonitor,
    InferenceEngine,
    YoloHeadDetector,
    auto_max_side,
    density_level,
    draw_head_boxes,
    heat_layer,
    privacy_blur,
    soft_blur,
)
from roi_motion import (
    BoxFlowTracker,
    MotionAnalyzer,
    apply_roi,
    draw_motion_vectors,
    draw_roi_overlay,
    normalize_roi_mode,
    parse_roi_points,
)
from storage import PostgresEventStore
from video_source import open_capture as open_network_capture
from youtube_source import DEFAULT_YOUTUBE_URL, YouTubeStreamError, resolve_youtube_live, youtube_url

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("CROWD_WEIGHTS", os.path.join(BASE, "weights", "csrnet_partB.pth"))
YOLO_FILE = "yolo11s_partB_best_count.pt"
YOLO_LOCAL = os.path.join(BASE, "weights", YOLO_FILE)
YOLO_ARTIFACT = os.path.abspath(os.path.join(BASE, "..", "train_yolo", "yolo_partb_outputs", YOLO_FILE))
YOLO_WEIGHTS = os.environ.get("CROWD_YOLO_WEIGHTS", YOLO_LOCAL if os.path.isfile(YOLO_LOCAL) else YOLO_ARTIFACT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DISPLAY_SIDE = 960          # cạnh dài tối đa của ảnh tĩnh gửi về trình duyệt
VIDEO_DISPLAY_SIDE = 800    # cạnh dài khung hình video gửi về (nhỏ gọn → FPS mượt & ít trễ băng thông)
JPEG_QUALITY = 72
VIDEO_JPEG_QUALITY = 60
AUTO_BUDGET_MS = 800        # chế độ tự chọn độ phân giải: ngân sách thời gian cho 1 lần suy luận
DEFAULT_MAX_SIDE = 512      # mặc định = kích thước ô cắt lúc huấn luyện (số đếm đúng nhất)

# Màu mức cảnh báo theo bảng màu đất của giao diện (Xanh, Vàng, Cam, Đỏ)
UI_LEVEL_COLORS = ["#325825", "#A8741A", "#C2531F", "#A3261A"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ---------------------------------------------------------------------------- Kho dữ liệu PostgreSQL
def _init_store():
    if os.environ.get("CROWD_NO_DB") == "1":
        st = PostgresEventStore(dsn="", start_worker=False)
        st.close()
        return st
    return PostgresEventStore()


STORE = _init_store()

# ---------------------------------------------------------------------------- Mô hình
MODEL_CONFIGS = {
    "csrnet": {
        "label": "CSRNet",
        "weights": WEIGHTS,
        "default_side": DEFAULT_MAX_SIDE,
        "visualization": "Bản đồ nhiệt",
    },
    "yolo": {
        "label": "YOLO11s",
        "weights": YOLO_WEIGHTS,
        "default_side": YoloHeadDetector.default_side,
        "visualization": "Hộp đầu người",
    },
}
ENGINES = {}
MODEL_ERRORS = {}
for key, config in MODEL_CONFIGS.items():
    if not os.path.isfile(config["weights"]):
        MODEL_ERRORS[key] = f"Thiếu trọng số {os.path.basename(config['weights'])}."
        continue
    try:
        if key == "csrnet":
            ENGINES[key] = InferenceEngine(
                config["weights"],
                DEVICE,
                reserve_cores=3,
                prefer_onnx=os.environ.get("CROWD_NO_ONNX") != "1",
            )
        else:
            ENGINES[key] = YoloHeadDetector(
                config["weights"],
                DEVICE,
                prefer_onnx=os.environ.get("CROWD_NO_ONNX") != "1",
                reserve_cores=3,
            )
    except Exception:
        app.logger.exception("Không nạp được %s", config["label"])
        MODEL_ERRORS[key] = f"Không nạp được {config['label']}. Kiểm tra trọng số và thư viện trên máy chủ."
DEFAULT_MODEL = next(iter(ENGINES), "csrnet")
_infer_lock = threading.Lock()  # nhiều yêu cầu cùng lúc → suy luận lần lượt


def model_meta(key):
    engine = ENGINES.get(key)
    config = MODEL_CONFIGS[key]
    out = {k: config[k] for k in ("label", "default_side", "visualization")}
    out.update(id=key, ok=engine is not None, error=MODEL_ERRORS.get(key), device=DEVICE)
    if engine is None:
        return out
    meta = getattr(engine, "meta", {})
    mae = meta.get("val_mae")
    out.update(
        dataset=meta.get("dataset"),
        epoch=meta.get("epoch"),
        val_mae=None if mae is None else round(float(mae), 2),
        backend=engine.backend,
    )
    if key == "csrnet":
        out.update(engine.info())
    return out


def infer(img_rgb, max_side, model_id):
    with _infer_lock:
        return ENGINES[model_id].run(img_rgb, max_side)


def resolve_side(p, img_rgb):
    if p["auto_res"] and p["model"] == "csrnet":
        with _infer_lock:
            return auto_max_side(ENGINES["csrnet"], img_rgb, AUTO_BUDGET_MS)
    return p["max_side"]


def to_data_url(img_rgb, quality=JPEG_QUALITY):
    h, w = img_rgb.shape[:2]
    s = min(1.0, DISPLAY_SIDE / max(h, w))
    if s < 1:
        img_rgb = cv2.resize(img_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def jpeg_data_url(bgr, quality=VIDEO_JPEG_QUALITY):
    """Nén khung BGR thành data URL JPEG cho luồng video (nhanh: không đổi hệ màu)."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def prepare_video_frame(frame, blur):
    """Thu nhỏ và làm mờ khung gửi tới trình duyệt khi bật chế độ riêng tư."""
    h, w = frame.shape[:2]
    scale = min(1.0, VIDEO_DISPLAY_SIDE / max(h, w))
    if scale < 1:
        size = (max(2, int(w * scale)), max(2, int(h * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return soft_blur(frame) if blur else frame


def _yolo_crowd_mask(boxes):
    """Tạo mặt nạ thưa (120, 160) quanh các hộp đầu YOLO để định vùng tính Optical Flow."""
    if boxes is None or len(boxes) == 0:
        return None
    mh, mw = 120, 160
    mask = np.zeros((mh, mw), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    px = np.rint(arr[:, :4] * [mw - 1, mh - 1, mw - 1, mh - 1]).astype(int)
    for x1, y1, x2, y2 in px:
        cv2.rectangle(mask, (max(0, int(x1)), max(0, int(y1))), (min(mw - 1, int(x2)), min(mh - 1, int(y2))), 1.0, -1)
    return mask


def read_params(src):
    """Đọc & giới hạn tham số người dùng gửi lên (bao gồm ROI đa giác & Optical Flow)."""
    def num(key, default, lo, hi):
        raw = src.get(key, default)
        if raw is None or raw == "":
            v = default
        else:
            try:
                v = float(raw)
                if not np.isfinite(v):
                    v = default
            except (TypeError, ValueError):
                v = default
        return min(hi, max(lo, v))

    def flag(key, default="1"):
        return str(src.get(key, default)).lower() in ("1", "true", "on", "yes")

    crowd_type = src.get("crowd_type", "di chuyển")
    if crowd_type not in THRESHOLDS:
        crowd_type = "di chuyển"
    model_id = src.get("model", DEFAULT_MODEL)
    require_model(model_id)
    default_side = MODEL_CONFIGS[model_id]["default_side"]
    raw_side = num("max_side", default_side, 0, 1600)
    auto_res = raw_side <= 0
    max_side = default_side if auto_res else int(raw_side)

    area_val = num("area", 100.0, 1.0, 100000.0)
    roi_points = parse_roi_points(src.get("roi_points"))
    roi_mode = normalize_roi_mode(src.get("roi_mode", "full"), roi_points)
    has_roi_area = src.get("roi_area") not in (None, "")
    roi_area = num("roi_area", area_val, 1.0, 100000.0)
    effective_area = roi_area if (roi_mode == "roi" and has_roi_area) else area_val

    raw_ct = num("counter_thr", 30.0, 0.05, 95.0)
    counter_thr = raw_ct / 100.0 if raw_ct > 1.0 else raw_ct

    return {
        "model": model_id,
        "area": area_val,
        "crowd_type": crowd_type,
        "blur": flag("blur", "1"),
        "max_side": max_side,
        # 0 = tự chọn độ phân giải theo ngân sách thời gian (không xuống dưới 384px)
        "auto_res": auto_res,
        "fps": num("fps", 15, 1, 30),
        "stride": num("stride", 0.0, 0.0, 5.0),
        "surge_ratio": num("surge_ratio", 50, 10, 200) / 100,
        "surge_window": num("surge_window", 10, 2, 60),
        "min_persist": int(num("min_persist", 3, 1, 10)),
        "max_secs": num("max_secs", 60, 10, 3600),
        "roi_points": roi_points,
        "roi_mode": roi_mode,
        "roi_area": roi_area,
        "effective_area": effective_area,
        "restricted_thr": num("restricted_thr", 3.0, 1.0, 1000.0),
        "enable_flow": flag("enable_flow", "1"),
        "show_vectors": flag("show_vectors", "1"),
        "speed_thr": num("speed_thr", 14.0, 2.0, 80.0),
        "entropy_thr": num("entropy_thr", 0.72, 0.20, 0.98),
        "counter_thr": counter_thr,
    }


def require_model(model_id):
    if model_id not in MODEL_CONFIGS:
        abort(Response(json.dumps({"error": "Mô hình không hợp lệ."}), status=400, mimetype="application/json"))
    if model_id not in ENGINES:
        abort(Response(json.dumps({"error": MODEL_ERRORS[model_id]}), status=503, mimetype="application/json"))


def ext_of(name):
    return os.path.splitext(name or "")[1].lower()


def _int_arg(name, default):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _opt_int_arg(name):
    raw = request.args.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _enrich_alert(item):
    out = dict(item)
    out.setdefault("t", out.get("t_sec", 0.0))
    out.setdefault("type", out.get("alert_type", "general"))
    out.setdefault("source", out.get("source_label", ""))
    out.setdefault("source_name", out.get("source_label", ""))
    out.setdefault("count", out.get("count_val"))
    out.setdefault("density", out.get("density_val"))
    return out


def _enrich_session(item):
    out = dict(item)
    out.setdefault("source", out.get("source_label", ""))
    out.setdefault("source_name", out.get("source_label", ""))
    out.setdefault("kind", out.get("source_kind", ""))
    return out


# ---------------------------------------------------------------------------- Trang & Trạng thái
@app.get("/")
def index():
    return render_template("index.html", contact_email=CONTACT_EMAIL, youtube_url=DEFAULT_YOUTUBE_URL)


@app.get("/api/status")
def status():
    return jsonify({
        "ok": bool(ENGINES),
        "model": model_meta(DEFAULT_MODEL) if ENGINES else None,
        "default_model": DEFAULT_MODEL,
        "models": {key: model_meta(key) for key in MODEL_CONFIGS},
        "weights": os.path.relpath(WEIGHTS, BASE),
        "thresholds": THRESHOLDS,
        "levels": [dict(lv, color=UI_LEVEL_COLORS[i]) for i, lv in enumerate(LEVELS)],
        "video": {
            "fps": 15,
            "auto_budget_ms": AUTO_BUDGET_MS,
            "display_side": VIDEO_DISPLAY_SIDE,
            "max_sides": [0, 256, 384, 512, 768, 1024],
        },
        "database": STORE.status_info(),
    })


# ---------------------------------------------------------------------------- Ảnh tĩnh
@app.post("/api/image")
def api_image():
    p = read_params(request.form)
    f = request.files.get("file")
    if f is None or ext_of(f.filename) not in IMAGE_EXT:
        return jsonify({"error": "Hãy chọn một ảnh JPG, PNG hoặc BMP."}), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Không đọc được ảnh."}), 400
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p["max_side"] = resolve_side(p, rgb)

    t0 = time.time()
    raw_count, raw_ann, resized = infer(rgb, p["max_side"], p["model"])
    roi_res = apply_roi(raw_ann, p["model"], p["roi_mode"], p["roi_points"])
    count = raw_count if roi_res["roi_mode"] != "roi" else roi_res["count"]
    restricted_count = roi_res["restricted_count"]
    ann = roi_res["annotation"]
    r_boxes = roi_res["restricted_boxes"]

    eff_area = p["effective_area"]
    rho = count / eff_area
    raw_lvl = density_level(rho, p["crowd_type"])
    lvl = raw_lvl
    roi_intrusion = bool(roi_res["roi_mode"] == "forbidden" and restricted_count >= p["restricted_thr"])
    if roi_intrusion:
        lvl = min(3, max(lvl + 1, 2))

    shown = privacy_blur(resized) if p["blur"] else resized.copy()
    if p["model"] == "csrnet":
        heat = heat_layer(ann, shown.shape[1], shown.shape[0])
        vis = cv2.addWeighted(shown, 0.5, heat, 0.5, 0)
    else:
        vis = draw_head_boxes(shown.copy(), ann)
    vis = draw_roi_overlay(
        vis,
        roi_res["roi_mode"],
        p["roi_points"],
        is_bgr=False,
        triggered=roi_intrusion,
        restricted_boxes=r_boxes,
    )
    ms = (time.time() - t0) * 1000

    alerts_out = []
    if lvl >= 1 or roi_intrusion:
        img_sid = uuid.uuid4().hex
        src_label = f.filename or "image"
        mdl_label = MODEL_CONFIGS[p["model"]]["label"]
        event_items = []
        if raw_lvl >= 1:
            event_items.append({
                "type": "density",
                "level": lvl,
                "message": f"Mật độ ảnh ở mức {LEVELS[raw_lvl]['name']} ({rho:.2f} người/m²)",
            })
        if roi_intrusion:
            event_items.append({
                "type": "roi_intrusion",
                "level": lvl,
                "message": f"Xâm nhập vùng cấm ({restricted_count:.0f} người ≥ ngưỡng {p['restricted_thr']:.0f})",
            })
        STORE.start_session(
            img_sid,
            source_kind="image",
            source_label=src_label,
            model=p["model"],
            model_label=mdl_label,
            area_m2=eff_area,
            crowd_type=p["crowd_type"],
            roi_mode=roi_res["roi_mode"],
            params=p,
        )
        alerts_out = STORE.record_alerts(
            img_sid,
            t_sec=0.0,
            source_label=src_label,
            model_label=mdl_label,
            count_val=count,
            density_val=rho,
            event_items=event_items,
            frame_bgr=cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            details={"roi_mode": roi_res["roi_mode"], "restricted_count": restricted_count},
        )
        STORE.finish_session(
            img_sid,
            {
                "samples": 1,
                "max_count": count,
                "max_level": lvl,
                "display_fps": None,
                "infer_fps": round(1000.0 / ms, 1) if ms > 0 else None,
                "infer_ms": round(ms, 1),
            },
        )

    return jsonify({
        "count": count,
        "density": rho,
        "level": lvl,
        "ms": ms,
        "overlay": to_data_url(vis),
        "original": None if p["blur"] else to_data_url(resized),
        "max_side": p["max_side"],
        "model": p["model"],
        "model_label": MODEL_CONFIGS[p["model"]]["label"],
        "visualization": MODEL_CONFIGS[p["model"]]["visualization"],
        "roi_mode": roi_res["roi_mode"],
        "restricted_count": restricted_count,
        "roi_intrusion": roi_intrusion,
        "alerts": alerts_out,
    })


# ---------------------------------------------------------------------------- Lịch sử & Ảnh bằng chứng PostgreSQL
@app.get("/api/history/alerts")
def api_history_alerts():
    limit = _int_arg("limit", 100)
    level = _opt_int_arg("level")
    alert_type = (request.args.get("alert_type") or request.args.get("type") or "").strip() or None
    session_id = (request.args.get("session_id") or request.args.get("session") or "").strip() or None
    rows = STORE.list_alerts(limit=limit, level=level, alert_type=alert_type, session_id=session_id)
    return jsonify({
        "ok": True,
        "database": STORE.status_info(),
        "alerts": [_enrich_alert(r) for r in rows],
    })


@app.get("/api/history/sessions")
def api_history_sessions():
    limit = _int_arg("limit", 50)
    rows = STORE.list_sessions(limit=limit)
    return jsonify({
        "ok": True,
        "sessions": [_enrich_session(r) for r in rows],
    })


@app.get("/api/history/sessions/<sid>/timeseries")
def api_history_session_timeseries(sid):
    return jsonify({
        "ok": True,
        "timeseries": STORE.get_session_timeseries(sid),
    })


@app.delete("/api/history")
def api_clear_history():
    return jsonify({"ok": True, **STORE.clear_history()})


@app.get("/api/evidence/<filename>")
def api_evidence(filename):
    path = STORE.evidence_path(filename)
    if path is None:
        return jsonify({"error": "Không tìm thấy ảnh bằng chứng."}), 404
    return send_file(path, mimetype="image/jpeg")


# ---------------------------------------------------------------------------- Video / camera / YouTube Live
JOBS = {}
_jobs_lock = threading.Lock()
JOB_TTL = 600  # phiên chưa được mở luồng sau 10 phút thì bị hủy


def _cleanup_job(job):
    if job.get("tmp") and os.path.exists(job["tmp"]):
        try:
            os.remove(job["tmp"])
        except OSError:
            pass


def _gc_jobs():
    now = time.time()
    with _jobs_lock:
        stale = [k for k, j in JOBS.items() if not j["started"] and now - j["created"] > JOB_TTL]
        for k in stale:
            _cleanup_job(JOBS.pop(k))


@app.post("/api/jobs")
def create_job():
    _gc_jobs()
    p = read_params(request.form)
    source = request.form.get("source", "file")
    job = {"params": p, "stop": False, "started": False, "created": time.time(), "tmp": None}
    if source == "file":
        f = request.files.get("file")
        if f is None or ext_of(f.filename) not in VIDEO_EXT:
            return jsonify({"error": "Hãy chọn video MP4, AVI, MOV hoặc MKV."}), 400
        fd, tmp = tempfile.mkstemp(suffix=ext_of(f.filename))
        with os.fdopen(fd, "wb") as out:
            f.save(out)
        job.update(kind="file", src=tmp, tmp=tmp, source_label=f.filename or "video")
    elif source == "camera":
        url = (request.form.get("url") or "0").strip()
        job.update(kind="camera", src=int(url) if url.isdigit() else url, source_label=f"camera:{url}")
    elif source == "youtube":
        try:
            url = youtube_url(request.form.get("url", ""))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        job.update(kind="youtube", src=url, source_label=f"youtube:{url}")
    else:
        return jsonify({"error": "Nguồn video không hợp lệ."}), 400
    jid = uuid.uuid4().hex
    with _jobs_lock:
        JOBS[jid] = job
    return jsonify({"id": jid, "model": p["model"]})


@app.post("/api/jobs/<jid>/stop")
def stop_job(jid):
    with _jobs_lock:
        job = JOBS.get(jid)
    if job:
        job["stop"] = True
    return jsonify({"ok": True})


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def open_capture(src, is_file):
    """Mở nguồn video. Windows webcam dùng DSHOW mở nhanh; RTSP/HTTP đặt timeout FFmpeg."""
    if is_file:
        return cv2.VideoCapture(src)
    cap = None
    if os.name == "nt" and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        if isinstance(src, str) and src.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
            try:
                cap = open_network_capture(src)
            except Exception:
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
    if cap is not None and hasattr(cap, "set"):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


class _Shared:
    """Trạng thái dùng chung giữa luồng đọc khung, luồng suy luận và luồng hiển thị."""

    __slots__ = (
        "lock",
        "frame",
        "frame_seq",
        "tsec",
        "frame_ring",
        "inferred_frame",
        "annotation",
        "restricted_boxes",
        "flow_vectors",
        "result",
        "alert_records",
        "infer_seq",
        "eof",
        "infer_done",
        "error",
        "infer_ms",
        "infer_fps",
        "max_side",
    )

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None              # khung BGR mới nhất (chưa qua xử lý)
        self.frame_seq = 0
        self.tsec = 0.0
        self.frame_ring = deque(maxlen=8)  # vòng đệm (seq, tsec, frame) phục vụ Optical Flow
        self.inferred_frame = None     # khung hình đi kèm với kết quả suy luận
        self.annotation = None         # bản đồ mật độ CSRNet hoặc hộp YOLO gần nhất
        self.restricted_boxes = None   # hộp đầu người xâm nhập vùng cấm (YOLO)
        self.flow_vectors = []         # danh sách vector luồng chuyển động chuẩn hóa
        self.result = None             # kết quả CrowdMonitor của lần suy luận gần nhất
        self.alert_records = []        # các bản ghi cảnh báo mới phát sinh chờ gửi ở SSE tick kế tiếp
        self.infer_seq = 0
        self.eof = False
        self.infer_done = False
        self.error = None
        self.infer_ms = 0.0
        self.infer_fps = 0.0
        self.max_side = 0


class VideoSession:
    """Ba luồng độc lập cho video/camera/YouTube Live kết hợp ROI đa giác, Optical Flow và PostgreSQL.

    - luồng đọc: giải mã khung theo nhịp nguồn (~0 ms copy vào frame_ring), luôn giữ khung mới nhất;
    - luồng suy luận: chạy CSRNet/YOLO + lọc ROI + Farneback Optical Flow + ghi bất đồng bộ vào STORE;
    - luồng hiển thị: gửi khung nguồn và khung nhận diện theo FPS mục tiêu của giao diện.
    """

    def __init__(self, jid, job):
        self.jid = jid
        self.job = job
        self.p = job["params"]
        self.is_file = job["kind"] == "file"
        self.source_label = job.get("source_label") or (
            f"youtube:{job.get('src', '')}"
            if job.get("kind") == "youtube"
            else (f"camera:{job.get('src', 0)}" if job.get("kind") == "camera" else "video")
        )
        self.s = _Shared()
        self.stop = False
        self.summary = {
            "samples": 0,
            "max_count": 0.0,
            "max_level": 0,
            "stopped": False,
            "display_fps": 0.0,
            "infer_fps": 0.0,
            "infer_ms": 0.0,
            "max_side": 0,
        }

    def _render_ai_bgr(self, bgr_frame, annotation, r_boxes, vectors, *,
                       roi_intrusion=False, panic=False, counter_flow=False):
        """Dựng khung BGR nhận diện AI (làm mờ riêng tư + nhiệt/hộp + ROI + mũi tên Optical Flow)."""
        p = self.p
        small = prepare_video_frame(bgr_frame, p["blur"])
        if p["model"] == "yolo":
            out = draw_head_boxes(small.copy(), annotation)
        else:
            h, w = small.shape[:2]
            heat = cv2.cvtColor(heat_layer(annotation, w, h), cv2.COLOR_RGB2BGR)
            out = cv2.addWeighted(small, 0.45, heat, 0.55, 0)
        if p["roi_mode"] != "full":
            out = draw_roi_overlay(
                out,
                p["roi_mode"],
                p["roi_points"],
                is_bgr=True,
                triggered=bool(roi_intrusion),
                restricted_boxes=r_boxes,
            )
        if p["enable_flow"] and p["show_vectors"] and vectors:
            out = draw_motion_vectors(
                out,
                vectors,
                is_bgr=True,
                panic=bool(panic),
                counter_flow=bool(counter_flow),
            )
        return out

    # -- vòng đọc ------------------------------------------------------------
    def _reader(self, cap, src_fps):
        s, job = self.s, self.job
        period = 1.0 / max(1.0, min(src_fps, 30.0))
        next_t = time.perf_counter()
        idx = 1  # khung đầu đã được đọc khi mở phiên
        error = None
        try:
            while not self.stop and not job["stop"]:
                ok, frame = cap.read()
                if not ok:
                    if job["kind"] == "youtube":
                        error = "Luồng YouTube đã ngừng gửi hình. Hãy thử kết nối lại."
                    break
                idx += 1
                tsec = idx / src_fps if self.is_file else time.perf_counter() - self.t_start
                with s.lock:
                    s.frame, s.tsec, s.frame_seq = frame, tsec, idx
                    s.frame_ring.append((idx, tsec, frame))
                # Chỉ giữ nhịp sleep với tệp video trên đĩa; nguồn trực tiếp (camera/RTSP/YouTube)
                # phải đọc liên tục để xả bộ đệm của OpenCV/FFmpeg, tránh trễ hình tích lũy.
                if self.is_file:
                    next_t += period
                    dt = next_t - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)
                    elif dt < -1.0:      # tụt lại quá xa (máy bận) → bỏ phần bù, không cố đuổi
                        next_t = time.perf_counter()
        except Exception:
            app.logger.exception("Lỗi khi đọc khung hình video")
            error = "Không đọc được khung hình từ nguồn video."
        finally:
            cap.release()
            _cleanup_job(job)
            with s.lock:
                s.eof = True
                if error and not self.stop and not job["stop"]:
                    s.error = error

    # -- vòng suy luận -------------------------------------------------------
    def _infer_worker(self):
        s, p = self.s, self.p
        model_label = MODEL_CONFIGS[p["model"]]["label"]
        mon = CrowdMonitor(
            p["effective_area"],
            p["crowd_type"],
            p["surge_ratio"],
            p["surge_window"],
            min_persist=p["min_persist"],
            restricted_thr=p["restricted_thr"],
        )
        flow_analyzer = (
            MotionAnalyzer(
                speed_thr=p["speed_thr"],
                entropy_thr=p["entropy_thr"],
                counter_thr=p["counter_thr"],
            )
            if p["enable_flow"]
            else None
        )
        last_seq, ema_ms = 0, None
        # "Phân tích mỗi N giây" (thanh trượt trong Cấu hình): giới hạn nhịp suy luận để
        # không chiếm hết CPU. 0 = chạy liền tay, luôn phân tích khung mới nhất.
        stride = max(0.0, p["stride"])
        next_due = 0.0
        while not self.stop and not self.job["stop"]:
            if stride > 0:
                wait = next_due - time.perf_counter()
                if wait > 0:
                    time.sleep(min(wait, 0.05))
                    continue
            prev_pair = None
            with s.lock:
                if s.frame is None or s.frame_seq == last_seq:
                    frame = None
                else:
                    frame, tsec, last_seq = s.frame, s.tsec, s.frame_seq
                    best_cand = None
                    best_err = None
                    for p_seq, p_tsec, p_frame in reversed(s.frame_ring):
                        if p_seq == last_seq:
                            continue
                        dt_cand = tsec - p_tsec
                        if 0.03 <= dt_cand <= 0.30:
                            err = abs(dt_cand - 0.08)
                            if best_err is None or err < best_err:
                                best_err = err
                                best_cand = (p_tsec, p_frame)
                    if best_cand is None and len(s.frame_ring) >= 2:
                        p_seq, p_tsec, p_frame = s.frame_ring[-2]
                        if p_seq != last_seq and tsec > p_tsec:
                            best_cand = (p_tsec, p_frame)
                    prev_pair = best_cand
            if frame is None:
                if s.eof:
                    break
                time.sleep(0.01)
                continue
            next_due = time.perf_counter() + stride
            try:
                t0 = time.perf_counter()
                raw_count, raw_ann, _ = infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self._side, p["model"])
                roi_res = apply_roi(raw_ann, p["model"], p["roi_mode"], p["roi_points"])
                eff_count = raw_count if roi_res["roi_mode"] != "roi" else roi_res["count"]

                vectors = []
                motion_meta = None
                if flow_analyzer is not None and prev_pair is not None:
                    prev_tsec, prev_frame = prev_pair
                    crowd_mask = (
                        roi_res["annotation"]
                        if p["model"] == "csrnet"
                        else _yolo_crowd_mask(roi_res["annotation"])
                    )
                    motion_out = flow_analyzer.update(
                        prev_frame,
                        frame,
                        tsec - prev_tsec,
                        roi_mode=roi_res["roi_mode"],
                        roi_points=p["roi_points"],
                        crowd_mask=crowd_mask,
                    )
                    vectors = motion_out.get("vectors", [])
                    motion_meta = {k: v for k, v in motion_out.items() if k != "vectors"}
                    motion_meta["speed"] = motion_meta.get("mean_speed", 0.0)
                    motion_meta["entropy"] = motion_meta.get("direction_entropy", 0.0)
                    motion_meta["dominant_angle"] = motion_meta.get("dominant_angle_deg")

                ms = (time.perf_counter() - t0) * 1000
                ema_ms = ms if ema_ms is None else 0.6 * ema_ms + 0.4 * ms
                r = mon.update(
                    tsec,
                    eff_count,
                    restricted_count=roi_res["restricted_count"],
                    motion=motion_meta,
                )

                STORE.enqueue_timeseries(
                    self.jid,
                    {
                        "t_sec": r["t"],
                        "count_raw": r["count"],
                        "count_smoothed": r["smoothed"],
                        "density": r["density"],
                        "restricted_count": r["restricted_count"],
                        "mean_speed": motion_meta.get("mean_speed") if motion_meta else None,
                        "direction_entropy": motion_meta.get("direction_entropy") if motion_meta else None,
                        "counter_flow_ratio": motion_meta.get("counter_flow_ratio") if motion_meta else None,
                        "level": r["level"],
                        "surge": r["surge"],
                        "panic": r["panic"],
                        "counter_flow": r["counter_flow"],
                    },
                )

                new_alerts = []
                if r["event_items"]:
                    ann_snap = roi_res["annotation"]
                    rboxes_snap = roi_res["restricted_boxes"]
                    vec_snap = vectors
                    new_alerts = STORE.record_alerts(
                        self.jid,
                        t_sec=tsec,
                        source_label=self.source_label,
                        model_label=model_label,
                        count_val=r["count"],
                        density_val=r["density"],
                        event_items=r["event_items"],
                        render_fn=lambda f=frame, a=ann_snap, rb=rboxes_snap, v=vec_snap, res=r: (
                            self._render_ai_bgr(
                                f,
                                a,
                                rb,
                                v,
                                roi_intrusion=bool(res.get("roi_intrusion")),
                                panic=bool(res.get("panic")),
                                counter_flow=bool(res.get("counter_flow")),
                            )
                        ),
                        details={
                            "motion": motion_meta,
                            "roi_mode": roi_res["roi_mode"],
                            "restricted_count": roi_res["restricted_count"],
                        },
                    )

                with s.lock:
                    s.inferred_frame = frame
                    s.annotation = roi_res["annotation"]
                    s.restricted_boxes = roi_res["restricted_boxes"]
                    s.flow_vectors = vectors
                    s.result = r
                    if new_alerts:
                        s.alert_records.extend(new_alerts)
                    s.infer_seq += 1
                    s.infer_ms, s.infer_fps = ema_ms, 1000.0 / ema_ms if ema_ms else 0.0
                self.summary["samples"] += 1
                self.summary["max_count"] = max(self.summary["max_count"], r["count"])
                self.summary["max_level"] = max(self.summary["max_level"], r["level"])
            except Exception as e:
                with s.lock:
                    s.error = f"Lỗi khi phân tích: {e}"
                break
        with s.lock:
            s.infer_done = True

    # -- vòng hiển thị -------------------------------------------------------
    def events(self):
        s, p = self.s, self.p
        model_id = p["model"]
        model = model_meta(model_id)
        youtube = self.job["kind"] == "youtube"
        self.summary.update(
            backend=model["backend"],
            model=model_id,
            model_label=model["label"],
            target_fps=p["fps"],
        )
        cap, reader, infer_thread = None, None, None
        session_started = False
        session_finished = False
        try:
            src = self.job["src"]
            if youtube:
                yield ": Đang kết nối YouTube\n\n"
                if not self.job["stop"]:
                    src = resolve_youtube_live(src)
                if not self.job["stop"]:
                    # Phân đoạn HLS có thể dài 5 giây và cần thêm thời gian tải/giải mã.
                    cap = open_network_capture(src, open_timeout_ms=18000, read_timeout_ms=12000)
                if self.job["stop"]:
                    self.summary["stopped"] = True
                    yield _sse("done", self.summary)
                    return
            else:
                cap = open_capture(src, self.is_file)
            if not cap.isOpened():
                error = (
                    "Không mở được luồng hình YouTube. Hãy thử kết nối lại."
                    if youtube
                    else "Không mở được nguồn video."
                )
                yield _sse("fail", {"error": error})
                return
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if not self.is_file:
                src_fps = min(max(src_fps, 5), 30)

            ok, first = cap.read()
            if not ok:
                yield _sse("fail", {"error": "Không đọc được khung hình nào."})
                return
            self.t_start = time.perf_counter()
            t_first = 1.0 / src_fps
            with s.lock:
                s.frame, s.frame_seq, s.tsec = first, 1, t_first
                s.frame_ring.append((1, t_first, first))

            # Suy luận sẽ tự đo thời gian thật ở lần chạy đầu tiên (cập nhật qua từng tick),
            # nên không đo trước ở đây: mỗi lần đo tốn ~0,5–1,5 s và làm chậm khung đầu tiên.
            self._side = resolve_side(p, cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
            s.max_side = self._side
            self.summary["max_side"] = self._side

            STORE.start_session(
                self.jid,
                source_kind=self.job["kind"],
                source_label=self.source_label,
                model=model_id,
                model_label=model["label"],
                area_m2=p["effective_area"],
                crowd_type=p["crowd_type"],
                roi_mode=p["roi_mode"],
                params=p,
            )
            session_started = True

            yield _sse(
                "start",
                {
                    "session_id": self.jid,
                    "kind": self.job["kind"],
                    "fps": src_fps,
                    "duration": total / src_fps if self.is_file and total else None,
                    "target_fps": p["fps"],
                    "max_side": self._side,
                    "infer_ms": None,
                    "infer_fps": None,
                    "backend": model["backend"],
                    "model": model_id,
                    "model_label": model["label"],
                    "visualization": model["visualization"],
                    "roi_mode": p["roi_mode"],
                },
            )

            reader = threading.Thread(target=self._reader, args=(cap, src_fps), daemon=True)
            reader.start()
            infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
            infer_thread.start()

            period = 1.0 / max(1.0, p["fps"])
            next_tick = time.perf_counter()
            sent_frame, sent_infer = -1, 0
            fps_marks = deque(maxlen=60)
            box_tracker = BoxFlowTracker()
            cached_heat_bgr = None
            max_secs = None if self.is_file else p["max_secs"]
            while not self.stop and not self.job["stop"]:
                now = time.perf_counter()
                if max_secs and now - self.t_start >= max_secs:
                    break
                if now < next_tick:
                    time.sleep(min(next_tick - now, 0.02))
                    continue
                next_tick = max(next_tick + period, now - period)

                with s.lock:
                    frame, seq, tsec = s.frame, s.frame_seq, s.tsec
                    inferred_frame = s.inferred_frame
                    annotation = s.annotation
                    r_boxes = s.restricted_boxes
                    vectors = s.flow_vectors
                    result, iseq, err = s.result, s.infer_seq, s.error
                    ema_ms, ema_fps, side = s.infer_ms, s.infer_fps, s.max_side
                    eof, infer_done = s.eof, s.infer_done
                    fresh_result = iseq != sent_infer
                    if fresh_result and s.alert_records:
                        tick_alerts = list(s.alert_records)
                        s.alert_records.clear()
                    else:
                        tick_alerts = []

                if err:
                    yield _sse("fail", {"error": err})
                    return
                if frame is None:
                    if eof:
                        break
                    continue
                if max_secs and tsec > max_secs:
                    break
                fresh_frame = seq != sent_frame
                if not fresh_frame and not fresh_result and eof and infer_done:
                    break

                base_bgr = None
                raw_url = None
                if fresh_frame:
                    base_bgr = prepare_video_frame(frame, p["blur"])
                    if p["roi_mode"] != "full":
                        raw_vis = draw_roi_overlay(
                            base_bgr.copy(),
                            p["roi_mode"],
                            p["roi_points"],
                            is_bgr=True,
                            triggered=bool(result and result.get("roi_intrusion")),
                        )
                    else:
                        raw_vis = base_bgr
                    raw_url = jpeg_data_url(raw_vis)
                    sent_frame = seq
                    fps_marks.append(time.perf_counter())

                analyzed_url = None
                if fresh_frame or fresh_result:
                    if base_bgr is None:
                        base_bgr = prepare_video_frame(frame, p["blur"])
                    if result is None:
                        # Hiển thị ngay khung trực tiếp từ giây đầu tiên thay vì chờ lượt AI đầu xong
                        analyzed_url = raw_url if raw_url is not None else jpeg_data_url(base_bgr)
                    else:
                        if p["model"] == "yolo":
                            if fresh_result:
                                cur_boxes, cur_rboxes = box_tracker.sync_from_inference(
                                    inferred_frame, frame, annotation, r_boxes
                                )
                            else:
                                cur_boxes, cur_rboxes = box_tracker.step(frame)
                            out = draw_head_boxes(base_bgr.copy(), cur_boxes)
                            active_rboxes = cur_rboxes
                        else:
                            h, w = base_bgr.shape[:2]
                            if fresh_result or cached_heat_bgr is None or cached_heat_bgr.shape[:2] != (h, w):
                                new_heat = cv2.cvtColor(heat_layer(annotation, w, h), cv2.COLOR_RGB2BGR)
                                if cached_heat_bgr is not None and cached_heat_bgr.shape == new_heat.shape:
                                    cached_heat_bgr = cv2.addWeighted(new_heat, 0.80, cached_heat_bgr, 0.20, 0)
                                else:
                                    cached_heat_bgr = new_heat
                            out = cv2.addWeighted(base_bgr, 0.45, cached_heat_bgr, 0.55, 0)
                            active_rboxes = r_boxes

                        if p["roi_mode"] != "full":
                            out = draw_roi_overlay(
                                out,
                                p["roi_mode"],
                                p["roi_points"],
                                is_bgr=True,
                                triggered=bool(result.get("roi_intrusion")),
                                restricted_boxes=active_rboxes,
                            )
                        if p["enable_flow"] and p["show_vectors"] and vectors:
                            out = draw_motion_vectors(
                                out,
                                vectors,
                                is_bgr=True,
                                panic=bool(result.get("panic")),
                                counter_flow=bool(result.get("counter_flow")),
                            )
                        analyzed_url = jpeg_data_url(out)
                        if fresh_result:
                            sent_infer = iseq

                payload = {
                    "t": result["t"] if result else tsec,
                    "source_t": tsec,
                    "count": result["count"] if result else None,
                    "smoothed": result["smoothed"] if result else None,
                    "density": result["density"] if result else None,
                    "level": result["level"] if result else 0,
                    "surge": result["surge"] if result else False,
                    "restricted_count": result["restricted_count"] if result else 0.0,
                    "roi_intrusion": result["roi_intrusion"] if result else False,
                    "panic": result["panic"] if result else False,
                    "counter_flow": result["counter_flow"] if result else False,
                    "overload_eta_s": result["overload_eta_s"] if result else None,
                    "motion": result["motion"] if result else None,
                    "events": result["events"] if result and fresh_result else [],
                    "alert_records": tick_alerts,
                    "progress": (
                        min(1.0, tsec * src_fps / total)
                        if self.is_file and total
                        else (min(1.0, tsec / max_secs) if max_secs else None)
                    ),
                    "infer_ms": round(ema_ms, 0),
                    "infer_fps": round(ema_fps, 1),
                    "max_side": side,
                    "fresh": fresh_result,
                    "raw_frame": raw_url,
                    "frame": analyzed_url,
                }
                yield _sse("tick", payload)

            self.summary["stopped"] = bool(self.stop or self.job["stop"])
            with s.lock:
                self.summary["infer_ms"] = round(s.infer_ms, 0)
                self.summary["infer_fps"] = round(s.infer_fps, 1)
            if len(fps_marks) > 1:
                span = fps_marks[-1] - fps_marks[0]
                self.summary["display_fps"] = round((len(fps_marks) - 1) / span, 1) if span > 0 else 0.0
            STORE.finish_session(self.jid, self.summary)
            session_finished = True
            yield _sse("done", self.summary)
        except YouTubeStreamError as e:
            if self.job["stop"]:
                self.summary["stopped"] = True
                yield _sse("done", self.summary)
            else:
                yield _sse("fail", {"error": str(e)})
        except Exception as e:  # báo lỗi cho trình duyệt thay vì cắt luồng im lặng
            yield _sse("fail", {"error": f"Lỗi khi phân tích: {e}"})
        finally:
            self.stop = True
            # Sau khi khởi động vòng đọc, chính luồng đó giải phóng cap khi read() kết thúc.
            if reader is not None and reader.ident is not None:
                reader.join(timeout=0.1)
            else:
                if cap is not None:
                    cap.release()
                _cleanup_job(self.job)
            if infer_thread is not None and infer_thread.ident is not None:
                infer_thread.join(timeout=0.1)
            if session_started and not session_finished:
                self.summary["stopped"] = True
                with s.lock:
                    self.summary["infer_ms"] = round(s.infer_ms, 0)
                    self.summary["infer_fps"] = round(s.infer_fps, 1)
                STORE.finish_session(self.jid, self.summary)
            with _jobs_lock:
                JOBS.pop(self.jid, None)


def run_job(jid, job):
    return VideoSession(jid, job).events()


@app.get("/api/jobs/<jid>/events")
def job_events(jid):
    with _jobs_lock:
        job = JOBS.get(jid)
        if job is None or job["started"]:
            return jsonify({"error": "Phiên không tồn tại hoặc đã chạy."}), 404
        job["started"] = True
    return Response(
        run_job(jid, job),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Tệp quá lớn (tối đa 500 MB)."}), 413


if __name__ == "__main__":
    host = os.environ.get("CROWD_HOST", "127.0.0.1")
    port = int(os.environ.get("CROWD_PORT", "5000"))
    names = ", ".join(MODEL_CONFIGS[key]["label"] for key in ENGINES)
    db_info = STORE.status_info()
    db_text = (
        f"PostgreSQL: {db_info['database']}@{db_info['host']} (Sẵn sàng)"
        if db_info.get("ok")
        else f"PostgreSQL: ngoại tuyến ({db_info.get('error') or 'tắt'})"
    )
    print(f"Mô hình: {names or 'chưa nạp được mô hình'} · {DEVICE} · {db_text} · http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)
