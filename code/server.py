"""Web app (Flask) đếm người & cảnh báo đám đông bất thường.

Chạy:   python server.py            → mở http://127.0.0.1:5000
Biến môi trường (tùy chọn):
    CROWD_WEIGHTS   đường dẫn trọng số (mặc định weights/csrnet_partB.pth)
    CROWD_HOST      địa chỉ lắng nghe (mặc định 127.0.0.1 — chỉ máy này truy cập được)
    CROWD_PORT      cổng (mặc định 5000)
    CROWD_THREADS   số luồng CPU cho suy luận (mặc định min(12, số nhân logic))
    CROWD_NO_ONNX   đặt "1" để tắt ONNX Runtime, chỉ dùng torch/TorchScript
    CONTACT_EMAIL   email hiện ở thanh liên hệ (bấm để sao chép); để trống thì ẩn

API:
    GET  /api/status                 trạng thái mô hình, backend suy luận, ngưỡng, các mức cảnh báo
    POST /api/image                  phân tích một ảnh (multipart: file + tham số)
    POST /api/jobs                   tạo phiên phân tích video/camera → {id}
    GET  /api/jobs/<id>/events       luồng kết quả theo thời gian (Server-Sent Events)
    POST /api/jobs/<id>/stop         dừng phiên

Luồng video/camera chạy ba vòng độc lập (xem VideoSession): đọc khung → suy luận → hiển thị.
Nhờ vậy khung hình vẫn phát đủ FPS mục tiêu trong khi CSRNet (vốn chậm trên CPU) chỉ cập nhật
bản đồ nhiệt khi tính xong; không còn hiện tượng video bị giật theo tốc độ suy luận.

Ảnh/video chỉ được xử lý trong bộ nhớ (video tải lên được ghi tạm rồi xóa ngay khi xong);
chỉ số liệu thống kê được trả về trình duyệt.
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
from flask import Flask, Response, abort, jsonify, render_template, request

from crowd_core import (LEVELS, THRESHOLDS, CrowdMonitor, InferenceEngine, auto_max_side,
                        density_level, heat_layer, load_model, predict, privacy_blur, soft_blur)

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("CROWD_WEIGHTS", os.path.join(BASE, "weights", "csrnet_partB.pth"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DISPLAY_SIDE = 960          # cạnh dài tối đa của ảnh tĩnh gửi về trình duyệt
VIDEO_DISPLAY_SIDE = 880    # cạnh dài khung hình video gửi về (nhỏ hơn → nhiều FPS hơn)
JPEG_QUALITY = 72
VIDEO_JPEG_QUALITY = 62
AUTO_BUDGET_MS = 800        # chế độ tự chọn độ phân giải: ngân sách thời gian cho 1 lần suy luận
DEFAULT_MAX_SIDE = 512      # mặc định = kích thước ô cắt lúc huấn luyện (số đếm đúng nhất)

# Màu mức cảnh báo theo bảng màu đất của giao diện (Xanh, Vàng, Cam, Đỏ)
UI_LEVEL_COLORS = ["#325825", "#A8741A", "#C2531F", "#A3261A"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# ---------------------------------------------------------------------------- Mô hình
MODEL = None
ENGINE = None
if os.path.exists(WEIGHTS):
    MODEL = load_model(WEIGHTS, DEVICE)  # (model, density_scale, meta)
    try:
        ENGINE = InferenceEngine(WEIGHTS, DEVICE, prefer_onnx=os.environ.get("CROWD_NO_ONNX") != "1",
                                 reserve_cores=3)
        MODEL = (ENGINE.model, ENGINE.density_scale, ENGINE.meta)
    except Exception as e:  # thiếu onnxruntime... vẫn chạy được bằng torch
        print(f"[cảnh báo] không khởi tạo được InferenceEngine ({e}); dùng torch thuần")
        ENGINE = None
_infer_lock = threading.Lock()  # nhiều yêu cầu cùng lúc → suy luận lần lượt


def model_meta():
    if MODEL is None:
        return None
    meta = MODEL[2]
    mae = meta.get("val_mae")
    out = {"dataset": str(meta.get("dataset", "CSRNet")), "epoch": meta.get("epoch", "?"),
           "val_mae": None if mae is None else round(float(mae), 2), "device": DEVICE}
    if ENGINE is not None:
        out.update(ENGINE.info())
    return out


def infer(img_rgb, max_side):
    """Một lần suy luận, dùng InferenceEngine (nhanh) nếu có."""
    if ENGINE is not None:
        with _infer_lock:
            return ENGINE.run(img_rgb, max_side)
    model, dscale, _ = MODEL
    with _infer_lock:
        return predict(model, img_rgb, dscale, DEVICE, max_side)


def analyse(img_rgb, p):
    count, den, img = infer(img_rgb, p["max_side"])
    shown = privacy_blur(img) if p["blur"] else img
    return count, heat_layer(den, shown.shape[1], shown.shape[0]), shown, img

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


def read_params(src):
    """Đọc & giới hạn tham số người dùng gửi lên."""
    def num(key, default, lo, hi):
        try:
            v = float(src.get(key, default))
        except (TypeError, ValueError):
            v = default
        return min(hi, max(lo, v))

    crowd_type = src.get("crowd_type", "di chuyển")
    if crowd_type not in THRESHOLDS:
        crowd_type = "di chuyển"
    raw_side = num("max_side", DEFAULT_MAX_SIDE, 0, 1600)
    auto_res = raw_side <= 0
    max_side = DEFAULT_MAX_SIDE if auto_res else int(raw_side)
    return {
        "area": num("area", 100.0, 1.0, 100000.0),
        "crowd_type": crowd_type,
        "blur": str(src.get("blur", "1")).lower() in ("1", "true", "on", "yes"),
        "max_side": max_side,
        # 0 = tự chọn độ phân giải theo ngân sách thời gian (không xuống dưới 384px)
        "auto_res": auto_res,
        "fps": num("fps", 15, 1, 30),
        "stride": num("stride", 0.5, 0.0, 5.0),        "surge_ratio": num("surge_ratio", 50, 10, 200) / 100,
        "surge_window": num("surge_window", 10, 2, 60),
        "min_persist": int(num("min_persist", 3, 1, 10)),
        "max_secs": num("max_secs", 60, 10, 3600),
    }



def require_model():
    if MODEL is None:
        abort(Response(json.dumps({"error": f"Không tìm thấy trọng số {WEIGHTS}. Hãy chép csrnet_partB.pth "
                                             "vào thư mục weights/."}, ensure_ascii=False),
                       status=503, mimetype="application/json"))


def ext_of(name):
    return os.path.splitext(name or "")[1].lower()


# ---------------------------------------------------------------------------- Trang
@app.get("/")
def index():
    return render_template("index.html", contact_email=CONTACT_EMAIL)


@app.get("/api/status")
def status():
    return jsonify({
        "ok": MODEL is not None,
        "model": model_meta(),
        "weights": os.path.relpath(WEIGHTS, BASE),
        "thresholds": THRESHOLDS,
        "levels": [dict(lv, color=UI_LEVEL_COLORS[i]) for i, lv in enumerate(LEVELS)],
        "video": {"fps": 15, "auto_budget_ms": AUTO_BUDGET_MS, "display_side": VIDEO_DISPLAY_SIDE,
                  "max_sides": [0, 256, 384, 512, 768, 1024]},
    })


# ---------------------------------------------------------------------------- Ảnh
@app.post("/api/image")
def api_image():
    require_model()
    f = request.files.get("file")
    if f is None or ext_of(f.filename) not in IMAGE_EXT:
        return jsonify({"error": "Hãy chọn một ảnh JPG, PNG hoặc BMP."}), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Không đọc được ảnh."}), 400
    p = read_params(request.form)
    if p["auto_res"]:
        p["max_side"] = auto_max_side(ENGINE, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), AUTO_BUDGET_MS) \
            if ENGINE is not None else 1024
    t0 = time.time()
    count, heat, shown, resized = analyse(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), p)
    ms = (time.time() - t0) * 1000
    vis = cv2.addWeighted(shown, 0.5, heat, 0.5, 0)
    rho = count / p["area"]
    return jsonify({
        "count": count, "density": rho, "level": density_level(rho, p["crowd_type"]), "ms": ms,
        "overlay": to_data_url(vis),
        "original": None if p["blur"] else to_data_url(resized),
        "max_side": p["max_side"],
    })



# ---------------------------------------------------------------------------- Video / camera
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
    require_model()
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
        job.update(kind="file", src=tmp, tmp=tmp)
    else:
        url = (request.form.get("url") or "0").strip()
        job.update(kind="camera", src=int(url) if url.isdigit() else url)
    jid = uuid.uuid4().hex
    with _jobs_lock:
        JOBS[jid] = job
    return jsonify({"id": jid})


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
    """Mở nguồn video. Trên Windows, camera dùng DSHOW: mở nhanh hơn MSMF ~5 lần."""
    if is_file:
        return cv2.VideoCapture(src)
    if os.name == "nt" and isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(src)


class _Shared:
    """Trạng thái dùng chung giữa luồng đọc khung, luồng suy luận và luồng hiển thị."""

    __slots__ = ("lock", "frame", "frame_seq", "tsec", "den", "result", "infer_seq",
                 "eof", "error", "infer_ms", "infer_fps", "max_side")

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None      # khung BGR mới nhất (chưa qua xử lý)
        self.frame_seq = 0
        self.tsec = 0.0
        self.den = None        # bản đồ mật độ 1/8 của lần suy luận gần nhất
        self.result = None     # kết quả CrowdMonitor của lần suy luận gần nhất
        self.infer_seq = 0
        self.eof = False
        self.error = None
        self.infer_ms = 0.0
        self.infer_fps = 0.0
        self.max_side = 0


class VideoSession:
    """Ba luồng độc lập cho video/camera.

    - luồng đọc: giải mã khung theo nhịp của nguồn (cost rất thấp, ~300 fps), luôn giữ khung mới nhất;
    - luồng suy luận: chạy CSRNet trên khung mới nhất ngay khi rảnh (chậm hơn nhiều, ~1–3 fps trên CPU);
    - luồng hiển thị: phát khung ở đúng FPS mục tiêu, ghép bản đồ nhiệt gần nhất vào.

    Nhờ tách rời, FPS hiển thị không còn bị kéo xuống theo tốc độ suy luận — trước đây mỗi
    khung phải chờ suy luận xong nên video chỉ chạy ~0.3–1 fps.
    """

    def __init__(self, jid, job):
        self.jid = jid
        self.job = job
        self.p = job["params"]
        self.is_file = job["kind"] == "file"
        s = _Shared()
        self.s = s
        self.stop = False
        self.summary = {"samples": 0, "max_count": 0.0, "max_level": 0, "stopped": False,
                        "display_fps": 0.0, "infer_fps": 0.0, "infer_ms": 0.0, "max_side": 0}

    # -- vòng đọc ------------------------------------------------------------
    def _reader(self, cap, src_fps):
        s, job = self.s, self.job
        period = 1.0 / max(1.0, min(src_fps, 30.0))
        next_t = time.perf_counter()
        idx = 0
        while not self.stop and not job["stop"]:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            tsec = idx / src_fps if self.is_file else time.perf_counter() - self.t_start
            with s.lock:
                s.frame, s.tsec, s.frame_seq = frame, tsec, idx
            next_t += period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            elif dt < -1.0:      # tụt lại quá xa (máy bận) → bỏ phần bù, không cố đuổi
                next_t = time.perf_counter()
        with s.lock:
            s.eof = True

    # -- vòng suy luận -------------------------------------------------------
    def _infer_worker(self):
        s = self.s
        mon = CrowdMonitor(self.p["area"], self.p["crowd_type"], self.p["surge_ratio"],
                           self.p["surge_window"], min_persist=self.p["min_persist"])
        last_seq, ema_ms = 0, None
        # "Phân tích mỗi N giây" (thanh trượt trong Cấu hình): giới hạn nhịp suy luận để
        # không chiếm hết CPU. 0 = chạy liền tay, luôn phân tích khung mới nhất.
        stride = max(0.0, self.p["stride"])
        next_due = 0.0
        while not self.stop and not self.job["stop"]:
            with s.lock:
                if s.frame is None or s.frame_seq == last_seq:
                    frame = None
                else:
                    frame, tsec, last_seq = s.frame, s.tsec, s.frame_seq
            if frame is None:
                if s.eof:
                    return
                time.sleep(0.01)
                continue
            if stride > 0:
                wait = next_due - time.perf_counter()
                if wait > 0:
                    time.sleep(min(wait, 0.05))
                    continue
                next_due = time.perf_counter() + stride
            try:
                t0 = time.perf_counter()
                count, den, _ = infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self._side)
                ms = (time.perf_counter() - t0) * 1000
                ema_ms = ms if ema_ms is None else 0.6 * ema_ms + 0.4 * ms
                r = mon.update(tsec, count)
                with s.lock:
                    s.den, s.result = den, r
                    s.infer_seq += 1
                    s.infer_ms, s.infer_fps = ema_ms, 1000.0 / ema_ms if ema_ms else 0.0
                self.summary["samples"] += 1
                self.summary["max_count"] = max(self.summary["max_count"], count)
                self.summary["max_level"] = max(self.summary["max_level"], r["level"])
            except Exception as e:
                with s.lock:
                    s.error = f"Lỗi khi phân tích: {e}"
                return


    # -- vòng hiển thị -------------------------------------------------------
    def events(self):
        s, p = self.s, self.p
        cap = open_capture(self.job["src"], self.is_file)
        if not cap.isOpened():
            yield _sse("fail", {"error": "Không mở được nguồn video."})
            return
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if not self.is_file:
                src_fps = min(max(src_fps, 5), 30)
            self.t_start = time.perf_counter()

            ok, first = cap.read()
            if not ok:
                yield _sse("fail", {"error": "Không đọc được khung hình nào."})
                return
            with s.lock:
                s.frame, s.frame_seq, s.tsec = first, 1, 1.0 / src_fps

            # Suy luận sẽ tự đo thời gian thật ở lần chạy đầu tiên (cập nhật qua từng tick),
            # nên không đo trước ở đây: mỗi lần đo tốn ~0,5–1,5 s và làm chậm khung đầu tiên.
            self._side = p["max_side"]
            if p["auto_res"] and ENGINE is not None:
                self._side = auto_max_side(ENGINE, cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
                                           AUTO_BUDGET_MS)
            s.max_side = self._side
            self.summary.update(max_side=self._side,
                                backend=(ENGINE.backend if ENGINE is not None else "torch"),
                                target_fps=p["fps"])

            yield _sse("start", {"kind": self.job["kind"], "fps": src_fps,
                                 "duration": total / src_fps if total else None,
                                 "target_fps": p["fps"], "max_side": self._side,
                                 "infer_ms": None, "infer_fps": None,
                                 "backend": ENGINE.backend if ENGINE is not None else "torch"})

            threads = [threading.Thread(target=self._reader, args=(cap, src_fps), daemon=True),
                       threading.Thread(target=self._infer_worker, daemon=True)]
            for th in threads:
                th.start()

            dw = dh = 0
            period = 1.0 / max(1.0, p["fps"])
            next_tick = time.perf_counter()
            sent_frame, sent_infer = -1, -1
            heat, heat_size = None, None
            last_t = 0.0
            last_frame_url = None      # JPEG đã nén của khung gần nhất (dùng lại khi không có gì mới)
            fps_marks = deque(maxlen=60)
            max_secs = None if self.is_file else p["max_secs"]
            while not self.stop and not self.job["stop"]:
                now = time.perf_counter()
                if now < next_tick:
                    time.sleep(min(next_tick - now, 0.02))
                    continue
                next_tick = max(next_tick + period, now - period)

                with s.lock:
                    frame, seq, tsec = s.frame, s.frame_seq, s.tsec
                    den, result, iseq, err = s.den, s.result, s.infer_seq, s.error
                    ema_ms, ema_fps, side = s.infer_ms, s.infer_fps, s.max_side
                    eof = s.eof
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
                fresh_result = iseq != sent_infer
                if not fresh_frame and not fresh_result and eof:
                    break

                if fresh_frame or fresh_result or last_frame_url is None:
                    # khung hiển thị: thu nhỏ + (tùy chọn) làm mờ, rồi ghép bản đồ nhiệt đã tính sẵn
                    h, w = frame.shape[:2]
                    sc = min(1.0, VIDEO_DISPLAY_SIDE / max(h, w))
                    dw, dh = (max(2, int(w * sc) // 2 * 2), max(2, int(h * sc) // 2 * 2)) if sc < 1 else (w, h)
                    small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if sc < 1 else frame.copy()
                    if p["blur"]:
                        small = soft_blur(small)
                    if fresh_result and den is not None:
                        heat = cv2.cvtColor(heat_layer(den, dw, dh), cv2.COLOR_RGB2BGR)
                        heat_size = (dw, dh)
                    out = cv2.addWeighted(small, 0.45, heat, 0.55, 0) \
                        if heat is not None and heat_size == (dw, dh) else small
                    last_frame_url = jpeg_data_url(out)
                    new_image = True
                else:
                    new_image = False   # khung không đổi → không gửi lại ảnh, chỉ giữ nhịp phát

                sent_frame, sent_infer = seq, iseq
                if result is not None:
                    last_t = result["t"]
                fps_marks.append(time.perf_counter())
                payload = {"t": last_t if result else tsec,
                           "count": result["count"] if result else None,
                           "smoothed": result["smoothed"] if result else None,
                           "density": result["density"] if result else None,
                           "level": result["level"] if result else 0,
                           "surge": result["surge"] if result else False,
                           "events": result["events"] if result else [],
                           "progress": (min(1.0, tsec * src_fps / total) if self.is_file and total else
                                        (min(1.0, tsec / max_secs) if max_secs else None)),
                           "infer_ms": round(ema_ms, 0), "infer_fps": round(ema_fps, 1),
                           "max_side": side, "fresh": fresh_result,
                           "frame": last_frame_url if new_image else None}
                yield _sse("tick", payload)

            self.summary["stopped"] = bool(self.stop or self.job["stop"])
            with s.lock:
                self.summary["infer_ms"] = round(s.infer_ms, 0)
                self.summary["infer_fps"] = round(s.infer_fps, 1)
            if len(fps_marks) > 1:
                span = fps_marks[-1] - fps_marks[0]
                self.summary["display_fps"] = round((len(fps_marks) - 1) / span, 1) if span > 0 else 0.0
            yield _sse("done", self.summary)
        except Exception as e:  # báo lỗi cho trình duyệt thay vì cắt luồng im lặng
            yield _sse("fail", {"error": f"Lỗi khi phân tích: {e}"})
        finally:
            self.stop = True
            time.sleep(0.05)
            cap.release()
            _cleanup_job(self.job)
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
    return Response(run_job(jid, job), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Tệp quá lớn (tối đa 500 MB)."}), 413


if __name__ == "__main__":
    host = os.environ.get("CROWD_HOST", "127.0.0.1")
    port = int(os.environ.get("CROWD_PORT", "5000"))
    print(f"Mô hình: {'đã nạp' if MODEL else 'CHƯA có trọng số'} · {DEVICE} · http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)
