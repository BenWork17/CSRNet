"""Thành phần dùng chung cho notebook huấn luyện và app demo.

- CSRNet: mạng ước lượng bản đồ mật độ (Li et al., CVPR 2018).
- Tiền xử lý / suy luận / vẽ bản đồ nhiệt.
- InferenceEngine: bọc mô hình với các tối ưu suy luận (TorchScript + channels_last,
  ONNX Runtime, chọn độ phân giải theo ngân sách thời gian) cho luồng video/camera.
- CrowdMonitor: luật cảnh báo đám đông bất thường (quá tải mật độ + tăng đột biến).
"""
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DOWNSAMPLE = 8  # CSRNet xuất bản đồ mật độ ở 1/8 độ phân giải ảnh vào


# ----------------------------------------------------------------------------
# Mô hình
# ----------------------------------------------------------------------------
def _make_layers(cfg, in_channels=3, dilation=1):
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_channels, v, 3, padding=dilation, dilation=dilation),
                       nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    FRONTEND = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
    BACKEND = [512, 512, 512, 256, 128, 64]

    def __init__(self, load_vgg=False):
        super().__init__()
        self.frontend = _make_layers(self.FRONTEND)                      # 10 lớp conv đầu của VGG-16
        self.backend = _make_layers(self.BACKEND, 512, dilation=2)       # 6 lớp conv giãn (dilation=2)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        for m in list(self.backend.modules()) + [self.output]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        if load_vgg:
            import torchvision
            vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self.frontend.load_state_dict(vgg.features[:23].state_dict())

    def forward(self, x):
        return self.output(self.backend(self.frontend(x)))


def load_model(weights_path, device="cpu"):
    """Đọc checkpoint do notebook xuất ra. Trả về (model, density_scale, meta)."""
    # weights_only=True: chỉ giải nén tensor/kiểu cơ bản, không thực thi mã pickle tùy ý trong file trọng số
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model = CSRNet(load_vgg=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, float(ckpt.get("density_scale", 1.0)), {k: v for k, v in ckpt.items() if k != "model"}


# ----------------------------------------------------------------------------
# Tiền xử lý & suy luận
# ----------------------------------------------------------------------------
def resize_for_model(img_rgb, max_side=1024):
    """Thu nhỏ để cạnh dài <= max_side và hai chiều chia hết cho 8."""
    h, w = img_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    nh = max(DOWNSAMPLE, int(h * s) // DOWNSAMPLE * DOWNSAMPLE)
    nw = max(DOWNSAMPLE, int(w * s) // DOWNSAMPLE * DOWNSAMPLE)
    if (nh, nw) != (h, w):
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return img_rgb


def to_tensor(img_rgb):
    x = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


@torch.no_grad()
def predict(model, img_rgb, density_scale=1.0, device="cpu", max_side=1024):
    """Trả về (số người ước lượng, bản đồ mật độ 1/8, ảnh đã resize)."""
    img = resize_for_model(img_rgb, max_side)
    den = model(to_tensor(img).to(device))[0, 0].float().cpu().numpy() / density_scale
    den = np.clip(den, 0, None)
    return float(den.sum()), den, img



def density_overlay(img_rgb, den, alpha=0.5):
    h, w = img_rgb.shape[:2]
    d = cv2.resize(den, (w, h), interpolation=cv2.INTER_CUBIC)
    d = np.clip(d / (d.max() + 1e-8), 0, 1)
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)


def heat_layer(den, w, h):
    """Bản đồ nhiệt đã tô màu ở kích thước hiển thị (w, h) — tính 1 lần cho mỗi kết quả suy luận."""
    d = cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR)
    d = np.clip(d / (d.max() + 1e-8), 0, 1)
    heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def privacy_blur(img_rgb, ksize=31):
    """Làm mờ ảnh hiển thị: người xem chỉ thấy phân bố đám đông, không nhận diện được cá nhân."""
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)


def soft_blur(bgr, alpha=0.35, factor=6):
    """Làm mờ nhanh cho luồng video: thu nhỏ rồi phóng lại (rẻ hơn GaussianBlur ~2.5 lần).

    Đủ mạnh để không nhận diện được cá nhân nhưng giữ được bố cục đám đông; dùng cho
    từng khung hình video nên chi phí quan trọng hơn chất lượng mờ.
    """
    h, w = bgr.shape[:2]
    if w // factor < 2 or h // factor < 2:
        return cv2.GaussianBlur(bgr, (0, 0), 8)
    q = cv2.resize(bgr, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    q = cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(bgr, alpha, q, 1 - alpha, 0)


# ----------------------------------------------------------------------------
# Bộ máy suy luận cho luồng video/camera
# ----------------------------------------------------------------------------
class InferenceEngine:
    """Bọc CSRNet với các tối ưu suy luận, chọn backend nhanh nhất có thể.

    Thứ tự ưu tiên: ONNX Runtime (nhanh nhất trên CPU, cùng kết quả) → TorchScript
    (`torch.jit.freeze`) + channels_last → torch thuần. `run()` trả về cùng định dạng
    như `predict()` nên phần còn lại của app không phải đổi.
    """

    ONNX_OPSET = 17

    def __init__(self, weights_path, device="cpu", threads=None, prefer_onnx=True, onnx_path=None,
                 reserve_cores=0):
        start = time.perf_counter()
        self.device = device
        self.model, self.density_scale, self.meta = load_model(weights_path, device)
        self.weights_path = weights_path
        self.onnx_path = onnx_path or os.path.splitext(weights_path)[0] + ".onnx"
        cores = os.cpu_count() or 4
        # chừa `reserve_cores` nhân cho luồng đọc/ghép ảnh/nén JPEG, nếu không vòng hiển thị
        # sẽ bị đói CPU (bị đẩy xuống ~11 FPS thay vì 15 FPS) khi suy luận chạy hết công suất
        self.threads = threads or max(1, min(12, cores - max(0, reserve_cores)))
        self.backend = "torch"
        self._sess = None
        self._input_name = None
        self._torch_model = self.model

        if device == "cpu":
            try:
                torch.set_num_threads(self.threads)
            except Exception:
                pass
            try:  # TorchScript: bỏ qua autograd/bản ghi đồ thị → nhanh hơn ~8%
                frozen = torch.jit.freeze(torch.jit.script(self.model.eval()))
                with torch.inference_mode():
                    frozen(torch.zeros(1, 3, 64, 64))
                self._torch_model = frozen.to(memory_format=torch.channels_last)
                self.backend = "torchscript"
            except Exception:
                self._torch_model = self.model

        if prefer_onnx and device == "cpu":
            try:
                self._init_onnx()
            except Exception as e:  # thiếu onnxruntime hoặc export lỗi → dùng torch
                self.onnx_error = f"{type(e).__name__}: {e}"
                self._sess = None
        self.load_ms = (time.perf_counter() - start) * 1000

    # -- ONNX ---------------------------------------------------------------
    def _export_onnx(self):
        """Xuất ONNX shape động (dùng lại file cache nếu đã có và còn mới)."""
        src = getattr(self, "weights_path", None)
        if (src and os.path.exists(self.onnx_path)
                and os.path.getmtime(self.onnx_path) >= os.path.getmtime(src)):
            return
        tmp = self.onnx_path + ".tmp"
        torch.onnx.export(
            self._torch_model if self.backend == "torchscript" else self.model,
            torch.zeros(1, 3, 256, 256), tmp,
            input_names=["input"], output_names=["output"], opset_version=self.ONNX_OPSET,
            dynamo=False, do_constant_folding=True,
            dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
        )
        os.replace(tmp, self.onnx_path)

    def _init_onnx(self):
        import onnxruntime as ort

        self._export_onnx()
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(self.onnx_path, so, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name
        self.backend = "onnx"

    # -- Suy luận -----------------------------------------------------------
    def run(self, img_rgb, max_side=1024):
        """Trả về (số người, bản đồ mật độ 1/8, ảnh đã resize) — như `predict()`."""
        img = resize_for_model(img_rgb, max_side)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
        if self._sess is not None:
            den = self._sess.run(None, {self._input_name: x})[0][0, 0]
        else:
            with torch.inference_mode():
                t = torch.from_numpy(x)
                if self.backend == "torchscript":
                    t = t.contiguous(memory_format=torch.channels_last)
                out = self._torch_model(t if self.device == "cpu" else t.to(self.device))
            den = out[0, 0].float().cpu().numpy()
        den = np.clip(den / self.density_scale, 0, None)
        return float(den.sum()), den, img

    def benchmark(self, img_rgb, max_side, reps=2):
        """Đo thời gian suy luận trung bình (ms) ở một cạnh dài cho trước."""
        self.run(img_rgb, max_side)
        t0 = time.perf_counter()
        for _ in range(reps):
            self.run(img_rgb, max_side)
        return (time.perf_counter() - t0) / reps * 1000

    def info(self):
        return {"backend": self.backend, "threads": self.threads, "device": self.device,
                "load_ms": round(self.load_ms, 0),
                "onnx": self.onnx_path if self._sess is not None else None}


class YoloHeadDetector:
    """YOLO11s đầu người với cấu hình đã chọn trên validation Part B."""

    backend = "pytorch"
    default_side = 1024

    def __init__(self, weights_path, device):
        from ultralytics import YOLO

        self.model = YOLO(weights_path, task="detect")
        if list(self.model.names.values()) != ["head"]:
            raise ValueError("Cần trọng số YOLO được huấn luyện cho một lớp head.")
        self.device = device

    def run(self, img_rgb, max_side):
        result = self.model.predict(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), imgsz=max(32, max_side),
            conf=0.20, iou=0.65, max_det=1200, rect=True,
            device=0 if self.device == "cuda" else "cpu",
            half=self.device == "cuda", verbose=False, save=False,
        )[0]
        boxes = result.boxes.xyxyn.float().cpu().numpy()
        return len(boxes), boxes, img_rgb


def draw_head_boxes(image, boxes):
    """Vẽ hộp chuẩn hóa lên ảnh đang hiển thị."""
    h, w = image.shape[:2]
    pixels = np.rint(boxes * [w - 1, h - 1, w - 1, h - 1]).astype(int)
    for x1, y1, x2, y2 in pixels:
        cv2.rectangle(image, (x1, y1), (x2, y2), (86, 235, 86), 2, cv2.LINE_AA)
    return image


def auto_max_side(engine, frame_rgb, budget_ms, candidates=(1024, 768, 640, 512, 448, 384)):
    """Chọn cạnh dài lớn nhất mà một lần suy luận vẫn kịp `budget_ms`.

    Chỉ đo một lần ở độ phân giải tham chiếu rồi ngoại suy theo diện tích (chi phí tỉ lệ
    thuận với số điểm ảnh) — đo từng mức sẽ tốn vài giây mỗi lần mở phiên.

    Sàn là 384px: model được huấn luyện trên ô cắt 512×512, hạ thấp hơn nữa làm đầu người
    nhỏ quá mức so với lúc học và số đếm sai lệch rất mạnh (đo trên hajj.jpg: 16 người ở
    256px so với 813 người ở 512px). Thà chậm hơn còn hơn đếm sai.
    """
    h, w = frame_rgb.shape[:2]
    ref = min(256, max(64, min(h, w)))
    ref = max(DOWNSAMPLE, ref // DOWNSAMPLE * DOWNSAMPLE)
    ms_ref = engine.benchmark(frame_rgb, ref, reps=1)
    if ms_ref <= 0:
        return 512
    # ms(s) ≈ ms_ref * (s/ref)²  →  s = ref * sqrt(budget/ms_ref)
    best = ref * (budget_ms / ms_ref) ** 0.5
    side = min(candidates, key=lambda c: abs(c - min(candidates[0], max(candidates[-1], best))))
    # xác nhận lại một lần (ngoại suy có thể lệch do đệm/nhánh CPU)
    while side > candidates[-1] and engine.benchmark(frame_rgb, side, reps=1) > budget_ms:
        side = next(c for c in candidates if c < side)
    return side



# ----------------------------------------------------------------------------
# Luật cảnh báo bất thường
# ----------------------------------------------------------------------------
LEVELS = [
    {"name": "XANH", "label": "Bình thường", "color": "#2e7d32", "action": "Giám sát thông thường"},
    {"name": "VÀNG", "label": "Chú ý", "color": "#f9a825", "action": "Theo dõi sát, chuẩn bị phương án phân luồng"},
    {"name": "CAM", "label": "Cảnh báo", "color": "#ef6c00", "action": "Hạn chế người vào, mở thêm lối thoát, điều lực lượng"},
    {"name": "ĐỎ", "label": "Nguy hiểm", "color": "#c62828", "action": "Dừng dòng người vào, giải tỏa khẩn cấp, thông báo loa"},
]
# Ngưỡng mật độ (người/m²) chuyển sang mức Vàng / Cam / Đỏ — tổng hợp từ Fruin và Still.
THRESHOLDS = {"di chuyển": (2.0, 3.0, 4.0), "đứng": (3.0, 4.0, 5.0)}


def density_level(rho, crowd_type="di chuyển"):
    return int(sum(rho >= t for t in THRESHOLDS[crowd_type]))


class CrowdMonitor:
    """Theo dõi chuỗi số đếm theo thời gian và phát cảnh báo.

    - Quá tải: mật độ rho = C / A vượt ngưỡng; mức chỉ đổi khi giữ nguyên `min_persist` lần đo liên tiếp.
    - Tăng đột biến: (C_t - C_{t-W}) / C_{t-W} >= surge_ratio với C_{t-W} >= min_count; khi đó nâng mức thêm một bậc.
    """

    def __init__(self, area_m2, crowd_type="di chuyển", surge_ratio=0.5, surge_window=10.0,
                 min_count=10, min_persist=3, ema_alpha=0.5):
        self.area = float(area_m2)
        self.crowd_type = crowd_type
        self.surge_ratio = surge_ratio
        self.surge_window = surge_window
        self.min_count = min_count
        self.min_persist = min_persist
        self.ema_alpha = ema_alpha
        self.history = deque()  # (t, số đếm đã làm trơn)
        self.smoothed = None
        self.level = 0
        self._cand, self._n = 0, 0
        self.surge = False
        self.final_level = 0

    def update(self, t, count):
        a = self.ema_alpha
        self.smoothed = count if self.smoothed is None else a * count + (1 - a) * self.smoothed
        rho = self.smoothed / self.area
        events = []

        raw = density_level(rho, self.crowd_type)
        if raw == self.level:
            self._n = 0
        else:
            self._n = self._n + 1 if raw == self._cand else 1
            self._cand = raw
            if self._n >= self.min_persist:
                if raw > self.level:
                    events.append(f"Mật độ tăng lên mức {LEVELS[raw]['name']} ({rho:.2f} người/m²)")
                else:
                    events.append(f"Mật độ giảm về mức {LEVELS[raw]['name']} ({rho:.2f} người/m²)")
                self.level, self._n = raw, 0

        while self.history and self.history[0][0] < t - self.surge_window:
            self.history.popleft()
        base = self.history[0][1] if self.history else None
        surge = (base is not None and base >= self.min_count
                 and (self.smoothed - base) / base >= self.surge_ratio)
        if surge and not self.surge:
            events.append(f"Số người tăng đột biến {base:.0f} → {self.smoothed:.0f} trong {self.surge_window:.0f}s")
        self.surge = surge
        self.history.append((t, self.smoothed))

        final = min(3, self.level + (1 if surge else 0))
        if final != self.final_level and not events:
            events.append(f"Mức cảnh báo: {LEVELS[final]['name']}")
        self.final_level = final
        return {"t": t, "count": count, "smoothed": self.smoothed, "density": rho,
                "level": final, "surge": surge, "events": events}
