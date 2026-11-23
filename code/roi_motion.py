"""Phân tích vùng quan tâm đa giác (Polygon ROI / Vùng cấm) và luồng quang học đám đông.

Cung cấp:
- Chuẩn hóa, mặt nạ đa giác có bộ nhớ đệm và lọc đếm theo ROI / vùng cấm cho cả CSRNet và YOLO.
- Vẽ lớp phủ đa giác ROI / vùng cấm và hộp cảnh báo xâm nhập.
- MotionAnalyzer: ước lượng luồng quang học đặc Farneback, tính vận tốc chuẩn hóa,
  entropy hướng Shannon 12-bin, tỉ lệ di chuyển ngược chiều, điểm bất thường trực tuyến,
  và phát hiện hoảng loạn / ngược chiều.
"""
from __future__ import annotations

import json
from functools import lru_cache

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Đa giác ROI & Vùng cấm (Polygon ROI / Restricted Zone)
# ----------------------------------------------------------------------------
def parse_roi_points(raw) -> list[tuple[float, float]]:
    """Đọc danh sách đỉnh đa giác chuẩn hóa [0, 1]. Trả về [] nếu không hợp lệ hoặc < 3 điểm."""
    if raw is None or raw == "":
        return []
    data = raw
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8")
        except Exception:
            return []
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except Exception:
            return []
    if not isinstance(data, (list, tuple, np.ndarray)):
        return []

    pts: list[tuple[float, float]] = []
    for item in data:
        try:
            if isinstance(item, dict) and "x" in item and "y" in item:
                x, y = float(item["x"]), float(item["y"])
            elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                x, y = float(item[0]), float(item[1])
            else:
                return []
        except (TypeError, ValueError):
            return []
        if not (np.isfinite(x) and np.isfinite(y)):
            return []
        pts.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))

    return pts if len(pts) >= 3 else []


def normalize_roi_mode(mode: str | None, points: list[tuple[float, float]]) -> str:
    """Chuẩn hóa chế độ ROI về 'full', 'roi' hoặc 'forbidden'."""
    if not isinstance(mode, str):
        return "full"
    m = mode.strip().lower()
    if m not in ("roi", "forbidden") or points is None or len(points) < 3:
        return "full"
    return m


@lru_cache(maxsize=128)
def _rasterize_polygon_cached(h: int, w: int, key: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    if len(key) >= 3:
        sx = max(0, w - 1)
        sy = max(0, h - 1)
        pts_px = np.array(
            [[int(round(x * sx)), int(round(y * sy))] for x, y in key],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts_px], 1.0)
    return mask


def rasterize_polygon_mask(h: int, w: int, points: list[tuple[float, float]]) -> np.ndarray:
    """Tạo mặt nạ đa giác float32 (h, w) nhận giá trị trong {0.0, 1.0} (có cache)."""
    hh = max(1, int(h))
    ww = max(1, int(w))
    if not points or len(points) < 3:
        return np.zeros((hh, ww), dtype=np.float32)
    key = tuple((round(float(x), 4), round(float(y), 4)) for x, y in points)
    return _rasterize_polygon_cached(hh, ww, key)


def apply_roi(
    annotation,
    model_id: str,
    roi_mode: str,
    roi_points: list[tuple[float, float]],
) -> dict:
    """Áp dụng vùng ROI / vùng cấm lên bản đồ mật độ CSRNet hoặc hộp đầu YOLO."""
    pts = parse_roi_points(roi_points) if isinstance(roi_points, (str, bytes, bytearray)) else (
        parse_roi_points(roi_points) if roi_points is not None else []
    )
    mode = normalize_roi_mode(roi_mode, pts)

    if model_id == "yolo":
        if annotation is None or len(annotation) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        elif isinstance(annotation, np.ndarray) and annotation.ndim == 2 and annotation.shape[1] == 4:
            boxes = annotation
        else:
            boxes = np.asarray(annotation, dtype=np.float32).reshape(-1, 4)

        empty_boxes = np.empty((0, 4), dtype=np.float32)
        if mode == "full":
            return {
                "roi_mode": mode,
                "count": int(len(boxes)),
                "restricted_count": 0.0,
                "annotation": boxes,
                "restricted_boxes": empty_boxes,
                "mask_ratio": 1.0,
            }

        mask1000 = rasterize_polygon_mask(1000, 1000, pts)
        mask_ratio = max(1e-4, float(mask1000.mean()))
        if len(boxes) > 0:
            cx = np.clip(((boxes[:, 0] + boxes[:, 2]) * 0.5 * 999).astype(int), 0, 999)
            cy = np.clip(((boxes[:, 1] + boxes[:, 3]) * 0.5 * 999).astype(int), 0, 999)
            inside = mask1000[cy, cx] > 0.5
        else:
            inside = np.zeros((0,), dtype=bool)

        if mode == "roi":
            active_boxes = boxes[inside]
            return {
                "roi_mode": mode,
                "count": int(len(active_boxes)),
                "restricted_count": 0.0,
                "annotation": active_boxes,
                "restricted_boxes": empty_boxes,
                "mask_ratio": float(mask_ratio),
            }

        # mode == "forbidden"
        restricted_boxes = boxes[inside]
        return {
            "roi_mode": mode,
            "count": int(len(boxes)),
            "restricted_count": float(len(restricted_boxes)),
            "annotation": boxes,
            "restricted_boxes": restricted_boxes,
            "mask_ratio": float(mask_ratio),
        }

    # Mặc định / model_id == "csrnet": annotation là mảng 2D (dh, dw)
    den = annotation if isinstance(annotation, np.ndarray) else np.asarray(annotation, dtype=np.float32)
    dh, dw = den.shape[:2]

    if mode == "full":
        return {
            "roi_mode": mode,
            "count": float(den.sum()),
            "restricted_count": 0.0,
            "annotation": den,
            "restricted_boxes": None,
            "mask_ratio": 1.0,
        }

    mask = rasterize_polygon_mask(dh, dw, pts)
    mask_ratio = max(1e-4, float(mask.mean()))

    if mode == "roi":
        active_den = den * mask
        return {
            "roi_mode": mode,
            "count": float(active_den.sum()),
            "restricted_count": 0.0,
            "annotation": active_den,
            "restricted_boxes": None,
            "mask_ratio": float(mask_ratio),
        }

    # mode == "forbidden"
    restricted_count = float((den * mask).sum())
    return {
        "roi_mode": mode,
        "count": float(den.sum()),
        "restricted_count": restricted_count,
        "annotation": den,
        "restricted_boxes": None,
        "mask_ratio": float(mask_ratio),
    }


def _to_color(rgb: tuple[int, int, int], is_bgr: bool) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0]) if is_bgr else rgb


def draw_roi_overlay(
    img: np.ndarray,
    roi_mode: str,
    roi_points: list[tuple[float, float]],
    *,
    is_bgr: bool = True,
    triggered: bool = False,
    restricted_boxes=None,
) -> np.ndarray:
    """Vẽ đa giác ROI / vùng cấm và các hộp đầu xâm nhập lên ảnh hiển thị."""
    if img is None or img.size == 0:
        return img
    pts = parse_roi_points(roi_points) if roi_points is not None else []
    mode = normalize_roi_mode(roi_mode, pts)
    if mode == "full":
        return img

    out = img.copy()
    h, w = out.shape[:2]

    if triggered:
        rgb = (225, 45, 35)
    elif mode == "roi":
        rgb = (52, 130, 60)
    else:
        rgb = (215, 130, 30)
    color = _to_color(rgb, is_bgr)
    red_color = _to_color((225, 45, 35), is_bgr)

    pts_px = np.array(
        [[int(round(x * max(0, w - 1))), int(round(y * max(0, h - 1)))] for x, y in pts],
        dtype=np.int32,
    )

    fill_layer = out.copy()
    cv2.fillPoly(fill_layer, [pts_px], color)
    out = cv2.addWeighted(fill_layer, 0.16, out, 0.84, 0)

    cv2.polylines(out, [pts_px], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    for px, py in pts_px:
        cv2.circle(out, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)

    if restricted_boxes is not None and len(restricted_boxes) > 0:
        boxes_arr = np.asarray(restricted_boxes, dtype=np.float32)
        if boxes_arr.ndim == 2 and boxes_arr.shape[1] >= 4:
            pixels = np.rint(boxes_arr[:, :4] * [w - 1, h - 1, w - 1, h - 1]).astype(int)
            for x1, y1, x2, y2 in pixels:
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), red_color, 2, cv2.LINE_AA)

    label = "ROI ZONE" if mode == "roi" else "RESTRICTED ZONE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    top_idx = int(np.lexsort((pts_px[:, 0], pts_px[:, 1]))[0])
    ax, ay = int(pts_px[top_idx, 0]), int(pts_px[top_idx, 1])
    pad_x, pad_y = 6, 4
    box_w = tw + pad_x * 2
    box_h = th + baseline + pad_y * 2
    bx0 = int(np.clip(ax, 4, max(4, w - box_w - 4)))
    by0 = int(np.clip(ay - box_h - 4 if ay - box_h - 4 >= 4 else ay + 6, 4, max(4, h - box_h - 4)))

    cv2.rectangle(out, (bx0, by0), (bx0 + box_w, by0 + box_h), color, -1)
    cv2.putText(
        out,
        label,
        (bx0 + pad_x, by0 + pad_y + th),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


# ----------------------------------------------------------------------------
# Phân tích chuyển động đám đông bằng Optical Flow đặc (Farneback)
# ----------------------------------------------------------------------------
class MotionAnalyzer:
    """Phân tích vận tốc, độ hỗn loạn hướng (entropy) và dòng ngược chiều từ luồng quang học."""

    def __init__(
        self,
        *,
        long_side: int = 240,
        min_mag: float = 0.75,
        min_active_ratio: float = 0.012,
        speed_thr: float = 14.0,
        entropy_thr: float = 0.72,
        counter_thr: float = 0.30,
        ema_alpha: float = 0.35,
    ):
        self.long_side = int(long_side)
        self.min_mag = float(min_mag)
        self.min_active_ratio = float(min_active_ratio)
        self.speed_thr = float(speed_thr)
        self.entropy_thr = float(entropy_thr)
        self.counter_thr = float(counter_thr)
        self.ema_alpha = float(ema_alpha)
        self.reset()

    def reset(self) -> None:
        self._dom_vx: float = 0.0
        self._dom_vy: float = 0.0
        self._active_steps: int = 0
        self._stat_mean = np.array([4.0, 0.30, 0.05], dtype=np.float64)
        self._stat_var = np.array([16.0, 0.04, 0.02], dtype=np.float64)
        self._stat_steps: int = 0
        self._anom_ema: float = 0.0

    @staticmethod
    def _idle_dict(active_ratio: float = 0.0) -> dict:
        return {
            "mean_speed": 0.0,
            "p90_speed": 0.0,
            "direction_entropy": 0.0,
            "dominant_angle_deg": None,
            "counter_flow_ratio": 0.0,
            "active_ratio": round(float(active_ratio), 4),
            "anomaly_score": 0.0,
            "panic": False,
            "counter_flow": False,
            "state_label": "Tĩnh",
            "vectors": [],
        }

    @staticmethod
    def _to_small_gray(img: np.ndarray, gw: int, gh: int) -> np.ndarray:
        if img.shape[1] != gw or img.shape[0] != gh:
            img = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return img.astype(np.uint8, copy=False)

    def update(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
        dt: float,
        *,
        roi_mode: str = "full",
        roi_points=(),
        crowd_mask: np.ndarray | None = None,
    ) -> dict:
        if prev_bgr is None or curr_bgr is None:
            return self._idle_dict(0.0)
        if not isinstance(prev_bgr, np.ndarray) or not isinstance(curr_bgr, np.ndarray):
            return self._idle_dict(0.0)
        if prev_bgr.size == 0 or curr_bgr.size == 0 or prev_bgr.shape[:2] != curr_bgr.shape[:2]:
            return self._idle_dict(0.0)
        if dt is None or not np.isfinite(dt) or dt <= 1e-4:
            return self._idle_dict(0.0)

        h0, w0 = curr_bgr.shape[:2]
        if h0 < 8 or w0 < 8:
            return self._idle_dict(0.0)

        scale = min(1.0, float(self.long_side) / float(max(h0, w0)))
        gw = max(8, int(round(w0 * scale)))
        gh = max(8, int(round(h0 * scale)))

        prev_g = self._to_small_gray(prev_bgr, gw, gh)
        curr_g = self._to_small_gray(curr_bgr, gw, gh)

        flow = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        ang = np.mod(ang, 2.0 * np.pi)

        region_mask = np.ones((gh, gw), dtype=bool)
        pts = parse_roi_points(roi_points) if roi_points is not None else []
        mode = normalize_roi_mode(roi_mode, pts)
        if mode == "roi":
            poly = rasterize_polygon_mask(gh, gw, pts) > 0.5
            if poly.any():
                region_mask &= poly

        if crowd_mask is not None:
            cm_arr = np.asarray(crowd_mask, dtype=np.float32)
            if cm_arr.ndim >= 2 and cm_arr.size > 0:
                if cm_arr.ndim > 2:
                    cm_arr = cm_arr[..., 0]
                cm_small = (
                    cv2.resize(cm_arr, (gw, gh), interpolation=cv2.INTER_LINEAR)
                    if cm_arr.shape[:2] != (gh, gw)
                    else cm_arr
                )
                cm_max = float(np.nanmax(cm_small)) if cm_small.size > 0 else 0.0
                if np.isfinite(cm_max) and cm_max > 1e-6:
                    thr = max(1e-5, cm_max * 0.08)
                    cm_bin = (cm_small >= thr).astype(np.uint8)
                    k = max(5, int(round(max(gw, gh) * 0.06)) | 1)
                    cm_dil = cv2.dilate(cm_bin, np.ones((k, k), dtype=np.uint8)) > 0
                    cand = region_mask & cm_dil
                    if int(cand.sum()) >= int(0.02 * gh * gw):
                        region_mask = cand

        active = region_mask & (mag >= self.min_mag)
        region_area = max(1, int(region_mask.sum()))
        active_ratio = float(active.sum()) / float(region_area)

        diag = float(np.hypot(gw, gh))
        dt_s = max(float(dt), 0.02)
        speed_pct_s = (mag / diag) / dt_s * 100.0

        if int(active.sum()) < 8 or active_ratio < self.min_active_ratio * 0.5:
            self._anom_ema *= 0.5
            return self._idle_dict(active_ratio)

        act_mag = mag[active]
        act_spd = speed_pct_s[active]
        act_ang = ang[active]
        mag_sum = float(act_mag.sum()) + 1e-9

        mean_speed = float(act_spd.mean())
        p90_speed = float(np.percentile(act_spd, 90))

        # Entropy hướng Shannon 12-bin có trọng số biên độ
        hist, _ = np.histogram(act_ang, bins=12, range=(0.0, 2.0 * np.pi), weights=act_mag)
        p = hist / (float(hist.sum()) + 1e-9)
        nz = p > 1e-9
        direction_entropy = float(-np.sum(p[nz] * np.log2(p[nz])) / np.log2(12.0))
        direction_entropy = min(1.0, max(0.0, direction_entropy))

        # Vector dòng chủ đạo tức thời & bán cầu chủ đạo
        vx = float(np.sum(np.cos(act_ang) * act_mag) / mag_sum)
        vy = float(np.sum(np.sin(act_ang) * act_mag) / mag_sum)

        smooth_hist = 0.25 * np.roll(hist, 1) + 0.5 * hist + 0.25 * np.roll(hist, -1)
        k_max = int(np.argmax(smooth_hist))
        seed_ang = (k_max + 0.5) * (2.0 * np.pi / 12.0)
        hemi = np.abs((act_ang - seed_ang + np.pi) % (2.0 * np.pi) - np.pi) <= (np.pi * 0.5)
        if hemi.any():
            hemi_w = act_mag[hemi]
            hemi_sum = float(hemi_w.sum()) + 1e-9
            hx = float(np.sum(np.cos(act_ang[hemi]) * hemi_w) / hemi_sum)
            hy = float(np.sum(np.sin(act_ang[hemi]) * hemi_w) / hemi_sum)
            hemi_ang = float(np.arctan2(hy, hx))
            hemi_share = float(hemi_w.sum()) / mag_sum
        else:
            hx, hy = vx, vy
            hemi_ang = float(np.arctan2(vy, vx))
            hemi_share = 1.0

        if self._active_steps >= 2 and float(np.hypot(self._dom_vx, self._dom_vy)) > 0.15:
            ref_ang = float(np.arctan2(self._dom_vy, self._dom_vx))
        else:
            ref_ang = hemi_ang

        # Cập nhật EMA vector tham chiếu chậm (0.2) để nhóm ngược chiều đột ngột không lật hướng chính ngay
        obs_vx, obs_vy = (vx, vy) if float(np.hypot(vx, vy)) >= 0.22 else (hx * hemi_share, hy * hemi_share)
        if self._active_steps == 0:
            self._dom_vx, self._dom_vy = obs_vx, obs_vy
        else:
            dom_alpha = 0.20
            self._dom_vx = (1.0 - dom_alpha) * self._dom_vx + dom_alpha * obs_vx
            self._dom_vy = (1.0 - dom_alpha) * self._dom_vy + dom_alpha * obs_vy
        self._active_steps += 1

        diff = np.abs((act_ang - ref_ang + np.pi) % (2.0 * np.pi) - np.pi)
        counter_mask = diff > (2.0 * np.pi / 3.0)
        counter_flow_ratio = float(act_mag[counter_mask].sum() / mag_sum)
        dominant_angle_deg = round(float(np.degrees(ref_ang) % 360.0), 1) % 360.0

        # Điểm bất thường thống kê trực tuyến (online EMA / phương sai)
        feat = np.array([mean_speed, direction_entropy, counter_flow_ratio], dtype=np.float64)
        std = np.sqrt(self._stat_var + 1e-6)
        z = np.maximum(0.0, (feat - self._stat_mean) / std)
        raw_anom = float(np.sqrt(np.mean(z * z)))

        adapt = 0.03 if raw_anom > 2.0 else 0.08
        delta = feat - self._stat_mean
        self._stat_mean = self._stat_mean + adapt * delta
        self._stat_var = (1.0 - adapt) * (self._stat_var + adapt * (delta * delta))

        if self._stat_steps == 0:
            self._anom_ema = raw_anom
        else:
            a = self.ema_alpha
            self._anom_ema = a * raw_anom + (1.0 - a) * self._anom_ema
        self._stat_steps += 1
        anomaly_score = max(0.0, float(self._anom_ema))

        # Phân loại trạng thái chuyển động
        panic = bool(
            active_ratio >= self.min_active_ratio
            and p90_speed >= self.speed_thr
            and direction_entropy >= self.entropy_thr
        )
        counter_flow = bool(
            active_ratio >= self.min_active_ratio
            and (not panic)
            and counter_flow_ratio >= self.counter_thr
            and direction_entropy < 0.88
        )
        if panic:
            state_label = "Hỗn loạn"
        elif counter_flow:
            state_label = "Ngược chiều"
        else:
            state_label = "Trật tự"

        # Lấy mẫu lưới vector luồng (~14x10 ô, tối đa 80 vector)
        nx = max(1, min(14, gw // 8))
        ny = max(1, min(10, gh // 8))
        xs = np.linspace(0, gw, nx + 1, dtype=int)
        ys = np.linspace(0, gh, ny + 1, dtype=int)
        candidates: list[tuple[float, list[float]]] = []
        for r in range(ny):
            y0, y1 = int(ys[r]), int(ys[r + 1])
            if y1 <= y0:
                continue
            for c in range(nx):
                x0, x1 = int(xs[c]), int(xs[c + 1])
                if x1 <= x0:
                    continue
                cell_act = active[y0:y1, x0:x1]
                if not cell_act.any():
                    continue
                dx = float(flow[y0:y1, x0:x1, 0][cell_act].mean())
                dy = float(flow[y0:y1, x0:x1, 1][cell_act].mean())
                cmag = float(np.hypot(dx, dy))
                if cmag < self.min_mag * 0.8:
                    continue
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                candidates.append(
                    (
                        cmag,
                        [
                            round(cx / gw, 4),
                            round(cy / gh, 4),
                            round(dx / gw, 4),
                            round(dy / gh, 4),
                        ],
                    )
                )

        if len(candidates) > 80:
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[:80]
        vectors = [vec for _, vec in candidates]

        return {
            "mean_speed": round(mean_speed, 2),
            "p90_speed": round(p90_speed, 2),
            "direction_entropy": round(direction_entropy, 3),
            "dominant_angle_deg": dominant_angle_deg,
            "counter_flow_ratio": round(counter_flow_ratio, 3),
            "active_ratio": round(active_ratio, 4),
            "anomaly_score": round(anomaly_score, 2),
            "panic": panic,
            "counter_flow": counter_flow,
            "state_label": state_label,
            "vectors": vectors,
        }


def draw_motion_vectors(
    img: np.ndarray,
    vectors,
    *,
    is_bgr: bool = True,
    panic: bool = False,
    counter_flow: bool = False,
) -> np.ndarray:
    """Vẽ các mũi tên luồng chuyển động lên ảnh hiển thị."""
    if img is None or img.size == 0 or not vectors:
        return img

    out = img.copy()
    h, w = out.shape[:2]

    if panic:
        rgb = (235, 55, 45)
    elif counter_flow:
        rgb = (245, 145, 30)
    else:
        rgb = (45, 215, 195)
    color = _to_color(rgb, is_bgr)

    max_len = max(18.0, 0.08 * float(np.hypot(w, h)))
    scale = 2.2

    for v in vectors:
        if v is None or len(v) < 4:
            continue
        x1 = int(round(float(v[0]) * w))
        y1 = int(round(float(v[1]) * h))
        vx = float(v[2]) * w * scale
        vy = float(v[3]) * h * scale
        length = float(np.hypot(vx, vy))
        if length < 0.75:
            continue
        if length < 5.0:
            vx *= 5.0 / length
            vy *= 5.0 / length
        elif length > max_len:
            vx *= max_len / length
            vy *= max_len / length
        x2 = int(round(np.clip(x1 + vx, 0, w - 1)))
        y2 = int(round(np.clip(y1 + vy, 0, h - 1)))
        if (x1, y1) != (x2, y2):
            cv2.arrowedLine(
                out,
                (x1, y1),
                (x2, y2),
                color,
                2,
                line_type=cv2.LINE_AA,
                tipLength=0.3,
            )
    return out


# ----------------------------------------------------------------------------
# Bám vết hộp đầu thời gian thực bằng Lucas-Kanade (Real-Time Box Flow Tracker)
# ----------------------------------------------------------------------------
class BoxFlowTracker:
    """Bám vết nhanh tâm các hộp YOLO giữa các lần suy luận (~1 ms/khung trên 320px).

    - Khi có kết quả YOLO mới trên `inferred_bgr` (vốn đã trễ ~150-250 ms so với khung live hiện tại),
      `sync_from_inference` lập tức dịch chuyển các hộp từ `inferred_bgr` tới `live_bgr` bằng
      quang học thưa Lucas-Kanade có kiểm tra xuôi-ngược (forward-backward check) để triệt tiêu độ trễ.
    - Ở các nhịp hiển thị tiếp theo (15-30 FPS) trong lúc chờ lượt YOLO kế tiếp, `step(live_bgr)`
      tiếp tục cập nhật vị trí hộp mượt mà theo chuyển động thật trên khung hình trực tiếp.
    """

    _LK_PARAMS = dict(
        winSize=(17, 17),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
    )

    def __init__(self, track_side: int = 320, fb_max_err: float = 3.0) -> None:
        self.track_side = max(96, int(track_side))
        self.fb_max_err = float(fb_max_err)
        self._prev_gray: np.ndarray | None = None
        self.boxes: np.ndarray = np.empty((0, 4), dtype=np.float32)
        self.restricted_boxes: np.ndarray = np.empty((0, 4), dtype=np.float32)

    def _to_small_gray(self, bgr: np.ndarray | None) -> np.ndarray | None:
        if bgr is None or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        s = min(1.0, self.track_side / float(max(h, w)))
        if s < 1.0:
            gw, gh = max(16, int(round(w * s))), max(16, int(round(h * s)))
            bgr = cv2.resize(bgr, (gw, gh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    def _warp_boxes(self, g0: np.ndarray, g1: np.ndarray, boxes: np.ndarray | None) -> np.ndarray:
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)
        arr = np.asarray(boxes, dtype=np.float32).copy()
        if g0 is None or g1 is None or g0.shape != g1.shape:
            return arr
        gh, gw = g0.shape[:2]
        cx = ((arr[:, 0] + arr[:, 2]) * 0.5 * (gw - 1)).astype(np.float32)
        cy = ((arr[:, 1] + arr[:, 3]) * 0.5 * (gh - 1)).astype(np.float32)
        p0 = np.stack([cx, cy], axis=-1).reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **self._LK_PARAMS)
        if p1 is None or st1 is None:
            return arr
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **self._LK_PARAMS)
        valid = st1.reshape(-1) == 1
        if p0r is not None and st2 is not None:
            fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
            valid = valid & (st2.reshape(-1) == 1) & (fb_err <= self.fb_max_err)
        if not np.any(valid):
            return arr
        dx = ((p1.reshape(-1, 2)[:, 0] - cx) / max(1.0, float(gw - 1))).astype(np.float32)
        dy = ((p1.reshape(-1, 2)[:, 1] - cy) / max(1.0, float(gh - 1))).astype(np.float32)
        dx = np.where(valid, np.clip(dx, -0.15, 0.15), 0.0)
        dy = np.where(valid, np.clip(dy, -0.15, 0.15), 0.0)
        bw = arr[:, 2] - arr[:, 0]
        bh = arr[:, 3] - arr[:, 1]
        nx1 = np.clip(arr[:, 0] + dx, 0.0, np.maximum(0.0, 1.0 - bw))
        ny1 = np.clip(arr[:, 1] + dy, 0.0, np.maximum(0.0, 1.0 - bh))
        arr[:, 0] = nx1
        arr[:, 1] = ny1
        arr[:, 2] = np.clip(nx1 + bw, 0.0, 1.0)
        arr[:, 3] = np.clip(ny1 + bh, 0.0, 1.0)
        return arr

    def sync_from_inference(
        self,
        inferred_bgr: np.ndarray | None,
        live_bgr: np.ndarray | None,
        boxes: np.ndarray | None,
        restricted_boxes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        g_inf = self._to_small_gray(inferred_bgr)
        g_live = self._to_small_gray(live_bgr)
        b_arr = np.empty((0, 4), dtype=np.float32) if boxes is None or len(boxes) == 0 else np.asarray(boxes, dtype=np.float32).copy()
        r_arr = (
            np.empty((0, 4), dtype=np.float32)
            if restricted_boxes is None or len(restricted_boxes) == 0
            else np.asarray(restricted_boxes, dtype=np.float32).copy()
        )
        if g_inf is not None and g_live is not None and inferred_bgr is not live_bgr:
            if g_inf.shape != g_live.shape:
                g_inf = cv2.resize(g_inf, (g_live.shape[1], g_live.shape[0]), interpolation=cv2.INTER_AREA)
            if len(b_arr) > 0:
                b_arr = self._warp_boxes(g_inf, g_live, b_arr)
            if len(r_arr) > 0:
                r_arr = self._warp_boxes(g_inf, g_live, r_arr)
        self._prev_gray = g_live if g_live is not None else g_inf
        self.boxes = b_arr
        self.restricted_boxes = r_arr
        return self.boxes, self.restricted_boxes

    def step(self, live_bgr: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        g_curr = self._to_small_gray(live_bgr)
        if g_curr is None:
            return self.boxes, self.restricted_boxes
        if self._prev_gray is not None and self._prev_gray.shape == g_curr.shape:
            if len(self.boxes) > 0:
                self.boxes = self._warp_boxes(self._prev_gray, g_curr, self.boxes)
            if len(self.restricted_boxes) > 0:
                self.restricted_boxes = self._warp_boxes(self._prev_gray, g_curr, self.restricted_boxes)
        self._prev_gray = g_curr
        return self.boxes, self.restricted_boxes
