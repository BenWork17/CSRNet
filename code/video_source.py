"""Đọc video tuần tự hoặc giữ duy nhất frame camera mới nhất."""
import threading
import time

import cv2


def open_capture(source):
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        # Timeout phải được đặt khi mở nguồn FFmpeg, không phải sau khi mở.
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ])
    return cv2.VideoCapture(source)


class LatestFrameCapture:
    """Một luồng liên tục đọc camera; frame mới ghi đè frame chưa xử lý."""

    def __init__(self, cap):
        self.cap = cap
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._finished = False
        self._error = None
        self._latest = None
        self._sequence = 0
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Một số backend không hỗ trợ.
        self._thread = threading.Thread(target=self._capture, daemon=True, name="crowd-camera")
        self._thread.start()

    def _capture(self):
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                captured_at = time.monotonic()
                with self._condition:
                    self._sequence += 1
                    self._latest = (self._sequence, captured_at, frame)
                    self._condition.notify_all()
        except Exception as exc:
            self._error = exc
        finally:
            self.cap.release()
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    @property
    def finished(self):
        with self._condition:
            return self._finished

    def read(self, after_sequence, timeout=0.1):
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._finished or self._stop.is_set(),
                timeout=timeout,
            )
            if self._sequence > after_sequence:
                return self._latest
            if self._error is not None:
                raise RuntimeError("Không đọc được khung hình camera.") from self._error
            return None

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        # Chỉ luồng đọc gọi release(), tránh giải phóng cap khi read() còn chạy.
        # Nguồn RTSP có read timeout 2 giây; webcam thường trả về sau một frame.
        self._thread.join(timeout=2.5)


def iter_frames(cap, *, is_file, stride, should_stop, max_secs=None):
    """Yield (thời gian nguồn, ảnh BGR, thời điểm nhận frame).

    Hàm sở hữu cap và đóng nó khi hết nguồn, dừng hoặc đóng generator.
    Video file giữ thứ tự lấy mẫu; camera bỏ frame cũ khi suy luận chậm.
    """
    if is_file:
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            idx, next_t = 0, 0.0
            while not should_stop() and cap.grab():
                tsec = idx / fps
                idx += 1
                if tsec + 1e-9 < next_t:
                    continue
                next_t += stride
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield tsec, frame, time.monotonic()
        finally:
            cap.release()
        return

    started = time.monotonic()
    reader = None
    sequence, next_due = 0, started
    try:
        reader = LatestFrameCapture(cap)
        while not should_stop():
            now = time.monotonic()
            if max_secs is not None and now - started >= max_secs:
                break
            if now < next_due:
                time.sleep(min(0.02, next_due - now))
                continue
            sample = reader.read(sequence)
            if sample is None:
                if reader.finished:
                    break
                continue
            sequence, captured_at, frame = sample
            if should_stop() or (max_secs is not None and time.monotonic() - started >= max_secs):
                break
            # Nếu xử lý chậm, chạy tiếp với frame mới; không chạy bù frame cũ.
            next_due = max(next_due + stride, time.monotonic())
            yield captured_at - started, frame, captured_at
    finally:
        if reader is not None:
            reader.close()
        else:
            cap.release()
