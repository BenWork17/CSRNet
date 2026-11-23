"""Đọc video tuần tự, giữ frame camera mới nhất, hoặc tải trước phân đoạn HLS mượt không khựng."""
import queue
import re
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import requests


_SQ_RE = re.compile(r"/sq/(\d+)(?:/|$)")


def _is_hls_url(source) -> bool:
    if not isinstance(source, str):
        return False
    low = source.lower()
    if not low.startswith(("http://", "https://")):
        return False
    path = low.split("?", 1)[0]
    return path.endswith(".m3u8") or "/hls_playlist/" in low or "/hls_variant/" in low


class HlsPrefetchCapture:
    """Bộ đọc luồng trực tiếp HLS (YouTube Live / IP Camera HLS) chống khựng phân đoạn.

    - Dùng `requests` (nén `gzip` giảm `.m3u8` DVR 3,4 MB xuống ~30 KB, kèm `Connection: close`
      tránh lỗi `403 Forbidden` keep-alive của `googlevideo.com`) tải trước các phân đoạn `.ts`
      ở luồng nền cách biên trực tiếp 2 phân đoạn (~8-10s đệm an toàn).
    - Giải mã liên tục qua một tiến trình `ffmpeg` duy nhất và nhả khung đều đặn ở đúng nhịp 1x
      (`target_fps = 20`), triệt tiêu hoàn toàn hiện tượng chạy 5s lại khựng 8-12s của OpenCV.
    """

    def __init__(self, m3u8_url: str, *, open_timeout_ms: int = 18000, read_timeout_ms: int = 12000,
                 max_side: int = 800, target_fps: float = 20.0) -> None:
        from youtube_source import get_stream_context, resolve_youtube_live

        ctx = get_stream_context(m3u8_url)
        self._m3u8_url = m3u8_url
        self._watch_url = ctx.get("watch_url")
        self._resolve_fn = resolve_youtube_live if self._watch_url else None
        self._headers = dict(ctx.get("headers") or {})
        self._headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        self._headers["Accept-Encoding"] = "gzip, deflate"

        src_w = max(64, int(ctx.get("width") or 1280))
        src_h = max(64, int(ctx.get("height") or 720))
        scale = min(1.0, float(max_side) / float(max(src_w, src_h)))
        self.width = max(16, int(round(src_w * scale)) // 2 * 2)
        self.height = max(16, int(round(src_h * scale)) // 2 * 2)
        self.fps = float(min(max(target_fps, 10.0), float(ctx.get("fps") or 25.0)))
        self._frame_bytes = self.width * self.height * 3
        self._open_timeout_s = max(25.0, open_timeout_ms / 1000.0)
        self._read_timeout_s = max(12.0, read_timeout_ms / 1000.0)

        self._stop = threading.Event()
        self._frame_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=int(self.fps * 12))
        self._opened = False
        self._released = False
        self._next_emit_ts: float | None = None
        self._proc: subprocess.Popen | None = None

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return

        ff_cmd = [
            ffmpeg_bin,
            "-loglevel", "quiet",
            "-probesize", "262144",
            "-analyzeduration", "500000",
            "-fflags", "+nobuffer+genpts",
            "-flags", "low_delay",
            "-f", "mpegts",
            "-i", "pipe:0",
            "-an", "-sn",
            "-vf", f"fps={self.fps:g},scale={self.width}:{self.height}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                ff_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._frame_bytes * 4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._proc = None
            return

        self._feeder_thread = threading.Thread(target=self._feed_loop, daemon=True, name="hls-segment-feeder")
        self._decoder_thread = threading.Thread(target=self._decode_loop, daemon=True, name="hls-frame-reader")
        self._feeder_thread.start()
        self._decoder_thread.start()

        # Đợi khung hình đầu tiên để xác nhận luồng thực sự mở thành công
        try:
            first = self._frame_q.get(timeout=self._open_timeout_s)
        except queue.Empty:
            first = None
        if first is None:
            self.release()
            return
        self._first_frame: np.ndarray | None = first
        self._opened = True

    def _parse_playlist(self, text: str) -> list[tuple[int, str]]:
        media_seq = 0
        raw_urls: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                try:
                    media_seq = int(s.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif not s.startswith("#"):
                raw_urls.append(s)
        out: list[tuple[int, str]] = []
        for idx, u in enumerate(raw_urls):
            m = _SQ_RE.search(u)
            sq = int(m.group(1)) if m else (media_seq + idx)
            out.append((sq, u))
        return out

    def _refresh_manifest_url(self) -> bool:
        if not self._resolve_fn or not self._watch_url:
            return False
        try:
            new_url = self._resolve_fn(self._watch_url)
            if new_url:
                from youtube_source import get_stream_context
                self._m3u8_url = new_url
                ctx = get_stream_context(new_url)
                if ctx.get("headers"):
                    self._headers.update(ctx["headers"])
                    self._headers["Accept-Encoding"] = "gzip, deflate"
                return True
        except Exception:
            pass
        return False

    def _feed_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        max_sq = -1
        consecutive_errors = 0
        seg_headers = dict(self._headers)
        seg_headers["Connection"] = "close"

        try:
            while not self._stop.is_set():
                try:
                    resp = requests.get(self._m3u8_url, headers=seg_headers, timeout=15)
                    if resp.status_code == 403 and self._refresh_manifest_url():
                        seg_headers = dict(self._headers)
                        seg_headers["Connection"] = "close"
                        continue
                    resp.raise_for_status()
                    numbered = self._parse_playlist(resp.text)
                except Exception:
                    consecutive_errors += 1
                    if consecutive_errors >= 4:
                        if not self._refresh_manifest_url():
                            break
                        consecutive_errors = 0
                    self._stop.wait(0.8)
                    continue

                if not numbered:
                    self._stop.wait(0.8)
                    continue

                if max_sq < 0:
                    # Bắt đầu lùi 2 phân đoạn (~8-10s) để có sẵn dữ liệu phát ngay không khựng
                    to_fetch = numbered[max(0, len(numbered) - 2):]
                else:
                    to_fetch = [(sq, u) for sq, u in numbered if sq > max_sq]

                if not to_fetch:
                    self._stop.wait(1.0)
                    continue

                fetched_ok = False
                for sq, seg_url in to_fetch:
                    if self._stop.is_set():
                        break
                    try:
                        with requests.get(seg_url, headers=seg_headers, stream=True, timeout=6) as sr:
                            if sr.status_code == 403:
                                self._refresh_manifest_url()
                                seg_headers = dict(self._headers)
                                seg_headers["Connection"] = "close"
                                break
                            sr.raise_for_status()
                            for chunk in sr.iter_content(chunk_size=65536):
                                if self._stop.is_set():
                                    break
                                if chunk:
                                    proc.stdin.write(chunk)
                            proc.stdin.flush()
                        max_sq = max(max_sq, sq)
                        fetched_ok = True
                        consecutive_errors = 0
                    except BrokenPipeError:
                        return
                    except Exception:
                        consecutive_errors += 1
                        break

                self._stop.wait(0.6 if fetched_ok else 0.8)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def _decode_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        fb = self._frame_bytes
        try:
            while not self._stop.is_set():
                buf = proc.stdout.read(fb)
                if not buf or len(buf) < fb:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                # Nếu hàng đợi đệm đầy (quá 12s), bỏ bớt khung cũ nhất để giữ bộ nhớ ổn định
                while not self._stop.is_set():
                    try:
                        self._frame_q.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        try:
                            self._frame_q.get_nowait()
                        except queue.Empty:
                            pass
        finally:
            try:
                self._frame_q.put_nowait(None)
            except queue.Full:
                pass

    def isOpened(self) -> bool:
        return self._opened and not self._released

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def read(self):
        if not self.isOpened():
            return False, None
        if self._first_frame is not None:
            frm = self._first_frame
            self._first_frame = None
            self._next_emit_ts = time.perf_counter() + (1.0 / self.fps)
            return True, frm

        # Giữ nhịp phát chuẩn 1x để kho đệm phân đoạn luôn đầy trước khi phân đoạn tiếp theo tới
        if self._next_emit_ts is not None:
            wait = self._next_emit_ts - time.perf_counter()
            if wait > 0:
                if self._stop.wait(wait):
                    return False, None
            elif wait < -0.5:
                self._next_emit_ts = time.perf_counter()
            self._next_emit_ts += 1.0 / self.fps

        try:
            frm = self._frame_q.get(timeout=self._read_timeout_s)
        except queue.Empty:
            return False, None
        if frm is None:
            return False, None
        return True, frm

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._opened = False
        self._stop.set()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass


def open_capture(source, *, open_timeout_ms=5000, read_timeout_ms=2000):
    if _is_hls_url(source) and shutil.which("ffmpeg"):
        return HlsPrefetchCapture(
            source,
            open_timeout_ms=open_timeout_ms,
            read_timeout_ms=read_timeout_ms,
        )
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        # Timeout phải được đặt khi mở nguồn FFmpeg, không phải sau khi mở.
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms,
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
