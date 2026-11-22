"""Chạy: python -m unittest discover -s code -p test_video_source.py -v"""
import json
import math
import queue
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_source import LatestFrameCapture, iter_frames, open_capture


class FileCapture:
    def __init__(self, fps=25, seconds=2):
        self.fps, self.total = fps, fps * seconds
        self.index = -1
        self.retrieved = []
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return self.fps if key == cv2.CAP_PROP_FPS else self.total

    def grab(self):
        self.index += 1
        return self.index < self.total

    def retrieve(self):
        self.retrieved.append(self.index)
        return True, self.index

    def release(self):
        self.released = True


class CameraCapture(FileCapture):
    def __init__(self):
        super().__init__()
        self.pending = queue.Queue()
        self.condition = threading.Condition()
        self.reads = 0

    def set(self, *_):
        return False  # Backend không hỗ trợ buffer size vẫn phải chạy được.

    def push(self, value):
        self.pending.put(value)

    def read(self):
        with self.condition:
            self.reads += 1
            self.condition.notify_all()
        value = self.pending.get(timeout=2)
        if value is None:
            return False, None
        return True, np.full((8, 8, 3), value, dtype=np.uint8)

    def wait_reads(self, count):
        with self.condition:
            if not self.condition.wait_for(lambda: self.reads >= count, timeout=2):
                raise AssertionError("Luồng đọc camera không tiến triển")


class VideoSourceTests(unittest.TestCase):
    def test_files_keep_every_scheduled_sample(self):
        for fps, target_fps in ((10, 10), (25, 10), (30, 10), (60, 10), (25, 15), (30, 15), (60, 15)):
            with self.subTest(fps=fps, target_fps=target_fps):
                cap = FileCapture(fps)
                frames = list(iter_frames(cap, is_file=True, stride=1 / target_fps, should_stop=lambda: False))
                expected = [math.ceil(k * fps / target_fps - 1e-9) for k in range(2 * target_fps)]
                self.assertEqual([frame for _, frame, _ in frames], expected)
                self.assertEqual(cap.retrieved, expected)
                self.assertTrue(cap.released)

    def test_reader_keeps_latest_frame_after_source_ends(self):
        cap = CameraCapture()
        reader = LatestFrameCapture(cap)
        try:
            for value in (1, 2, 3, 4, None):
                cap.push(value)
            reader._thread.join(timeout=2)
            self.assertTrue(reader.finished)
            seq, _, frame = reader.read(0)
            self.assertEqual(seq, 4)
            self.assertEqual(int(frame[0, 0, 0]), 4)
            self.assertIsNone(reader.read(seq))
        finally:
            cap.push(None)
            reader.close()
        self.assertTrue(cap.released)

    def test_camera_drops_frames_produced_during_inference(self):
        cap = CameraCapture()
        cap.push(1)
        frames = iter_frames(cap, is_file=False, stride=0.01, should_stop=lambda: False)
        try:
            first = next(frames)
            self.assertEqual(int(first[1][0, 0, 0]), 1)
            # Trong lúc consumer xử lý frame 1, camera vẫn nhận frame 2, 3, 4.
            for value in (2, 3, 4):
                cap.push(value)
            cap.wait_reads(5)
            latest = next(frames)
            self.assertEqual(int(latest[1][0, 0, 0]), 4)
            self.assertGreaterEqual(latest[0], first[0])
        finally:
            cap.push(None)
            frames.close()
        self.assertTrue(cap.released)

    def test_camera_stop_closes_capture(self):
        cap = CameraCapture()
        cap.push(1)
        stop = threading.Event()
        frames = iter_frames(cap, is_file=False, stride=0.1, should_stop=stop.is_set)
        next(frames)
        stop.set()
        cap.push(None)
        self.assertEqual(list(frames), [])
        self.assertTrue(cap.released)

    def test_camera_duration_expires_without_a_new_frame(self):
        cap = CameraCapture()
        cap.push(1)
        with patch("video_source.time.monotonic", return_value=100.0) as clock:
            frames = iter_frames(cap, is_file=False, stride=0.1, should_stop=lambda: False, max_secs=10)
            next(frames)
            clock.return_value = 111.0
            cap.push(None)
            self.assertEqual(list(frames), [])
        self.assertTrue(cap.released)

    def test_read_errors_close_capture(self):
        cap = CameraCapture()
        with patch.object(cap, "read", side_effect=RuntimeError("camera disconnected")):
            with self.assertRaisesRegex(RuntimeError, "camera"):
                list(iter_frames(cap, is_file=False, stride=0.1, should_stop=lambda: False))
        self.assertTrue(cap.released)

    def test_network_capture_has_open_and_read_timeouts(self):
        with patch("video_source.cv2.VideoCapture") as capture:
            open_capture("rtsp://camera/live")
            capture.assert_called_once_with("rtsp://camera/live", cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
            ])


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Kiểm thử lifecycle streaming không cần nạp trọng số hay GPU.
        with patch("crowd_core.load_model", return_value=(None, 100, {})):
            import server
        cls.server = server

    def job(self):
        self.assertAlmostEqual(self.server.read_params({})["stride"], 1 / 15)
        return dict(params=self.server.read_params({}), kind="camera", src=0, stop=False, tmp=None)

    @staticmethod
    def decode(event):
        lines = event.splitlines()
        return lines[0], json.loads(lines[1][6:])

    def test_disconnect_before_first_frame_releases_capture(self):
        cap = CameraCapture()
        with patch.object(self.server, "open_capture", return_value=cap):
            events = self.server.run_job("test", self.job())
            self.assertEqual(self.decode(next(events))[0], "event: start")
            events.close()
        self.assertTrue(cap.released)

    def test_camera_stream_uses_latest_frame_and_reports_metrics(self):
        cap = CameraCapture()
        cap.push(1)
        job, seen = self.job(), []

        def analyse(image, _params):
            seen.append(int(image[0, 0, 0]))
            if len(seen) == 1:
                for value in (2, 3, 4):
                    cap.push(value)
                cap.wait_reads(5)
            else:
                job["stop"] = True
                cap.push(None)
            return 10, image, image

        with patch.object(self.server, "open_capture", return_value=cap), \
                patch.object(self.server, "analyse", side_effect=analyse), \
                patch.object(self.server, "to_data_url", return_value="data:image/jpeg;base64,test"):
            events = [self.decode(e) for e in self.server.run_job("test", job)]
        self.assertEqual(seen, [1, 4])
        ticks = [data for kind, data in events if kind == "event: tick"]
        self.assertEqual(len(ticks), 2)
        self.assertTrue(all(t["fps"] > 0 and t["latency_ms"] >= 0 for t in ticks))
        self.assertEqual(events[-1][0], "event: done")
        self.assertTrue(events[-1][1]["stopped"])
        self.assertTrue(cap.released)


if __name__ == "__main__":
    unittest.main()
