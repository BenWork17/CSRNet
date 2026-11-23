"""Lưu trữ bất đồng bộ phiên phân tích, chuỗi thời gian và cảnh báo vào PostgreSQL 16.

Thiết kế hướng hiệu năng cho luồng video thời gian thực:
- Tự động khởi tạo database `crowd_monitor`, schema và các bảng/chỉ mục khi khởi động
  (nếu Docker PostgreSQL tạm dừng, ứng dụng vẫn chạy bình thường và tự thử kết nối lại sau).
- Luồng nền (`queue.Queue`) gom lô `timeseries` (`executemany`) và ghi `sessions`/`alerts`
  bất đồng bộ để không gây độ trễ cho vòng đọc/hiển thị khung hình.
- Ảnh bằng chứng JPEG của cảnh báo được nén nhanh và ghi nguyên tử (`.tmp` -> `os.replace`)
  ngay khi phát hiện sự kiện (có áp dụng cooldown theo loại cảnh báo & vượt cooldown khi tăng cấp)
  để giao diện có thể tải `/api/evidence/<file>` tức thì không bị lỗi 404.
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Callable

import cv2
import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SAFE_EVIDENCE_RE = re.compile(r"[A-Za-z0-9_.-]+\.jpg")

# Ngưỡng gom lô chuỗi thời gian trong luồng nền
_BATCH_MAX_ROWS = 25
_BATCH_FLUSH_SEC = 0.8
_CONNECT_TIMEOUT_SEC = 4
_EVIDENCE_MAX_SIDE = 880
_EVIDENCE_JPEG_QUALITY = 80

# Màu BGR tương ứng mức cảnh báo 0..3 (Xanh, Vàng, Cam, Đỏ) cho chấm trạng thái trên ảnh bằng chứng
_LEVEL_BGR = {
    0: (37, 88, 50),
    1: (26, 116, 168),
    2: (31, 83, 194),
    3: (26, 38, 163),
}


def _opt_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) or np.isinf(f) else f


def _json_safe(obj: Any) -> Any:
    """Chuyển các kiểu NumPy/datetime sang kiểu chuẩn để сериalize JSONB an toàn."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _iso_or_none(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _is_missing_database_error(exc: BaseException, dbname: str) -> bool:
    """Nhận diện lỗi database chưa tồn tại (SQLSTATE 3D000 hoặc thông báo FATAL của libpq)."""
    if isinstance(exc, psycopg.errors.InvalidCatalogName):
        return True
    if getattr(exc, "sqlstate", None) == "3D000":
        return True
    msg = str(exc).lower()
    return "does not exist" in msg and ("database" in msg or (bool(dbname) and dbname.lower() in msg))


def _parse_dsn_metadata(dsn: str | None) -> tuple[dict[str, Any], str, str, str]:
    """Phân tách chuỗi kết nối để lấy (conn_dict, host, port, dbname)."""
    if not dsn or not dsn.strip():
        return {}, "127.0.0.1", "5432", ""
    raw = dsn.strip()
    try:
        info = conninfo_to_dict(raw)
        host = str(info.get("host") or "127.0.0.1")
        port = str(info.get("port") or "5432")
        dbname = str(info.get("dbname") or "crowd_monitor")
        info.setdefault("dbname", dbname)
        return info, host, port, dbname
    except Exception:
        parsed = urllib.parse.urlsplit(raw)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 5432)
        dbname = (parsed.path or "").lstrip("/") or "crowd_monitor"
        return {}, host, port, dbname


class PostgresEventStore:
    """Kho lưu trữ sự kiện đám đông trên PostgreSQL 16 kèm ảnh chụp bằng chứng JPEG."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        schema: str = "public",
        evidence_dir: str | None = None,
        cooldown_sec: float = 4.0,
        start_worker: bool = True,
    ) -> None:
        if not isinstance(schema, str) or not _IDENT_RE.fullmatch(schema):
            raise ValueError(f"Tên schema PostgreSQL không hợp lệ: {schema!r}")

        if dsn is None:
            load_dotenv(os.path.join(BASE, ".env"))
            dsn = os.environ.get("CROWD_DATABASE_URL")

        self.dsn: str | None = dsn.strip() if isinstance(dsn, str) and dsn.strip() else None
        self.schema: str = schema
        self._schema_ident = sql.Identifier(schema)
        self.evidence_dir: str = os.path.abspath(
            evidence_dir if evidence_dir is not None else os.path.join(BASE, "data", "evidence")
        )
        os.makedirs(self.evidence_dir, exist_ok=True)

        self.cooldown_sec: float = max(0.0, float(cooldown_sec))
        self._conn_dict, self._host, self._port, self._dbname = _parse_dsn_metadata(self.dsn)
        if self._dbname == "dss_jobs":
            raise ValueError("Không được phép sử dụng database 'dss_jobs'; hãy trỏ tới 'crowd_monitor'.")

        self._init_lock = threading.Lock()
        self._ready = False
        self._last_error: str | None = None
        self._last_retry_ts: float = 0.0

        self._cooldown_lock = threading.Lock()
        self._cooldown_state: dict[tuple[str, str], tuple[float, int]] = {}

        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._closed = False
        self._worker: threading.Thread | None = None

        # Khởi tạo database/bảng ngay khi tạo đối tượng (không làm sập server nếu Docker tắt)
        self.ensure_ready()

        if start_worker:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"pg-event-store-{self.schema}",
                daemon=True,
            )
            self._worker.start()

    # ------------------------------------------------------------------ Kết nối & Khởi tạo DDL
    def _connect_target(self, *, autocommit: bool = True, row_factory: Any = None) -> psycopg.Connection:
        if not self.dsn:
            raise psycopg.OperationalError("Chưa cấu hình biến môi trường CROWD_DATABASE_URL.")
        kwargs: dict[str, Any] = {"autocommit": autocommit}
        if "connect_timeout" not in self._conn_dict:
            kwargs["connect_timeout"] = _CONNECT_TIMEOUT_SEC
        if row_factory is not None:
            kwargs["row_factory"] = row_factory
        conn = psycopg.connect(self.dsn, **kwargs)
        try:
            conn.execute(sql.SQL("SET search_path TO {}, public").format(self._schema_ident))
        except Exception:
            conn.close()
            raise
        return conn

    def _create_database_if_missing(self) -> None:
        if not self._dbname or not _IDENT_RE.fullmatch(self._dbname):
            raise ValueError(f"Tên database PostgreSQL không hợp lệ: {self._dbname!r}")
        if self._dbname == "dss_jobs":
            raise ValueError("Không được phép can thiệp vào database 'dss_jobs'.")

        last_exc: Exception | None = None
        for maint_db in ("postgres", "template1"):
            try:
                if self._conn_dict:
                    maint_cfg = dict(self._conn_dict)
                    maint_cfg["dbname"] = maint_db
                    maint_cfg.setdefault("connect_timeout", _CONNECT_TIMEOUT_SEC)
                    maint_dsn = make_conninfo(**maint_cfg)
                    conn = psycopg.connect(maint_dsn, autocommit=True)
                else:
                    parts = urllib.parse.urlsplit(self.dsn or "")
                    maint_url = urllib.parse.urlunsplit(
                        (parts.scheme, parts.netloc, f"/{maint_db}", parts.query, parts.fragment)
                    )
                    conn = psycopg.connect(maint_url, autocommit=True, connect_timeout=_CONNECT_TIMEOUT_SEC)
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self._dbname,))
                        if cur.fetchone() is None:
                            try:
                                cur.execute(
                                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self._dbname))
                                )
                            except psycopg.errors.DuplicateDatabase:
                                pass
                conn.close()
                return
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc

    def ensure_ready(self) -> bool:
        """Đảm bảo database `crowd_monitor`, schema và các bảng/chỉ mục đã sẵn sàng."""
        if not self.dsn:
            self._ready = False
            self._last_error = "Chưa cấu hình biến môi trường CROWD_DATABASE_URL."
            return False

        with self._init_lock:
            self._last_retry_ts = time.monotonic()
            conn: psycopg.Connection | None = None
            try:
                kwargs: dict[str, Any] = {"autocommit": True}
                if "connect_timeout" not in self._conn_dict:
                    kwargs["connect_timeout"] = _CONNECT_TIMEOUT_SEC
                try:
                    conn = psycopg.connect(self.dsn, **kwargs)
                except psycopg.Error as exc:
                    if _is_missing_database_error(exc, self._dbname):
                        self._create_database_if_missing()
                        conn = psycopg.connect(self.dsn, **kwargs)
                    else:
                        raise

                s_ident = self._schema_ident
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(s_ident))
                        cur.execute(sql.SQL("SET search_path TO {}, public").format(s_ident))

                        cur.execute(
                            sql.SQL(
                                """
                                CREATE TABLE IF NOT EXISTS {s}.sessions (
                                    id VARCHAR(64) PRIMARY KEY,
                                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                    ended_at TIMESTAMPTZ,
                                    source_kind VARCHAR(24) NOT NULL,
                                    source_label TEXT NOT NULL,
                                    model VARCHAR(24) NOT NULL,
                                    model_label VARCHAR(48) NOT NULL,
                                    area_m2 DOUBLE PRECISION NOT NULL,
                                    crowd_type VARCHAR(32) NOT NULL,
                                    roi_mode VARCHAR(24) NOT NULL DEFAULT 'full',
                                    params JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                                    samples INTEGER NOT NULL DEFAULT 0,
                                    max_count DOUBLE PRECISION NOT NULL DEFAULT 0,
                                    max_level SMALLINT NOT NULL DEFAULT 0,
                                    display_fps DOUBLE PRECISION,
                                    infer_fps DOUBLE PRECISION,
                                    infer_ms DOUBLE PRECISION
                                )
                                """
                            ).format(s=s_ident)
                        )
                        cur.execute(
                            sql.SQL(
                                "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON {s}.sessions(created_at DESC)"
                            ).format(s=s_ident)
                        )

                        cur.execute(
                            sql.SQL(
                                """
                                CREATE TABLE IF NOT EXISTS {s}.timeseries (
                                    id BIGSERIAL PRIMARY KEY,
                                    session_id VARCHAR(64) NOT NULL REFERENCES {s}.sessions(id) ON DELETE CASCADE,
                                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                    t_sec DOUBLE PRECISION NOT NULL,
                                    count_raw DOUBLE PRECISION NOT NULL,
                                    count_smoothed DOUBLE PRECISION NOT NULL,
                                    density DOUBLE PRECISION NOT NULL,
                                    restricted_count DOUBLE PRECISION NOT NULL DEFAULT 0,
                                    mean_speed DOUBLE PRECISION,
                                    direction_entropy DOUBLE PRECISION,
                                    counter_flow_ratio DOUBLE PRECISION,
                                    level SMALLINT NOT NULL,
                                    surge BOOLEAN NOT NULL DEFAULT FALSE,
                                    panic BOOLEAN NOT NULL DEFAULT FALSE,
                                    counter_flow BOOLEAN NOT NULL DEFAULT FALSE
                                )
                                """
                            ).format(s=s_ident)
                        )
                        cur.execute(
                            sql.SQL(
                                "CREATE INDEX IF NOT EXISTS idx_timeseries_session_t ON {s}.timeseries(session_id, t_sec)"
                            ).format(s=s_ident)
                        )

                        cur.execute(
                            sql.SQL(
                                """
                                CREATE TABLE IF NOT EXISTS {s}.alerts (
                                    id VARCHAR(64) PRIMARY KEY,
                                    session_id VARCHAR(64) REFERENCES {s}.sessions(id) ON DELETE CASCADE,
                                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                    t_sec DOUBLE PRECISION NOT NULL DEFAULT 0,
                                    source_label TEXT NOT NULL,
                                    model_label VARCHAR(48) NOT NULL,
                                    level SMALLINT NOT NULL,
                                    alert_type VARCHAR(32) NOT NULL,
                                    message TEXT NOT NULL,
                                    count_val DOUBLE PRECISION NOT NULL,
                                    density_val DOUBLE PRECISION NOT NULL,
                                    details JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                                    evidence_file TEXT
                                )
                                """
                            ).format(s=s_ident)
                        )
                        cur.execute(
                            sql.SQL(
                                "CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON {s}.alerts(created_at DESC)"
                            ).format(s=s_ident)
                        )
                        cur.execute(
                            sql.SQL(
                                "CREATE INDEX IF NOT EXISTS idx_alerts_type_level ON {s}.alerts(alert_type, level)"
                            ).format(s=s_ident)
                        )
                        cur.execute(
                            sql.SQL(
                                "CREATE INDEX IF NOT EXISTS idx_alerts_session_id ON {s}.alerts(session_id)"
                            ).format(s=s_ident)
                        )

                self._ready = True
                self._last_error = None
                return True
            except Exception as exc:
                self._ready = False
                self._last_error = f"{type(exc).__name__}: {exc}".strip()
                return False
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    def _ensure_ready_lazy(self, min_interval: float = 2.0) -> bool:
        if self._ready:
            return True
        if time.monotonic() - self._last_retry_ts >= min_interval:
            return self.ensure_ready()
        return False

    def status_info(self) -> dict[str, Any]:
        """Trả về thông tin trạng thái PostgreSQL cho endpoint `/api/status`."""
        self._ensure_ready_lazy(min_interval=2.0)
        host_port = f"{self._host}:{self._port}" if self._host else ""
        return {
            "ok": bool(self._ready),
            "engine": "PostgreSQL 16",
            "database": self._dbname,
            "schema": self.schema,
            "host": host_port,
            "error": self._last_error,
        }

    def evidence_path(self, filename: str) -> str | None:
        """Kiểm tra tên tệp an toàn và trả về đường dẫn tuyệt đối của ảnh bằng chứng."""
        if not filename or not _SAFE_EVIDENCE_RE.fullmatch(filename):
            return None
        full = os.path.abspath(os.path.join(self.evidence_dir, filename))
        if os.path.commonpath([self.evidence_dir, full]) != self.evidence_dir:
            return None
        return full if os.path.isfile(full) else None

    # ------------------------------------------------------------------ Chụp ảnh bằng chứng
    def _save_evidence_snapshot(
        self,
        evidence_file: str,
        *,
        frame_bgr: np.ndarray | None,
        render_fn: Callable[[], np.ndarray | None] | None,
        t_sec: float,
        count_val: float,
        density_val: float,
        top_level: int,
    ) -> bool:
        bgr: np.ndarray | None = None
        if callable(render_fn):
            try:
                bgr = render_fn()
            except Exception:
                bgr = None
        if bgr is None and frame_bgr is not None:
            bgr = frame_bgr
        if not isinstance(bgr, np.ndarray) or bgr.ndim < 2 or bgr.size == 0:
            return False

        if bgr.ndim == 2:
            canvas = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            canvas = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        else:
            canvas = bgr.copy()

        h, w = canvas.shape[:2]
        if max(h, w) > _EVIDENCE_MAX_SIDE:
            scale = _EVIDENCE_MAX_SIDE / float(max(h, w))
            canvas = cv2.resize(
                canvas,
                (max(2, int(w * scale)), max(2, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            h, w = canvas.shape[:2]

        # Vẽ thanh thông tin gọn ở cạnh dưới (chỉ dùng ký tự ASCII để Hershey font hiển thị sắc nét)
        if h >= 36 and w >= 120:
            bar_h = min(28, max(20, h // 14))
            y0 = h - bar_h
            roi = canvas[y0:h, 0:w]
            dark = np.zeros_like(roi)
            cv2.addWeighted(roi, 0.32, dark, 0.68, 0, dst=roi)
            dot_color = _LEVEL_BGR.get(int(top_level), _LEVEL_BGR[1])
            cy = y0 + bar_h // 2
            cv2.circle(canvas, (12, cy), 5, dot_color, -1, cv2.LINE_AA)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            text = f"{stamp} | t={t_sec:.1f}s | N={count_val:.1f} | D={density_val:.2f}/m2 | L{int(top_level)}"
            cv2.putText(
                canvas,
                text,
                (24, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (242, 242, 242),
                1,
                cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, _EVIDENCE_JPEG_QUALITY])
        if not ok:
            return False

        os.makedirs(self.evidence_dir, exist_ok=True)
        dst_path = os.path.join(self.evidence_dir, evidence_file)
        tmp_path = os.path.join(self.evidence_dir, f".{evidence_file}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp_path, "wb") as fp:
                fp.write(buf.tobytes())
            os.replace(tmp_path, dst_path)
            return True
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False

    # ------------------------------------------------------------------ Ghi bất đồng bộ (0 ms latency)
    def start_session(
        self,
        session_id: str,
        *,
        source_kind: str,
        source_label: str,
        model: str,
        model_label: str,
        area_m2: float,
        crowd_type: str,
        roi_mode: str = "full",
        params: dict[str, Any] | None = None,
    ) -> None:
        if self._closed or not session_id:
            return
        payload = (
            str(session_id)[:64],
            str(source_kind or "file")[:24],
            str(source_label or "unknown"),
            str(model or "csrnet")[:24],
            str(model_label or model or "CSRNet")[:48],
            float(area_m2),
            str(crowd_type or "di chuyển")[:32],
            str(roi_mode or "full")[:24],
            Jsonb(_json_safe(params or {})),
        )
        self._queue.put(("start_session", payload))

    def enqueue_timeseries(self, session_id: str, row: dict[str, Any]) -> None:
        if self._closed or not session_id or not row:
            return
        count_raw = float(row.get("count_raw", row.get("count", 0.0)) or 0.0)
        count_smoothed = float(row.get("count_smoothed", row.get("smoothed", count_raw)) or 0.0)
        ts_tuple = (
            str(session_id)[:64],
            float(row.get("t_sec", row.get("t", 0.0)) or 0.0),
            count_raw,
            count_smoothed,
            float(row.get("density", 0.0) or 0.0),
            float(row.get("restricted_count", 0.0) or 0.0),
            _opt_float(row.get("mean_speed")),
            _opt_float(row.get("direction_entropy")),
            _opt_float(row.get("counter_flow_ratio")),
            int(row.get("level", 0) or 0),
            bool(row.get("surge", False)),
            bool(row.get("panic", False)),
            bool(row.get("counter_flow", False)),
        )
        self._queue.put(("timeseries", ts_tuple))

    def record_alerts(
        self,
        session_id: str | None,
        *,
        t_sec: float,
        source_label: str,
        model_label: str,
        count_val: float,
        density_val: float,
        event_items: list[dict[str, Any]],
        frame_bgr: np.ndarray | None = None,
        render_fn: Callable[[], np.ndarray | None] | None = None,
        details: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._closed or not event_items:
            return []

        sess_key = str(session_id)[:64] if session_id else None
        cooldown_scope = sess_key or "stateless"
        now_mono = time.monotonic()

        accepted: list[tuple[str, str, int, str, dict[str, Any]]] = []
        with self._cooldown_lock:
            for item in event_items:
                if isinstance(item, dict):
                    alert_type = str(item.get("type") or item.get("alert_type") or "general")[:32]
                    level = int(item.get("level", 1) or 0)
                    message = str(item.get("message") or "").strip()
                    item_details = item.get("details") if isinstance(item.get("details"), dict) else None
                else:
                    alert_type = "general"
                    level = 1
                    message = str(item).strip()
                    item_details = None

                if not message:
                    continue

                cd_key = (cooldown_scope, alert_type)
                prev = self._cooldown_state.get(cd_key)
                if prev is not None:
                    prev_ts, prev_level = prev
                    if (now_mono - prev_ts) < self.cooldown_sec and level <= prev_level:
                        continue

                self._cooldown_state[cd_key] = (now_mono, level)
                merged_details: dict[str, Any] = {}
                if details:
                    merged_details.update(details)
                if item_details:
                    merged_details.update(item_details)
                accepted.append((uuid.uuid4().hex, alert_type, level, message, merged_details))

        if not accepted:
            return []

        # Nén & ghi nguyên tử một tệp JPEG dùng chung cho các cảnh báo phát sinh ở cùng tick suy luận
        shared_evidence_file: str | None = f"{accepted[0][0]}.jpg"
        top_level = max(it[2] for it in accepted)
        saved_ok = self._save_evidence_snapshot(
            shared_evidence_file,
            frame_bgr=frame_bgr,
            render_fn=render_fn,
            t_sec=float(t_sec),
            count_val=float(count_val),
            density_val=float(density_val),
            top_level=top_level,
        )
        if not saved_ok:
            shared_evidence_file = None

        evidence_url = f"/api/evidence/{shared_evidence_file}" if shared_evidence_file else None
        db_rows: list[tuple[Any, ...]] = []
        out_items: list[dict[str, Any]] = []

        src_lbl = str(source_label or "unknown")
        mdl_lbl = str(model_label or "CSRNet")[:48]
        t_val = float(t_sec)
        c_val = float(count_val)
        d_val = float(density_val)

        for alert_id, alert_type, level, message, merged_details in accepted:
            db_rows.append(
                (
                    alert_id,
                    sess_key,
                    t_val,
                    src_lbl,
                    mdl_lbl,
                    int(level),
                    alert_type,
                    message,
                    c_val,
                    d_val,
                    Jsonb(_json_safe(merged_details)),
                    shared_evidence_file,
                )
            )
            out_items.append(
                {
                    "id": alert_id,
                    "session_id": sess_key,
                    "t": round(t_val, 1),
                    "type": alert_type,
                    "level": int(level),
                    "message": message,
                    "count": round(c_val, 1),
                    "density": round(d_val, 3),
                    "source_label": src_lbl,
                    "model_label": mdl_lbl,
                    "evidence_url": evidence_url,
                }
            )

        self._queue.put(("alerts", db_rows))
        return out_items

    def finish_session(self, session_id: str, summary: dict[str, Any]) -> None:
        if self._closed or not session_id:
            return
        summary = summary or {}
        payload = (
            int(summary.get("samples", 0) or 0),
            float(summary.get("max_count", 0.0) or 0.0),
            int(summary.get("max_level", 0) or 0),
            _opt_float(summary.get("display_fps")),
            _opt_float(summary.get("infer_fps")),
            _opt_float(summary.get("infer_ms")),
            str(session_id)[:64],
        )
        self._queue.put(("finish_session", payload))

    # ------------------------------------------------------------------ Luồng nền thực thi SQL
    def _exec_start_session(self, conn: psycopg.Connection, payload: tuple[Any, ...]) -> None:
        q = sql.SQL(
            """
            INSERT INTO {s}.sessions (
                id, source_kind, source_label, model, model_label,
                area_m2, crowd_type, roi_mode, params
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """
        ).format(s=self._schema_ident)
        conn.execute(q, payload)

    def _exec_timeseries_batch(self, conn: psycopg.Connection, batch: list[tuple[Any, ...]]) -> None:
        if not batch:
            return
        q = sql.SQL(
            """
            INSERT INTO {s}.timeseries (
                session_id, t_sec, count_raw, count_smoothed, density,
                restricted_count, mean_speed, direction_entropy, counter_flow_ratio,
                level, surge, panic, counter_flow
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        ).format(s=self._schema_ident)
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(q, batch)
        except psycopg.errors.ForeignKeyViolation:
            # Nếu một phiên bị xóa giữa chừng, ghi từng dòng hợp lệ để không mất phần còn lại
            for row in batch:
                try:
                    conn.execute(q, row)
                except psycopg.errors.ForeignKeyViolation:
                    pass

    def _exec_alerts(self, conn: psycopg.Connection, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        q = sql.SQL(
            """
            INSERT INTO {s}.alerts (
                id, session_id, t_sec, source_label, model_label,
                level, alert_type, message, count_val, density_val, details, evidence_file
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """
        ).format(s=self._schema_ident)
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(q, rows)
        except psycopg.errors.ForeignKeyViolation:
            for r in rows:
                try:
                    conn.execute(q, r)
                except psycopg.errors.ForeignKeyViolation:
                    fallback = (r[0], None, *r[2:])
                    conn.execute(q, fallback)

    def _exec_finish_session(self, conn: psycopg.Connection, payload: tuple[Any, ...]) -> None:
        q = sql.SQL(
            """
            UPDATE {s}.sessions
            SET ended_at = NOW(),
                samples = %s,
                max_count = %s,
                max_level = %s,
                display_fps = %s,
                infer_fps = %s,
                infer_ms = %s
            WHERE id = %s
            """
        ).format(s=self._schema_ident)
        conn.execute(q, payload)

    def _run_with_conn(
        self,
        conn: psycopg.Connection | None,
        fn: Callable[[psycopg.Connection], None],
    ) -> psycopg.Connection | None:
        """Mở kết nối lười và tự phục hồi 1 lần nếu kết nối bị ngắt đột ngột."""
        for attempt in range(2):
            if conn is None or conn.closed:
                if not self._ensure_ready_lazy(min_interval=0.0 if attempt == 1 else 1.5):
                    return None
                try:
                    conn = self._connect_target(autocommit=True)
                except Exception as exc:
                    self._ready = False
                    self._last_error = f"{type(exc).__name__}: {exc}".strip()
                    return None
            try:
                fn(conn)
                self._ready = True
                self._last_error = None
                return conn
            except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
                self._ready = False
                self._last_error = f"{type(exc).__name__}: {exc}".strip()
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}".strip()
                return conn
        return conn

    def _worker_loop(self) -> None:
        conn: psycopg.Connection | None = None
        ts_batch: list[tuple[Any, ...]] = []
        last_flush = time.monotonic()

        def flush_ts() -> None:
            nonlocal conn, last_flush
            if ts_batch:
                batch_copy = list(ts_batch)
                ts_batch.clear()
                conn = self._run_with_conn(conn, lambda c: self._exec_timeseries_batch(c, batch_copy))
            last_flush = time.monotonic()

        try:
            while True:
                if ts_batch:
                    wait_sec = max(0.01, _BATCH_FLUSH_SEC - (time.monotonic() - last_flush))
                else:
                    wait_sec = _BATCH_FLUSH_SEC

                try:
                    op, data = self._queue.get(timeout=wait_sec)
                except queue.Empty:
                    if ts_batch:
                        flush_ts()
                    continue

                if op == "timeseries":
                    if not ts_batch:
                        last_flush = time.monotonic()
                    ts_batch.append(data)
                    if len(ts_batch) >= _BATCH_MAX_ROWS or (time.monotonic() - last_flush) >= _BATCH_FLUSH_SEC:
                        flush_ts()
                elif op == "start_session":
                    flush_ts()
                    conn = self._run_with_conn(conn, lambda c, d=data: self._exec_start_session(c, d))
                elif op == "alerts":
                    if ts_batch and (time.monotonic() - last_flush) >= _BATCH_FLUSH_SEC:
                        flush_ts()
                    conn = self._run_with_conn(conn, lambda c, d=data: self._exec_alerts(c, d))
                elif op == "finish_session":
                    flush_ts()
                    conn = self._run_with_conn(conn, lambda c, d=data: self._exec_finish_session(c, d))
                elif op == "flush":
                    flush_ts()
                    if isinstance(data, threading.Event):
                        data.set()
                elif op == "reset_conn":
                    flush_ts()
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    if isinstance(data, threading.Event):
                        data.set()
                elif op == "stop":
                    flush_ts()
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None
                    if isinstance(data, threading.Event):
                        data.set()
                    break
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _drain_sync(self) -> None:
        """Xử lý đồng bộ hàng đợi khi `start_worker=False`."""
        conn: psycopg.Connection | None = None
        ts_batch: list[tuple[Any, ...]] = []
        try:
            while True:
                try:
                    op, data = self._queue.get_nowait()
                except queue.Empty:
                    break
                if op == "timeseries":
                    ts_batch.append(data)
                    if len(ts_batch) >= _BATCH_MAX_ROWS:
                        batch_copy = list(ts_batch)
                        ts_batch.clear()
                        conn = self._run_with_conn(conn, lambda c, b=batch_copy: self._exec_timeseries_batch(c, b))
                else:
                    if ts_batch:
                        batch_copy = list(ts_batch)
                        ts_batch.clear()
                        conn = self._run_with_conn(conn, lambda c, b=batch_copy: self._exec_timeseries_batch(c, b))
                    if op == "start_session":
                        conn = self._run_with_conn(conn, lambda c, d=data: self._exec_start_session(c, d))
                    elif op == "alerts":
                        conn = self._run_with_conn(conn, lambda c, d=data: self._exec_alerts(c, d))
                    elif op == "finish_session":
                        conn = self._run_with_conn(conn, lambda c, d=data: self._exec_finish_session(c, d))
                    elif isinstance(data, threading.Event):
                        data.set()
            if ts_batch:
                batch_copy = list(ts_batch)
                ts_batch.clear()
                conn = self._run_with_conn(conn, lambda c, b=batch_copy: self._exec_timeseries_batch(c, b))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def flush(self, timeout: float = 5.0) -> None:
        """Đợi cho tới khi toàn bộ tác vụ và lô chuỗi thời gian trong hàng đợi đã commit."""
        if self._worker is not None and self._worker.is_alive():
            done = threading.Event()
            self._queue.put(("flush", done))
            done.wait(timeout=max(0.1, float(timeout)))
        elif not self._closed:
            self._drain_sync()

    def close(self) -> None:
        """Đẩy hết dữ liệu còn lại và dừng luồng nền an toàn."""
        if self._closed:
            return
        if self._worker is not None and self._worker.is_alive():
            done = threading.Event()
            self._queue.put(("stop", done))
            done.wait(timeout=5.0)
            self._worker.join(timeout=5.0)
        else:
            self._drain_sync()
        self._closed = True

    # ------------------------------------------------------------------ Truy vấn dữ liệu
    def list_alerts(
        self,
        *,
        limit: int = 100,
        level: int | None = None,
        alert_type: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._ensure_ready_lazy():
            return []

        lim = max(1, min(int(limit), 1000))
        clauses: list[sql.Composable] = []
        params: list[Any] = []
        if level is not None:
            clauses.append(sql.SQL("level = %s"))
            params.append(int(level))
        if alert_type:
            clauses.append(sql.SQL("alert_type = %s"))
            params.append(str(alert_type))
        if session_id:
            clauses.append(sql.SQL("session_id = %s"))
            params.append(str(session_id))

        where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
        query = sql.SQL(
            """
            SELECT
                id, session_id, created_at, t_sec, source_label, model_label,
                level, alert_type, message, count_val, density_val, details, evidence_file
            FROM {s}.alerts
            {w}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """
        ).format(s=self._schema_ident, w=where_sql)
        params.append(lim)

        try:
            with self._connect_target(autocommit=True, row_factory=dict_row) as conn:
                rows = conn.execute(query, params).fetchall()
        except Exception as exc:
            self._ready = False
            self._last_error = f"{type(exc).__name__}: {exc}".strip()
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            ev_file = r.get("evidence_file")
            out.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "created_at": _iso_or_none(r.get("created_at")),
                    "t_sec": float(r["t_sec"]),
                    "source_label": r["source_label"],
                    "model_label": r["model_label"],
                    "level": int(r["level"]),
                    "alert_type": r["alert_type"],
                    "message": r["message"],
                    "count_val": float(r["count_val"]),
                    "density_val": float(r["density_val"]),
                    "details": r["details"] if isinstance(r.get("details"), dict) else {},
                    "evidence_file": ev_file,
                    "evidence_url": f"/api/evidence/{ev_file}" if ev_file else None,
                }
            )
        return out

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self._ensure_ready_lazy():
            return []

        lim = max(1, min(int(limit), 500))
        query = sql.SQL(
            """
            SELECT
                s.id,
                s.created_at,
                s.ended_at,
                s.source_kind,
                s.source_label,
                s.model,
                s.model_label,
                s.area_m2,
                s.crowd_type,
                s.roi_mode,
                s.params,
                s.samples,
                s.max_count,
                s.max_level,
                s.display_fps,
                s.infer_fps,
                s.infer_ms,
                COALESCE(a.alert_count, 0)::int AS alert_count
            FROM {s}.sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*)::int AS alert_count
                FROM {s}.alerts
                WHERE session_id IS NOT NULL
                GROUP BY session_id
            ) a ON a.session_id = s.id
            ORDER BY s.created_at DESC
            LIMIT %s
            """
        ).format(s=self._schema_ident)

        try:
            with self._connect_target(autocommit=True, row_factory=dict_row) as conn:
                rows = conn.execute(query, (lim,)).fetchall()
        except Exception as exc:
            self._ready = False
            self._last_error = f"{type(exc).__name__}: {exc}".strip()
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "created_at": _iso_or_none(r.get("created_at")),
                    "ended_at": _iso_or_none(r.get("ended_at")),
                    "source_kind": r["source_kind"],
                    "source_label": r["source_label"],
                    "model": r["model"],
                    "model_label": r["model_label"],
                    "area_m2": float(r["area_m2"]),
                    "crowd_type": r["crowd_type"],
                    "roi_mode": r["roi_mode"],
                    "params": r["params"] if isinstance(r.get("params"), dict) else {},
                    "samples": int(r["samples"]),
                    "max_count": float(r["max_count"]),
                    "max_level": int(r["max_level"]),
                    "display_fps": _opt_float(r.get("display_fps")),
                    "infer_fps": _opt_float(r.get("infer_fps")),
                    "infer_ms": _opt_float(r.get("infer_ms")),
                    "alert_count": int(r.get("alert_count") or 0),
                }
            )
        return out

    def get_session_timeseries(self, session_id: str) -> list[dict[str, Any]]:
        if not session_id or not self._ensure_ready_lazy():
            return []

        query = sql.SQL(
            """
            SELECT
                id, session_id, recorded_at, t_sec, count_raw, count_smoothed,
                density, restricted_count, mean_speed, direction_entropy,
                counter_flow_ratio, level, surge, panic, counter_flow
            FROM {s}.timeseries
            WHERE session_id = %s
            ORDER BY t_sec ASC, id ASC
            """
        ).format(s=self._schema_ident)

        try:
            with self._connect_target(autocommit=True, row_factory=dict_row) as conn:
                rows = conn.execute(query, (str(session_id),)).fetchall()
        except Exception as exc:
            self._ready = False
            self._last_error = f"{type(exc).__name__}: {exc}".strip()
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "session_id": r["session_id"],
                    "recorded_at": _iso_or_none(r.get("recorded_at")),
                    "t_sec": float(r["t_sec"]),
                    "count_raw": float(r["count_raw"]),
                    "count_smoothed": float(r["count_smoothed"]),
                    "density": float(r["density"]),
                    "restricted_count": float(r["restricted_count"]),
                    "mean_speed": _opt_float(r.get("mean_speed")),
                    "direction_entropy": _opt_float(r.get("direction_entropy")),
                    "counter_flow_ratio": _opt_float(r.get("counter_flow_ratio")),
                    "level": int(r["level"]),
                    "surge": bool(r["surge"]),
                    "panic": bool(r["panic"]),
                    "counter_flow": bool(r["counter_flow"]),
                }
            )
        return out

    def clear_history(self) -> dict[str, int]:
        """Xóa toàn bộ cảnh báo, phiên phân tích trong schema hiện tại và ảnh chụp bằng chứng."""
        self.flush()
        with self._cooldown_lock:
            self._cooldown_state.clear()

        deleted_alerts = 0
        deleted_sessions = 0
        if self._ensure_ready_lazy():
            try:
                with self._connect_target(autocommit=True) as conn:
                    with conn.transaction():
                        with conn.cursor() as cur:
                            cur.execute(sql.SQL("DELETE FROM {s}.alerts").format(s=self._schema_ident))
                            deleted_alerts = max(0, cur.rowcount or 0)
                            cur.execute(sql.SQL("DELETE FROM {s}.sessions").format(s=self._schema_ident))
                            deleted_sessions = max(0, cur.rowcount or 0)
            except Exception as exc:
                self._ready = False
                self._last_error = f"{type(exc).__name__}: {exc}".strip()

        if os.path.isdir(self.evidence_dir):
            for name in os.listdir(self.evidence_dir):
                if name.lower().endswith((".jpg", ".tmp")):
                    try:
                        os.remove(os.path.join(self.evidence_dir, name))
                    except OSError:
                        pass

        return {
            "deleted_sessions": int(deleted_sessions),
            "deleted_alerts": int(deleted_alerts),
        }

    def drop_schema(self) -> None:
        """Xóa schema thử nghiệm (`DROP SCHEMA IF EXISTS <schema> CASCADE`) — cấm chạy trên `public`."""
        if self.schema.lower() == "public":
            raise ValueError("Không được phép xóa schema 'public'.")

        if self._worker is not None and self._worker.is_alive():
            done = threading.Event()
            self._queue.put(("reset_conn", done))
            done.wait(timeout=5.0)
        elif not self._closed:
            self._drain_sync()

        with self._connect_target(autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(self._schema_ident))
        self._ready = False


EventStore = PostgresEventStore

__all__ = ["PostgresEventStore", "EventStore"]
