"""Lấy địa chỉ luồng HLS của một buổi phát trực tiếp trên YouTube."""
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit


DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=G_G8A6JU_LI"
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
INVALID_URL = "Hãy nhập liên kết video YouTube hoặc mã nhúng iframe hợp lệ."
_STREAM_CONTEXT = {}  # m3u8_url -> {"headers": dict, "watch_url": str, "fps": float}


class YouTubeStreamError(RuntimeError):
    pass


class _EmbedParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.src = None

    def handle_starttag(self, tag, attrs):
        if tag == "iframe" and self.src is None:
            self.src = dict(attrs).get("src")


def youtube_url(value):
    """Chỉ chuyển ID, liên kết video hoặc iframe YouTube thành URL xem chuẩn."""
    value = value.strip()
    if value.startswith("<"):
        parser = _EmbedParser()
        parser.feed(value)
        value = (parser.src or "").strip()

    video_id = value
    if not VIDEO_ID.fullmatch(video_id):
        try:
            url = urlsplit(value)
        except ValueError:
            raise ValueError(INVALID_URL) from None
        if url.scheme not in ("http", "https"):
            raise ValueError(INVALID_URL)
        host = url.netloc.lower()
        if host == "youtu.be":
            video_id = url.path[1:]
        elif host in ("youtube.com", "www.youtube.com", "m.youtube.com",
                      "youtube-nocookie.com", "www.youtube-nocookie.com"):
            if url.path == "/watch":
                video_id = parse_qs(url.query).get("v", [""])[0]
            elif url.path.startswith(("/embed/", "/live/")):
                video_id = url.path.split("/", 2)[-1]
        if not VIDEO_ID.fullmatch(video_id):
            raise ValueError(INVALID_URL)
    return f"https://www.youtube.com/watch?v={video_id}"


def _extract_hls_info(url, player_clients=None):
    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": 8,
        "retries": 1,
        "extractor_retries": 1,
        "ignore_no_formats_error": True,
        "js_runtimes": {"deno": {}, "node": {}},
        "format": ("bv[protocol^=m3u8][vcodec^=avc1][height<=720][fps<=?30]"
                   "/b[protocol^=m3u8][vcodec^=avc1][height<=720][fps<=?30]"
                   "/b[protocol^=m3u8][height<=720]"
                   "/bv*[protocol^=m3u8]"),
    }
    if player_clients:
        options["extractor_args"] = {"youtube": {"player_client": list(player_clients)}}
    with YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


def resolve_youtube_live(url):
    """Lấy địa chỉ HLS mới mỗi phiên, ưu tiên android_vr (nhanh, không chặn 403 sau 2 phân đoạn)."""
    try:
        from yt_dlp.utils import DownloadError
    except ImportError:
        raise YouTubeStreamError("Thiếu yt-dlp trên máy chủ. Hãy cài thư viện trong requirements.txt.") from None

    info = None
    try:
        try:
            info = _extract_hls_info(url, player_clients=["android_vr"])
        except Exception:
            info = None
        if not info or not info.get("url"):
            info = _extract_hls_info(url, player_clients=None)
    except DownloadError:
        raise YouTubeStreamError("Không kết nối được luồng YouTube. Hãy kiểm tra đường dẫn và thử lại.") from None

    if not info or info.get("live_status") != "is_live":
        raise YouTubeStreamError("Video YouTube này hiện không phát trực tiếp.")
    m3u8_url = info.get("url")
    if not m3u8_url or not (info.get("protocol") or "").startswith("m3u8"):
        raise YouTubeStreamError("YouTube không cung cấp luồng HLS có thể đọc cho video này.")

    if len(_STREAM_CONTEXT) > 32:
        _STREAM_CONTEXT.clear()
    _STREAM_CONTEXT[m3u8_url] = {
        "headers": dict(info.get("http_headers") or {}),
        "watch_url": url,
        "fps": float(info.get("fps") or 25.0),
        "width": int(info.get("width") or 1280),
        "height": int(info.get("height") or 720),
    }
    return m3u8_url


def get_stream_context(m3u8_url):
    """Trả về thông tin HTTP headers và URL gốc gắn với luồng HLS đã phân giải."""
    return _STREAM_CONTEXT.get(m3u8_url) or {}
