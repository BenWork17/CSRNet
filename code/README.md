# Hệ thống đếm người, giám sát vùng đa giác & cảnh báo đám đông bất thường (CSRNet · YOLO11s · YouTube Live · Optical Flow · PostgreSQL 16)

```
code/
├── train_csrnet_shanghaitech.ipynb  # B1–B3: tải dữ liệu, tiền xử lý, huấn luyện, đánh giá & xuất trọng số CSRNet
├── crowd_core.py                    # CSRNet, YOLO11s, tối ưu ONNX/TorchScript & bộ giám sát CrowdMonitor
├── roi_motion.py                    # Đa giác ROI / Vùng cấm & phân tích chuyển động Farneback Optical Flow
├── storage.py                       # Lưu trữ bất đồng bộ PostgreSQL 16 (sessions, timeseries, alerts) & ảnh bằng chứng
├── video_source.py                  # Đọc nguồn video / camera mới nhất & đặt timeout FFmpeg cho RTSP/HTTP/HLS
├── youtube_source.py                # Chuẩn hóa liên kết/iframe & lấy địa chỉ HLS trực tiếp YouTube Live qua yt-dlp
├── server.py                        # B4: Web app Flask (ảnh, video, webcam/RTSP, YouTube Live) + SSE + REST API lịch sử
├── evaluate_pipeline.py             # Kịch bản đánh giá định lượng độ chính xác, tốc độ & phát hiện bất thường (outputCsrnet/)
├── test_video_source.py             # Kiểm thử đơn vị cho nguồn video, camera và YouTube Live
├── test_full_system.py              # Kiểm thử tích hợp toàn hệ thống (ROI, Optical Flow, PostgreSQL 16, Flask API)
├── templates/index.html             # Giao diện giám sát 3 phân khu (Ảnh tĩnh, Video/Camera/YouTube Live, Kho Lịch sử)
├── static/css/, static/js/          # style.css, app.js (vẽ đa giác ROI trực quan, biểu đồ, Lightbox bằng chứng)
├── static/assets/                   # Nền giấy, hạt nhiễu, ảnh minh họa giao diện
├── data/evidence/                   # Ảnh chụp bằng chứng JPEG tự động sinh khi có cảnh báo
├── .env.example                     # Mẫu cấu hình kết nối PostgreSQL 16
├── app.py                           # Bản demo Streamlit cũ (không bắt buộc)
├── requirements.txt                 # Danh sách thư viện Python
└── weights/                         # csrnet_partB.pth (tự xuất .onnx) và vị trí tùy chọn cho yolo11s_partB_best_count.pt
```

---

## 1. Kiến trúc tính năng chính

1. **Hai mô hình đếm người bổ trợ (`crowd_core.py`):**
   - **CSRNet** (Li et al., CVPR 2018): ước lượng bản đồ mật độ 2D cho đám đông dày đặc; tự động xuất và chạy trên **ONNX Runtime** (nhanh hơn PyTorch thuần ~1,5–2 lần trên CPU, kết quả khớp tuyệt đối, mặc định cạnh dài `512 px`).
   - **YOLO11s Head**: phát hiện hộp đầu người trực tiếp cho đám đông thưa đến trung bình (mặc định cạnh dài `1024 px`, `conf=0.20`, `iou=0.65`, tối đa `1.200` hộp/ảnh). App tìm CSRNet tại `code/weights/csrnet_partB.pth` và YOLO tại `code/weights/` hoặc `train_yolo/yolo_partb_outputs/yolo11s_partB_best_count.pt`.

2. **Đa nguồn đầu vào trực tiếp (Ảnh, Video, Camera/RTSP, YouTube Live):**
   - **Ảnh tĩnh:** hiển thị số người, mật độ người/m², lớp phủ (bản đồ nhiệt hoặc hộp đầu người), đa giác ROI và mức cảnh báo.
   - **Video / Camera:** xử lý tệp tải lên (`.mp4`, `.avi`, `.mov`, `.mkv`), webcam USB (mở nhanh ~1 s qua `CAP_DSHOW` trên Windows) hoặc camera mạng RTSP/HTTP (timeout FFmpeg `5s` mở / `2s` đọc). Chế độ riêng tư làm mờ nhẹ bằng thu nhỏ/phóng to (nhanh hơn `GaussianBlur` ~2,5 lần).
   - **YouTube Live (`youtube_source.py`):** chọn nguồn **YouTube Live**, giữ liên kết camera Phương Trần Đà Nẵng có sẵn hoặc dán liên kết video / thẻ `<iframe>` của một luồng đang phát trực tiếp. Máy chủ dùng `yt-dlp` lấy luồng HLS H.264 `<= 720p` mới mỗi phiên và giải mã trực tiếp qua FFmpeg (timeout mở `18s`, đọc `12s`) để hiển thị khung nguồn, khung nhận diện và thống kê thời gian thực.

3. **Đa giác vùng quan tâm (Polygon ROI) & Vùng cấm xâm nhập (`roi_motion.py`):**
   - Cho phép chấm đỉnh vẽ đa giác bất kỳ trực tiếp trên ảnh tĩnh hoặc khung nguồn video/camera/YouTube Live:
     - `full`: giám sát toàn khung hình với diện tích `area` (m²).
     - `roi`: chỉ tích phân mật độ CSRNet hoặc lọc tâm hộp đầu YOLO nằm trong đa giác ROI, tính mật độ theo diện tích riêng `roi_area` (m²).
     - `forbidden`: giữ nguyên đếm toàn cảnh nhưng đếm riêng số người lọt vào vùng cấm (`restricted_count`); kích hoạt cảnh báo **`roi_intrusion`** và khoanh đỏ các hộp vi phạm khi vượt ngưỡng `restricted_thr`.

4. **Phân tích chuyển động đám đông bằng Optical Flow đặc (`MotionAnalyzer`):**
   - Ước lượng luồng quang học đặc **Farneback** giữa cặp khung hình gần nhất trong bộ đệm vòng (`frame_ring`, khoảng cách lý tưởng `~0.08s`), kết hợp mặt nạ đám đông (`crowd_mask`) để loại nhiễu hậu cảnh.
   - Trích xuất thời gian thực:
     - **Tốc độ dịch chuyển chuẩn hóa** (`mean_speed`, `p90_speed`, đơn vị `%/s` theo đường chéo khung).
     - **Entropy hướng Shannon 12-bin** (`direction_entropy` ∈ `[0, 1]`): đo mức độ hỗn loạn đa hướng.
     - **Tỉ lệ di chuyển ngược chiều** (`counter_flow_ratio`): tỉ lệ dòng người ngược quá `120°` so với vector dòng chủ đạo EMA.
     - **Điểm bất thường trực tuyến** (`anomaly_score`) và dự báo thời gian chạm ngưỡng Đỏ (`overload_eta_s`).
   - Phát hiện tự động 5 nhóm sự kiện: **Quá tải mật độ** (`density`), **Tăng đột biến** (`surge`), **Xâm nhập vùng cấm** (`roi_intrusion`), **Hoảng loạn / Chạy tán loạn** (`motion_panic`), và **Ngược chiều** (`counter_flow`).

5. **Tốc độ khung hình 3 vòng tách rời (`VideoSession` trong `server.py`):**
   - CSRNet chạy trên CPU đạt ~0,3–1,5 lần suy luận/giây, nên ứng dụng tách rời 3 vòng độc lập và **chừa 3 nhân CPU** cho vòng hiển thị để không bị nghẽn khung:

   | Vòng | Công việc | Tốc độ |
   |---|---|---|
   | Đọc khung | Giải mã nguồn video/camera/HLS, cập nhật khung mới nhất & `frame_ring` | Theo nhịp nguồn |
   | Suy luận | CSRNet/YOLO + lọc ROI + Farneback Optical Flow + đẩy hàng đợi PostgreSQL | ~1,5 lần/s (CPU), nhanh hơn nhiều trên GPU |
   | Hiển thị | Phát khung nguồn đều đặn và gửi riêng khung nhận diện AI khi có kết quả mới | Theo FPS mục tiêu (15–30 FPS thực tế trên Ryzen 5 5600H CPU) |

6. **Lưu trữ bất đồng bộ PostgreSQL 16 & Ảnh bằng chứng (`storage.py`):**
   - Tự động khởi tạo database `crowd_monitor` và 3 bảng `sessions`, `timeseries`, `alerts` khi máy chủ khởi động (nếu PostgreSQL tạm tắt, ứng dụng vẫn hoạt động bình thường và tự kết nối lại khi DB sẵn sàng).
   - Hàng đợi nền (`queue.Queue`) gom lô `timeseries` (`executemany`) và ghi `sessions`/`alerts` với độ trễ `~0 ms` lên luồng hiển thị video.
   - Mỗi cảnh báo (có cơ chế chống lặp `cooldown` 4 giây theo từng loại và tự vượt cooldown khi tăng mức nguy hiểm) được chụp lại **ảnh bằng chứng JPEG** đã vẽ lớp phủ nhiệt/hộp + đa giác ROI + mũi tên Optical Flow vào `code/data/evidence/`.

---

## 2. Cài đặt & Cấu hình `.env`

### Bước 1: Cài đặt thư viện Python & JS runtime cho YouTube Live
```bash
cd code
pip install -r requirements.txt
```
> **Lưu ý cho nguồn YouTube Live:** Cài đặt [Node.js 22+ hoặc Deno 2.3+](https://github.com/yt-dlp/yt-dlp/wiki/EJS) trên máy chạy Flask để `yt-dlp` giải quyết thử thách JS khi lấy luồng HLS trực tiếp từ YouTube.

### Bước 2: Cấu hình PostgreSQL 16 qua `.env`
Sao chép tệp `.env.example` thành `.env` trong thư mục `code/` và cập nhật thông tin kết nối PostgreSQL:
```ini
CROWD_DATABASE_URL=postgresql://dss:your_password_here@127.0.0.1:5433/crowd_monitor
```
> Hệ thống tự tạo database `crowd_monitor` và các bảng/chỉ mục nếu chưa tồn tại. Nếu muốn chạy không cần cơ sở dữ liệu, đặt biến môi trường `CROWD_NO_DB=1`.

### Bước 3: Khởi chạy máy chủ
```bash
python server.py          # mở http://127.0.0.1:5000
```

Các biến môi trường hỗ trợ:

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `CROWD_DATABASE_URL` | Đọc từ `code/.env` | Chuỗi kết nối PostgreSQL 16 (`crowd_monitor`) |
| `CROWD_NO_DB` | `0` | Đặt `1` để tắt lưu trữ PostgreSQL |
| `CROWD_WEIGHTS` | `weights/csrnet_partB.pth` | Đường dẫn trọng số CSRNet |
| `CROWD_YOLO_WEIGHTS` | `weights/yolo11s_partB_best_count.pt` (hoặc `../train_yolo/...`) | Đường dẫn trọng số YOLO11s đầu người |
| `CROWD_HOST` | `127.0.0.1` | Địa chỉ lắng nghe của Flask |
| `CROWD_PORT` | `5000` | Cổng dịch vụ web |
| `CROWD_THREADS` | Tự động (chừa 3 nhân CPU) | Số luồng CPU cho suy luận CSRNet |
| `CROWD_NO_ONNX` | `0` | Đặt `1` để tắt ONNX Runtime, chỉ dùng PyTorch/TorchScript |
| `CONTACT_EMAIL` | *(trống)* | Email hiển thị ở thanh liên hệ (bấm để sao chép) |

---

## 3. Danh mục REST API & Server-Sent Events

| Phương thức & Đường dẫn | Mô tả |
|---|---|
| `GET /api/status` | Trạng thái các mô hình (`csrnet`, `yolo`), ngưỡng mật độ, bảng màu cảnh báo và trạng thái kết nối PostgreSQL (`database`). |
| `POST /api/image` | Phân tích ảnh tĩnh (`multipart/form-data`: `file` + tham số cấu hình/ROI). Trả về số người, mật độ, mức cảnh báo, ảnh phủ (`overlay`), thông tin ROI (`roi_mode`, `restricted_count`, `roi_intrusion`) và danh sách `alerts` kèm đường dẫn bằng chứng nếu vượt ngưỡng. |
| `POST /api/jobs` | Tạo phiên xử lý video tải lên (`source=file`), camera/RTSP/HTTP (`source=camera`, `url=...`) hoặc YouTube Live (`source=youtube`, `url=...` hoặc thẻ `<iframe>`) → trả về `{id, model}`. |
| `GET /api/jobs/<id>/events` | Luồng **Server-Sent Events** phát các sự kiện `start`, `tick` (khung nguồn, khung AI, số người, mật độ, thống kê chuyển động `motion`, dự báo `overload_eta_s`, `alert_records`), `done` và `fail`. |
| `POST /api/jobs/<id>/stop` | Yêu cầu dừng phiên phân tích video/camera/YouTube Live đang chạy. |
| `GET /api/history/alerts` | Truy vấn danh sách cảnh báo từ PostgreSQL; hỗ trợ lọc theo `limit` (mặc định `100`), `level`, `type` / `alert_type`, `session_id`. |
| `GET /api/history/sessions` | Danh sách các phiên phân tích đã ghi nhận (`limit` mặc định `50`) kèm số lượng cảnh báo mỗi phiên. |
| `GET /api/history/sessions/<sid>/timeseries` | Toàn bộ chuỗi thời gian (`t_sec`, số đếm, mật độ, vận tốc, entropy hướng, tỉ lệ ngược chiều, cờ cảnh báo) của một phiên. |
| `DELETE /api/history` | Xóa toàn bộ lịch sử phiên, chuỗi thời gian, cảnh báo trong PostgreSQL và dọn ảnh bằng chứng. |
| `GET /api/evidence/<filename>` | Phục vụ tệp ảnh bằng chứng `.jpg` an toàn (kiểm tra chống leo thư mục). |

---

## 4. Huấn luyện mô hình, Kiểm thử & Đánh giá (`outputCsrnet/`)

### B1–B3: Huấn luyện CSRNet (Google Colab, GPU T4)
1. Mở `train_csrnet_shanghaitech.ipynb` trên Colab → Runtime → Change runtime type → **T4 GPU** → **Run all** (tự tải ShanghaiTech qua `kagglehub`, mặc định 150 epoch ~1 giờ).
2. Tải kết quả huấn luyện (lưu đối chiếu tại thư mục `outputCsrnet/`: `csrnet_partB.pth`, `results.json`, `hist_counts.png`, `density_gt.png`, `training_curves.png`, `scatter_test.png`, `test_samples.png`, `alert_levels.png`) và đặt `csrnet_partB.pth` vào `code/weights/`. Ở lần khởi động đầu tiên, `server.py` tự xuất `weights/csrnet_partB.onnx` để tăng tốc suy luận CPU.

### Kiểm thử & Đánh giá định lượng toàn hệ thống
```bash
# Chạy toàn bộ kiểm thử đơn vị & tích hợp (nguồn video, YouTube Live, ROI, Optical Flow, PostgreSQL, API)
python -X utf8 -m unittest discover -s code -p "test_*.py" -v

# Chạy kịch bản đánh giá toàn trình (đọc outputCsrnet/demo_surge.mp4, xuất outputCsrnet/anomaly_eval.json & anomaly_eval_curves.png)
python -X utf8 code/evaluate_pipeline.py
```
