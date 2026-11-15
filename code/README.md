# Đếm người & cảnh báo đám đông bất thường (CSRNet + luật mật độ)

```
code/
├── train_csrnet_shanghaitech.ipynb  # B1–B3: tải dữ liệu, tiền xử lý, huấn luyện, đánh giá, xuất trọng số
├── crowd_core.py                    # CSRNet, suy luận (InferenceEngine), luật cảnh báo (CrowdMonitor)
├── server.py                        # B4: web app Flask (ảnh, video, webcam/RTSP) + API
├── templates/index.html             # giao diện
├── static/css, static/js            # style.css, app.js (không dùng framework)
├── static/assets/                   # nền giấy, hạt nhiễu, ảnh minh họa
├── app.py                           # bản demo Streamlit cũ (không bắt buộc)
├── requirements.txt
└── weights/                         # csrnet_partB.pth (+ .onnx tự sinh khi chạy lần đầu)
```

## B1–B3. Huấn luyện (Google Colab, GPU T4)
1. Mở `train_csrnet_shanghaitech.ipynb` trên Colab → Runtime → Change runtime type → **T4 GPU**.
2. Run all. Notebook tự tải ShanghaiTech qua `kagglehub`; nếu lỗi, tải thủ công rồi gán `DATA_ROOT`.
3. Mặc định 150 epoch (~1 giờ trên T4). Muốn chạy nhanh để thử: sửa `epochs` trong `CFG`.
4. Cuối notebook tải về `outputs.zip`, gồm:
   - `csrnet_partB.pth`: trọng số mô hình tốt nhất
   - `results.json`: MAE, RMSE, tốc độ (điền vào Bảng 4.3 của báo cáo)
   - `hist_counts.png`, `density_gt.png`, `training_curves.png`, `scatter_test.png`, `test_samples.png`, `alert_levels.png`: hình cho báo cáo

## B4. Chạy web app (máy cá nhân, CPU cũng chạy được)
```bash
pip install -r requirements.txt
# chép outputs/csrnet_partB.pth vào code/weights/
python server.py          # mở http://127.0.0.1:5000
```
- **Ảnh:** thả/chọn ảnh → số người, mật độ người/m², bản đồ nhiệt, mức cảnh báo.
- **Video / Camera:** phân tích video tải lên hoặc webcam/RTSP theo thời gian thực (Server-Sent Events):
  khung hình, biểu đồ số người kèm ngưỡng cảnh báo, nhật ký sự kiện, tải CSV.
- **Cấu hình** (nút trên thanh điều hướng nổi): diện tích vùng quan sát, loại đám đông, chế độ riêng tư,
  độ phân giải suy luận, FPS hiển thị mục tiêu, ngưỡng & cửa sổ tăng đột biến, số lần đo xác nhận.
  Được lưu trong trình duyệt.
- Biến môi trường: `CROWD_WEIGHTS`, `CROWD_HOST` (mặc định 127.0.0.1), `CROWD_PORT` (5000),
  `CROWD_THREADS` (số luồng CPU cho suy luận), `CROWD_NO_ONNX=1` (tắt ONNX Runtime),
  `CONTACT_EMAIL` (hiện nút Email trên thanh liên hệ, bấm để sao chép).

### Tốc độ khung hình (video / camera)
CSRNet chạy trên CPU chỉ đạt ~0,3–1,5 lần suy luận/giây, nên app **tách rời ba vòng** thay vì
chờ suy luận xong mới phát khung (`VideoSession` trong `server.py`):

| Vòng | Việc | Tốc độ |
|---|---|---|
| đọc khung | giải mã nguồn, giữ khung mới nhất | theo nhịp nguồn |
| suy luận | CSRNet trên khung mới nhất | ~1,5 lần/s (CPU), nhanh hơn nhiều trên GPU |
| hiển thị | ghép bản đồ nhiệt gần nhất, nén JPEG | đúng FPS mục tiêu |

Kết quả đo trên Ryzen 5 5600H (CPU, ONNX Runtime): **15,1 FPS** ở mức 15, 20,1 ở mức 20,
25,1 ở mức 25 và 30,1 ở mức 30 — bản đồ nhiệt cập nhật khi có kết quả mới. Camera mở trong ~1 s
(DSHOW trên Windows).

Các tối ưu kèm theo:
- **ONNX Runtime** thay torch cho suy luận: nhanh hơn ~1,5–2 lần, kết quả khớp tuyệt đối
  (tự xuất `weights/csrnet_partB.onnx` ở lần chạy đầu, dùng lại ở các lần sau).
- **Chừa 3 nhân CPU** cho vòng hiển thị; nếu không, luồng phát bị đói CPU và tụt còn ~11 FPS.
- Bản đồ nhiệt chỉ tô màu lại khi có kết quả mới; làm mờ bằng thu nhỏ/phóng to (rẻ hơn
  `GaussianBlur` ~2,5 lần).
- Độ phân giải suy luận mặc định **512px** — đúng bằng ô cắt lúc huấn luyện. Hạ xuống 256px
  tuy nhanh hơn nhưng số đếm sai lệch rất mạnh (ví dụ `hajj.jpg`: 16 người so với 813 người ở 512px).

