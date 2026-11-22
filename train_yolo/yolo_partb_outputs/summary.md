# Kết quả đếm người trên ShanghaiTech Part B

Tập test chính thức gồm 316 ảnh.

- CSRNet đã lưu tại epoch 116 có MAE 9.5539 và RMSE 14.8724 người/ảnh.
- YOLO11s tại epoch 130, confidence 0.20, có MAE 8.3956 và RMSE 14.3475 người/ảnh.
- Theo MAE, YOLO11s thấp hơn (Δ = -1.1583 người/ảnh).
- Theo RMSE, YOLO11s thấp hơn (Δ = -0.5249 người/ảnh).

Hộp huấn luyện YOLO được xấp xỉ từ tâm đầu người. Checkpoint và confidence được chọn trên validation. Sai số test tính từ số điểm gốc trong file MAT.
