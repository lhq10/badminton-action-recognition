# Dự án Nhận diện Hành động Cầu lông (Badminton Action Recognition)

Dự án này sử dụng mô hình học sâu để nhận diện các hành động/kỹ thuật khác nhau trong môn cầu lông từ video. Quy trình bao gồm trích xuất khung hình từ video, phát hiện và trích xuất keypoints của người chơi, xử lý dữ liệu keypoints và huấn luyện một mô hình BiLSTM với cơ chế Attention để phân loại hành động.

## Cấu trúc Thư mục

```text
project_root/
├── data/ # Thư mục chứa dữ liệu
│ ├── filter_dataset/ # Đặt dataset video gốc tại đây
│ │ ├── Class1/ # Ví dụ: Smash
│ │ │ └── video1.mp4
│ │ └── ...
│ ├── frames_test/ # Frame được trích xuất sẽ lưu ở đây
│ └── keypoints_test/ # Keypoints dạng JSON sẽ lưu ở đây
│ └── processed_data/ # Dữ liệu đã xử lý cho huấn luyện sẽ lưu ở đây
├── frame_extractor.py # Script trích xuất frame từ video
├── rtmlib_utils.py # Tiện ích và khởi tạo mô hình RTMLib
├── keypoint_extractor.py # Script trích xuất keypoints từ frame
├── data_processor.py # Script xử lý keypoints thành dữ liệu huấn luyện
├── model.py # Định nghĩa kiến trúc mô hình BiLSTM + Attention
├── train.py # Script huấn luyện mô hình
├── predict.py # Script dự đoán trên một video ngẫu nhiên
├── config.py # Tệp cấu hình đường dẫn và tham số
├── requirements.txt # Danh sách các thư viện cần thiết
├── best_model.pth # Model được huấn luyện tốt nhất sẽ lưu ở đây
├── training_plots.png # Biểu đồ accuracy/loss trong quá trình huấn luyện
└── confusion_matrix.png # Ma trận nhầm lẫn trên tập kiểm tra
```

### Yêu cầu Hệ thống
Python 3.8+
OpenCV
RTMLib
ONNXRuntime (và onnxruntime-gpu nếu sử dụng GPU cho ONNX)
PyTorch
Scikit-learn
TensorFlow (chỉ cho to_categorical, có thể thay thế)
NumPy
Matplotlib
TQDM

### Cài đặt
Clone repository:

```bash
git clone https://github.com/lhq10/badminton-action-recognition.git
cd badminton-action-recognition
```

### Tạo và kích hoạt môi trường ảo (khuyến nghị):
```bash
python -m venv venv
```
#### Trên Windows
```bash
venv\Scripts\activate
```
#### Trên macOS/Linux
```bash
source venv/bin/activate
```
### Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
Nếu bạn muốn sử dụng GPU cho ONNXRuntime, hãy đảm bảo bạn đã cài đặt CUDA và cuDNN tương thích, sau đó cài đặt onnxruntime-gpu.

### Chuẩn bị dữ liệu:

Tạo thư mục data/filter_dataset trong thư mục gốc của dự án.

Bên trong filter_dataset, tạo các thư mục con đặt tên theo từng lớp hành động (ví dụ: Smash, Drop Shot, Clear, v.v.).

Sao chép các tệp video huấn luyện của bạn vào các thư mục lớp tương ứng.

Kiểm tra và chỉnh sửa ```bash config.py ``` (Nếu cần):

Mở tệp ```bash config.py ```

Đảm bảo các biến đường dẫn như VIDEO_INPUT_DIR_FE, FRAME_OUTPUT_DIR_FE, KEYPOINT_INPUT_DIR_DP, VIDEO_PREDICT_INPUT_DIR_PRED được thiết lập chính xác nếu bạn thay đổi cấu trúc thư mục mặc định.

Kiểm tra các tham số MAX_FRAMES_PER_VIDEO, MODEL_HIDDEN_DIM, v.v., nếu bạn muốn thử nghiệm với các giá trị khác nhau.

Thiết lập DEVICE ('cuda' hoặc 'cpu') và RTMLIB_BACKEND.

### Quy trình Thực thi

Chạy các script sau từ thư mục gốc của dự án (project_root) theo thứ tự:

#### Trích xuất Frame từ Video:
```bash
python frame_extractor.py
```
Thao tác này sẽ đọc video từ data/filter_dataset, trích xuất các frame và lưu chúng vào data/frames_test (hoặc đường dẫn bạn cấu hình trong config.py).

#### Trích xuất Keypoints từ Frame:

Trước khi chạy, đảm bảo rtmlib_utils.py có thể khởi tạo mô hình wholebody_model thành công. Nếu có lỗi, kiểm tra cài đặt ONNXRuntime và CUDA (nếu dùng GPU).
```bash
python keypoint_extractor.py
```
Script này sẽ đọc các frame từ data/frames_test, sử dụng RTMLib để trích xuất keypoints, và lưu kết quả dưới dạng tệp JSON vào data/keypoints_test.

#### Xử lý Dữ liệu Keypoints:
```bash
python data_processor.py
```
Script này sẽ đọc các tệp JSON keypoints, xử lý chúng thành các chuỗi đặc trưng (bao gồm chuẩn hóa và vector chuyển động), thực hiện padding, mã hóa nhãn, và chia thành các tập huấn luyện/kiểm tra. Dữ liệu đã xử lý sẽ được lưu vào data/processed_data/.

#### Huấn luyện Model:
```bash
python train.py
```
Script này sẽ tải dữ liệu đã xử lý, khởi tạo mô hình BiLSTM + Attention, và bắt đầu quá trình huấn luyện. Model tốt nhất (dựa trên validation accuracy) sẽ được lưu vào best_model.pth. Biểu đồ accuracy/loss và ma trận nhầm lẫn cũng sẽ được tạo.

#### Dự đoán trên một Video Ngẫu nhiên (Tùy chọn):
Sau khi đã huấn luyện và có tệp best_model.pth:
```bash
python predict.py
```
Script này sẽ chọn một video ngẫu nhiên từ data/filter_dataset, trích xuất frame và keypoints cho video đó, xử lý keypoints, sau đó sử dụng model đã huấn luyện để dự đoán hành động và so sánh với nhãn thực tế.

### Ghi chú
Quá trình trích xuất frame và keypoint có thể mất nhiều thời gian tùy thuộc vào số lượng và độ dài của video.

Huấn luyện model cũng có thể cần thời gian đáng kể, đặc biệt nếu không có GPU.

Mô hình RTMLib (wholebody_model) sẽ tự động tải xuống các tệp trọng số cần thiết trong lần chạy đầu tiên (nếu thư mục cache ~/.cache/rtmlib chưa có).

Hiệu suất của mô hình phụ thuộc vào chất lượng và số lượng dữ liệu huấn luyện, cũng như các tham số được chọn.
