import os

# ----- Cấu hình đường dẫn -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục gốc của dự án
DATA_DIR = os.path.join(BASE_DIR, "data")

# Đường dẫn cho frame_extractor.py
VIDEO_INPUT_DIR_FE = os.path.join(DATA_DIR, "filter_dataset") # Thay "temp_test" bằng "filter_dataset"
FRAME_OUTPUT_DIR_FE = os.path.join(DATA_DIR, "frames_test")

# Đường dẫn cho keypoint_extractor.py
FRAME_INPUT_DIR_KE = FRAME_OUTPUT_DIR_FE # Output của frame_extractor là input của keypoint_extractor
KEYPOINT_OUTPUT_DIR_KE = os.path.join(DATA_DIR, "keypoints_test")

# Đường dẫn cho data_processor.py
KEYPOINT_INPUT_DIR_DP = os.path.join(BASE_DIR, "keypoints") # Hoặc KEYPOINT_OUTPUT_DIR_KE nếu bạn muốn xử lý data test
PROCESSED_DATA_OUTPUT_DIR_DP = os.path.join(DATA_DIR, "processed_data")

# Đường dẫn cho train.py và predict.py
PROCESSED_DATA_INPUT_DIR_TRAIN_PRED = PROCESSED_DATA_OUTPUT_DIR_DP
MODEL_SAVE_PATH_TRAIN = os.path.join(BASE_DIR, "best_model.pth")

# Đường dẫn cho predict.py
VIDEO_PREDICT_INPUT_DIR_PRED = os.path.join(DATA_DIR, "filter_dataset")


# ----- Cấu hình trích xuất Keypoint -----
# Mapping từ index sang tên các bộ phận cơ thể liên quan đến cơ bắp
# Đây là các index gốc trong COCO (hoặc bộ tương tự mà RTMLib trả về)
MUSCLE_KEYPOINT_MAPPING_ORIGINAL = {
    5: "L_shoulder", 6: "R_shoulder",
    7: "L_elbow",    8: "R_elbow",
    9: "L_wrist",   10: "R_wrist",
    11: "L_hip",    12: "R_hip",
    13: "L_knee",   14: "R_knee",
    15: "L_ankle",  16: "R_ankle"
}
MUSCLE_INDICES_ORIGINAL = list(MUSCLE_KEYPOINT_MAPPING_ORIGINAL.keys()) # Index gốc (0-16 cho COCO)

# Tìm index của L hip và R hip trong danh sách MUSCLE_INDICES_ORIGINAL
# để sử dụng trong `anchor_indices_in_muscle_list`
# Ví dụ: nếu MUSCLE_INDICES_ORIGINAL là [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# thì index của 11 (L_hip) trong list này là 6, và của 12 (R_hip) là 7.
L_HIP_ORIGINAL_IDX = 11
R_HIP_ORIGINAL_IDX = 12

ANCHOR_INDICES_IN_MUSCLE_LIST = []
if L_HIP_ORIGINAL_IDX in MUSCLE_INDICES_ORIGINAL:
    ANCHOR_INDICES_IN_MUSCLE_LIST.append(MUSCLE_INDICES_ORIGINAL.index(L_HIP_ORIGINAL_IDX))
if R_HIP_ORIGINAL_IDX in MUSCLE_INDICES_ORIGINAL:
    ANCHOR_INDICES_IN_MUSCLE_LIST.append(MUSCLE_INDICES_ORIGINAL.index(R_HIP_ORIGINAL_IDX))

if not ANCHOR_INDICES_IN_MUSCLE_LIST:
    if MUSCLE_INDICES_ORIGINAL:
        ANCHOR_INDICES_IN_MUSCLE_LIST = [0] # Fallback nếu không có hông
        print("Warning: No hip keypoints found. Using first muscle keypoint as anchor for normalization.")
    else:
        print("Warning: MUSCLE_INDICES_ORIGINAL is empty. Normalization might not work as expected.")


# ----- Cấu hình Model -----
MAX_FRAMES_PER_VIDEO = 10 # Số frame tối đa lấy từ mỗi video

# Tham số mô hình BiLSTM + Attention (phải khớp với lúc train)
# num_features_per_frame sẽ được tính tự động trong data_processor
# num_classes sẽ được tính tự động trong data_processor
MODEL_HIDDEN_DIM = 256
MODEL_DENSE_DIM = 112
MODEL_DROPOUT_RATE = 0.4

# ----- Cấu hình Training -----
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# ----- Thiết bị -----
DEVICE = 'cuda' # hoặc 'cpu' nếu không có GPU
RTMLIB_BACKEND = 'onnxruntime' # 'opencv', 'openvino'