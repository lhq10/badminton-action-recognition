from rtmlib import Wholebody
from config import RTMLIB_BACKEND, DEVICE, MUSCLE_KEYPOINT_MAPPING_ORIGINAL, MUSCLE_INDICES_ORIGINAL

# Khởi tạo mô hình wholebody toàn cục để có thể tái sử dụng
# Thiết lập style skeleton
OPENPOSE_SKELETON = False  # True nếu muốn dùng kiểu openpose

try:
    wholebody_model = Wholebody(
        to_openpose=OPENPOSE_SKELETON,
        mode='balanced', # 'performance', 'lightweight', 'balanced'
        backend=RTMLIB_BACKEND,
        device=DEVICE
    )
    print(f"RTMLib WholeBody model initialized successfully on {DEVICE} with {RTMLIB_BACKEND} backend.")
except Exception as e:
    print(f"Error initializing RTMLib WholeBody model: {e}")
    print("Please ensure ONNXRuntime (and CUDA if using GPU) is installed correctly.")
    wholebody_model = None

# Danh sách index của keypoint cơ (từ config)
# MUSCLE_INDICES_ORIGINAL là danh sách các index gốc (ví dụ: COCO) mà chúng ta quan tâm.
# Ví dụ: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

if __name__ == '__main__':
    if wholebody_model:
        print("wholebody_model is available.")
    else:
        print("wholebody_model is NOT available. Check initialization errors.")
    print(f"Muscle keypoint original indices: {MUSCLE_INDICES_ORIGINAL}")
    print(f"Muscle keypoint mapping: {MUSCLE_KEYPOINT_MAPPING_ORIGINAL}")