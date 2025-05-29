import os
import cv2
from config import VIDEO_INPUT_DIR_FE, FRAME_OUTPUT_DIR_FE, MAX_FRAMES_PER_VIDEO

def extract_frames_from_video(video_path, output_folder, max_frames=10):
    if not os.path.exists(video_path):
        print(f"Video path not found: {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    frames_extracted = []
    frame_count = 0
    while frame_count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360)) # Resize frame
        frames_extracted.append(frame)
        frame_count += 1
    cap.release()

    os.makedirs(output_folder, exist_ok=True)
    for i, frame in enumerate(frames_extracted):
        frame_path = os.path.join(output_folder, f"{i:03d}.jpg")
        cv2.imwrite(frame_path, frame)
    # print(f"Extracted {len(frames_extracted)} frames from {video_path} to {output_folder}")


def process_all_videos(input_root, output_root, max_frames=10):
    if not os.path.isdir(input_root):
        print(f"Input root directory not found: {input_root}")
        return

    os.makedirs(output_root, exist_ok=True)
    print(f"Starting frame extraction from: {input_root}")
    print(f"Frames will be saved to: {output_root}")

    for class_name in os.listdir(input_root):
        class_input_path = os.path.join(input_root, class_name)
        class_output_path = os.path.join(output_root, class_name)

        if os.path.isdir(class_input_path):
            os.makedirs(class_output_path, exist_ok=True)
            print(f"Processing class: {class_name}")

            for video_file in os.listdir(class_input_path):
                if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')): # Xử lý các định dạng video phổ biến
                    continue
                
                video_name_no_ext = os.path.splitext(video_file)[0]
                video_full_path = os.path.join(class_input_path, video_file)
                
                # Tạo thư mục output dựa trên tên video (không bao gồm phần mở rộng)
                # và có thể thêm một hậu tố để phân biệt thư mục frame với video gốc
                # Ví dụ: video1.mp4 -> thư mục output là video1
                output_folder_for_video = os.path.join(class_output_path, video_name_no_ext)
                
                if os.path.exists(output_folder_for_video) and os.listdir(output_folder_for_video):
                    print(f"  Skipping already extracted video: {video_file}")
                    continue
                
                print(f"  Extracting frames from: {video_file}")
                extract_frames_from_video(video_full_path, output_folder_for_video, max_frames)
    print("Frame extraction complete.")

if __name__ == "__main__":
    # Sử dụng đường dẫn từ config.py
    process_all_videos(VIDEO_INPUT_DIR_FE, FRAME_OUTPUT_DIR_FE, max_frames=MAX_FRAMES_PER_VIDEO)