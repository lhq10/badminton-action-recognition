import os
import json
import cv2
import numpy as np
from rtmlib_utils import wholebody_model, MUSCLE_INDICES_ORIGINAL # MUSCLE_KEYPOINT_MAPPING_ORIGINAL
from config import FRAME_INPUT_DIR_KE, KEYPOINT_OUTPUT_DIR_KE

def extract_and_save_keypoints(frame_dir_root, output_base_dir):
    if wholebody_model is None:
        print("RTMLib WholeBody model is not initialized. Cannot extract keypoints.")
        return
    if not os.path.isdir(frame_dir_root):
        print(f"Frame input directory not found: {frame_dir_root}")
        return

    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Starting keypoint extraction from: {frame_dir_root}")
    print(f"Keypoints will be saved to: {output_base_dir}")

    r_ankle_index_original = 16 # Index của R ankle trong bộ keypoint gốc (COCO)

    for class_folder in os.listdir(frame_dir_root):
        class_path = os.path.join(frame_dir_root, class_folder)
        if os.path.isdir(class_path):
            print(f"Processing class: {class_folder}")
            class_output_dir = os.path.join(output_base_dir, class_folder)
            os.makedirs(class_output_dir, exist_ok=True)

            for action_folder in os.listdir(class_path): # Đây là thư mục chứa frames của 1 video
                action_frames_path = os.path.join(class_path, action_folder)
                if os.path.isdir(action_frames_path):
                    # Tên file JSON sẽ là tên thư mục frame + "_keypoints.json"
                    output_json_path = os.path.join(class_output_dir, f"{action_folder}_keypoints.json")

                    if os.path.exists(output_json_path):
                        print(f"  Skipping already processed action (frames folder): {action_folder}")
                        continue
                    print(f"  Processing action (frames folder): {action_folder}")

                    frame_keypoints_list = []
                    frame_files = sorted([f for f in os.listdir(action_frames_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

                    for frame_file in frame_files:
                        frame_path = os.path.join(action_frames_path, frame_file)
                        img = cv2.imread(frame_path)
                        if img is None:
                            print(f"    Could not read frame: {frame_file}")
                            continue

                        keypoints_all_people, scores_all_people = wholebody_model(img)

                        person_with_highest_r_ankle_idx = None
                        highest_r_ankle_y = -1

                        if keypoints_all_people is not None and len(keypoints_all_people) > 0:
                            for person_idx, person_kps in enumerate(keypoints_all_people):
                                if len(person_kps) > r_ankle_index_original:
                                    r_ankle_kpt = person_kps[r_ankle_index_original]
                                    if r_ankle_kpt[0] != -1 and r_ankle_kpt[1] != -1:
                                        if r_ankle_kpt[1] > highest_r_ankle_y:
                                            highest_r_ankle_y = r_ankle_kpt[1]
                                            person_with_highest_r_ankle_idx = person_idx
                        
                        frame_data = {
                            "frame_file": frame_file,
                            "muscle_keypoints": None,
                            "person_with_highest_r_ankle_index": person_with_highest_r_ankle_idx
                        }

                        if person_with_highest_r_ankle_idx is not None:
                            # Lấy keypoints của người được chọn
                            selected_person_keypoints = keypoints_all_people[person_with_highest_r_ankle_idx]
                            # Lọc ra chỉ các keypoint cơ bắp dựa trên MUSCLE_INDICES_ORIGINAL
                            muscle_keypoints_for_frame = selected_person_keypoints[MUSCLE_INDICES_ORIGINAL]
                            frame_data["muscle_keypoints"] = muscle_keypoints_for_frame.tolist()
                        
                        frame_keypoints_list.append(frame_data)

                    with open(output_json_path, 'w') as f:
                        json.dump(frame_keypoints_list, f, indent=2)
                    # print(f"    Saved keypoints to {output_json_path}")
    print("Keypoint extraction complete.")

if __name__ == "__main__":
    # Sử dụng đường dẫn từ config.py
    extract_and_save_keypoints(FRAME_INPUT_DIR_KE, KEYPOINT_OUTPUT_DIR_KE)