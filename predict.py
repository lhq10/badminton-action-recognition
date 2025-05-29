import os
import cv2
import numpy as np
import json
import torch
import random
import pickle

from rtmlib_utils import wholebody_model, MUSCLE_INDICES_ORIGINAL, ANCHOR_INDICES_IN_MUSCLE_LIST
from model import BiLSTMAttention
from config import (
    MODEL_SAVE_PATH_TRAIN, 
    VIDEO_PREDICT_INPUT_DIR_PRED, 
    MAX_FRAMES_PER_VIDEO,
    MODEL_HIDDEN_DIM, 
    MODEL_DENSE_DIM, 
    MODEL_DROPOUT_RATE,
    DEVICE,
    PROCESSED_DATA_INPUT_DIR_TRAIN_PRED # Để lấy data_params và label_encoder
)

def process_single_video_for_prediction(video_path, wholebody, muscle_indices_orig, 
                                        anchor_indices_in_muscle, max_vid_frames, 
                                        target_seq_length, target_num_features):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while frame_count < max_vid_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        frames.append(frame)
        frame_count +=1
    cap.release()

    if not frames:
        print(f"Could not extract frames from {video_path}")
        return None

    sequence_data_processed = []
    previous_flat_features = None
    r_ankle_index_original = 16

    for frame_img in frames:
        keypoints_all_people, _ = wholebody(frame_img)
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
        
        current_frame_features = None
        if person_with_highest_r_ankle_idx is not None:
            selected_person_keypoints = keypoints_all_people[person_with_highest_r_ankle_idx]
            muscle_keypoints_np = selected_person_keypoints[muscle_indices_orig]
            
            muscle_keypoints_normalized = muscle_keypoints_np
            if anchor_indices_in_muscle and muscle_keypoints_np.shape[0] > max(anchor_indices_in_muscle if anchor_indices_in_muscle else [-1]):
                anchor_points = muscle_keypoints_np[anchor_indices_in_muscle]
                valid_anchor_points = anchor_points[~np.all(anchor_points == -1, axis=1)]
                if valid_anchor_points.shape[0] > 0:
                    anchor_mean = np.mean(valid_anchor_points, axis=0)
                    muscle_keypoints_normalized = np.copy(muscle_keypoints_np)
                    valid_keypoint_mask = ~np.all(muscle_keypoints_normalized == -1, axis=1)
                    muscle_keypoints_normalized[valid_keypoint_mask] -= anchor_mean
            
            flat_features_shape = muscle_keypoints_normalized.flatten()
            flat_features_movement = np.zeros_like(flat_features_shape)
            if previous_flat_features is not None and flat_features_shape.shape == previous_flat_features.shape:
                movement_mask = (flat_features_shape != -1) & (previous_flat_features != -1)
                flat_features_movement[movement_mask] = flat_features_shape[movement_mask] - previous_flat_features[movement_mask]
            
            current_frame_features = np.concatenate([flat_features_shape, flat_features_movement])
            previous_flat_features = flat_features_shape
        
        if current_frame_features is None:
            num_features_per_frame_expected = len(muscle_indices_orig) * 2 * 2
            current_frame_features = np.array([0.0] * num_features_per_frame_expected, dtype=np.float32)
            previous_flat_features = None
            
        sequence_data_processed.append(current_frame_features.tolist())

    if not sequence_data_processed:
        return None

    sequence_np = np.array(sequence_data_processed, dtype=np.float32)
    current_seq_length = sequence_np.shape[0]
    
    # Đảm bảo rằng sequence_np có đúng số features trước khi padding/cắt
    if current_seq_length > 0 and sequence_np.shape[1] != target_num_features:
        print(f"Warning: Feature mismatch for video {video_path}. Expected {target_num_features}, got {sequence_np.shape[1]}. This might lead to errors.")
        # Có thể cần xử lý lỗi này tốt hơn, ví dụ: padding features nếu thiếu, hoặc bỏ qua video
        # Tạm thời, nếu sai số features, trả về None
        return None


    padded_sequence_np = np.zeros((target_seq_length, target_num_features), dtype=np.float32)
    if current_seq_length > target_seq_length:
        padded_sequence_np = sequence_np[:target_seq_length, :]
    elif current_seq_length > 0: # Chỉ copy nếu có frame
        padded_sequence_np[:current_seq_length, :] = sequence_np
    
    return np.expand_dims(padded_sequence_np, axis=0)


def predict_random_video():
    if wholebody_model is None:
        print("RTMLib WholeBody model is not initialized. Cannot predict.")
        return

    if not os.path.isdir(VIDEO_PREDICT_INPUT_DIR_PRED):
        print(f"Video input directory not found: {VIDEO_PREDICT_INPUT_DIR_PRED}")
        return
        
    # Load data parameters and label encoder
    try:
        with open(os.path.join(PROCESSED_DATA_INPUT_DIR_TRAIN_PRED, "data_params.json"), "r") as f:
            data_params = json.load(f)
        max_seq_length = data_params["max_seq_length"]
        num_features_per_frame = data_params["num_features_per_frame"]
        num_classes = data_params["num_classes"]

        with open(os.path.join(PROCESSED_DATA_INPUT_DIR_TRAIN_PRED, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(PROCESSED_DATA_INPUT_DIR_TRAIN_PRED, "classes.json"), "r") as f:
            classes_list = json.load(f) # Đây là danh sách tên lớp, dùng để map index sang tên

    except FileNotFoundError:
        print("Error: Processed data files (data_params.json, label_encoder.pkl, classes.json) not found.")
        print(f"Please run data_processor.py first and ensure files are in {PROCESSED_DATA_INPUT_DIR_TRAIN_PRED}")
        return
    
    class_folders = [d for d in os.listdir(VIDEO_PREDICT_INPUT_DIR_PRED) if os.path.isdir(os.path.join(VIDEO_PREDICT_INPUT_DIR_PRED, d))]
    if not class_folders:
        print(f"No class folders found in {VIDEO_PREDICT_INPUT_DIR_PRED}")
        return

    random_class = random.choice(class_folders)
    random_class_path = os.path.join(VIDEO_PREDICT_INPUT_DIR_PRED, random_class)
    video_files = [f for f in os.listdir(random_class_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"No video files found in {random_class_path}")
        return

    random_video_name = random.choice(video_files)
    random_video_path = os.path.join(random_class_path, random_video_name)
    true_label = random_class

    print(f"Selected video: {random_video_path}")
    print(f"True label: {true_label}")

    device_to_use = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
    
    inference_model = BiLSTMAttention(
        input_dim=num_features_per_frame, 
        hidden_dim=MODEL_HIDDEN_DIM, 
        dense_dim=MODEL_DENSE_DIM, 
        num_classes=num_classes, 
        dropout_rate=MODEL_DROPOUT_RATE
    ).to(device_to_use)
    
    try:
        inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH_TRAIN, map_location=device_to_use))
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {MODEL_SAVE_PATH_TRAIN}. Please train the model first.")
        return
        
    inference_model.eval()

    processed_sequence = process_single_video_for_prediction(
        random_video_path,
        wholebody_model,
        MUSCLE_INDICES_ORIGINAL,
        ANCHOR_INDICES_IN_MUSCLE_LIST,
        MAX_FRAMES_PER_VIDEO,
        max_seq_length,
        num_features_per_frame
    )

    if processed_sequence is not None:
        sequence_tensor = torch.tensor(processed_sequence, dtype=torch.float32).to(device_to_use)
        with torch.no_grad():
            output = inference_model(sequence_tensor)
            _, predicted_index_tensor = torch.max(output, 1)
            predicted_label_index = predicted_index_tensor.item()
        
        predicted_label = classes_list[predicted_label_index] # Sử dụng classes_list đã load

        print(f"\nPredicted label: {predicted_label}")
        print(f"True label:      {true_label}")
        if predicted_label == true_label:
            print("Prediction is CORRECT!")
        else:
            print("Prediction is INCORRECT!")
    else:
        print("Could not process the video for prediction.")

if __name__ == "__main__":
    predict_random_video()