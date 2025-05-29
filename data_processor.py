import os
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # Hoặc sử dụng cách khác nếu không muốn dùng TF
from config import (
    KEYPOINT_INPUT_DIR_DP, 
    PROCESSED_DATA_OUTPUT_DIR_DP, 
    MUSCLE_INDICES_ORIGINAL, 
    ANCHOR_INDICES_IN_MUSCLE_LIST
)


def load_and_process_data(keypoints_data_dir, anchor_indices_in_muscle_list):
    if not os.path.isdir(keypoints_data_dir):
        print(f"Keypoints data directory not found: {keypoints_data_dir}")
        return None, None, None, None, None, None, 0, 0

    classes = sorted(os.listdir(keypoints_data_dir))
    num_classes = len(classes)
    print(f"Found classes: {classes}")

    sequences_processed = []
    labels_processed = []

    for class_folder in classes:
        class_path = os.path.join(keypoints_data_dir, class_folder)
        if os.path.isdir(class_path):
            print(f"Processing class for data loading: {class_folder}")
            for json_file in os.listdir(class_path):
                if json_file.endswith('_keypoints.json'):
                    json_path = os.path.join(class_path, json_file)
                    with open(json_path, 'r') as f:
                        frame_keypoints_list = json.load(f)

                    sequence_data_for_video = []
                    previous_flat_features = None

                    for frame_data in frame_keypoints_list:
                        muscle_keypoints_raw = frame_data.get("muscle_keypoints")
                        current_frame_features = None

                        if muscle_keypoints_raw is not None:
                            muscle_keypoints_np = np.array(muscle_keypoints_raw, dtype=np.float32)

                            if anchor_indices_in_muscle_list and muscle_keypoints_np.shape[0] > max(anchor_indices_in_muscle_list if anchor_indices_in_muscle_list else [-1]): # Đảm bảo index không vượt quá
                                anchor_points = muscle_keypoints_np[anchor_indices_in_muscle_list]
                                valid_anchor_points = anchor_points[~np.all(anchor_points == -1, axis=1)]

                                if valid_anchor_points.shape[0] > 0:
                                    anchor_mean = np.mean(valid_anchor_points, axis=0)
                                    muscle_keypoints_normalized = np.copy(muscle_keypoints_np)
                                    valid_keypoint_mask = ~np.all(muscle_keypoints_normalized == -1, axis=1)
                                    muscle_keypoints_normalized[valid_keypoint_mask] -= anchor_mean
                                else:
                                    muscle_keypoints_normalized = muscle_keypoints_np
                            else:
                                muscle_keypoints_normalized = muscle_keypoints_np

                            flat_features_shape = muscle_keypoints_normalized.flatten()
                            flat_features_movement = np.zeros_like(flat_features_shape)

                            if previous_flat_features is not None and flat_features_shape.shape == previous_flat_features.shape:
                                movement_mask = (flat_features_shape != -1) & (previous_flat_features != -1)
                                flat_features_movement[movement_mask] = flat_features_shape[movement_mask] - previous_flat_features[movement_mask]
                            
                            current_frame_features = np.concatenate([flat_features_shape, flat_features_movement])
                            previous_flat_features = flat_features_shape
                        
                        if current_frame_features is None:
                            num_muscle_points = len(MUSCLE_INDICES_ORIGINAL)
                            num_features_per_frame_expected = num_muscle_points * 2 * 2
                            current_frame_features = np.array([0.0] * num_features_per_frame_expected, dtype=np.float32)
                            previous_flat_features = None
                        
                        sequence_data_for_video.append(current_frame_features.tolist())

                    if sequence_data_for_video:
                        sequences_processed.append(sequence_data_for_video)
                        labels_processed.append(class_folder)
    
    if not sequences_processed:
        print("No sequences were processed. Check keypoint data.")
        return None, None, None, None, None, None, 0, 0

    max_seq_length = max(len(seq) for seq in sequences_processed)
    # num_features_per_frame nên được lấy từ frame đầu tiên của sequence đầu tiên,
    # giả sử tất cả các frame feature vectors có cùng kích thước.
    if sequences_processed and sequences_processed[0]:
         num_features_per_frame = len(sequences_processed[0][0])
    else: # Xử lý trường hợp không có dữ liệu
        num_features_per_frame = len(MUSCLE_INDICES_ORIGINAL) * 2 * 2 # Dự phòng
        print(f"Warning: Could not determine num_features_per_frame from data, defaulting to {num_features_per_frame}")


    sequences_padded = np.zeros((len(sequences_processed), max_seq_length, num_features_per_frame), dtype=np.float32)
    for i, sequence in enumerate(sequences_processed):
        seq_len = len(sequence)
        if seq_len > 0 and len(sequence[0]) == num_features_per_frame: # Đảm bảo số features khớp
             sequences_padded[i, :min(seq_len, max_seq_length), :] = np.array(sequence[:min(seq_len, max_seq_length)], dtype=np.float32)
        elif seq_len > 0:
            print(f"Warning: Feature length mismatch for sequence {i}. Expected {num_features_per_frame}, got {len(sequence[0])}. Padding with zeros.")


    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_processed)
    labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        sequences_padded, labels_categorical, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    
    print(f"Data processing complete.")
    print(f"  X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Number of features per frame: {num_features_per_frame}")

    return X_train, X_test, y_train, y_test, label_encoder, classes, max_seq_length, num_features_per_frame

def save_processed_data(output_dir, X_train, X_test, y_train, y_test, label_encoder, classes, max_seq_length, num_features_per_frame):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(classes, f)
        
    data_params = {
        "max_seq_length": max_seq_length,
        "num_features_per_frame": num_features_per_frame,
        "num_classes": len(classes)
    }
    with open(os.path.join(output_dir, "data_params.json"), "w") as f:
        json.dump(data_params, f)
    
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, classes_list, max_len, n_features = load_and_process_data(
        KEYPOINT_INPUT_DIR_DP, 
        ANCHOR_INDICES_IN_MUSCLE_LIST
    )
    if X_train is not None:
        save_processed_data(
            PROCESSED_DATA_OUTPUT_DIR_DP, 
            X_train, X_test, y_train, y_test, 
            le, classes_list, max_len, n_features
        )