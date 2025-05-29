import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import json
from tqdm import tqdm # Use tqdm instead of tqdm.notebook for terminal
import matplotlib.pyplot as plt

from model import BiLSTMAttention
from config import (
    PROCESSED_DATA_INPUT_DIR_TRAIN_PRED, 
    MODEL_SAVE_PATH_TRAIN,
    MODEL_HIDDEN_DIM, 
    MODEL_DENSE_DIM, 
    MODEL_DROPOUT_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_training_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    with open(os.path.join(data_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(data_dir, "classes.json"), "r") as f:
        classes = json.load(f)
    with open(os.path.join(data_dir, "data_params.json"), "r") as f:
        data_params = json.load(f)
        
    return X_train, X_test, y_train, y_test, label_encoder, classes, data_params

def plot_history(history, save_path="training_plots.png"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training plots saved to {save_path}")
    # plt.show() # Comment out if running in a non-GUI environment

def plot_confusion_matrix_func(model_to_eval, data_loader, device_to_use, label_enc, class_names, save_path="confusion_matrix.png"):
    model_to_eval.eval()
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device_to_use)
            outputs = model_to_eval(inputs)
            predicted = torch.argmax(outputs, dim=1)
            true = torch.argmax(labels, dim=1) # Assuming labels are one-hot
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels.extend(true.cpu().numpy())

    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10)) # Adjusted for better label display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax, values_format='d')
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    # plt.show()

def train_model():
    X_train, X_test, y_train, y_test, label_encoder, classes, data_params = load_training_data(PROCESSED_DATA_INPUT_DIR_TRAIN_PRED)

    input_dim = data_params["num_features_per_frame"]
    num_classes = data_params["num_classes"]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
    print(f"Using device: {device}")

    model = BiLSTMAttention(
        input_dim=input_dim, 
        hidden_dim=MODEL_HIDDEN_DIM, 
        dense_dim=MODEL_DENSE_DIM, 
        num_classes=num_classes, 
        dropout_rate=MODEL_DROPOUT_RATE
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss() # Matches your notebook
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted_labels = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1) # Assuming one-hot
            total_train += labels.size(0)
            correct_train += (predicted_labels == true_labels).sum().item()
            train_pbar.set_postfix({'loss': loss.item(), 'acc': correct_train/total_train if total_train > 0 else 0})


        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                predicted_labels = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(labels, dim=1)
                total_val += labels.size(0)
                correct_val += (predicted_labels == true_labels).sum().item()
                val_pbar.set_postfix({'val_loss': loss.item(), 'val_acc': correct_val/total_val if total_val > 0 else 0})

        epoch_val_loss = running_val_loss / len(test_dataset)
        epoch_val_accuracy = correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH_TRAIN)
            print(f"Saved best model with Val Acc: {best_val_accuracy:.4f}")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    plot_history(history)
    
    # Plot confusion matrix for the test set using the best model
    print("Plotting confusion matrix for the test set...")
    best_model = BiLSTMAttention(
        input_dim=input_dim, 
        hidden_dim=MODEL_HIDDEN_DIM, 
        dense_dim=MODEL_DENSE_DIM, 
        num_classes=num_classes, 
        dropout_rate=MODEL_DROPOUT_RATE
    ).to(device)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH_TRAIN, map_location=device))
    plot_confusion_matrix_func(best_model, test_loader, device, label_encoder, classes)


if __name__ == "__main__":
    train_model()