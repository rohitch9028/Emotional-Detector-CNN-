"""
Emotion Detector using PyTorch + OpenCV
---------------------------------------
Compatible with Python 3.14

Setup:
1. Download FER-2013 dataset (fer2013.csv) from Kaggle:
   https://www.kaggle.com/datasets/msambare/fer2013
   Place it in the same folder.

2. Install dependencies (run these one by one):
   pip install torch torchvision torchaudio
   pip install opencv-python numpy pandas scikit-learn matplotlib

3. Train:
   python emotion_detector_pytorch.py --mode train --epochs 30

4. Run webcam demo:
   python emotion_detector_pytorch.py --mode infer --model_path best_model.pth
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 48
LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
N_CLASSES = len(LABELS)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==========================
# Dataset loader
# ==========================
class FER2013Dataset(Dataset):
    def __init__(self, csv_path, usage="Training"):
        df = pd.read_csv(csv_path)
        df = df[df['Usage'] == usage]
        self.emotions = df['emotion'].values
        self.pixels = df['pixels'].apply(lambda x: np.fromstring(x, dtype=np.uint8, sep=' ').reshape(48, 48))
        self.transform = lambda x: torch.tensor(x/255.0, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        img = self.transform(self.pixels.iloc[idx])
        label = torch.tensor(self.emotions[idx], dtype=torch.long)
        return img, label


# ==========================
# CNN Model
# ==========================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ==========================
# Training
# ==========================
def train_model(csv_path, epochs=30, batch_size=64, model_out='best_model.pth'):
    train_data = FER2013Dataset(csv_path, usage="Training")
    val_data = FER2013Dataset(csv_path, usage="PublicTest")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = EmotionCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_acc = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_out)
            print("✅ Saved best model")

    # Plot loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Training History")
    plt.savefig("training_history.png")
    plt.close()

    print("Training Complete. Best accuracy:", best_acc, "%")


# ==========================
# Webcam inference
# ==========================
def infer_webcam(model_path='best_model.pth'):
    model = EmotionCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = torch.tensor(roi / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                preds = model(roi)
                label = LABELS[int(torch.argmax(preds))]
                score = float(F.softmax(preds, dim=1).max())
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================
# CLI
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','infer'], default='train')
    parser.add_argument('--csv', default='fer2013.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_path', default='best_model.pth')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, model_out=args.model_path)
    elif args.mode == 'infer':
        infer_webcam(model_path=args.model_path)


if __name__ == '__main__':
    main()
