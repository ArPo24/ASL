import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import torch
import numpy as np
import time

from model import ASLModel



id_to_label = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
    25: "Z", 26: "del", 27: "nothing", 28: "space" 
}


# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Загрузка обученной модели
model = torch.load('asl_model.pth', weights_only=False, map_location='cpu')
model.eval()

# Трансформы для изображения
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def process_frame(image):
    # Конвертация и предсказание
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return id_to_label[predicted.item()]

def main():
    # Иницилизация камеры
    cap = cv2.VideoCapture(0)
    recognized_text = ""
    last_gesture = None
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Обнаружение руки
        frame.flags.writeable = False
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True


        if results.multi_hand_landmarks:
            # Получение bounding box руки
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            # Обрезка области с рукой + padding
            margin = 50
            x_min, x_max = int(min(x_coords)) - margin, int(max(x_coords)) + margin
            y_min, y_max = int(min(y_coords)) - margin, int(max(y_coords)) + margin
            
            # Проверка границ
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            hand_roi = frame[y_min:y_max, x_min:x_max]

            
            if hand_roi.size > 0:
                # Классификация жеста
                current_gesture = process_frame(hand_roi)
                # Логика обновления текста
                if current_gesture != last_gesture:
                    if current_gesture == "del":
                        recognized_text = recognized_text[:-1]
                    elif current_gesture == "space":
                        recognized_text += " "
                    else:
                        recognized_text += current_gesture
                    last_gesture = current_gesture

                # Отрисовка
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # Отображение результатов
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, recognized_text, (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('ASL Translator', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()