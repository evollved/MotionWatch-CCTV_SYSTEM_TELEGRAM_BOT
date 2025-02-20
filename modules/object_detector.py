import cv2
import numpy as np
import os
import requests

class ObjectDetector:
    def __init__(self, config_path, weights_path, classes_path):
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes_path = classes_path

        # Загружаем классы объектов
        with open(self.classes_path, 'r') as f:
            self.classes = f.read().strip().split('\n')

        # Загружаем модель YOLO
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Получаем имена слоев вывода
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        """Обнаружение объектов на кадре с помощью YOLO."""
        height, width, channels = frame.shape

        # Подготавливаем кадр для YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Параметры для обнаружения объектов
        class_ids = []
        confidences = []
        boxes = []

        # Обрабатываем выходные данные YOLO
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Порог уверенности
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Координаты прямоугольника
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Применяем Non-Maximum Suppression для устранения дублирующих прямоугольников
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append((label, confidence, (x, y, w, h)))

        return detected_objects, frame