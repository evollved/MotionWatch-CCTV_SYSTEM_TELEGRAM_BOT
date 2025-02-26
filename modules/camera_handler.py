import cv2
import threading
import time
import json
import os
import requests
from datetime import datetime
import numpy as np
import subprocess
import asyncio
from ultralytics import YOLO
import aiohttp
import collections
import aiofiles.os as aio_os
import torch
import psutil
import logging
from logging.handlers import RotatingFileHandler

def select_yolo_model():
    """
    Автоматически выбирает модель YOLO в зависимости от характеристик системы.
    Возвращает имя модели (например, "yolov8n.pt", "yolov8s.pt" и т.д.).
    """
    # Проверяем наличие GPU
    has_gpu = torch.cuda.is_available()

    # Получаем объем оперативной памяти в ГБ
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Выбираем модель в зависимости от характеристик системы
    if has_gpu:
        if ram_gb >= 16:
            return "yolov8m.pt"  # Для мощных систем с GPU
        elif ram_gb >= 8:
            return "yolov8s.pt"  # Для систем с GPU и 8 ГБ ОЗУ
        else:
            return "yolov8n.pt"  # Для слабых систем с GPU
    else:
        if ram_gb >= 16:
            return "yolov8s.pt"  # Для систем без GPU, но с 16 ГБ ОЗУ
        elif ram_gb >= 8:
            return "yolov8n.pt"  # Для систем без GPU и 8 ГБ ОЗУ
        else:
            return "yolov8n.pt"  # Для слабых систем без GPU

def setup_logging():
    """
    Настройка логирования с временными метками, записью в файл и ротацией по размеру.
    """
    # Создаем логгер
    logger = logging.getLogger("cctv_bot")
    logger.setLevel(logging.INFO)

    # Формат логов: [время] [уровень] сообщение
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Логирование в файл с ротацией по размеру (5 МБ)
    file_handler = RotatingFileHandler(
        "cctv_bot.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Логирование в консоль (опционально)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

class CameraHandler(threading.Thread):
    def __init__(self, camera_config, telegram_config):
        threading.Thread.__init__(self)
        self.camera_config = camera_config
        self.telegram_config = telegram_config

        # Логирование
        self.logger = setup_logging()

        # Автоматический выбор модели YOLO
        self.model_name = select_yolo_model()
        self.logger.info(f"Выбрана модель YOLO: {self.model_name}")

        # Инициализация модели YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name).to(self.device)
        self.logger.info(f"Модель YOLO загружена на устройство: {self.device}")

        # Остальные параметры камеры
        self.camera_id = camera_config['id']
        self.name = camera_config['name']
        self.rtsp_url = camera_config['rtsp_url']
        self.enabled = camera_config['enabled']
        self.detect_motion = camera_config['detect_motion']
        self.detect_objects = camera_config['detect_objects']
        self.object_types = camera_config.get('object_types', ["person", "cat", "dog"])
        self.object_confidence = camera_config.get('object_confidence', 0.5)
        self.telegram_chat_id = camera_config['telegram_chat_id']
        self.send_photo = camera_config['send_photo']
        self.send_video = camera_config['send_video']
        self.draw_boxes = camera_config['draw_boxes']
        self.test_frame_on_start = camera_config['test_frame_on_start']
        self.test_video_on_start = camera_config['test_video_on_start']
        self.motion_sensitivity = camera_config['motion_sensitivity']
        self.codec = camera_config.get('codec', 'h264').lower()
        self.record_audio = camera_config.get('record_audio', True)
        self.reconnect_attempts = camera_config.get('reconnect_attempts', 5)
        self.reconnect_delay = camera_config.get('reconnect_delay', 10)
        self.is_running = False
        self.motion_detected = False
        self.frame_count = 0
        self.motion_frame_count = 0
        self.last_frame = None
        self.last_motion_time = None
        self.motion_threshold = 1000
        self.min_motion_frames = 3
        self.loop = asyncio.new_event_loop()
        self.fps = 30
        self.buffer_size = int(30 * self.fps)
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)
        self.show_live_feed = camera_config.get("show_live_feed", False)

        # Инициализация ширины и высоты кадра
        self.camera_width = camera_config.get('width')
        self.camera_height = camera_config.get('height')
        self.frame_width = None  # Ширина кадра, полученная из потока
        self.frame_height = None  # Высота кадра, полученная из потока

    async def get_frame_size_ffmpeg(self):
        """Получает размеры кадра из RTSP-потока с помощью FFmpeg."""
        try:
            command = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', self.rtsp_url,
                '-vframes', '1',
                '-f', 'image2pipe',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-'
            ]
            process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.error(f"Ошибка FFmpeg при получении размеров кадра: {stderr.decode()}")
                return None, None

            # Получаем размеры кадра из вывода FFmpeg
            frame = np.frombuffer(stdout, dtype=np.uint8)
            if frame.size == 0:
                self.logger.error("FFmpeg вернул пустой кадр.")
                return None, None

            # Предполагаем, что кадр имеет стандартное соотношение сторон
            height, width = 480, 640  # Значения по умолчанию
            if len(frame) > 0:
                height, width, _ = frame.shape

            return width, height

        except Exception as e:
            self.logger.error(f"Ошибка при получении размеров кадра: {e}")
            return None, None

    async def capture_frame_async(self):
        """Асинхронный захват кадра с помощью FFmpeg."""
        try:
            command = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', self.rtsp_url,
                '-frames:v', '1',
                '-f', 'image2pipe',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-'
            ]
            process = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.error(f"Ошибка FFmpeg: {stderr.decode()}")
                return None

            frame = np.frombuffer(stdout, dtype=np.uint8)
            if frame.size == 0:
                self.logger.error("FFmpeg вернул пустой кадр.")
                return None

            # Предполагаем, что кадр имеет стандартное соотношение сторон
            if self.frame_width is None or self.frame_height is None:
                self.frame_width, self.frame_height = await self.get_frame_size_ffmpeg()
                if self.frame_width is None or self.frame_height is None:
                    self.logger.error("Не удалось получить размеры кадра.")
                    return None

            frame = frame.reshape((self.frame_height, self.frame_width, 3))
            return frame.copy()  # Возвращаем копию кадра

        except Exception as e:
            self.logger.error(f"Ошибка при захвате кадра FFmpeg: {e}")
            return None

    async def run_async(self):
        if not self.enabled:
            self.logger.info(f"Камера {self.name} отключена.")
            return

        self.logger.info(f"Запуск обработки камеры {self.name}...")
        self.is_running = True

        attempt = 0
        while self.is_running and attempt < self.reconnect_attempts:
            try:
                if self.camera_width and self.camera_height:
                    self.frame_width = self.camera_width
                    self.frame_height = self.camera_height
                    self.logger.info(f"Используем размеры кадра из конфигурации: width={self.frame_width}, height={self.frame_height}")
                else:
                    self.frame_width, self.frame_height = await self.get_frame_size_ffmpeg()
                    if self.frame_width is None or self.frame_height is None:
                        self.logger.error(f"Не удалось получить размеры кадра с камеры {self.name}")
                        self.is_running = False
                        return
                    self.logger.info(f"Размеры кадра получены из потока: width={self.frame_width}, height={self.frame_height}")

                if self.test_frame_on_start:
                    await self.capture_test_frame_async()
                if self.test_video_on_start:
                    await self.capture_test_video_async(5)

                bg_frame = None
                while self.is_running:
                    frame = await self.capture_frame_async()
                    if frame is None:
                        self.logger.error(f"Ошибка при захвате кадра с камеры {self.name} (FFmpeg)")
                        break

                    self.frame_buffer.append(frame)  # Добавляем кадр в буфер

                    self.frame_count += 1
                    if self.frame_count % 3 == 0:  # Обрабатываем каждый 3-й кадр
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (21, 21), 0)

                        if bg_frame is None:
                            bg_frame = gray
                            continue

                        frame_delta = cv2.absdiff(bg_frame, gray)
                        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh, None, iterations=2)
                        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        motion_detected = False
                        for contour in contours:
                            if cv2.contourArea(contour) < self.motion_sensitivity:
                                continue
                            motion_detected = True
                            break

                        if motion_detected:
                            self.logger.info(f"Движение обнаружено в кадре {self.frame_count}")
                            self.motion_frame_count += 1
                            if self.motion_frame_count >= self.min_motion_frames:
                                self.logger.info(f"Движение подтверждено, начало записи видео...")
                                self.motion_detected = True
                                self.last_motion_time = datetime.now()

                                if self.detect_objects:
                                    results = self.model(frame, conf=self.object_confidence, classes=[0, 15, 16])  # Только люди, коты и собаки
                                    detected_objects = results[0].boxes

                                    # Фильтруем объекты по типам
                                    filtered_objects = []
                                    object_counts = {}  # Словарь для подсчета количества объектов каждого типа
                                    for obj in detected_objects:
                                        class_name = self.model.names[int(obj.cls)]  # Получаем название класса
                                        if class_name in self.object_types:  # Сравниваем с object_types
                                            filtered_objects.append(obj)
                                            # Подсчитываем количество объектов каждого типа
                                            if class_name in object_counts:
                                                object_counts[class_name] += 1
                                            else:
                                                object_counts[class_name] = 1

                                    self.logger.info(f"Всего обнаружено объектов: {len(detected_objects)}, отфильтровано: {len(filtered_objects)}")

                                    if len(filtered_objects) > 0:  # Если объекты обнаружены
                                        self.logger.info(f"Обнаружены объекты: {filtered_objects}")
                                        if self.draw_boxes:
                                            frame = results[0].plot()  # Рисуем рамки на кадре

                                        # Формируем сообщение с информацией об объектах
                                        object_message = "Обнаружены объекты:\n"
                                        for obj_type, count in object_counts.items():
                                            object_message += f"- {obj_type}: {count}\n"

                                        if self.send_photo:
                                            self.logger.info("Попытка отправки фото в Telegram...")
                                            await self.send_telegram_photo_async(frame, message=object_message)  # Передаем сообщение об объектах
                                        if self.send_video:
                                            self.logger.info("Попытка отправки видео в Telegram...")
                                            await self.record_video_with_buffer(duration=15, message=object_message)  # Увеличено до 15 секунд
                                    else:
                                        self.logger.info("Нет объектов, соответствующих фильтру.")

                                else:
                                    # Если обнаружение объектов отключено, отправляем уведомления только о движении
                                    if self.send_photo:
                                        self.logger.info("Попытка отправки фото в Telegram...")
                                        await self.send_telegram_photo_async(frame)
                                    if self.send_video:
                                        self.logger.info("Попытка отправки видео в Telegram...")
                                        await self.record_video_with_buffer(duration=15)  # Увеличено до 15 секунд

                                self.motion_frame_count = 0

                        else:
                            self.motion_frame_count = 0
                            self.motion_detected = False

                        bg_frame = gray

            except Exception as e:
                self.logger.error(f"Ошибка при обработке камеры {self.name}: {e}")
                attempt += 1
                if attempt < self.reconnect_attempts:
                    self.logger.info(f"Попытка переподключения {attempt} из {self.reconnect_attempts} через {self.reconnect_delay} секунд...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    self.logger.error(f"Превышено количество попыток переподключения для камеры {self.name}.")
                    break

        self.logger.info(f"Завершение обработки камеры {self.name}")
        self.is_running = False

    def stop(self):
        """Останавливает поток."""
        self.is_running = False

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_async())