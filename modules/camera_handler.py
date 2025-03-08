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
import tempfile
import logging

class CameraHandler(threading.Thread):
    def __init__(self, camera_config, telegram_config, yolo_model, device="cpu"):
        threading.Thread.__init__(self)
        self.camera_config = camera_config
        self.telegram_config = telegram_config
        self.device = device
        self.camera_id = camera_config['id']
        self.name = camera_config['name']
        self.rtsp_url = camera_config['rtsp_url']
        self.enabled = camera_config['enabled']
        self.camera_width = camera_config.get('width')
        self.camera_height = camera_config.get('height')
        self.detect_motion = camera_config['detect_motion']
        self.detect_objects = camera_config['detect_objects']
        self.object_types = camera_config.get("object_types", ["person", "cat", "dog"])
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
        self.model = yolo_model
        self.fps = 15  # Уменьшенный FPS
        self.buffer_size = int(10 * self.fps)  # Буфер на 10 секунд
        self.buffer_dir = tempfile.mkdtemp()  # Временная папка для буфера
        self.frame_buffer = []  # Список для хранения путей к файлам
        self.show_live_feed = camera_config.get("show_live_feed", False)
        self.back_sub = cv2.createBackgroundSubtractorMOG2()

    async def capture_frame_async(self):
        """Асинхронный захват кадра с помощью FFmpeg."""
        try:
            command = [
                'ffmpeg',
                '-threads', '8',  # Использование всех ядер
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
                logging.error(f"Ошибка FFmpeg: {stderr.decode()}")
                return None

            frame = np.frombuffer(stdout, dtype=np.uint8)
            if frame.size == 0:
                logging.warning("FFmpeg вернул пустой кадр.")
                return None

            frame = frame.reshape((self.camera_height, self.camera_width, 3))
            return frame.copy()

        except Exception as e:
            logging.error(f"Ошибка при захвате кадра FFmpeg: {e}")
            return None

    def get_motion_zone(self, frame, motion_mask):
        """
        Определяет зону, где обнаружено движение.
        :param frame: Входной кадр.
        :param motion_mask: Маска движения.
        :return: Строка с описанием зоны (например, "слева вверху").
        """
        h, w = frame.shape[:2]
        motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not motion_contours:
            return "движение не обнаружено"

        # Находим центр движения
        motion_center = None
        for contour in motion_contours:
            (x, y, w_contour, h_contour) = cv2.boundingRect(contour)
            if motion_center is None:
                motion_center = (x + w_contour // 2, y + h_contour // 2)
            else:
                motion_center = ((motion_center[0] + x + w_contour // 2) // 2, (motion_center[1] + y + h_contour // 2) // 2)

        if motion_center is None:
            return "движение не обнаружено"

        # Определяем зону
        frame_center_x = w // 2
        frame_center_y = h // 2

        if motion_center[0] < frame_center_x and motion_center[1] < frame_center_y:
            return "слева вверху"
        elif motion_center[0] < frame_center_x and motion_center[1] >= frame_center_y:
            return "слева внизу"
        elif motion_center[0] >= frame_center_x and motion_center[1] < frame_center_y:
            return "справа вверху"
        else:
            return "справа внизу"

    def draw_motion_zone(self, frame, motion_mask):
        """
        Рисует прямоугольник вокруг области движения.
        :param frame: Входной кадр.
        :param motion_mask: Маска движения.
        :return: Кадр с выделенной областью движения.
        """
        motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in motion_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Зелёный прямоугольник
        return frame

    async def run_async(self):
        if not self.enabled:
            logging.info(f"Камера {self.name} отключена.")
            return

        logging.info(f"Запуск обработки камеры {self.name}...")
        self.is_running = True

        attempt = 0
        while self.is_running and attempt < self.reconnect_attempts:
            try:
                if self.camera_width is None or self.camera_height is None:
                    self.camera_width, self.camera_height = self.get_frame_size_ffmpeg()
                    if self.camera_width is None or self.camera_height is None:
                        logging.error(f"Не удалось определить разрешение кадра для камеры {self.name}")
                        self.is_running = False
                        return
                    logging.info(f"Автоматически определено разрешение: width={self.camera_width}, height={self.camera_height}")

                if self.test_frame_on_start:
                    await self.capture_test_frame_async()
                if self.test_video_on_start:
                    await self.capture_test_video_async(5)

                bg_frame = None
                while self.is_running:
                    frame = await self.capture_frame_async()
                    if frame is None:
                        logging.error(f"Ошибка при захвате кадра с камеры {self.name}")
                        break

                    # Сохранение кадра на диск
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    filename = os.path.join(self.buffer_dir, f"frame_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    self.frame_buffer.append(filename)

                    # Ограничение размера буфера
                    if len(self.frame_buffer) > self.buffer_size:
                        old_frame = self.frame_buffer.pop(0)
                        if os.path.exists(old_frame):
                            os.remove(old_frame)

                    self.frame_count += 1

                    if self.frame_count % 5 == 0:  # Обрабатываем каждый 5-й кадр
                        resized_frame = cv2.resize(frame, (640, 360))  # Масштабирование для детекции
                        fg_mask = self.back_sub.apply(resized_frame)
                        motion_detected = np.sum(fg_mask) > self.motion_sensitivity

                        if motion_detected:
                            motion_zone = self.get_motion_zone(resized_frame, fg_mask)
                            logging.info(f"Камера {self.name}: Движение обнаружено в зоне {motion_zone}")
                            self.motion_frame_count += 1
                            if self.motion_frame_count >= self.min_motion_frames:
                                logging.info(f"Камера {self.name}: Движение подтверждено, начало записи видео...")
                                self.motion_detected = True
                                self.last_motion_time = datetime.now()

                                if self.detect_objects:
                                    results = self.model(resized_frame, conf=self.object_confidence, classes=[0, 15, 16])
                                    detected_objects = results[0].boxes

                                    filtered_objects = []
                                    object_counts = {}
                                    for obj in detected_objects:
                                        class_name = self.model.names[int(obj.cls)]
                                        if class_name in self.object_types:
                                            filtered_objects.append(obj)
                                            if class_name in object_counts:
                                                object_counts[class_name] += 1
                                            else:
                                                object_counts[class_name] = 1

                                    logging.info(f"Камера {self.name}: Всего обнаружено объектов: {len(detected_objects)}, отфильтровано: {len(filtered_objects)}")

                                    if len(filtered_objects) > 0:
                                        for obj in filtered_objects:
                                            # Получаем координаты объекта
                                            x1, y1, x2, y2 = obj.xyxy[0].tolist()
                                            confidence = obj.conf.item()

                                            # Определяем расположение объекта
                                            frame_center_x = resized_frame.shape[1] / 2
                                            frame_center_y = resized_frame.shape[0] / 2

                                            if x1 < frame_center_x and y1 < frame_center_y:
                                                location = "слева вверху"
                                            elif x1 < frame_center_x and y1 >= frame_center_y:
                                                location = "слева внизу"
                                            elif x1 >= frame_center_x and y1 < frame_center_y:
                                                location = "справа вверху"
                                            else:
                                                location = "справа внизу"

                                            logging.info(f"Камера {self.name}: Обнаружен объект '{class_name}' с точностью {confidence:.2f}, расположение: {location}")

                                        if self.draw_boxes:
                                            frame = results[0].plot()

                                        object_message = "Обнаружены объекты:\n"
                                        for obj_type, count in object_counts.items():
                                            object_message += f"- {obj_type}: {count}\n"

                                        if self.send_photo:
                                            logging.info(f"Камера {self.name}: Попытка отправки фото в Telegram...")
                                            await self.send_telegram_photo_async(frame, message=object_message)
                                        if self.send_video:
                                            logging.info(f"Камера {self.name}: Попытка отправки видео в Telegram...")
                                            await self.record_video_with_buffer(duration=15, message=object_message)
                                    else:
                                        logging.info(f"Камера {self.name}: Нет объектов, соответствующих фильтру.")

                                else:
                                    # Если детекция объектов отключена, просто логируем зону движения
                                    if self.send_photo:
                                        frame_with_motion = self.draw_motion_zone(frame, fg_mask)
                                        logging.info(f"Камера {self.name}: Попытка отправки фото в Telegram...")
                                        await self.send_telegram_photo_async(frame_with_motion, message=f"Движение обнаружено в зоне {motion_zone}")
                                    if self.send_video:
                                        logging.info(f"Камера {self.name}: Попытка отправки видео в Telegram...")
                                        await self.record_video_with_buffer(duration=15, message=f"Движение обнаружено в зоне {motion_zone}")

                                self.motion_frame_count = 0

                        else:
                            self.motion_frame_count = 0
                            self.motion_detected = False

            except Exception as e:
                logging.error(f"Камера {self.name}: Ошибка при обработке камеры: {e}")
                attempt += 1
                if attempt < self.reconnect_attempts:
                    delay = min(self.reconnect_delay * (2 ** (attempt - 1)), 300)
                    logging.info(f"Камера {self.name}: Попытка переподключения {attempt} из {self.reconnect_attempts} через {delay} секунд...")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"Камера {self.name}: Превышено количество попыток переподключения.")
                    break

        logging.info(f"Камера {self.name}: Завершение обработки.")
        self.is_running = False

    async def capture_test_frame_async(self):
        """Асинхронный захват и отправка тестового кадра."""
        frame = await self.capture_frame_async()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"test_frame_{self.name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            await self.send_telegram_photo_async(frame, message=f"Тестовый кадр с камеры {self.name}")
            await aio_os.remove(filename)
        else:
            logging.error(f"Не удалось захватить тестовый кадр с камеры {self.name}")

    async def capture_test_video_async(self, duration=5):
        """Асинхронный захват и отправка тестового видео."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"test_video_{self.name}_{timestamp}.mp4"
        temp_filename = f"temp_test_video_{self.name}_{timestamp}.mp4"

        try:
            command_record = [
                'ffmpeg',
                '-threads', '8',
                '-rtsp_transport', 'tcp',
                '-i', self.rtsp_url,
                '-t', str(duration),
                '-c:v', 'copy',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
            ]
            if self.record_audio:
                command_record.extend(['-c:a', 'aac'])
            else:
                command_record.append('-an')
            command_record.append(temp_filename)
            process_record = await asyncio.create_subprocess_exec(*command_record, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await process_record.communicate()

            if process_record.returncode != 0:
                logging.error(f"Ошибка FFmpeg при записи тестового видео.")
                return

            if self.codec != 'h264':
                command_convert = [
                    'ffmpeg',
                    '-threads', '8',
                    '-i', temp_filename,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',
                    '-movflags', 'faststart',
                    filename
                ]
                process_convert = await asyncio.create_subprocess_exec(*command_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                await process_convert.communicate()

                if process_convert.returncode != 0:
                    logging.error(f"Ошибка FFmpeg при перекодировании тестового видео.")
                    await aio_os.remove(temp_filename)
                    return
            else:
                await aio_os.rename(temp_filename, filename)

            await self.send_telegram_video_async(filename, message=f"Тестовое видео с камеры {self.name}")

        except Exception as e:
            logging.error(f"Ошибка при записи тестового видео: {e}")
        finally:
            if await aio_os.path.exists(temp_filename):
                await aio_os.remove(temp_filename)
            if await aio_os.path.exists(filename):
                await aio_os.remove(filename)

    async def record_video_with_buffer(self, duration=15, message="Движение обнаружено!"):
        """Записывает видео с буфером и продолжает запись после."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"motion_video_{self.name}_{timestamp}.mp4"
        temp_filename = f"temp_video_{self.name}_{timestamp}.mp4"

        try:
            buffer_frames = list(self.frame_buffer)
            self.frame_buffer.clear()

            command_record = [
                'ffmpeg',
                '-threads', '8',
                '-rtsp_transport', 'tcp',
                '-i', self.rtsp_url,
                '-t', str(duration),
                '-c:v', 'copy',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
            ]
            if self.record_audio:
                command_record.extend(['-c:a', 'aac'])
            else:
                command_record.append('-an')
            command_record.append(temp_filename)
            process_record = await asyncio.create_subprocess_exec(*command_record, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await process_record.communicate()

            if process_record.returncode != 0:
                logging.error(f"Ошибка FFmpeg при записи видео.")
                return

            if self.codec != 'h264':
                command_convert = [
                    'ffmpeg',
                    '-threads', '8',
                    '-i', temp_filename,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',
                    '-movflags', 'faststart',
                    filename
                ]
                process_convert = await asyncio.create_subprocess_exec(*command_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                await process_convert.communicate()

                if process_convert.returncode != 0:
                    logging.error(f"Ошибка FFmpeg при перекодировании видео.")
                    await aio_os.remove(temp_filename)
                    return
            else:
                await aio_os.rename(temp_filename, filename)

            await self.send_telegram_video_async(filename, message=message)

        except Exception as e:
            logging.error(f"Ошибка при записи/перекодировании видео: {e}")
        finally:
            if await aio_os.path.exists(temp_filename):
                await aio_os.remove(temp_filename)
            if await aio_os.path.exists(filename):
                await aio_os.remove(filename)

    async def send_telegram_photo_async(self, frame, message="Движение обнаружено!"):
        """Асинхронная отправка фото в Telegram."""
        try:
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_chat_id

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"motion_photo_{self.name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            async with aiohttp.ClientSession() as session:
                with open(filename, 'rb') as photo:
                    data = aiohttp.FormData()
                    data.add_field('chat_id', chat_id)
                    data.add_field('caption', message)
                    data.add_field('photo', photo, filename=filename)
                    async with session.post(url, data=data) as response:
                        if response.status != 200:
                            logging.error(f"Ошибка при отправке фото в Telegram: {response.status} - {await response.text()}")
                        else:
                            logging.info("Фото успешно отправлено в Telegram.")

            await aio_os.remove(filename)
        except Exception as e:
            logging.error(f"Ошибка при отправке фото в Telegram: {e}")

    async def send_telegram_video_async(self, video_path, message="Движение обнаружено!"):
        """Асинхронная отправка видео в Telegram."""
        try:
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_chat_id

            url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
            async with aiohttp.ClientSession() as session:
                with open(video_path, 'rb') as video:
                    data = aiohttp.FormData()
                    data.add_field('chat_id', chat_id)
                    data.add_field('caption', message)
                    data.add_field('video', video, filename=os.path.basename(video_path))
                    async with session.post(url, data=data) as response:
                        if response.status != 200:
                            logging.error(f"Ошибка при отправке видео в Telegram: {response.status} - {await response.text()}")
                        else:
                            logging.info("Видео успешно отправлено в Telegram.")

        except Exception as e:
            logging.error(f"Ошибка при отправке видео в Telegram: {e}")

    def stop(self):
        """Останавливает поток."""
        self.is_running = False

    def run(self):
        """Запускает основной цикл обработки."""
        self.is_running = True
        asyncio.run(self.run_async())
