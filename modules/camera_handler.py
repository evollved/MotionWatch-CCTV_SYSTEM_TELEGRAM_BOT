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
import aiofiles.os as aio_os  # Импортируем асинхронную версию os

class CameraHandler(threading.Thread):
    def __init__(self, camera_config, telegram_config):
        threading.Thread.__init__(self)
        self.camera_config = camera_config
        self.telegram_config = telegram_config
        self.camera_id = camera_config['id']
        self.name = camera_config['name']
        self.rtsp_url = camera_config['rtsp_url']
        self.enabled = camera_config['enabled']
        self.camera_width = camera_config.get('width')
        self.camera_height = camera_config.get('height')
        self.detect_motion = camera_config['detect_motion']
        self.detect_objects = camera_config['detect_objects']
        self.object_types = camera_config.get('object_types', [])  # Типы объектов для обнаружения
        self.object_confidence = camera_config.get('object_confidence', 0.5)  # Порог уверенности для объектов
        self.telegram_chat_id = camera_config['telegram_chat_id']
        self.send_photo = camera_config['send_photo']
        self.send_video = camera_config['send_video']
        self.draw_boxes = camera_config['draw_boxes']
        self.test_frame_on_start = camera_config['test_frame_on_start']
        self.test_video_on_start = camera_config['test_video_on_start']
        self.motion_sensitivity = camera_config['motion_sensitivity']
        self.codec = camera_config.get('codec', 'h264').lower()
        self.record_audio = camera_config.get('record_audio', True)
        self.reconnect_attempts = camera_config.get('reconnect_attempts', 5)  # Количество попыток переподключения
        self.reconnect_delay = camera_config.get('reconnect_delay', 10)  # Задержка между попытками (в секундах)
        self.is_running = False
        self.motion_detected = False
        self.frame_count = 0
        self.motion_frame_count = 0
        self.last_frame = None
        self.last_motion_time = None
        self.motion_threshold = 1000
        self.min_motion_frames = 5
        self.loop = asyncio.new_event_loop()
        self.model = YOLO("yolov8n.pt")  # Загружаем модель YOLOv8
        self.fps = 30  # Предполагаемый FPS (можно настроить)
        self.buffer_size = int(10 * self.fps)  # 10 секунд буфера
        self.frame_buffer = collections.deque(maxlen=self.buffer_size)  # Кольцевой буфер
        self.show_live_feed = camera_config.get("show_live_feed", False)  # Новый параметр для отображения видео

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
                print(f"Ошибка FFmpeg: {stderr.decode()}")
                return None

            frame = np.frombuffer(stdout, dtype=np.uint8)
            if frame.size == 0:
                print("FFmpeg вернул пустой кадр.")
                return None

            frame = frame.reshape((self.frame_height, self.frame_width, 3))
            return frame.copy()  # Возвращаем копию кадра

        except Exception as e:
            print(f"Ошибка при захвате кадра FFmpeg: {e}")
            return None

    async def show_processed_feed(self):
        """Отображает видео с наложенными результатами обработки."""
        try:
            bg_frame = None
            while self.is_running and self.show_live_feed:  # Проверяем параметр
                frame = await self.capture_frame_async()
                if frame is None:
                    print("Не удалось захватить кадр.")
                    break

                # Создаем копию кадра для обработки
                frame_copy = frame.copy()

                # Обработка движения
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if bg_frame is None:
                    bg_frame = gray
                    continue

                frame_delta = cv2.absdiff(bg_frame, gray)
                thresh = cv2.threshold(frame_delta, self.motion_sensitivity, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < self.motion_sensitivity:
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Обнаружение объектов (если включено)
                if self.detect_objects:
                    results = self.model(frame_copy, conf=self.object_confidence)
                    detected_objects = results[0].boxes

                    for obj in detected_objects:
                        if obj.cls in self.model.names and self.model.names[obj.cls] in self.object_types:
                            label = self.model.names[obj.cls]
                            confidence = obj.conf
                            (x, y, w, h) = obj.xyxy[0].int().tolist()
                            cv2.rectangle(frame_copy, (x, y), (w, h), (255, 0, 0), 2)
                            cv2.putText(frame_copy, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Отображение кадра с результатами обработки
                cv2.imshow(f"Обработанное видео: {self.name}", frame_copy)

                # Выход по нажатию клавиши 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Ошибка при отображении обработанного видео: {e}")
        finally:
            cv2.destroyAllWindows()

    async def run_async(self):
        if not self.enabled:
            print(f"Камера {self.name} отключена.")
            return

        print(f"Запуск обработки камеры {self.name}...")
        self.is_running = True

        # Запуск отображения обработанного видео (если включено)
        if self.show_live_feed:
            asyncio.create_task(self.show_processed_feed())

        attempt = 0
        while self.is_running and attempt < self.reconnect_attempts:
            try:
                if self.camera_width and self.camera_height:
                    self.frame_width = self.camera_width
                    self.frame_height = self.camera_height
                    print(f"Используем размеры кадра из конфигурации: width={self.frame_width}, height={self.frame_height}")
                else:
                    self.frame_width, self.frame_height = self.get_frame_size_ffmpeg()
                    if self.frame_width is None or self.frame_height is None:
                        print(f"Не удалось получить размеры кадра с камеры {self.name}")
                        self.is_running = False
                        return
                    print(f"Размеры кадра получены из потока: width={self.frame_width}, height={self.frame_height}")

                if self.test_frame_on_start:
                    await self.capture_test_frame_async()
                if self.test_video_on_start:
                    await self.capture_test_video_async(5)

                bg_frame = None
                while self.is_running:
                    frame = await self.capture_frame_async()
                    if frame is None:
                        print(f"Ошибка при захвате кадра с камеры {self.name} (FFmpeg)")
                        break

                    self.frame_buffer.append(frame)  # Добавляем кадр в буфер

                    self.frame_count += 1
                    if self.frame_count % 5 == 0:
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
                            self.motion_frame_count += 1
                            if self.motion_frame_count >= self.min_motion_frames:
                                self.motion_detected = True
                                self.last_motion_time = datetime.now()

                                if self.detect_objects:
                                    results = self.model(frame, conf=self.object_confidence)
                                    detected_objects = results[0].boxes

                                    # Фильтруем объекты по типам
                                    filtered_objects = []
                                    for obj in detected_objects:
                                        class_name = self.model.names[int(obj.cls)]  # Получаем название класса
                                        print(f"Обнаружен объект: {class_name}")  # Логирование
                                        if class_name in self.object_types:  # Сравниваем с object_types
                                            filtered_objects.append(obj)

                                    print(f"Всего обнаружено объектов: {len(detected_objects)}, отфильтровано: {len(filtered_objects)}")  # Логирование

                                    if len(filtered_objects) > 0:  # Если объекты обнаружены
                                        print(f"Обнаружены объекты: {filtered_objects}")  # Логирование
                                        if self.draw_boxes:
                                            frame = results[0].plot()  # Рисуем рамки на кадре

                                        if self.send_photo:
                                            print("Попытка отправки фото в Telegram...")  # Логирование
                                            await self.send_telegram_photo_async(frame)
                                        if self.send_video:
                                            print("Попытка отправки видео в Telegram...")  # Логирование
                                            await self.record_video_with_buffer(duration=6)
                                    else:
                                        print("Нет объектов, соответствующих фильтру.")  # Логирование

                                else:
                                    # Если обнаружение объектов отключено, отправляем уведомления только о движении
                                    if self.send_photo:
                                        print("Попытка отправки фото в Telegram...")  # Логирование
                                        await self.send_telegram_photo_async(frame)
                                    if self.send_video:
                                        print("Попытка отправки видео в Telegram...")  # Логирование
                                        await self.record_video_with_buffer(duration=6)

                                self.motion_frame_count = 0

                        else:
                            self.motion_frame_count = 0
                            self.motion_detected = False

                        bg_frame = gray

            except Exception as e:
                print(f"Ошибка при обработке камеры {self.name}: {e}")
                attempt += 1
                if attempt < self.reconnect_attempts:
                    print(f"Попытка переподключения {attempt} из {self.reconnect_attempts} через {self.reconnect_delay} секунд...")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    print(f"Превышено количество попыток переподключения для камеры {self.name}.")
                    break

        print(f"Завершение обработки камеры {self.name}")
        self.is_running = False

    async def capture_test_frame_async(self):
        """Асинхронный захват и отправка тестового кадра."""
        frame = await self.capture_frame_async()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"test_frame_{self.name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            await self.send_telegram_photo_async(frame, message=f"Тестовый кадр с камеры {self.name}")
            await aio_os.remove(filename)  # Асинхронное удаление файла
        else:
            print(f"Не удалось захватить тестовый кадр с камеры {self.name}")

    async def capture_test_video_async(self, duration=5):
        """Асинхронный захват и отправка тестового видео."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"test_video_{self.name}_{timestamp}.mp4"
        temp_filename = f"temp_test_video_{self.name}_{timestamp}.mp4"

        try:
            # Записываем видео во временный файл
            command_record = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', self.rtsp_url,
                '-t', str(duration),
                '-c:v', 'copy',
            ]
            if self.record_audio:
                command_record.extend(['-c:a', 'aac'])
            else:
                command_record.append('-an')
            command_record.append(temp_filename)
            process_record = await asyncio.create_subprocess_exec(*command_record, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await process_record.communicate()

            if process_record.returncode != 0:
                print(f"Ошибка FFmpeg при записи тестового видео.")
                return

            # Перекодируем видео в H.264, если кодек не h264
            if self.codec != 'h264':
                command_convert = [
                    'ffmpeg',
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
                    print(f"Ошибка FFmpeg при перекодировании тестового видео.")
                    await aio_os.remove(temp_filename)  # Асинхронное удаление временного файла
                    return
            else:
                await aio_os.rename(temp_filename, filename)  # Асинхронное переименование

            # Отправляем видео в Telegram
            await self.send_telegram_video_async(filename, message=f"Тестовое видео с камеры {self.name}")

        except Exception as e:
            print(f"Ошибка при записи тестового видео: {e}")
        finally:
            if await aio_os.path.exists(temp_filename):
                await aio_os.remove(temp_filename)  # Асинхронное удаление временного файла
            if await aio_os.path.exists(filename):
                await aio_os.remove(filename)  # Асинхронное удаление финального файла

    async def record_video_with_buffer(self, duration=6):
        """Записывает видео с буфером (10 секунд до движения) и продолжает запись после."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"motion_video_{self.name}_{timestamp}.mp4"
        temp_filename = f"temp_video_{self.name}_{timestamp}.mp4"

        try:
            # Записываем кадры из буфера
            buffer_frames = list(self.frame_buffer)

            # Записываем видео во временный файл
            command_record = [
                'ffmpeg',
                '-rtsp_transport', self.camera_config.get('rtsp_transport', 'tcp'),
                '-i', self.rtsp_url,
                '-t', str(duration),
                '-c:v', 'copy',
            ]
            if self.record_audio:
                command_record.extend(['-c:a', 'aac'])
            else:
                command_record.append('-an')
            command_record.append(temp_filename)
            print(f"Запуск FFmpeg для записи видео: {' '.join(command_record)}")
            process_record = await asyncio.create_subprocess_exec(*command_record, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process_record.communicate()

            if process_record.returncode != 0:
                print(f"Ошибка FFmpeg при записи видео: {stderr.decode()}")
                return

            # Перекодируем видео в H.264, если кодек не h264
            if self.codec != 'h264':
                command_convert = [
                    'ffmpeg',
                    '-i', temp_filename,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',
                    '-movflags', 'faststart',
                    filename
                ]
                print(f"Запуск FFmpeg для перекодирования видео: {' '.join(command_convert)}")
                process_convert = await asyncio.create_subprocess_exec(*command_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = await process_convert.communicate()

                if process_convert.returncode != 0:
                    print(f"Ошибка FFmpeg при перекодировании видео: {stderr.decode()}")
                    await aio_os.remove(temp_filename)  # Асинхронное удаление временного файла
                    return
            else:
                await aio_os.rename(temp_filename, filename)  # Асинхронное переименование

            # Отправляем видео в Telegram
            print(f"Отправка видео в Telegram: {filename}")
            await self.send_telegram_video_async(filename, message=f"Движение обнаружено на камере {self.name}!")

        except Exception as e:
            print(f"Ошибка при записи/перекодировании видео: {e}")
        finally:
            if await aio_os.path.exists(temp_filename):
                await aio_os.remove(temp_filename)  # Асинхронное удаление временного файла
            if await aio_os.path.exists(filename):
                await aio_os.remove(filename)  # Асинхронное удаление финального файла

    async def send_telegram_photo_async(self, frame, message="Движение обнаружено!"):
        """Асинхронная отправка фото в Telegram."""
        try:
            print("Попытка отправки фото в Telegram...")  # Логирование
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_chat_id

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"motion_photo_{self.name}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Файл {filename} создан.")  # Логирование

            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            async with aiohttp.ClientSession() as session:
                with open(filename, 'rb') as photo:
                    data = aiohttp.FormData()
                    data.add_field('chat_id', chat_id)
                    data.add_field('caption', message)
                    data.add_field('photo', photo, filename=filename)
                    async with session.post(url, data=data) as response:
                        if response.status != 200:
                            print(f"Ошибка при отправке фото в Telegram: {response.status} - {await response.text()}")
                        else:
                            print("Фото успешно отправлено в Telegram.")  # Логирование

            await aio_os.remove(filename)  # Асинхронное удаление файла
        except Exception as e:
            print(f"Ошибка при отправке фото в Telegram: {e}")

    async def send_telegram_video_async(self, video_path, message="Движение обнаружено!"):
        """Асинхронная отправка видео в Telegram."""
        try:
            print("Попытка отправки видео в Telegram...")  # Логирование
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
                            print(f"Ошибка при отправке видео в Telegram: {response.status} - {await response.text()}")
                        else:
                            print("Видео успешно отправлено в Telegram.")  # Логирование

        except Exception as e:
            print(f"Ошибка при отправке видео в Telegram: {e}")

    def stop(self):
        """Останавливает поток."""
        self.is_running = False

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_async())
