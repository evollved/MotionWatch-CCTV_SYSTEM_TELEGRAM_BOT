import json
import threading
import time
import os
import asyncio
from modules.camera_handler import CameraHandler
from telegram_bot import start_telegram_bot

def load_config(filename):
    """Загружает конфигурацию из JSON файла."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации {filename} не найден.")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка: Некорректный JSON в файле {filename}.")
        return None

def validate_camera_config(camera_config):
    """Проверяет, что конфигурация камеры заполнена правильно."""
    required_keys = ["id", "name", "rtsp_url", "enabled", "detect_motion", "detect_objects", "telegram_chat_id", "send_photo", "send_video", "draw_boxes", "test_frame_on_start", "test_video_on_start"]
    optional_keys = ["object_types", "width", "height", "motion_sensitivity", "codec", "record_audio", "reconnect_attempts", "reconnect_delay"]

    for camera in camera_config:
        for key in required_keys:
            if key not in camera:
                print(f"Ошибка: Отсутствует ключ '{key}' в конфигурации камеры с id {camera.get('id', 'N/A')}")
                return False
        # Проверка опциональных полей
        for key in optional_keys:
            if key not in camera:
                print(f"Предупреждение: Отсутствует опциональный ключ '{key}' в конфигурации камеры с id {camera.get('id', 'N/A')}")
    return True

def main():
    """Основная функция."""
    cameras_config = load_config("config/cameras.json")
    telegram_config = load_config("config/telegram.json")

    if not cameras_config or not telegram_config:
        print("Завершение работы.")
        return

    if not validate_camera_config(cameras_config):
        print("Обнаружены ошибки в конфигурации камер. Завершение работы.")
        return

    # Запуск Telegram бота
    start_telegram_bot(telegram_config["bot_token"], cameras_config)

    threads = []
    for camera in cameras_config:
        camera_thread = CameraHandler(camera, telegram_config)
        threads.append(camera_thread)
        camera_thread.start()

    try:
        while True:
            time.sleep(1)  # Держим основной поток активным
    except KeyboardInterrupt:
        print("Завершение работы...")
        for thread in threads:
            thread.stop()  # Отправляем сигнал остановки каждому потоку
        for thread in threads:
            thread.join()  # Ждем завершения всех потоков
        print("Все потоки завершены.")

if __name__ == "__main__":
    main()