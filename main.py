import json
import yaml
import threading
import time
import logging
from modules.camera_handler import CameraHandler
from ultralytics import YOLO

def load_config(filename):
    """Загружает конфигурацию из JSON или YAML файла."""
    try:
        with open(filename, 'r') as f:
            if filename.endswith('.json'):
                return json.load(f)
            elif filename.endswith('.yaml') or filename.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                print(f"Ошибка: Неподдерживаемый формат файла {filename}.")
                return None
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации {filename} не найден.")
        return None
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"Ошибка: Некорректный формат в файле {filename}. {e}")
        return None

def validate_camera_config(camera_config):
    """Проверяет, что конфигурация камеры заполнена правильно."""
    required_keys = ["id", "name", "rtsp_url", "enabled", "detect_motion", "detect_objects", "telegram_chat_id", "send_photo", "send_video", "draw_boxes", "test_frame_on_start", "test_video_on_start"]
    optional_keys = ["object_types", "width", "height", "fps", "codec", "record_audio", "reconnect_attempts", "reconnect_delay"]

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cameras_config = load_config("config/cameras.yaml")
    telegram_config = load_config("config/telegram.yaml")

    if not cameras_config or not telegram_config:
        logging.error("Завершение работы.")
        return

    # Извлекаем список камер из конфигурации
    if "cameras" in cameras_config:
        cameras_list = cameras_config["cameras"]
    else:
        logging.error("Ошибка: Конфигурация камер не найдена или имеет неправильный формат.")
        return

    if not validate_camera_config(cameras_list):
        logging.error("Обнаружены ошибки в конфигурации камер. Завершение работы.")
        return

    # Загрузка модели YOLO
    yolo_model = YOLO("yolov8n.pt")
    yolo_model.fuse()  # Ускорение инференса

    threads = []
    for camera in cameras_list:
        camera_thread = CameraHandler(camera, telegram_config, yolo_model, device="cpu")
        threads.append(camera_thread)
        camera_thread.start()

    try:
        while True:
            time.sleep(1)  # Держим основной поток активным
    except KeyboardInterrupt:
        logging.info("Завершение работы...")
        for thread in threads:
            thread.stop()  # Отправляем сигнал остановки каждому потоку
        for thread in threads:
            thread.join()  # Ждем завершения всех потоков
        logging.info("Все потоки завершены.")

if __name__ == "__main__":
    main()
