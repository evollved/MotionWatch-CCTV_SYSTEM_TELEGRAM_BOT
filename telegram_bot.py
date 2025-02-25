from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ConversationHandler,
)
import json

# Состояния для ConversationHandler
SELECT_CAMERA, SELECT_SETTING, ENTER_VALUE = range(3)

# Глобальные переменные для хранения состояния
current_camera = None
current_setting = None

# Загрузка конфигурации камер
def load_cameras_config():
    with open("config/cameras.json", "r") as f:
        return json.load(f)

# Сохранение конфигурации камер
def save_cameras_config(cameras_config):
    with open("config/cameras.json", "w") as f:
        json.dump(cameras_config, f, indent=4)

# Получение списка камер
def get_camera_names(cameras_config):
    return [camera["name"] for camera in cameras_config]

# Получение списка возможных настроек для камеры
def get_camera_settings(camera):
    # Возвращаем только те настройки, которые можно изменить через Telegram
    return [
        "detect_motion",
        "detect_objects",
        "send_photo",
        "send_video",
        "motion_sensitivity",
        "object_confidence",
        "show_live_feed",
    ]

# Формирование пронумерованного списка настроек
def format_settings_list(settings):
    return "\n".join([f"{i + 1}. {setting}" for i, setting in enumerate(settings)])

# Команда /start
async def start(update: Update, context: CallbackContext):
    cameras_config = load_cameras_config()
    camera_names = get_camera_names(cameras_config)
    await update.message.reply_text(
        "Выберите камеру для настройки:",
        reply_markup=ReplyKeyboardMarkup([camera_names], one_time_keyboard=True)
    )
    return SELECT_CAMERA

# Обработка выбора камеры
async def handle_camera_selection(update: Update, context: CallbackContext):
    global current_camera
    camera_name = update.message.text
    cameras_config = load_cameras_config()

    # Находим камеру по имени
    for camera in cameras_config:
        if camera["name"] == camera_name:
            current_camera = camera
            break

    if current_camera:
        settings = get_camera_settings(current_camera)
        formatted_settings = format_settings_list(settings)
        await update.message.reply_text(
            f"Выбрана камера: {current_camera['name']}\nВыберите настройку для изменения:\n{formatted_settings}"
        )
        return SELECT_SETTING
    else:
        await update.message.reply_text("Камера не найдена.")
        return ConversationHandler.END

# Обработка выбора настройки
async def handle_setting_selection(update: Update, context: CallbackContext):
    global current_setting
    setting_number = update.message.text

    if not setting_number.isdigit():
        await update.message.reply_text("Пожалуйста, введите номер настройки.")
        return SELECT_SETTING

    settings = get_camera_settings(current_camera)
    setting_index = int(setting_number) - 1

    if 0 <= setting_index < len(settings):
        current_setting = settings[setting_index]
        await update.message.reply_text(f"Выбрана настройка: {current_setting}\nВведите новое значение (true/false для boolean, число для числовых настроек):")
        return ENTER_VALUE
    else:
        await update.message.reply_text("Некорректный номер настройки. Попробуйте снова.")
        return SELECT_SETTING

# Обработка ввода нового значения
async def handle_new_value(update: Update, context: CallbackContext):
    global current_camera, current_setting
    new_value = update.message.text

    # Преобразуем значение в правильный тип
    if new_value.lower() in ["true", "false"]:
        new_value = new_value.lower() == "true"
    elif new_value.isdigit():
        new_value = int(new_value)
    elif new_value.replace(".", "", 1).isdigit():  # Проверка на float
        new_value = float(new_value)
    else:
        await update.message.reply_text("Некорректное значение. Попробуйте снова.")
        return ENTER_VALUE

    # Обновляем настройку камеры
    cameras_config = load_cameras_config()
    for camera in cameras_config:
        if camera["name"] == current_camera["name"]:
            camera[current_setting] = new_value
            break

    # Сохраняем изменения
    save_cameras_config(cameras_config)
    await update.message.reply_text(f"Настройка {current_setting} для камеры {current_camera['name']} изменена на {new_value}.")

    # Сбрасываем состояние
    current_camera = None
    current_setting = None
    return ConversationHandler.END

# Отмена операции
async def cancel(update: Update, context: CallbackContext):
    await update.message.reply_text("Операция отменена.")
    return ConversationHandler.END

# Основная функция для запуска бота
def start_telegram_bot(token, cameras_config):
    application = Application.builder().token(token).build()

    # Создаем ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECT_CAMERA: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_camera_selection)],
            SELECT_SETTING: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_setting_selection)],
            ENTER_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_new_value)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Регистрация ConversationHandler
    application.add_handler(conv_handler)

    # Запуск бота
    application.run_polling()