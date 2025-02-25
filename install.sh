#!/bin/bash

# Путь к папке с виртуальным окружением
VENV_DIR="$HOME/python_venv/motiowatch"

# Создание папки для виртуального окружения, если она не существует
mkdir -p $VENV_DIR

# Переход в папку с виртуальным окружением
cd $VENV_DIR

# Создание виртуального окружения
python3 -m venv .

# Активация виртуального окружения
source bin/activate

# Установка зависимостей из requirements.txt
pip install -r $HOME/my_cctv_bot/requirements.txt

# Деактивация виртуального окружения
deactivate

echo "Виртуальное окружение и зависимости успешно установлены в $VENV_DIR"