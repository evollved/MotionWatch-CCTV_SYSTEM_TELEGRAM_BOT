#!/bin/bash

# Путь к папке с виртуальным окружением
VENV_DIR="$HOME/python_venv/motiowatch"

# Активация виртуального окружения
source $VENV_DIR/bin/activate

# Переход в папку с проектом
cd $HOME/my_cctv_bot

# Запуск основного скрипта
python main.py

# Деактивация виртуального окружения
deactivate