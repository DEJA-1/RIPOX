#!/bin/bash

echo "Sprawdzanie środowiska..."

OS="$(uname -s)"

if [ "$OS" == "Darwin" ]; then
    ACTIVATE=".venv/bin/activate"
elif [ "$OS" == "Linux" ]; then
    ACTIVATE=".venv/bin/activate"
elif [[ "$OS" == MINGW* || "$OS" == CYGWIN* || "$OS" == MSYS* ]]; then
    ACTIVATE=".venv/Scripts/activate"
else
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Tworzenie środowiska virtualenv..."
    python3 -m venv .venv
fi

echo "Aktywacja środowiska..."
source "$ACTIVATE"

echo "Instalacja zależności..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Uruchamianie systemu..."
python src/main.py
