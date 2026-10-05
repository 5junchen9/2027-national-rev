#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$task_root"
if (( EUID == 0 )); then
    echo "Run this script as the desktop user; it uses sudo only for apt."
    exit 1
fi
if [[ "$(dpkg --print-architecture)" != "arm64" ]]; then
    echo "The included face/emotion models require 64-bit Raspberry Pi OS (arm64)."
    exit 1
fi
sudo apt update
sudo apt install -y python3 python3-venv python3-dev build-essential \
    python3-opencv python3-numpy python3-picamera2 python3-pyaudio python3-rpi.gpio python3-tk \
    libzbar0 portaudio19-dev mplayer alsa-utils i2c-tools usbutils v4l-utils \
    wget unzip file fonts-noto-cjk
if [[ ! -f venv/pyvenv.cfg ]]; then
    python3 -m venv --system-site-packages venv
fi
venv/bin/python3 -m pip install -r requirements.txt
chmod +x assets/models/face/face.eim assets/models/emotion/emotion.eim
venv/bin/python3 main.py check
echo "Installation complete. No startup service or boot configuration was created."
echo "Next: venv/bin/python3 main.py simulate"
