#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
test "$(uname -m)" = "aarch64" || { echo "需要 64 位树莓派系统（aarch64）"; exit 1; }

sudo apt update
sudo apt install -y python3-pip python3-opencv
python3 -m pip install --user --break-system-packages edge_impulse_linux
chmod +x models/face/face.eim
./models/face/face.eim --print-info
