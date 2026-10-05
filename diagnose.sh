#!/usr/bin/env bash
set -u
echo "=== Kernel and architecture ==="
uname -a
getconf LONG_BIT
echo "=== User permissions ==="
id
echo "=== USB and serial ==="
lsusb
ls -l /dev/serial/by-path/ /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
echo "=== Audio ==="
aplay -l
arecord -l
echo "=== Camera ==="
if command -v rpicam-still >/dev/null; then
    rpicam-still --list-cameras
fi
v4l2-ctl --list-devices
echo "=== Boot audio/I2C configuration (read only) ==="
if [[ -f /boot/firmware/config.txt ]]; then
    grep -nE 'dtoverlay|dtparam|include' /boot/firmware/config.txt
fi
