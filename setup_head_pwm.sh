#!/usr/bin/env bash
# Run manually; no systemd service, boot configuration or robot motion.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$task_root/.head_pwm"
model="$(tr -d '\0' < /proc/device-tree/model)"
case "$model" in
  'Raspberry Pi 4 '*) ;;
  *) echo "This setup was prepared for Raspberry Pi 4. Found: $model"; exit 1 ;;
esac
case "${1:-prepare}" in
  prepare)
    sudo apt install -y python3-pigpio build-essential wget ca-certificates
    if [ ! -x "$runtime_dir/pigpiod" ]; then
      build_dir="$(mktemp -d /tmp/robot-pigpio.XXXXXX)"
      trap 'rm -rf -- "$build_dir"' EXIT
      # Official v79 commit, pinned so reruns use the same driver source.
      wget -O "$build_dir/source.tar.gz" \
        'https://codeload.github.com/joan2937/pigpio/tar.gz/c33738a320a3e28824af7807edafda440952c05d'
      tar -xzf "$build_dir/source.tar.gz" -C "$build_dir" --strip-components=1
      make -C "$build_dir" -j2 pigpiod
      mkdir -p "$runtime_dir"
      install -m 0755 "$build_dir/pigpiod" "$build_dir/libpigpio.so.1" "$runtime_dir/"
    fi
    ;;
  start) ;;
  *) echo 'Usage: bash setup_head_pwm.sh [prepare|start]'; exit 1 ;;
esac
if [ ! -x "$runtime_dir/pigpiod" ]; then
  echo 'First run: bash setup_head_pwm.sh'; exit 1
fi
if pgrep -x pigpiod >/dev/null; then
  # Do not silently replace another application's daemon/clock configuration.
  echo 'pigpiod already running. Current command:'
  ps -C pigpiod -o args=
else
  # PWM clock leaves PCM/I2S for WM8960; updates restricted to BCM27.
  # Foreground mode preserves initialization errors which daemon mode hides.
  # An IPv4 allowlist avoids -l binding only ::1 while clients use 127.0.0.1.
  sudo -v
  sudo env LD_LIBRARY_PATH="$runtime_dir" nohup "$runtime_dir/pigpiod" \
    -g -t 0 -l -n 127.0.0.1 -x 0x08000000 >"$runtime_dir/pigpiod.log" 2>&1 </dev/null &
fi
if /usr/bin/python3 - <<'PY'
import pigpio
import socket
import time
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    try:
        with socket.create_connection(('127.0.0.1',8888), timeout=.3):
            pass
        break
    except OSError:
        time.sleep(.25)
else:
    raise SystemExit('pigpiod port not ready after 15 seconds.')
pi = pigpio.pi('127.0.0.1',8888)
connected = pi.connected
pi.stop()
if not connected:
    raise SystemExit('pigpiod connection failed.')
print('pigpio ready. No servo command sent. No boot startup installed.')
PY
then
  exit 0
else
  echo 'Driver startup failed. Process status:'
  ps -C pigpiod -o pid=,args= || true
  echo "Startup log: $runtime_dir/pigpiod.log"
  if [ -f "$runtime_dir/pigpiod.log" ]; then
    tail -n 60 "$runtime_dir/pigpiod.log"
  else
    echo 'No local startup log; the existing daemon was not replaced.'
  fi
  exit 1
fi
