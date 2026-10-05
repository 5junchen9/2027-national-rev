"""统一路径与环境配置；外部环境变量优先于配置文件。"""
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "hardware"
VOICE_DIR = ROOT / "assets/voice"
for directory in (ROOT / "app", MODULE_DIR, ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def read_config(path):
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = shlex.split(line, comments=True)
        if not fields:
            continue
        if len(fields) != 1 or "=" not in fields[0]:
            raise ValueError(f"Invalid configuration line {number}: {path}")
        key, value = fields[0].split("=", 1)
        if not (key.startswith("ROBOT_") or key == "VOICE_SERIAL_PORT"):
            raise ValueError(f"Unsupported configuration key: {key}")
        if key in values:
            raise ValueError(f"Duplicate configuration key: {key}")
        values[key] = value
    return values


for key, value in read_config(ROOT / "config/robot.env").items():
    os.environ.setdefault(key, value)
