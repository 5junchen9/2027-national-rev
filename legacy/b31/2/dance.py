# -*- coding: utf-8 -*-
"""兼容旧入口：使用根目录的 DreamMaker 五段舞蹈任务。"""

import os
import sys


DESKTOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if DESKTOP_DIR not in sys.path:
    sys.path.insert(0, DESKTOP_DIR)

from dance_main import run


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
