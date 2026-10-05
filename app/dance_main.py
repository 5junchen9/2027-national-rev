# -*- coding: utf-8 -*-
"""DreamMaker 舞蹈任务：按顺序播放四段启用的动作并同步播放背景音乐。"""

import os
import subprocess
import sys
from robot_audio import start


from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
MODULE_DIR = str(MODULE_DIR)
VOICE_DIR = str(VOICE_DIR)
BACKGROUND_MUSIC = str(ROOT / "assets/music/dance.mp3")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

PLUGIN_NAME = "dance"
VOICE_COMMANDS = ("跳舞", "开始跳舞")
# 第 2 段包含突然后仰姿态，实机会躺倒，先从表演序列中停用。
DANCE_ACTIONS = ("DANCE1", "DANCE3", "DANCE4", "DANCE5")


def run(robot=None):
    """播放四段启用的舞蹈动作；动作全部完成后返回 True。"""
    music = None
    owns_robot = robot is None
    if owns_robot:
        from robotmove import RobotMove
        robot = RobotMove(None)
    try:
        if os.path.isfile(BACKGROUND_MUSIC):
            try:
                music = start(BACKGROUND_MUSIC)
            except OSError as error:
                print("无法播放跳舞背景音乐：{}".format(error))
        else:
            print("缺少跳舞背景音乐：{}".format(BACKGROUND_MUSIC))

        for action in DANCE_ACTIONS:
            robot.robotMove(action)
        return True
    finally:
        if owns_robot:
            robot.close()
        if music is not None and music.poll() is None:
            music.terminate()
            try:
                music.wait(timeout=3)
            except subprocess.TimeoutExpired:
                music.kill()
                music.wait(timeout=3)


if __name__ == "__main__":
    try:
        success = run()
    except KeyboardInterrupt:
        print("跳舞任务已取消。")
        success = False
    raise SystemExit(0 if success else 1)
