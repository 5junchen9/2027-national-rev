# -*- coding: utf-8 -*-
"""USB-TTL 语音总控：读取数字指令并调用实际任务插件。"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time


VOICE_BAUD = 9600
from robot_audio import CARD as AUDIO_CARD, SETTINGS as AUDIO_SETTINGS, configure, play
from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
VOICE_DIR = str(VOICE_DIR)
READY_AUDIO = os.path.join(VOICE_DIR, "小童已就绪.wav")
SONG_AUDIO = os.path.join(VOICE_DIR, "唱歌.mp3")
JOKE_AUDIO = os.path.join(VOICE_DIR, "笑话.mp3")
MODULE_DIR = str(MODULE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


def _run_face(_robot):
    import face_main
    return face_main.run(quick=True)


def _run_qr(robot):
    import qr_main
    return qr_main.run(robot=robot)


def _run_carry(robot):
    import carry_main
    return carry_main.run(robot=robot)


def _run_kick(robot):
    import kick_main
    return kick_main.run(robot=robot)


def _run_dance(robot):
    import dance_main
    return dance_main.run(robot=robot)


def _run_emotion(robot):
    import emotion_main
    return emotion_main.run(robot=robot)


def _play_audio(audio_path, description):
    try:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)
        play(audio_path)
        return True
    except (OSError, subprocess.SubprocessError) as error:
        print("[语音] {}播放失败：{}".format(description, error))
        return False


def _run_song(_robot):
    return _play_audio(SONG_AUDIO, "歌曲")


def _run_joke(_robot):
    return _play_audio(JOKE_AUDIO, "笑话")


def _run_action(robot, action):
    robot.robotMove(action)
    return True


TASKS = {
    "1": ("人脸识别", _run_face, "完成_人脸识别.mp3"),
    "2": ("二维码循迹", _run_qr, "完成_二维码循迹.mp3"),
    "3": ("搬运", _run_carry, "搬运完成准备踢球.mp3"),
    "4": ("踢球", _run_kick, "完成_踢球.mp3"),
    "5": ("跳舞", _run_dance, "完成_跳舞.mp3"),
    "6": ("向左转", lambda robot: _run_action(robot, "BIGLEFT"), None),
    "7": ("向右转", lambda robot: _run_action(robot, "RIGHT1"), None),
    "8": ("向前", lambda robot: _run_action(robot, "UP_LITTLE"), None),
    "9": ("向后", lambda robot: _run_action(robot, "BACK"), None),
    "12": ("跳舞", _run_dance, None),
    "13": ("情绪识别", _run_emotion, None),
    "14": ("唱歌", _run_song, None),
    "15": ("讲笑话", _run_joke, None),
}

PHRASE_COMMANDS = {
    "开始向左转": "6",
    "开始向右转": "7",
    "开始向前": "8",
    "开始向后": "9",
}


def configure_audio():
    try:
        return configure()
    except (OSError, subprocess.SubprocessError) as error:
        print("Audio configuration failed: {}".format(error))
        return False


def announce_ready():
    return _play_audio(READY_AUDIO, "就绪提示")


def parse_command(data):
    """兼容中文口令、ASCII 整数或单字节数值 0~9。"""
    if isinstance(data, str):
        text = data
        data = data.encode("utf-8")
    else:
        text = data.decode("utf-8", errors="ignore")
    for phrase, key in PHRASE_COMMANDS.items():
        if phrase in text:
            return key

    # 先按完整整数解析，避免 ASCII "12" 被误判为旧指令 "1"。
    numbers = re.findall(r"\d+", text)
    if numbers:
        return next((number for number in numbers
                     if number in TASKS or number == "0"), None)

    for value in data:
        if 0 <= value <= 9:
            return str(value)
    return None


def unique_physical_ports(paths, resolver=os.path.realpath):
    """合并同一串口的 by-id 别名和 tty 设备节点。"""
    unique = {}
    for path in paths:
        unique.setdefault(resolver(path), path)
    return list(unique.values())


def voice_port_candidates(requested=None):
    paths = ([requested] if requested else []) + (
        sorted(glob.glob("/dev/serial/by-path/*"))
        + sorted(glob.glob("/dev/serial/by-id/*"))
        + sorted(glob.glob("/dev/ttyUSB*"))
        + sorted(glob.glob("/dev/ttyACM*"))
    )
    return unique_physical_ports(paths)


def find_voice_port(requested=None):
    if requested:
        return requested
    candidates = voice_port_candidates()
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError("未发现 USB-TTL 串口")
    raise RuntimeError("发现多个串口，请用 --port 指定：{}".format(", ".join(candidates)))


def open_voice_serial(serial_module, requested, robot_port, baud):
    errors = []
    for port in voice_port_candidates(requested):
        if os.path.realpath(port) == os.path.realpath(robot_port):
            continue
        try:
            return serial_module.Serial(port, baud, timeout=0.5), port
        except (OSError, serial_module.SerialException) as error:
            errors.append("{}: {}".format(port, error))
    detail = "; ".join(errors) or "没有机器人端口以外的可用串口"
    raise RuntimeError("无法打开语音串口：{}".format(detail))


def run_task(key, robot):
    description, task, completion_audio = TASKS[key]
    print("[语音] 开始：{}".format(description))
    try:
        success = bool(task(robot))
    except Exception as error:
        print("[语音] {}异常：{}".format(description, error))
        return False
    print("[语音] {}：{}".format(description, "完成" if success else "失败"))
    if success and completion_audio:
        audio_path = os.path.join(VOICE_DIR, completion_audio)
        if os.path.isfile(audio_path):
            _play_audio(audio_path, "完成提示")
        else:
            print("[语音] 缺少完成提示音：{}".format(audio_path))
    return success


def voice_switch_mode(voice_serial, robot):
    """处理一条语音模块输入；返回 True 表示退出总控。"""
    data = voice_serial.readline()
    if not data:
        return False
    key = parse_command(data)
    if key is None:
        numbers = re.findall(r"\d+", data.decode("utf-8", errors="ignore"))
        if any(number in {"10", "11"} for number in numbers):
            return False
        print("[语音] 无法识别的命令：{}".format(repr(data)))
        return False
    if key == "0":
        print("[语音] 收到退出指令")
        return True
    voice_serial.reset_input_buffer()
    try:
        run_task(key, robot)
    finally:
        # 丢弃任务期间语音模块积压的旧指令。
        voice_serial.reset_input_buffer()
    print("[语音] 继续等待指令：1-9原任务 12跳舞 13情绪识别 14唱歌 15讲笑话 0退出")
    return False


def self_check():
    assert parse_command(b"1\r\n") == "1"
    assert parse_command(b"CMD:2") == "2"
    assert parse_command(bytes((3,))) == "3"
    assert parse_command(bytes((4,))) == "4"
    assert parse_command(bytes((5,))) == "5"
    assert parse_command(bytes((6,))) == "6"
    assert parse_command(b"9\r\n") == "9"
    assert parse_command(b"CMD:12\r\n") == "12"
    assert parse_command(b"13\r\n") == "13"
    assert parse_command(b"14\r\n") == "14"
    assert parse_command(b"15\r\n") == "15"
    assert parse_command(b"10\r\n") is None
    assert parse_command(b"CMD:11\r\n") is None
    assert parse_command("开始向左转") == "6"
    assert parse_command("请开始向右转") == "7"
    assert parse_command("开始向前") == "8"
    assert parse_command("开始向后") == "9"
    assert parse_command(b"noise") is None
    assert AUDIO_CARD == os.environ.get("ROBOT_AUDIO_CARD", "wm8960soundcard")
    assert len(AUDIO_SETTINGS) == 6
    assert os.path.isfile(READY_AUDIO)
    aliases = ["/dev/serial/by-id/voice", "/dev/ttyUSB0"]
    targets = {aliases[0]: "/dev/ttyUSB0", aliases[1]: "/dev/ttyUSB0"}
    assert unique_physical_ports(aliases, targets.get) == [aliases[0]]
    assert all(callable(task) for _, task, _ in TASKS.values())
    assert all(task.__code__.co_argcount == 1 for _, task, _ in TASKS.values())
    assert all(not audio or os.path.isfile(os.path.join(VOICE_DIR, audio))
               for _, _, audio in TASKS.values())
    assert os.path.isfile(SONG_AUDIO)
    if not os.path.isfile(JOKE_AUDIO):
        print("提示：笑话音频待放入 {}".format(JOKE_AUDIO))
    print("USB-TTL 语音总控映射自检通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=os.environ.get("VOICE_SERIAL_PORT") or None,
                        help="语音模块串口；不填时自动查找")
    parser.add_argument("--robot-port", default=os.environ.get(
        "ROBOT_SERIAL_PORT", "/dev/ttyUSB1"), help="机器人控制串口")
    parser.add_argument("--baud", type=int, default=VOICE_BAUD, help="波特率，默认9600")
    parser.add_argument("--check", action="store_true", help="只检查指令映射，不打开硬件")
    args = parser.parse_args()
    if args.check:
        self_check()
        return True

    import serial

    voice_serial = None
    robot = None
    try:
        voice_serial, port = open_voice_serial(
            serial, args.port, args.robot_port, args.baud
        )
        from robotmove import RobotMove
        robot = RobotMove(None, port=args.robot_port)
        if not configure_audio():
            raise RuntimeError("Cannot configure audio card")
        announce_ready()
    except Exception as error:
        if robot is not None:
            robot.close()
        if voice_serial is not None:
            voice_serial.close()
        print("初始化语音或机器人串口失败：{}".format(error))
        print("请检查接线后运行：lsusb; ls -l /dev/ttyUSB* /dev/ttyACM*")
        return False

    print("USB-TTL 语音串口已打开：{} @ {}".format(port, args.baud))
    print("机器人串口已联调并进入站立姿态：{} @ 115200".format(args.robot_port))
    print("等待指令：1-9原任务 12跳舞 13情绪识别 14唱歌 15讲笑话 0退出")
    try:
        while True:
            try:
                if voice_switch_mode(voice_serial, robot):
                    break
            except (OSError, serial.SerialException) as error:
                print("语音USB串口已断开：{}".format(error))
                try:
                    voice_serial.close()
                except Exception:
                    pass
                while True:
                    try:
                        voice_serial, port = open_voice_serial(
                            serial, args.port, args.robot_port, args.baud
                        )
                        print("语音USB串口已重新连接：{} @ {}".format(
                            port, args.baud
                        ))
                        break
                    except RuntimeError as reconnect_error:
                        print("等待语音USB串口：{}".format(reconnect_error))
                        time.sleep(1)
    except KeyboardInterrupt:
        print("\n手动中断，退出")
    finally:
        robot.close()
        if voice_serial is not None:
            voice_serial.close()
        print("语音串口已关闭")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
