# -*- coding: utf-8 -*-
"""扫码读取图片路线或 F6,L90,F5 指令，并立即发送给 STM32。"""

import argparse
import os
import re
import subprocess
import sys
import time
from robot_audio import play


from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
MODULE_DIR = str(MODULE_DIR)
if not os.path.isdir(MODULE_DIR):
    MODULE_DIR = os.path.join(BASE_DIR, "modules")
sys.path.insert(0, MODULE_DIR)

PLUGIN_NAME = "qr_route"
VOICE_COMMANDS = ("二维码循迹", "扫描二维码")
START_AUDIO = str(VOICE_DIR / "识别到二维码开始动作.mp3")


def parse_route(payload, left_turn_actions=7, right_turn_actions=7):
    """把二维码文本转换为动作列表；不认识的内容直接报错，不移动机器人。"""
    if not 1 <= left_turn_actions <= 30 or not 1 <= right_turn_actions <= 30:
        raise ValueError("90 度转向的动作次数必须为 1 至 30")
    if len(payload) > 1000:
        raise ValueError("路线文本过长")
    actions = []
    for token in payload.upper().replace(" ", "").split(","):
        forward = re.fullmatch(r"F([1-9][0-9]{0,2})", token)
        turn = re.fullmatch(r"([LR])90", token)
        if forward:
            steps = int(forward.group(1))
            if steps > 100:
                raise ValueError("每段最多 100 次前进动作")
            actions.extend(["UP"] * steps)
        elif turn:
            action = "TURN_LEFT" if turn.group(1) == "L" else "TURN_RIGHT"
            actions.extend([action] * (left_turn_actions if turn.group(1) == "L" else right_turn_actions))
        else:
            raise ValueError("路线格式错误：{}".format(token))
        if len(actions) > 500:
            raise ValueError("路线总动作次数超过 500")
    if not actions:
        raise ValueError("二维码没有路线指令")
    return actions


def format_action_plan(actions):
    groups = []
    for action in actions:
        if groups and groups[-1][0] == action:
            groups[-1][1] += 1
        else:
            groups.append([action, 1])
    return " ".join("{}x{}".format(action, count) for action, count in groups)


def scan_route():
    from robotbarcode import RobotBarcode
    from roboteye import RobotEye

    eye = RobotEye()
    barcode = RobotBarcode()
    print("请将路线二维码对准摄像头；按 Ctrl+C 退出。")
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            ok, image = eye.getImage()
            if not ok:
                time.sleep(0.05)
                continue
            found, image, codes = barcode.GetBarcodes(image)
            eye.showImage(image)
            if found:
                payload = codes[0].data.decode("utf-8").strip()
                print("扫码结果：{}".format(payload))
                return eye, payload
        raise RuntimeError("60 秒未扫描到二维码，请检查摄像头及二维码清晰度")
    except BaseException:
        eye.close()
        raise


def execute(eye, actions, robot=None):
    # 延迟导入：--dry-run 扫码时不打开机器人串口。
    owns_robot = robot is None
    if owns_robot:
        from robotmove import RobotMove
        robot = RobotMove(eye)
    try:
        for action in actions:
            robot.robotMove(action)
    finally:
        if owns_robot:
            robot.close()


def run(source=None, image_path=None, dry_run=False, longest_steps=8,
        reverse=False, left_turn_actions=7, right_turn_actions=7, robot=None):
    """二维码任务插件入口；返回是否成功。"""
    eye = None
    try:
        payload = source
        if payload is None and image_path is None:
            eye, payload = scan_route()
        if image_path or payload.lower().startswith(("https://", "http://")):
            import cv2
            import numpy as np
            from qr_route_image import fetch_image, extract_route

            image = (cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
                     if image_path else fetch_image(payload))
            if image is None:
                raise ValueError("无法读取路线图片")
            payload, preview = extract_route(image, longest_steps, reverse)
            output = os.path.join(BASE_DIR, "qr_route_preview.png")
            cv2.imencode(".png", preview)[1].tofile(output)
            print("路线标注图：" + output)
            print("起点为左侧且最接近画面中部的线段头（reverse=True 可反向）；机器人初始朝向须对准第一段。")
        actions = parse_route(payload, left_turn_actions, right_turn_actions)
        print("路线指令：" + payload)
        print("动作计划：" + format_action_plan(actions))
        if dry_run:
            print("预览完成，未向 STM32 发送动作。")
            return True
        if os.path.isfile(START_AUDIO):
            play(START_AUDIO)
        else:
            print("缺少二维码开始提示音：{}".format(START_AUDIO))
        execute(eye, actions, robot)
        print("路线已转换为 DreamMaker 动作帧并发送完成。")
        return True
    finally:
        if eye is not None:
            eye.close()
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", help="直接测试二维码内的网址或路线文字")
    source.add_argument("--image", help="直接测试本地路线图片")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不向 STM32 发送动作")
    parser.add_argument("--longest-steps", type=int, default=8, help="图片中最长一段的前进次数，默认8")
    parser.add_argument("--reverse", action="store_true", help="从另一端点开始")
    parser.add_argument("--left-turn-actions", type=int, default=7, help="左转90度的完整动作次数，默认7")
    parser.add_argument(
        "--right-turn-actions", type=int, default=7,
        help="右转90度的完整动作次数，默认7",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    return run(
        source=args.source,
        image_path=args.image,
        dry_run=args.dry_run,
        longest_steps=args.longest_steps,
        reverse=args.reverse,
        left_turn_actions=args.left_turn_actions,
        right_turn_actions=args.right_turn_actions,
    )


if __name__ == "__main__":
    try:
        success = main()
    except KeyboardInterrupt:
        print("已取消。")
        success = False
    except Exception as error:
        print("任务停止：{}".format(error), file=sys.stderr)
        success = False
    raise SystemExit(0 if success else 1)
