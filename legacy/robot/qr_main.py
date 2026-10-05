# -*- coding: utf-8 -*-
"""扫码读取图片路线或 F6,L90,F5 指令。默认仅预览，--execute 才执行。"""

import argparse
import os
import re
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "b31", "2")
if not os.path.isdir(MODULE_DIR):
    MODULE_DIR = os.path.join(BASE_DIR, "modules")
sys.path.insert(0, MODULE_DIR)

def parse_route(payload, left_turn_actions=1, right_turn_actions=1):
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
            action = "BIGLEFT" if turn.group(1) == "L" else "BIGRIGHT"
            actions.extend([action] * (left_turn_actions if turn.group(1) == "L" else right_turn_actions))
        else:
            raise ValueError("路线格式错误：{}".format(token))
        if len(actions) > 500:
            raise ValueError("路线总动作次数超过 500")
    if not actions:
        raise ValueError("二维码没有路线指令")
    return actions


def scan_route():
    from robotbarcode import RobotBarcode
    from roboteye import RobotEye

    eye = RobotEye()
    barcode = RobotBarcode()
    print("请将路线二维码对准摄像头；按 Ctrl+C 退出。")
    deadline = time.monotonic() + 60
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


def execute(eye, actions):
    # 延迟导入：--dry-run 扫码时不打开机器人串口。
    from robotmove import RobotMove

    robot = RobotMove(eye)
    for action in actions:
        robot.robotMove(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", help="直接测试二维码内的网址或路线文字")
    source.add_argument("--image", help="直接测试本地路线图片")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只预览（默认）")
    mode.add_argument("--execute", action="store_true", help="调用机器人动作")
    parser.add_argument("--longest-steps", type=int, default=8, help="图片中最长一段的前进次数，默认8")
    parser.add_argument("--reverse", action="store_true", help="从另一端点开始")
    parser.add_argument("--left-turn-actions", type=int, help="实测左转90度需要的 BIGLEFT 次数")
    parser.add_argument("--right-turn-actions", type=int, help="实测右转90度需要的 BIGRIGHT 次数")
    args = parser.parse_args()
    if args.execute and (args.left_turn_actions is None or args.right_turn_actions is None):
        parser.error("执行前须填写实测的 --left-turn-actions 和 --right-turn-actions")
    eye = None
    payload = args.source
    if payload is None and args.image is None:
        eye, payload = scan_route()
    if args.image or payload.lower().startswith(("https://", "http://")):
        import cv2
        import numpy as np
        from qr_route_image import fetch_image, extract_route

        image = (cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR)
                 if args.image else fetch_image(payload))
        if image is None:
            raise ValueError("无法读取路线图片")
        payload, preview = extract_route(image, args.longest_steps, args.reverse)
        output = os.path.join(BASE_DIR, "qr_route_preview.png")
        cv2.imencode(".png", preview)[1].tofile(output)
        print("路线标注图：" + output)
        print("起点为最左侧端点（--reverse 可反向）；机器人初始朝向须对准第一段。")
    actions = parse_route(payload,
                          args.left_turn_actions if args.left_turn_actions is not None else 1,
                          args.right_turn_actions if args.right_turn_actions is not None else 1)
    print("路线指令：" + payload)
    if not args.execute:
        print("预览完成，未发送动作。请核对标注图起点和顺序；L90/R90 不是一次原有转向动作。")
        return
    if eye is None:
        from roboteye import RobotEye
        eye = RobotEye()
    execute(eye, actions)
    print("路线执行完成。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已取消。")
    except Exception as error:
        print("任务停止：{}".format(error), file=sys.stderr)
        sys.exit(1)
