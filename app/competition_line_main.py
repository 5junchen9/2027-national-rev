"""额外的寻线比赛版本，原competition.py保持不变。"""
import argparse
import json
from pathlib import Path
import time
from contextlib import ExitStack
from unittest.mock import Mock

import cv2
from robot_config import ROOT
from competition_main import (CompetitionIO, GuardedRobot, BODY_PORT, CONFIG_FILE,
                              load_settings, preflight)
from carry_vision import steering, find_blocks
from kick_shapes import overlap
from dual_kick import camera_settings
from line_search_route import detect_line, LinePlanner, run_course


class LineCompetitionIO(CompetitionIO):
    line_color = "white"

    def follow_to_carry(self, color):
        self.open_views()
        self.stream.watch(None, None)  # 不等待left和action1，也不让二维码触发转向。
        planner = LinePlanner()
        self.set_head(127)
        block_frames = 0
        previous_block = None
        deadline = min(self.deadline, time.monotonic() + self.settings["route_timeout_seconds"])
        for _ in range(self.settings["route_max_steps"]):
            head, belly, _ = self.observe_ready()
            if time.monotonic() >= deadline:
                raise RuntimeError("寻线超时，停止比赛")
            blocks = find_blocks(head, color)
            block = max(blocks, key=lambda box: box.width * box.height) if blocks else None
            if block is not None:
                same_block = previous_block is not None and overlap(previous_block, block) > .3
                block_frames = block_frames + 1 if same_block else 1
                previous_block = block
                # 出现候选时停步确认，由搬运模块继续找物和对位。
                if block_frames >= 3:
                    print("指定颜色物块连续确认3帧，转入搬运。", flush=True)
                    return
                continue
            block_frames = 0
            previous_block = None
            x = detect_line(belly, self.line_color, planner.previous_x)
            # 连续两幅新画面没线后才开始局部搜索。
            if x is None:
                _, belly, _ = self.observe_ready()
                x = detect_line(belly, self.line_color, planner.previous_x)
            far_x = detect_line(belly, self.line_color, x, band=(.55, .75))
            flip = camera_settings()["belly"]["flip"]
            action = planner.decide(x, belly.shape[1], far_x, mirrored=flip in ("1", "-1"))
            display = belly.copy()
            top = int(display.shape[0] * .8)
            cv2.line(display, (0, top), (display.shape[1]-1, top), (0, 255, 255), 2)
            if x is not None:
                cv2.circle(display, (round(x), (top+display.shape[0])//2), 8, (0, 0, 255), 2)
            if far_x is not None:
                cv2.circle(display, (round(far_x), round(display.shape[0]*.65)), 8, (255, 0, 255), 2)
            cv2.putText(display, action, (10, 30), 0, .7, (0, 255, 0), 2)
            cv2.imshow("line search", display)
            if action == "STOP":
                raise RuntimeError("向右搜索4次仍未找到线或指定物块，停止比赛")
            print("[寻线]", action, "线位置：", x, flush=True)
            if action.startswith("SIDE_"):
                action = steering(action, flip)
            self.move(action)
        raise RuntimeError("寻线超过动作上限，停止比赛")


def preview_line(settings, line_color):
    io = LineCompetitionIO(settings, None, time.monotonic() + settings["total_seconds"])
    planner = LinePlanner()
    try:
        io.open_views()
        print("仅预览线路，不连接身体串口；Q退出。")
        while True:
            _, belly, _ = io.observe()
            x = detect_line(belly, line_color, planner.previous_x)
            far_x = detect_line(belly, line_color, x, band=(.55, .75))
            if x is not None:
                planner.previous_x = x
            display = belly.copy()
            top = int(display.shape[0] * .8)
            cv2.line(display, (0, top), (display.shape[1]-1, top), (0, 255, 255), 2)
            if x is not None:
                cv2.circle(display, (round(x), (top+display.shape[0])//2), 8, (0, 0, 255), 2)
            for fraction in (.55, .75):
                y = round(display.shape[0] * fraction)
                cv2.line(display, (0, y), (display.shape[1]-1, y), (255, 0, 255), 1)
            if far_x is not None:
                cv2.circle(display, (round(far_x), round(display.shape[0]*.65)), 8, (255, 0, 255), 2)
            cv2.putText(display, "NO LINE" if x is None else f"LINE x={x:.0f}",
                        (10, 30), 0, .7, (0, 255, 0), 2)
            cv2.imshow("line search", display)
    finally:
        io.close_views()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--color", choices=("red", "blue", "yellow"), default="red")
    parser.add_argument("--line-color", choices=("white", "black"), default="white")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--actions", action="store_true")
    args = parser.parse_args()
    if sum((args.check, args.simulate, args.preview)) > 1:
        parser.error("check、simulate、preview只能选一种")
    if args.actions and (not args.run or args.check or args.simulate or args.preview):
        parser.error("真实动作只用 --run --actions")
    settings = load_settings(args.config)
    if args.simulate:
        io = Mock()
        io.settings = settings
        run_course(io, args.color)
        for call in io.method_calls:
            print("[模拟]", call)
        return True
    if args.preview:
        if not args.run:
            parser.error("摄像头预览须指定 --run --preview")
        return preview_line(settings, args.line_color)
    if args.check or not args.run:
        return preflight(settings)
    if not args.actions:
        parser.error("真实比赛须同时指定 --run --actions")
    if not preflight(settings):
        return False
    from robotmove import RobotMove
    from robot_audio import configure
    from chinese_speech import warm_up
    configure()
    warm_up()
    deadline = time.monotonic() + settings["total_seconds"]
    io = LineCompetitionIO(settings, None, deadline)
    io.line_color = args.line_color
    with ExitStack() as stack:
        stack.callback(io.close_views)
        stack.callback(cv2.destroyAllWindows)
        io.open_views()
        head, belly, _ = io.observe()
        shapes = dict(head=list(head.shape[:2]), belly=list(belly.shape[:2]))
        carry = json.loads((ROOT / "config/carry_dual_reference.json").read_text())
        head_ref = json.loads((ROOT / "config/head_calibration.json").read_text())
        foot_ref = json.loads((ROOT / "config/right_foot_reference.json").read_text())
        if carry.get("shapes") != shapes or head_ref.get("shapes") != shapes or foot_ref.get("shape") != shapes["belly"]:
            raise ValueError("实际双摄尺寸与标定不同，身体未启动")
        io.check_time()
        robot = RobotMove(None, port=BODY_PORT)
        stack.callback(robot.close)
        io.robot = GuardedRobot(robot, deadline)
        io.flush()
        return run_course(io, args.color)
