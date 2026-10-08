# -*- coding: utf-8 -*-
"""固定体育动作：直接执行左脚踢球。"""

import argparse
import os
import sys


from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
MODULE_DIR = str(MODULE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

PLUGIN_NAME = "kick_ball"
VOICE_COMMANDS = ("踢球", "开始踢球")
KICK_ACTION = "LEFT_BALL"


def run(config=None, dry_run=False, robot=None):
    mode = config or os.environ.get("ROBOT_KICK_MODE", "fixed")
    if mode not in ("fixed", "dual"):
        raise ValueError("ROBOT_KICK_MODE must be fixed or dual")
    if mode == "dual":
        if dry_run:
            print("Dual plan: head ball/goal -> belly handover -> align -> one RIGHT_BALL. No hardware opened.")
            return True
        from dual_kick import run as run_dual
        return run_dual(robot=robot)
    del config
    if dry_run:
        print("[模拟动作]", KICK_ACTION)
        return True

    move = robot
    owns_move = False
    if move is None:
        from robotmove import RobotMove
        move = RobotMove(None)
        owns_move = True

    from Head import RobotHeadServoOnly
    head = RobotHeadServoOnly()
    try:
        print("== 固定体育动作：左脚踢球 ==")
        head.look_down()
        move.robotMove(KICK_ACTION)
        return True
    finally:
        head.level()
        head.cleanup()
        if owns_move:
            move.close()


def _check():
    assert KICK_ACTION == "LEFT_BALL"
    assert os.environ.get("ROBOT_KICK_MODE", "fixed") in ("fixed", "dual")
    print("踢球配置自检通过；fixed保留左脚，dual使用双摄右脚")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查固定动作")
    parser.add_argument("--run", action="store_true", help="允许发送真实机器人动作")
    parser.add_argument("--mode", choices=("fixed", "dual"), default=os.environ.get("ROBOT_KICK_MODE", "fixed"),
                        help="fixed preserves the old left kick; dual uses two-camera right kick")
    parser.add_argument("--preview", action="store_true", help="dual-camera preview and calibration; no motion")
    parser.add_argument("--calibration", help="dual-camera calibration JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check:
        _check()
        raise SystemExit(0)
    if args.mode == "dual":
        if args.preview or args.run:
            from dual_kick import run as run_dual
            success = run_dual(preview=args.preview, calibration_path=args.calibration)
        else:
            success = run(config="dual", dry_run=True)
    else:
        if args.preview or args.calibration:
            raise SystemExit("--preview/--calibration require --mode dual")
        success = run(config="fixed", dry_run=not args.run)
    raise SystemExit(0 if success else 1)
