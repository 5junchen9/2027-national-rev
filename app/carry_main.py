# -*- coding: utf-8 -*-
"""固定物品搬运：直接抱起、前进并放下。"""

import argparse
import os
import sys
from dataclasses import dataclass


from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
MODULE_DIR = str(MODULE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

PLUGIN_NAME = "carry_items"
VOICE_COMMANDS = ("物品搬运", "开始搬运")


@dataclass
class CarryConfig:
    carry_steps: int = 3


def action_sequence(config):
    return (("HOLD_BOX",) +
            ("UP_HOLDBOX",) * max(0, config.carry_steps) +
            ("DOWN_BOX",))


def run(config=None, dry_run=False, robot=None):
    config = config or CarryConfig()
    actions = action_sequence(config)
    if dry_run:
        for action in actions:
            print("[模拟动作]", action)
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
        print("== 固定搬运：抱起后前进 {} 步并放下 ==".format(
            config.carry_steps))
        head.look_down()
        for action in actions:
            move.robotMove(action)
        return True
    finally:
        head.level()
        head.cleanup()
        if owns_move:
            move.close()


def _check():
    assert action_sequence(CarryConfig(0)) == ("HOLD_BOX", "DOWN_BOX")
    assert action_sequence(CarryConfig(2)) == (
        "HOLD_BOX", "UP_HOLDBOX", "UP_HOLDBOX", "DOWN_BOX"
    )
    print("固定搬运动作顺序自检通过")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('fixed','visual'), default='fixed')
    parser.add_argument('--color', choices=('red','green','blue','yellow','pink'))
    parser.add_argument('--target-qr', help='目的地二维码的完整内容')
    parser.add_argument('--preview', action='store_true', help='视觉预览和位置标定，不发送身体动作')
    parser.add_argument('--actions', action='store_true', help='视觉模式允许真实搬运动作')
    parser.add_argument("--check", action="store_true", help="只检查动作顺序")
    parser.add_argument("--run", action="store_true", help="允许发送真实机器人动作")
    parser.add_argument("--steps", type=int, default=3, help="抱起后前进次数，默认3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check:
        _check()
        raise SystemExit(0)
    if args.mode == 'visual':
        if not args.run:
            raise SystemExit('视觉模式使用摄像头和头部舵机，请加 --run')
        if args.preview and args.actions:
            raise SystemExit('--preview 和 --actions 不能同时使用')
        from carry_vision import choose_task, run as run_visual
        color,target_qr = choose_task(args.color,args.target_qr)
        from competition_qr import read_codes
        success = run_visual(color,target_qr,actions=args.actions,qr_reader=read_codes)
    else:
        success = run(CarryConfig(max(0, args.steps)), dry_run=not args.run)
    raise SystemExit(0 if success else 1)
