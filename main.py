"""统一入口；默认自检，实机任务需 --run。"""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
import robot_config

TASK_MODULES = {
    "voice": "voice_main", "face": "face_main", "qr": "qr_main",
    "carry": "carry_main", "kick": "kick_main", "sport": "kick_main",
    "dance": "dance_main", "fun": "dance_main", "emotion": "emotion_main",
    "speech": "chinese_speech", "serial": "serial_monitor",
    "protocol": "dreammaker_protocol", "action": "robotmove",
    "camera": "roboteye", "round": "round_object",
    "mic": "voice_control", "calibrate": "color_calibration", "vision": "vision_check",
    "head": "head_debug",
    "ball": "ball_debug",
    "handover": "handover_debug",
}


def run_sequence(keys):
    import voice_main
    from robotmove import RobotMove
    robot = RobotMove(None)
    try:
        if not voice_main.configure_audio():
            return False
        for key in keys:
            if not voice_main.run_task(key, robot):
                return False
        return True
    finally:
        robot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="check",
                        choices=tuple(TASK_MODULES) + ("check", "simulate", "all", "song", "joke"))
    parser.add_argument("--run", action="store_true", help="run a real hardware task")
    parser.add_argument("--check", action="store_true", help="offline checks only")
    args, remaining = parser.parse_known_args()
    if args.check or args.task == "check":
        from check_environment import check
        return check()
    if args.task == "simulate":
        from simulate import run
        return run()
    if args.task == "qr" and not args.run:
        if not any(option in remaining for option in ("--source", "--image")):
            parser.error("QR preview requires --source or --image; real camera use requires --run")
        remaining.append("--dry-run")
    elif args.task in ("carry", "kick", "sport"):
        if args.run:
            remaining.append("--run")
    elif args.task == "protocol":
        remaining = remaining or ["check"]
        if remaining[0] != "check" and not args.run:
            parser.error("Protocol hardware commands require --run")
        if args.run and any(command in remaining for command in ("play", "known", "action")):
            remaining.append("--run")
    elif args.task == "mic" and "--list-devices" in remaining:
        pass
    elif args.task not in ("speech", "serial", "vision") and not args.run:
        parser.error("Add --run to use real hardware, or use simulate for an offline rehearsal")
    if args.task in ("all", "song", "joke"):
        if remaining:
            parser.error("Unexpected arguments: " + " ".join(remaining))
        if args.task == "all":
            return run_sequence(("1", "2", "3", "4", "5"))
        import voice_main
        return voice_main.run_task("14" if args.task == "song" else "15", None)
    sys.argv = [TASK_MODULES[args.task] + ".py", *remaining]
    runpy.run_module(TASK_MODULES[args.task], run_name="__main__")
    return True


if __name__ == "__main__":
    try:
        success = main()
    except KeyboardInterrupt:
        print("Stopped.")
        success = False
    except Exception as error:
        print(f"Failed: {error}", file=sys.stderr)
        success = False
    raise SystemExit(0 if success else 1)
