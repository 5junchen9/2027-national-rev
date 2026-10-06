"""独立比赛入口，保留原 main.py 的所有任务。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
from competition_main import main

if __name__ == "__main__":
    try:
        success = main()
    except KeyboardInterrupt:
        print("比赛停止；已发送的动作不能由Ctrl+C撤回。")
        success = False
    except Exception as error:
        print("比赛停止：", error, file=sys.stderr)
        success = False
    raise SystemExit(0 if success else 1)
