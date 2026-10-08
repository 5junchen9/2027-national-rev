"""独立寻线比赛入口：不覆盖competition.py。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
from competition_line_main import main

if __name__ == "__main__":
    try:
        success = main()
    except KeyboardInterrupt:
        print("寻线比赛停止；已发送的动作不能撤回。")
        success = False
    except Exception as error:
        print("寻线比赛停止：", error, file=sys.stderr)
        success = False
    raise SystemExit(0 if success else 1)
