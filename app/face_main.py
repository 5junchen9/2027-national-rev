"""独立及语音人脸入口：复用当前姓名OCR、性别识别和播报。"""
import argparse

from competition_identity import recognize
from competition_main import CONFIG_FILE, legacy_root, load_settings

PLUGIN_NAME = "face_recognition"
VOICE_COMMANDS = ("人脸识别", "识别人脸")


def run(quick=False):
    # 保留语音总控已有的quick调用；当前流程只播报姓名和性别。
    settings = load_settings(CONFIG_FILE)
    return recognize(legacy_root(settings), settings["face_head_position"], 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="face", choices=("face",))
    parser.add_argument("--quick", action="store_true", help="保留语音入口参数，不播放开场音频")
    args = parser.parse_args()
    return run(quick=args.quick)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
