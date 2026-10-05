# -*- coding: utf-8 -*-
"""表情识别任务：识别开心/伤心，未看清时最多重试一次。"""

import argparse
import os
import stat
import subprocess
import sys
import time
from robot_audio import play

from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
MODULE_DIR = str(MODULE_DIR)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

HAAR_PATH = str(ROOT / "assets/haar/haarcascade_frontalface_default.xml")
EIM_PATH = os.path.join(BASE_DIR, "assets", "models", "emotion", "emotion.eim")
VOICE_DIR = str(VOICE_DIR)

EMOTION_CONFIDENCE = float(os.environ.get("ROBOT_EMOTION_CONFIDENCE", "0.52"))
SAD_CONFIDENCE = float(os.environ.get("ROBOT_SAD_CONFIDENCE", "0.95"))
FACE_MARGIN = 0.15
CONFIRM_FRAMES = 3
ATTEMPT_TIMEOUT_SECONDS = 6
MAX_ATTEMPTS = 2

EMOTION_AUDIO = {
    "happy": "情绪_开心.mp3",
    "sad": "情绪_伤心.mp3",
    "unknown": "情绪_未看清.mp3",
    "failed": "情绪_识别失败.mp3",
}


def normalize_emotion(label):
    """把模型标签收敛为业务使用的三种结果。"""
    normalized = str(label).strip().lower()
    if normalized in ("happy", "sad"):
        return normalized
    return "unknown"


def decide_emotion(label, score):
    """按模型置信度判断单帧情绪。"""
    emotion = normalize_emotion(label)
    if emotion == "happy" and score >= EMOTION_CONFIDENCE:
        return "happy"
    if emotion == "sad" and score >= SAD_CONFIDENCE:
        return "sad"
    return "unknown"


def square_face_box(image_shape, face):
    """给人脸四周留出余量，并生成不越界的正方形裁剪框。"""
    image_height, image_width = image_shape[:2]
    x, y, width, height = (int(value) for value in face)
    size = min(
        max(1, int(round(max(width, height) * (1 + 2 * FACE_MARGIN)))),
        image_width,
        image_height,
    )
    left = int(round(x + width / 2 - size / 2))
    top = int(round(y + height / 2 - size / 2))
    left = max(0, min(left, image_width - size))
    top = max(0, min(top, image_height - size))
    return left, top, size


def play_prompt(name):
    audio_path = os.path.join(VOICE_DIR, EMOTION_AUDIO[name])
    if not os.path.isfile(audio_path):
        print("[情绪] 缺少提示音：{}".format(audio_path))
        return False
    try:
        play(audio_path)
    except (OSError, subprocess.SubprocessError) as error:
        print("[情绪] 无法播放提示音：{}".format(error))
        return False
    return True


class EmotionRecognitionPlugin:
    def __init__(self):
        import cv2
        from face_eim import FaceClassifier
        from roboteye import RobotEye

        if not os.path.isfile(EIM_PATH):
            raise FileNotFoundError("缺少情绪模型：{}".format(EIM_PATH))
        if os.name == "posix" and not os.access(EIM_PATH, os.X_OK):
            os.chmod(EIM_PATH, os.stat(EIM_PATH).st_mode | stat.S_IXUSR)

        self.classifier = None
        self.eye = None
        self.cv2 = cv2
        try:
            self.classifier = FaceClassifier(model_path=EIM_PATH)
            self.eye = RobotEye()
            self.face_detector = self.cv2.CascadeClassifier(HAAR_PATH)
            if self.face_detector.empty():
                raise RuntimeError("无法加载人脸检测文件：{}".format(HAAR_PATH))
        except Exception:
            self.close()
            raise

    def recognize_once(self):
        """在一次观察窗口内返回 happy、sad 或 unknown。"""
        candidate = None
        count = 0
        deadline = time.monotonic() + ATTEMPT_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            ok, image = self.eye.getImage()
            if not ok:
                candidate, count = None, 0
                continue
            gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=5, minSize=(60, 60)
            )
            if not len(faces):
                candidate, count = None, 0
                continue

            detected_face = max(faces, key=lambda face: face[2] * face[3])
            left, top, size = square_face_box(image.shape, detected_face)
            face = image[top:top + size, left:left + size]
            label, score = self.classifier.classify(face)
            emotion = decide_emotion(label, score)
            if emotion == "unknown":
                candidate, count = None, 0
            elif emotion == candidate:
                count += 1
            else:
                candidate, count = emotion, 1
            print("[情绪] 模型={} {:.1%} -> {}，连续={}/{}".format(
                label, score, emotion, count, CONFIRM_FRAMES
            ))
            if count >= CONFIRM_FRAMES:
                return candidate

        return "unknown"

    def close(self):
        if self.classifier is not None:
            self.classifier.close()
            self.classifier = None
        if self.eye is not None:
            self.eye.close()
            self.eye = None


def run(robot=None):
    """执行一次情绪交互；unknown 时只额外重试一次。"""
    plugin = EmotionRecognitionPlugin()
    emotion = "unknown"
    try:
        for attempt in range(MAX_ATTEMPTS):
            emotion = plugin.recognize_once()
            if emotion != "unknown":
                break
            if attempt + 1 < MAX_ATTEMPTS:
                print("[情绪] 我没有看清，你可以正对着我再试一次。")
                play_prompt("unknown")
                plugin.eye.discard_frames()
    finally:
        plugin.close()

    if emotion == "happy":
        print("[情绪] 你现在看起来很开心呀！可以跟我分享一下有什么开心事吗？")
        play_prompt("happy")
        return True
    if emotion == "sad":
        print("[情绪] 你看起来有点难过，那我给你跳支舞蹈吧。")
        play_prompt("sad")
        import dance_main
        return dance_main.run(robot=robot)

    print("[情绪] 重试后仍未看清，本次识别结束")
    play_prompt("failed")
    return False


def self_check():
    assert normalize_emotion("happy") == "happy"
    assert normalize_emotion(" SAD ") == "sad"
    assert normalize_emotion("none") == "unknown"
    assert normalize_emotion("unknown") == "unknown"
    assert decide_emotion("happy", EMOTION_CONFIDENCE) == "happy"
    assert decide_emotion("happy", EMOTION_CONFIDENCE - 0.01) == "unknown"
    assert decide_emotion("sad", SAD_CONFIDENCE) == "sad"
    assert square_face_box((480, 640, 3), (100, 100, 100, 80)) == (85, 75, 130)
    assert square_face_box((480, 640, 3), (0, 0, 100, 100)) == (0, 0, 130)
    assert CONFIRM_FRAMES == 3
    assert MAX_ATTEMPTS == 2
    assert os.path.isfile(EIM_PATH)
    assert os.path.isfile(HAAR_PATH)
    assert all(os.path.isfile(os.path.join(VOICE_DIR, filename))
               for filename in EMOTION_AUDIO.values())
    print("情绪识别配置自检通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查配置，不打开硬件")
    args = parser.parse_args()
    if args.check:
        self_check()
        return True
    return run()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
