# -*- coding: utf-8 -*-
"""人脸识别任务插件，可独立运行，也可由后续语音总控调用。"""

import argparse
import os
import stat
import sys
import time

import cv2
from chinese_speech import speak_chinese
from robot_audio import play

from robot_config import ROOT, MODULE_DIR, VOICE_DIR
BASE_DIR = str(ROOT)
MODULE_DIR = str(MODULE_DIR)
if not os.path.isdir(MODULE_DIR):
    MODULE_DIR = os.path.join(BASE_DIR, "modules")
if os.path.isdir(MODULE_DIR):
    sys.path.insert(0, MODULE_DIR)

HAAR_PATH = str(ROOT / "assets/haar/haarcascade_frontalface_default.xml")
EIM_PATH = os.path.join(BASE_DIR, "assets", "models", "face", "face.eim")
VOICE_DIR = str(VOICE_DIR)
PLUGIN_NAME = "face_recognition"
VOICE_COMMANDS = ("人脸识别", "识别人脸")

FACE_NAMES = {
    "haowenwan": "郝文菀",
    "wujuncheng": "吴俊成",
    "xieyuting": "谢雨婷",
    "yangchengrui": "杨承睿",
    "zhaobowen": "赵博闻",
}
FACE_AUDIO = {
    label: "人脸_{}.mp3".format(label)
    for label in FACE_NAMES
}
FACE_CONFIDENCE = float(os.environ.get("ROBOT_FACE_CONFIDENCE", "0.65"))
FACE_CONFIRM_FRAMES = int(os.environ.get("ROBOT_FACE_CONFIRM_FRAMES", "2"))
if not 0 <= FACE_CONFIDENCE <= 1 or FACE_CONFIRM_FRAMES < 1:
    raise ValueError("Face confidence must be 0..1; confirmation frames must be positive")
FACE_TIMEOUT_SECONDS = 60
class FaceRecognitionPlugin:
    def __init__(self):
        # 只在运行任务时加载硬件和模型；import face_main 不占用资源。
        from Head import RobotHeadServoOnly
        from face_eim import FaceClassifier
        from roboteye import RobotEye

        if os.name == "posix" and not os.access(EIM_PATH, os.X_OK):
            os.chmod(EIM_PATH, os.stat(EIM_PATH).st_mode | stat.S_IXUSR)
        self.eye = self.head = self.face_classifier = None
        try:
            self.eye = RobotEye()
            self.head = RobotHeadServoOnly()
            self.face_engine = cv2.CascadeClassifier(HAAR_PATH)
            if self.face_engine.empty():
                raise RuntimeError("Cannot load face detector: " + HAAR_PATH)
            self.face_classifier = FaceClassifier()
        except BaseException:
            self.close()
            raise

    def _welcome(self, label):
        text = "{}同学，欢迎你".format(FACE_NAMES[label])
        print("[语音] " + text)
        audio_path = os.path.join(VOICE_DIR, FACE_AUDIO[label])
        return speak_chinese(text, audio_path)

    def run(self, quick=False):
        print("== 人脸识别任务 ==")
        self.head.level()
        if not quick:
            play(os.path.join(VOICE_DIR, "开头.mp3"))

        candidate = None
        count = 0
        deadline = time.monotonic() + FACE_TIMEOUT_SECONDS

        while time.monotonic() < deadline:
            ok, image = self.eye.getImage()
            if not ok:
                candidate, count = None, 0
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_engine.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces):
                x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
                label, score = self.face_classifier.classify(
                    image[y:y + height, x:x + width]
                )
                accepted = label in FACE_NAMES and score >= FACE_CONFIDENCE
                count = count + 1 if accepted and label == candidate else 1 if accepted else 0
                candidate = label if accepted else None
                cv2.rectangle(
                    image, (x, y), (x + width, y + height),
                    (0, 255, 0) if accepted else (0, 0, 255), 2,
                )
                cv2.putText(
                    image, "{} {:.0%}".format(label, score), (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            else:
                candidate, count = None, 0

            self.eye.showImage(image)
            cv2.waitKey(1)
            if count >= FACE_CONFIRM_FRAMES:
                return self._welcome(candidate)

        print("[语音] 未识别到指定人脸，任务停止")
        return False

    def close(self):
        # 每项都尝试释放，避免一个关闭错误留下摄像头或模型进程。
        error = None
        callbacks = []
        if self.head is not None:
            callbacks.append(self.head.level)
        if self.face_classifier is not None:
            callbacks.append(self.face_classifier.close)
        if self.eye is not None:
            callbacks.extend((self.eye.close, cv2.destroyAllWindows))
        if self.head is not None:
            callbacks.append(self.head.cleanup)
        for callback in callbacks:
            try:
                callback()
            except Exception as failure:
                error = failure
        if error:
            raise error


def run(quick=False):
    """插件入口，成功返回 True，失败返回 False。"""
    plugin = FaceRecognitionPlugin()
    try:
        return plugin.run(quick=quick)
    finally:
        plugin.close()


def main():
    parser = argparse.ArgumentParser(description="人脸识别任务插件")
    parser.add_argument("task", nargs="?", default="face", choices=("face",))
    parser.add_argument("--quick", action="store_true", help="跳过开场音频")
    args = parser.parse_args()
    return run(quick=args.quick)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
