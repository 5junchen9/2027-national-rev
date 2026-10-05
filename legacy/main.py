# -*- coding: utf-8 -*-
"""
童星小伴（XingXing）—— 人形机器人竞赛主控制程序
====================================================
依据《童星小伴.txt》五项任务流程编写，按顺序执行：

  任务1  人脸识别任务   —— 识别场地内指定人脸，准确播报对应姓名
  任务2  二维码循迹任务 —— 识别二维码路径指示，沿路径自主行走至目标区
  任务3  物品搬运任务   —— 抓取指定物品，搬运至目标位置并平稳放置
  任务4  体育运动任务   —— 踢球入门（球与球门适配小人形机器人规格）
  任务5  娱乐休闲任务   —— 固定舞蹈动作 + 简单语音交互

说明：所有识别模型均复用文件夹内现有代码——
  人脸识别：b3/2/haar/ 下的 Haar 级联（haarcascade_frontalface_default.xml、
            haarcascade_eye.xml），逻辑同 day3.py 的 __handleFaceRecognition
  颜色识别：findbox.py 的 FindBox（读 Line/img/ 下 *.hsv.json 标定）
  二维码  ：robotbarcode.py 的 RobotBarcode（pyzbar）
  动作/舵机：robotmove.py（串口）、Head.py（GPIO PWM）

运行环境：树莓派（Linux）。
用法：
    python3 main.py             # 依次跑完五项任务
    python3 main.py face        # 只测任务1
    python3 main.py qr          # 只测任务2
    python3 main.py carry       # 只测任务3
    python3 main.py sport       # 只测任务4
    python3 main.py fun         # 只测任务5
"""

import os
import sys
import time

import cv2

# ----------------------------------------------------------------------
# 让本文件能 import 到 b3/2 目录下的既有硬件模块
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "b3", "2")
HAAR_DIR = os.path.join(MODULE_DIR, "haar")   # 现有人脸 Haar 模型目录
if os.path.isdir(MODULE_DIR):
    sys.path.insert(0, MODULE_DIR)

from roboteye import RobotEye          # 摄像头
from robotmove import RobotMove        # 身体动作（串口发单字符指令）
from Head import RobotHeadServoOnly    # 头部俯仰舵机（GPIO PWM）
from robotbarcode import RobotBarcode  # 二维码识别（pyzbar）
from findbox import FindBox            # 颜色识别找目标（HSV + 轮廓）

# TTS 文字转语音（语音交互 / 播报用；无则退化为打印）
try:
    import pyttsx3
    TTS_OK = True
except Exception:
    pyttsx3 = None
    TTS_OK = False

# 语音包目录（树莓派上的预录音频）
VOICE_DIR = "/home/pi/Desktop/语音包/"

# 摄像头画面尺寸（与既有代码一致）
IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_CENTER_X = IMG_WIDTH // 2       # 320


# ======================================================================
# 主控制类
# ======================================================================
class XingXingRobot:
    def __init__(self):
        # 硬件对象
        self.eye = RobotEye()               # 摄像头
        self.head = RobotHeadServoOnly()    # 头部舵机
        self.move = RobotMove(self.eye)     # 身体动作
        self.barcode = RobotBarcode()       # 二维码
        self.findbox = FindBox()            # 颜色找目标

        # 现有人脸 Haar 级联（复用 b3/2/haar/ 下的模型文件）
        self.face_engine = cv2.CascadeClassifier(
            os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(HAAR_DIR, "haarcascade_eye.xml"))

        # TTS
        self.tts = None
        if TTS_OK:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty("rate", 150)
            except Exception:
                self.tts = None

        self.img = None                     # 当前帧

    # ==================== 基础工具 ====================
    def _mv(self, action):
        """身体动作：发串口指令给下位机执行。"""
        self.move.robotMove(action)

    def _head(self, angle):
        """头部舵机转到指定角度（85~180）。"""
        self.head.turn_vertical(angle)

    def _speak_mp3(self, filename):
        """播放预录音频。"""
        os.system("mplayer {}{}".format(VOICE_DIR, filename))

    def _speak_text(self, text):
        """TTS 播报（语音交互 / 播报用）。"""
        print("[语音] " + text)
        if self.tts is not None:
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception:
                pass

    def _get_image(self):
        """取一帧。"""
        ret, self.img = self.eye.getImage()
        return ret

    def _start_camera(self):
        """稳定摄像头：连读若干帧清缓存、降延迟。"""
        for _ in range(10):
            self._get_image()

    # ==================== 任务1：人脸识别 ====================
    def task_face_recognition(self):
        """
        人脸识别：复用现成的 Haar 级联检测人脸 + 眼睛（同 day3.py）。
        人脸尺寸 w>50 连续命中 5 次即判定识别成功并播报。
        """
        print("== 任务1：人脸识别 ==")
        self._head(95)                      # 平视，对准人脸高度
        self._speak_text("开始人脸识别")

        self._mv("BIGRIGHT")                # 原地右转，扫描人脸
        self._mv("BIGRIGHT")

        count = 0
        while True:
            if not self._get_image():
                continue

            faces = self.face_engine.detectMultiScale(self.img, 1.1, 10)
            for (x, y, w, h) in faces:
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_area = self.img[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(face_area, 1.1, 5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh),
                                  (0, 255, 0), 1)
                if w > 50 or h > 50:        # 人脸足够大，计一次命中
                    count += 1

            self.eye.showImage(self.img)
            cv2.waitKey(1)

            if count >= 5:                  # 连续 5 次命中 → 识别成功
                self._speak_mp3("扫码成功开始搬运_1.mp3")
                time.sleep(0.2)
                self._mv("BIGLEFT")
                self._mv("BIGLEFT")
                for _ in range(6):
                    self._mv("UP")
                break

        print("== 任务1结束 ==")

    # ==================== 任务2：二维码循迹 ====================
    def task_qr_navigate(self):
        """识别二维码路径指示，沿路径走到目标区域。"""
        print("== 任务2：二维码循迹 ==")
        self._head(110)                     # 低头看地面二维码

        # 二维码内容 -> 动作序列（可据场地实际修改）
        ACTION_MAP = {
            "face":    ["UP", "UP"],                 # 前进
            "left":    ["UP", "BIGLEFT", "UP"],      # 左转
            "right":   ["UP", "BIGRIGHT", "UP"],     # 右转
            "back":    ["BACK"],                      # 后退
        }

        self._start_camera()
        self._mv("UP")                      # 起步
        self._mv("UP")

        while True:
            if not self._get_image():
                continue
            ret, self.img, barcodes = self.barcode.GetBarcodes(self.img)
            if not ret or not barcodes:
                self._mv("RIGHT1")          # 没扫到就小步右转再找
                continue

            for b in barcodes:
                code = b.data.decode("utf-8")
                print("扫到二维码：", code)
                if code in ("action1", "action2", "goal", "end"):
                    self._speak_mp3("开头.mp3")   # 到达目标区提示
                    print("== 任务2结束：已到达目标区 ==")
                    return code
                for act in ACTION_MAP.get(code, ["UP"]):
                    self._mv(act)

    # ==================== 任务3：物品搬运 ====================
    def _find_color_target(self, color, max_size=270):
        """复用 findbox 在画面中找指定颜色目标块，返回 (centerX, centerY) 或 None。"""
        if self.img is None:
            return None
        ok, _ = self.findbox.findRect(color, self.img)
        if not ok:
            return None
        r = self.findbox.cFindBoxRect
        if 8 < r.width < max_size and 8 < r.height < max_size:
            return r.centerX, r.centerY
        return None

    def task_carry_item(self, color="yellow"):
        """抓取指定颜色物品，搬运到目标位置（扫码定位）并放置。"""
        print("== 任务3：物品搬运（颜色：{}）==".format(color))
        self._head(120)                     # 抬头找远处物品

        # ---- 3.1 走近并抓取 ----
        while True:
            if not self._get_image():
                continue
            pos = self._find_color_target(color)
            if pos is None:
                self._mv("LEFT2")           # 没看到就左转搜索
                continue

            cx, cy = pos
            self.eye.showImage(self.img)
            cv2.waitKey(1)

            if cy > 380:                    # 目标已经足够近
                self._mv("HOLD_BOX")        # 抱 / 抓物品
                break
            if cx < 200:                    # 偏左 -> 右移
                self._mv("RIGHT")
            elif cx > 450:                  # 偏右 -> 左移
                self._mv("LEFT1")
            else:                           # 居中 -> 前进
                self._mv("UP_LITTLE")

        self._speak_mp3("开始搬运.mp3")

        # ---- 3.2 扫码寻找放置点 ----
        self._head(100)                     # 低头找放置点的二维码
        while True:
            if not self._get_image():
                continue
            ret, self.img, barcodes = self.barcode.GetBarcodes(self.img)
            if ret and barcodes:
                self._mv("UP_HOLDBOX")      # 抱箱前进
                self._mv("DOWN_BOX")        # 放下
                break
            self._mv("RIGHT_HOLDBOX")       # 没找到就抱着右转找

        self._speak_mp3("搬运完成准备踢球.mp3")
        print("== 任务3结束 ==")

    # ==================== 任务4：体育运动（踢球入门） ====================
    def task_sport_kick(self, color="blue"):
        """识别球，走近后踢球入门。"""
        print("== 任务4：体育运动（踢球，球色：{}）==".format(color))
        self._head(120)

        while True:
            if not self._get_image():
                continue
            pos = self._find_color_target(color)
            if pos is None:
                self._mv("LEFT2")
                continue

            cx, cy = pos
            self.eye.showImage(self.img)
            cv2.waitKey(1)

            if cy > 380:                    # 球够近 -> 踢球
                if cx <= IMG_CENTER_X:      # 球在左 -> 左脚
                    self._mv("LEFT_BALL")
                else:                       # 球在右 -> 右脚
                    self._mv("RIGHT_BALL")
                break
            if cx < 200:
                self._mv("RIGHT")
            elif cx > 450:
                self._mv("LEFT1")
            else:
                self._mv("UP_LITTLE")

        self._speak_text("射门完成")
        print("== 任务4结束 ==")

    # ==================== 任务5：娱乐休闲（舞蹈 + 语音交互） ====================
    def task_dance_and_interact(self):
        """固定舞蹈动作展示 + 简单语音交互。"""
        print("== 任务5：娱乐休闲 ==")
        self._speak_text("下面为大家表演一段舞蹈")
        self._mv("DANCE")                   # 串口触发舞蹈动作
        self._speak_mp3("跳舞bgm.mp3")      # 播放 BGM

        # 简单语音交互：本机无语音识别输入，用固定话术演示；
        # 若需真正语音交互，需另加麦克风 + 语音识别方案
        time.sleep(2)
        self._speak_text("大家好，我是童星小伴，很高兴认识你们")
        time.sleep(2)
        self._speak_text("表演结束，谢谢观看")
        print("== 任务5结束 ==")

    # ==================== 主流程 ====================
    def run(self):
        """按竞赛顺序依次执行五项任务。"""
        print("=" * 40)
        print(" 童星小伴 竞赛流程启动")
        print("=" * 40)
        self._start_camera()

        self.task_face_recognition()   # 1. 人脸识别
        self.task_qr_navigate()        # 2. 二维码循迹
        self.task_carry_item()         # 3. 物品搬运
        self.task_sport_kick()         # 4. 体育运动（踢球）
        self.task_dance_and_interact() # 5. 娱乐休闲

        print("全部任务完成！")


# ======================================================================
# 入口
# ======================================================================
if __name__ == "__main__":
    robot = XingXingRobot()

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "face":
        robot.task_face_recognition()
    elif arg == "qr":
        robot.task_qr_navigate()
    elif arg == "carry":
        robot.task_carry_item()
    elif arg == "sport":
        robot.task_sport_kick()
    elif arg == "fun":
        robot.task_dance_and_interact()
    else:
        robot.run()
