import time
import os
import cv2
import numpy
import datetime

import RobotEnum
from roboteye import RobotEye
from robotmove import RobotMove
from robotbarcode import RobotBarcode
from basedata import BaseData
from myangle import MyAngle
from holdbox import RobotHoldBox
from ball import RobotHoldBox1
from findrectangle import FindRectangle
from touying import TouYing
#from cv2 import resize,INTER_CUBIC,waitKey,imshow,cvtColor,COLOR_BGR2GRAY,rectangle,destroyAllWindows,CascadeClassifier,VideoCapture,namedWindow,WINDOW_NORMAL,WINDOW_KEEPRATIO,WINDOW_GUI_EXPANDED
import numpy as np

from Head import RobotHeadServoOnly

class RobotState:
    def __init__(self):
        self.current_mode = 0
        self.face_detection_count = 0
        self.face_dimensions = {'width': 0, 'height': 0}
        self.process_state = RobotEnum.BoxProcessState.NONE
        self.head_angle = 90
        self.current_image = None
        self.not_find_count = 0

class Day:
    def __init__(self):
        # Initialize robot components
        self.__robotEye = RobotEye()
        self.__robotMove = RobotMove(self.__robotEye)
        self.__robotHead = RobotHeadServoOnly()
        self.__holdBox = RobotHoldBox(self.__robotEye, self.__robotHead, self.__robotMove)
        self.__ball2 = RobotHoldBox1(self.__robotEye, self.__robotHead, self.__robotMove)
        self.__robotBarcode = RobotBarcode()
        self.__findRectangle = FindRectangle()
        self.__touYing = TouYing()

        # Initialize state management
        self.state = RobotState()

        # Initialize face detection components
        self.face_engine = cv2.CascadeClassifier('/home/pi/Desktop/b3/2/haar/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('/home/pi/Desktop/b3/2/haar/haarcascade_eye.xml')

        # Initialize tracking variables
        self.mode = 0
        self.ww = 0
        self.hh = 0

    def __getImage(self):
        """获取图像"""
        ret, self.state.current_image = self.__robotEye.getImage()
        if not ret:
            self.state.current_image = None
        return ret
    def __showImage(self):
        """显示图像"""
        if self.state.current_image is None:
            print("********* error in {0}.{1}()".format(self.__class__, self.__class__.__name__))
        else:
            self.__robotEye.showImage(self.state.current_image)


    # 移动控制
    def __move(self, action):
        """移动控制"""
        self.__robotMove.robotMove(action)

    # 通过摄像头获取条码
    def __findBarCode(self):
        """查找二维码"""
        if self.__getImage():
            self.__showImage()

        loopCount = 0
        while loopCount < BaseData.Barcode.findBarcodeLoopCount:
            loopCount += 1
            if self.__getImage():
                ret2, self.state.current_image, barcodes = self.__robotBarcode.GetBarcodes(self.state.current_image)
                if ret2:
                    self.__showImage()
                    return True, barcodes
        print("barcode Not found")
        return False, None

    # 获取条码中心点和条码角度
    def __getAngleError(self, x, y):
        """计算角度误差"""
        myAngle = MyAngle()
        if abs(BaseData.Image.half_Width - x) < 2:
            return 0
        __angle = myAngle.get_angle(BaseData.Image.half_Width, BaseData.Image.height, round(x), round(y))
        return 90 - __angle if __angle > 0 else -(90 + __angle)

    # 获取条码中心点和条码角度
    def __getBarCodeCenter(self):
        """获取二维码中心点和角度"""
        barcodeX = self.__robotBarcode.cRobotBarcodeRect.left + int(self.__robotBarcode.cRobotBarcodeRect.width / 2)
        barcodeY = self.__robotBarcode.cRobotBarcodeRect.top + int(self.__robotBarcode.cRobotBarcodeRect.height / 2)
        angle = round(self.__getAngleError(barcodeX, barcodeY))
        print(f"码中心点X:={barcodeX}, 码中心点Y:={barcodeY}, 角度:={angle}, 头角度:={self.state.head_angle}")
        return barcodeX, barcodeY, angle

    # 划线
    def __showLine(self, x, y):
        """显示指向线"""
        if self.state.current_image is not None:
            cv2.line(self.state.current_image, (BaseData.Image.half_Width, BaseData.Image.height), (x, y), (0, 255, 0), 1, 4)
        self.__showImage()

    # 走到条码处
    def __moveBasic(self, action):
        """基础移动控制"""
        barcodeX, barcodeY, angle = self.__getBarCodeCenter()
        self.__showLine(barcodeX, barcodeY)

        if barcodeY < BaseData.Image.half_Height:
            if abs(angle) > 25:
                self.__move("RIGHT1" if angle > 25 else "LEFT")
            else:
                self.__move("UP")
        else:
            if abs(angle) > 30:
                self.__move("RIGHT1" if angle > 30 else "LEFT")
            elif barcodeY < 330:
                self.__move("UP")
            else:
                return action
        return ""
    
    # 走到前进条码(UP)
    def __moveUp(self):
        """向前移动"""
        self.state.process_state = RobotEnum.BoxProcessState.MOVE_UP
        if self.__moveBasic("UP") == "UP":
            self.__move("UP")
            self.__move("UP")

    def __inTurnFindBarcode(self):
        action = ""
        ret, barcodes = self.__findBarCode()
        if ret:
            if barcodes[0] == "face":
                action = "UP"
            if barcodes[0] == "left":
                action = "LEFT"
            if barcodes[0] == "action1":
                action = "ACTION1"

        return action

    # 走到右转条码(目前条码是left)
    def __moveRight(self):
        """向右移动"""
        if self.__moveBasic("LEFT") == "LEFT":
            self.__move("UP")
            self.__move("UP")
            self.__move("RIGHT1")
            self.__move("UP")
            self.__move("UP")

    # 走到动作1(ACTION1)条码处
    def __moveAction1(self):
        """移动到动作1位置"""
        self.__move("UP")
        self.state.process_state = RobotEnum.BoxProcessState.FIND_BOX

    # 稳定摄像头
    def __startCamera(self):
        """初始化摄像头"""
        for _ in range(20):
            if self.__getImage():
                self.__showImage()

    def __findRectMove(self):
        if self.__getImage():

            imgP = self.__touYing.getTouYingImg(self.state.current_image)

            ret, squares, imgP, cx, cy = self.__findRectangle.getNearRectangle(imgP)

            if ret:
                x, y = self.__touYing.getXY(cx, cy)
                self.__showLine(int(x), int(y))

                angle = round(self.__getAngleError(x, y))

                if abs(angle) > 25:
                    if angle > 25:
                        self.__move("RIGHT1")
                    elif angle < -25:
                        self.__move("LEFT")
                else:
                    self.__move("UP")
                return "RECTMOVE"
            else:
                return ""

        else:
            return ""
    #
    # 主流程
    def run(self):
        """主运行流程"""
        self.__startCamera()
        self.__robotHead.turn_vertical(110)
        self.__currentHeadVerticalAngle = BaseData.Barcode.HeadAnger.near
        #self.__robotHead.turn_vertical(self.__currentHeadVerticalAngle)
        #self.__robotHead.turn_vertical(self.__currentHeadVerticalAngle)
        
        # 初始移动
        for _ in range(2):
            self.__move("UP")

        while True:
            action = ""
            modeflag = False
            ret, barcodes = self.__findBarCode()
            
            if ret:
                for barcode in barcodes:
                    barcode_data = barcode.data.decode("utf-8")
                    
                    # 处理不同模式的二维码
                    print(str(self.state.current_mode) + "barcode mode")  #add
                    
                    if self.state.current_mode == 0 and barcode_data == "face":
                        self.state.current_mode = 1
                        if self.__handleFaceRecognition():
                            self.__findNextBarcode("sber")
                            
                    elif self.state.current_mode == 1 and barcode_data == "sber":
                        self.state.current_mode = 1
                        barcodeY = self.__robotBarcode.cRobotBarcodeRect.top + int(self.__robotBarcode.cRobotBarcodeRect.height / 2)
                        #if barcodeY > 250:
                        print("sber")
                        os.system('mplayer /home/pi/Desktop/语音包/开头.mp3')
                        for _ in range(4):
                            self.__move("UP")
                                
                    elif self.state.current_mode == 1 and barcode_data == "left":
                        #barcodeY = self.__robotBarcode.cRobotBarcodeRect.top + int(self.__robotBarcode.cRobotBarcodeRect.height / 2)
                        #if barcodeY > 250:
                            self.state.current_mode = 1#san er wei ma
                            self.__move("UP")
                            self.__move("UP")
                            self.__move("BIGRIGHT")
                            self.__move("BIGRIGHT")
                            self.__move("UP")
                            self.__move("UP")
                            self.__move("BIGRIGHT")
                            #for _ in range(2):
                                #self.__move("BIGRIGHT")
                            #for _ in range(3):
                                #self.__move("UP")
                                
                    elif self.state.current_mode == 3 and barcode_data == "action1":
                        os.system('mplayer /home/pi/Desktop/语音包/准备搬运.mp3')
                        self.__moveAction1()
                        barcodeY = self.__robotBarcode.cRobotBarcodeRect.top + int(self.__robotBarcode.cRobotBarcodeRect.height / 2)
                        if barcodeY > 250:
                            for _ in range(3):
                                self.__move("BIGRIGHT")
                            print("3")
            else:
                # 处理未找到二维码的情况
                if self.state.not_find_count >= 2:
                    #self.__move("RIGHT1")
                    print("not found")
                    self.state.not_find_count = 0
                else:
                    self.state.not_find_count += 1

            # 处理箱子状态
            if self.state.process_state == RobotEnum.BoxProcessState.FIND_BOX:
                if self.__holdBox.Run():
                    self.state.process_state = RobotEnum.BoxProcessState.NONE
                    print("方箱子到条码处流程结束!")
                    self.__robotHead.turn_vertical(95)
                    self.__currentHeadVerticalAngle = BaseData.Barcode.HeadAnger.near
                    self.__robotHead.turn_vertical(self.__currentHeadVerticalAngle)
                    self.__robotHead.turn_vertical(self.__currentHeadVerticalAngle)
                    continue

    def __handleFaceRecognition(self):
        """处理人脸识别"""
        i = 0
        self.__move("BIGRIGHT")
        self.__move("BIGRIGHT")
        
        while True:
            if self.__getImage():
                faces = self.face_engine.detectMultiScale(self.state.current_image, 1.1, 10)
                for (x, y, w, h) in faces:
                    self.state.current_image = cv2.rectangle(self.state.current_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_area = self.state.current_image[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(face_area, 1.1, 5)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(face_area, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
                    if w > 50 or h > 50:
                        i += 1
                self.__showImage()

                if i == 5:
                    os.system('mplayer /home/pi/Desktop/语音包/扫码成功开始搬运_1.mp3')
                    time.sleep(0.2)
                    self.__move("BIGLEFT")
                    self.__move("BIGLEFT")
                    #self.__move("BIGLEFT")
                    self.__move("UP")
                    self.__move("UP")
                    self.__move("UP")
                    #self.__move("BIGRIGHT")
                    self.__move("UP")
                    self.__move("UP")
                    self.__move("UP")
                    return True
        return False

    def __findNextBarcode(self, target_barcode):
        """寻找下一个二维码"""
        while True:
            ret, barcodes = self.__findBarCode()
            if ret:
                for barcode in barcodes:
                    if barcode.data.decode("utf-8") == target_barcode:
                        barcodeY = self.__robotBarcode.cRobotBarcodeRect.top + int(self.__robotBarcode.cRobotBarcodeRect.height / 2)
                        if barcodeY > 250:
                            return True
            else:
                action = self.__moveBasic(target_barcode)
                if action == "":
                    self.__move("RIGHT1")
                    #self.__move("UP")
                    #self.__move("BIGRIGHT")
                    #self.__move("BIGRIGHT")
                    #self.__move("UP")
                    #self.__move("UP")
                    
        return False

if __name__ == "__main__":
    day = Day()
    day.run()
