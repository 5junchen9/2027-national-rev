import os

import cv2
import numpy
import time
import sys
import datetime

import RobotEnum
from robotmove import RobotMove

from Head import RobotHeadServoOnly
from findball import FindBox1
from robotbarcode import RobotBarcode
from basedata import BaseData
from myangle import MyAngle

from roboteye import RobotEye


class RobotHoldBox1:
    __camera = None
    boxProcessState = "TAKE_BOX"
    __currentHeadVerticalAngle = 40  # 当前抬头角度

    __robotEye: RobotEye
    # 机器人头部舵机控制
    __robotHead: RobotHeadServoOnly
    __robotMove: RobotMove

    def __init__(self, robotEye: RobotEye, robotHead: RobotHeadServoOnly, robotMove: RobotMove):
        self.__robotEye = robotEye
        self.__robotHead = robotHead
        self.__robotMove = robotMove

    cListFindBoxRect = []
    listBoxCenter = []

    ball_color = 'ball'

    # 在图片中查找盒子对象
    cFindBox = FindBox1()
    # 查找条码对象
    cRobotBarcode = RobotBarcode()

    __currentImg = None  # 当前图像

    def exit1(self):
        print("in RobotBox.exit()")
        sys.exit()

    def __getImage(self):
        ret, self.__currentImg = self.__robotEye.getImage()
        if not ret:
            self.__currentImg = None
        return ret

    def __showImage(self):
        if self.__currentImg is None:
            print("********* error in {0}.{1}()".format(self.__class__, self.__class__.__name__))
        else:
            self.__robotEye.showImage(self.__currentImg)

    def __move(self, action):
        self.__robotMove.robotMove(action)

    def move(self, ccDistance, ccCenterX, ccAngle):
        centerX = ccCenterX
        distance = ccDistance
        print("distance:={0}.".format(distance))
        print("ccAngle:={0}.".format(ccAngle))
        if 35 < ccAngle < 75:
            print("斜")
            if BaseData.FindBox.nearRightLimit <= centerX < BaseData.FindBox.nearLeftLimit:
                if distance >= 400:
                    self.boxProcessState = RobotEnum.BoxProcessState.UP_BOX
                    self.__move("RIGHT_BALL")
                    #self.__move("HOLD_BOX")
                    #self.__timeSleep(1.3)
                    #self.__move("RIGHT_HOLDBOX")
                else:
                    self.__move("UP_LITTLE")
            else:
                if centerX < BaseData.FindBox.nearRightLimit:
                    self.__move("RIGHT")
                elif centerX > BaseData.FindBox.nearLeftLimit:
                    self.__move("LEFT")

        else:
            print("水平")
            if BaseData.FindBox.nearRightLimit <= centerX < BaseData.FindBox.nearLeftLimit:
                if distance >= BaseData.FindBox.boxNearLimit:  # 距离很近
                    self.boxProcessState = RobotEnum.BoxProcessState.UP_BOX
                    #self.__move("UP_LITTLE")
                    #self.__move("HOLD_BOX")
                    self.__move("RIGHT_BALL")
                else:
                    self.__move("UP_LITTLE")
            else:
                if centerX < BaseData.FindBox.nearRightLimit:
                    self.__move("RIGHT")
                elif centerX > BaseData.FindBox.nearLeftLimit:
                    self.__move("LEFT")
        if -75 < abs(ccAngle) < -35:
            print("斜")
            if BaseData.FindBox.nearRightLimit <= centerX < BaseData.FindBox.nearLeftLimit:
                if distance >= 300:
                    self.boxProcessState = RobotEnum.BoxProcessState.UP_BOX
                    self.__move("LEFT_BALL")
                    #self.__move("HOLD_BOX")
                    #self.__timeSleep(1.3)
                    #self.__move("RIGHT_HOLDBOX")
                else:
                    self.__move("UP_LITTLE")
            else:
                if centerX < BaseData.FindBox.nearRightLimit:
                    self.__move("RIGHT")
                elif centerX > BaseData.FindBox.nearLeftLimit:
                    self.__move("LEFT")

        else:
            print("水平")
            if BaseData.FindBox.nearRightLimit <= centerX < BaseData.FindBox.nearLeftLimit:
                if distance >= BaseData.FindBox.boxNearLimit:  # 距离很近
                    self.boxProcessState = RobotEnum.BoxProcessState.UP_BOX
                    #self.__move("UP_LITTLE")
                    #self.__move("HOLD_BOX")
                    self.__move("RIGHT_BALL")
                else:
                    self.__move("UP_LITTLE")
            else:
                if centerX < BaseData.FindBox.nearRightLimit:
                    self.__move("RIGHT")
                elif centerX > BaseData.FindBox.nearLeftLimit:
                    self.__move("LEFT")
    def getAngleError1(self, x, y):
        myAngle = MyAngle()
        if abs(320 - x) < 2:
            return 0
        else:
            __angle = myAngle.get_angle(320, 480, round(x), round(y))
            if __angle > 0:
                return 90 - __angle
            else:
                return -(90 + __angle)

    # 划线
    def __showLine1(self, x, y):
        if self.__currentImg is not None:
            cv2.line(self.__currentImg, (BaseData.Image.half_Width, BaseData.Image.height), (x, y), (0, 255, 0), 1, 4)
        self.__showImage()

    # 放箱子到自己的条码处
    def moveHoldBox1(self, ccLeft, ccTop, ccWidth, ccHeight):
        print("in RobotCtrl.moveHoldBox()")
        barcodeX = ccLeft + int(ccWidth / 2)
        barcodeY = ccTop + int(ccHeight / 2)
        angle = round(self.getAngleError(barcodeX, barcodeY))

        print("码中心点X:={0}, 码中心点Y:={1}, 角度:={2}, 头角度:={3}".format(barcodeX, barcodeY, angle, self.__currentHeadVerticalAngle))

        self.__showLine(barcodeX, barcodeY)

        if self.__currentHeadVerticalAngle == BaseData.FindBox.HeadAnger.FindBarcode.near:
            if abs(angle) > 20:
                if angle > 20:
                    self.__move("RIGHT_HOLDBOX")
                elif angle < -20:
                    self.__move("LEFT_HOLDBOX")
                return False
        elif self.__currentHeadVerticalAngle == BaseData.FindBox.HeadAnger.FindBarcode.far:
            if abs(angle) > 30:
                if angle > 30:
                    self.__move("RIGHT_HOLDBOX")
                elif angle < -30:
                    self.__move("LEFT_HOLDBOX")
                return False

        if barcodeY > 240:
            self.__move("UP_HOLDBOX")
            self.__move("UP_HOLDBOX")
            time.sleep(1)
            self.__move("DOWN_BOX")
            self.boxProcessState = RobotEnum.BoxProcessState.DOWN_BOX
            return True
        if barcodeY > 220:
            self.__move("UP_HOLDBOX")
            self.__move("UP_HOLDBOX")
            self.__move("UP_HOLDBOX")
            time.sleep(1)
            self.__move("DOWN_BOX")
            self.boxProcessState = RobotEnum.BoxProcessState.DOWN_BOX
            return True
        else:
            self.__move("UP_HOLDBOX")
            return False

    def findBox1(self, ccBoxColor):
        print("in RoboxBox.findBox()")
        loopCount = 0
        self.cListFindBoxRect.clear()
        t11 = datetime.datetime.now()

        if self.__getImage():
            self.__showImage()
        while True:
            # print("loopCount:=", loopCount)
            loopCount += 1
            if loopCount > BaseData.FindBox.findBoxLoopCount:
                print("not find ball!")
                print("totalSecond:={}.".format((datetime.datetime.now() - t11).total_seconds()))
                return False
            if self.__getImage():
                try:
                    ret2, self.__currentImg = self.cFindBox.findRect1(ccBoxColor, self.__currentImg)
                except Exception as e:
                    print(e)
                    continue
                if ret2:
                    self.__showImage()

                    if self.cFindBox.matchSizeFind1(10, 270):  # 匹配盒子大小 矩形长宽 50<box<270
                        self.cListFindBoxRect.append(self.cFindBox.cFindBoxRect)
                        # print(self.cFindBox.cFindBoxRect.printStr,
                        #       " , findBoxCount:=%d , loopCount:=%d" % (len(self.cListFindBoxRect), loopCount))

                        if len(self.cListFindBoxRect) > BaseData.FindBox.findBoxCount:
                            print("totalSecond:={}.".format((datetime.datetime.now() - t11).total_seconds()))
                            return True

    # 获取距离平均值
    def __getDistance1(self):
        total = 0
        for x in self.cListFindBoxRect:
            total = total + x.centerY

        return round(total / len(self.cListFindBoxRect))

    # 获取中心轴的平均值
    def __getCenter1(self):
        total = 0
        for x in self.cListFindBoxRect:
            total += x.centerX
        return round(total / len(self.cListFindBoxRect))

    # 获取角度的平均值
    def __getAnge1(self):
        total = 0
        for x in self.cListFindBoxRect:
            total += x.angle
        return round(total / len(self.cListFindBoxRect))

    # 取箱子 主流程
    def __moveToTakeBox1(self, ccBoxColor):
        self.boxProcessState = RobotEnum.BoxProcessState.FIND_BALL
        print("33")
        while 1:
            if self.findBox1(ccBoxColor):
                boxX = self.__getCenter1()  # 中心点 X 平均值
                boxY = self.__getDistance1()  # 中心点 Y 平均值
                boxAngle = self.__getAnge1()  # 箱子的角度

                angle = round(self.getAngleError1(boxX, boxY))

                print("箱子X:={0}, 箱子Y:={1}, 箱子角度:={2}, 头角度:={3}, 角度:={4} "
                      .format(boxX, boxY, boxAngle, self.__currentHeadVerticalAngle, angle))

                self.__showLine1(boxX, boxY)

                if self.__currentHeadVerticalAngle == BaseData.FindBox.HeadAnger.FindBox.far:
                    if abs(angle) > 30:
                        if angle > 30:
                            self.__move("RIGHT")
                            print("44")
                        elif angle < -30:
                            self.__move("LEFT")
                            print("55")
                        continue

                if self.__currentHeadVerticalAngle == BaseData.FindBox.HeadAnger.FindBox.far:
                    if boxY < 350:
                        self.__move("UP")
                        continue
                    else:
                        self.__currentHeadVerticalAngle = BaseData.FindBox.HeadAnger.FindBox.near
                        #self.__currentHeadVerticalAngle = 40
                        self.__robotHead.turn_vertical(95)
                        continue
                if self.__currentHeadVerticalAngle == BaseData.FindBox.HeadAnger.FindBox.near:
                #if self.__currentHeadVerticalAngle == 40:
                    print("66")
                    if boxY < 200:
                        if boxX < 200:
                            self.__move("RIGHT")
                            print("66")
                        elif boxX > 450:
                            self.__move("LEFT")
                            print("77")
                        self.cListFindBoxRect.clear()
                        self.__move("UP")
                        print("88")
                    else:
                        print("22")
                        self.move(boxY, boxX, angle)

                        if self.boxProcessState == RobotEnum.BoxProcessState.UP_BOX:
                            print("boxProcessState=UP_BOX")
                            #self.__move("HOLD_BOX")
                            return True
            else:
                self.__move("LEFT2")  # 没有看到箱子左转身

    # 检测是否拿起来了盒子
    def boxIsHandUp1(self, ccBoxColor):
        print("in RobotBox.boxIsHandUp()")
        loopCount = 0
        listBox = []
        while 1:
            for i in range(10):  # 等待下释放下摄像头延迟
                if self.__getImage():
                    self.__showImage()

            loopCount += 1
            if loopCount > 20:
                print("Find box Error", " , findBoxCount:=%d , loopCount:=%d" % (len(listBox), loopCount))
                return False

            if self.__getImage():
                self.__showImage()
                ret2, self.__currentImg = self.cFindBox.findRect1(ccBoxColor, self.__currentImg)

                if ret2:
                    self.__showImage()

                    if self.cFindBox.matchSizeHandUp1(330):
                        listBox.append(self.cFindBox.cFindBoxRect)

                        print("Find box ", self.cFindBox.cFindBoxRect.printStr,
                              " , findBoxCount:=%d , loopCount:=%d" % (len(listBox), loopCount))

                        if len(listBox) > 3:
                            print("box is in hand")
                            return True

    # 通过摄像头获取条码
    def findBarCode1(self, ccBarcode):
        print("in RobotBox.findBarCode({})".format(ccBarcode))
        loopCount = 0
        t11 = datetime.datetime.now()

        if self.__getImage():
            self.__showImage()

        while 1:
            loopCount += 1
            if loopCount > BaseData.Barcode.findBarcodeLoopCount:
                print("barcode Not found")
                print("loopCount:={0},t:={1}".format(loopCount, (datetime.datetime.now() - t11).total_seconds()))
                return False

            if self.__getImage():
                t1 = datetime.datetime.now()
                ret2, img2, angle = self.cRobotBarcode.decodeDisplay(self.__currentImg, ccBarcode)
                print("loopCount:={0},t:={1}".format(loopCount, (datetime.datetime.now() - t1).total_seconds()))
                if ret2:
                    print("barcode is found")
                    print("loopCount:={0},t:={1}".format(loopCount, (datetime.datetime.now() - t11).total_seconds()))
                    return True
                else:
                    continue

    # 放箱子到条码处
    def putBoxToBarcode1(self, ccBarcode):
        print("in RobotBox.putBoxToBarcode({})".format(ccBarcode))
        self.boxProcessState = RobotEnum.BoxProcessState.BARCODE

        self.__robotHead.turn_vertical(95)
        self.__currentHeadVerticalAngle = BaseData.FindBox.HeadAnger.FindBarcode.far
        self.__robotHead.turn_vertical(self.__currentHeadVerticalAngle)

        while 1:
            ret3 = self.findBarCode1(ccBarcode)
            if ret3:
                print("Barcode rect:=", self.cRobotBarcode.cRobotBarcodeRect.printStr, datetime.datetime.now())
                self.moveHoldBox1(self.cRobotBarcode.cRobotBarcodeRect.left,
                                 self.cRobotBarcode.cRobotBarcodeRect.top,
                                 self.cRobotBarcode.cRobotBarcodeRect.width,
                                 self.cRobotBarcode.cRobotBarcodeRect.height)

                if self.boxProcessState == RobotEnum.BoxProcessState.DOWN_BOX:
                    return True
            else:
                self.__move("RIGHT_HOLDBOX")
                continue

    def TurnHeadStartFindBox1(self):
        self.__robotHead.turn_vertical(95)
        self.__currentHeadVerticalAngle = BaseData.FindBox.HeadAnger.FindBox.far
        self.__robotHead.turn_vertical(self.__currentHeadVerticalAngle)

    # 稳定摄像头
    def __startCamera1(self):
        for x in range(10):
            if self.__getImage():
                self.__showImage()

    def Run(self):
        # 稳定摄像头画面
        self.__startCamera1()

        fBoxColor = "ball"

        self.TurnHeadStartFindBox1()

        while True:
            ret = self.__moveToTakeBox1(fBoxColor)
            if ret:
                ret1 = self.boxIsHandUp1(fBoxColor)
                if ret1:
                    self.putBoxToBarcode1("action2")
                    self.TurnHeadStartFindBox1()
                    return True
                else:
                    print("take up box fail continue")
                    self.__move("BACK")
                    time.sleep(2)
                    continue
            else:
                self.release()


if __name__ == "__main__":
    robotEye = RobotEye()
    robotHead = RobotHeadServoOnly()
    robotMove = RobotMove(robotEye)
    ball2 = RobotHoldBox1(robotEye, robotHead, robotMove)

    ball2.Run()
