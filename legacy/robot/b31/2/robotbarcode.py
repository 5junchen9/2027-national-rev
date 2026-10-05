import datetime
import os
import sys

import cv2
import numpy as np
import time
import pyzbar.pyzbar as pyzbar
import math
import imutils

class RobotBarcodeRect:
    left = 0
    top = 0
    width = 0
    height = 0
    angle = 0
    printStr = ""

    def __init__(self):
        pass

    def __init__(self, ccRect):
        (self.left, self.top, self.width, self.height) = ccRect
        self.printStr = "({0}, {1}, {2}, {3})".format(self.left, self.top, self.width, self.height)


class RobotBarcode:
    def __int__(self):
        pass

    cRobotBarcodeRect = None

    def azimuthangle(self, x1, y1, x2, y2):
        """ 已知两点坐标计算角度 -
        :param x1: 原点横坐标值
        :param y1: 原点纵坐标值
        :param x2: 目标点横坐标值
        :param y2: 目标纵坐标值
        """
        dx = x2 - x1
        dy = y2 - y1
        # 求斜率
        k = dy / dx
        # 结果是弧度值
        angle = math.atan(k)
        # 弧度值转为角度
        return angle * 180 / math.pi

    def get_angle(self, qr_item):
        """
        获取出进行矫正所需要的角度
        """
        # 将坐标从下到上，从左到右进行排序
        locs = {qr_item.polygon[0], qr_item.polygon[1], qr_item.polygon[2], qr_item.polygon[3]}
        locs = sorted(locs, key=lambda x: x.y * 100000 + x.x * 1000)
        return self.azimuthangle(locs[2].x, locs[2].y, locs[3].x, locs[3].y)

    def decodeDisplay(self, image, ccBarcode):
        print("in RobotBarcode.decodeDisplay({0})".format(ccBarcode))
        # 转为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        #    print("decodes:", barcodes)
        if len(barcodes) > 0:
            print("len(barcodes):=", len(barcodes))
            for barcode in barcodes:
                # 提取条形码的边界框的位置
                # 画出图像中条形码的边界框
                (x, y, w, h) = barcode.rect
                fAngle = self.get_angle(barcode)
                print("fAngle =", fAngle)
                #   print(barcode.rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 条形码数据为字节对象，所以如果我们想在输出图像上
                # 画出来，就需要先将它转换成字符串
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type

                # 绘出图像上条形码的数据和条形码类型
                text = "{} ({})".format(barcodeData, barcodeType)
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 向终端打印条形码数据和条形码类型
                print("barcode:=", barcodeData)

                if barcodeData == ccBarcode:
                    print("barcodeData == ccBarcode")
                    self.cRobotBarcodeRect = RobotBarcodeRect(barcode.rect)
                    return True, image, abs(fAngle)

        return False, image, 0

    # 获取所有条码
    def GetBarcodes(self, image):
        print("in {0} {1}()".format(self.__class__, sys._getframe().f_code.co_name))

        # 转为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        #    print("decodes:", barcodes)
        if len(barcodes) > 0:
            print("len(barcodes):=", len(barcodes))
            for barcode in barcodes:
                # 提取条形码的边界框的位置
                # 画出图像中条形码的边界框
                #   print(barcode.rect)
                (x, y, w, h) = barcode.rect
                # fAngle = self.get_angle(barcode)
                # print("fAngle =", fAngle)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 条形码数据为字节对象，所以如果我们想在输出图像上
                # 画出来，就需要先将它转换成字符串
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type

                # 绘出图像上条形码的数据和条形码类型
                text = "{} ({})".format(barcodeData, barcodeType)
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.cRobotBarcodeRect = RobotBarcodeRect(barcode.rect)
                # 向终端打印条形码数据和条形码类型
                # print("barcode:=", barcodeData)

            return True, image, barcodes
        return False, image, None


# 只取需要的条码
def OneBarcode(robotBarcode: RobotBarcode, barcode):
    x = robotBarcode
    # 读取当前帧
    ret, frame = camera.read()
    if ret:
        ret2, img, angle = x.decodeDisplay(frame, "UP")
        if ret2:
            print("barcode Rect:= ", x.cRobotBarcodeRect.printStr)
        cv2.imshow("camera", img)
        cv2.waitKey(1)


# 取得所有条码
def AllBarcode(robotBarcode: RobotBarcode):
    x = robotBarcode
    ret, frame = camera.read()
    if ret:
        ret2, img, barcodes = x.GetBarcodes(frame)
        if ret2:
            for barcode in barcodes:
                print("条码值:={0} ".format(barcode.data.decode("utf-8")))

        cv2.imshow("camera", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    x = RobotBarcode()

    index = 0
    while True:
        index += 1
        if index <= 1:
            t1 = datetime.datetime.now()
        if index >= 1000:
            print("******运行{0}次总秒数:={1}".format(index, (datetime.datetime.now() - t1).total_seconds()))
            index = 0
            break

        # OneBarcode(x, "YELLOW")
        AllBarcode(x)

    print("release()")
    camera.release()
    cv2.destroyAllWindows()
