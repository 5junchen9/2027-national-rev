import datetime
import json
import os
import time

import cv2
import numpy as np

# 3 保存 blackhsv.json
# 4 保存 bluehsv.json
# 5 保存 yellowhsv.json
# 6 保存 redhsv.json

# 采用固定点方式, 查找黑色线条的hsv,并保存到blackhsv.json文件

if os.sep == "/":
    path = "/home/pi/Desktop/b3/Line/img/"
else:
    path = "F:\\_AAA-BXL\\IMG\\"


class ColorHsvSave:
    count = 0
    lowerH = 0
    lowerS = 0
    lowerV = 0
    upperH = 0
    upperS = 0
    upperV = 0
    saveDatetime = ""
    color = ""

    def __init__(self, count=0, lowerH=0, lowerS=0, lowerV=0, upperH=0, upperS=0,
                 upperV=0, color=""):
        self.count = count
        self.lowerH = int(lowerH)
        self.lowerS = int(lowerS)
        self.lowerV = int(lowerV)

        self.upperH = int(upperH)
        self.upperS = int(upperS)
        self.upperV = int(upperV)
        self.saveDatetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.color = color


class ColorHsvData:
    count = 0
    lower = [0, 0, 0]
    upper = [0, 0, 0]


class ColorHsv:
    __colorHsvData = ColorHsvData()
    __hsvImg = None
    __hsvFileName = "blackhsv"

    def getColorLower(self):
        return self.__colorHsvData.lower

    def getColorUpper(self):
        return self.__colorHsvData.upper

    def __init__(self):
        pass

    def canShowHsv(self):
        return self.__colorHsvData.count > 0

    def open(self, hsvFileName):
        path1 = "{0}{1}{2}".format(path, hsvFileName, ".json")
        print(path1)
        with open(path1) as file_obj:
            numbers = json.load(file_obj)
        print(numbers)
        x = ColorHsvSave()
        x.__dict__ = numbers

        print(x.lowerH, x.lowerS, x.lowerV)
        self.__colorHsvData.count = x.count
        self.__colorHsvData.lower = np.array([x.lowerH, x.lowerS, x.lowerV])
        self.__colorHsvData.upper = np.array([x.upperH, x.upperS, x.upperV])
        print("self.__colorHsvData", x.saveDatetime, x.color, self.__colorHsvData.count, self.__colorHsvData.lower,
              self.__colorHsvData.upper)

    def save(self, hsvFileName):
        s = self.__colorHsvData
        x = ColorHsvSave(s.count, s.lower[0], s.lower[1], s.lower[2], s.upper[0],
                         s.upper[1], s.upper[2], hsvFileName)
        jsonStr1 = json.dumps(x.__dict__)
        print(jsonStr1)

        # now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        now = hsvFileName  # "blackhsv"
        path1 = "{0}{1}{2}".format(path, now, ".json")
        # print(path1)
        with open(path1, 'w') as file_obj:
            json.dump(x.__dict__, file_obj)

    def getBlackHsvAuto(self, px, py):

        x = self.__colorHsvData
        print("lower:=", x.lower)
        print("upper:=", x.upper)

        y1 = py - 10  # 行
        y2 = py + 10
        x1 = px - 10  # 列
        x2 = px + 10

        w = 20
        h = 20

        a = self.__hsvImg[y1:y2, x1:x2]  # 数值切片 取需要的行列
        b = np.reshape(a, (w * h, 3))  # 把三维数值变成二维数值
        min = np.amin(b, axis=0)
        max = np.amax(b, axis=0)
        print("min:=", min)
        print("max:=", max)

        if min[0] == 0 or min[1] == 0 or min[2] == 0:
            print("ERROR 错误的数据")
            return

        if x.count > 0:
            # if int(x.lower[0]) - int(min[0]) > 20:
            #     print("ERROR 数据0误差超限错误")
            #     return
            # if int(x.lower[1]) - int(min[1]) > 20:
            #     print("ERROR 数据1误差超限错误")
            #     return
            # if int(x.lower[2]) - int(min[2]) > 20:
            #     print("ERROR 数据2误差超限错误")
            #     return

            v = [min, x.lower]
            x.lower = np.amin(v, axis=0)
            w = [max, x.upper]
            x.upper = np.amax(w, axis=0)
        else:
            x.lower = min
            x.upper = max

        print("lower:=", x.lower)
        print("upper:=", x.upper)
        x.count += 1

        print("黑色计数:=", x.count)

    # HSV方式查找黑线
    def findBlackHsv(self, ccImage):
        # print("in findbhsv")
        # 把 BGR 转为 HSV
        gs_frame = cv2.GaussianBlur(ccImage, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
        self.__hsvImg = hsv
        cv2.namedWindow("hsv")
        cv2.imshow("hsv", hsv)
        if self.__colorHsvData.count > 0:
            erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
            # erode_hsv = cv2.dilate(hsv, None, iterations=2)  # 腐蚀 粗的变细
            inRange_hsv = cv2.inRange(erode_hsv, self.__colorHsvData.lower,
                                      self.__colorHsvData.upper)
            cv2.namedWindow("inRange_hsv")
            cv2.imshow("inRange_hsv", inRange_hsv)
        else:
            cv2.namedWindow("inRange_hsv")
            cv2.imshow("inRange_hsv", hsv)

    # HSV方式查找黑线
    def showHsvImg(self, ccImage):
        # 把 BGR 转为 HSV
        gs_frame = cv2.GaussianBlur(ccImage, (5, 5), 0)  # 高斯模糊
        hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像

        if self.__colorHsvData.count <= 0:
            print("self.__colorHsvData", self.__colorHsvData.count, self.__colorHsvData.lower,
                  self.__colorHsvData.upper)
            print("blackHsv没有赋初值")
            return

        erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
        # erode_hsv = cv2.dilate(hsv, None, iterations=2)  # 腐蚀 粗的变细
        inRange_hsv = cv2.inRange(erode_hsv, self.__colorHsvData.lower,
                                  self.__colorHsvData.upper)

        cv2.imshow("inRange_hsv", inRange_hsv)


colorHsv = ColorHsv()


def printShow():
    print("")
    print("-------使用说明----------")
    print("**在camera窗口用鼠标点击需要的颜色**")
    print("(1) 保存黑色hsv数据文件 按1")
    print("(2) 保存蓝色hsv数据文件 按2")
    print("(3) 保存黄色hsv数据文件 按3")
    print("(4) 保存红色hsv数据文件 按4")
    print("(5) 保存greenhsv数据文件 按5")
    print("-----")
    print("(6) 读取文件ballhsv数据 按6")
    print("(7) 读取文件蓝色hsv数据 按7")
    print("(8) 读取文件黄色hsv数据 按8")
    print("(9) 读取文件红色hsv数据 按9")
    print("")


def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "(%d,%d)" % (x, y)  # 设置坐标显示格式
        print(xy)
        colorHsv.getBlackHsvAuto(x, y)
        printShow()


# 测试下
if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    flag = 2
    isFirstRun = True

    if camera.isOpened():
        isPress = True
        a = 20  # 过滤前面的视频帧
        while True:
            ret, frame = camera.read()
            frame = cv2.flip(frame, -1)
            if a > 0:
                a -= 1
                time.sleep(0.01)
                continue

            if isPress:
                printShow()
                isPress = False

            if cv2.waitKey(10) & 0xFF == 27:
                print("exit 1")
                break

            if cv2.waitKey(100) & 0xFF == ord("1"):
                colorHsv.save("blackhsv")
                colorHsv.open("blackhsv")
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("2"):
                colorHsv.save("bluehsv")
                colorHsv.open("bluehsv")
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("3"):
                colorHsv.save("yellowhsv")
                colorHsv.open("yellowhsv")
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("4"):
                colorHsv.save("redhsv")
                colorHsv.open("redhsv")
                isPress = True
                flag = 1
            if cv2.waitKey(100) & 0xFF == ord("5"):
                colorHsv.save("greenhsv")
                colorHsv.open("greenhsv")
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("6"):
                colorHsv.save("ballhsv")
                colorHsv.open("ballhsv")
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("7"):
                colorHsv.open("bluehsv")
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("8"):
                colorHsv.open("yellowhsv")
                colorHsv.findBlackHsv(frame)
                isPress = True
                flag = 1

            if cv2.waitKey(100) & 0xFF == ord("9"):
                colorHsv.open("redhsv")
                colorHsv.findBlackHsv(frame)
                isPress = True
                flag = 1

            time.sleep(0.1)
            # if flag == 2:
            cv2.imshow("camera", frame)
            cv2.waitKey(1)
            colorHsv.findBlackHsv(frame.copy())

            if colorHsv.canShowHsv():
                colorHsv.showHsvImg(frame)

            if isFirstRun:
                cv2.setMouseCallback("camera", mouse)
                isFirstRun = False

    print("release")
    camera.release()
    cv2.destroyAllWindows()

2