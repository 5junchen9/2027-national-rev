import cv2
import numpy as np
from colorhsv_mouse import ColorHsv


# 找到的盒子
class FindBoxRect:
    centerX = 0
    centerY = 0
    width = 0
    height = 0
    angle = 0
    printStr = ""

    def __init__(self):
        pass

    def __init__(self, rect):
        self.centerX = round(rect[0][0])
        self.centerY = round(rect[0][1])
        self.width = round(rect[1][0])
        self.height = round(rect[1][1])
        self.angle = round(rect[2])
        self.printStr = "(({0}, {1}), ({2}, {3}), {4})".format(self.centerX, self.centerY, self.width, self.height,
                                                               self.angle)


class FindBox:
    __blackLower = [0, 0, 0]
    __blackUpper = [0, 0, 0]
    __color_dist = {
        'red': {'Lower': [0, 0, 0], 'Upper': [0, 0, 0]},
        'blue': {'Lower': [0, 0, 0], 'Upper': [0, 0, 0]},
        'yellow': {'Lower': [0, 0, 0], 'Upper': [0, 0, 0]},
        'ball': {'Lower': [0, 0, 0], 'Upper': [0, 0, 0]},
    }

    cFindBoxRect = None
    cCurrentColor = ""

    def __init__(self):
        colorHsv = ColorHsv()

        color = 'yellow'
        colorFileName = color + 'hsv'
        colorHsv.open(colorFileName)
        self.__color_dist[color]["Lower"] = np.add(colorHsv.getColorLower(), self.__color_dist[color]["Lower"])
        self.__color_dist[color]["Upper"] = np.add(colorHsv.getColorUpper(), self.__color_dist[color]["Upper"])
        print(self.__color_dist[color]["Lower"], self.__color_dist[color]["Upper"])

        color = 'ball'
        colorFileName = color + 'hsv'
        colorHsv.open(colorFileName)
        self.__color_dist[color]["Lower"] = np.add(colorHsv.getColorLower(), self.__color_dist[color]["Lower"])
        self.__color_dist[color]["Upper"] = np.add(colorHsv.getColorUpper(), self.__color_dist[color]["Upper"])
        print(self.__color_dist[color]["Lower"], self.__color_dist[color]["Upper"])

        color = 'blue'
        colorFileName = color + 'hsv'
        colorHsv.open(colorFileName)
        self.__color_dist[color]["Lower"] = np.add(colorHsv.getColorLower(), self.__color_dist[color]["Lower"])
        self.__color_dist[color]["Upper"] = np.add(colorHsv.getColorUpper(), self.__color_dist[color]["Upper"])
        print(self.__color_dist[color]["Lower"], self.__color_dist[color]["Upper"])

        color = 'red'
        colorFileName = color + 'hsv'
        colorHsv.open(colorFileName)
        self.__color_dist[color]["Lower"] = np.add(colorHsv.getColorLower(), self.__color_dist[color]["Lower"])
        self.__color_dist[color]["Upper"] = np.add(colorHsv.getColorUpper(), self.__color_dist[color]["Upper"])
        print(self.__color_dist[color]["Lower"], self.__color_dist[color]["Upper"])

    def findRect(self, ccBoxColor, ccImage):
        # print("in FindBox.findRect({})".format(ccBoxColor))
        if ccImage is not None:

            gs_frame = cv2.GaussianBlur(ccImage, (5, 5), 0)
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)

            erode_hsv = cv2.erode(hsv, None, iterations=2)
            # print(self.__color_dist[ccBoxColor]['Lower'], self.__color_dist[ccBoxColor]['Upper'])
            inRange_hsv = cv2.inRange(erode_hsv, self.__color_dist[ccBoxColor]['Lower'], self.__color_dist[ccBoxColor]['Upper'])

            # cv2.imshow('3', inRange_hsv)
            # cv2.waitKey(1)

            cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                #                print("rect: ",rect)
                #                print("box: ", box)
                cv2.drawContours(ccImage, [np.int0(box)], -1, (0, 255, 255), 2)

                self.cFindBoxRect = FindBoxRect(rect)

                return True, ccImage
            else:
                return False, ccImage

        else:
            return False, ccImage

    # 大小是否匹配地上的盒子
    def matchSizeFind(self, ccMinSize, ccMaxSize):
        if ccMinSize < self.cFindBoxRect.width < ccMaxSize and ccMinSize < self.cFindBoxRect.height < ccMaxSize:
            # print("in matchSizeFind() match box size:={0}x{1}".format(self.cFindBoxRect.width,self.cFindBoxRect.height))
            return True
        else:
            return False

    # 大小是否匹配手上的盒子
    def matchSizeHandUp(self, ccMaxSize):
        if self.cFindBoxRect.width > ccMaxSize or self.cFindBoxRect.height > ccMaxSize:
            return True
        else:
            return False

    cFindBoxRect = None
    cCurrentColor = ""

    def findRect(self, ccBoxColor, ccImage):
        # print("in FindBox.findRect({})".format(ccBoxColor))
        if ccImage is not None:

            gs_frame = cv2.GaussianBlur(ccImage, (5, 5), 0)
            hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)

            erode_hsv = cv2.erode(hsv, None, iterations=2)
            # print(self.__color_dist[ccBoxColor]['Lower'], self.__color_dist[ccBoxColor]['Upper'])
            inRange_hsv = cv2.inRange(erode_hsv, self.__color_dist[ccBoxColor]['Lower'], self.__color_dist[ccBoxColor]['Upper'])

            # cv2.imshow('3', inRange_hsv)
            # cv2.waitKey(1)

            cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                #                print("rect: ",rect)
                #                print("box: ", box)
                cv2.drawContours(ccImage, [np.int0(box)], -1, (0, 255, 255), 2)

                self.cFindBoxRect = FindBoxRect(rect)

                return True, ccImage
            else:
                return False, ccImage

        else:
            return False, ccImage

    # 大小是否匹配地上的盒子
    def matchSizeFind(self, ccMinSize, ccMaxSize):
        if ccMinSize < self.cFindBoxRect.width < ccMaxSize and ccMinSize < self.cFindBoxRect.height < ccMaxSize:
            # print("in matchSizeFind() match box size:={0}x{1}".format(self.cFindBoxRect.width,self.cFindBoxRect.height))
            return True
        else:
            return False

    # 大小是否匹配手上的盒子
    def matchSizeHandUp(self, ccMaxSize):
        if self.cFindBoxRect.width > ccMaxSize or self.cFindBoxRect.height > ccMaxSize:
            return True
        else:
            return False


# 测试下
if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    x = FindBox()

    while True:

        if cv2.waitKey(10) & 0xFF == 27:
            print("exit 1")
            break
        # 读取当前帧
        ret, frame = camera.read()
        ret2, img = x.findRect("red", frame)
        if ret2:
            if x.matchSizeFind(10, 270):  # 匹配盒子大小 矩形长宽 50<box<270
                print(ret2, x.cFindBoxRect.printStr)

        cv2.imshow("camera", img)
        cv2.waitKey(1)

    camera.release()
    cv2.destroyAllWindows()
