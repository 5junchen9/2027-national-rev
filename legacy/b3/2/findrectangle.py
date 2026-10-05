import datetime
import os
import cv2
import numpy as np
from touying import TouYing
from basedata import BaseData

# if os.sep == "/":
#     system = "linux"
# else:
#     system = "windows"


class FindRectangle:
    # 计算两边夹角额cos值
    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    def find_squares(self, img):
        squares = []
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin = cv2.Canny(gray, 30, 300, apertureSize=3)
        if os.sep == "/":
            img2, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print("轮廓数量：%d" % len(contours))
        minY = 0
        index = 0
        # 轮廓遍历
        for cnt in contours:
            cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
            cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
            # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
            if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):

                M = cv2.moments(cnt)  # 计算轮廓的矩
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])  # 轮廓重心

                print("index:={3}, 边数量:={0}, 轮廓面积:={1}, 轮廓是否为凸的:={2}, 轮廓重心:={4},{5} , 时间:={6}".format(len(cnt), cv2.contourArea(cnt), cv2.isContourConvex(cnt), index, cx, cy, datetime.datetime.now()))

                cnt = cnt.reshape(-1, 2)
                max_cos = np.max([self.angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                print("max_cos:=", max_cos)
                # 只检测矩形（cos90° = 0）
                if max_cos < 1:
                    # 检测四边形（不限定角度范围）
                    # if True:
                    index = index + 1
                    squares.append(cnt)

                    if minY == 0:
                        minY = cy
                        retX = cx
                        retY = cy
                    elif cy > minY:
                        retX = cx
                        retY = cy

        if len(squares) > 0:
            return True, squares, img, retX, retY
        else:
            return False, squares, img, 0, 0

    # 获取最近的矩形
    def getNearRectangle(self, image):
        print("in {0}.{1}()".format(self.__class__, self.__class__.__name__))
        flist = []
        fSquares = []
        for i in range(BaseData.FindRectangle.findTimes):
            ret, squares, img, retX, retY = self.find_squares(image)
            if ret:
                fSquares = squares
                flist.append([retX, retY])

        if len(flist) > 0:
            x, y = max(flist, key=lambda x: x[1])
            return True, fSquares, img, x, y
        else:
            return False, fSquares, img, 0, 0


if __name__ == '__main__':
    print(__name__)

    touying = TouYing()
    findRect = FindRectangle()
    camera = cv2.VideoCapture(0)

    # 读取当前帧
    for i in range(20):
        ret, frame = camera.read()
        if ret:
            img = frame
        else:
            print("camera error")

    while 1:
        ret, frame = camera.read()
        if ret:
            img = frame
        else:
            print("camera error")

        imgP = touying.getTouYingImg(img)
        #imgP = img
        t1 = datetime.datetime.now()
        ret, squares, imgP, cx, cy = findRect.getNearRectangle(imgP)
        print("timepass:=", (datetime.datetime.now()-t1).total_seconds())
        cv2.drawContours(imgP, squares, -1, (0, 0, 255), 2)

        if ret:
            x, y = touying.getXY(cx, cy)
            cv2.line(imgP, (520, 980), (cx, cy), (0, 255, 0), 1, 4)
            cv2.line(img, (320, 480), (int(x), int(y)), (0, 255, 0), 1, 4)
        cv2.imshow("img", img)
        cv2.imshow('squares', imgP)
        ch = cv2.waitKey(100)

        print('Done')



    cv2.destroyAllWindows()
