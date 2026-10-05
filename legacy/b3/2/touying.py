import cv2
import numpy as np


class TouYing:

    def getTouYingImg(self, image):
        h, w = image.shape[:2]  # 图片的高度和宽度
        pointSrc = np.float32([[0, 0], [w - 1, 0], [0, h-1], [w - 1, h - 1]])  # 原始图像中 4点坐标
        pointDst = np.float32([[0, 0], [w+400, 0], [200, h + 500], [w - 1 + 200, h + 500]])  # 变换图像中 4点坐标
        MP = cv2.getPerspectiveTransform(pointSrc, pointDst)  # 计算投影变换矩阵 M
        imgP = cv2.warpPerspective(image.copy(), MP, (1040, 980))  # 用变换矩阵 M 进行投影变换
        return imgP

    def getXY(self, cx, cy):
        rx = 640 / 1040
        ry = 480 / 980
        x = cx * rx
        y = cy * ry
        return x, y

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)
    # 读取当前帧
    for i in range(20):
        ret, frame = camera.read()
        if ret:
            img = frame
        else:
            print("camera error")

    t = TouYing()
    imgP = t.getTouYingImg(img)
    cv2.imshow('img', img)
    cv2.imshow('imgP', imgP)
    cv2.waitKey(0)
