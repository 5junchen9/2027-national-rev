import datetime

import cv2


class RobotEye:
    __camera = None

    def __init__(self):
        self.__camera = cv2.VideoCapture(0)

    # 闲了就读取摄像头缓存,减少画面延迟
    def __getImage(self):
        if not self.__camera.isOpened():
            print("error in __getImage() camera.isOpened()")
            return
        self.__camera.read()

    def timeSleep(self, ccSecond):
        t2 = datetime.datetime.now() + datetime.timedelta(seconds=ccSecond)
        while 1:
            self.__getImage()
            if datetime.datetime.now() > t2:
                return

    def getImage(self):
        if self.__camera.isOpened:
            ret, frame = self.__camera.read()
            if ret:
                image = cv2.flip(frame, 0)
                return True, image
            else:
                return False, None
        else:
            return False, None
    def getImage1(self):
        if self.__camera.isOpened:
            ret, frame = self.__camera.read()
            
            if ret:
                image = cv2.flip(frame, -1)
                return True, ret,image
            else:
                return False, None
        else:
            return False, None

    def showImage(self, ccImg):
        cv2.imshow("camera", ccImg)
        cv2.waitKey(1)


if __name__ == "__main__":
    x = RobotEye()

    while 1:
        ret, img = x.getImage()
        if ret:
            x.showImage(img)
        else:
            print("error in x.getImage() ")
