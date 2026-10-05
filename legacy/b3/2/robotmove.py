import os
import serial
from roboteye import RobotEye


class RobotMove:
    if os.sep == "/":
        __ser = serial.Serial("/dev/ttyserial0", 9600)
    else:
        __ser = serial.Serial("COM1", 9600)

    __robotEye = None

    def __init__(self, eye):
        self.__robotEye = eye

    def __timeSleep(self, second):
        self.__robotEye.timeSleep(second)

    def robotMove(self, ccAction):
        print("--", self.__class__)
        x = ccAction
        if not self.__ser.isOpen:
            self.__ser.open()

        if x.upper() == "UP":
            print("控制--> 前进 ^^")
            self.__ser.write(b'1')
            self.__timeSleep(1.1)
        elif x.upper() == "LEFT":
            print("控制--> 小步左转<<")            
            self.__ser.write(b'7')
            self.__timeSleep(2)
        elif x.upper() == "LEFT1":
            print("控制--> 左平移  <<")             
            self.__ser.write(b'3')
            self.__timeSleep(2)  
        elif x.upper() == "RIGHT":
            print("控制--> 右平移 >>111")           
            self.__ser.write(b'4')
            self.__timeSleep(2)
        elif x.upper() == "LEFT2":
            print("控制--> 小步左转  <<")            
            self.__ser.write(b'7')
            self.__timeSleep(2)  
        elif x.upper() == "RIGHT1":
            print("控制--> 小步右转 >>")           
            self.__ser.write(b'8')
            self.__timeSleep(2)            
        elif x.upper() == "BACK":
            print("控制--> 后退 vv")
            self.__ser.write(b'2')
            self.__timeSleep(1)
        elif x.upper() == "UP_LITTLE":
            print("控制--> 小步前进 ^")
            self.__ser.write(b'1')
            self.__timeSleep(1)
        elif x.upper() == "HOLD_BOX":
            print("控制--> 抱箱子")
            self.__ser.write(b'D')
            self.__timeSleep(3)
        elif x.upper() == "DOWN_BOX":
            print("控制--> 放箱子")
            self.__ser.write(b'a')
            self.__timeSleep(2)
        elif x.upper() == "UP_HOLDBOX":
            print("控制--> 抱箱子前进 ^^")
            self.__ser.write(b'd')
            self.__timeSleep(1.1)
        elif x.upper() == "LEFT_HOLDBOX":
            print("控制--> 抱箱子左转 <<")
            self.__ser.write(b'f')
            self.__timeSleep(1.3)
        elif x.upper() == "RIGHT_HOLDBOX":
            print("控制--> 抱箱子右转 >>")
            self.__ser.write(b'5')
            self.__timeSleep(1.3)
        elif x.upper() == "LEFT_BALL":
            print("控制--> 左脚踢球 >>")
            self.__ser.write(b'b')
            self.__timeSleep(1.3)
        elif x.upper() == "RIGHT_BALL":
            print("控制--> 右脚踢球 >>")
            self.__ser.write(b'c')
            self.__timeSleep(1.3)
        elif x.upper() == "DANCE":
            print("控制--> dance >>")
            self.__ser.write(b'B')
            self.__timeSleep(1.3)
        elif x.upper() == "BIGLEFT":
            print("控制--> 大步左转 >>")
            self.__ser.write(b'7')
            self.__timeSleep(1.3)
        elif x.upper() == "BIGRIGHT":
            print("控制-->大步右转  >>")
            self.__ser.write(b'8')
            self.__timeSleep(1.3)