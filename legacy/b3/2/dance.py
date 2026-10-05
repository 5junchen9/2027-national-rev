import os
import serial
import time
time.sleep(3)
ser = serial.Serial("/dev/ttyAMA0", 9600)  #位置1
if ser.isOpen == False:
    ser.open()
ser.flushInput()  #清空缓冲区域
time.sleep(0.01)  #
def main():
    ser.write("B".encode())
    os.system('mplayer /home/pi/Desktop/语音包/跳舞bgm.mp3')
if __name__ == '__main__':
    main()