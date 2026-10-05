import time
import RPi.GPIO as GPIO


class HeadServo:
    servopin1 = 22  # 舵机1,方向为左右转head
    servopin2 = 27  # 舵机2,方向为上下转

    def __int__(self):
        pass

    def Init(self):
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servopin1, GPIO.OUT)
        GPIO.setup(self.servopin2, GPIO.OUT)

    def __setServoAngle(self, servo, num):
        assert 0 <= num <= 180
        pwm = GPIO.PWM(servo, 50)
        pwm.start(8)
        dutyCycle = 1.8 + num / 360 * 20
        pwm.ChangeDutyCycle(dutyCycle)
        time.sleep(0.3)

    def InitServo(self):
        self.__setServoAngle(22, 95)
        time.sleep(1) 
        self.__setServoAngle(22, 130)
        time.sleep(1)
        self.__setServoAngle(22, 50)
        time.sleep(1)
        self.__setServoAngle(22, 93)
        time.sleep(1)
        
        self.__setServoAngle(27, 100)
        time.sleep(1)
        self.__setServoAngle(27, 60)
        time.sleep(1)
        # os._exit()

    def TakeBox(self):
        self.__setServoAngle(22, 50)
        time.sleep(1)
        self.__setServoAngle(22, 45)
        time.sleep(1)

    def FindBarCode(self):
        self.__setServoAngle(22, 60)
        time.sleep(1)

if __name__ == "__main__":
    x = HeadServo()
    x.Init()
    x.InitServo()