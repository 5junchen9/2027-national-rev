import os
import time

if os.sep == "/":
    import RPi.GPIO as GPIO

class RobotHeadServoOnly:
    servopin2 = 27  # 舵机上下转动

    def __init__(self):
        self.__initGPIO()

    def __initGPIO(self):
        if os.sep == "/":
            GPIO.cleanup()
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.servopin2, GPIO.OUT)

    def __setServoAngle(self, servo, num):
        assert 85 <= num <= 180
        if os.sep == "/":
            pwm = GPIO.PWM(servo, 50)
            pwm.start(8)
            dutyCycle = 1.8 + num / 360 * 20
            pwm.ChangeDutyCycle(dutyCycle)
            time.sleep(0.3)
            pwm.stop()
        print(f"Servo {servo} set to angle {num}")

    def turn_vertical(self, angle):
        self.__setServoAngle(self.servopin2, angle)
        time.sleep(1)

    def cleanup(self):
        if os.sep == "/":
            GPIO.cleanup()

    def head_action(self, command):
        if command == "抬头":
            self.turn_vertical(85)
        elif command == "低头":
            self.turn_vertical(120)
        elif command == "a":
            self.turn_vertical(95)

