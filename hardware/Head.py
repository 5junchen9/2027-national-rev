import os
import time

HEAD_LEVEL_ANGLE = int(os.environ.get("ROBOT_HEAD_LEVEL", "95"))
HEAD_DOWN_ANGLE = int(os.environ.get("ROBOT_HEAD_DOWN", "120"))
SERVO_DRIVE_SECONDS = float(os.environ.get("ROBOT_HEAD_DRIVE_SECONDS", "0.8"))
HEAD_PWM_BACKEND = os.environ.get("ROBOT_HEAD_PWM_BACKEND", "rpi").lower()

if os.sep == "/":
    import RPi.GPIO as GPIO

class RobotHeadServoOnly:
    servopin2 = 27  # 舵机上下转动

    def __init__(self, hold=False, backend=None):
        self._pwm = None
        self._pi = None
        self._driving = False
        self._hold = hold
        self.backend = HEAD_PWM_BACKEND if backend is None else backend
        if self.backend not in ('rpi','pigpio'):
            raise ValueError('ROBOT_HEAD_PWM_BACKEND must be rpi or pigpio')
        self._drive_until = self._ready_at = 0.0
        if self.backend == 'pigpio':
            try:
                import pigpio
            except ImportError as error:
                raise RuntimeError('Install python3-pigpio; use a system-site-packages venv') from error
            self._pi = pigpio.pi('127.0.0.1',8888)
            if not self._pi.connected:
                self._pi.stop()
                self._pi = None
                raise RuntimeError('pigpiod is not running. Run: bash /home/yuting/robot_env/setup_head_pwm.sh start; check its startup log if it fails.')
        else:
            self.__initGPIO()

    def __initGPIO(self):
        if os.sep == "/":
            GPIO.cleanup(self.servopin2)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.servopin2, GPIO.OUT)

    def begin_vertical(self, angle, settle_seconds=None):
        """Start a move; callers must poll is_moving and always clean up."""
        assert 85 <= angle <= 180
        if settle_seconds is not None and (not self._hold or not .08 <= settle_seconds <= 2.0):
            raise ValueError('Short settling requires hold=True and 0.08..2.0 seconds')
        if self.backend == 'pigpio':
            # Preserve calibrated pulse widths: 117 -> 1660us, 127 -> 1771us.
            pulse_us = round((1.8 + angle / 360 * 20)*200)
            result = self._pi.set_servo_pulsewidth(self.servopin2,pulse_us)
            if result < 0:
                raise RuntimeError('pigpio rejected head pulse: '+str(result))
        elif os.sep == "/":
            duty = 1.8 + angle / 360 * 20
            if self._pwm is None:
                self._pwm = GPIO.PWM(self.servopin2, 50)
                self._pwm.start(duty)
            else:
                self._pwm.ChangeDutyCycle(duty)
        self._driving = True
        self._drive_until = time.monotonic() + max(0.3, SERVO_DRIVE_SECONDS)
        self._ready_at = self._drive_until + 0.2
        if settle_seconds is not None:
            self._ready_at = time.monotonic() + settle_seconds
        print(f"Servo {self.servopin2} set to angle {angle} ({self.backend})")

    def _stop_output(self):
        if self._pi is not None and self._driving:
            self._pi.set_servo_pulsewidth(self.servopin2,0)
        if self._pwm is not None:
            self._pwm.stop()
            self._pwm = None
        self._driving = False

    def is_moving(self):
        now = time.monotonic()
        if now >= self._drive_until and not self._hold:
            self._stop_output()
        return now < self._ready_at

    def turn_vertical(self, angle):
        # Existing tasks retain their synchronous move-and-settle behavior.
        self.begin_vertical(angle)
        time.sleep(max(0.3, SERVO_DRIVE_SECONDS))
        self.is_moving()
        time.sleep(0.2)

    def level(self):
        self.turn_vertical(HEAD_LEVEL_ANGLE)

    def look_down(self):
        self.turn_vertical(HEAD_DOWN_ANGLE)

    def cleanup(self):
        try:
            self._stop_output()
        finally:
            if self._pi is not None:
                self._pi.stop()
                self._pi = None
        self._drive_until = self._ready_at = 0.0
        if self.backend == 'rpi' and os.sep == "/":
            GPIO.cleanup(self.servopin2)

    def head_action(self, command):
        if command == "抬头":
            self.turn_vertical(85)
        elif command == "低头":
            self.look_down()
        elif command == "a":
            self.level()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fixed head test. No cameras or body motion.')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--angle',type=int,default=124)
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--backend',choices=('rpi','pigpio'),default='pigpio')
    args = parser.parse_args()
    if not args.run: parser.error('Add --run to move the head')
    if not 85 <= args.angle <= 180 or not 0 < args.seconds <= 60:
        parser.error('angle must be 85..180; seconds must be >0 and <=60')
    servo = RobotHeadServoOnly(hold=True,backend=args.backend)
    try:
        servo.begin_vertical(args.angle)
        print('Fixed head: observe jitter. Ctrl+C stops output.')
        deadline = time.monotonic()+args.seconds
        while time.monotonic() < deadline:
            servo.is_moving()
            time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        servo.cleanup()
