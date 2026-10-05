import types
import unittest
from unittest.mock import Mock,patch,call
import Head


class HeadServoTests(unittest.TestCase):
    def test_small_tracking_step_settles_early_while_holding_output(self):
        pi=Mock(connected=True)
        pi.set_servo_pulsewidth.return_value=0
        with patch.dict('sys.modules',{'pigpio':types.SimpleNamespace(pi=lambda *args:pi)}), \
             patch.object(Head.time,'monotonic',side_effect=[0,0,.17,.19]):
            servo=Head.RobotHeadServoOnly(hold=True,backend='pigpio')
            servo.begin_vertical(119,settle_seconds=.18)
            self.assertTrue(servo.is_moving())
            self.assertFalse(servo.is_moving())
            pi.set_servo_pulsewidth.assert_called_once_with(27,1682)
            servo.cleanup()
            pi.set_servo_pulsewidth.assert_called_with(27,0)

    def test_pigpio_preserves_calibrated_pulse_mapping_and_holds_until_cleanup(self):
        pi=Mock(connected=True)
        pi.set_servo_pulsewidth.return_value=0
        module=types.SimpleNamespace(pi=Mock(return_value=pi))
        gpio=Mock()
        with patch.dict('sys.modules',{'pigpio':module}), \
             patch.object(Head,'GPIO',gpio,create=True), \
             patch.object(Head.time,'monotonic',side_effect=[0,2,2,4]):
            servo=Head.RobotHeadServoOnly(hold=True,backend='pigpio')
            servo.begin_vertical(117)
            self.assertFalse(servo.is_moving())
            servo.begin_vertical(127)
            self.assertFalse(servo.is_moving())
            self.assertEqual(pi.set_servo_pulsewidth.call_args_list,[call(27,1660),call(27,1771)])
            gpio.PWM.assert_not_called()
            gpio.cleanup.assert_not_called()
            servo.cleanup()
            pi.set_servo_pulsewidth.assert_called_with(27,0)
            pi.stop.assert_called_once()
            module.pi.assert_called_once_with('127.0.0.1',8888)

    def test_pigpio_connection_failure_does_not_fall_back_to_software_pwm(self):
        pi=Mock(connected=False)
        with patch.dict('sys.modules',{'pigpio':types.SimpleNamespace(pi=lambda *args:pi)}), \
             patch.object(Head,'GPIO',Mock(),create=True) as gpio:
            with self.assertRaisesRegex(RuntimeError,'not running'):
                Head.RobotHeadServoOnly(backend='pigpio')
            pi.stop.assert_called_once()
            pi.set_servo_pulsewidth.assert_not_called()
            gpio.PWM.assert_not_called()

    def test_missing_pigpio_fails_before_output(self):
        with patch.dict('sys.modules',{'pigpio':None}):
            with self.assertRaisesRegex(RuntimeError,'Install python3-pigpio'):
                Head.RobotHeadServoOnly(backend='pigpio')

    def test_pigpio_command_error_is_reported_and_connection_cleaned_up(self):
        pi=Mock(connected=True)
        pi.set_servo_pulsewidth.return_value=-1
        with patch.dict('sys.modules',{'pigpio':types.SimpleNamespace(pi=lambda *args:pi)}):
            servo=Head.RobotHeadServoOnly(backend='pigpio')
            with self.assertRaisesRegex(RuntimeError,'rejected'):
                servo.begin_vertical(117)
            servo.cleanup()
            pi.stop.assert_called_once()

    def test_hold_reuses_pwm_without_release_between_tracking_steps(self):
        gpio=Mock()
        with patch.object(Head,'GPIO',gpio,create=True), \
             patch.object(Head,'os',types.SimpleNamespace(sep='/')), \
             patch.object(Head.time,'monotonic',side_effect=[0,2,2,4]):
            servo=Head.RobotHeadServoOnly(hold=True)
            servo.begin_vertical(117)
            self.assertFalse(servo.is_moving())
            servo.begin_vertical(118)
            self.assertFalse(servo.is_moving())
            gpio.PWM.assert_called_once_with(27,50)
            gpio.PWM.return_value.start.assert_called_once_with(1.8+117/360*20)
            gpio.PWM.return_value.ChangeDutyCycle.assert_called_once_with(1.8+118/360*20)
            gpio.PWM.return_value.stop.assert_not_called()
            servo.cleanup()
            gpio.PWM.return_value.stop.assert_called_once()

    def test_nonblocking_move_keeps_drive_and_settle_time_without_sleep(self):
        gpio=Mock()
        with patch.object(Head,'GPIO',gpio,create=True), \
             patch.object(Head,'os',types.SimpleNamespace(sep='/')), \
             patch.object(Head,'SERVO_DRIVE_SECONDS',.8), \
             patch.object(Head.time,'monotonic',side_effect=[0,.79,.81,1.01]), \
             patch.object(Head.time,'sleep') as sleep:
            servo=Head.RobotHeadServoOnly()
            servo.begin_vertical(118)
            self.assertTrue(servo.is_moving())
            gpio.PWM.return_value.stop.assert_not_called()
            self.assertTrue(servo.is_moving())
            gpio.PWM.return_value.stop.assert_called_once()
            self.assertFalse(servo.is_moving())
            sleep.assert_not_called()
            gpio.PWM.return_value.start.assert_called_once_with(1.8+118/360*20)
            servo.cleanup()
            gpio.PWM.return_value.stop.assert_called_once()

    def test_quit_during_move_stops_pwm_immediately(self):
        gpio=Mock()
        with patch.object(Head,'GPIO',gpio,create=True), \
             patch.object(Head,'os',types.SimpleNamespace(sep='/')):
            servo=Head.RobotHeadServoOnly()
            servo.begin_vertical(117)
            servo.cleanup()
            gpio.PWM.return_value.stop.assert_called_once()
            self.assertFalse(servo.is_moving())

    def test_old_blocking_call_still_waits_for_drive_and_settle(self):
        gpio=Mock()
        with patch.object(Head,'GPIO',gpio,create=True), \
             patch.object(Head,'os',types.SimpleNamespace(sep='/')), \
             patch.object(Head,'SERVO_DRIVE_SECONDS',.8), \
             patch.object(Head.time,'monotonic',side_effect=[0,.8]), \
             patch.object(Head.time,'sleep') as sleep:
            servo=Head.RobotHeadServoOnly()
            servo.turn_vertical(117)
            self.assertEqual(sleep.call_args_list,[call(.8),call(.2)])
            gpio.PWM.return_value.stop.assert_called_once()
            servo.cleanup()
