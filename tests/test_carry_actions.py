"""只测试动作文件和模拟发送，不打开真实串口。"""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import dreammaker_protocol as protocol
import robotmove


class CarryActionTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(robotmove.ACTION_DIR)
        self.old = self.folder/'搬运右转.dzz'
        self.fixed = self.folder/'搬运右转-补齐收尾.dzz'

    def test_original_frames_and_timing_are_preserved(self):
        old = protocol.load_dzz(self.old)
        fixed = protocol.load_dzz(self.fixed)
        self.assertTrue(self.fixed.read_bytes().startswith(self.old.read_bytes()))
        self.assertEqual(fixed[:-1],old)
        self.assertEqual(len(fixed),10)
        self.assertEqual(fixed[-1][1],50)

    def test_final_pose_recovers_dynamic_channels_and_keeps_hold(self):
        fixed = protocol.load_dzz(self.fixed)
        normal = protocol.load_dzz(self.folder/'右转.dzz')
        for (offsets,duration),(turn_offsets,turn_duration) in zip(fixed,normal):
            self.assertEqual(offsets[:7],fixed[0][0][:7])
            self.assertEqual(offsets[7:],turn_offsets[7:])
            self.assertEqual(duration,turn_duration)
        self.assertEqual(fixed[-1][0][7:],[0]*17)
        # 左转原有收尾正常，本轮保持调用原动作。
        self.assertEqual(robotmove.ACTIONS['LEFT_HOLDBOX'],('搬运左转.dzz',1))
        left = protocol.load_dzz(self.folder/'搬运左转.dzz')
        self.assertEqual(left[-1][0][7:],[0]*17)

    def test_right_hold_action_sends_ten_frames_without_speed_change(self):
        controller = protocol.DreamMakerController(serial_port=Mock(),online_protocol=True)
        with patch.object(robotmove,'DreamMakerController',return_value=controller), \
             patch.object(controller,'open'),patch.object(controller,'stand'), \
             patch.object(controller,'_write') as write,patch.object(protocol.time,'sleep'):
            robot = robotmove.RobotMove(None)
            robot.robotMove('RIGHT_HOLDBOX')
            robot.close()
        frames = protocol.load_dzz(self.fixed)
        self.assertEqual(write.call_count,10)
        for call,(offsets,duration) in zip(write.call_args_list,frames):
            positions = protocol.apply_offsets(protocol.DEFAULT_INITIAL_POSITIONS,offsets)
            expected = protocol.motion_frame(positions,protocol.scaled_duration_ms(duration))
            self.assertEqual(call.args[0],expected)
        final_positions = protocol.apply_offsets(protocol.DEFAULT_INITIAL_POSITIONS,frames[-1][0])
        self.assertEqual(final_positions[7:],list(protocol.DEFAULT_INITIAL_POSITIONS[7:]))


if __name__ == '__main__':
    unittest.main()
