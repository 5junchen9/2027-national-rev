import unittest
from unittest.mock import Mock, patch
import numpy as np

from line_search_route import detect_line, LinePlanner, run_course
from competition_line_main import LineCompetitionIO
from competition_main import load_settings, CONFIG_FILE


class LineTests(unittest.TestCase):
    def test_white_line_on_blue_floor_and_black_line_on_white_floor(self):
        white = np.full((480,640,3), (100,30,20), np.uint8)
        white[384:,310:330] = 255
        self.assertAlmostEqual(detect_line(white),319.5)
        black = np.full((480,640,3),255,np.uint8)
        black[384:,310:330] = 0
        self.assertAlmostEqual(detect_line(black,'black'),319.5)

    def test_whole_bright_floor_is_not_a_line(self):
        self.assertIsNone(detect_line(np.full((480,640,3),255,np.uint8)))

    def test_follow_previous_local_candidate(self):
        image = np.zeros((480,640,3),np.uint8)
        image[384:,200:220] = 255
        image[384:,360:380] = 255
        self.assertLess(detect_line(image,previous_x=210),230)

    def test_loss_search_is_bounded_and_reacquisition_resets_it(self):
        planner=LinePlanner()
        self.assertEqual([planner.decide(None,640) for _ in range(5)],
                         ['TURN_RIGHT','TURN_LEFT','TURN_LEFT','TURN_RIGHT','STOP'])
        self.assertEqual(planner.decide(320,640),'UP_LITTLE')
        self.assertEqual(planner.decide(None,640),'TURN_RIGHT')
        self.assertEqual(planner.decide(240,640),'SIDE_LEFT')
        self.assertEqual(planner.decide(450,640),'TURN_RIGHT')

    def make_io(self):
        io=LineCompetitionIO(load_settings(CONFIG_FILE),Mock(),float('inf'))
        io.open_views=Mock()
        io.stream=Mock()
        frame=np.zeros((480,640,3),np.uint8)
        io.observe_ready=Mock(return_value=(frame,frame,{}))
        io.move=Mock()
        return io

    @patch('competition_line_main.cv2.imshow')
    def test_initial_missing_line_stops_instead_of_carry(self, show):
        io=self.make_io()
        with self.assertRaisesRegex(RuntimeError,'尚未沿线前进'):
            io.follow_to_carry('red')
        self.assertEqual(io.move.call_count,4)

    @patch('competition_line_main.cv2.imshow')
    def test_forward_then_exit_search_enters_carry(self, show):
        io=self.make_io()
        with patch('competition_line_main.detect_line',side_effect=[320]+[None]*10):
            io.follow_to_carry('red')
        self.assertEqual(io.move.call_args_list[0].args,('UP_LITTLE',))
        self.assertEqual(io.move.call_count,5)

    def test_course_has_no_left_or_action1_scan_and_preserves_later_tasks(self):
        io=Mock();io.settings=load_settings(CONFIG_FILE)
        run_course(io,'red')
        io.follow_to_carry.assert_called_once_with('red')
        self.assertEqual([call.args[0] for call in io.scan_until.call_args_list],['face','dance'])
        io.carry.assert_called_once_with('red','action2')
        names=[call[0] for call in io.method_calls]
        self.assertLess(names.index('stand'),names.index('follow_to_carry'))
        self.assertLess(names.index('follow_to_carry'),names.index('carry'))
        self.assertLess(names.index('carry'),names.index('sport'))
        io.dance.assert_called_once()


if __name__ == '__main__':
    unittest.main()
