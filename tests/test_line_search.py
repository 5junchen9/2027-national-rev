import unittest
from unittest.mock import Mock, patch
import numpy as np

from line_search_route import detect_line, LinePlanner, run_course
from competition_line_main import LineCompetitionIO
from competition_main import load_settings, CONFIG_FILE
from kick_shapes import Box


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
                         ['TURN_RIGHT','TURN_RIGHT','TURN_RIGHT','TURN_RIGHT','STOP'])
        self.assertEqual(planner.decide(320,640),'UP_LITTLE')
        self.assertEqual(planner.decide(None,640),'TURN_RIGHT')
        self.assertEqual(planner.decide(240,640),'SIDE_LEFT')
        self.assertEqual(planner.decide(450,640),'TURN_RIGHT')

    def test_far_right_bend_turns_before_near_line_leaves_center(self):
        planner=LinePlanner()
        self.assertEqual(planner.decide(320,640,410),'TURN_RIGHT')
        self.assertEqual(planner.decide(320,640,230,mirrored=True),'TURN_RIGHT')
        self.assertEqual(planner.decide(None,640,mirrored=True),'TURN_RIGHT')
        self.assertEqual(planner.decide(320,640,330),'UP_LITTLE')

    def test_near_and_far_bands_are_separate(self):
        image=np.zeros((480,640,3),np.uint8)
        image[384:,310:330]=255
        image[264:360,400:420]=255
        self.assertAlmostEqual(detect_line(image),319.5)
        self.assertAlmostEqual(detect_line(image,band=(.55,.75)),409.5)

    def make_io(self):
        io=LineCompetitionIO(load_settings(CONFIG_FILE),Mock(),float('inf'))
        io.open_views=Mock()
        io.stream=Mock()
        frame=np.zeros((480,640,3),np.uint8)
        io.observe_ready=Mock(return_value=(frame,frame,{}))
        io.move=Mock()
        io.set_head=Mock()
        return io

    @patch('competition_line_main.cv2.imshow')
    def test_initial_missing_line_stops_instead_of_carry(self, show):
        io=self.make_io()
        with self.assertRaisesRegex(RuntimeError,'向右搜索4次'):
            io.follow_to_carry('red')
        self.assertEqual(io.move.call_count,4)

    @patch('competition_line_main.cv2.imshow')
    def test_forward_then_line_loss_does_not_enter_carry(self, show):
        io=self.make_io()
        with patch('competition_line_main.detect_line',side_effect=[320,None]+[None]*15):
            with self.assertRaisesRegex(RuntimeError,'向右搜索4次'):
                io.follow_to_carry('red')
        self.assertEqual(io.move.call_args_list[0].args,('UP_LITTLE',))
        self.assertEqual(io.move.call_count,5)

    @patch('competition_line_main.cv2.imshow')
    def test_same_color_block_confirms_before_any_search_motion(self, show):
        io=self.make_io()
        with patch('competition_line_main.BlockTracker') as tracker:
            tracker.return_value.update.return_value=Box(100,100,60,60)
            io.follow_to_carry('red')
        io.move.assert_not_called()
        self.assertEqual(io.observe_ready.call_count,3)
        io.set_head.assert_called_once_with(127)

    @patch('competition_line_main.cv2.imshow')
    def test_different_blocks_do_not_accumulate_confirmation(self, show):
        io=self.make_io()
        boxes=[[Box(10 if i%2 else 200,100,60,60)] for i in range(30)]
        with patch('competition_line_main.BlockTracker') as tracker:
            tracker.return_value.update.side_effect=[items[0] for items in boxes]
            with self.assertRaisesRegex(RuntimeError,'动作上限'):
                io.follow_to_carry('red')
        io.move.assert_not_called()

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
