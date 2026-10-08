import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from football_align import find_court, alignment_action, run
from competition_route import run_course
from line_search_route import run_course as run_line_course


class FootballAlignTests(unittest.TestCase):
    def court(self):
        image = np.full((480, 640, 3), (110, 40, 20), np.uint8)
        cv2.rectangle(image, (155, 45), (485, 435), (240, 240, 240), 3)
        cv2.rectangle(image, (170, 60), (470, 420), (240, 240, 240), 3)
        cv2.line(image, (170, 240), (470, 240), (240, 240, 240), 3)
        cv2.ellipse(image, (320, 240), (40, 34), 0, 0, 360, (240, 240, 240), 3)
        return image

    def test_inner_border_detected_with_midfield(self):
        quad = find_court(self.court())
        self.assertIsNotNone(quad)
        self.assertAlmostEqual(quad[:, 0].min(), 170, delta=4)
        self.assertEqual(alignment_action(quad, (480, 640, 3)), 'DONE')

    def test_routes_rectangles_and_qr_like_boxes_are_rejected(self):
        image = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(image, (155, 45), (485, 435), (255, 255, 255), 3)
        cv2.rectangle(image, (170, 60), (470, 420), (255, 255, 255), 3)
        cv2.line(image, (170, 240), (470, 240), (255, 255, 255), 3)
        self.assertIsNone(find_court(image))
        self.assertIsNone(find_court(np.full_like(image, 255)))

    def test_both_routes_align_before_sport_and_skip_on_scan_limit(self):
        settings = dict(face_qr='face', factory_qr='left', carry_qr='action1',
                        drop_qr='action2', dance_qr='dance', left_right_actions=3,
                        action1_right_actions=3, after_sport_right_actions=7,
                        dance_head_position=120)
        for course in (run_course, run_line_course):
            io = Mock(); io.settings = settings
            course(io, 'red')
            calls = [call[0] for call in io.method_calls]
            self.assertLess(calls.index('carry'), calls.index('align_football'))
            self.assertLess(calls.index('align_football'), calls.index('sport'))
            io = Mock(); io.settings = settings; io.carry.return_value = 'scan_limit'
            course(io, 'red')
            io.align_football.assert_not_called()
            io.sport.assert_not_called()
            io = Mock(); io.settings = settings
            io.align_football.side_effect = RuntimeError('not confirmed')
            with self.assertRaises(RuntimeError): course(io, 'red')
            io.sport.assert_not_called()

    def test_alignment_confirmation_and_lost_border_stop(self):
        quad = find_court(self.court())
        io = Mock(); io.settings = dict(sport_head_position=125); io.deadline = float('inf')
        io.observe_ready.return_value = (self.court(), None, None)
        with patch('football_align.cv2.imshow'):
            self.assertTrue(run(io))
        io.move.assert_not_called()
        with patch('football_align.cv2.imshow'), patch('football_align.find_court', side_effect=[quad, None, None, None]):
            with self.assertRaises(RuntimeError): run(io)
        io.move.assert_not_called()

    def test_heading_correction_and_perspective_detection(self):
        quad = np.float32([[350, 100], [550, 100], [520, 420], [120, 420]])
        self.assertEqual(alignment_action(quad, (480, 640)), 'TURN_RIGHT')
        self.assertEqual(alignment_action(quad, (480, 640), True), 'TURN_LEFT')
        source = np.float32([[155,45], [485,45], [485,435], [155,435]])
        target = np.float32([[210,70], [440,90], [500,430], [120,410]])
        matrix = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(self.court(), matrix, (640,480))
        self.assertIsNotNone(find_court(image))

    def test_translation_mapping_and_mirror(self):
        quad = np.float32([[300, 50], [600, 50], [600, 420], [300, 420]])
        self.assertEqual(alignment_action(quad, (480, 640)), 'SIDE_LEFT')
        self.assertEqual(alignment_action(quad, (480, 640), True), 'SIDE_RIGHT')


if __name__ == '__main__':
    unittest.main()
