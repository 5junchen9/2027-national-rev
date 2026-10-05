import unittest
from unittest.mock import patch
import cv2
import numpy as np
from kick_shapes import Box
from goal_debug import GoalTracker, find_goal_frames


class GoalTrackerTests(unittest.TestCase):
    def test_u_shaped_goal_is_confirmed_over_five_frames(self):
        frame=np.full((480,640,3),255,np.uint8)
        cv2.line(frame,(150,230),(150,60),(0,0,0),7)
        cv2.line(frame,(150,60),(440,60),(0,0,0),7)
        cv2.line(frame,(440,60),(440,230),(0,0,0),7)
        tracker=GoalTracker()
        for _ in range(5): box=tracker.update(frame)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box.cx,295,delta=8)
        self.assertEqual(tracker.stable_frames,5)

    def test_missing_multiple_or_jumping_goal_cannot_keep_confirmation(self):
        frame=np.zeros((480,640,3),np.uint8)
        tracker=GoalTracker()
        center=Box(190,60,250,160)
        with patch('goal_debug.find_goal_frames',return_value=[center]):
            for _ in range(5): tracker.update(frame)
        with patch('goal_debug.find_goal_frames',return_value=[Box(20,60,120,100)]):
            tracker.update(frame)
        self.assertEqual(tracker.stable_frames,1)
        with patch('goal_debug.find_goal_frames',return_value=[center,Box(20,60,120,100)]):
            self.assertIsNone(tracker.update(frame))
        self.assertEqual(tracker.stable_frames,0)
        with patch('goal_debug.find_goal_frames',return_value=[]):
            self.assertIsNone(tracker.update(frame))
        tracker.reset()
        self.assertIsNone(tracker.box)

    def test_ground_outline_without_two_upright_posts_is_rejected(self):
        frame=np.full((480,640,3),255,np.uint8)
        cv2.line(frame,(80,400),(150,270),(0,0,0),5)
        cv2.line(frame,(150,270),(460,270),(0,0,0),5)
        cv2.line(frame,(460,270),(560,400),(0,0,0),5)
        cv2.line(frame,(560,400),(80,400),(0,0,0),5)
        self.assertEqual(find_goal_frames(frame),[])
