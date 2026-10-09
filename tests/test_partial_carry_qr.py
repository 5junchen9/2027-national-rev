import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from competition_qr import partial_edge_qr
from carry_vision import scan_while_moving_right
from kick_shapes import Box


def finder(frame,x,y,size=70):
    cv2.rectangle(frame,(x,y),(x+size-1,y+size-1),(0,0,0),-1)
    cv2.rectangle(frame,(x+10,y+10),(x+size-11,y+size-11),(255,255,255),-1)
    cv2.rectangle(frame,(x+20,y+20),(x+size-21,y+size-21),(0,0,0),-1)


class PartialQRTests(unittest.TestCase):
    def test_blank_and_isolated_finder_do_not_trigger(self):
        frame=np.full((300,400,3),255,np.uint8)
        self.assertIsNone(partial_edge_qr(frame))
        finder(frame,30,220)
        self.assertIsNone(partial_edge_qr(frame))

    def test_two_edge_finders_trigger_but_centered_pair_does_not(self):
        frame=np.full((1000,1200,3),255,np.uint8)
        finder(frame,400,450);finder(frame,600,450)
        self.assertIsNone(partial_edge_qr(frame))
        edge=np.full((300,400,3),255,np.uint8)
        finder(edge,40,220);finder(edge,200,220)
        self.assertIsNotNone(partial_edge_qr(edge))

    def test_checkerboard_without_finder_does_not_trigger(self):
        frame=np.full((300,400,3),255,np.uint8)
        for x in range(0,200,20):
            for y in range(180,300,20):
                cv2.rectangle(frame,(x,y),(x+9,y+9),(0,0,0),-1)
        self.assertIsNone(partial_edge_qr(frame))

    def test_single_edge_finder_with_module_texture_is_accepted(self):
        frame=np.full((300,400,3),255,np.uint8)
        finder(frame,200,220)
        for x in (110,130,150):
            for y in (220,240,260):
                cv2.rectangle(frame,(x,y),(x+9,y+9),(0,0,0),-1)
        self.assertIsNotNone(partial_edge_qr(frame))

    def test_clipped_ring_without_center_block_needs_adjacent_modules(self):
        frame=np.full((300,400,3),255,np.uint8)
        # 底边出画、中心黑块嵌套断开，仍可见外框和内白孔。
        cv2.rectangle(frame,(200,240),(269,309),(0,0,0),-1)
        cv2.rectangle(frame,(210,250),(259,290),(255,255,255),-1)
        self.assertIsNone(partial_edge_qr(frame))
        for x in (110,130,150):
            for y in (240,260):
                cv2.rectangle(frame,(x,y),(x+9,y+9),(0,0,0),-1)
        self.assertIsNotNone(partial_edge_qr(frame))

    def run_scan(self,boxes,contents=None,quit_after=False):
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        frame=np.zeros((480,640,3),np.uint8)
        def image():
            time.sleep(.002)
            return True,frame
        head.getImage.side_effect=belly.getImage.side_effect=image
        servo.is_moving.return_value=False
        observed=[];done=threading.Event()
        def partial(frame):
            index=len(observed)
            box=boxes[index] if index<len(boxes) else None
            observed.append(box)
            if index>=len(boxes)-1:
                done.set()
            return box
        decoder=Mock(return_value=[] if contents is None else contents)
        def key(_):
            return ord('q') if quit_after and done.is_set() else -1
        with patch('competition_qr.partial_edge_qr',side_effect=partial) as detect, \
             patch('carry_vision.cv2.imshow'),patch('carry_vision.cv2.waitKey',side_effect=key):
            result=scan_while_moving_right(robot,head,belly,servo,'action2',
                     dict(shapes=dict(head=[480,640],belly=[480,640])),decoder,
                     time.monotonic()+.5,7)
        return result,robot,observed,detect

    def test_one_detection_releases_once(self):
        result,robot,observed,_=self.run_scan([Box(10,400,100,70)])
        self.assertIs(result,True)
        self.assertGreaterEqual(len(observed),1)
        robot.robotMove.assert_called_once_with('DOWN_BOX')

    def test_no_detection_does_not_release(self):
        result,robot,_,_=self.run_scan([None],quit_after=True)
        self.assertIs(result,False)
        robot.robotMove.assert_not_called()

    def test_decoded_other_qr_never_uses_partial_condition(self):
        result,robot,_,detect=self.run_scan([],contents=[('dance',None)])
        self.assertIs(result,False)
        detect.assert_not_called()
        robot.robotMove.assert_not_called()


if __name__ == '__main__':
    unittest.main()
