import json
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from carry_vision import find_blocks, CarryPlanner
from qr_geometry import qr_box, view_matches, quad_size_ratios
from competition_qr import read_codes
from kick_shapes import Box


class CarryGeometryTests(unittest.TestCase):
    def reference(self):
        box=qr_box([(240,160),(400,160),(400,320),(240,320)]).normalized((480,640))
        return box,dict(drop_camera='belly',drop=[box.x,box.y,box.width,box.height],
                        drop_quad=box.corners,shapes=dict(belly=[480,640],head=[480,640]))

    def test_quadrilateral_survives_normalization_and_json(self):
        box,reference=self.reference()
        restored=json.loads(json.dumps(reference))
        self.assertTrue(view_matches(box,restored,'belly'))
        np.testing.assert_allclose(quad_size_ratios(box,restored,'belly'),[1,1])

    def test_different_perspective_cannot_release_even_with_matching_bbox(self):
        box,reference=self.reference()
        slanted=qr_box([(285,160),(355,160),(400,320),(240,320)]).normalized((480,640))
        self.assertFalse(view_matches(slanted,reference,'belly'))
        planner=CarryPlanner(reference,'none');planner.phase='DELIVER'
        self.assertEqual([planner.decide(slanted,5) for _ in range(4)],['WAIT']*4)
        self.assertEqual(planner.decide(Box(box.x,box.y,box.width,box.height),5),'WAIT')

    def test_matching_geometry_releases_only_after_confirmation(self):
        box,reference=self.reference()
        planner=CarryPlanner(reference,'none');planner.phase='DELIVER'
        self.assertEqual([planner.decide(box,5) for _ in range(3)],['WAIT','WAIT','DOWN_BOX'])

    def test_same_slanted_view_can_be_calibrated(self):
        box=qr_box([(285,160),(355,160),(400,320),(240,320)]).normalized((480,640))
        reference=dict(drop_camera='belly',drop=[box.x,box.y,box.width,box.height],
                       drop_quad=box.corners,shapes=dict(belly=[480,640]))
        planner=CarryPlanner(reference,'none');planner.phase='DELIVER'
        self.assertEqual([planner.decide(box,5) for _ in range(3)],['WAIT','WAIT','DOWN_BOX'])

    def test_smaller_same_shape_is_approached(self):
        _,reference=self.reference()
        box=qr_box([(280,200),(360,200),(360,280),(280,280)]).normalized((480,640))
        planner=CarryPlanner(reference,'none');planner.phase='DELIVER'
        self.assertEqual([planner.decide(box,5) for _ in range(2)],['WAIT','UP_HOLDBOX'])

    def test_blue_box_can_be_found_when_color_mask_merges_with_floor(self):
        hsv=np.full((480,640,3),(115,180,95),np.uint8)
        hsv[180:300,250:390]=(105,130,220)
        image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        boxes=find_blocks(image,'blue')
        self.assertTrue(any(abs(box.cx-320)<10 and abs(box.cy-240)<10 for box in boxes))
        flat=cv2.cvtColor(np.full((480,640,3),(115,180,95),np.uint8),cv2.COLOR_HSV2BGR)
        self.assertEqual(find_blocks(flat,'blue'),[])

    def test_small_glare_on_blue_floor_is_not_a_blue_box(self):
        hsv=np.full((480,640,3),(115,180,95),np.uint8)
        hsv[180:300,250:390]=(0,0,255)
        self.assertEqual(find_blocks(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),'blue'),[])

    def test_decoded_polygon_preserves_exact_text_and_corners(self):
        import types
        points=[types.SimpleNamespace(x=x,y=y) for x,y in [(20,20),(60,20),(60,60),(20,60)]]
        code=types.SimpleNamespace(data=b'action2',rect=(20,20,40,40),polygon=points)
        module=types.SimpleNamespace(decode=Mock(return_value=[code]),ZBarSymbol=types.SimpleNamespace(QRCODE='QR'))
        with patch.dict('sys.modules',{'pyzbar.pyzbar':module}):
            decoded=read_codes(np.zeros((100,100,3),np.uint8))
        self.assertEqual(decoded[0][0],'action2')
        self.assertEqual(len(decoded[0][1].corners),4)


if __name__=='__main__':
    unittest.main()
