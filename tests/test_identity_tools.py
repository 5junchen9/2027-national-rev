import unittest
from unittest.mock import Mock, patch
import numpy as np
from identity_tools import read_local_name, tracked_head_position
from competition_main import CompetitionIO, load_settings, CONFIG_FILE
from line_search_route import detect_line, line_mask


class IdentityImprovements(unittest.TestCase):
    def test_local_ocr_is_enlarged_and_rejects_weak_or_non_name_text(self):
        image=np.zeros((480,640,3),np.uint8)
        ocr=Mock()
        ocr._detect_pairs.return_value=[('错误',.6),('ABC张',.99),('姓名：李娜',.91)]
        self.assertEqual(read_local_name(ocr,image,(100,100,80,80)),('李娜',.91))
        self.assertEqual(ocr._detect_pairs.call_args.args[0].shape,(360,480,3))
        ocr._detect_pairs.assert_called_once()

    def test_no_full_frame_fallback_when_label_is_outside_image(self):
        ocr=Mock()
        self.assertEqual(read_local_name(ocr,np.zeros((100,100,3),np.uint8),(20,80,20,20)),(None,None))
        ocr._detect_pairs.assert_not_called()

    def test_tracking_steps_one_unit_and_stays_inside_six_unit_band(self):
        shape=(480,640,3)
        self.assertEqual(tracked_head_position(131,131,(100,350,80,80),shape),132)
        self.assertEqual(tracked_head_position(137,131,(100,350,80,80),shape),137)
        self.assertEqual(tracked_head_position(125,131,(100,0,80,80),shape),125)
        self.assertEqual(tracked_head_position(131,131,None,shape),131)

    def test_unconfirmed_identity_resumes_route_instead_of_aborting(self):
        io=CompetitionIO(load_settings(CONFIG_FILE),Mock(),float('inf'))
        io.stop_route=Mock();io.resume_route=Mock();io.check_time=Mock()
        io.head_eye=Mock();io.servo=Mock()
        with patch('competition_identity.recognize',return_value=False):
            io.identity()
        io.resume_route.assert_called_once()

    def test_qr_polygon_is_removed_even_without_decoding(self):
        image=np.zeros((480,640,3),np.uint8)
        image[360:,250:350]=255
        detector=Mock()
        detector.detectMulti.return_value=(True,np.array([[[250,360],[350,360],[350,479],[250,479]]],np.float32))
        with patch('line_search_route.cv2.QRCodeDetector',return_value=detector):
            mask=line_mask(image)
        self.assertEqual(np.count_nonzero(mask),0)

    def test_short_glare_patch_and_far_away_line_are_rejected(self):
        image=np.zeros((480,640,3),np.uint8)
        image[400:420,250:300]=255
        self.assertIsNone(detect_line(image))
        image[:]=0
        image[384:,500:520]=255
        self.assertIsNone(detect_line(image,previous_x=320))
        self.assertIsNotNone(detect_line(image))


if __name__ == '__main__':
    unittest.main()
