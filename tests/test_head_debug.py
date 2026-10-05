import unittest
from head_debug import HeadPosition


class HeadDebugTests(unittest.TestCase):
    def test_bounds_and_save_require_actual_references(self):
        p = HeadPosition()
        self.assertFalse(p.active)
        with self.assertRaises(ValueError): p.profile({}, {})
        p.forward = 95; p.down_sign = 1
        p.command = 115
        self.assertEqual(p.candidate(1),115)
        p.handover = 116
        with self.assertRaises(ValueError): p.profile({}, {})
        p.handover = 110
        self.assertEqual(p.profile({}, {})['handover'],110)
        p.forward = 140; p.down_sign = -1; p.command = 120; p.handover = 120
        self.assertEqual(p.candidate(-1),120)
        self.assertEqual(p.profile({}, {})['bounds'],[120,180])
        with self.assertRaises(ValueError): HeadPosition(45)

    def test_field_direction_and_manual_overlap_preserve_provenance(self):
        p = HeadPosition(117)
        self.assertEqual(p.down_sign,1)
        p.forward = 117; p.command = 127
        with self.assertRaises(ValueError): p.record_handover('manual')
        p.active = True
        p.record_handover('operator_visual_confirmation; shape detection incomplete')
        profile = p.profile({}, {})
        self.assertEqual((profile['forward'],profile['handover']), (117,127))
        self.assertIn('operator_visual_confirmation',profile['overlap_confirmation'])
        p.command = 138
        with self.assertRaises(ValueError): p.record_handover('manual')
