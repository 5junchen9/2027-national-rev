"""无摄像头、无串口的路线解析检查：python3 test_qr_routes.py。"""
import unittest
import cv2
import numpy as np
from qr_main import format_action_plan, parse_route
from qr_route_image import _left_middle_start, extract_route


def drawing(points):
    image = np.full((600, 900, 3), 255, np.uint8)
    cv2.putText(image, "route", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.polylines(image, [np.array(points)], False, (0, 0, 0), 5)
    return image


class RouteTests(unittest.TestCase):
    def test_left_middle_start(self):
        nodes = [(20, 40), (70, 300)]
        self.assertEqual(_left_middle_start(nodes, [0, 1], 900, 600, 10), 1)

    def test_shapes_and_reverse(self):
        route1 = drawing([(100, 450), (400, 450), (400, 200), (800, 200), (800, 450)])
        text, _ = extract_route(route1)
        self.assertEqual(text.split(",")[1::2], ["L90", "R90", "R90"])
        self.assertEqual([int(s[1:]) for s in text.split(",")[::2]], [6, 5, 8, 5])
        reverse, _ = extract_route(route1, reverse=True)
        self.assertEqual(reverse.split(",")[1::2], ["L90", "L90", "R90"])
        route2 = drawing([(100, 500), (800, 500), (800, 100), (300, 100), (300, 350)])
        self.assertEqual(extract_route(route2)[0].split(",")[1::2], ["L90"] * 3)

    def test_reject_ambiguous_image(self):
        for image in (np.full((400, 400, 3), 255, np.uint8),
                      drawing([(100, 100), (700, 100), (700, 500), (100, 500), (100, 100)])):
            with self.assertRaises(ValueError):
                extract_route(image)

    def test_action_bounds_and_calibration(self):
        self.assertEqual(parse_route("L90,R90"),
                         ["TURN_LEFT"] * 7 + ["TURN_RIGHT"] * 7)
        self.assertEqual(parse_route("F2,L90,R90", 2, 3), ["UP"] * 2 + ["TURN_LEFT"] * 2 + ["TURN_RIGHT"] * 3)
        self.assertEqual(format_action_plan(parse_route("F2,L90,R90", 2, 3)),
                         "UPx2 TURN_LEFTx2 TURN_RIGHTx3")
        for value in ("", "F0", "F99999999999", "left", "F2,Q90", "F1," , "F100," * 6 + "F1"):
            with self.assertRaises(ValueError):
                parse_route(value)


if __name__ == "__main__":
    unittest.main()
