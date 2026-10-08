"""后半程测试只验证流程调用，不连接真实硬件。"""
import unittest
from unittest.mock import Mock, call, patch

from competition_main import CONFIG_FILE, load_settings
from competition_line_main import main
from line_search_route import run_from_line


class LineToDanceTests(unittest.TestCase):
    def make_io(self):
        io = Mock()
        io.settings = load_settings(CONFIG_FILE)
        io.carry.return_value = True
        return io

    def test_normal_chain_starts_at_line_and_aligns_before_sport(self):
        io = self.make_io()
        self.assertTrue(run_from_line(io, "blue"))
        actions = [item for item in io.method_calls if item[0] != "phase"]
        self.assertEqual(actions, [
            call.follow_to_carry("blue"),
            call.carry("blue", "action2"),
            call.align_football(), call.sport(), call.right(7),
            call.set_head(120), call.scan_until("dance", "head", confirm_frames=2),
            call.enter_blue(), call.dance(),
        ])

    def test_scan_limit_skips_football_and_continues_to_dance(self):
        io = self.make_io()
        io.carry.return_value = "scan_limit"
        self.assertTrue(run_from_line(io, "red"))
        io.align_football.assert_not_called()
        io.sport.assert_not_called()
        io.right.assert_not_called()  # 超限右转由搬运模块执行，不重复转。
        io.dance.assert_called_once()

    def test_alignment_failure_prevents_sport_and_dance(self):
        io = self.make_io()
        io.align_football.side_effect = RuntimeError("足球区对齐失败")
        with self.assertRaisesRegex(RuntimeError, "足球区对齐失败"):
            run_from_line(io, "red")
        io.sport.assert_not_called()
        io.dance.assert_not_called()

    def test_simulation_selects_partial_course(self):
        with patch("sys.argv", ["competition_line.py", "--from-line", "--simulate"]), \
                patch("competition_line_main.run_from_line") as partial, \
                patch("competition_line_main.run_course") as full:
            self.assertTrue(main())
        partial.assert_called_once()
        full.assert_not_called()

    def test_real_chain_requires_actions_flag(self):
        with patch("sys.argv", ["competition_line.py", "--from-line", "--run"]), \
                patch("competition_line_main.preflight") as preflight:
            with self.assertRaises(SystemExit):
                main()
        preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
