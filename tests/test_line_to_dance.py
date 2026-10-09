"""后半程测试只验证流程调用，不连接真实硬件。"""
import unittest
from unittest.mock import Mock, call, patch

from competition_main import CONFIG_FILE, load_settings
from competition_line_main import main
from line_search_route import run_from_line, run_from_carry


class LineToDanceTests(unittest.TestCase):
    def make_io(self):
        io = Mock()
        io.settings = load_settings(CONFIG_FILE)
        io.carry.return_value = True
        return io

    def test_normal_chain_starts_at_line_and_enters_sport_directly(self):
        io = self.make_io()
        self.assertTrue(run_from_line(io, "blue"))
        actions = [item for item in io.method_calls if item[0] != "phase"]
        self.assertEqual(actions, [
            call.follow_to_carry("blue"),
            call.carry("blue", "action2"),
            call.sport(), call.find_dance(), call.forward(10), call.dance(),
        ])

    def test_scan_limit_continues_football_then_dance(self):
        io = self.make_io()
        io.carry.return_value = "scan_limit"
        self.assertTrue(run_from_line(io, "red"))
        io.align_football.assert_not_called()
        io.sport.assert_called_once()
        io.find_dance.assert_called_once()
        io.forward.assert_called_once_with(10)
        io.dance.assert_called_once()

    def test_carry_start_skips_line_and_enters_sport_directly(self):
        io = self.make_io()
        self.assertTrue(run_from_carry(io, "red"))
        actions = [item for item in io.method_calls if item[0] != "phase"]
        self.assertEqual(actions, [
            call.carry("red", "action2"), call.sport(), call.find_dance(), call.forward(10), call.dance(),
        ])

    def test_carry_start_simulation_selects_carry_course(self):
        with patch("sys.argv", ["competition_line.py", "--from-carry", "--simulate"]), \
                patch("competition_line_main.run_from_carry") as carry, \
                patch("competition_line_main.run_course") as full:
            self.assertTrue(main())
        carry.assert_called_once()
        full.assert_not_called()

    def test_carry_start_rejects_conflicting_start(self):
        with patch("sys.argv", ["competition_line.py", "--from-carry", "--from-line", "--simulate"]):
            with self.assertRaises(SystemExit):
                main()

    def test_sport_failure_prevents_dance(self):
        io = self.make_io()
        io.sport.side_effect = RuntimeError("足球阶段失败")
        with self.assertRaisesRegex(RuntimeError, "足球阶段失败"):
            run_from_line(io, "red")
        io.align_football.assert_not_called()
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
