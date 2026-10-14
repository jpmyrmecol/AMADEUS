"""Monitor selection regression tests; no graphical display required."""

import subprocess
import unittest
from unittest.mock import patch

from gui.splash_geometry import (
    MonitorBounds,
    configure_splash_window,
    linux_primary_monitor,
    parse_xrandr_monitors,
)


class MonitorSelectionTests(unittest.TestCase):
    SAMPLE = (
        "Monitors: 2\n"
        " 0: +HDMI-1 1920/510x1080/287+0+0  HDMI-1\n"
        " 1: +*DP-1 2560/600x1440/340+1920+0  DP-1\n"
    )

    def test_parse_and_primary_selection(self):
        expected = (
            MonitorBounds(1920, 1080, 0, 0),
            MonitorBounds(2560, 1440, 1920, 0, True),
        )
        self.assertEqual(parse_xrandr_monitors(self.SAMPLE), expected)
        with patch("gui.splash_geometry.os.environ", {"XDG_SESSION_TYPE": "x11"}), patch(
            "gui.splash_geometry.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, self.SAMPLE, ""),
        ):
            self.assertEqual(linux_primary_monitor(), expected[1])

    def test_single_monitor_without_physical_dimensions(self):
        self.assertEqual(
            parse_xrandr_monitors("Monitors: 1\n 0: +*eDP-1 1920x1200+0+0 eDP-1\n"),
            (MonitorBounds(1920, 1200, 0, 0, True),),
        )

    def test_wayland_skips_xrandr(self):
        with patch("gui.splash_geometry.os.environ", {"XDG_SESSION_TYPE": "wayland"}), patch(
            "gui.splash_geometry.subprocess.run"
        ) as run:
            self.assertIsNone(linux_primary_monitor())
            run.assert_not_called()

    def test_xrandr_failure_requests_monitor_scoped_fullscreen(self):
        class Window:
            def __init__(self):
                self.actions = []
            def attributes(self, *args):
                self.actions.append(("attributes", args))
            def update(self):
                self.actions.append(("update",))
            def winfo_width(self):
                return 1920
            def winfo_height(self):
                return 1080

        window = Window()
        with patch("gui.splash_geometry.sys.platform", "linux"), patch(
            "gui.splash_geometry.linux_primary_monitor", return_value=None
        ):
            self.assertEqual(configure_splash_window(window), (1920, 1080))
        self.assertEqual(window.actions, [("attributes", ("-fullscreen", True)), ("update",)])

    def test_negative_monitor_offset_uses_window_manager(self):
        class Window:
            def __init__(self):
                self.actions = []
            def attributes(self, *args):
                self.actions.append(("attributes", args))
            def update(self):
                self.actions.append(("update",))
            def winfo_width(self):
                return 1920
            def winfo_height(self):
                return 1080

        window = Window()
        with patch("gui.splash_geometry.sys.platform", "linux"), patch(
            "gui.splash_geometry.linux_primary_monitor",
            return_value=MonitorBounds(1920, 1080, -1920, 0, True),
        ):
            self.assertEqual(configure_splash_window(window), (1920, 1080))
        self.assertEqual(window.actions, [
            ("attributes", ("-fullscreen", True)), ("update",)
        ])

    def test_borderless_window_uses_primary_monitor_offset(self):
        class Window:
            def __init__(self):
                self.actions = []
            def overrideredirect(self, value):
                self.actions.append(("overrideredirect", value))
            def geometry(self, value):
                self.actions.append(("geometry", value))

        window = Window()
        with patch("gui.splash_geometry.sys.platform", "linux"), patch(
            "gui.splash_geometry.linux_primary_monitor",
            return_value=MonitorBounds(2560, 1440, 1920, 0, True),
        ):
            self.assertEqual(configure_splash_window(window), (2560, 1440))
        self.assertEqual(window.actions, [
            ("overrideredirect", True), ("geometry", "2560x1440+1920+0")
        ])


if __name__ == "__main__":
    unittest.main()
