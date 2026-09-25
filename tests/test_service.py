from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skillset import service


class ServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config with % and $.json"
        self.config.write_text(json.dumps({"schema_version": 1, "sets": {}}))
        self.unit = self.root / "systemd/user/skillset-watch.service"
        self.calls = []
        for context in (
            patch.object(service, "unit_path", return_value=self.unit),
            patch.object(service.sys, "platform", "linux"),
            patch.object(service, "_systemctl", side_effect=self.control),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def control(self, *args):
        self.calls.append(args)
        return 0

    def test_install_enable_and_repeat_without_rewriting(self):
        self.assertEqual(service.manage_service("install", self.config), 0)
        self.assertTrue(self.unit.read_text().startswith(service.MARKER))
        self.assertIn("%% and $$", self.unit.read_text())
        self.assertIn(("enable", "--now", service.UNIT_NAME), self.calls)
        before = self.unit.stat().st_mtime_ns
        self.assertEqual(service.manage_service("install", self.config), 0)
        self.assertEqual(self.unit.stat().st_mtime_ns, before)

    def test_foreign_unit_or_symlink_is_preserved(self):
        self.unit.parent.mkdir(parents=True)
        self.unit.write_text("[Service]\nExecStart=/bin/true\n")
        for action in ("install", "uninstall", "stop"):
            self.assertEqual(service.manage_service(action, self.config), 3)
        self.assertEqual(self.calls, [])
        self.unit.unlink()
        self.unit.symlink_to(self.config)
        self.assertEqual(service.manage_service("install", self.config), 3)
        self.assertTrue(self.unit.is_symlink())

    def test_uninstall_stops_and_removes_only_owned_unit(self):
        self.assertEqual(service.manage_service("install", self.config), 0)
        self.calls.clear()
        self.assertEqual(service.manage_service("uninstall", self.config), 0)
        self.assertEqual(self.calls, [("disable", "--now", service.UNIT_NAME), ("daemon-reload",)])
        self.assertFalse(self.unit.exists())
        self.assertTrue(self.config.exists())
        self.assertEqual(service.manage_service("uninstall", self.config), 0)

    def test_config_update_restarts_running_unit(self):
        self.assertEqual(service.manage_service("install", self.config), 0)
        alternate = self.root / "alternate.json"
        alternate.write_bytes(self.config.read_bytes())
        self.assertEqual(service.manage_service("install", alternate), 0)
        self.assertIn(("restart", service.UNIT_NAME), self.calls)

    def test_bad_config_and_unavailable_systemd_report_errors(self):
        self.assertEqual(service.manage_service("install", self.root / "missing.json"), 2)
        self.assertFalse(self.unit.exists())
        with patch.object(service, "_systemctl", side_effect=FileNotFoundError("systemctl")):
            self.assertEqual(service.manage_service("install", self.config), 1)

    def test_status_passes_inactive_exit_and_other_os_is_unsupported(self):
        with patch.object(service, "_systemctl", return_value=3):
            self.assertEqual(service.manage_service("status", self.config), 3)
        with patch.object(service.sys, "platform", "darwin"):
            self.assertEqual(service.manage_service("install", self.config), 2)
