"""Independent acceptance checks for automatic folder scopes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AutoCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="skillset-auto-cli-")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.env = os.environ | {
            "HOME": str(self.temp), "XDG_CONFIG_HOME": str(self.temp / "config"),
            "XDG_STATE_HOME": str(self.temp / "state"), "CODEX_HOME": str(self.temp / "codex"),
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        }
        self.area = self.temp / "clients"
        self.area.mkdir()
        self.source = self.temp / "source"
        self.source.mkdir()
        (self.source / "SKILL.md").write_text("# test skill\n")
        self.config = self.temp / "settings.json"
        self.config.write_text(json.dumps({"schema_version": 1, "sets": {
            "company": {"agents": ["codex", "claude"], "skills": {"company-tool": str(self.source)}},
            "personal": {"agents": ["codex"], "skills": {}},
        }}))

    def cli(self, *args, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "skillset", *map(str, args), "--config", str(self.config)],
            cwd=self.temp, env=self.env, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def git(self, *args, cwd=None):
        return subprocess.run(["git", *map(str, args)], cwd=cwd, env=self.env,
                              capture_output=True, text=True, check=True)

    def repo(self, name):
        path = self.area / name
        self.git("init", "-q", path)
        return path

    def register(self):
        self.cli("roots", "add", self.area, "--set", "company")

    def test_registration_sync_exclusion_and_no_automatic_removal(self):
        project = self.repo("project")
        self.register()
        self.assertFalse((project / ".skillset").exists())
        self.cli("sync", "--dry-run")
        self.assertFalse((project / ".skillset").exists())
        self.cli("sync")
        for base in (".agents/skills", ".claude/skills"):
            self.assertEqual(os.readlink(project / base / "company-tool"), str(self.source))
        self.assertEqual(self.git("status", "--porcelain", cwd=project).stdout, "")
        self.assertFalse((project / ".gitignore").exists())
        excludes = project / ".git/info/exclude"
        before = excludes.read_bytes()
        self.cli("sync")
        self.assertEqual(excludes.read_bytes(), before)
        self.cli("roots", "remove", self.area)
        self.cli("sync")
        self.assertTrue((project / ".agents/skills/company-tool").is_symlink())

    def test_nested_disabled_scope_and_manual_set_are_respected(self):
        personal = self.repo("personal")
        excluded = self.repo("disabled")
        ordinary = self.repo("ordinary")
        self.register()
        self.cli("roots", "disable", excluded)
        self.cli("apply", "personal", "--project-root", personal)
        before = (personal / ".skillset/state.json").read_bytes()
        self.cli("sync", expected=3)
        self.assertEqual((personal / ".skillset/state.json").read_bytes(), before)
        self.assertFalse((excluded / ".skillset").exists())
        self.assertTrue((ordinary / ".agents/skills/company-tool").is_symlink())

    @unittest.skipUnless(sys.platform == "linux", "native event monitoring requires Linux")
    def test_watch_detects_new_worktree_and_config_opt_out(self):
        self.register()
        log = self.temp / "watch.log"
        with log.open("w") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", "skillset", "watch", "--config", str(self.config)],
                cwd=self.temp, env=self.env, stdout=output, stderr=subprocess.STDOUT,
            )
            def stop():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            self.addCleanup(stop)
            def until(predicate):
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    if predicate():
                        return
                    if process.poll() is not None:
                        self.fail(log.read_text())
                    time.sleep(0.05)
                self.fail("watch timeout:\n" + log.read_text())
            until(lambda: "watch: ready" in log.read_text())
            main = self.repo("main")
            until(lambda: (main / ".agents/skills/company-tool").is_symlink())
            self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                     "commit", "--allow-empty", "-qm", "initial", cwd=main)
            worktree = self.area / "new-worktree"
            self.git("worktree", "add", "-qb", "work", worktree, cwd=main)
            until(lambda: (worktree / ".claude/skills/company-tool").is_symlink())
            self.assertEqual(self.git("status", "--porcelain", cwd=worktree).stdout, "")
            disabled_parent = self.area / "private"
            disabled_parent.mkdir()
            self.cli("roots", "disable", disabled_parent)
            # A second CLI sync observes the same config immediately; the
            # watcher must also re-read before any queued candidate mutation.
            private_repo = self.repo("private/new")
            self.cli("sync", "--project-root", private_repo)
            time.sleep(0.6)
            self.assertFalse((private_repo / ".skillset").exists(), log.read_text())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 0, log.read_text())
