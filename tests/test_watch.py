from __future__ import annotations

import fcntl
import contextlib
import io
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

from skillset.inotify import (
    IN_CREATE,
    IN_DONT_FOLLOW,
    IN_ISDIR,
    IN_ONLYDIR,
    IN_Q_OVERFLOW,
    INOTIFY_EVENT_HEADER,
    Event,
    Inotify,
    InotifyEventParseError,
    parse_events,
)
from skillset.watch import _WatchDaemon, _Retry


REPO_ROOT = Path(__file__).resolve().parents[1]
_WATCH_SCRIPT = (
    "from pathlib import Path; import sys; "
    "from skillset.watch import run_watch; "
    "raise SystemExit(run_watch(Path(sys.argv[1])))"
)


def _record(wd: int, mask: int, name: bytes = b"") -> bytes:
    encoded_name = name + b"\0" if name else b""
    padded_length = (len(encoded_name) + 3) & ~3
    encoded_name += b"\0" * (padded_length - len(encoded_name))
    return INOTIFY_EVENT_HEADER.pack(wd, mask, 42, len(encoded_name)) + encoded_name


class InotifyBindingTests(unittest.TestCase):
    def test_parses_multiple_records_and_rejects_truncated_records(self) -> None:
        events = parse_events(_record(7, IN_CREATE | IN_ISDIR, b"folder") + _record(-1, IN_Q_OVERFLOW))
        self.assertEqual(events, [
            Event(wd=7, mask=IN_CREATE | IN_ISDIR, cookie=42, name="folder"),
            Event(wd=-1, mask=IN_Q_OVERFLOW, cookie=42, name=""),
        ])
        with self.assertRaises(InotifyEventParseError):
            parse_events(b"\x01\x02")
        with self.assertRaises(InotifyEventParseError):
            parse_events(INOTIFY_EVENT_HEADER.pack(1, IN_CREATE, 0, 8) + b"name")

    @unittest.skipUnless(sys.platform.startswith("linux"), "native inotify exists on Linux only")
    def test_fd_is_nonblocking_cloexec_and_receives_a_real_event(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillset-inotify-") as temporary:
            directory = Path(temporary)
            with Inotify() as watcher:
                descriptor_flags = fcntl_getfd(watcher.fd)
                status_flags = fcntl_getfl(watcher.fd)
                self.assertTrue(descriptor_flags & fcntl.FD_CLOEXEC)
                self.assertTrue(status_flags & os.O_NONBLOCK)
                watcher.add_watch(directory, IN_CREATE | IN_ONLYDIR | IN_DONT_FOLLOW)
                selector = selectors.DefaultSelector()
                self.addCleanup(selector.close)
                selector.register(watcher.fd, selectors.EVENT_READ)
                (directory / "created").mkdir()
                self.assertTrue(selector.select(2), "kernel did not report the directory creation")
                self.assertTrue(any(event.name == "created" for event in watcher.read_events()))


def fcntl_getfd(descriptor: int) -> int:
    return fcntl.fcntl(descriptor, fcntl.F_GETFD)


def fcntl_getfl(descriptor: int) -> int:
    return fcntl.fcntl(descriptor, fcntl.F_GETFL)


class _FakeBackend:
    def __init__(self) -> None:
        self.removed: list[int] = []
        self.fd = 123
        self.paths: dict[Path, int] = {}
        self.closed = False

    def add_watch(self, path: Path, _mask: int) -> int:
        if path not in self.paths:
            self.paths[path] = len(self.paths) + 1
        return self.paths[path]

    def remove_watch(self, wd: int) -> None:
        self.removed.append(wd)

    def read_events(self) -> list[Event]:
        return []

    def close(self) -> None:
        self.closed = True


class _StoppingSelector:
    def __init__(self) -> None:
        self.daemon: _WatchDaemon | None = None
        self.timeouts: list[float | None] = []
        self.closed = False

    def register(self, _descriptor: int, _events: int, _data: object = None) -> object:
        return object()

    def unregister(self, _descriptor: int) -> object:
        return object()

    def select(self, timeout: float | None = None) -> list[tuple[object, int]]:
        self.timeouts.append(timeout)
        assert self.daemon is not None
        self.daemon.stop_requested = True
        return []

    def close(self) -> None:
        self.closed = True


class WatchPolicyTests(unittest.TestCase):
    def test_idle_has_no_timeout_and_candidate_retries_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillset-watch-policy-") as temporary:
            daemon = _WatchDaemon(Path(temporary) / "config.json", monotonic=lambda: 10.0)
            self.assertIsNone(daemon._next_timeout())
            daemon.retry_queue[Path(temporary)] = _Retry(10.2)
            self.assertAlmostEqual(daemon._next_timeout(), 0.2)

    def test_drop_removes_anchor_watch_for_moved_out_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillset-watch-drop-") as temporary:
            root = Path(temporary)
            daemon = _WatchDaemon(root / "config.json")
            daemon.backend = _FakeBackend()
            daemon.wd_paths[11] = root
            daemon.path_wds[root] = 11
            daemon.roles[root] = {"anchor", "boundary"}
            daemon._drop_subtree(root)
            self.assertNotIn(root, daemon.path_wds)
            self.assertNotIn(11, daemon.wd_paths)
            self.assertNotIn(root, daemon.roles)
            self.assertEqual(daemon.backend.removed, [11])

    def test_idle_loop_uses_an_indefinite_selector_wait_after_startup(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("watch service uses Linux inotify")
        with tempfile.TemporaryDirectory(prefix="skillset-watch-idle-") as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"schema_version": 1, "sets": {}, "roots": {}}),
                encoding="utf-8",
            )
            backend = _FakeBackend()
            selector = _StoppingSelector()
            daemon = _WatchDaemon(
                config_path,
                backend_factory=lambda: backend,
                selector_factory=lambda: selector,
            )
            selector.daemon = daemon
            with contextlib.ExitStack() as stack:
                stack.enter_context(unittest.mock.patch.dict(os.environ, {"XDG_STATE_HOME": str(root / "state")}))
                sync = stack.enter_context(unittest.mock.patch("skillset.watch.auto.sync_projects", return_value=0))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                self.assertEqual(daemon.run(), 0)
            self.assertEqual(selector.timeouts, [None])
            sync.assert_called_once_with(config_path.resolve(), quiet=True)
            self.assertTrue(selector.closed)
            self.assertTrue(backend.closed)

    def test_new_boundary_directory_refreshes_only_its_subtree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillset-watch-local-") as temporary:
            root = Path(temporary)
            child = root / "new"
            daemon = _WatchDaemon(root / "config.json")
            daemon.wd_paths[1] = root
            daemon.path_wds[root] = 1
            daemon.roles[root] = {"boundary"}
            daemon._rebuild_watch_graph = Mock()
            daemon._refresh_subtree = Mock()
            daemon._schedule_subtree_scan = Mock()
            daemon._process_events([Event(1, IN_CREATE | IN_ISDIR, 0, "new")])
            daemon._rebuild_watch_graph.assert_not_called()
            daemon._refresh_subtree.assert_called_once_with(child)
            daemon._schedule_subtree_scan.assert_called_once_with(child)

    def test_overflow_reloads_once_and_logs_full_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="skillset-watch-overflow-") as temporary:
            daemon = _WatchDaemon(Path(temporary) / "config.json")
            daemon._load_config_and_reconcile = Mock()
            with _captured_stderr() as output:
                daemon._process_events([Event(-1, IN_Q_OVERFLOW, 0, "")])
            daemon._load_config_and_reconcile.assert_called_once_with()
            self.assertIn("queue overflowed", output.getvalue())


class _captured_stderr:
    def __enter__(self):
        import contextlib
        import io

        self.buffer = io.StringIO()
        self._context = contextlib.redirect_stderr(self.buffer)
        self._context.__enter__()
        return self.buffer

    def __exit__(self, *args):
        return self._context.__exit__(*args)


@unittest.skipUnless(sys.platform.startswith("linux"), "watch service uses Linux inotify")
class NativeWatchIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="skillset-watch-e2e-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.config_home = self.root / "config-home"
        self.state_home = self.root / "state-home"
        self.base = self.root / "projects"
        for directory in (self.home, self.config_home, self.state_home, self.base):
            directory.mkdir(parents=True)
        self.config_path = self.config_home / "skillset" / "config.json"
        self.config_path.parent.mkdir(parents=True)
        self.source = self.root / "sources" / "team-skill"
        self.source.mkdir(parents=True)
        (self.source / "SKILL.md").write_text("# team skill\n", encoding="utf-8")
        self.special_source = self.root / "sources" / "special-skill"
        self.special_source.mkdir(parents=True)
        (self.special_source / "SKILL.md").write_text("# special skill\n", encoding="utf-8")
        self.env = os.environ.copy()
        self.env.update(
            {
                "HOME": str(self.home),
                "XDG_CONFIG_HOME": str(self.config_home),
                "XDG_STATE_HOME": str(self.state_home),
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": str(self.root / "empty-git-config"),
            }
        )
        self.process: subprocess.Popen[bytes] | None = None
        self._stdout_seen = bytearray()
        self._stderr_seen = bytearray()

    def write_config(self, roots: dict[Path, str | None], *, atomic: bool = False) -> None:
        value = {
            "schema_version": 1,
            "sets": {
                "team": {"agents": ["codex"], "skills": {"team-skill": str(self.source)}},
                "special": {"agents": ["claude"], "skills": {"special-skill": str(self.special_source)}},
            },
            "roots": {str(path): set_name for path, set_name in roots.items()},
        }
        contents = json.dumps(value, indent=2) + "\n"
        if atomic:
            temporary_path = self.config_path.with_name("config.json.next")
            temporary_path.write_text(contents, encoding="utf-8")
            os.replace(temporary_path, self.config_path)
        else:
            self.config_path.write_text(contents, encoding="utf-8")

    def git(self, *arguments: str | Path, cwd: Path | None = None) -> None:
        subprocess.run(
            ["git", *(str(argument) for argument in arguments)],
            cwd=cwd,
            env=self.env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )

    def make_repo(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.git("init", "--quiet", cwd=path)
        self.git("config", "user.name", "Skillset Watch Test", cwd=path)
        self.git("config", "user.email", "watch-test@example.invalid", cwd=path)
        (path / "README.md").write_text("temporary repository\n", encoding="utf-8")
        self.git("add", "README.md", cwd=path)
        self.git("commit", "--quiet", "-m", "initial", cwd=path)
        return path

    def start_watch(self) -> None:
        self._stdout_seen.clear()
        self._stderr_seen.clear()
        self.process = subprocess.Popen(
            [sys.executable, "-c", _WATCH_SCRIPT, str(self.config_path)],
            cwd=REPO_ROOT,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._wait_for_output("stdout", lambda value: "watch: ready" in value.splitlines())

    def _wait_for_output(self, stream_name: str, predicate, *, timeout: float = 10.0) -> str:
        assert self.process is not None
        stream = self.process.stdout if stream_name == "stdout" else self.process.stderr
        seen = self._stdout_seen if stream_name == "stdout" else self._stderr_seen
        assert stream is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(stream, selectors.EVENT_READ)
            deadline = time.monotonic() + timeout
            while True:
                decoded = seen.decode(errors="replace")
                if predicate(decoded):
                    return decoded
                if self.process.poll() is not None:
                    remaining_out, remaining_err = self.process.communicate()
                    self._stdout_seen.extend(remaining_out)
                    self._stderr_seen.extend(remaining_err)
                    self.fail(
                        f"watcher exited before expected {stream_name} output "
                        f"(exit={self.process.returncode}): "
                        f"stdout={self._stdout_seen.decode(errors='replace')} "
                        f"stderr={self._stderr_seen.decode(errors='replace')}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self.fail(
                        f"timed out waiting for watcher {stream_name} output; "
                        f"seen={decoded!r}"
                    )
                chunk = os.read(stream.fileno(), 4096)
                if not chunk:
                    self.fail(f"watcher closed its {stream_name} stream")
                seen.extend(chunk)
        finally:
            selector.close()

    def stop_watch(self, signum: int = signal.SIGTERM) -> tuple[str, str]:
        assert self.process is not None
        if self.process.poll() is None:
            self.process.send_signal(signum)
        stdout, stderr = self.process.communicate(timeout=8)
        combined_stdout = bytes(self._stdout_seen) + stdout
        combined_stderr = bytes(self._stderr_seen) + stderr
        stdout_text = combined_stdout.decode(errors="replace")
        stderr_text = combined_stderr.decode(errors="replace")
        self.assertEqual(
            self.process.returncode,
            0,
            f"watcher failed\nstdout:\n{stdout_text}\nstderr:\n{stderr_text}",
        )
        return stdout_text, stderr_text

    def wait_for(self, predicate, *, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("timed out waiting for native watcher result")

    def codex_link(self, repo: Path) -> Path:
        return repo / ".agents" / "skills" / "team-skill"

    def claude_link(self, repo: Path) -> Path:
        return repo / ".claude" / "skills" / "special-skill"

    def tearDown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            self.process.communicate(timeout=8)

    def test_new_repository_moved_in_and_worktree_dotgit_file(self) -> None:
        self.write_config({self.base: "team"})
        self.start_watch()

        new_repo = self.make_repo(self.base / "created-after-start")
        self.wait_for(lambda: self.codex_link(new_repo).is_symlink())

        outside = self.make_repo(self.root / "outside" / "moved-in")
        moved = self.base / "moved-in"
        shutil.move(outside, moved)
        self.wait_for(lambda: self.codex_link(moved).is_symlink())

        worktree_source = self.make_repo(self.root / "worktree-source")
        worktree = self.base / "linked-worktree"
        self.git("worktree", "add", "--quiet", "--detach", str(worktree), "HEAD", cwd=worktree_source)
        self.assertTrue((worktree / ".git").is_file())
        self.wait_for(lambda: self.codex_link(worktree).is_symlink())
        stdout, _stderr = self.stop_watch()
        self.assertIn(f"project: {new_repo}", stdout)
        self.assertIn(f"project: {moved}", stdout)
        self.assertIn(f"project: {worktree}", stdout)

    def test_nested_rules_and_atomic_config_reload_pause_and_repair(self) -> None:
        nested = self.base / "disabled" / "override"
        normal = self.make_repo(self.base / "normal")
        disabled = self.make_repo(self.base / "disabled" / "inactive")
        override = self.make_repo(nested)
        self.write_config(
            {
                self.base: "team",
                self.base / "disabled": None,
                nested: "special",
            }
        )
        self.start_watch()
        self.wait_for(lambda: self.codex_link(normal).is_symlink())
        self.wait_for(lambda: self.claude_link(override).is_symlink())
        self.assertFalse(self.codex_link(disabled).exists())

        self.config_path.write_text("{ malformed json", encoding="utf-8")
        self._wait_for_output(
            "stderr",
            lambda value: "configuration unavailable; waiting for repair" in value,
        )
        during_invalid = self.make_repo(self.base / "created-while-invalid")
        time.sleep(0.45)
        self.assertFalse(self.codex_link(during_invalid).exists())

        self.write_config(
            {
                self.base: "team",
                self.base / "disabled": None,
                nested: "special",
            },
            atomic=True,
        )
        self.wait_for(lambda: self.codex_link(during_invalid).is_symlink())
        _stdout, stderr = self.stop_watch()
        self.assertIn("configuration unavailable; waiting for repair", stderr)

    def test_incomplete_git_marker_upgrades_to_repo_and_stops_source_tree_watches(self) -> None:
        incomplete = self.base / "partial"
        incomplete.mkdir()
        (incomplete / ".git").mkdir()
        (incomplete / "source").mkdir()
        self.write_config({self.base: "team"})
        self.start_watch()
        self.assertFalse(self.codex_link(incomplete).exists())

        self.git("init", "--quiet", cwd=incomplete)
        self.git("config", "user.name", "Skillset Watch Test", cwd=incomplete)
        self.git("config", "user.email", "watch-test@example.invalid", cwd=incomplete)
        (incomplete / "README.md").write_text("ready\n", encoding="utf-8")
        self.git("add", "README.md", cwd=incomplete)
        self.git("commit", "--quiet", "-m", "initial", cwd=incomplete)
        self.wait_for(lambda: self.codex_link(incomplete).is_symlink())

        nested_repo = self.make_repo(incomplete / "source" / "nested")
        time.sleep(0.45)
        self.assertFalse(self.codex_link(nested_repo).exists())
        self.stop_watch()

    def test_sigterm_shuts_down_cleanly_after_idle_block(self) -> None:
        self.write_config({self.base: "team"})
        self.start_watch()
        stdout, stderr = self.stop_watch(signal.SIGINT)
        self.assertEqual(stdout.strip(), "watch: ready")
        self.assertEqual(stderr, "")
        self.start_watch()
        stdout, stderr = self.stop_watch(signal.SIGTERM)
        self.assertEqual(stdout.strip(), "watch: ready")
        self.assertEqual(stderr, "")

    def test_config_parent_removal_and_recreation_rearms_reload(self) -> None:
        self.write_config({self.base: "team"})
        self.start_watch()
        shutil.rmtree(self.config_path.parent)
        self._wait_for_output(
            "stderr",
            lambda value: "configuration unavailable; waiting for repair" in value,
        )
        waiting_repo = self.make_repo(self.base / "appeared-while-config-missing")
        self.assertFalse(self.codex_link(waiting_repo).exists())

        self.config_path.parent.mkdir(parents=True)
        self.write_config({self.base: "team"}, atomic=True)
        self.wait_for(lambda: self.codex_link(waiting_repo).is_symlink())
        self.stop_watch()


if __name__ == "__main__":
    unittest.main()
