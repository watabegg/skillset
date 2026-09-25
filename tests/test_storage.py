from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skillset.models import Journal, State
from skillset.planner import plan_remove, plan_set
from skillset.storage import (
    ConcurrentChangeError,
    CorruptStoreError,
    LockBusyError,
    PendingJournalError,
    UnsafePathError,
    execute_plan,
    load_snapshot,
)


DEST = ".agents/skills/sample-skill"


def write_skill(path: Path, content: str = "# sample\n") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(content, encoding="utf-8")
    return path.resolve()


def write_state(root: Path, state: State) -> Path:
    directory = root / ".skillset"
    directory.mkdir(exist_ok=True)
    path = directory / "state.json"
    path.write_text(json.dumps(state.to_json()), encoding="utf-8")
    return path


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = write_skill(self.root / "source")
        self.other = write_skill(self.root / "other", "# other\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_read_only_snapshot_does_not_create_anything(self) -> None:
        before = set(self.root.iterdir())
        snapshot = load_snapshot(self.root)
        self.assertEqual(snapshot.state, State())
        self.assertIsNone(snapshot.journal)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_apply_writes_link_manifest_and_clears_journal(self) -> None:
        plan = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
        execute_plan(self.root, State(), plan)

        destination = self.root / DEST
        self.assertTrue(destination.is_symlink())
        self.assertEqual(os.readlink(destination), str(self.source))
        snapshot = load_snapshot(self.root)
        self.assertEqual(snapshot.state, State("team", {DEST: str(self.source)}))
        self.assertIsNone(snapshot.journal)
        self.assertEqual(
            json.loads((self.root / ".skillset" / "state.json").read_text(encoding="utf-8")),
            {
                "schema_version": 1,
                "active_set": "team",
                "links": {DEST: str(self.source)},
            },
        )
        self.assertTrue((self.root / ".skillset" / "lock").is_file())

    def test_switch_updates_only_owned_link_then_remove_records_unmanaged(self) -> None:
        initial = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="old")
        execute_plan(self.root, State(), initial)
        before = State("old", {DEST: str(self.source)})

        switched = plan_set(self.root, before, {DEST: str(self.other)}, active_set="new")
        execute_plan(self.root, before, switched)
        self.assertEqual(os.readlink(self.root / DEST), str(self.other))
        self.assertEqual(load_snapshot(self.root).state, State("new", {DEST: str(self.other)}))

        managed = State("new", {DEST: str(self.other)})
        removal = plan_remove(self.root, managed)
        execute_plan(self.root, managed, removal)
        self.assertFalse((self.root / DEST).exists())
        self.assertEqual(load_snapshot(self.root).state, State())

    def test_remove_allows_a_broken_owned_symlink_and_forgets_missing_links(self) -> None:
        destination = self.root / DEST
        destination.parent.mkdir(parents=True)
        dead_source = self.root / "deleted-source"
        destination.symlink_to(dead_source, target_is_directory=True)
        before = State("team", {DEST: str(dead_source)})
        write_state(self.root, before)

        plan = plan_remove(self.root, before)
        execute_plan(self.root, before, plan)
        self.assertFalse(os.path.lexists(destination))
        self.assertEqual(load_snapshot(self.root).state, State())

    def test_matching_noop_preserves_state_bytes_and_mtime(self) -> None:
        first = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
        execute_plan(self.root, State(), first)
        state_path = self.root / ".skillset" / "state.json"
        before_bytes = state_path.read_bytes()
        before_mtime = state_path.stat().st_mtime_ns

        state = State("team", {DEST: str(self.source)})
        no_op = plan_set(self.root, state, {DEST: str(self.source)}, active_set="team")
        execute_plan(self.root, state, no_op)
        self.assertEqual(state_path.read_bytes(), before_bytes)
        self.assertEqual(state_path.stat().st_mtime_ns, before_mtime)

    def test_pending_journal_blocks_mutation_and_is_validated(self) -> None:
        after = State("team", {DEST: str(self.source)})
        operation = plan_set(self.root, State(), after.links, active_set="team").operations[0]
        directory = self.root / ".skillset"
        directory.mkdir()
        (directory / "pending.json").write_text(
            json.dumps(Journal(State(), after, (operation,)).to_json()), encoding="utf-8"
        )
        snapshot = load_snapshot(self.root)
        self.assertIsNotNone(snapshot.journal)
        plan = plan_set(self.root, State(), after.links, active_set="team")
        with self.assertRaises(PendingJournalError):
            execute_plan(self.root, State(), plan)
        self.assertFalse(os.path.lexists(self.root / DEST))

    def test_io_failure_after_journal_leaves_recoverable_pending_record(self) -> None:
        plan = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
        with patch("skillset.storage._install_link", side_effect=OSError("injected interruption")):
            with self.assertRaisesRegex(OSError, "injected interruption"):
                execute_plan(self.root, State(), plan)

        pending_path = self.root / ".skillset" / "pending.json"
        self.assertTrue(pending_path.is_file())
        self.assertIsNotNone(load_snapshot(self.root).journal)
        with self.assertRaises(PendingJournalError):
            execute_plan(self.root, State(), plan)

    def test_lock_contention_refuses_without_journaling(self) -> None:
        directory = self.root / ".skillset"
        directory.mkdir()
        lock_path = directory / "lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
            with self.assertRaises(LockBusyError):
                execute_plan(self.root, State(), plan)
            self.assertFalse((directory / "pending.json").exists())
            self.assertFalse(os.path.lexists(self.root / DEST))
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_partial_failure_can_be_restored_from_before_and_retried(self) -> None:
        from skillset import storage

        second = ".claude/skills/sample-skill"
        plan = plan_set(
            self.root, State(), {DEST: str(self.source), second: str(self.other)},
            active_set="team",
        )
        install = storage._install_link

        def fail_second(operation, destination):
            if operation.destination == second:
                raise OSError("interrupted after first link")
            install(operation, destination)

        with patch("skillset.storage._install_link", side_effect=fail_second):
            with self.assertRaises(OSError):
                execute_plan(self.root, State(), plan)
        self.assertTrue((self.root / DEST).is_symlink())
        self.assertFalse(os.path.lexists(self.root / second))
        journal = load_snapshot(self.root).journal
        self.assertIsNotNone(journal)
        with self.assertRaises(PendingJournalError):
            execute_plan(self.root, State(), plan)

        # Follow the documented manual recovery using recorded targets only.
        for destination, source in journal.after.links.items():
            path = self.root / destination
            if destination not in journal.before.links and path.is_symlink():
                self.assertEqual(os.readlink(path), source)
                path.unlink()
        write_state(self.root, journal.before)
        (self.root / ".skillset" / "pending.json").unlink()
        restored = load_snapshot(self.root)
        self.assertEqual(restored.state, State())
        execute_plan(self.root, restored.state, plan)
        self.assertEqual(load_snapshot(self.root).state, plan.after)
        self.assertIsNone(load_snapshot(self.root).journal)

    def test_destination_tamper_after_journal_is_preserved(self) -> None:
        from skillset import storage

        first = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
        execute_plan(self.root, State(), first)
        before = load_snapshot(self.root).state
        update = plan_set(self.root, before, {DEST: str(self.other)}, active_set="team")
        atomic_write = storage._write_json_atomic

        def tamper_after_journal(path, value):
            atomic_write(path, value)
            if path.name == "pending.json":
                (self.root / DEST).unlink()
                (self.root / DEST).write_text("user replacement", encoding="utf-8")

        with patch("skillset.storage._write_json_atomic", side_effect=tamper_after_journal):
            with self.assertRaises(ConcurrentChangeError):
                execute_plan(self.root, before, update)
        self.assertEqual((self.root / DEST).read_text(), "user replacement")
        self.assertEqual(load_snapshot(self.root).state, before)
        self.assertIsNotNone(load_snapshot(self.root).journal)

    def test_concurrent_metadata_directory_creation_uses_shared_lock(self) -> None:
        mkdir = Path.mkdir
        metadata = self.root / ".skillset"

        def create_then_report_exists(path, *args, **kwargs):
            if path == metadata:
                mkdir(path, *args, **kwargs)
                raise FileExistsError("another process created metadata")
            return mkdir(path, *args, **kwargs)

        plan = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
        with patch.object(Path, "mkdir", create_then_report_exists):
            execute_plan(self.root, State(), plan)
        self.assertEqual(load_snapshot(self.root).state, plan.after)

    def test_state_and_journal_corruption_and_path_escape_are_rejected(self) -> None:
        state_path = write_state(self.root, State())
        state_path.write_text('{"schema_version":true,"active_set":null,"links":{}}', encoding="utf-8")
        with self.assertRaises(CorruptStoreError):
            load_snapshot(self.root)

        state_path.write_text(
            json.dumps({"schema_version": 1, "active_set": "team", "links": {"../../outside": "/tmp/x"}}),
            encoding="utf-8",
        )
        with self.assertRaises(CorruptStoreError):
            load_snapshot(self.root)

        state_path.unlink()
        journal_path = self.root / ".skillset" / "pending.json"
        journal_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "before": State().to_json(),
                    "after": {"schema_version": 1, "active_set": "team", "links": {".claude/skills/../x": "/tmp/x"}},
                    "operations": [],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(CorruptStoreError):
            load_snapshot(self.root)

    def test_parent_symlinks_and_metadata_symlinks_are_refused(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / ".agents").symlink_to(outside, target_is_directory=True)
        plan = plan_set(self.root, State(), {DEST: str(self.source)}, active_set="team")
        from skillset.storage import validate_managed_parents

        with self.assertRaises(UnsafePathError):
            validate_managed_parents(self.root, (DEST,))

        (self.root / ".agents").unlink()
        metadata_target = self.root / "metadata.json"
        metadata_target.write_text(json.dumps(State().to_json()), encoding="utf-8")
        (self.root / ".skillset").mkdir()
        (self.root / ".skillset" / "state.json").symlink_to(metadata_target)
        with self.assertRaises(UnsafePathError):
            load_snapshot(self.root)


if __name__ == "__main__":
    unittest.main()
