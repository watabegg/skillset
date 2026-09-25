from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from skillset.models import State
from skillset.planner import inspect_recorded, plan_remove, plan_set


DEST = ".agents/skills/sample-skill"


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "SKILL.md").write_text("# sample\n", encoding="utf-8")
        self.other = self.root / "other"
        self.other.mkdir()
        (self.other / "SKILL.md").write_text("# other\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_plans_add_then_keep_and_update(self) -> None:
        desired = {DEST: str(self.source)}
        add = plan_set(self.root, State(), desired, active_set="team")
        self.assertEqual([row.label for row in add.rows], ["add"])
        self.assertEqual(add.operations[0].action, "add")

        destination = self.root / DEST
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.source, target_is_directory=True)
        owned = State("team", {DEST: str(self.source)})
        keep = plan_set(self.root, owned, desired, active_set="team")
        self.assertEqual([row.label for row in keep.rows], ["keep"])
        self.assertEqual(keep.operations, ())

        update = plan_set(
            self.root,
            owned,
            {DEST: str(self.other)},
            active_set="team",
        )
        self.assertEqual([row.label for row in update.rows], ["update"])
        self.assertEqual(update.operations[0].previous_source, str(self.source))

    def test_unowned_same_target_symlink_and_tampered_owned_link_conflict(self) -> None:
        destination = self.root / DEST
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.source, target_is_directory=True)
        desired = {DEST: str(self.source)}
        unowned = plan_set(self.root, State(), desired, active_set="team")
        self.assertEqual([row.label for row in unowned.rows], ["conflict"])

        owned = State("team", {DEST: str(self.other)})
        tampered = plan_set(self.root, owned, desired, active_set="team")
        self.assertEqual([row.label for row in tampered.rows], ["conflict"])

    def test_owned_missing_link_is_recreated_or_forgotten(self) -> None:
        state = State("old", {DEST: str(self.source)})
        add = plan_set(self.root, state, {DEST: str(self.other)}, active_set="new")
        self.assertEqual([row.label for row in add.rows], ["add"])

        forget = plan_set(self.root, state, {}, active_set="new")
        self.assertEqual([row.label for row in forget.rows], ["remove"])
        self.assertEqual(forget.after.links, {})

    def test_remove_allows_broken_target_and_detects_tampering(self) -> None:
        destination = self.root / DEST
        destination.parent.mkdir(parents=True)
        destination.symlink_to(self.root / "gone", target_is_directory=True)
        broken = State("team", {DEST: str(self.root / "gone")})
        plan = plan_remove(self.root, broken)
        self.assertEqual([row.label for row in plan.rows], ["remove"])

        destination.unlink()
        destination.symlink_to(self.source, target_is_directory=True)
        tampered = plan_remove(self.root, broken)
        self.assertEqual([row.label for row in tampered.rows], ["conflict"])

    def test_unavailable_requested_source_is_broken_and_recorded_status_is_read_only(self) -> None:
        missing = str(self.root / "gone")
        plan = plan_set(
            self.root,
            State(),
            {DEST: missing},
            active_set="team",
            unavailable_sources={missing},
        )
        self.assertEqual([row.label for row in plan.rows], ["broken"])

        destination = self.root / DEST
        destination.parent.mkdir(parents=True)
        destination.symlink_to(missing, target_is_directory=True)
        recorded = inspect_recorded(
            self.root,
            State("team", {DEST: missing}),
            check_source=lambda _path: object(),
        )
        self.assertEqual([row.label for row in recorded], ["broken"])

    def test_owned_absent_undesired_has_remove_row_without_touching_tree(self) -> None:
        state = State("team", {".claude/skills/sample-skill": str(self.source)})
        snapshot = set(self.root.iterdir())
        plan = plan_set(self.root, state, {}, active_set="team")
        self.assertEqual([row.label for row in plan.rows], ["remove"])
        self.assertEqual(set(self.root.iterdir()), snapshot)


if __name__ == "__main__":
    unittest.main()
