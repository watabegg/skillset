from __future__ import annotations

import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from skillset.auto import discover_projects, select_set, sync_projects
from skillset.cli import _destination_map, main
from skillset.config import load_config
from skillset.models import Config, State
from skillset.planner import plan_set
from skillset.storage import execute_plan


class AutoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="skillset-auto-")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def make_repo(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--quiet", str(path)], check=True, capture_output=True)
        return path.resolve()

    def make_skill(self, name: str = "demo-skill") -> Path:
        source = self.root / "sources" / name
        source.mkdir(parents=True, exist_ok=True)
        (source / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
        return source.resolve()

    def make_config(
        self,
        roots: dict[Path, str | None],
        *,
        skill_sets: dict[str, dict[str, object]] | None = None,
    ) -> Path:
        if skill_sets is None:
            source = self.make_skill()
            skill_sets = {
                "team": {"agents": ["codex"], "skills": {"demo-skill": str(source)}},
                "other": {"agents": ["claude"], "skills": {"demo-skill": str(source)}},
            }
        path = self.root / "config.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sets": skill_sets,
                    "roots": {str(root): name for root, name in roots.items()},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_longest_component_ancestor_wins_and_disabled_root_stops_parent_rule(self) -> None:
        base = self.root / "projects"
        nested = base / "team"
        disabled = base / "private"
        config = Config(
            sets={},
            roots={
                base: "default",
                nested: "specific",
                disabled: None,
            },
        )

        self.assertEqual(select_set(config, nested / "repo"), "specific")
        self.assertEqual(select_set(config, disabled / "repo"), None)
        self.assertEqual(select_set(config, self.root / "projects-other"), None)

    def test_discovery_prunes_ignored_disabled_and_symlink_trees_but_scans_explicit_nested_roots(self) -> None:
        workspace = self.root / "workspace"
        direct = self.make_repo(workspace / "direct")
        unregistered_nested = self.make_repo(direct / "unregistered-nested")
        explicit_nested = self.make_repo(direct / "explicit-nested")
        hidden = self.make_repo(workspace / ".hidden" / "project")
        ignored = self.make_repo(workspace / "node_modules" / "ignored")
        disabled = self.make_repo(workspace / "blocked" / "project")
        enabled_below_disabled = self.make_repo(workspace / "blocked" / "enabled" / "project")
        outside = self.make_repo(self.root / "outside")
        symlink = workspace / "linked-outside"
        symlink.symlink_to(outside, target_is_directory=True)

        config = Config(
            sets={},
            roots={
                workspace.resolve(): "team",
                (direct / "explicit-nested").resolve(): "team",
                (workspace / "blocked").resolve(): None,
                (workspace / "blocked" / "enabled").resolve(): "team",
            },
        )
        self.assertEqual(
            discover_projects(config),
            sorted([direct, explicit_nested, hidden, enabled_below_disabled], key=str),
        )
        self.assertNotIn(unregistered_nested, discover_projects(config))
        self.assertNotIn(ignored, discover_projects(config))
        self.assertNotIn(disabled, discover_projects(config))
        self.assertNotIn(outside, discover_projects(config))
        self.assertEqual(discover_projects(config, start=workspace / "blocked"), [enabled_below_disabled])

    def _manual_apply(self, root: Path, config_path: Path, set_name: str = "team") -> None:
        config = load_config(config_path)
        skill_set = config.sets[set_name]
        plan = plan_set(
            root,
            State(),
            _destination_map(skill_set),
            active_set=set_name,
        )
        execute_plan(root, State(), plan)

    def test_same_set_manually_applied_state_is_adopted_and_different_set_conflicts(self) -> None:
        repo = self.make_repo(self.root / "repo")
        config_path = self.make_config({repo: "team"})
        self._manual_apply(repo, config_path)
        exclude = repo / ".git" / "info" / "exclude"
        before = exclude.read_bytes()

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = sync_projects(config_path, repo)
        self.assertEqual(status, 0)
        self.assertIn("active_set: team", output.getvalue())
        self.assertTrue((repo / ".agents" / "skills" / "demo-skill").is_symlink())
        added = exclude.read_text(encoding="utf-8").splitlines()
        self.assertIn("/.skillset/", added)
        self.assertIn("/.agents/skills/demo-skill", added)
        self.assertNotEqual(exclude.read_bytes(), before)

        data = json.loads(config_path.read_text(encoding="utf-8"))
        data["roots"] = {str(repo): "other"}
        config_path.write_text(json.dumps(data), encoding="utf-8")
        before_conflict = exclude.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = sync_projects(config_path, repo)
        self.assertEqual(status, 3)
        self.assertIn("conflict", output.getvalue())
        self.assertEqual(exclude.read_bytes(), before_conflict)

    def test_dry_run_is_read_only_and_successful_sync_does_not_duplicate_excludes(self) -> None:
        repo = self.make_repo(self.root / "repo")
        config_path = self.make_config({repo: "team"})
        exclude = repo / ".git" / "info" / "exclude"
        original_exclude = exclude.read_bytes()

        self.assertEqual(sync_projects(config_path, repo, dry_run=True), 0)
        self.assertFalse((repo / ".skillset").exists())
        self.assertFalse((repo / ".agents").exists())
        self.assertEqual(exclude.read_bytes(), original_exclude)
        self.assertFalse((exclude.parent / "exclude.skillset.lock").exists())

        self.assertEqual(sync_projects(config_path, repo, quiet=True), 0)
        after_first = exclude.read_bytes()
        exclude_mtime = exclude.stat().st_mtime_ns
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(sync_projects(config_path, repo, quiet=True), 0)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(exclude.read_bytes(), after_first)
        self.assertEqual(exclude.stat().st_mtime_ns, exclude_mtime)

    def test_tracked_skillset_metadata_and_tracked_destination_block_sync(self) -> None:
        for tracked_relative in (".skillset/user-data", ".agents/skills/demo-skill"):
            with self.subTest(tracked_relative=tracked_relative):
                repo = self.make_repo(self.root / tracked_relative.replace("/", "-"))
                config_path = self.make_config({repo: "team"})
                tracked_path = repo / tracked_relative
                tracked_path.parent.mkdir(parents=True, exist_ok=True)
                if tracked_relative == ".agents/skills/demo-skill":
                    self._manual_apply(repo, config_path)
                else:
                    tracked_path.write_text("user-owned\n", encoding="utf-8")
                subprocess.run(["git", "-C", str(repo), "add", "-f", tracked_relative], check=True)
                exclude = repo / ".git" / "info" / "exclude"
                before = exclude.read_bytes()

                self.assertEqual(sync_projects(config_path, repo), 3)
                self.assertEqual(exclude.read_bytes(), before)
                if tracked_relative == ".skillset/user-data":
                    self.assertFalse((repo / ".agents").exists())

    def test_unsafe_exclude_and_busy_exclude_lock_block_before_link_mutation(self) -> None:
        repo = self.make_repo(self.root / "repo")
        config_path = self.make_config({repo: "team"})
        exclude = repo / ".git" / "info" / "exclude"
        original = exclude.read_bytes()
        saved = self.root / "saved-exclude"
        saved.write_bytes(original)
        exclude.unlink()
        exclude.symlink_to(saved)

        self.assertEqual(sync_projects(config_path, repo), 3)
        self.assertFalse((repo / ".agents").exists())
        self.assertTrue(exclude.is_symlink())

        exclude.unlink()
        exclude.write_bytes(original)
        lock_path = exclude.with_name("exclude.skillset.lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(sync_projects(config_path, repo), 3)
            self.assertFalse((repo / ".agents").exists())
            self.assertFalse((repo / ".skillset").exists())
            self.assertEqual(exclude.read_bytes(), original)
        finally:
            os.close(descriptor)

    def test_roots_cli_registers_disables_lists_and_removes_normalized_paths_without_syncing(self) -> None:
        repo = self.make_repo(self.root / "repo")
        config_path = self.make_config({})
        alias = self.root / "repo-alias"
        alias.symlink_to(repo, target_is_directory=True)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                main(["roots", "add", str(alias), "--set", "team", "--config", str(config_path)]),
                0,
            )
        self.assertEqual(output.getvalue(), f"registered\t{repo}\tteam\n")
        self.assertEqual(load_config(config_path).roots, {repo: "team"})
        self.assertFalse((repo / ".skillset").exists())

        original_stat = config_path.stat()
        self.assertEqual(
            main(["roots", "add", str(repo), "--set", "team", "--config", str(config_path)]),
            0,
        )
        self.assertEqual((config_path.stat().st_ino, config_path.stat().st_mtime_ns), (original_stat.st_ino, original_stat.st_mtime_ns))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["roots", "disable", str(repo), "--config", str(config_path)]), 0)
        self.assertEqual(output.getvalue(), f"disabled\t{repo}\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["roots", "list", "--config", str(config_path)]), 0)
        self.assertEqual(output.getvalue(), f"{repo}\tdisabled\n")
        self.assertEqual(main(["roots", "remove", str(self.root / "missing"), "--config", str(config_path)]), 0)
        self.assertEqual(main(["roots", "remove", str(alias), "--config", str(config_path)]), 0)
        self.assertEqual(load_config(config_path).roots, {})


if __name__ == "__main__":
    unittest.main()
