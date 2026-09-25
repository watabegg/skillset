"""Black-box acceptance tests for the public ``skillset`` CLI.

Each invocation runs in a fresh temporary HOME/XDG/CODEX_HOME and uses only
temporary Git repositories and skill sources. These tests deliberately exercise
the CLI in a subprocess so they do not depend on internal Python APIs.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


class SkillsetCliContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(prefix="skillset-cli-")
        self.addCleanup(self.temporary_directory.cleanup)
        self.temp = Path(self.temporary_directory.name)
        self.home = self.temp / "home"
        self.xdg = self.temp / "xdg"
        self.codex_home = self.temp / "codex-home"
        for directory in (self.home, self.xdg, self.codex_home):
            directory.mkdir(parents=True)

        self.env = os.environ.copy()
        self.env.update(
            {
                "HOME": str(self.home),
                "XDG_CONFIG_HOME": str(self.xdg),
                "CODEX_HOME": str(self.codex_home),
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": str(self.temp / "empty-git-config"),
            }
        )
        self.config = self.temp / "config.json"

    def cli(
        self,
        *arguments: str | Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-m", "skillset", *(str(arg) for arg in arguments)]
        return subprocess.run(
            command,
            cwd=cwd or self.temp,
            env=env or self.env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

    def git(self, *arguments: str | Path, cwd: Path | None = None) -> None:
        subprocess.run(
            ["git", *(str(arg) for arg in arguments)],
            cwd=cwd,
            env=self.env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def make_repo(self, name: str = "project") -> Path:
        root = self.temp / name
        root.mkdir()
        self.git("init", "--quiet", cwd=root)
        self.git("config", "user.name", "CLI Contract Test", cwd=root)
        self.git("config", "user.email", "skillset-test@example.invalid", cwd=root)
        (root / "README.md").write_text("temporary repository\n", encoding="utf-8")
        self.git("add", "README.md", cwd=root)
        self.git("commit", "--quiet", "-m", "initial", cwd=root)
        return root

    def make_skill(self, name: str = "skill-one", *, contents: str | None = None) -> Path:
        source = self.temp / "sources" / name
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text(
            contents if contents is not None else f"# {name}\n",
            encoding="utf-8",
        )
        return source

    def write_config(self, value: dict[str, Any], path: Path | None = None) -> Path:
        output = path or self.config
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return output

    def config_for(
        self,
        sets: dict[str, dict[str, Any]],
        *,
        path: Path | None = None,
    ) -> Path:
        return self.write_config({"schema_version": 1, "sets": sets}, path)

    def one_set_config(
        self,
        source: Path,
        *,
        name: str = "demo",
        agents: list[str] | None = None,
        skill_name: str = "skill-one",
        path: Path | None = None,
    ) -> Path:
        return self.config_for(
            {
                name: {
                    "agents": agents if agents is not None else ["codex"],
                    "skills": {skill_name: str(source)},
                }
            },
            path=path,
        )

    def run_apply(
        self,
        root: Path,
        set_name: str = "demo",
        *,
        config: Path | None = None,
        dry_run: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        args: list[str | Path] = ["apply", set_name]
        if dry_run:
            args.append("--dry-run")
        if config is not None:
            args.extend(["--config", config])
        return self.cli(*args, cwd=root)

    def destination(self, root: Path, agent: str, skill_name: str) -> Path:
        agent_root = ".agents/skills" if agent == "codex" else ".claude/skills"
        return root / agent_root / skill_name

    def state_file(self, root: Path) -> Path:
        return root / ".skillset" / "state.json"

    def assert_exit(self, result: subprocess.CompletedProcess[str], expected: int) -> None:
        self.assertEqual(
            result.returncode,
            expected,
            msg=f"expected exit {expected}, got {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def rows(self, result: subprocess.CompletedProcess[str]) -> list[tuple[str, str, str]]:
        parsed: list[tuple[str, str, str]] = []
        for line in result.stdout.splitlines():
            if "\t" not in line:
                continue
            fields = line.split("\t")
            self.assertEqual(len(fields), 3, f"unexpected output row: {line!r}")
            parsed.append((fields[0], fields[1], fields[2]))
        return parsed

    def tree_snapshot(self, root: Path) -> dict[str, tuple[str, int, bytes | str | None]]:
        snapshot: dict[str, tuple[str, int, bytes | str | None]] = {}

        def walk(directory: Path) -> None:
            for entry in sorted(os.scandir(directory), key=lambda item: item.name):
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                info = entry.stat(follow_symlinks=False)
                mode = stat.S_IMODE(info.st_mode)
                if entry.is_symlink():
                    snapshot[relative] = ("symlink", mode, os.readlink(path))
                elif entry.is_dir(follow_symlinks=False):
                    snapshot[relative] = ("directory", mode, None)
                    walk(path)
                elif entry.is_file(follow_symlinks=False):
                    snapshot[relative] = ("file", mode, path.read_bytes())
                else:
                    snapshot[relative] = ("other", mode, None)

        walk(root)
        return snapshot

    def test_dry_run_apply_status_and_remove_round_trip(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source, agents=["codex", "claude"])

        dry_run = self.run_apply(root, config=self.config, dry_run=True)
        self.assert_exit(dry_run, 0)
        expected_destinations = sorted(
            [
                self.destination(root, "codex", "skill-one"),
                self.destination(root, "claude", "skill-one"),
            ],
            key=str,
        )
        dry_rows = self.rows(dry_run)
        self.assertEqual([row[0] for row in dry_rows], ["add", "add"])
        self.assertEqual([row[1] for row in dry_rows], [str(path) for path in expected_destinations])
        self.assertEqual([row[2] for row in dry_rows], [str(source.resolve())] * 2)
        self.assertFalse((root / ".skillset").exists())
        self.assertFalse((root / ".agents").exists())
        self.assertFalse((root / ".claude").exists())

        applied = self.run_apply(root, config=self.config)
        self.assert_exit(applied, 0)
        self.assertEqual(applied.stdout.splitlines()[0], f"project: {root.resolve()}")
        self.assertEqual([row[0] for row in self.rows(applied)], ["add", "add"])
        for destination in expected_destinations:
            self.assertTrue(destination.is_symlink())
            self.assertEqual(os.readlink(destination), str(source.resolve()))

        status = self.cli("status", "--config", self.config, cwd=root)
        self.assert_exit(status, 0)
        self.assertIn("active_set: demo", status.stdout.splitlines())
        self.assertEqual([row[0] for row in self.rows(status)], ["keep", "keep"])

        before = self.state_file(root).stat()
        before_bytes = self.state_file(root).read_bytes()
        no_op = self.run_apply(root, config=self.config)
        self.assert_exit(no_op, 0)
        after = self.state_file(root).stat()
        self.assertEqual(self.state_file(root).read_bytes(), before_bytes)
        self.assertEqual((after.st_ino, after.st_mtime_ns), (before.st_ino, before.st_mtime_ns))

        removed = self.cli("remove", "--config", self.temp / "missing-config.json", cwd=root)
        self.assert_exit(removed, 0)
        self.assertEqual(removed.stdout.splitlines()[0], f"project: {root.resolve()}")
        self.assertEqual([row[0] for row in self.rows(removed)], ["remove", "remove"])
        for destination in expected_destinations:
            self.assertFalse(destination.is_symlink())
        self.assertEqual(json.loads(self.state_file(root).read_text(encoding="utf-8")), {
            "schema_version": 1,
            "active_set": None,
            "links": {},
        })

        removed_again = self.cli("remove", cwd=root)
        self.assert_exit(removed_again, 0)
        self.assertEqual(self.rows(removed_again), [])

    def test_switching_sets_and_applying_an_empty_set(self) -> None:
        root = self.make_repo()
        first_source = self.make_skill("first")
        second_source = self.make_skill("second")
        self.config_for(
            {
                "first": {"agents": ["codex"], "skills": {"first-skill": str(first_source)}},
                "second": {
                    "agents": ["codex", "claude"],
                    "skills": {"second-skill": str(second_source)},
                },
                "empty": {"agents": ["codex"], "skills": {}},
            }
        )

        self.assert_exit(self.run_apply(root, "first", config=self.config), 0)
        first_destination = self.destination(root, "codex", "first-skill")
        self.assertTrue(first_destination.is_symlink())

        switched = self.run_apply(root, "second", config=self.config)
        self.assert_exit(switched, 0)
        self.assertFalse(first_destination.is_symlink())
        second_destinations = [
            self.destination(root, agent, "second-skill") for agent in ("codex", "claude")
        ]
        self.assertTrue(all(path.is_symlink() for path in second_destinations))
        self.assertEqual([row[0] for row in self.rows(switched)], ["remove", "add", "add"])
        state = json.loads(self.state_file(root).read_text(encoding="utf-8"))
        self.assertEqual(state["active_set"], "second")

        emptied = self.run_apply(root, "empty", config=self.config)
        self.assert_exit(emptied, 0)
        self.assertEqual([row[0] for row in self.rows(emptied)], ["remove", "remove"])
        self.assertTrue(all(not path.is_symlink() for path in second_destinations))
        state = json.loads(self.state_file(root).read_text(encoding="utf-8"))
        self.assertEqual(state, {"schema_version": 1, "active_set": "empty", "links": {}})

    def test_duplicate_unknown_and_unavailable_configuration_errors_are_preflighted(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        bad_configs: list[tuple[str, Path]] = []

        duplicate_agents = self.config_for(
            {"demo": {"agents": ["codex", "codex"], "skills": {"skill-one": str(source)}}},
            path=self.temp / "duplicate-agents.json",
        )
        bad_configs.append(("duplicate agents", duplicate_agents))
        unknown_field = self.config_for(
            {
                "demo": {
                    "agents": ["codex"],
                    "skills": {"skill-one": str(source)},
                    "unexpected": True,
                }
            },
            path=self.temp / "unknown-field.json",
        )
        bad_configs.append(("unknown field", unknown_field))

        for description, config in bad_configs:
            with self.subTest(description=description):
                result = self.run_apply(root, config=config)
                self.assert_exit(result, 2)
                self.assertFalse((root / ".skillset").exists())
                self.assertFalse((root / ".agents").exists())

        unavailable = self.one_set_config(
            self.temp / "sources" / "not-created",
            path=self.temp / "unavailable-source.json",
        )
        unavailable_result = self.run_apply(root, config=unavailable)
        self.assert_exit(unavailable_result, 2)
        self.assertFalse((root / ".skillset").exists())

        valid = self.one_set_config(source)
        unknown_set = self.cli("apply", "absent", "--config", valid, cwd=root)
        self.assert_exit(unknown_set, 2)
        self.assertFalse((root / ".skillset").exists())

    def test_duplicate_json_keys_are_rejected(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        config = self.temp / "duplicate-json-key.json"
        config.write_text(
            '{"schema_version":1,"sets":{"demo":{"agents":["codex"],'
            f'"skills":{{"skill-one":{json.dumps(str(source))}}},'
            '"skills":{}}}}\n',
            encoding="utf-8",
        )

        result = self.run_apply(root, config=config)
        self.assert_exit(result, 2)
        self.assertFalse((root / ".skillset").exists())

    def test_conflict_preserves_existing_file_bytes_mode_and_managed_state(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        destination = self.destination(root, "codex", "skill-one")
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"user data\x00must remain\n")
        destination.chmod(0o640)
        original_bytes = destination.read_bytes()
        original_mode = stat.S_IMODE(destination.stat().st_mode)

        unowned_conflict = self.run_apply(root, config=self.config)
        self.assert_exit(unowned_conflict, 3)
        self.assertEqual(destination.read_bytes(), original_bytes)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), original_mode)
        self.assertFalse((root / ".skillset").exists())

        destination.unlink()
        first_apply = self.run_apply(root, config=self.config)
        self.assert_exit(first_apply, 0)
        state_path = self.state_file(root)
        state_before = state_path.read_bytes()
        destination.unlink()
        destination.write_bytes(b"replacement data\n")
        destination.chmod(0o604)
        conflict_bytes = destination.read_bytes()
        conflict_mode = stat.S_IMODE(destination.stat().st_mode)

        managed_conflict = self.run_apply(root, config=self.config)
        self.assert_exit(managed_conflict, 3)
        self.assertEqual(destination.read_bytes(), conflict_bytes)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), conflict_mode)
        self.assertEqual(state_path.read_bytes(), state_before)
        self.assertFalse((root / ".skillset" / "pending.json").exists())

    def test_unmanaged_same_target_symlink_is_a_conflict_and_remove_leaves_it(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        destination = self.destination(root, "codex", "skill-one")
        destination.parent.mkdir(parents=True)
        destination.symlink_to(source.resolve())

        apply_result = self.run_apply(root, config=self.config)
        self.assert_exit(apply_result, 3)
        self.assertEqual(os.readlink(destination), str(source.resolve()))
        self.assertFalse((root / ".skillset").exists())

        remove_result = self.cli("remove", cwd=root)
        self.assert_exit(remove_result, 0)
        self.assertEqual(os.readlink(destination), str(source.resolve()))

    def test_remove_uses_manifest_when_config_and_source_are_gone(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        self.assert_exit(self.run_apply(root, config=self.config), 0)
        destination = self.destination(root, "codex", "skill-one")
        self.assertTrue(destination.is_symlink())

        self.config.unlink()
        for child in source.iterdir():
            child.unlink()
        source.rmdir()
        missing_config_remove = self.cli("remove", "--config", self.config, cwd=root)
        self.assert_exit(missing_config_remove, 0)
        self.assertFalse(destination.is_symlink())
        state = json.loads(self.state_file(root).read_text(encoding="utf-8"))
        self.assertEqual(state, {"schema_version": 1, "active_set": None, "links": {}})

    def test_status_reports_missing_source_as_broken_and_apply_rejects_it(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        self.assert_exit(self.run_apply(root, config=self.config), 0)
        source.joinpath("SKILL.md").unlink()
        source.rmdir()

        status = self.cli("status", "--config", self.config, cwd=root)
        self.assert_exit(status, 3)
        self.assertIn("broken", [row[0] for row in self.rows(status)])

        apply_result = self.run_apply(root, config=self.config)
        self.assert_exit(apply_result, 2)
        self.assertTrue(self.destination(root, "codex", "skill-one").is_symlink())

    def test_default_xdg_config_and_tilde_source_resolution(self) -> None:
        root = self.make_repo()
        source = self.home / "sources" / "tilde-skill"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("# from home\n", encoding="utf-8")
        config = self.xdg / "skillset" / "config.json"
        self.config_for(
            {
                "demo": {
                    "agents": ["codex"],
                    "skills": {"tilde-skill": "~/sources/tilde-skill"},
                }
            },
            path=config,
        )

        result = self.cli("apply", "demo", cwd=root)
        self.assert_exit(result, 0)
        destination = self.destination(root, "codex", "tilde-skill")
        self.assertTrue(destination.is_symlink())
        self.assertEqual(os.readlink(destination), str(source.resolve()))

        status = self.cli("status", cwd=root)
        self.assert_exit(status, 0)
        self.assertIn("active_set: demo", status.stdout.splitlines())
        self.assertEqual([row[0] for row in self.rows(status)], ["keep"])

    def test_status_falls_back_to_recorded_links_when_config_is_missing_or_set_removed(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        self.assert_exit(self.run_apply(root, config=self.config), 0)
        destination = self.destination(root, "codex", "skill-one")
        recorded_link = os.readlink(destination)

        self.config.unlink()
        missing_config = self.cli("status", "--config", self.config, cwd=root)
        self.assert_exit(missing_config, 0)
        self.assertNotIn("remove", [row[0] for row in self.rows(missing_config)])
        self.assertTrue(missing_config.stderr.strip(), "missing config should produce a warning")
        self.assertTrue(destination.is_symlink())
        self.assertEqual(os.readlink(destination), recorded_link)

        self.config_for(
            {"other": {"agents": ["codex"], "skills": {}}},
        )
        removed_definition = self.cli("status", "--config", self.config, cwd=root)
        self.assert_exit(removed_definition, 0)
        self.assertNotIn("remove", [row[0] for row in self.rows(removed_definition)])
        self.assertTrue(removed_definition.stderr.strip(), "removed set should produce a warning")
        self.assertTrue(destination.is_symlink())
        self.assertEqual(os.readlink(destination), recorded_link)

    def test_global_duplicate_warning_does_not_block_apply(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        duplicate = self.home / ".agents" / "skills" / "skill-one"
        duplicate.mkdir(parents=True)
        (duplicate / "SKILL.md").write_text("# existing global skill\n", encoding="utf-8")

        result = self.run_apply(root, config=self.config)
        self.assert_exit(result, 0)
        self.assertIn("skill-one", result.stderr)
        self.assertTrue(self.destination(root, "codex", "skill-one").is_symlink())

    def test_symlinked_launcher_works_from_a_foreign_working_directory(self) -> None:
        launcher = REPO_ROOT / "bin" / "skillset"
        self.assertTrue(launcher.is_file(), f"expected public launcher at {launcher}")
        self.assertNotEqual(stat.S_IMODE(launcher.stat().st_mode) & 0o111, 0)
        foreign_root = self.make_repo("foreign-project")
        source = self.make_skill()
        self.one_set_config(source)
        alias_dir = self.temp / "external-bin"
        alias_dir.mkdir()
        alias = alias_dir / "skillset"
        alias.symlink_to(launcher)

        result = subprocess.run(
            [str(alias), "status", "--set", "demo", "--config", str(self.config)],
            cwd=foreign_root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        self.assert_exit(result, 0)
        self.assertEqual(
            self.rows(result),
            [("add", str(self.destination(foreign_root, "codex", "skill-one")), str(source.resolve()))],
        )
        self.assertFalse((foreign_root / ".skillset").exists())
        self.assertFalse((foreign_root / ".agents").exists())

    def test_tampered_state_path_traversal_is_rejected_without_touching_outside(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        self.assert_exit(self.run_apply(root, config=self.config), 0)
        state_path = self.state_file(root)
        original_state = json.loads(state_path.read_text(encoding="utf-8"))
        outside = self.temp / "outside"
        outside.write_bytes(b"safe\n")
        outside_before = outside.read_bytes()

        malformed_states = [
            {
                **original_state,
                "links": {".agents/skills/../../outside": str(source.resolve())},
            },
            {
                **original_state,
                "links": {
                    ".agents/skills/skill-one": str(source.parent / ".." / source.name),
                },
            },
        ]
        for index, malformed in enumerate(malformed_states):
            with self.subTest(index=index):
                state_path.write_text(json.dumps(malformed), encoding="utf-8")
                result = self.cli("remove", cwd=root)
                self.assert_exit(result, 1)
                self.assertTrue(self.destination(root, "codex", "skill-one").is_symlink())
                self.assertEqual(outside.read_bytes(), outside_before)

    def test_parent_symlink_is_refused_before_repository_state_is_created(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        redirected = self.temp / "redirected"
        redirected.mkdir()
        sentinel = redirected / "sentinel"
        sentinel.write_bytes(b"keep\n")
        (root / ".agents").symlink_to(redirected, target_is_directory=True)

        result = self.run_apply(root, config=self.config)
        self.assert_exit(result, 3)
        self.assertEqual(sentinel.read_bytes(), b"keep\n")
        self.assertTrue((root / ".agents").is_symlink())
        self.assertFalse((root / ".skillset").exists())

    def test_nondirectory_parent_is_rejected_for_status_apply_and_remove(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        parent = root / ".agents"
        parent.write_text("user file", encoding="utf-8")
        for arguments in (("status", "--set", "demo"), ("apply", "demo")):
            with self.subTest(arguments=arguments):
                result = self.cli(*arguments, "--config", self.config, cwd=root)
                self.assert_exit(result, 3)
                self.assertEqual(parent.read_text(), "user file")
                self.assertFalse((root / ".skillset").exists())
        state = self.state_file(root)
        state.parent.mkdir()
        state.write_text(json.dumps({
            "schema_version": 1, "active_set": "demo",
            "links": {".agents/skills/skill-one": str(source.resolve())},
        }), encoding="utf-8")
        self.assert_exit(self.cli("remove", cwd=root), 3)
        self.assertEqual(parent.read_text(), "user file")

    def test_non_git_bare_and_invalid_arguments_exit_two(self) -> None:
        self.assert_exit(self.cli("status", cwd=self.temp), 2)
        bare = self.temp / "bare.git"
        self.git("init", "--quiet", "--bare", bare)
        self.assert_exit(self.cli("status", "--project-root", bare), 2)
        self.assert_exit(self.cli("apply"), 2)

    def test_deleted_owned_link_is_recreated_then_forgotten_by_remove(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        self.assert_exit(self.run_apply(root, config=self.config), 0)
        destination = self.destination(root, "codex", "skill-one")
        destination.unlink()
        self.assert_exit(self.run_apply(root, config=self.config), 0)
        self.assertEqual(os.readlink(destination), str(source.resolve()))
        destination.unlink()
        self.assert_exit(self.cli("remove", cwd=root), 0)
        self.assertEqual(json.loads(self.state_file(root).read_text())["links"], {})

    def test_worktree_manifests_and_links_are_independent(self) -> None:
        main = self.make_repo("main")
        worktree = self.temp / "linked-worktree"
        self.git("-C", main, "worktree", "add", "--quiet", "-b", "linked", worktree)
        source = self.make_skill()
        self.one_set_config(source)

        self.assert_exit(self.run_apply(main, config=self.config), 0)
        self.assert_exit(self.run_apply(worktree, config=self.config), 0)
        main_state = self.state_file(main)
        worktree_state = self.state_file(worktree)
        self.assertTrue(main_state.is_file())
        self.assertTrue(worktree_state.is_file())
        self.assertNotEqual(main_state.resolve(), worktree_state.resolve())

        removed_worktree = self.cli("remove", cwd=worktree)
        self.assert_exit(removed_worktree, 0)
        self.assertFalse(self.destination(worktree, "codex", "skill-one").is_symlink())
        self.assertTrue(self.destination(main, "codex", "skill-one").is_symlink())
        self.assertEqual(json.loads(main_state.read_text(encoding="utf-8"))["active_set"], "demo")

    def test_status_and_dry_run_do_not_create_files_directories_or_locks(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        before = self.tree_snapshot(self.temp)

        unmanaged = self.cli("status", cwd=root)
        self.assert_exit(unmanaged, 0)
        self.assertIn("unmanaged", unmanaged.stdout.splitlines())
        self.assertEqual(self.tree_snapshot(self.temp), before)

        planned = self.cli("status", "--set", "demo", "--config", self.config, cwd=root)
        self.assert_exit(planned, 0)
        self.assertEqual(self.tree_snapshot(self.temp), before)

        dry_run = self.run_apply(root, config=self.config, dry_run=True)
        self.assert_exit(dry_run, 0)
        self.assertEqual(self.tree_snapshot(self.temp), before)
        for name in (".skillset", ".agents", ".claude"):
            self.assertFalse((root / name).exists())

    def test_pending_journal_is_reported_and_blocks_dry_run_without_mutation(self) -> None:
        root = self.make_repo()
        source = self.make_skill()
        self.one_set_config(source)
        skill_key = ".agents/skills/skill-one"
        before = {"schema_version": 1, "active_set": None, "links": {}}
        after = {
            "schema_version": 1,
            "active_set": "demo",
            "links": {skill_key: str(source.resolve())},
        }
        state_dir = root / ".skillset"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(json.dumps(before), encoding="utf-8")
        pending_path = state_dir / "pending.json"
        pending_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "before": before,
                    "after": after,
                    "operations": [
                        {
                            "action": "add",
                            "destination": skill_key,
                            "source": str(source.resolve()),
                            "previous_source": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        state_before = (state_dir / "state.json").read_bytes()
        pending_before = pending_path.read_bytes()
        invalid_config = self.temp / "malformed.json"
        invalid_config.write_text("{ invalid", encoding="utf-8")

        status = self.cli("status", "--config", invalid_config, cwd=root)
        self.assert_exit(status, 3)
        self.assertIn("pending", (status.stdout + status.stderr).lower())
        self.assertIn(skill_key, status.stdout + status.stderr)
        self.assertTrue(
            any(word in (status.stdout + status.stderr).lower() for word in ("before", "after")),
            "pending diagnostics should distinguish the journal expectations",
        )

        dry_run = self.run_apply(root, config=self.config, dry_run=True)
        self.assert_exit(dry_run, 3)
        self.assertEqual((state_dir / "state.json").read_bytes(), state_before)
        self.assertEqual(pending_path.read_bytes(), pending_before)
        self.assertFalse(self.destination(root, "codex", "skill-one").exists())


if __name__ == "__main__":
    unittest.main()
