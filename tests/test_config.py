from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from skillset.config import ConfigError, load_config, source_problem, validate_selected_sources


def write_skill(path: Path, content: str = "# skill\n") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(content, encoding="utf-8")
    return path


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_config(self, value: object) -> None:
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

    def test_loads_multiple_sets_and_resolves_sources(self) -> None:
        source = write_skill(self.root / "source")
        alias = self.root / "alias"
        alias.symlink_to(source, target_is_directory=True)
        self.write_config(
            {
                "schema_version": 1,
                "sets": {
                    "team": {
                        "agents": ["codex", "claude"],
                        "skills": {"alpha-skill": str(alias)},
                    },
                    "empty": {"agents": ["codex"], "skills": {}},
                },
            }
        )

        config = load_config(self.config_path)
        self.assertEqual(config.sets["team"].agents, ("codex", "claude"))
        self.assertEqual(config.sets["team"].skills["alpha-skill"], source.resolve())
        self.assertEqual(config.sets["empty"].skills, {})

    def test_validates_all_set_shapes_but_not_unselected_source_availability(self) -> None:
        self.write_config(
            {
                "schema_version": 1,
                "sets": {
                    "chosen": {"agents": ["codex"], "skills": {}},
                    "unused": {
                        "agents": ["claude"],
                        "skills": {"later": str(self.root / "not-created")},
                    },
                },
            }
        )

        config = load_config(self.config_path)
        self.assertIn("unused", config.sets)

    def test_expands_only_home_prefix(self) -> None:
        self.write_config(
            {
                "schema_version": 1,
                "sets": {"team": {"agents": ["codex"], "skills": {"skill": "~/work/skill"}}},
            }
        )
        config = load_config(self.config_path)
        self.assertEqual(config.sets["team"].skills["skill"], (Path.home() / "work/skill").resolve())

        self.write_config({
            "schema_version": 1,
            "sets": {"team": {"agents": ["codex"], "skills": {"skill": "~//work/skill"}}},
        })
        config = load_config(self.config_path)
        self.assertEqual(config.sets["team"].skills["skill"], (Path.home() / "work/skill").resolve())

        for bad_path in ("relative/path", "~someone/skill", "~"):
            with self.subTest(bad_path=bad_path):
                self.write_config(
                    {
                        "schema_version": 1,
                        "sets": {"team": {"agents": ["codex"], "skills": {"skill": bad_path}}},
                    }
                )
                with self.assertRaises(ConfigError):
                    load_config(self.config_path)

    def test_rejects_duplicate_json_keys_even_in_nested_objects(self) -> None:
        self.config_path.write_text(
            '{"schema_version":1,"sets":{"team":{"agents":["codex"],'
            '"skills":{},"skills":{}}}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ConfigError, "duplicate JSON key"):
            load_config(self.config_path)

    def test_rejects_invalid_schema_fields_names_agents_and_version_types(self) -> None:
        invalid_values = [
            {"schema_version": True, "sets": {}},
            {"schema_version": 1.0, "sets": {}},
            {"schema_version": 1, "sets": {}, "extra": True},
            {"schema_version": 1, "sets": {"": {"agents": ["codex"], "skills": {}}}},
            {"schema_version": 1, "sets": {"team": {"agents": [], "skills": {}}}},
            {"schema_version": 1, "sets": {"team": {"agents": ["codex", "codex"], "skills": {}}}},
            {"schema_version": 1, "sets": {"team": {"agents": ["other"], "skills": {}}}},
            {"schema_version": 1, "sets": {"team": {"agents": ["codex"], "skills": {"Bad": "/x"}}}},
            {"schema_version": 1, "sets": {"team": {"agents": ["codex"], "skills": {}, "extra": 0}}},
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                self.write_config(value)
                with self.assertRaises(ConfigError):
                    load_config(self.config_path)

    def test_selected_source_requires_readable_directory_and_regular_nonempty_skill_file(self) -> None:
        valid = write_skill(self.root / "valid")
        invalid_dir = self.root / "missing-skill-file"
        invalid_dir.mkdir()
        empty = write_skill(self.root / "empty", "")
        symlink_file = self.root / "symlink-skill"
        symlink_file.mkdir()
        target = self.root / "target.md"
        target.write_text("# target\n", encoding="utf-8")
        (symlink_file / "SKILL.md").symlink_to(target)

        self.assertIsNone(source_problem("valid", valid))
        self.assertIsNotNone(source_problem("missing", invalid_dir))
        self.assertIsNotNone(source_problem("empty", empty))
        self.assertIsNotNone(source_problem("link", symlink_file))

        self.write_config(
            {
                "schema_version": 1,
                "sets": {
                    "team": {
                        "agents": ["codex"],
                        "skills": {"valid": str(valid), "missing": str(invalid_dir)},
                    }
                },
            }
        )
        problems = validate_selected_sources(load_config(self.config_path).sets["team"])
        self.assertEqual([problem.skill for problem in problems], ["missing"])


if __name__ == "__main__":
    unittest.main()
