"""Strict JSON configuration loading and source validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Any

from .models import Config, SkillSet, valid_skill_name


class ConfigError(ValueError):
    """The user-supplied configuration is invalid or unusable."""


@dataclass(frozen=True)
class SourceProblem:
    skill: str
    source: Path
    reason: str


def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "skillset" / "config.json"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except ConfigError:
        raise
    except FileNotFoundError:
        raise ConfigError(f"configuration file does not exist: {path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from None


def _source_path(value: object, *, set_name: str, skill: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"set {set_name!r} skill {skill!r}: source must be a nonempty string")
    if value.startswith("~/"):
        path = Path(os.path.expanduser(value))
    elif value.startswith("~"):
        raise ConfigError(f"set {set_name!r} skill {skill!r}: only ~/ paths are expanded")
    else:
        path = Path(value)
        if not path.is_absolute():
            raise ConfigError(f"set {set_name!r} skill {skill!r}: source must be absolute or start with ~/ ")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"set {set_name!r} skill {skill!r}: cannot normalize source: {exc}") from None


def _root_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("root paths must be nonempty absolute paths or start with '~/'.")
    if value.startswith("~/"):
        path = Path(os.path.expanduser(value))
    elif value.startswith("~"):
        raise ConfigError("root paths may only expand the ~/ prefix")
    else:
        path = Path(value)
        if not path.is_absolute():
            raise ConfigError(f"root path must be absolute or start with '~/': {value!r}")
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"cannot normalize root path {value!r}: {exc}") from None
    if resolved == Path("/"):
        raise ConfigError("filesystem root cannot be a configured root")
    try:
        info = resolved.stat()
    except FileNotFoundError:
        return resolved
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"cannot inspect root path {resolved}: {exc}") from None
    if not stat.S_ISDIR(info.st_mode):
        raise ConfigError(f"configured root is not a directory: {resolved}")
    return resolved


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Read configuration and validate every set's schema, without stat'ing sources."""
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    value = _load_json(config_path)
    if not isinstance(value, dict) or not {"schema_version", "sets"}.issubset(value) or set(value) - {
        "schema_version",
        "sets",
        "roots",
    }:
        raise ConfigError("configuration must contain schema_version and sets, with optional roots only")
    version = value["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ConfigError("unsupported schema_version (expected integer 1)")
    raw_sets = value["sets"]
    if not isinstance(raw_sets, dict):
        raise ConfigError("sets must be an object")

    sets: dict[str, SkillSet] = {}
    for name, raw_set in raw_sets.items():
        if not isinstance(name, str) or not name:
            raise ConfigError("set names must be nonempty strings")
        if not isinstance(raw_set, dict) or set(raw_set) != {"agents", "skills"}:
            raise ConfigError(f"set {name!r} must contain agents and skills only")

        raw_agents = raw_set["agents"]
        if not isinstance(raw_agents, list) or not raw_agents:
            raise ConfigError(f"set {name!r}: agents must be a nonempty array")
        agents: list[str] = []
        for agent in raw_agents:
            if agent not in ("codex", "claude"):
                raise ConfigError(f"set {name!r}: unknown agent {agent!r}")
            if agent in agents:
                raise ConfigError(f"set {name!r}: duplicate agent {agent!r}")
            agents.append(agent)

        raw_skills = raw_set["skills"]
        if not isinstance(raw_skills, dict):
            raise ConfigError(f"set {name!r}: skills must be an object")
        skills: dict[str, Path] = {}
        for skill, source in raw_skills.items():
            if not valid_skill_name(skill):
                raise ConfigError(f"set {name!r}: invalid skill name {skill!r}")
            skills[skill] = _source_path(source, set_name=name, skill=skill)
        sets[name] = SkillSet(name=name, agents=tuple(agents), skills=skills)
    raw_roots = value.get("roots", {})
    if not isinstance(raw_roots, dict):
        raise ConfigError("roots must be an object")
    roots: dict[Path, str | None] = {}
    for raw_path, set_name in raw_roots.items():
        root = _root_path(raw_path)
        if root in roots:
            raise ConfigError(f"duplicate normalized root path: {root}")
        if set_name is not None and (not isinstance(set_name, str) or set_name not in sets):
            raise ConfigError(f"root {root} must map to an existing set name or null")
        roots[root] = set_name
    return Config(sets=sets, roots=roots)


def source_problem(skill: str, source: Path) -> SourceProblem | None:
    """Return why a source cannot be used, without exposing its content."""
    try:
        directory_stat = source.stat()
        if not stat.S_ISDIR(directory_stat.st_mode):
            return SourceProblem(skill, source, "source is not a directory")
        if not os.access(source, os.R_OK | os.X_OK):
            return SourceProblem(skill, source, "source directory is not readable/searchable")
        # Opening the directory catches permission and filesystem errors that
        # os.access alone can miss. No entry names or file contents are logged.
        with os.scandir(source):
            pass
        skill_file = source / "SKILL.md"
        file_stat = skill_file.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            return SourceProblem(skill, source, "SKILL.md is not a regular file")
        if file_stat.st_size == 0:
            return SourceProblem(skill, source, "SKILL.md is empty")
        if not os.access(skill_file, os.R_OK):
            return SourceProblem(skill, source, "SKILL.md is not readable")
        with skill_file.open("rb") as stream:
            if not stream.read(1):
                return SourceProblem(skill, source, "SKILL.md is empty")
    except (OSError, RuntimeError, ValueError) as exc:
        return SourceProblem(skill, source, f"source is unavailable: {getattr(exc, 'strerror', None) or exc}")
    return None


def validate_selected_sources(skill_set: SkillSet) -> list[SourceProblem]:
    """Check that every source in one selected set is usable."""
    return [
        problem
        for skill, source in skill_set.skills.items()
        if (problem := source_problem(skill, source)) is not None
    ]
