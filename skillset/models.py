"""Typed data structures shared by skillset's readers and planners."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any


SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
DESTINATION_PREFIXES = (".agents/skills/", ".claude/skills/")


def valid_skill_name(value: object) -> bool:
    return isinstance(value, str) and SKILL_NAME_RE.fullmatch(value) is not None


def validate_destination(value: object) -> str:
    """Validate and return an allowed repo-relative skill destination."""
    if not isinstance(value, str):
        raise ValueError("destination must be a string")
    for prefix in DESTINATION_PREFIXES:
        if value.startswith(prefix) and valid_skill_name(value[len(prefix) :]):
            return value
    raise ValueError(f"invalid managed destination: {value!r}")


def validate_absolute_source(value: object) -> str:
    """Validate a normalized absolute path without resolving it again."""
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError("source must be an absolute path")
    if "\x00" in value:
        raise ValueError("source path contains a NUL byte")
    if os.path.normpath(value) != value:
        raise ValueError("source path is not normalized")
    parts = value.split("/")
    if any(part in (".", "..") for part in parts):
        raise ValueError("source path contains a dot component")
    if "//" in value or (len(value) > 1 and value.endswith("/")):
        raise ValueError("source path is not normalized")
    return value


@dataclass(frozen=True)
class SkillSet:
    name: str
    agents: tuple[str, ...]
    skills: dict[str, Path]


@dataclass(frozen=True)
class Config:
    sets: dict[str, SkillSet]
    roots: dict[Path, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class State:
    active_set: str | None = None
    links: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "active_set": self.active_set,
            "links": {key: self.links[key] for key in sorted(self.links)},
        }


@dataclass(frozen=True)
class Operation:
    action: str
    destination: str
    source: str
    previous_source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "destination": self.destination,
            "source": self.source,
            "previous_source": self.previous_source,
        }


@dataclass(frozen=True)
class DiffRow:
    label: str
    destination: str
    source: str


@dataclass(frozen=True)
class Plan:
    rows: tuple[DiffRow, ...]
    operations: tuple[Operation, ...]
    after: State


@dataclass(frozen=True)
class Journal:
    before: State
    after: State
    operations: tuple[Operation, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "before": self.before.to_json(),
            "after": self.after.to_json(),
            "operations": [op.to_json() for op in self.operations],
        }


def state_from_json(value: object) -> State:
    if not isinstance(value, dict) or set(value) != {"schema_version", "active_set", "links"}:
        raise ValueError("state must contain schema_version, active_set, and links only")
    version = value["schema_version"]
    if type(version) is not int or version != 1:
        raise ValueError("unsupported state schema_version")
    active_set = value["active_set"]
    if active_set is not None and (not isinstance(active_set, str) or not active_set):
        raise ValueError("active_set must be a nonempty string or null")
    raw_links = value["links"]
    if not isinstance(raw_links, dict):
        raise ValueError("links must be an object")
    links: dict[str, str] = {}
    for destination, source in raw_links.items():
        links[validate_destination(destination)] = validate_absolute_source(source)
    if active_set is None and links:
        raise ValueError("unmanaged state cannot own links")
    return State(active_set=active_set, links=links)


def operation_from_json(value: object) -> Operation:
    if not isinstance(value, dict) or set(value) != {
        "action",
        "destination",
        "source",
        "previous_source",
    }:
        raise ValueError("operation must contain action, destination, source, and previous_source only")
    action = value["action"]
    if action not in ("add", "update", "remove"):
        raise ValueError("unknown operation action")
    destination = validate_destination(value["destination"])
    source = validate_absolute_source(value["source"])
    previous = value["previous_source"]
    if previous is not None:
        previous = validate_absolute_source(previous)
    if action == "update" and (previous is None or previous == source):
        raise ValueError("update requires a different previous_source")
    if action in ("add", "remove") and previous is not None:
        raise ValueError(f"{action} operation cannot have previous_source")
    return Operation(action, destination, source, previous)
