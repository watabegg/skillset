"""Manifest, pending journal, filesystem safety, locking, and link mutations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import Any, Iterator

from .config import source_problem
from .models import Journal, Operation, Plan, State, operation_from_json, state_from_json, validate_destination
from .planner import plan_remove, plan_set


class StoreError(Exception):
    """Base class for expected state-storage failures."""


class CorruptStoreError(StoreError):
    """A state or pending JSON file is malformed or unsafe internally."""


class UnsafePathError(StoreError):
    """A managed path is a symlink or has an unsafe filesystem shape."""


class PendingJournalError(StoreError):
    """An unresolved journal blocks a requested mutation."""


class LockBusyError(StoreError):
    """Another skillset process holds the repository lock."""


class ConcurrentChangeError(StoreError):
    """A destination changed after planning and before its operation."""


class SelectedSourceError(StoreError):
    """A selected source became unavailable while re-planning under lock."""


@dataclass(frozen=True)
class Snapshot:
    state: State
    journal: Journal | None


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _check_directory(path: Path, *, label: str) -> bool:
    """Validate a path without following its final component; return existence."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise UnsafePathError(f"unsafe {label}: expected an ordinary directory at {path}")
    return True


def _check_regular_file(path: Path, *, label: str) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafePathError(f"unsafe {label}: expected an ordinary file at {path}")
    return True


def validate_internal_layout(root: Path) -> None:
    """Reject symlink/non-directory metadata parents and non-file metadata."""
    skillset_dir = root / ".skillset"
    if not _check_directory(skillset_dir, label=".skillset directory"):
        return
    for name in ("state.json", "pending.json", "lock"):
        _check_regular_file(skillset_dir / name, label=f".skillset/{name}")


def validate_managed_parents(root: Path, destinations: set[str] | tuple[str, ...] | list[str]) -> None:
    """Check relevant .agents/.claude parents without following symlinks."""
    for destination in destinations:
        relative = validate_destination(destination)
        first, second, _name = relative.split("/")
        _check_directory(root / first, label=f"{first} parent")
        _check_directory(root / first / second, label=f"{first}/{second} parent")


def _read_json_file(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_json_pairs)
    except Exception as exc:
        raise CorruptStoreError(f"cannot read {path}: {exc}") from None


def _journal_from_json(value: object) -> Journal:
    if not isinstance(value, dict) or set(value) != {"schema_version", "before", "after", "operations"}:
        raise ValueError("journal must contain schema_version, before, after, and operations only")
    version = value["schema_version"]
    if type(version) is not int or version != 1:
        raise ValueError("unsupported journal schema_version")
    before = state_from_json(value["before"])
    after = state_from_json(value["after"])
    raw_operations = value["operations"]
    if not isinstance(raw_operations, list):
        raise ValueError("journal operations must be an array")
    operations = tuple(operation_from_json(item) for item in raw_operations)
    destinations = [item.destination for item in operations]
    if len(set(destinations)) != len(destinations):
        raise ValueError("journal contains duplicate destinations")
    for operation in operations:
        old = before.links.get(operation.destination)
        new = after.links.get(operation.destination)
        if operation.action == "add":
            if new != operation.source:
                raise ValueError("add operation does not match after state")
        elif operation.action == "update":
            if old != operation.previous_source or new != operation.source:
                raise ValueError("update operation does not match before/after states")
        elif old != operation.source or new is not None:
            raise ValueError("remove operation does not match before/after states")
    changed_links = {
        destination
        for destination in set(before.links) | set(after.links)
        if before.links.get(destination) != after.links.get(destination)
    }
    if not changed_links.issubset(set(destinations)):
        raise ValueError("journal omits a changed link from operations")
    return Journal(before=before, after=after, operations=operations)


def load_snapshot(root: Path) -> Snapshot:
    """Load and validate state and pending journal without creating files."""
    validate_internal_layout(root)
    directory = root / ".skillset"
    state_path = directory / "state.json"
    journal_path = directory / "pending.json"
    state = State()
    if _exists(state_path):
        value = _read_json_file(state_path)
        try:
            state = state_from_json(value)
        except (TypeError, ValueError) as exc:
            raise CorruptStoreError(f"invalid state file {state_path}: {exc}") from None
    journal: Journal | None = None
    if _exists(journal_path):
        value = _read_json_file(journal_path)
        try:
            journal = _journal_from_json(value)
        except (TypeError, ValueError) as exc:
            raise CorruptStoreError(f"invalid pending journal {journal_path}: {exc}") from None
    return Snapshot(state=state, journal=journal)


def _write_json_atomic(path: Path, value: object) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _current_target(path: Path) -> tuple[bool, str | None, bool]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False, None, False
    if stat.S_ISLNK(info.st_mode):
        return True, os.readlink(path), True
    return True, None, False


def _ensure_parent_directories(root: Path, destination: str) -> Path:
    relative = validate_destination(destination)
    parts = relative.split("/")
    current = root
    for component in parts[:-1]:
        current = current / component
        if not _check_directory(current, label=f"managed parent {current}"):
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                pass
            _check_directory(current, label=f"managed parent {current}")
    return current / parts[-1]


def _check_expected(operation: Operation, path: Path) -> tuple[bool, str | None, bool]:
    exists, raw_target, is_link = _current_target(path)
    if operation.action == "add":
        if exists:
            raise ConcurrentChangeError(f"destination appeared after planning: {path}")
    elif operation.action == "update":
        if not exists or not is_link or raw_target != operation.previous_source:
            raise ConcurrentChangeError(f"owned link changed after planning: {path}")
    elif operation.action == "remove":
        if exists and (not is_link or raw_target != operation.source):
            raise ConcurrentChangeError(f"owned link changed after planning: {path}")
    return exists, raw_target, is_link


def _install_link(operation: Operation, destination: Path) -> None:
    if operation.action == "add":
        os.symlink(operation.source, destination)
        return

    # Update a link through a temporary symlink and atomic rename, checking the
    # old value again after creating the temporary entry.
    temp_link = destination.parent / f".skillset-link-{secrets.token_hex(8)}"
    os.symlink(operation.source, temp_link)
    try:
        exists, raw_target, is_link = _current_target(destination)
        if not exists or not is_link or raw_target != operation.previous_source:
            raise ConcurrentChangeError(f"owned link changed after planning: {destination}")
        os.replace(temp_link, destination)
    finally:
        try:
            temp_link.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    directory = root / ".skillset"
    if not _check_directory(directory, label=".skillset directory"):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            # Another first-time invocation may create this before acquiring
            # the shared lock. Validate it, then contend on that same lock.
            pass
        _check_directory(directory, label=".skillset directory")
    lock_path = directory / "lock"
    _check_regular_file(lock_path, label=".skillset/lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno in (getattr(os, "ELOOP", 40),):
            raise UnsafePathError(f"unsafe .skillset/lock: {exc}") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafePathError(f"unsafe .skillset/lock: expected an ordinary file at {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockBusyError(f"another skillset mutation holds {lock_path}") from None
        yield
    finally:
        os.close(descriptor)


def _replan(root: Path, state: State, requested: Plan) -> Plan:
    if requested.after.active_set is None:
        return plan_remove(root, state)
    unavailable: set[str] = set()
    for destination, source in requested.after.links.items():
        skill = destination.rsplit("/", 1)[-1]
        if source_problem(skill, Path(source)) is not None:
            unavailable.add(source)
    return plan_set(
        root,
        state,
        requested.after.links,
        active_set=requested.after.active_set,
        unavailable_sources=unavailable,
    )


def _raise_for_plan(plan: Plan) -> None:
    if any(row.label == "conflict" for row in plan.rows):
        raise ConcurrentChangeError("a destination conflicts with the current plan")
    if any(row.label == "broken" for row in plan.rows):
        raise SelectedSourceError("a selected skill source is unavailable")


def execute_plan(root: Path, expected_before: State, requested: Plan) -> None:
    """Apply a preflighted plan under flock, leaving pending.json on failure."""
    validate_internal_layout(root)
    validate_managed_parents(root, tuple(set(expected_before.links) | set(requested.after.links)))
    initial = load_snapshot(root)
    if initial.journal is not None:
        raise PendingJournalError("pending journal requires manual recovery")
    if initial.state != expected_before:
        raise ConcurrentChangeError("state changed after the plan was calculated")
    _raise_for_plan(requested)
    if not requested.operations and requested.after == expected_before:
        return

    with _locked(root):
        validate_internal_layout(root)
        current = load_snapshot(root)
        if current.journal is not None:
            raise PendingJournalError("pending journal requires manual recovery")
        replan = _replan(root, current.state, requested)
        _raise_for_plan(replan)
        relevant = tuple(set(current.state.links) | set(replan.after.links))
        validate_managed_parents(root, relevant)
        if not replan.operations and replan.after == current.state:
            return

        journal_path = root / ".skillset" / "pending.json"
        if _exists(journal_path):
            raise PendingJournalError("pending journal requires manual recovery")
        journal = Journal(current.state, replan.after, replan.operations)
        _write_json_atomic(journal_path, journal.to_json())

        for operation in sorted(replan.operations, key=lambda item: item.destination):
            validate_internal_layout(root)
            validate_managed_parents(root, (operation.destination,))
            if operation.action == "remove":
                destination = root / operation.destination
            else:
                destination = _ensure_parent_directories(root, operation.destination)
            exists, _target, _is_link = _check_expected(operation, destination)
            if operation.action == "remove":
                if exists:
                    destination.unlink()
            else:
                _install_link(operation, destination)

        _write_json_atomic(root / ".skillset" / "state.json", replan.after.to_json())
        validate_internal_layout(root)
        journal_path.unlink()
        _fsync_directory(journal_path.parent)
