"""Read-only link inspection and deterministic set diff planning."""

from __future__ import annotations

import os
from pathlib import Path

from .models import DiffRow, Operation, Plan, State, validate_absolute_source, validate_destination


def _current_link(path: Path) -> tuple[bool, str | None, bool]:
    """Return (exists, raw symlink target, is_symlink), following no links."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False, None, False
    if os.path.islink(path):
        return True, os.readlink(path), True
    return True, None, False


def plan_set(
    root: Path,
    state: State,
    desired: dict[str, str],
    *,
    active_set: str,
    unavailable_sources: set[str] | None = None,
) -> Plan:
    """Plan an apply/switch without touching the filesystem."""
    unavailable = unavailable_sources or set()
    normalized_desired = {
        validate_destination(destination): validate_absolute_source(source)
        for destination, source in desired.items()
    }
    rows: list[DiffRow] = []
    operations: list[Operation] = []

    for destination in sorted(set(state.links) | set(normalized_desired)):
        path = root / destination
        exists, raw_target, is_link = _current_link(path)
        old_source = state.links.get(destination)
        new_source = normalized_desired.get(destination)

        if old_source is None:
            if exists:
                # A same-target symlink can still have been created by another
                # actor; only the state file can establish tool ownership.
                rows.append(DiffRow("conflict", destination, new_source or "-"))
            elif new_source is not None:
                label = "broken" if new_source in unavailable else "add"
                rows.append(DiffRow(label, destination, new_source))
                operations.append(Operation("add", destination, new_source))
            continue

        # The ownership check always precedes source availability checks.
        if exists and (not is_link or raw_target != old_source):
            rows.append(DiffRow("conflict", destination, new_source or old_source))
            continue

        if not exists:
            if new_source is None:
                rows.append(DiffRow("remove", destination, old_source))
                operations.append(Operation("remove", destination, old_source))
            else:
                label = "broken" if new_source in unavailable else "add"
                rows.append(DiffRow(label, destination, new_source))
                operations.append(Operation("add", destination, new_source))
            continue

        if new_source is None:
            rows.append(DiffRow("remove", destination, old_source))
            operations.append(Operation("remove", destination, old_source))
        elif new_source in unavailable:
            rows.append(DiffRow("broken", destination, new_source))
            operations.append(Operation("update", destination, new_source, old_source))
        elif new_source == old_source:
            rows.append(DiffRow("keep", destination, old_source))
        else:
            rows.append(DiffRow("update", destination, new_source))
            operations.append(Operation("update", destination, new_source, old_source))

    after = State(active_set=active_set, links=normalized_desired)
    return Plan(tuple(rows), tuple(operations), after)


def plan_remove(root: Path, state: State) -> Plan:
    """Plan removal of all state-owned links, retaining conflicts safely."""
    rows: list[DiffRow] = []
    operations: list[Operation] = []
    for destination in sorted(state.links):
        source = state.links[destination]
        exists, raw_target, is_link = _current_link(root / destination)
        if exists and (not is_link or raw_target != source):
            rows.append(DiffRow("conflict", destination, source))
        else:
            rows.append(DiffRow("remove", destination, source))
            operations.append(Operation("remove", destination, source))
    return Plan(tuple(rows), tuple(operations), State())


def inspect_recorded(root: Path, state: State, *, check_source) -> tuple[DiffRow, ...]:
    """Inspect only the manifest's links, without consulting current config."""
    rows: list[DiffRow] = []
    for destination in sorted(state.links):
        source = state.links[destination]
        exists, raw_target, is_link = _current_link(root / destination)
        if exists and (not is_link or raw_target != source):
            label = "conflict"
        elif not exists:
            label = "broken"
        elif check_source(Path(source)) is not None:
            label = "broken"
        else:
            label = "keep"
        rows.append(DiffRow(label, destination, source))
    return tuple(rows)

