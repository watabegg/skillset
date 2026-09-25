"""Root-based discovery and safe synchronization for registered projects."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Iterator, Sequence

from .cli import _check_sources_are_outside_managed_trees, _destination_map, _print_header, _print_rows, _unavailable_sources
from .config import ConfigError, load_config
from .models import Config, Plan
from .planner import plan_set
from .storage import (
    ConcurrentChangeError,
    CorruptStoreError,
    LockBusyError,
    PendingJournalError,
    SelectedSourceError,
    UnsafePathError,
    execute_plan,
    load_snapshot,
    validate_managed_parents,
)


IGNORED_DIRS = {
    ".git",
    ".skillset",
    ".agents",
    ".claude",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    "dist",
    "build",
    "target",
    "vendor",
}


def _normalized(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OSError(f"cannot normalize path {path}: {exc}") from None


def select_set(config: Config, project: Path) -> str | None:
    """Return the set selected by the most specific containing root rule."""
    project_path = _normalized(project)
    matches: list[tuple[int, Path, str | None]] = []
    for root, set_name in config.roots.items():
        if project_path == root or project_path.is_relative_to(root):
            matches.append((len(root.parts), root, set_name))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], str(item[1])))[2]


def _git_project(candidate: Path) -> bool:
    """Recognize a complete, non-bare worktree rooted exactly at candidate."""
    marker = candidate / ".git"
    try:
        marker_info = marker.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(marker_info.st_mode) or not (
        stat.S_ISDIR(marker_info.st_mode) or stat.S_ISREG(marker_info.st_mode)
    ):
        return False
    try:
        top = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise
    if top.returncode != 0:
        return False
    try:
        bare = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--is-bare-repository"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise
    if bare.returncode != 0 or bare.stdout.strip() != "false":
        return False
    try:
        reported_root = Path(top.stdout.strip()).resolve(strict=False)
        actual_root = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False
    return reported_root == actual_root


def discover_projects(config: Config, start: Path | None = None) -> list[Path]:
    """Find sorted, real Git worktree roots within enabled registered scopes."""
    configured_roots = set(config.roots)
    scan_roots: set[Path] = set()
    if start is None:
        scan_roots.update(root for root, name in config.roots.items() if name is not None)
    else:
        start_path = _normalized(start)
        if select_set(config, start_path) is not None:
            scan_roots.add(start_path)
        scan_roots.update(
            root
            for root, name in config.roots.items()
            if name is not None and (root == start_path or root.is_relative_to(start_path))
        )

    projects: set[Path] = set()

    def visit(directory: Path, scan_root: Path, boundaries: set[Path]) -> None:
        if directory != scan_root and directory in boundaries:
            return
        try:
            info = directory.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            return
        if _git_project(directory):
            resolved = directory.resolve(strict=True)
            if select_set(config, resolved) is not None:
                projects.add(resolved)
            # A registered nested root is scanned independently below.
            return
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except FileNotFoundError:
            return
        for entry in children:
            if entry.name in IGNORED_DIRS:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except FileNotFoundError:
                continue
            visit(Path(entry.path), scan_root, boundaries)

    for scan_root in sorted(scan_roots, key=str):
        boundaries = {
            root for root in configured_roots if root != scan_root and root.is_relative_to(scan_root)
        }
        visit(scan_root, scan_root, boundaries)
    return sorted(projects, key=str)


def _git_toplevel(project: Path) -> Path:
    target = project.expanduser()
    try:
        top = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise OSError("git executable was not found") from None
    if top.returncode != 0:
        raise ValueError(f"not a Git working tree: {target}")
    root = Path(top.stdout.strip()).resolve(strict=False)
    try:
        bare = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--is-bare-repository"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise OSError("git executable was not found") from None
    if bare.returncode != 0:
        raise ValueError(f"cannot inspect Git repository: {target}")
    if bare.stdout.strip() == "true":
        raise ValueError(f"bare Git repositories are not supported: {target}")
    if not root.is_dir():
        raise ValueError(f"Git returned an invalid project root: {root}")
    return root


def _git_exclude(root: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-path", "info/exclude"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise OSError("git executable was not found") from None
    if result.returncode != 0 or not result.stdout.strip():
        raise OSError(f"cannot locate Git info/exclude for {root}")
    reported = Path(os.path.abspath(result.stdout.strip()))
    try:
        common = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise OSError("git executable was not found") from None
    if common.returncode != 0 or not common.stdout.strip():
        raise OSError(f"cannot locate Git common directory for {root}")
    common_dir = Path(os.path.abspath(common.stdout.strip()))
    _check_plain_directory(common_dir, "Git common directory")
    candidate = common_dir / "info" / "exclude"
    # Git may print the symlink target for info/exclude. Keep the lexical path
    # so preflight can reject a symlink instead of accidentally editing it.
    if candidate.resolve(strict=False) != reported.resolve(strict=False):
        raise OSError(f"Git returned inconsistent info/exclude paths for {root}")
    return candidate


def _check_plain_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise UnsafePathError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise UnsafePathError(f"unsafe {label}: expected an ordinary directory at {path}")


def _read_exclude(path: Path) -> bytes:
    _check_plain_directory(path.parent, "Git info directory")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return b""
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafePathError(f"unsafe Git info/exclude: expected an ordinary file at {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafePathError(f"unsafe Git info/exclude: expected an ordinary file at {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65536):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _exclude_lines(destinations: Sequence[str]) -> list[bytes]:
    lines = [b"/.skillset/"]
    for destination in sorted(set(destinations)):
        if "\n" in destination or "\r" in destination:
            raise ValueError("managed destination contains a newline")
        lines.append(("/" + destination).encode("ascii"))
    return lines


def _preflight_exclude(path: Path) -> None:
    _read_exclude(path)
    lock_path = path.with_name(path.name + ".skillset.lock")
    try:
        info = lock_path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafePathError(f"unsafe Git exclude lock: expected an ordinary file at {lock_path}")


@contextmanager
def _exclude_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(path.name + ".skillset.lock")
    _check_plain_directory(lock_path.parent, "Git info directory")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno == getattr(os, "ELOOP", 40):
            raise UnsafePathError(f"unsafe Git exclude lock: {lock_path}") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafePathError(f"unsafe Git exclude lock: expected an ordinary file at {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockBusyError(f"another skillset sync holds {lock_path}") from None
        yield
    finally:
        os.close(descriptor)


def _append_missing_excludes_locked(path: Path, lines: Sequence[bytes]) -> bool:
    contents = _read_exclude(path)
    existing = {line.rstrip(b"\r") for line in contents.splitlines()}
    missing = [line for line in lines if line not in existing]
    if not missing:
        return False
    prefix = b"" if not contents or contents.endswith(b"\n") else b"\n"
    payload = prefix + b"\n".join(missing) + b"\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafePathError(f"unsafe Git info/exclude: expected an ordinary file at {path}")
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _tracked_destinations(root: Path, destinations: Sequence[str]) -> list[str]:
    if not destinations:
        return []
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        raise OSError("git executable was not found") from None
    if result.returncode != 0:
        raise OSError(f"cannot inspect tracked paths in {root}: {os.fsdecode(result.stderr).strip()}")
    tracked = [os.fsdecode(item) for item in result.stdout.split(b"\0") if item]
    conflicts: set[str] = set()
    for destination in destinations:
        prefix = destination.rstrip("/") + "/"
        for path in tracked:
            if path == destination or path.startswith(prefix):
                conflicts.add(destination)
                break
    return sorted(conflicts)


def _print_custom_conflict(
    root: Path,
    message: str,
    *,
    destination: str | None = None,
    active_set: str | None = None,
) -> None:
    print(f"project: {root}", flush=True)
    _print_header(active_set)
    absolute = (root / destination).absolute() if destination is not None else root
    print(f"conflict\t{absolute}\t{message}")


def _process_project(
    config: Config,
    root: Path,
    selected: str,
    *,
    dry_run: bool,
    quiet: bool,
) -> tuple[int, bool]:
    """Sync one project; return its status and whether it made visible changes."""
    snapshot = load_snapshot(root)
    if snapshot.journal is not None:
        raise PendingJournalError("pending journal requires manual recovery")
    if snapshot.state.active_set is not None and snapshot.state.active_set != selected:
        _print_custom_conflict(
            root,
            f"active set {snapshot.state.active_set!r} differs from root mapping {selected!r}",
            active_set=snapshot.state.active_set,
        )
        return 3, False

    skill_set = config.sets[selected]
    desired = _destination_map(skill_set)
    validate_managed_parents(root, tuple(set(snapshot.state.links) | set(desired)))
    _check_sources_are_outside_managed_trees(root, skill_set)
    unavailable, source_messages = _unavailable_sources(skill_set)
    plan = plan_set(
        root,
        snapshot.state,
        desired,
        active_set=selected,
        unavailable_sources=unavailable,
    )
    if any(row.label == "conflict" for row in plan.rows):
        print(f"project: {root}", flush=True)
        _print_header(selected)
        _print_rows(root, plan.rows)
        return 3, False
    if any(row.label == "broken" for row in plan.rows):
        print(f"project: {root}", flush=True)
        _print_header(selected)
        _print_rows(root, plan.rows)
        for message in source_messages:
            print(f"error: selected source is unavailable: {message}", file=sys.stderr)
        return 2, False

    link_destinations = tuple(sorted(set(snapshot.state.links) | set(plan.after.links)))
    tracked = _tracked_destinations(root, (".skillset", *link_destinations))
    if tracked:
        _print_custom_conflict(
            root,
            "tracked in Git index",
            destination=tracked[0],
            active_set=snapshot.state.active_set,
        )
        return 3, False

    exclude_path = _git_exclude(root)
    exclude_lines = _exclude_lines(tuple(plan.after.links))
    _preflight_exclude(exclude_path)
    if dry_run:
        if not quiet or plan.operations or plan.after != snapshot.state:
            print(f"project: {root}", flush=True)
            _print_header(selected)
            _print_rows(root, plan.rows)
        return 0, False

    changed = bool(plan.operations or plan.after != snapshot.state)
    with _exclude_lock(exclude_path):
        _preflight_exclude(exclude_path)
        if changed:
            print(f"project: {root}", flush=True)
            _print_header(selected)
            _print_rows(root, plan.rows)
            execute_plan(root, snapshot.state, plan)
        excludes_changed = _append_missing_excludes_locked(exclude_path, exclude_lines)
    if not quiet and not changed:
        print(f"project: {root}", flush=True)
        _print_header(selected)
        _print_rows(root, plan.rows)
    elif quiet and not changed and excludes_changed:
        print(f"project: {root}", flush=True)
        _print_header(selected)
        _print_rows(root, plan.rows)
    return 0, changed or excludes_changed


def sync_projects(
    config_path: Path,
    project_root: Path | None = None,
    *,
    dry_run: bool = False,
    quiet: bool = False,
) -> int:
    """Reconcile registered Git projects, returning the public CLI status."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if project_root is not None:
        try:
            root = _git_toplevel(project_root)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"error: system I/O failure: {exc}", file=sys.stderr)
            return 1
        selected = select_set(config, root)
        if selected is None:
            if not quiet:
                print(f"unmatched\t{root}")
            return 0
        projects = [(root, selected)]
    else:
        try:
            roots = discover_projects(config)
        except OSError as exc:
            print(f"error: project discovery failed: {exc}", file=sys.stderr)
            return 1
        projects = [(root, select_set(config, root)) for root in roots]
        projects = [(root, selected) for root, selected in projects if selected is not None]

    saw_io = False
    saw_conflict = False
    saw_invalid = False
    for root, selected in projects:
        assert selected is not None
        try:
            status, _changed = _process_project(
                config,
                root,
                selected,
                dry_run=dry_run,
                quiet=quiet,
            )
        except (CorruptStoreError, OSError) as exc:
            print(f"project: {root}", flush=True)
            print(f"error: {exc}", file=sys.stderr)
            saw_io = True
        except (UnsafePathError, PendingJournalError, LockBusyError, ConcurrentChangeError) as exc:
            print(f"project: {root}", flush=True)
            print(f"error: {exc}", file=sys.stderr)
            saw_conflict = True
        except SelectedSourceError as exc:
            print(f"project: {root}", flush=True)
            print(f"error: {exc}", file=sys.stderr)
            saw_invalid = True
        except ConfigError as exc:
            print(f"project: {root}", flush=True)
            print(f"error: {exc}", file=sys.stderr)
            saw_invalid = True
        except Exception as exc:
            print(f"project: {root}", flush=True)
            print(f"error: unexpected system failure: {exc}", file=sys.stderr)
            saw_io = True
        else:
            if status == 1:
                saw_io = True
            elif status == 2:
                saw_invalid = True
            elif status == 3:
                saw_conflict = True

    if saw_io:
        return 1
    if saw_conflict:
        return 3
    if saw_invalid:
        return 2
    return 0
