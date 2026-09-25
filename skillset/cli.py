"""Command-line interface for project-local skill sets."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Iterator, Sequence

from .config import ConfigError, _load_json, _root_path, default_config_path, load_config, source_problem
from .models import DiffRow, Plan, SkillSet, State
from .planner import inspect_recorded, plan_remove, plan_set
from .storage import (
    ConcurrentChangeError,
    CorruptStoreError,
    LockBusyError,
    PendingJournalError,
    SelectedSourceError,
    Snapshot,
    UnsafePathError,
    execute_plan,
    load_snapshot,
    validate_managed_parents,
)


class ProjectError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skillset", description="Manage project-scoped agent skill links.")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="show recorded state or a requested set's diff")
    status.add_argument("--set", dest="set_name", metavar="NAME")
    status.add_argument("--project-root", metavar="DIR")
    status.add_argument("--config", metavar="FILE")

    apply = commands.add_parser("apply", help="apply one named skill set")
    apply.add_argument("name", metavar="NAME")
    apply.add_argument("--dry-run", action="store_true")
    apply.add_argument("--project-root", metavar="DIR")
    apply.add_argument("--config", metavar="FILE")

    remove = commands.add_parser("remove", help="remove links recorded by skillset")
    remove.add_argument("--dry-run", action="store_true")
    remove.add_argument("--project-root", metavar="DIR")
    remove.add_argument("--config", metavar="FILE")

    sync = commands.add_parser("sync", help="synchronize projects under registered roots")
    sync.add_argument("--project-root", metavar="DIR")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--config", metavar="FILE")

    watch = commands.add_parser("watch", help="watch registered roots and synchronize projects")
    watch.add_argument("--config", metavar="FILE")

    service = commands.add_parser("service", help="manage the user-level skillset watch service")
    service.add_argument("action", choices=("install", "start", "stop", "status", "uninstall"))
    service.add_argument("--config", metavar="FILE")

    roots = commands.add_parser("roots", help="manage automatic synchronization roots")
    root_commands = roots.add_subparsers(dest="roots_command", required=True)
    root_add = root_commands.add_parser("add", help="register a root with a skill set")
    root_add.add_argument("directory", metavar="DIR")
    root_add.add_argument("--set", dest="set_name", metavar="NAME", required=True)
    root_add.add_argument("--config", metavar="FILE")
    root_disable = root_commands.add_parser("disable", help="disable automatic sync under a root")
    root_disable.add_argument("directory", metavar="DIR")
    root_disable.add_argument("--config", metavar="FILE")
    root_remove = root_commands.add_parser("remove", help="remove a root rule")
    root_remove.add_argument("directory", metavar="DIR")
    root_remove.add_argument("--config", metavar="FILE")
    root_list = root_commands.add_parser("list", help="list configured root rules")
    root_list.add_argument("--config", metavar="FILE")
    return parser


def _project_root(directory: str | None) -> Path:
    target = Path(directory).expanduser() if directory else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise ProjectError("git executable was not found") from None
    except subprocess.CalledProcessError:
        raise ProjectError(f"not a Git working tree: {target}") from None
    root = Path(result.stdout.strip()).resolve(strict=False)
    if not root.is_dir():
        raise ProjectError(f"Git returned an invalid project root: {root}")
    try:
        bare = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--is-bare-repository"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        raise ProjectError(f"cannot inspect Git repository: {target}") from None
    if bare.stdout.strip() == "true":
        raise ProjectError(f"bare Git repositories are not supported: {target}")
    return root


def _destination_map(skill_set: SkillSet) -> dict[str, str]:
    result: dict[str, str] = {}
    for agent in skill_set.agents:
        prefix = ".agents/skills" if agent == "codex" else ".claude/skills"
        for skill, source in skill_set.skills.items():
            result[f"{prefix}/{skill}"] = str(source)
    return result


def _check_sources_are_outside_managed_trees(root: Path, skill_set: SkillSet) -> None:
    managed_roots = (
        (root / ".agents" / "skills").resolve(strict=False),
        (root / ".claude" / "skills").resolve(strict=False),
    )
    for skill, source in skill_set.skills.items():
        for managed_root in managed_roots:
            try:
                inside_managed_tree = source == managed_root or source.is_relative_to(managed_root)
            except (OSError, RuntimeError):
                inside_managed_tree = False
            if inside_managed_tree:
                raise ConfigError(
                    f"set {skill_set.name!r} skill {skill!r}: source resolves into a managed destination tree"
                )


def _unavailable_sources(skill_set: SkillSet) -> tuple[set[str], list[str]]:
    unavailable: set[str] = set()
    diagnostics: list[str] = []
    for skill, source in skill_set.skills.items():
        problem = source_problem(skill, source)
        if problem is not None:
            unavailable.add(str(source))
            diagnostics.append(f"{problem.reason}: {source}")
    return unavailable, diagnostics


def _print_header(active_set: str | None) -> None:
    print(f"active_set: {active_set}" if active_set else "unmanaged")


def _print_rows(root: Path, rows: Sequence[DiffRow]) -> None:
    for row in sorted(rows, key=lambda item: item.destination):
        destination = (root / row.destination).absolute()
        print(f"{row.label}\t{destination}\t{row.source}")


def _print_plan(root: Path, active_set: str | None, plan: Plan, *, project: bool = False) -> None:
    if project:
        print(f"project: {root}", flush=True)
    _print_header(active_set)
    _print_rows(root, plan.rows)


def _has_label(plan: Plan, label: str) -> bool:
    return any(row.label == label for row in plan.rows)


def _configuration_path(argument: str | None) -> Path:
    return Path(argument).expanduser() if argument is not None else default_config_path()


def _root_argument(directory: str, *, must_exist: bool) -> Path:
    try:
        root = Path(directory).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"cannot normalize root path {directory!r}: {exc}") from None
    if root == Path("/"):
        raise ConfigError("filesystem root cannot be registered")
    if must_exist:
        try:
            info = root.lstat()
        except OSError as exc:
            raise ConfigError(f"root directory does not exist: {root}") from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ConfigError(f"root path must resolve to an ordinary directory: {root}")
    return root


@contextmanager
def _config_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.skillset.lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno == getattr(os, "ELOOP", 40):
            raise UnsafePathError(f"unsafe configuration lock: {lock_path}") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafePathError(f"unsafe configuration lock: expected an ordinary file at {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockBusyError(f"another skillset root update holds {lock_path}") from None
        yield
    finally:
        os.close(descriptor)


def _write_config_atomic(path: Path, value: object) -> None:
    try:
        existing = path.lstat()
    except OSError as exc:
        raise ConfigError(f"cannot inspect configuration {path}: {exc}") from None
    if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
        raise ConfigError(f"configuration must be an ordinary file: {path}")
    mode = stat.S_IMODE(existing.st_mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _roots_command(args: argparse.Namespace) -> int:
    config_path = _configuration_path(args.config)
    config = load_config(config_path)
    if args.roots_command == "list":
        for root, set_name in sorted(config.roots.items(), key=lambda item: str(item[0])):
            print(f"{root}\t{set_name if set_name is not None else 'disabled'}")
        return 0

    must_exist = args.roots_command in ("add", "disable")
    root = _root_argument(args.directory, must_exist=must_exist)
    if args.roots_command == "add" and args.set_name not in config.sets:
        raise ConfigError(f"unknown set: {args.set_name}")

    try:
        config_path = config_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"cannot resolve configuration path: {exc}") from None
    with _config_lock(config_path):
        current_config = load_config(config_path)
        if args.roots_command == "add" and args.set_name not in current_config.sets:
            raise ConfigError(f"unknown set: {args.set_name}")
        raw = _load_json(config_path)
        if not isinstance(raw, dict):
            raise ConfigError("configuration must be an object")
        roots = raw.get("roots", {})
        if not isinstance(roots, dict):
            raise ConfigError("roots must be an object")
        key = str(root)
        matching_keys = [raw_key for raw_key in roots if _root_path(raw_key) == root]
        existing_key = matching_keys[0] if matching_keys else None
        if args.roots_command == "add":
            value: str | None = args.set_name
        elif args.roots_command == "disable":
            value = None
        elif args.roots_command == "remove":
            value = None
        else:
            raise ConfigError(f"unknown roots command: {args.roots_command}")

        if args.roots_command == "remove":
            changed = existing_key is not None
            if changed:
                del roots[existing_key]
        else:
            target_key = existing_key if existing_key is not None else key
            changed = target_key not in roots or roots[target_key] != value
            roots[target_key] = value
        if changed:
            raw["roots"] = roots
            _write_config_atomic(config_path, raw)

    if args.roots_command == "add":
        print(f"registered\t{root}\t{args.set_name}")
    elif args.roots_command == "disable":
        print(f"disabled\t{root}")
    return 0


def _path_exists_including_symlink(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _global_roots(agents: Sequence[str]) -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    if "codex" in agents:
        roots.extend((home / ".agents" / "skills", home / ".codex" / "skills"))
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            roots.append(Path(codex_home).expanduser() / "skills")
    if "claude" in agents:
        roots.append(home / ".claude" / "skills")

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve(strict=False))
        except (OSError, RuntimeError):
            key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _warn_global_collisions(skill_set: SkillSet) -> None:
    names = set(skill_set.skills)
    for root in _global_roots(skill_set.agents):
        try:
            if not _path_exists_including_symlink(root):
                continue
            with os.scandir(root) as entries:
                children = {entry.name for entry in entries}
            for name in sorted(names & children):
                skill_file = root / name / "SKILL.md"
                try:
                    skill_file.lstat()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    print(f"warning: cannot inspect global skill {skill_file}: {exc}", file=sys.stderr)
                    continue
                print(f"warning: global skill with the same name exists: {name} at {skill_file}", file=sys.stderr)
        except (OSError, NotADirectoryError) as exc:
            print(f"warning: cannot scan global skills at {root}: {exc}", file=sys.stderr)


def _status_pending(root: Path, snapshot: Snapshot) -> int:
    journal = snapshot.journal
    assert journal is not None
    _print_header(snapshot.state.active_set)
    print("pending: manual recovery required")
    print(f"pending_before_active_set: {journal.before.active_set or 'unmanaged'}")
    print(f"pending_after_active_set: {journal.after.active_set or 'unmanaged'}")
    destinations = sorted(set(journal.before.links) | set(journal.after.links) | {op.destination for op in journal.operations})
    unsafe = False
    for destination in destinations:
        absolute = (root / destination).absolute()
        before = journal.before.links.get(destination, "absent")
        after = journal.after.links.get(destination, "absent")
        try:
            validate_managed_parents(root, (destination,))
            path = root / destination
            try:
                info = path.lstat()
            except FileNotFoundError:
                current = "absent"
            else:
                if os.path.islink(path):
                    current = os.readlink(path)
                else:
                    current = "non-symlink object"
        except UnsafePathError:
            current = "unsafe parent"
            unsafe = True
        print(f"pending_before\t{absolute}\t{before}")
        print(f"pending_after\t{absolute}\t{after}")
        print(f"pending_current\t{absolute}\t{current}")
    return 3 if unsafe or journal is not None else 0


def _status(args: argparse.Namespace, root: Path, snapshot: Snapshot) -> int:
    if snapshot.journal is not None:
        return _status_pending(root, snapshot)

    config_path = _configuration_path(args.config)
    if args.set_name is not None:
        config = load_config(config_path)
        try:
            skill_set = config.sets[args.set_name]
        except KeyError:
            raise ConfigError(f"unknown set: {args.set_name}") from None
        desired = _destination_map(skill_set)
        validate_managed_parents(root, tuple(set(snapshot.state.links) | set(desired)))
        _check_sources_are_outside_managed_trees(root, skill_set)
        unavailable, _messages = _unavailable_sources(skill_set)
        plan = plan_set(
            root,
            snapshot.state,
            desired,
            active_set=skill_set.name,
            unavailable_sources=unavailable,
        )
        _warn_global_collisions(skill_set)
        _print_plan(root, skill_set.name, plan)
        return 3 if _has_label(plan, "conflict") or _has_label(plan, "broken") else 0

    if snapshot.state.active_set is None:
        _print_header(None)
        return 0

    if not _path_exists_including_symlink(config_path):
        print(f"warning: configuration file is missing; inspecting recorded links only: {config_path}", file=sys.stderr)
        validate_managed_parents(root, tuple(snapshot.state.links))
        rows = inspect_recorded(root, snapshot.state, check_source=lambda source: source_problem(source.name, source))
        _print_header(snapshot.state.active_set)
        _print_rows(root, rows)
        return 3 if any(row.label in ("conflict", "broken") for row in rows) else 0

    config = load_config(config_path)
    skill_set = config.sets.get(snapshot.state.active_set)
    if skill_set is None:
        print(
            f"warning: active set {snapshot.state.active_set!r} is absent from configuration; inspecting recorded links only",
            file=sys.stderr,
        )
        validate_managed_parents(root, tuple(snapshot.state.links))
        rows = inspect_recorded(root, snapshot.state, check_source=lambda source: source_problem(source.name, source))
        _print_header(snapshot.state.active_set)
        _print_rows(root, rows)
        return 3 if any(row.label in ("conflict", "broken") for row in rows) else 0

    desired = _destination_map(skill_set)
    validate_managed_parents(root, tuple(set(snapshot.state.links) | set(desired)))
    _check_sources_are_outside_managed_trees(root, skill_set)
    unavailable, _messages = _unavailable_sources(skill_set)
    plan = plan_set(
        root,
        snapshot.state,
        desired,
        active_set=skill_set.name,
        unavailable_sources=unavailable,
    )
    _warn_global_collisions(skill_set)
    _print_plan(root, snapshot.state.active_set, plan)
    return 3 if _has_label(plan, "conflict") or _has_label(plan, "broken") else 0


def _apply(args: argparse.Namespace, root: Path, snapshot: Snapshot) -> int:
    config = load_config(_configuration_path(args.config))
    try:
        skill_set = config.sets[args.name]
    except KeyError:
        raise ConfigError(f"unknown set: {args.name}") from None
    desired = _destination_map(skill_set)
    validate_managed_parents(root, tuple(set(snapshot.state.links) | set(desired)))
    _check_sources_are_outside_managed_trees(root, skill_set)
    unavailable, source_messages = _unavailable_sources(skill_set)
    plan = plan_set(
        root,
        snapshot.state,
        desired,
        active_set=skill_set.name,
        unavailable_sources=unavailable,
    )
    _warn_global_collisions(skill_set)
    if _has_label(plan, "conflict"):
        _print_plan(root, skill_set.name, plan)
        return 3
    if _has_label(plan, "broken"):
        _print_plan(root, skill_set.name, plan)
        for message in source_messages:
            print(f"error: selected source is unavailable: {message}", file=sys.stderr)
        return 2
    if args.dry_run:
        _print_plan(root, skill_set.name, plan)
        return 0
    if not plan.operations and plan.after == snapshot.state:
        _print_plan(root, skill_set.name, plan)
        return 0
    _print_plan(root, skill_set.name, plan, project=True)
    execute_plan(root, snapshot.state, plan)
    return 0


def _remove(args: argparse.Namespace, root: Path, snapshot: Snapshot) -> int:
    validate_managed_parents(root, tuple(snapshot.state.links))
    plan = plan_remove(root, snapshot.state)
    if _has_label(plan, "conflict"):
        _print_plan(root, None, plan)
        return 3
    if args.dry_run:
        _print_plan(root, None, plan)
        return 0
    if not plan.operations and plan.after == snapshot.state:
        _print_plan(root, None, plan)
        return 0
    _print_plan(root, None, plan, project=True)
    execute_plan(root, snapshot.state, plan)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "roots":
        return _roots_command(args)
    if args.command == "sync":
        from .auto import sync_projects

        return sync_projects(
            _configuration_path(args.config),
            Path(args.project_root).expanduser() if args.project_root is not None else None,
            dry_run=args.dry_run,
        )
    if args.command == "watch":
        from .watch import run_watch

        return run_watch(_configuration_path(args.config))
    if args.command == "service":
        from .service import manage_service

        return manage_service(args.action, _configuration_path(args.config))

    root = _project_root(args.project_root)
    snapshot = load_snapshot(root)
    if snapshot.journal is not None:
        if args.command == "status":
            return _status_pending(root, snapshot)
        raise PendingJournalError("pending journal requires manual recovery")
    if args.command == "status":
        return _status(args, root, snapshot)
    if args.command == "apply":
        return _apply(args, root, snapshot)
    return _remove(args, root, snapshot)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its public exit status."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        return _dispatch(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (CorruptStoreError,) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (UnsafePathError, PendingJournalError, LockBusyError, ConcurrentChangeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except SelectedSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: system I/O failure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: unexpected system failure: {exc}", file=sys.stderr)
        return 1
