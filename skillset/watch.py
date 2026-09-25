"""Event-driven automatic skill-set application for registered folders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Callable

from . import auto
from .config import ConfigError, load_config
from .inotify import (
    IN_ATTRIB,
    IN_CLOSE_WRITE,
    IN_CREATE,
    IN_DELETE,
    IN_DELETE_SELF,
    IN_DONT_FOLLOW,
    IN_IGNORED,
    IN_ISDIR,
    IN_MOVED_FROM,
    IN_MOVED_TO,
    IN_MOVE_SELF,
    IN_ONLYDIR,
    IN_Q_OVERFLOW,
    IN_UNMOUNT,
    Inotify,
    InotifyEventParseError,
    is_available,
)


_WATCH_MASK = (
    IN_ATTRIB
    | IN_CLOSE_WRITE
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_MOVE_SELF
    | IN_UNMOUNT
    | IN_IGNORED
    | IN_Q_OVERFLOW
)
_STRUCTURAL_MASK = IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO
_DEBOUNCE_SECONDS = 0.2
_RETRY_DELAYS = (0.2, 0.5, 1.0, 2.0, 5.0)
_ROLE_ANCHOR = "anchor"
_ROLE_BOUNDARY = "boundary"
_ROLE_REPOSITORY = "repository"
_ROLE_GIT_METADATA = "git-metadata"
_ROLE_GIT_METADATA_ANCESTOR = "git-metadata-ancestor"


class WatcherError(RuntimeError):
    """Watcher setup or event processing could not safely continue."""


@dataclass(frozen=True)
class _Retry:
    due: float
    attempts: int = 0


class _WatchDaemon:
    """One foreground daemon. Methods are injectable enough for focused tests."""

    def __init__(
        self,
        config_path: Path,
        *,
        backend_factory: Callable[[], Inotify] = Inotify,
        selector_factory: Callable[[], selectors.BaseSelector] = selectors.DefaultSelector,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config_path = config_path.expanduser().resolve(strict=False)
        self.backend_factory = backend_factory
        self.selector_factory = selector_factory
        self.monotonic = monotonic
        self.config = None
        self.backend = None
        self.selector = None
        self.wd_paths: dict[int, Path] = {}
        self.path_wds: dict[Path, int] = {}
        self.roles: dict[Path, set[str]] = defaultdict(set)
        self.git_metadata_owners: dict[Path, set[Path]] = defaultdict(set)
        self.git_metadata_ancestor_owners: dict[Path, set[Path]] = defaultdict(set)
        self.target_paths: list[Path] = []
        self.enabled_roots: set[Path] = set()
        self.git_candidates: set[Path] = set()
        self.retry_queue: dict[Path, _Retry] = {}
        self.subtree_scans: dict[Path, float] = {}
        self.reload_due: float | None = None
        self.expected_ignored_wds: set[int] = set()
        self.stop_requested = False
        self._lock_file: int | None = None
        self._signal_read: int | None = None
        self._signal_write: int | None = None
        self._old_wakeup_fd: int | None = None
        self._old_signal_handlers: dict[int, object] = {}

    def run(self) -> int:
        if not sys.platform.startswith("linux") or not is_available():
            print("error: native skillset watch is supported on Linux only", file=sys.stderr)
            return 2
        try:
            self._acquire_lock()
        except BlockingIOError:
            print(f"error: another watcher already owns {self.config_path}", file=sys.stderr)
            return 3
        except OSError as exc:
            print(f"error: cannot create watcher lock: {exc}", file=sys.stderr)
            self._release_lock()
            return 1

        try:
            self.backend = self.backend_factory()
            self.selector = self.selector_factory()
            self.selector.register(self.backend.fd, selectors.EVENT_READ, "inotify")
            self._install_signal_wakeup()

            # Establish config watches before reading or scanning any projects.
            self._rebuild_watch_graph(None)
            self._load_config_and_reconcile()
            print("watch: ready", flush=True)

            while not self.stop_requested:
                ready = self.selector.select(self._next_timeout())
                if self._signal_read is not None:
                    for key, _mask in ready:
                        if key.data == "signal":
                            self._drain_signal_pipe()
                            break
                if self.stop_requested:
                    break

                events = []
                if any(key.data == "inotify" for key, _mask in ready):
                    try:
                        events = self.backend.read_events()
                    except (OSError, InotifyEventParseError) as exc:
                        raise WatcherError(f"cannot read inotify events: {exc}") from None
                self._process_events(events)
                self._process_due_work()
            return 0
        except KeyboardInterrupt:
            return 0
        except WatcherError as exc:
            print(f"error: watch stopped: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"error: watch stopped: {exc}", file=sys.stderr)
            return 1
        finally:
            self._restore_signal_wakeup()
            if self.selector is not None:
                try:
                    self.selector.close()
                except OSError:
                    pass
            if self.backend is not None:
                self.backend.close()
            self._release_lock()

    def _lock_path(self) -> Path:
        state_home = os.environ.get("XDG_STATE_HOME")
        state_root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
        config_hash = hashlib.sha256(os.fsencode(self.config_path)).hexdigest()
        return state_root / "skillset" / f"watch-{config_hash}.lock"

    def _acquire_lock(self) -> None:
        path = self._lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
        self._lock_file = descriptor

    def _release_lock(self) -> None:
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_file)
                self._lock_file = None

    def _install_signal_wakeup(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._signal_read, self._signal_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self._old_wakeup_fd = signal.set_wakeup_fd(self._signal_write)
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        self.selector.register(self._signal_read, selectors.EVENT_READ, "signal")

    def _handle_signal(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def _drain_signal_pipe(self) -> None:
        if self._signal_read is None:
            return
        while True:
            try:
                if not os.read(self._signal_read, 4096):
                    break
            except BlockingIOError:
                break
            except InterruptedError:
                continue

    def _restore_signal_wakeup(self) -> None:
        if self._old_wakeup_fd is not None:
            try:
                signal.set_wakeup_fd(self._old_wakeup_fd)
            except (ValueError, OSError):
                pass
            self._old_wakeup_fd = None
        for signum, handler in self._old_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
        self._old_signal_handlers.clear()
        if self.selector is not None and self._signal_read is not None:
            try:
                self.selector.unregister(self._signal_read)
            except (KeyError, OSError, ValueError):
                pass
        for descriptor in (self._signal_read, self._signal_write):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self._signal_read = None
        self._signal_write = None

    def _next_timeout(self) -> float | None:
        deadlines: list[float] = []
        if self.reload_due is not None:
            deadlines.append(self.reload_due)
        deadlines.extend(retry.due for retry in self.retry_queue.values())
        deadlines.extend(self.subtree_scans.values())
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - self.monotonic())

    @staticmethod
    def _canonical_path(path: Path) -> Path:
        return Path(os.path.abspath(os.fspath(path)))

    @staticmethod
    def _directory_info(path: Path) -> os.stat_result | None:
        try:
            info = path.lstat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            return None
        return info

    def _add_directory(self, path: Path, role: str) -> bool:
        """Subscribe before the caller enumerates any children."""
        path = self._canonical_path(path)
        if self._directory_info(path) is None:
            return False
        mask = _WATCH_MASK | IN_ONLYDIR | IN_DONT_FOLLOW
        try:
            wd = self.backend.add_watch(path, mask)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
                return False
            if exc.errno in (errno.ENOSPC, errno.ENOMEM):
                raise WatcherError(
                    f"inotify watch limit reached at {path}; raise fs.inotify.max_user_watches or register fewer folders"
                ) from None
            raise WatcherError(f"cannot watch directory {path}: {exc}") from None

        prior_path = self.wd_paths.get(wd)
        if prior_path is not None and prior_path != path:
            # Multiple path aliases to one inode cannot be represented safely
            # by inotify's single wd-to-path mapping. Keep the first alias.
            self.roles[prior_path].add(role)
            return True
        self.wd_paths[wd] = path
        self.path_wds[path] = wd
        self.roles[path].add(role)
        return True

    @staticmethod
    def _path_chain(directory: Path) -> list[Path]:
        chain: list[Path] = []
        current = directory
        while True:
            chain.append(current)
            if current.parent == current:
                break
            current = current.parent
        chain.reverse()
        return chain

    def _build_targets(self, config: object | None) -> None:
        targets = [self.config_path]
        enabled: set[Path] = set()
        if config is not None:
            for root, set_name in getattr(config, "roots", {}).items():
                if set_name is not None:
                    enabled.add(self._canonical_path(Path(root)))
            targets.extend(sorted(enabled, key=os.fspath))
        self.target_paths = targets
        self.enabled_roots = enabled

    def _watch_anchor_chains(self) -> None:
        directories: set[Path] = set()
        for target in self.target_paths:
            watch_dir = target.parent if target == self.config_path else target
            directories.update(self._path_chain(watch_dir))
        for directory in sorted(directories, key=lambda item: (len(item.parts), os.fspath(item))):
            self._add_directory(directory, _ROLE_ANCHOR)

    def _has_enabled_descendant(self, path: Path) -> bool:
        return any(root != path and root.is_relative_to(path) for root in self.enabled_roots)

    def _select_set(self, path: Path) -> str | None:
        return auto.select_set(self.config, path)

    @staticmethod
    def _git_marker(path: Path) -> Path | None:
        marker = path / ".git"
        try:
            info = marker.lstat()
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return None
        if stat.S_ISLNK(info.st_mode):
            return None
        if stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode):
            return marker
        return None

    @staticmethod
    def _is_worktree_root(path: Path, marker: Path) -> bool:
        try:
            completed = subprocess.run(
                ["git", "-C", os.fspath(path), "rev-parse", "--show-toplevel"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        try:
            reported = Path(completed.stdout.strip()).resolve(strict=False)
            candidate = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return False
        return bool(marker) and reported == candidate

    def _watch_metadata_dir(self, directory: Path, owner: Path) -> None:
        directory = self._canonical_path(directory)
        if self._directory_info(directory) is not None:
            if self._add_directory(directory, _ROLE_GIT_METADATA):
                self.git_metadata_owners[directory].add(owner)
            return
        current = directory.parent
        while current != current.parent:
            if self._directory_info(current) is not None:
                if self._add_directory(current, _ROLE_GIT_METADATA_ANCESTOR):
                    self.git_metadata_ancestor_owners[current].add(owner)
                return
            current = current.parent

    def _watch_git_metadata(self, root: Path, marker: Path) -> None:
        try:
            marker_info = marker.lstat()
        except OSError:
            return
        if stat.S_ISDIR(marker_info.st_mode) and not stat.S_ISLNK(marker_info.st_mode):
            self._watch_metadata_dir(marker, root)
            return
        if not stat.S_ISREG(marker_info.st_mode) or stat.S_ISLNK(marker_info.st_mode):
            return
        try:
            line = marker.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            return
        prefix = "gitdir: "
        if not line.startswith(prefix):
            return
        target = Path(line[len(prefix) :])
        git_dir = target if target.is_absolute() else root / target
        git_dir = git_dir.resolve(strict=False)
        self._watch_metadata_dir(git_dir, root)

        common_file = git_dir / "commondir"
        try:
            common_line = common_file.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            return
        common = Path(common_line)
        common_dir = common if common.is_absolute() else git_dir / common
        self._watch_metadata_dir(common_dir.resolve(strict=False), root)

    def _walk_directory(self, path: Path, *, route_only: bool = False) -> None:
        """Watch a selected tree recursively, pruning repos and ignored trees."""
        path = self._canonical_path(path)
        if self.config is None or self._directory_info(path) is None:
            return
        if path not in self.enabled_roots:
            try:
                selected = self._select_set(path)
            except (OSError, RuntimeError, ValueError):
                selected = None
            if selected is None and not self._has_enabled_descendant(path):
                return
        else:
            route_only = False

        # Subscribe before inspecting .git and enumerating children. A .git
        # marker can appear while rev-parse or directory enumeration is running.
        self._add_directory(path, _ROLE_BOUNDARY)
        marker = self._git_marker(path)
        valid_repo = marker is not None and self._is_worktree_root(path, marker)
        if valid_repo:
            self.roles[path].discard(_ROLE_BOUNDARY)
            self.roles[path].add(_ROLE_REPOSITORY)
        if marker is not None:
            self._watch_git_metadata(path, marker)
            if not valid_repo:
                self.git_candidates.add(path)

        if valid_repo or marker is not None or route_only:
            for name in sorted(self._route_children(path)):
                child = path / name
                if self._directory_info(child) is None:
                    continue
                self._walk_directory(child, route_only=child not in self.enabled_roots)
            return

        try:
            entries = list(os.scandir(path))
        except OSError:
            return
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.name == ".git":
                # Git metadata is watched directly and never traversed.
                continue
            if entry.name in auto.IGNORED_DIRS:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            self._walk_directory(path / entry.name)

    def _route_children(self, path: Path) -> set[str]:
        children: set[str] = set()
        for root in self.enabled_roots:
            if root == path or not root.is_relative_to(path):
                continue
            relative = root.relative_to(path)
            if relative.parts:
                children.add(relative.parts[0])
        return children

    def _forget_subtree(self, root: Path, *, preserve_anchors: bool) -> None:
        """Remove watches and metadata ownership beneath one path."""
        root = self._canonical_path(root)
        # Owners are repo roots. A gitdir may be outside the watched subtree,
        # so remove those ownership edges as well as paths below root.
        removed_owners = {
            owner
            for owner_set in self.git_metadata_owners.values()
            for owner in owner_set
            if owner == root or owner.is_relative_to(root)
        }
        removed_owners.update(
            owner
            for owner_set in self.git_metadata_ancestor_owners.values()
            for owner in owner_set
            if owner == root or owner.is_relative_to(root)
        )
        for table in (self.git_metadata_owners, self.git_metadata_ancestor_owners):
            for path, owner_set in tuple(table.items()):
                owner_set.difference_update(removed_owners)
                if not owner_set:
                    table.pop(path, None)
                    roles = self.roles.get(path)
                    if roles is not None:
                        roles.discard(_ROLE_GIT_METADATA)
                        roles.discard(_ROLE_GIT_METADATA_ANCESTOR)
                        if not roles:
                            self._remove_watch(path)

        for candidate in tuple(self.git_candidates):
            if candidate == root or candidate.is_relative_to(root):
                self.git_candidates.discard(candidate)
        for candidate in tuple(self.retry_queue):
            if candidate == root or candidate.is_relative_to(root):
                self.retry_queue.pop(candidate, None)
        for candidate in tuple(self.subtree_scans):
            if candidate == root or candidate.is_relative_to(root):
                self.subtree_scans.pop(candidate, None)

        for path, roles in tuple(self.roles.items()):
            if path != root and not path.is_relative_to(root):
                continue
            if preserve_anchors:
                roles.difference_update(
                    {
                        _ROLE_BOUNDARY,
                        _ROLE_REPOSITORY,
                        _ROLE_GIT_METADATA,
                        _ROLE_GIT_METADATA_ANCESTOR,
                    }
                )
            else:
                roles.clear()
            if roles:
                continue
            self._remove_watch(path)

    def _remove_watch(self, path: Path) -> None:
        self.roles.pop(path, None)
        wd = self.path_wds.pop(path, None)
        if wd is None:
            return
        self.wd_paths.pop(wd, None)
        self.expected_ignored_wds.add(wd)
        try:
            self.backend.remove_watch(wd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.EBADF):
                raise WatcherError(f"cannot remove inotify watch for {path}: {exc}") from None

    def _refresh_subtree(self, path: Path) -> None:
        """Reattach just one changed subtree, subscribing before enumeration."""
        path = self._canonical_path(path)
        self._forget_subtree(path, preserve_anchors=True)
        if self.config is not None and self._directory_info(path) is not None:
            self._walk_directory(path)
            for candidate in self.git_candidates:
                if candidate == path or candidate.is_relative_to(path):
                    self._schedule_candidate(candidate, restart=False)

    def _drop_subtree(self, path: Path) -> None:
        self._forget_subtree(path, preserve_anchors=False)

    def _refresh_target_anchors(self, event_path: Path) -> None:
        """Restore ancestor watches and only visit registered roots below an event."""
        self._watch_anchor_chains()
        if event_path == self.config_path or self.config_path.is_relative_to(event_path):
            self._schedule_reload()
        for root in sorted(self.enabled_roots, key=os.fspath):
            if root != event_path and not root.is_relative_to(event_path):
                continue
            if self._directory_info(root) is None:
                self._drop_subtree(root)
                continue
            self._refresh_subtree(root)
            self._schedule_subtree_scan(root)

    def _rebuild_watch_graph(self, config: object | None) -> None:
        """Refresh path mappings while retaining unchanged kernel watches."""
        old_wds = set(self.wd_paths)
        self.wd_paths = {}
        self.path_wds = {}
        self.roles = defaultdict(set)
        self.git_metadata_owners = defaultdict(set)
        self.git_metadata_ancestor_owners = defaultdict(set)
        self.git_candidates = set()
        self.config = config
        self._build_targets(config)
        self._watch_anchor_chains()
        if config is not None:
            for root in sorted(self.enabled_roots, key=os.fspath):
                if self._directory_info(root) is not None:
                    self._walk_directory(root)

        stale_wds = old_wds - set(self.wd_paths)
        for wd in stale_wds:
            self.expected_ignored_wds.add(wd)
            try:
                self.backend.remove_watch(wd)
            except OSError as exc:
                if exc.errno not in (errno.EINVAL, errno.EBADF):
                    raise WatcherError(f"cannot remove stale inotify watch {wd}: {exc}") from None

        if config is not None:
            for candidate in self.git_candidates:
                self._schedule_candidate(candidate, restart=False)
        for candidate in tuple(self.retry_queue):
            if candidate not in self.git_candidates and self._git_marker(candidate) is None:
                self.retry_queue.pop(candidate, None)

    def _schedule_candidate(self, path: Path, *, restart: bool = True) -> None:
        path = self._canonical_path(path)
        existing = self.retry_queue.get(path)
        if existing is not None and not restart:
            return
        if restart:
            self.retry_queue[path] = _Retry(self.monotonic() + _RETRY_DELAYS[0], attempts=0)
        elif existing is None:
            self.retry_queue[path] = _Retry(self.monotonic() + _RETRY_DELAYS[0], attempts=0)

    def _schedule_subtree_scan(self, path: Path) -> None:
        self.subtree_scans[self._canonical_path(path)] = self.monotonic() + _DEBOUNCE_SECONDS

    def _schedule_reload(self) -> None:
        self.reload_due = self.monotonic() + _DEBOUNCE_SECONDS

    def _target_related_change(self, event_path: Path) -> bool:
        return any(target == event_path or target.is_relative_to(event_path) for target in self.target_paths)

    def _process_events(self, events: list[object]) -> None:
        overflow = any(getattr(event, "mask", 0) & IN_Q_OVERFLOW for event in events)
        invalidation = any(getattr(event, "mask", 0) & IN_UNMOUNT for event in events)
        full_reconcile = overflow or invalidation
        reload_config = False
        drop_paths: set[Path] = set()
        refresh_paths: set[Path] = set()
        scan_paths: set[Path] = set()
        target_events: set[Path] = set()
        candidate_paths: set[Path] = set()

        if overflow:
            print("watch: inotify queue overflowed; rebuilding watches and reconciling", file=sys.stderr, flush=True)
        elif invalidation:
            print("watch: watched filesystem was unmounted; rebuilding watches and reconciling", file=sys.stderr, flush=True)

        for event in events:
            mask = getattr(event, "mask", 0)
            wd = getattr(event, "wd", -1)
            if mask & IN_IGNORED and wd in self.expected_ignored_wds:
                self.expected_ignored_wds.discard(wd)
                continue
            if mask & (IN_Q_OVERFLOW | IN_UNMOUNT):
                continue
            path = self.wd_paths.get(wd)
            if path is None:
                continue
            name = getattr(event, "name", "")
            event_path = path / name if name else path
            roles = self.roles.get(path, set())

            if mask & IN_IGNORED:
                # Directory deletion and move can remove a watch normally.
                # A still-present path indicates unexpected invalidation.
                if self._directory_info(path) is None:
                    drop_paths.add(path)
                else:
                    print(
                        f"watch: inotify watch was invalidated: {path}; rebuilding and reconciling",
                        file=sys.stderr,
                        flush=True,
                    )
                    full_reconcile = True
                continue
            if name and event_path == self.config_path:
                reload_config = True
                continue
            if _ROLE_ANCHOR in roles and name and self._target_related_change(event_path):
                target_events.add(event_path)

            if _ROLE_GIT_METADATA in roles:
                for owner in self.git_metadata_owners.get(path, ()):
                    candidate_paths.add(owner)
                continue
            if _ROLE_GIT_METADATA_ANCESTOR in roles:
                if name and mask & IN_ISDIR:
                    for owner in self.git_metadata_ancestor_owners.get(path, ()):
                        candidate_paths.add(owner)
                        refresh_paths.add(owner)
                continue

            if not name and mask & (IN_DELETE_SELF | IN_MOVE_SELF):
                drop_paths.add(path)
                continue

            if name == ".git" and roles & {_ROLE_BOUNDARY, _ROLE_REPOSITORY}:
                refresh_paths.add(path)
                candidate_paths.add(path)
                continue

            if _ROLE_REPOSITORY in roles:
                if name in self._route_children(path) and mask & _STRUCTURAL_MASK:
                    if mask & (IN_CREATE | IN_MOVED_TO):
                        refresh_paths.add(event_path)
                        scan_paths.add(event_path)
                    elif mask & (IN_DELETE | IN_MOVED_FROM):
                        drop_paths.add(event_path)
                continue

            if _ROLE_BOUNDARY in roles and name and mask & _STRUCTURAL_MASK:
                if mask & IN_ISDIR:
                    if name in auto.IGNORED_DIRS:
                        continue
                    if mask & (IN_CREATE | IN_MOVED_TO):
                        refresh_paths.add(event_path)
                        scan_paths.add(event_path)
                    elif mask & (IN_DELETE | IN_MOVED_FROM):
                        drop_paths.add(event_path)
                continue

            if _ROLE_ANCHOR in roles and name and self._target_related_change(event_path):
                if mask & (IN_DELETE | IN_MOVED_FROM):
                    drop_paths.add(event_path)
                if mask & (IN_CREATE | IN_MOVED_TO):
                    refresh_paths.add(event_path)

        if reload_config:
            self._schedule_reload()
        elif full_reconcile:
            self._load_config_and_reconcile()
            return
        else:
            for path in sorted(drop_paths, key=lambda item: (-len(item.parts), os.fspath(item))):
                self._drop_subtree(path)
            for path in sorted(refresh_paths, key=lambda item: (len(item.parts), os.fspath(item))):
                self._refresh_subtree(path)
            for path in sorted(target_events, key=os.fspath):
                self._refresh_target_anchors(path)
            for path in scan_paths:
                self._schedule_subtree_scan(path)
            for path in candidate_paths:
                self._schedule_candidate(path)

    def _load_config_and_reconcile(self) -> None:
        try:
            config = load_config(self.config_path)
        except (ConfigError, OSError) as exc:
            print(f"watch: configuration unavailable; waiting for repair: {exc}", file=sys.stderr, flush=True)
            self.retry_queue.clear()
            self.subtree_scans.clear()
            self._rebuild_watch_graph(None)
            return
        self.retry_queue.clear()
        self.subtree_scans.clear()
        self._rebuild_watch_graph(config)
        self._sync_all()

    def _sync_all(self) -> None:
        try:
            status = auto.sync_projects(self.config_path, quiet=True)
        except Exception as exc:
            print(f"watch: reconciliation failed: {exc}", file=sys.stderr, flush=True)
            return
        if status:
            print(f"watch: reconciliation returned status {status}", file=sys.stderr, flush=True)

    def _latest_config(self):
        try:
            return load_config(self.config_path)
        except (ConfigError, OSError) as exc:
            print(f"watch: configuration unavailable; waiting for repair: {exc}", file=sys.stderr, flush=True)
            self.retry_queue.clear()
            self.subtree_scans.clear()
            self._rebuild_watch_graph(None)
            return None

    def _ensure_current_config(self):
        config = self._latest_config()
        if config is None:
            return None
        if config != self.config:
            self.retry_queue.clear()
            self.subtree_scans.clear()
            self._rebuild_watch_graph(config)
            self._sync_all()
            return None
        return config

    def _process_due_work(self) -> None:
        now = self.monotonic()
        if self.reload_due is not None and self.reload_due <= now:
            self.reload_due = None
            self._load_config_and_reconcile()
            return

        due_scans = sorted(path for path, due in self.subtree_scans.items() if due <= now)
        for path in due_scans:
            self.subtree_scans.pop(path, None)
        due_retries = sorted(
            ((path, retry) for path, retry in self.retry_queue.items() if retry.due <= now),
            key=lambda item: os.fspath(item[0]),
        )
        if not due_scans and not due_retries:
            return
        config = self._ensure_current_config()
        if config is None:
            return

        for path in due_scans:
            self._discover_and_sync(config, path)
        for path, retry in due_retries:
            if path not in self.retry_queue:
                continue
            self.retry_queue.pop(path, None)
            if not self._in_scope_directory(config, path):
                continue
            projects = self._discover_and_sync(config, path)
            if projects or self._git_marker(path) is None:
                continue
            if retry.attempts < len(_RETRY_DELAYS) - 1:
                next_attempt = retry.attempts + 1
                self.retry_queue[path] = _Retry(
                    self.monotonic() + _RETRY_DELAYS[next_attempt],
                    attempts=next_attempt,
                )
            else:
                print(
                    f"watch: repository still incomplete; waiting for another event: {path}",
                    file=sys.stderr,
                    flush=True,
                )

    def _in_scope_directory(self, config: object, path: Path) -> bool:
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                return False
            if path.resolve(strict=True) != path:
                return False
            return auto.select_set(config, path) is not None or any(
                root != path and root.is_relative_to(path)
                for root, set_name in getattr(config, "roots", {}).items()
                if set_name is not None
            )
        except (OSError, RuntimeError, ValueError):
            return False

    def _discover_and_sync(self, config: object, start: Path) -> list[Path]:
        if not self._in_scope_directory(config, start):
            return []
        try:
            projects = auto.discover_projects(config, start=start)
        except Exception as exc:
            print(f"watch: cannot discover projects under {start}: {exc}", file=sys.stderr, flush=True)
            return []
        synced: list[Path] = []
        normalized = {Path(project).resolve(strict=False) for project in projects}
        for project in sorted(normalized, key=os.fspath):
            if not self._in_scope_directory(config, project):
                continue
            # A bounded retry can turn an incomplete .git marker into a valid
            # repository after its tree was first classified. Upgrade that
            # subtree before applying and stop watching ordinary source dirs.
            self._refresh_subtree(project)
            try:
                status = auto.sync_projects(self.config_path, project, quiet=True)
            except Exception as exc:
                print(f"watch: sync failed for {project}: {exc}", file=sys.stderr, flush=True)
                continue
            synced.append(project)
            if status:
                print(f"watch: sync returned status {status} for {project}", file=sys.stderr, flush=True)
        return synced


def run_watch(config_path: Path) -> int:
    """Run the native foreground watcher for one canonical config path."""
    return _WatchDaemon(Path(config_path)).run()
