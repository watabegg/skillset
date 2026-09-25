"""Small Linux inotify binding used by the foreground watcher.

This module deliberately exposes the kernel event stream without adding a
polling or platform fallback.  Directory recursion and event policy belong in
``watch.py``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import os
import struct
import sys
from pathlib import Path


# Values from <sys/inotify.h>.
IN_ACCESS = 0x00000001
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_UNMOUNT = 0x00002000
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
IN_ONLYDIR = 0x01000000
IN_DONT_FOLLOW = 0x02000000

INOTIFY_EVENT_HEADER = struct.Struct("iIII")


class InotifyUnavailableError(RuntimeError):
    """The current platform does not provide Linux inotify."""


class InotifyEventParseError(ValueError):
    """The kernel event buffer did not contain complete event records."""


@dataclass(frozen=True)
class Event:
    wd: int
    mask: int
    cookie: int
    name: str


def parse_events(data: bytes) -> list[Event]:
    """Decode complete ``struct inotify_event`` records from one read buffer."""
    events: list[Event] = []
    offset = 0
    size = len(data)
    while offset < size:
        if size - offset < INOTIFY_EVENT_HEADER.size:
            raise InotifyEventParseError("truncated inotify event header")
        wd, mask, cookie, name_length = INOTIFY_EVENT_HEADER.unpack_from(data, offset)
        offset += INOTIFY_EVENT_HEADER.size
        end = offset + name_length
        if end > size:
            raise InotifyEventParseError("truncated inotify event name")
        raw_name = data[offset:end].split(b"\0", 1)[0]
        try:
            name = os.fsdecode(raw_name)
        except (UnicodeError, ValueError) as exc:
            raise InotifyEventParseError(f"invalid inotify event name: {exc}") from None
        events.append(Event(wd=wd, mask=mask, cookie=cookie, name=name))
        offset = end
    return events


def is_available() -> bool:
    """Return whether libc exports the Linux inotify initialization call."""
    return sys.platform.startswith("linux") and hasattr(ctypes.CDLL(None), "inotify_init1")


class Inotify:
    """Nonblocking, close-on-exec inotify instance with typed libc signatures."""

    def __init__(self) -> None:
        if not sys.platform.startswith("linux"):
            raise InotifyUnavailableError("native skillset watch is supported on Linux only")
        self._libc = ctypes.CDLL(None, use_errno=True)
        try:
            init1 = self._libc.inotify_init1
            add_watch = self._libc.inotify_add_watch
            remove_watch = self._libc.inotify_rm_watch
        except AttributeError:
            raise InotifyUnavailableError("libc does not provide Linux inotify") from None

        init1.argtypes = (ctypes.c_int,)
        init1.restype = ctypes.c_int
        add_watch.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
        add_watch.restype = ctypes.c_int
        remove_watch.argtypes = (ctypes.c_int, ctypes.c_int)
        remove_watch.restype = ctypes.c_int

        fd = init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            self._raise_errno("inotify_init1")
        self.fd = fd
        self._closed = False

    @staticmethod
    def _raise_errno(operation: str, path: Path | None = None) -> None:
        error = ctypes.get_errno()
        if operation == "inotify_init1" and error in (errno.EMFILE, errno.ENFILE):
            raise OSError(
                error,
                "inotify_init1 could not allocate an instance; check "
                "/proc/sys/fs/inotify/max_user_instances and the process/system file-descriptor limits",
            )
        if path is None:
            raise OSError(error, f"{operation}: {os.strerror(error)}")
        raise OSError(error, f"{operation} {path}: {os.strerror(error)}", os.fspath(path))

    def add_watch(self, path: Path, mask: int) -> int:
        if self._closed:
            raise OSError("inotify instance is closed")
        wd = self._libc.inotify_add_watch(self.fd, os.fsencode(path), mask)
        if wd < 0:
            self._raise_errno("inotify_add_watch", path)
        return wd

    def remove_watch(self, wd: int) -> None:
        if self._closed:
            return
        if self._libc.inotify_rm_watch(self.fd, wd) < 0:
            error = ctypes.get_errno()
            # The kernel already removes watches for unlinked directories.
            if error not in (getattr(os, "EINVAL", 22),):
                raise OSError(error, f"inotify_rm_watch({wd}): {os.strerror(error)}")

    def read_events(self, buffer_size: int = 256 * 1024) -> list[Event]:
        """Read all currently queued records; never wait for another event."""
        if self._closed:
            return []
        events: list[Event] = []
        while True:
            try:
                chunk = os.read(self.fd, buffer_size)
            except InterruptedError:
                continue
            except BlockingIOError:
                break
            if not chunk:
                break
            events.extend(parse_events(chunk))
        return events

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self.fd)

    def __enter__(self) -> Inotify:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
