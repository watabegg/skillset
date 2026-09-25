"""Install and control the current user's Linux systemd watcher service."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

from .config import ConfigError, load_config


UNIT_NAME = "skillset-watch.service"
MARKER = "# Managed by skillset service install\n"


def unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "systemd" / "user" / UNIT_NAME


def _quote(value: str, *, command: bool = False) -> str:
    """Quote a systemd argument, not a shell command."""
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    value = value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    value = value.replace("%", "%%")
    if command:
        value = value.replace("$", "$$")
    return '"' + value + '"'


def render_unit(config_path: Path) -> str:
    launcher = Path(__file__).resolve().parents[1] / "bin" / "skillset"
    arguments = [str(Path(sys.executable).resolve()), str(launcher), "watch", "--config", str(config_path)]
    return (
        MARKER
        + "# Configuration: " + json.dumps(str(config_path)) + "\n"
        + "[Unit]\nDescription=Automatic project skill sets (filesystem events)\n\n"
        + "[Service]\nType=simple\n"
        + "ExecStart=" + " ".join(_quote(arg, command=True) for arg in arguments) + "\n"
        + "Environment=PYTHONUNBUFFERED=1\n"
        + "Restart=on-failure\nRestartSec=5\nRestartPreventExitStatus=2 3\n"
        + "TimeoutStopSec=10\nUMask=0077\n\n"
        + "[Install]\nWantedBy=default.target\n"
    )


def _existing_unit(path: Path) -> str | None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise FileExistsError(f"refusing non-regular service unit: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.startswith(MARKER):
        raise FileExistsError(f"refusing a service unit not managed by skillset: {path}")
    return content


def _write_unit(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".skillset-unit-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _systemctl(*arguments: str) -> int:
    result = subprocess.run(["systemctl", "--user", *arguments], check=False)
    return result.returncode


def manage_service(action: str, config_path: Path) -> int:
    """Return CLI status; no root-level service or privilege escalation is used."""
    if sys.platform != "linux":
        print("error: automatic service management currently requires Linux/systemd", file=sys.stderr)
        return 2
    if action not in {"install", "start", "stop", "status", "uninstall"}:
        print(f"error: unknown service action: {action}", file=sys.stderr)
        return 2
    try:
        path = unit_path()
        previous = _existing_unit(path)
        if action == "status":
            return _systemctl("status", "--no-pager", UNIT_NAME)
        if action == "install":
            config_path = config_path.expanduser().resolve()
            load_config(config_path)
            content = render_unit(config_path)
            if content != previous:
                _write_unit(path, content)
            if _systemctl("daemon-reload") != 0:
                return 1
            if _systemctl("enable", "--now", UNIT_NAME) != 0:
                return 1
            # enable --now does not restart an already running unit after an
            # ExecStart/config path update.
            if previous is not None and previous != content:
                if _systemctl("restart", UNIT_NAME) != 0:
                    return 1
            print(f"service: installed and enabled {path}")
            return 0
        if action == "uninstall":
            if previous is None:
                print("service: not installed")
                return 0
            if _systemctl("disable", "--now", UNIT_NAME) != 0:
                return 1
            path.unlink()
            if _systemctl("daemon-reload") != 0:
                return 1
            print("service: uninstalled; project links were retained")
            return 0
        if previous is None:
            print("error: service is not installed; run skillset service install", file=sys.stderr)
            return 2
        return 0 if _systemctl(action, UNIT_NAME) == 0 else 1
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError) as exc:
        print(f"error: cannot manage user service: {exc}", file=sys.stderr)
        return 1
