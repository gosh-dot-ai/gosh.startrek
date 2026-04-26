# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import logging
import os
import platform
import sys

log = logging.getLogger(__name__)


def apply_memory_sandbox(data_dir: str) -> None:
    """Apply Landlock sandbox restricting filesystem access.

    Memory process needs:
      - Read-only:  system libs, Python runtime, project source, $HOME
      - Read-write: data_dir, /tmp
    """
    if platform.system() != "Linux":
        log.info("sandbox: unavailable (not Linux), running without isolation")
        return

    try:
        from landlock import FSAccess, Ruleset
    except ImportError:
        log.warning(
            "sandbox: landlock package not installed, running without isolation. "
            "Install with: pip install landlock"
        )
        return

    try:
        ro_paths, rw_paths = _build_allowlists(data_dir)

        # Dry-run: verify data_dir is writable under Landlock.
        if not _probe_landlock_write(data_dir, ro_paths, rw_paths):
            log.warning(
                "sandbox: data_dir %s not writable under Landlock "
                "(likely Docker bind mount on non-Linux host). "
                "Skipping sandbox.",
                data_dir,
            )
            return

        # Real sandbox
        ro_rules = (
            FSAccess.READ_FILE | FSAccess.READ_DIR | FSAccess.EXECUTE
        )

        rs = Ruleset()

        for p in ro_paths:
            if os.path.exists(p):
                rs.allow(p, rules=ro_rules)

        for p in rw_paths:
            if os.path.exists(p):
                rs.allow(p)  # full access (default = all flags)

        rs.apply()
        log.info(
            "sandbox: active (Landlock), %d RO paths, %d RW paths",
            len(ro_paths),
            len(rw_paths),
        )

    except OSError as e:
        log.warning("sandbox: failed to activate: %s", e)
    except Exception as e:
        log.warning("sandbox: unexpected error: %s", e)


def _build_allowlists(data_dir: str) -> tuple[list[str], list[str]]:
    """Build separate read-only and read-write path lists."""
    ro = set()
    rw = set()

    # ── Read-only: system libraries and standard paths ──────────────
    for p in ["/usr", "/etc", "/lib", "/lib64", "/proc", "/dev"]:
        ro.add(p)

    # Python runtime (sys.prefix covers venv or system Python)
    ro.add(sys.prefix)
    if hasattr(sys, "real_prefix"):
        ro.add(sys.real_prefix)
    if sys.base_prefix != sys.prefix:
        ro.add(sys.base_prefix)

    # PYTHONPATH entries (covers src-layout in Docker: /app)
    for entry in sys.path:
        if entry and os.path.isabs(entry) and os.path.exists(entry):
            ro.add(entry)

    # Site-packages
    try:
        import site
        for sp in site.getsitepackages():
            if os.path.exists(sp):
                ro.add(sp)
        user_sp = site.getusersitepackages()
        if isinstance(user_sp, str) and os.path.exists(user_sp):
            ro.add(user_sp)
    except Exception:
        pass

    # Working directory (for running from a checkout)
    cwd = os.getcwd()
    if cwd:
        ro.add(cwd)

    # Home directory (read-only — config/token files are read, not written
    # at runtime; token is saved before sandbox activates)
    import pathlib
    home = str(pathlib.Path.home())
    ro.add(home)

    # ── Read-write: data directory and temp ─────────────────────────
    rw.add(data_dir)
    import tempfile
    rw.add(tempfile.gettempdir())

    # Remove any RW paths from RO set to avoid conflicting rules
    ro -= rw

    return sorted(ro), sorted(rw)


def _probe_landlock_write(
    data_dir: str, ro_paths: list[str], rw_paths: list[str]
) -> bool:
    """Fork a child process that applies Landlock and tests write access.

    Returns True if writes succeed under Landlock, False otherwise.
    Uses fork so the parent process is never sandboxed by the probe.
    """
    os.makedirs(data_dir, exist_ok=True)

    pid = os.fork()
    if pid == 0:
        # Child: apply sandbox and test
        try:
            from landlock import FSAccess, Ruleset

            ro_rules = (
                FSAccess.READ_FILE | FSAccess.READ_DIR | FSAccess.EXECUTE
            )

            rs = Ruleset()
            for p in ro_paths:
                if os.path.exists(p):
                    rs.allow(p, rules=ro_rules)
            for p in rw_paths:
                if os.path.exists(p):
                    rs.allow(p)
            rs.apply()

            probe = os.path.join(data_dir, ".landlock_probe")
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
            os._exit(0)  # success
        except Exception:
            os._exit(1)  # failure
    else:
        # Parent: wait for child
        _, status = os.waitpid(pid, 0)
        return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
