"""Run manifest: every experiment writes one JSON record of exactly what
produced its numbers — versions, hardware, settings, seeds. A result without
its manifest is unciteable. All probes degrade gracefully off the Spark
(no GPU, no sglang installed) so the module works on any dev machine."""

import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from importlib import metadata


def _sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except OSError:
        return None


def _pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _src_commit(name):
    """Source installs have no pip metadata; the git commit of the checkout is
    the real version pin. Returns None for wheel installs or missing packages."""
    try:
        from importlib.util import find_spec
        spec = find_spec(name)
        if spec is None or not spec.origin:
            return None
        return _sh(f"git -C {pathlib.Path(spec.origin).parent} rev-parse --short HEAD")
    except (ImportError, ValueError):
        return None


def collect_manifest(extra=None):
    m = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
        # library versions via package metadata — cheap, no CUDA import at collect time
        "versions": {name: _pkg_version(name) for name in ("sglang", "torch", "transformers", "requests")},
        "sglang_src_commit": _src_commit("sglang"),
        "gpu": _sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        "clocks_temp": _sh("nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader"),
    }
    freeze = _sh(f"{sys.executable} -m pip freeze")
    # fingerprint of the whole environment: two runs with equal hashes ran identical packages
    m["pip_freeze_sha"] = hashlib.sha256(freeze.encode()).hexdigest()[:12] if freeze else None
    if extra:
        m.update(extra)
    return m


def write_manifest(manifest, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
