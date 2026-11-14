"""Shared engine resolution: Fennec (native Rust) first, the Colibri launcher as fallback.

Fennec self-launches (`target/release/fennec[.exe] serve ...`); Colibri was a
Python launcher (`python c/coli serve ...`). Both speak the same OpenAI-compatible
HTTP contract on the same default port, so a script only needs to know which argv
prefix to build and which capabilities exist.

This mirrors fennec-core's convention (fennec_core/engine.py::_engine_binary):
probe `target/release`, `target/debug`, `.` for the fennec binary, else fall back
to `c/coli`. Stdlib only - preflight must not import torch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

FENNEC = "fennec"
COLIBRI = "colibri"

# Engine knobs Fennec still reads under the Colibri spelling. FENNEC_* is accepted
# as an alias; COLI_* wins when both are set, so an existing setup keeps working.
ALIASES = ("MODEL", "API_KEY", "VULKAN", "METAL", "CUDA",
           "VK_TIER_HOST_GB", "VK_TIER_GB", "VK_DEV2")
FENNEC_VARS = frozenset("FENNEC_" + name for name in ALIASES)

DEFAULT_PORT = 8184


class EngineError(RuntimeError):
    """No usable engine under the requested root."""


class ColibriOnlyError(EngineError):
    """The requested eval path has no Fennec equivalent; a Colibri checkout is required."""


class Engine(NamedTuple):
    kind: str
    binary: Path
    root: Path

    @property
    def is_fennec(self) -> bool:
        return self.kind == FENNEC


def binary_names() -> tuple:
    return ("fennec.exe", "fennec") if os.name == "nt" else ("fennec",)


def default_engine_root() -> str:
    """FENNEC_ENGINE_ROOT / FENNEC_HOME, else an installed Fennec, else ~/Colibri."""
    for variable in ("FENNEC_ENGINE_ROOT", "FENNEC_HOME"):
        value = os.environ.get(variable)
        if value:
            return value
    for candidate in (Path.home() / "Fennec", Path.home() / "Colibri"):
        if candidate.is_dir():
            return str(candidate)
    return str(Path.home() / "Colibri")


def resolve(root) -> Engine:
    """Fennec wins when both are installed; it is the engine this suite is built around."""
    root = Path(root).expanduser().resolve()
    for rel in ("target/release", "target/debug", "."):
        for name in binary_names():
            candidate = root / rel / name
            if candidate.is_file():
                return Engine(FENNEC, candidate, root)
    launcher = root / "c" / "coli"
    if launcher.is_file():
        return Engine(COLIBRI, launcher, root)
    raise EngineError(
        f"no engine found under {root}: build Fennec (cargo build --release) so "
        f"{root / 'target' / 'release' / binary_names()[0]} exists, or point --engine-root "
        "at a Colibri checkout that still has c/coli")


def command(engine: Engine, args, *, python=None) -> list:
    """Fennec needs no interpreter prefix; the Colibri launcher does."""
    argv = [str(arg) for arg in args]
    if engine.is_fennec:
        return [str(engine.binary), *argv]
    return [str(python or sys.executable), str(engine.binary), *argv]


def serve_command(engine: Engine, model, port, *, host="127.0.0.1", extra=(),
                  api_key=None, python=None) -> list:
    """`serve --auto-tier --model <dir> --host <host> --port <n>` for either engine.

    Fennec reads the key from COLI_API_KEY/FENNEC_API_KEY (see engine_env); the
    Colibri launcher also accepts --api-key, which is kept for compatibility.
    """
    args = ["serve", "--auto-tier", "--model", str(model), "--host", host, "--port", str(port)]
    args += [str(arg) for arg in extra]
    if api_key and not engine.is_fennec:
        args += ["--api-key", str(api_key)]
    return command(engine, args, python=python)


def env_alias(name, env=None):
    """FENNEC_<name> or COLI_<name>, whichever is set (Fennec spelling first)."""
    env = os.environ if env is None else env
    return env.get("FENNEC_" + name) or env.get("COLI_" + name)


def engine_env(overrides=None, *, base=None) -> dict:
    """The environment plus FENNEC_* aliases mirrored onto the COLI_* names both read."""
    env = dict(os.environ if base is None else base)
    for name in ALIASES:
        value = env.get("FENNEC_" + name)
        if value and not env.get("COLI_" + name):
            env["COLI_" + name] = value
    if overrides:
        env.update({key: str(value) for key, value in overrides.items()})
    return env


def strip_engine_vars(env) -> dict:
    """Drop every engine knob, both spellings, so a profile starts from a clean slate."""
    return {key: value for key, value in env.items()
            if not key.startswith("COLI_") and key not in FENNEC_VARS}


def require_colibri(engine: Engine, capability) -> None:
    """Refuse a Colibri-only path loudly: never fake an equivalent, never skip it."""
    if not engine.is_fennec:
        return
    raise ColibriOnlyError(
        f"{engine.binary} is Fennec (native Rust), which has no equivalent for {capability}. "
        "This eval path requires a Colibri checkout: pass --engine-root <colibri-dir> "
        "(or set FENNEC_ENGINE_ROOT) pointing at a checkout that still has c/coli. "
        "Against Fennec use the HTTP-only suites instead: scripts/bench_holdout.py, "
        "scripts/bench_engine_vs_airllm.py or scripts/bench_tier_profiles.py.")
