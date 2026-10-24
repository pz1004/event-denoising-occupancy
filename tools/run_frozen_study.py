#!/usr/bin/env python3
"""Run a frozen study with its unchanged protocols stored in protocols/.

Examples (run from the release root):
  python tools/run_frozen_study.py tstar --stage all --output reruns/phase79.json
  python tools/run_frozen_study.py dsec --stage download
  python tools/run_frozen_study.py rotation --stage preflight

Arguments following the study name are passed to its original runner. Use
<study> --help for that runner's options. This launcher redirects protocol paths
in memory; scientific source files, integrity checks and settings are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "tstar": "native3d_ssm.phase79_tstar_cost_quality",
    "dsec": "native3d_ssm.phase80_frozen_dsec_downstream",
    "rotation": "native3d_ssm.phase81_frozen_rotation_application",
}
PROTOCOL_BINDINGS = {
    "tstar": {"PROTOCOL_PATH": "protocol_tstar_fullstream_20261022.md"},
    "dsec": {"PROTOCOL": "protocol_frozen_dsec_downstream_20261022.md"},
    "rotation": {
        "PROTOCOL": "protocol_frozen_rotation_application_20261023.md",
        "TIMESTAMP_ADAPTER": "protocol_rotation_timestamp_adapter_20261023.md",
    },
}
PROTOCOL_SHA256 = {
    "protocol_tstar_fullstream_20261022.md": "b5c7735a3a6d09709ec766df871702790bebadc55ba7e48e5064dd20221a5917",
    "protocol_frozen_dsec_downstream_20261022.md": "9ecc9eb38a546f6a0490f3758a2602419422725238495c0eb81005fdf6624dd3",
    "protocol_frozen_rotation_application_20261023.md": "fe0ff7cdff47d07404aa8b13e083fe79d5205f19e896629c98ef56c89a2ecb64",
    "protocol_rotation_timestamp_adapter_20261023.md": "839652013d86c09b3bf1b6d519c023f0f1406ab41c6f5721adaa49108d2e7e8a",
}


def protocol_paths(study: str) -> dict[str, Path]:
    """Resolve and verify every protocol required by the selected study."""
    paths = {}
    for variable, name in PROTOCOL_BINDINGS[study].items():
        path = ROOT / "protocols" / name
        if not path.is_file():
            raise ValueError(f"Required frozen protocol is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != PROTOCOL_SHA256[name]:
            raise ValueError(f"Frozen protocol SHA-256 mismatch: {path}")
        paths[variable] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=tuple(STUDIES))
    parser.add_argument("runner_args", nargs=argparse.REMAINDER, metavar="ARG",
                        help="arguments forwarded unchanged to the selected runner")
    args = parser.parse_args(argv)
    try:
        paths = protocol_paths(args.study)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    saved_path = sys.path[:]
    sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module(STUDIES[args.study])
        expected_source = ROOT.joinpath(*STUDIES[args.study].split(".")).with_suffix(".py")
        if Path(module.__file__).resolve() != expected_source.resolve():
            raise RuntimeError("The selected study was imported from a different checkout")
        previous_bindings = {name: getattr(module, name) for name in paths}
        previous_argv = sys.argv
        missing = object()
        previous_main = sys.modules.get("__main__", missing)
        try:
            for name, path in paths.items():
                setattr(module, name, path)
            forwarded = args.runner_args
            if forwarded[:1] == ["--"]:
                forwarded = forwarded[1:]
            sys.argv = [module.__file__, *forwarded]
            # Rotation preflight records both canonical and -m dispatcher aliases.
            # Pointing the entry-point alias to the same module preserves that
            # contract without changing its functions or source-file hashes.
            sys.modules["__main__"] = module
            result = module.main()
            return 0 if result is None else result
        finally:
            for name, previous in previous_bindings.items():
                setattr(module, name, previous)
            sys.argv = previous_argv
            if previous_main is missing:
                sys.modules.pop("__main__", None)
            else:
                sys.modules["__main__"] = previous_main
    finally:
        sys.path[:] = saved_path


if __name__ == "__main__":
    raise SystemExit(main())
