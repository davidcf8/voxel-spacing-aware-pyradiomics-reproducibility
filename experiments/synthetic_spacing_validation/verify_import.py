from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    expected = Path(args.expected_source).resolve()

    import radiomics  # noqa: PLC0415
    from radiomics import _cmatrices, _cshape  # noqa: PLC0415

    radiomics_path = Path(radiomics.__file__).resolve()
    cmatrices_path = Path(_cmatrices.__file__).resolve()
    cshape_path = Path(_cshape.__file__).resolve()

    info = {
        "variant": args.variant,
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "expected_source": str(expected),
        "radiomics_file": str(radiomics_path),
        "cmatrices_file": str(cmatrices_path),
        "cshape_file": str(cshape_path),
        "ok": False,
    }

    for label, path in [
        ("radiomics", radiomics_path),
        ("_cmatrices", cmatrices_path),
        ("_cshape", cshape_path),
    ]:
        if expected not in path.parents and path != expected:
            info["error"] = f"{label} imported from unexpected path: {path}"
            print(json.dumps(info, indent=2))
            if args.output_json:
                Path(args.output_json).write_text(json.dumps(info, indent=2))
            return 2

    info["ok"] = True
    print(json.dumps(info, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
