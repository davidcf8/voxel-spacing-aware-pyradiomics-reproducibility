from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import pandas as pd
import SimpleITK as sitk


TEXTURE_FAMILIES = {"glcm", "glrlm", "gldm", "ngtdm", "glszm"}


def is_number(value: object) -> bool:
    try:
        f = float(value)
    except Exception:
        return False
    return math.isfinite(f)


def parse_feature_name(name: str) -> tuple[str | None, str | None]:
    parts = name.split("_")
    if len(parts) >= 3 and parts[0].lower() == "original" and parts[1].lower() in TEXTURE_FAMILIES:
        return parts[1].lower(), "_".join(parts[2:])
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--weighting-norm")
    parser.add_argument("--bin-width", type=float, default=25.0)
    args = parser.parse_args()

    expected = Path(args.expected_source).resolve()

    import radiomics  # noqa: PLC0415
    from radiomics import _cmatrices, _cshape, featureextractor  # noqa: PLC0415

    radiomics_path = Path(radiomics.__file__).resolve()
    cmatrices_path = Path(_cmatrices.__file__).resolve()
    cshape_path = Path(_cshape.__file__).resolve()
    for label, path in [
        ("radiomics", radiomics_path),
        ("_cmatrices", cmatrices_path),
        ("_cshape", cshape_path),
    ]:
        if expected not in path.parents and path != expected:
            raise RuntimeError(f"{label} imported from unexpected path: {path}; expected under {expected}")

    image = sitk.ReadImage(args.image)
    mask = sitk.ReadImage(args.mask)
    spacing = tuple(float(x) for x in image.GetSpacing())
    mask_spacing = tuple(float(x) for x in mask.GetSpacing())
    print(f"[{args.variant}] image spacing: {spacing}", flush=True)
    print(f"[{args.variant}] mask spacing:  {mask_spacing}", flush=True)
    print(f"[{args.variant}] radiomics: {radiomics_path}", flush=True)
    print(f"[{args.variant}] _cmatrices: {cmatrices_path}", flush=True)
    print(f"[{args.variant}] _cshape: {cshape_path}", flush=True)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=args.bin_width,
        label=1,
        force2D=False,
        additionalInfo=False,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for family in sorted(TEXTURE_FAMILIES):
        extractor.enableFeatureClassByName(family)
    if args.weighting_norm:
        extractor.settings["weightingNorm"] = args.weighting_norm

    result = extractor.execute(str(args.image), str(args.mask))
    rows: list[dict[str, object]] = []
    for key, value in result.items():
        family, short_name = parse_feature_name(str(key))
        if family is None or not is_number(value):
            continue
        rows.append(
            {
                "feature": str(key),
                "family": family,
                "short_feature": short_name,
                "value": float(value),
            }
        )

    if not rows:
        raise RuntimeError("No original texture features were extracted.")

    out = pd.DataFrame(rows).sort_values(["family", "feature"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    metadata = {
        "variant": args.variant,
        "weightingNorm": args.weighting_norm or "default",
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "expected_source": str(expected),
        "radiomics_file": str(radiomics_path),
        "cmatrices_file": str(cmatrices_path),
        "cshape_file": str(cshape_path),
        "image": str(Path(args.image).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "image_spacing": spacing,
        "mask_spacing": mask_spacing,
        "n_features": int(len(out)),
    }
    Path(args.metadata_json).write_text(json.dumps(metadata, indent=2))
    print(f"[{args.variant}] wrote {len(out)} features to {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
