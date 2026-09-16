from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


TEXTURE_FAMILIES = ("glcm", "glrlm", "glszm", "gldm", "ngtdm")


def prepare_isolated_import(source_dir: Path, staging_dir: Path) -> Path:
    """Expose a source tree as package name `radiomics` in a private staging dir."""
    source_dir = source_dir.resolve()
    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    package_link = staging_dir / "radiomics"
    if package_link.exists() or package_link.is_symlink():
        if package_link.is_symlink() or package_link.is_file():
            package_link.unlink()
        else:
            shutil.rmtree(package_link)
    package_link.symlink_to(source_dir, target_is_directory=True)
    sys.path.insert(0, str(staging_dir))
    return package_link


def import_radiomics(source_dir: Path, staging_dir: Path):
    prepare_isolated_import(source_dir, staging_dir)
    import radiomics  # noqa: WPS433
    from radiomics import featureextractor  # noqa: WPS433

    return radiomics, featureextractor


def build_extractor(featureextractor, weighting_norm: str | None = None):
    settings = {
        "binWidth": 25,
        "label": 1,
        "force2D": False,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "normalize": False,
        "correctMask": False,
        "additionalInfo": False,
    }
    if weighting_norm is not None:
        settings["weightingNorm"] = weighting_norm

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    for family in TEXTURE_FAMILIES:
        extractor.enableFeatureClassByName(family)
    return extractor


def feature_family(feature_name: str) -> str:
    parts = feature_name.split("_")
    return parts[1].lower() if len(parts) >= 3 else "other"


def execute_extraction(
    source_dir: Path,
    staging_dir: Path,
    image_path: Path,
    mask_path: Path,
    output_csv: Path,
    metadata_json: Path,
    weighting_norm: str | None = None,
) -> None:
    radiomics, featureextractor = import_radiomics(source_dir, staging_dir)
    print(f"IMPORTED_RADIOMICS_PATH={Path(radiomics.__file__).resolve()}", flush=True)

    extractor = build_extractor(featureextractor, weighting_norm=weighting_norm)
    result = extractor.execute(str(image_path), str(mask_path))

    feature_items = {
        k: v
        for k, v in result.items()
        if isinstance(k, str)
        and k.startswith("original_")
        and any(k.startswith(f"original_{fam}_") for fam in TEXTURE_FAMILIES)
    }
    df = pd.DataFrame(
        [
            {
                "feature": k,
                "family": feature_family(k),
                "value": float(v),
            }
            for k, v in sorted(feature_items.items())
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    metadata = {
        "source_dir": str(source_dir.resolve()),
        "radiomics_import_path": str(Path(radiomics.__file__).resolve()),
        "weightingNorm": weighting_norm if weighting_norm is not None else "default",
        "image": str(image_path),
        "mask": str(mask_path),
        "n_features": int(len(df)),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))


def cli(default_source_env: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(os.environ[default_source_env]))
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--weighting-norm", default=None)
    args = parser.parse_args()
    execute_extraction(
        source_dir=args.source_dir,
        staging_dir=args.staging_dir,
        image_path=args.image,
        mask_path=args.mask,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        weighting_norm=args.weighting_norm,
    )
