#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REQUIRED_UPSTREAM_COMMIT = "8ed579383b44806651c463d5e691f3b2b57522ab"
REFERENCE_VERSION = "3.1.1.dev111+g8ed579383.d20260712"
FAMILIES = ("glcm", "glrlm", "gldm", "glrlm", "glszm", "ngtdm")
TEXTURE_FAMILIES = ("glcm", "glrlm", "gldm", "glszm", "ngtdm")
BUILD_ROOT = ROOT / ".repro_build"
UPSTREAM_CACHE = ROOT / ".repro_pyradiomics"
RERUN_ROOT = ROOT / "results" / "rerun"
ENV_JSON = ROOT / "results" / "reproduction_environment.json"
EPS = 1e-12
ABS_TOL = 1e-8
REL_TOL = 1e-6


WORKER_CODE = r'''
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import SimpleITK as sitk


TEXTURE_FAMILIES = ("glcm", "glrlm", "gldm", "glszm", "ngtdm")


def parse_feature_name(name):
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] == "original" and parts[1].lower() in TEXTURE_FAMILIES:
        return parts[1].lower(), "_".join(parts[2:])
    return None, None


def geometry_from_image(image, mask):
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    spacing_zyx = tuple(reversed(spacing_xyz))
    shape_zyx = tuple(int(v) for v in sitk.GetArrayFromImage(image).shape)
    mask_array = sitk.GetArrayFromImage(mask)
    h = min(spacing_zyx)
    q = tuple(max(1, int(round(v / h))) for v in spacing_zyx)
    represented = tuple(float(v * h) for v in q)
    abs_error = tuple(float(r - s) for r, s in zip(represented, spacing_zyx))
    rel_error = tuple(float(e / s) if s else math.nan for e, s in zip(abs_error, spacing_zyx))
    expanded_shape = tuple(int(s * qi) for s, qi in zip(shape_zyx, q))
    expansion_factor = int(q[0] * q[1] * q[2])
    before = int(np.count_nonzero(mask_array == 1))
    return {
        "spacing_xyz": spacing_xyz,
        "spacing_zyx": spacing_zyx,
        "shape_zyx": shape_zyx,
        "h": float(h),
        "q_zyx": q,
        "expansion_factor_R": expansion_factor,
        "represented_spacing_zyx": represented,
        "absolute_spacing_error_zyx": abs_error,
        "relative_spacing_error_zyx": rel_error,
        "expanded_shape_zyx": expanded_shape,
        "mask_voxels_before": before,
        "mask_voxels_after": int(before * expansion_factor),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--expected-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--spacing-label", default="")
    parser.add_argument("--bin-width", type=float, default=25.0)
    parser.add_argument("--weighting-norm")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    expected = Path(args.expected_root).resolve()
    sys.meta_path = [
        finder for finder in sys.meta_path
        if not finder.__class__.__module__.startswith("_editable_")
    ]
    sys.path.insert(0, str(expected))
    os.environ["PYTHONNOUSERSITE"] = "1"

    import radiomics
    from radiomics import _cmatrices, _cshape, featureextractor

    module_paths = {
        "radiomics": Path(radiomics.__file__).resolve(),
        "_cmatrices": Path(_cmatrices.__file__).resolve(),
        "_cshape": Path(_cshape.__file__).resolve(),
    }
    for label, path in module_paths.items():
        if expected not in path.parents and path != expected:
            raise RuntimeError(f"{label} resolved outside local build: {path}; expected under {expected}")

    image = sitk.ReadImage(str(args.image))
    mask = sitk.ReadImage(str(args.mask))
    geom = geometry_from_image(image, mask)

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=args.bin_width,
        label=1,
        force2D=False,
        additionalInfo=False,
        resampledPixelSpacing=None,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    for family in TEXTURE_FAMILIES:
        extractor.enableFeatureClassByName(family)
    if args.weighting_norm:
        extractor.settings["weightingNorm"] = args.weighting_norm

    start = time.perf_counter()
    if args.profile:
        tracemalloc.start()
    result = extractor.execute(str(args.image), str(args.mask))
    elapsed = time.perf_counter() - start
    peak_mb = None
    if args.profile:
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        tracemalloc.stop()

    rows = []
    for key, value in result.items():
        family, short_feature = parse_feature_name(str(key))
        if family is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if not math.isfinite(numeric):
            continue
        rows.append({
            "feature": str(key),
            "family": family,
            "short_feature": short_feature,
            "value": numeric,
        })
    if not rows:
        raise RuntimeError("No texture features were extracted.")
    rows.sort(key=lambda r: (r["family"], r["feature"]))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "family", "short_feature", "value"])
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "variant": args.variant,
        "method": args.method,
        "spacing_label": args.spacing_label,
        "weightingNorm": args.weighting_norm or "default",
        "binWidth": args.bin_width,
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "radiomics_version": getattr(radiomics, "__version__", ""),
        "radiomics_file": str(module_paths["radiomics"]),
        "cmatrices_file": str(module_paths["_cmatrices"]),
        "cshape_file": str(module_paths["_cshape"]),
        "image": str(Path(args.image).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "n_features": len(rows),
        "extraction_time_seconds": elapsed,
        "peak_memory_mb": peak_mb,
        "geometry": geom,
    }
    Path(args.metadata_json).write_text(json.dumps(metadata, indent=2))
    print(json.dumps({
        "variant": args.variant,
        "n_features": len(rows),
        "radiomics": str(module_paths["radiomics"]),
        "_cmatrices": str(module_paths["_cmatrices"]),
        "seconds": elapsed,
        "peak_memory_mb": peak_mb,
    }, indent=2))


if __name__ == "__main__":
    main()
'''


@dataclass
class BuildInfo:
    name: str
    source_root: Path
    overlay: bool
    overlay_files: list[str]
    python: Path


def log(message: str) -> None:
    print(f"[rerun] {message}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, verbose: bool = False) -> subprocess.CompletedProcess[str]:
    if verbose:
        log(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stdout)
    return proc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean() -> None:
    for path in [BUILD_ROOT, UPSTREAM_CACHE, RERUN_ROOT, ENV_JSON]:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            log(f"removed {path.relative_to(ROOT)}")


def candidate_pythons() -> list[Path]:
    candidates = [Path(sys.executable)]
    env_python = os.environ.get("PYRADIOMICS_REPRO_PYTHON")
    if env_python:
        candidates.insert(0, Path(env_python))
    candidates.extend([
        ROOT.parent / "work" / "pyradiomics_pr_test" / ".venv" / "bin" / "python",
        ROOT / ".repro_env" / "bin" / "python",
    ])
    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        candidate = candidate.expanduser().absolute()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def python_has_extraction_deps(python: Path) -> bool:
    code = "import numpy, SimpleITK, pywt, pykwalify, setuptools; import sys; print(sys.version)"
    proc = subprocess.run([str(python), "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode == 0


def choose_extraction_python() -> Path:
    for python in candidate_pythons():
        if python_has_extraction_deps(python):
            return python
    raise RuntimeError(
        "No Python with PyRadiomics extraction dependencies was found. "
        "Install requirements-repro.txt or set PYRADIOMICS_REPRO_PYTHON to a compatible Python."
    )


def dependency_versions(python: Path) -> dict[str, str]:
    code = dedent(
        """
        import importlib, json, platform, sys
        names = ["numpy", "SimpleITK", "pywt", "pykwalify", "pytest", "setuptools"]
        out = {"python_executable": sys.executable, "python_version": sys.version.replace("\\n", " "), "platform": platform.platform(), "machine": platform.machine()}
        for name in names:
            try:
                mod = importlib.import_module(name)
                out[name] = getattr(mod, "__version__", "available")
                out[name + "_file"] = getattr(mod, "__file__", "")
            except Exception as exc:
                out[name] = "missing: " + repr(exc)
        print(json.dumps(out))
        """
    )
    proc = run([str(python), "-c", code])
    return json.loads(proc.stdout)


def default_local_upstream() -> Path | None:
    path = ROOT.parent / "work" / "pyradiomics_pr_test" / "pyradiomics_upstream"
    return path if path.exists() else None


def obtain_upstream(upstream_source: Path | None, verbose: bool) -> Path:
    if upstream_source is None:
        upstream_source = default_local_upstream()
    if upstream_source is None:
        target = UPSTREAM_CACHE / "pyradiomics"
        if not target.exists():
            UPSTREAM_CACHE.mkdir(parents=True, exist_ok=True)
            log("No local upstream source found; attempting official PyRadiomics clone.")
            run(["git", "clone", "https://github.com/AIM-Harvard/pyradiomics.git", str(target)], verbose=verbose)
        upstream_source = target
    upstream_source = upstream_source.resolve()
    if not (upstream_source / ".git").exists():
        raise RuntimeError(f"Upstream source is not a Git repository: {upstream_source}")
    proc = run(["git", "-C", str(upstream_source), "cat-file", "-t", REQUIRED_UPSTREAM_COMMIT], verbose=verbose)
    if proc.stdout.strip() != "commit":
        raise RuntimeError(f"Required commit is not present in upstream source: {REQUIRED_UPSTREAM_COMMIT}")
    return upstream_source


def archive_commit(upstream: Path, destination: Path, verbose: bool) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    proc = run(["git", "-C", str(upstream), "archive", "--format=tar", REQUIRED_UPSTREAM_COMMIT], verbose=verbose)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout.encode("latin1")), mode="r:") as tar:
        tar.extractall(destination)


def archive_commit_bytes(upstream: Path, destination: Path, verbose: bool) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "-C", str(upstream), "archive", "--format=tar", REQUIRED_UPSTREAM_COMMIT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tar:
        try:
            tar.extractall(destination, filter="data")
        except TypeError:
            tar.extractall(destination)


def overlay_implementation(source_root: Path) -> list[str]:
    overlay_root = ROOT / "implementation" / "radiomics"
    overlay_files: list[str] = []
    for src in sorted(overlay_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(overlay_root)
        dst = source_root / "radiomics" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        overlay_files.append(("radiomics" / rel).as_posix())
    return overlay_files


def write_build_helpers(source_root: Path) -> None:
    version_file = source_root / "radiomics" / "_version.py"
    version_file.write_text(
        "\n".join([
            "version = __version__ = '3.1.1.dev111+g8ed579383.d20260712'",
            "__version_tuple__ = version_tuple = (3, 1, 1, 'dev111', 'g8ed579383.d20260712')",
            "__commit_id__ = commit_id = 'g8ed579383'",
            "",
        ]),
        encoding="utf-8",
    )
    setup_py = source_root / "setup.py"
    setup_py.write_text(
        dedent(
            """
            from setuptools import Extension, setup
            import numpy

            setup(
                name="pyradiomics-paper-vs",
                version="3.1.1.dev111+g8ed579383.d20260712",
                packages=["radiomics", "radiomics.scripts"],
                package_data={"radiomics": ["schemas/*.yaml"]},
                ext_modules=[
                    Extension(
                        "radiomics._cmatrices",
                        ["radiomics/src/_cmatrices.c", "radiomics/src/cmatrices.c"],
                        include_dirs=[numpy.get_include()],
                    ),
                    Extension(
                        "radiomics._cshape",
                        ["radiomics/src/_cshape.c", "radiomics/src/cshape.c"],
                        include_dirs=[numpy.get_include()],
                    ),
                ],
            )
            """
        ),
        encoding="utf-8",
    )


def build_source_tree(name: str, upstream: Path, python: Path, overlay: bool, verbose: bool) -> BuildInfo:
    source_root = BUILD_ROOT / name
    archive_commit_bytes(upstream, source_root, verbose=verbose)
    overlay_files: list[str] = []
    if overlay:
        overlay_files = overlay_implementation(source_root)
    write_build_helpers(source_root)
    run([str(python), "setup.py", "build_ext", "--inplace"], cwd=source_root, verbose=verbose)
    return BuildInfo(name=name, source_root=source_root, overlay=overlay, overlay_files=overlay_files, python=python)


def write_worker() -> Path:
    path = BUILD_ROOT / "extract_worker.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(WORKER_CODE, encoding="utf-8")
    return path


def verify_local_build(build: BuildInfo, verbose: bool) -> dict[str, str]:
    code = dedent(
        f"""
        import json, sys
        from pathlib import Path
        expected = Path({str(build.source_root)!r}).resolve()
        sys.meta_path = [
            finder for finder in sys.meta_path
            if not finder.__class__.__module__.startswith("_editable_")
        ]
        sys.path.insert(0, str(expected))
        import radiomics
        from radiomics import _cmatrices, _cshape
        paths = {{
            "radiomics_package_path": str(Path(radiomics.__file__).resolve()),
            "radiomics_version": getattr(radiomics, "__version__", ""),
            "cmatrices_path": str(Path(_cmatrices.__file__).resolve()),
            "cshape_path": str(Path(_cshape.__file__).resolve()),
        }}
        for label, value in paths.items():
            if label.endswith("_path") and expected not in Path(value).parents:
                raise RuntimeError(f"{{label}} outside local build: {{value}}")
        print(json.dumps(paths))
        """
    )
    proc = run([str(build.python), "-c", code], verbose=verbose)
    return json.loads(proc.stdout)


def extract_features(
    build: BuildInfo,
    worker: Path,
    image: Path,
    mask: Path,
    out_csv: Path,
    out_json: Path,
    variant: str,
    method: str,
    spacing_label: str = "",
    bin_width: float = 25.0,
    weighting_norm: str | None = None,
    profile: bool = False,
    verbose: bool = False,
) -> None:
    cmd = [
        str(build.python),
        str(worker),
        "--image",
        str(image),
        "--mask",
        str(mask),
        "--output-csv",
        str(out_csv),
        "--metadata-json",
        str(out_json),
        "--expected-root",
        str(build.source_root),
        "--variant",
        variant,
        "--method",
        method,
        "--spacing-label",
        spacing_label,
        "--bin-width",
        str(bin_width),
    ]
    if weighting_norm:
        cmd.extend(["--weighting-norm", weighting_norm])
    if profile:
        cmd.append("--profile")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    run(cmd, cwd=ROOT, env=env, verbose=verbose)


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).sort_values(["family", "feature"]).reset_index(drop=True)


def compare_features(reference: pd.DataFrame, comparison: pd.DataFrame, reference_label: str, comparison_label: str) -> pd.DataFrame:
    merged = reference.rename(columns={"value": reference_label})[
        ["feature", "family", "short_feature", reference_label]
    ].merge(
        comparison.rename(columns={"value": comparison_label})[["feature", comparison_label]],
        on="feature",
        how="inner",
    )
    merged["abs_diff"] = (merged[reference_label] - merged[comparison_label]).abs()
    denom = np.maximum(np.maximum(merged[reference_label].abs(), merged[comparison_label].abs()), EPS)
    merged["rel_diff"] = merged["abs_diff"] / denom
    merged["changed"] = (merged["abs_diff"] > ABS_TOL) & (merged["rel_diff"] > REL_TOL)
    return merged.sort_values(["family", "feature"]).reset_index(drop=True)


def global_summary(df: pd.DataFrame, check: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "check": check,
        "n_features_compared": int(len(df)),
        "mean_abs_difference": float(df["abs_diff"].mean()),
        "max_abs_difference": float(df["abs_diff"].max()),
        "n_changed_strict": int(df["changed"].sum()),
        "n_changed_abs_1e_5": int((df["abs_diff"] > 1e-5).sum()),
    }])


def activation_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("family", as_index=False)
        .agg(
            n_features=("feature", "count"),
            n_changed_strict=("changed", "sum"),
            n_changed_abs_1e_5=("abs_diff", lambda s: int((s > 1e-5).sum())),
            median_abs_difference=("abs_diff", "median"),
            median_relative_difference=("rel_diff", "median"),
            max_abs_difference=("abs_diff", "max"),
        )
        .sort_values("family")
    )


def spearman(a: pd.Series, b: pd.Series) -> float:
    ar = a.rank(method="average").to_numpy(dtype=float)
    br = b.rank(method="average").to_numpy(dtype=float)
    if np.std(ar) == 0 or np.std(br) == 0:
        return math.nan
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(a: pd.Series, b: pd.Series) -> float:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    if np.std(av) == 0 or np.std(bv) == 0:
        return math.nan
    return float(np.corrcoef(av, bv)[0, 1])


def geometry_row(meta_path: Path, spacing_label: str) -> dict[str, object]:
    meta = json.loads(meta_path.read_text())
    geom = meta["geometry"]
    z, y, x = geom["spacing_zyx"]
    qz, qy, qx = geom["q_zyx"]
    rz, ry, rx = geom["represented_spacing_zyx"]
    ez, ey, ex = geom["absolute_spacing_error_zyx"]
    rez, rey, rex = geom["relative_spacing_error_zyx"]
    return {
        "spacing_z": z,
        "spacing_y": y,
        "spacing_x": x,
        "h": geom["h"],
        "q_z": qz,
        "q_y": qy,
        "q_x": qx,
        "expansion_factor_R": geom["expansion_factor_R"],
        "represented_spacing_z": rz,
        "represented_spacing_y": ry,
        "represented_spacing_x": rx,
        "absolute_spacing_error_z": ez,
        "relative_spacing_error_z": rez,
        "spacing_label": spacing_label,
    }


def run_synthetic_validation(original: BuildInfo, modified: BuildInfo, worker: Path, verbose: bool) -> Path:
    exp = ROOT / "experiments" / "synthetic_spacing_validation"
    out = RERUN_ROOT / "synthetic_spacing_validation"
    out.mkdir(parents=True, exist_ok=True)
    runs = [
        (original, exp / "synthetic_anisotropic_image.nii.gz", exp / "synthetic_anisotropic_mask.nii.gz", "features_original_anisotropic", "native_original", None),
        (modified, exp / "synthetic_anisotropic_image.nii.gz", exp / "synthetic_anisotropic_mask.nii.gz", "features_modified_default_anisotropic", "modified_default", None),
        (modified, exp / "synthetic_anisotropic_image.nii.gz", exp / "synthetic_anisotropic_mask.nii.gz", "features_modified_voxelSpacing_anisotropic", "voxel_spacing_aware", "voxelSpacing"),
        (modified, exp / "synthetic_isotropic_image.nii.gz", exp / "synthetic_isotropic_mask.nii.gz", "features_modified_default_isotropic", "modified_default", None),
        (modified, exp / "synthetic_isotropic_image.nii.gz", exp / "synthetic_isotropic_mask.nii.gz", "features_modified_voxelSpacing_isotropic", "voxel_spacing_aware", "voxelSpacing"),
    ]
    for build, image, mask, stem, method, weighting in runs:
        extract_features(build, worker, image, mask, out / f"{stem}.csv", out / f"{stem}_metadata.json", stem, method, weighting_norm=weighting, verbose=verbose)

    original_anis = read_features(out / "features_original_anisotropic.csv")
    modified_default_anis = read_features(out / "features_modified_default_anisotropic.csv")
    modified_vs_anis = read_features(out / "features_modified_voxelSpacing_anisotropic.csv")
    modified_default_iso = read_features(out / "features_modified_default_isotropic.csv")
    modified_vs_iso = read_features(out / "features_modified_voxelSpacing_isotropic.csv")
    backward = compare_features(original_anis, modified_default_anis, "original", "modified_default")
    activation = compare_features(modified_default_anis, modified_vs_anis, "modified_default", "modified_voxelSpacing")
    isotropic = compare_features(modified_default_iso, modified_vs_iso, "modified_default", "modified_voxelSpacing")
    backward.to_csv(out / "comparison_backward_compatibility.csv", index=False)
    activation.to_csv(out / "comparison_voxelSpacing_effect.csv", index=False)
    isotropic.to_csv(out / "comparison_isotropic_sanity.csv", index=False)
    global_summary(backward, "original anisotropic vs modified default anisotropic").to_csv(out / "comparison_backward_compatibility_summary.csv", index=False)
    activation_summary(activation).to_csv(out / "comparison_voxelSpacing_effect_summary.csv", index=False)
    global_summary(isotropic, "modified default isotropic vs modified voxelSpacing isotropic").to_csv(out / "comparison_isotropic_sanity_summary.csv", index=False)
    return out


def run_rounding(modified: BuildInfo, worker: Path, verbose: bool) -> Path:
    exp = ROOT / "experiments" / "finite_volume_rounding_sensitivity"
    out = RERUN_ROOT / "finite_volume_rounding_sensitivity" / "results"
    out.mkdir(parents=True, exist_ok=True)
    labels = ["z2", "z2.5", "z3", "z3.5", "z5", "z5.5"]
    feature_frames = {}
    geometry_rows = []
    for label in labels:
        extract_features(
            modified,
            worker,
            exp / "data" / f"{label}_image.nii.gz",
            exp / "data" / f"{label}_mask.nii.gz",
            out / f"{label}_voxel_spacing_aware_features.csv",
            out / f"{label}_metadata.json",
            label,
            "voxel_spacing_aware",
            spacing_label=label,
            weighting_norm="voxelSpacing",
            verbose=verbose,
        )
        feature_frames[label] = read_features(out / f"{label}_voxel_spacing_aware_features.csv")
        geometry_rows.append(geometry_row(out / f"{label}_metadata.json", label))
    pd.DataFrame(geometry_rows).to_csv(out / "rounding_geometry_table.csv", index=False)

    comparisons = [("z2", "z2.5"), ("z2.5", "z3"), ("z3", "z3.5"), ("z5", "z5.5")]
    all_rows = []
    summaries = []
    for a, b in comparisons:
        cmp = compare_features(feature_frames[a], feature_frames[b], a, b)
        wide = cmp[["feature", "family", "short_feature", a, b, "abs_diff", "rel_diff", "changed"]].copy()
        wide.insert(0, "comparison", f"{a}_vs_{b}")
        for label in labels:
            if label not in (a, b):
                wide[label] = np.nan
        all_rows.append(wide)
        summaries.append(
            cmp.groupby("family", as_index=False).agg(
                n_features=("feature", "count"),
                n_changed=("changed", "sum"),
                median_abs_diff=("abs_diff", "median"),
                median_rel_diff=("rel_diff", "median"),
            )
        )
        summaries[-1]["comparison"] = f"{a}_vs_{b}"
        corrs = []
        for family, sub in cmp.groupby("family"):
            corrs.append((family, spearman(sub[a], sub[b])))
        for family, corr in corrs:
            summaries[-1].loc[summaries[-1]["family"] == family, "spearman_corr"] = corr
    pd.concat(all_rows, ignore_index=True).to_csv(out / "rounding_feature_sensitivity.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary[["comparison", "family", "n_features", "n_changed", "median_abs_diff", "median_rel_diff", "spearman_corr"]]
    summary.to_csv(out / "rounding_summary_by_family.csv", index=False)
    return out


def run_binning(original: BuildInfo, modified: BuildInfo, worker: Path, verbose: bool) -> Path:
    exp = ROOT / "experiments" / "binning_sensitivity"
    out = RERUN_ROOT / "binning_sensitivity" / "results"
    out.mkdir(parents=True, exist_ok=True)
    methods = {
        "native_original": (original, exp / "data" / "z5_native_image.nii.gz", exp / "data" / "z5_native_mask.nii.gz", None),
        "resample_linear": (original, exp / "data" / "z5_resample_linear_image.nii.gz", exp / "data" / "z5_resample_linear_mask.nii.gz", None),
        "voxel_spacing_aware": (modified, exp / "data" / "z5_native_image.nii.gz", exp / "data" / "z5_native_mask.nii.gz", "voxelSpacing"),
    }
    rows = []
    summary_rows = []
    for bin_width in [5, 10, 25, 50]:
        feature_by_method = {}
        for method, (build, image, mask, weighting) in methods.items():
            stem = f"bin{bin_width}_{method}"
            extract_features(
                build,
                worker,
                image,
                mask,
                out / f"{stem}_features.csv",
                out / f"{stem}_metadata.json",
                stem,
                method,
                bin_width=float(bin_width),
                weighting_norm=weighting,
                verbose=verbose,
            )
            df = read_features(out / f"{stem}_features.csv")
            df["binWidth"] = bin_width
            df["method"] = method
            rows.append(df)
            feature_by_method[method] = df
        native = feature_by_method["native_original"]
        linear = feature_by_method["resample_linear"]
        vs = feature_by_method["voxel_spacing_aware"]
        cmp_native = compare_features(native, vs, "native_original", "voxel_spacing_aware")
        cmp_linear = compare_features(linear, vs, "resample_linear", "voxel_spacing_aware")
        summary_rows.append({
            "binWidth": bin_width,
            "n_valid_features": int(len(cmp_native)),
            "median_rel_diff_vs_native": float(cmp_native["rel_diff"].median()),
            "median_rel_diff_vs_resample_linear": float(cmp_linear["rel_diff"].median()),
            "spearman_corr_vs_native": spearman(cmp_native["native_original"], cmp_native["voxel_spacing_aware"]),
            "spearman_corr_vs_resample_linear": spearman(cmp_linear["resample_linear"], cmp_linear["voxel_spacing_aware"]),
        })
    pd.concat(rows, ignore_index=True).to_csv(out / "binning_features.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "binning_summary.csv", index=False)
    return out


def run_resampling(original: BuildInfo, modified: BuildInfo, worker: Path, verbose: bool) -> Path:
    exp = ROOT / "experiments" / "resampling_baseline_comparison"
    out = RERUN_ROOT / "resampling_baseline_comparison" / "results"
    out.mkdir(parents=True, exist_ok=True)
    phantoms = {
        "mild_z2": (2.0, 1.0, 1.0),
        "moderate_z3": (3.0, 1.0, 1.0),
        "strong_z5": (5.0, 1.0, 1.0),
    }
    methods = ["native_original", "resample_nearest", "resample_linear", "resample_bspline", "voxel_spacing_aware"]
    all_features = []
    metadata_rows = []
    for label, spacing in phantoms.items():
        method_frames = {}
        for method in methods:
            if method == "voxel_spacing_aware":
                build = modified
                image = exp / "data" / f"{label}_native_image.nii.gz"
                mask = exp / "data" / f"{label}_native_mask.nii.gz"
                weighting = "voxelSpacing"
            else:
                build = original
                suffix = "native" if method == "native_original" else method
                image = exp / "data" / f"{label}_{suffix}_image.nii.gz"
                mask = exp / "data" / f"{label}_{suffix}_mask.nii.gz"
                weighting = None
            stem = f"{label}_{method}"
            extract_features(build, worker, image, mask, out / f"{stem}_features.csv", out / f"{stem}_metadata.json", stem, method, spacing_label=label, weighting_norm=weighting, verbose=verbose)
            df = read_features(out / f"{stem}_features.csv")
            df["spacing_label"] = label
            df["spacing_z"], df["spacing_y"], df["spacing_x"] = spacing
            df["method"] = method
            all_features.append(df)
            method_frames[method] = df
            metadata_rows.append({"spacing_label": label, "method": method, "features": str((out / f"{stem}_features.csv").relative_to(ROOT))})
    features_all = pd.concat(all_features, ignore_index=True)
    features_all.to_csv(out / "features_all_methods.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(out / "extraction_metadata.csv", index=False)

    diff_all = []
    linear_diff_all = []
    for label in phantoms:
        native = features_all[(features_all.spacing_label == label) & (features_all.method == "native_original")]
        linear = features_all[(features_all.spacing_label == label) & (features_all.method == "resample_linear")]
        method_frames = {m: features_all[(features_all.spacing_label == label) & (features_all.method == m)] for m in methods}
        for method, frame in method_frames.items():
            cmp = compare_features(native, frame, "reference_value", "comparison_value")
            cmp.insert(0, "method", method)
            cmp.insert(0, "spacing_label", label)
            for m, fdf in method_frames.items():
                cmp = cmp.merge(fdf[["feature", "value"]].rename(columns={"value": m}), on="feature", how="left")
            diff_all.append(cmp)
            cmp_linear = compare_features(linear, frame, "reference_value", "comparison_value")
            cmp_linear.insert(0, "method", method)
            cmp_linear.insert(0, "spacing_label", label)
            linear_diff_all.append(cmp_linear)
    diffs = pd.concat(diff_all, ignore_index=True)
    diffs.to_csv(out / "feature_differences_vs_native.csv", index=False)
    pd.concat(linear_diff_all, ignore_index=True).to_csv(out / "feature_differences_vs_linear_resample.csv", index=False)

    summary_rows = []
    for (label, method, family), sub in diffs.groupby(["spacing_label", "method", "family"]):
        native_values = sub["reference_value"]
        comparison_values = sub["comparison_value"]
        linear_values = sub["resample_linear"]
        summary_rows.append({
            "spacing_label": label,
            "method": method,
            "family": family,
            "n_features": int(len(sub)),
            "n_changed_vs_native": int(sub["changed"].sum()),
            "median_abs_diff_vs_native": float(sub["abs_diff"].median()),
            "median_rel_diff_vs_native": float(sub["rel_diff"].median()),
            "spearman_corr_with_native": spearman(native_values, comparison_values),
            "spearman_corr_with_resample_linear": spearman(linear_values, comparison_values),
            "pearson_corr_with_resample_linear": pearson(linear_values, comparison_values),
        })
    pd.DataFrame(summary_rows).sort_values(["spacing_label", "method", "family"]).to_csv(out / "summary_by_family.csv", index=False)
    return out


def run_profiling(modified: BuildInfo, worker: Path, verbose: bool) -> Path:
    exp = ROOT / "experiments" / "computational_profiling"
    out = RERUN_ROOT / "computational_profiling" / "results"
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted((exp / "data").glob("*_image.nii.gz"))
    rows = []
    for image in cases:
        stem = image.name.removesuffix("_image.nii.gz")
        mask = exp / "data" / f"{stem}_mask.nii.gz"
        extract_features(
            modified,
            worker,
            image,
            mask,
            out / f"{stem}_features.csv",
            out / f"{stem}_metadata.json",
            stem,
            "voxel_spacing_aware",
            weighting_norm="voxelSpacing",
            profile=True,
            verbose=verbose,
        )
        meta = json.loads((out / f"{stem}_metadata.json").read_text())
        geom = meta["geometry"]
        size_label = stem.split("_")[0]
        spacing = geom["spacing_zyx"]
        represented = geom["represented_spacing_zyx"]
        abs_error = geom["absolute_spacing_error_zyx"]
        rel_error = geom["relative_spacing_error_zyx"]
        q = geom["q_zyx"]
        rows.append({
            "size_label": size_label,
            "spacing_z": spacing[0],
            "spacing_y": spacing[1],
            "spacing_x": spacing[2],
            "h": geom["h"],
            "q_z": q[0],
            "q_y": q[1],
            "q_x": q[2],
            "expansion_factor_R": geom["expansion_factor_R"],
            "represented_spacing_z": represented[0],
            "represented_spacing_y": represented[1],
            "represented_spacing_x": represented[2],
            "absolute_spacing_error_z": abs_error[0],
            "relative_spacing_error_z": rel_error[0],
            "input_shape": "x".join(str(v) for v in geom["shape_zyx"]),
            "expanded_shape": "x".join(str(v) for v in geom["expanded_shape_zyx"]),
            "success": True,
            "failure_reason": "",
            "feature_families": "glcm,glrlm,gldm,glszm,ngtdm",
            "number_of_mask_voxels_before": geom["mask_voxels_before"],
            "number_of_mask_voxels_after": geom["mask_voxels_after"],
            "extraction_time_seconds": meta["extraction_time_seconds"],
            "peak_memory_mb": meta["peak_memory_mb"],
        })
    prof = pd.DataFrame(rows).sort_values(["size_label", "expansion_factor_R"])
    prof.to_csv(out / "runtime_memory_profile.csv", index=False)
    summary = (
        prof.groupby("expansion_factor_R", as_index=False)
        .agg(
            n_cases=("success", "count"),
            n_success=("success", "sum"),
            median_runtime_seconds=("extraction_time_seconds", "median"),
            max_runtime_seconds=("extraction_time_seconds", "max"),
            median_peak_memory_mb=("peak_memory_mb", "median"),
            max_peak_memory_mb=("peak_memory_mb", "max"),
        )
        .sort_values("expansion_factor_R")
    )
    summary.to_csv(out / "runtime_summary_by_expansion_factor.csv", index=False)
    return out


def compare_csv_numeric(name: str, frozen: Path, generated: Path, checks: list[dict[str, object]], tol: float = 1e-5) -> None:
    if not generated.exists():
        checks.append({"check": name, "status": "FAIL", "detail": f"missing generated {generated}"})
        return
    a = pd.read_csv(frozen)
    b = pd.read_csv(generated)
    if list(a.columns) != list(b.columns):
        checks.append({"check": name, "status": "FAIL", "detail": "column mismatch"})
        return
    if len(a) != len(b):
        checks.append({"check": name, "status": "FAIL", "detail": f"row count {len(b)} != {len(a)}"})
        return
    details = []
    ok = True
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            av = pd.to_numeric(a[col], errors="coerce")
            bv = pd.to_numeric(b[col], errors="coerce")
            diff = (av - bv).abs().max()
            if pd.notna(diff) and float(diff) > tol:
                ok = False
                details.append(f"{col} max diff {diff}")
        else:
            if not a[col].astype(str).fillna("").equals(b[col].astype(str).fillna("")):
                ok = False
                details.append(f"{col} differs")
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": "; ".join(details) if details else "matched"})


def compare_key_outputs(checks: list[dict[str, object]]) -> None:
    pairs = [
        ("synthetic backward summary", "experiments/synthetic_spacing_validation/comparison_backward_compatibility_summary.csv", "results/rerun/synthetic_spacing_validation/comparison_backward_compatibility_summary.csv"),
        ("synthetic isotropic summary", "experiments/synthetic_spacing_validation/comparison_isotropic_sanity_summary.csv", "results/rerun/synthetic_spacing_validation/comparison_isotropic_sanity_summary.csv"),
        ("synthetic activation summary", "experiments/synthetic_spacing_validation/comparison_voxelSpacing_effect_summary.csv", "results/rerun/synthetic_spacing_validation/comparison_voxelSpacing_effect_summary.csv"),
        ("rounding geometry", "experiments/finite_volume_rounding_sensitivity/results/rounding_geometry_table.csv", "results/rerun/finite_volume_rounding_sensitivity/results/rounding_geometry_table.csv"),
        ("rounding family summary", "experiments/finite_volume_rounding_sensitivity/results/rounding_summary_by_family.csv", "results/rerun/finite_volume_rounding_sensitivity/results/rounding_summary_by_family.csv"),
        ("binning summary", "experiments/binning_sensitivity/results/binning_summary.csv", "results/rerun/binning_sensitivity/results/binning_summary.csv"),
        ("resampling family summary", "experiments/resampling_baseline_comparison/results/summary_by_family.csv", "results/rerun/resampling_baseline_comparison/results/summary_by_family.csv"),
    ]
    for name, frozen, generated in pairs:
        compare_csv_numeric(name, ROOT / frozen, ROOT / generated, checks)
    # Profiling timing and memory are environment-dependent; verify structure and key geometry only.
    frozen = pd.read_csv(ROOT / "experiments" / "computational_profiling" / "results" / "runtime_memory_profile.csv")
    generated = pd.read_csv(RERUN_ROOT / "computational_profiling" / "results" / "runtime_memory_profile.csv")
    ok = (
        set(generated["input_shape"]) == {"16x64x64", "32x96x96", "48x128x128"}
        and set(generated["expansion_factor_R"].astype(int)) == {1, 2, 3, 5, 7}
        and generated["success"].all()
    )
    demanding = generated[(generated.size_label == "large") & (generated.expansion_factor_R == 7)]
    ok = ok and len(demanding) == 1 and demanding.iloc[0]["expanded_shape"] == "48x128x896" and int(demanding.iloc[0]["number_of_mask_voxels_after"]) == 973196
    checks.append({"check": "profiling geometry/success", "status": "PASS" if ok else "FAIL", "detail": "timing and memory recorded but not exact-compared"})


def run_published_verification(checks: list[dict[str, object]], verbose: bool) -> None:
    proc = subprocess.run([sys.executable, "reproduce_paper.py"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = ROOT / "reproduced_results" / "REPRODUCIBILITY_REPORT.md"
    ok = proc.returncode == 0 and report.exists() and "Overall status: **PASS**" in report.read_text(encoding="utf-8")
    checks.append({"check": "published-result verification", "status": "PASS" if ok else "FAIL", "detail": proc.stdout.strip()[-500:]})
    if verbose or not ok:
        print(proc.stdout)


def write_environment_json(upstream: Path, original: BuildInfo, modified: BuildInfo, verification: dict[str, dict[str, str]], deps: dict[str, str]) -> None:
    overlay_hashes = []
    for rel in modified.overlay_files:
        path = modified.source_root / rel
        overlay_hashes.append({"path": rel, "sha256": sha256_file(path)})
    state = subprocess.run(["git", "-C", str(ROOT), "status", "--short"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": REQUIRED_UPSTREAM_COMMIT,
        "upstream_source_path": str(upstream),
        "original_build_path": str(original.source_root),
        "modified_build_path": str(modified.source_root),
        "overlay_file_list": modified.overlay_files,
        "overlay_hashes": overlay_hashes,
        "dependency_versions": deps,
        "build_verification": verification,
        "repository_status_short": state.stdout,
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
    }
    ENV_JSON.parent.mkdir(parents=True, exist_ok=True)
    ENV_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(checks: list[dict[str, object]]) -> bool:
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(checks)
    df.to_csv(RERUN_ROOT / "end_to_end_checks.csv", index=False)
    ok = bool((df["status"] == "PASS").all())
    lines = ["# End-to-end rerun report", "", f"End-to-end rerun: **{'PASS' if ok else 'FAIL'}**", ""]
    for _, row in df.iterrows():
        lines.append(f"- **{row.status}** {row.check}: {row.detail}")
    (RERUN_ROOT / "END_TO_END_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ok


def run_upstream_tests(modified: BuildInfo, verbose: bool, checks: list[dict[str, object]]) -> None:
    shim = BUILD_ROOT / "pytest_shims"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "readline.py").write_text("# macOS libedit/readline shim for pytest collection\n", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(shim) + os.pathsep + str(modified.source_root)
    proc = subprocess.run([str(modified.python), "-X", "faulthandler", "-m", "pytest", "-ra"], cwd=modified.source_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (RERUN_ROOT / "upstream_pytest_output.txt").write_text(proc.stdout, encoding="utf-8")
    ok = proc.returncode == 0
    detail = "passed" if ok else "failed; see results/rerun/upstream_pytest_output.txt"
    checks.append({"check": "optional upstream PyRadiomics test suite", "status": "PASS" if ok else "FAIL", "detail": detail})
    if verbose or not ok:
        print(proc.stdout[-4000:])


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end reproducibility runner for the voxel-spacing-aware technical paper.")
    parser.add_argument("--clean", action="store_true", help="Remove only disposable reproduction artifacts and exit.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing local builds only if provenance files are present.")
    parser.add_argument("--upstream-source", type=Path, help="Local PyRadiomics Git source containing the required upstream commit.")
    parser.add_argument("--run-upstream-tests", action="store_true", help="Run the full upstream PyRadiomics test suite after building.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.clean:
        clean()
        return 0

    required_inputs = [
        ROOT / "experiments" / "synthetic_spacing_validation" / "synthetic_anisotropic_image.nii.gz",
        ROOT / "experiments" / "finite_volume_rounding_sensitivity" / "data" / "z2.5_image.nii.gz",
        ROOT / "experiments" / "binning_sensitivity" / "data" / "z5_native_image.nii.gz",
        ROOT / "experiments" / "resampling_baseline_comparison" / "data" / "mild_z2_native_image.nii.gz",
        ROOT / "experiments" / "computational_profiling" / "data" / "large_z0p7_y0p7_x5p0_image.nii.gz",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_inputs if not path.exists()]
    if missing:
        raise RuntimeError("Missing required synthetic phantom inputs: " + ", ".join(missing))

    extraction_python = choose_extraction_python()
    deps = dependency_versions(extraction_python)
    if not deps["python_version"].startswith("3.10.9"):
        log("WARNING: extraction Python differs from validated Python 3.10.9: " + deps["python_version"])
    upstream = obtain_upstream(args.upstream_source, verbose=args.verbose)

    if args.skip_build:
        if not ENV_JSON.exists():
            raise RuntimeError("--skip-build requested, but existing build/provenance is incomplete.")
        previous = json.loads(ENV_JSON.read_text(encoding="utf-8"))
        if previous.get("upstream_commit") != REQUIRED_UPSTREAM_COMMIT:
            raise RuntimeError("--skip-build provenance commit does not match the required upstream commit.")
        previous_python = previous.get("dependency_versions", {}).get("python_executable")
        if previous_python and Path(previous_python).absolute() != extraction_python.absolute():
            raise RuntimeError("--skip-build provenance Python does not match the selected extraction Python.")
        original = BuildInfo("pyradiomics_upstream_original", Path(previous["original_build_path"]), False, [], extraction_python)
        modified = BuildInfo(
            "pyradiomics_paper_overlay",
            Path(previous["modified_build_path"]),
            True,
            list(previous.get("overlay_file_list", [])),
            extraction_python,
        )
        if not original.source_root.exists() or not modified.source_root.exists():
            raise RuntimeError("--skip-build requested, but the recorded build directories are missing.")
    else:
        log("Building pristine upstream PyRadiomics from exact commit.")
        original = build_source_tree("pyradiomics_upstream_original", upstream, extraction_python, overlay=False, verbose=args.verbose)
        log("Building paper overlay PyRadiomics from exact commit plus implementation/radiomics.")
        modified = build_source_tree("pyradiomics_paper_overlay", upstream, extraction_python, overlay=True, verbose=args.verbose)

    verification = {
        "original": verify_local_build(original, verbose=args.verbose),
        "modified": verify_local_build(modified, verbose=args.verbose),
    }
    worker = write_worker()
    write_environment_json(upstream, original, modified, verification, deps)

    checks: list[dict[str, object]] = []
    log("Running synthetic spacing validation.")
    run_synthetic_validation(original, modified, worker, verbose=args.verbose)
    log("Running finite-volume rounding sensitivity.")
    run_rounding(modified, worker, verbose=args.verbose)
    log("Running binWidth/discretization sensitivity.")
    run_binning(original, modified, worker, verbose=args.verbose)
    log("Running resampling baseline comparison.")
    run_resampling(original, modified, worker, verbose=args.verbose)
    log("Running computational profiling.")
    run_profiling(modified, worker, verbose=args.verbose)

    compare_key_outputs(checks)
    if args.run_upstream_tests:
        run_upstream_tests(modified, verbose=args.verbose, checks=checks)
    run_published_verification(checks, verbose=args.verbose)
    ok = write_report(checks)

    e2e = "PASS" if ok else "FAIL"
    pub = "PASS" if any(c["check"] == "published-result verification" and c["status"] == "PASS" for c in checks) else "FAIL"
    overall = "PASS" if ok and pub == "PASS" else "FAIL"
    print(f"End-to-end rerun: {e2e}")
    print(f"Published-result verification: {pub}")
    print(f"Overall status: {overall}")
    print(f"Report: {RERUN_ROOT / 'END_TO_END_REPORT.md'}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
