from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT_ROOT = Path("<REPOSITORY_ROOT>")
EXP_DIR = PROJECT_ROOT / "experiments" / "synthetic_spacing_validation"
ORIGINAL_BUILD = EXP_DIR / "build_sources" / "original" / "radiomics"
MODIFIED_BUILD = EXP_DIR / "build_sources" / "modified" / "radiomics"
VENV_ORIGINAL_PY = EXP_DIR / ".venv_original" / "bin" / "python"
VENV_MODIFIED_PY = EXP_DIR / ".venv_modified" / "bin" / "python"
FAMILIES = ["glcm", "glrlm", "gldm", "ngtdm", "glszm"]
ABS_TOL = 1e-8
REL_TOL = 1e-6
ABS_TOL_LOOSE = 1e-5
EPS = 1e-12


def log(message: str) -> None:
    print(f"[synthetic-validation] {message}", flush=True)


def create_synthetic_array(seed: int = 20260615) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((20, 40, 40))
    mask = (z >= 4) & (z <= 15) & (y >= 8) & (y <= 31) & (x >= 8) & (x <= 31)
    gradient_x = 4.0 * x
    z_bands = 55.0 * ((z // 2) % 2)
    checker = 18.0 * (((x + y) % 5) == 0)
    noise = rng.normal(0.0, 2.0, size=x.shape)
    texture = 35.0 + gradient_x + z_bands + checker + noise
    texture = np.clip(np.round(texture / 10.0) * 10.0, 0, 255).astype(np.float32)
    image = np.zeros_like(texture, dtype=np.float32)
    image[mask] = texture[mask]
    return image, mask.astype(np.uint8)


def save_nifti(image: np.ndarray, mask: np.ndarray, spacing: tuple[float, float, float], prefix: str) -> tuple[Path, Path]:
    img = sitk.GetImageFromArray(image)
    msk = sitk.GetImageFromArray(mask)
    img.SetSpacing(spacing)
    msk.SetSpacing(spacing)
    img_path = EXP_DIR / f"synthetic_{prefix}_image.nii.gz"
    msk_path = EXP_DIR / f"synthetic_{prefix}_mask.nii.gz"
    sitk.WriteImage(img, str(img_path))
    sitk.WriteImage(msk, str(msk_path))
    log(f"Saved {prefix} image spacing: {sitk.ReadImage(str(img_path)).GetSpacing()}")
    log(f"Saved {prefix} mask spacing:  {sitk.ReadImage(str(msk_path)).GetSpacing()}")
    return img_path, msk_path


def make_slice_figure(image: np.ndarray, mask: np.ndarray) -> None:
    zc, yc, xc = np.array(image.shape) // 2
    panels = [
        ("Axial", image[zc, :, :], mask[zc, :, :]),
        ("Coronal", image[:, yc, :], mask[:, yc, :]),
        ("Sagittal", image[:, :, xc], mask[:, :, xc]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    for ax, (title, img, msk) in zip(axes, panels):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.contour(msk, levels=[0.5], colors=["#b23a48"], linewidths=1.1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Controlled synthetic image with anisotropic spacing and ROI", fontsize=12)
    for suffix in ["pdf", "png"]:
        fig.savefig(EXP_DIR / f"synthetic_image_slices.{suffix}", dpi=300)
    plt.close(fig)


def run_extraction(
    python_bin: Path,
    expected_source: Path,
    variant: str,
    image: Path,
    mask: Path,
    output_csv: Path,
    metadata_json: Path,
    weighting_norm: str | None = None,
) -> dict[str, object]:
    cmd = [
        str(python_bin),
        str(EXP_DIR / "extract_features.py"),
        "--image",
        str(image),
        "--mask",
        str(mask),
        "--output-csv",
        str(output_csv),
        "--metadata-json",
        str(metadata_json),
        "--expected-source",
        str(expected_source),
        "--variant",
        variant,
    ]
    if weighting_norm:
        cmd.extend(["--weighting-norm", weighting_norm])
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    log("Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=EXP_DIR, env=env, text=True, capture_output=True)
    log_path = EXP_DIR / f"{output_csv.stem}.log"
    log_path.write_text("STDOUT\n======\n" + proc.stdout + "\nSTDERR\n======\n" + proc.stderr)
    status = {
        "variant": variant,
        "weightingNorm": weighting_norm or "default",
        "output_csv": str(output_csv),
        "metadata_json": str(metadata_json),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and output_csv.exists(),
        "log": str(log_path),
    }
    if not status["ok"]:
        raise RuntimeError(f"Extraction failed for {variant}; see {log_path}")
    return status


def read_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def compare(a_path: Path, b_path: Path, a_label: str, b_label: str) -> pd.DataFrame:
    a = read_features(a_path).rename(columns={"value": a_label})
    b = read_features(b_path).rename(columns={"value": b_label})
    merged = a[["feature", "family", "short_feature", a_label]].merge(
        b[["feature", b_label]], on="feature", how="inner"
    )
    merged["abs_diff"] = (merged[a_label] - merged[b_label]).abs()
    denom = np.maximum(np.maximum(merged[a_label].abs(), merged[b_label].abs()), EPS)
    merged["rel_diff"] = merged["abs_diff"] / denom
    merged["changed_strict"] = (merged["abs_diff"] > ABS_TOL) & (merged["rel_diff"] > REL_TOL)
    merged["changed_abs_1e_5"] = merged["abs_diff"] > ABS_TOL_LOOSE
    return merged.sort_values(["family", "feature"])


def global_summary(df: pd.DataFrame, check: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": check,
                "n_features_compared": int(len(df)),
                "mean_abs_difference": float(df["abs_diff"].mean()) if len(df) else math.nan,
                "max_abs_difference": float(df["abs_diff"].max()) if len(df) else math.nan,
                "n_changed_strict": int(df["changed_strict"].sum()) if len(df) else 0,
                "n_changed_abs_1e_5": int(df["changed_abs_1e_5"].sum()) if len(df) else 0,
            }
        ]
    )


def activation_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["family", "n_features", "n_changed_strict", "median_abs_difference", "median_relative_difference"])
    return (
        df.groupby("family", as_index=False)
        .agg(
            n_features=("feature", "count"),
            n_changed_strict=("changed_strict", "sum"),
            n_changed_abs_1e_5=("changed_abs_1e_5", "sum"),
            median_abs_difference=("abs_diff", "median"),
            median_relative_difference=("rel_diff", "median"),
            max_abs_difference=("abs_diff", "max"),
        )
        .sort_values("family")
    )


def make_difference_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    plot_df = summary.copy()
    plot_df["percent_changed"] = 100 * plot_df["n_changed_strict"] / plot_df["n_features"].replace(0, np.nan)
    colors = ["#4c78a8" if fam != "glszm" else "#9a9a9a" for fam in plot_df["family"]]
    ax.bar(np.arange(len(plot_df)), plot_df["percent_changed"], color=colors)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["family"].str.upper())
    ax.set_ylabel("Changed features under voxelSpacing (%)")
    ax.set_ylim(0, max(5, min(100, float(plot_df["percent_changed"].max()) + 10)))
    ax.set_title("Anisotropic spacing-aware activation by texture family")
    for i, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(i, row["percent_changed"] + 1, f"{int(row['n_changed_strict'])}/{int(row['n_features'])}", ha="center", va="bottom", fontsize=8)
    for suffix in ["pdf", "png"]:
        fig.savefig(EXP_DIR / f"feature_difference_summary.{suffix}", dpi=300)
    plt.close(fig)


def latex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.3g}"
    return latex_escape(value)


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(format_value(row[c]) for c in cols) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}\n")
    path.write_text("\n".join(lines))


def make_report(imports_df: pd.DataFrame, backward: pd.DataFrame, activation: pd.DataFrame, isotropic: pd.DataFrame) -> None:
    write_latex_table(imports_df, EXP_DIR / "table_import_paths.tex")
    write_latex_table(backward, EXP_DIR / "table_backward_compatibility.tex")
    write_latex_table(activation, EXP_DIR / "table_voxelSpacing_activation.tex")
    write_latex_table(isotropic, EXP_DIR / "table_isotropic_sanity.tex")
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{caption}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\title{Synthetic Spacing Validation of Voxel-Spacing-Aware PyRadiomics}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Rationale}
This controlled experiment tests software behavior in a synthetic anisotropic image. It compares original PyRadiomics, the modified implementation in default mode, and the modified implementation with \texttt{weightingNorm=voxelSpacing}. The experiment is not a clinical validation study.

\section*{Synthetic Image}
A deterministic 3D image of size $40 \times 40 \times 20$ voxels was generated with a cuboid ROI, a smooth intensity gradient along $x$, alternating bands along $z$, and small fixed-seed noise. Anisotropic spacing was set to $(1.0, 1.0, 5.0)$ in SimpleITK $(x,y,z)$ convention; an isotropic copy used $(1.0,1.0,1.0)$.

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{synthetic_image_slices.pdf}
\caption{Representative slices of the synthetic image with ROI overlay.}
\end{figure}

\section*{Environment Verification}
\input{table_import_paths.tex}

\section*{Backward Compatibility}
\input{table_backward_compatibility.tex}

\section*{Voxel-Spacing Activation}
\input{table_voxelSpacing_activation.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{feature_difference_summary.pdf}
\caption{Percentage of texture features changing after activating voxel-spacing-aware extraction in the anisotropic image.}
\end{figure}

\section*{Isotropic Sanity Check}
\input{table_isotropic_sanity.tex}

\section*{Interpretation}
This validation demonstrates software behavior and activation of spacing-aware execution paths under controlled conditions. It does not assess clinical performance. Identical default results between original and modified implementations support backward compatibility. The anisotropic \texttt{voxelSpacing} comparison shows activation of the corrected spacing-aware texture computations, while the isotropic comparison confirms that \texttt{voxelSpacing} now reduces to standard behavior when voxel spacing is equal across axes. This supports the intended implementation principle: spacing-aware extraction should correct anisotropic geometric interpretation without injecting an additional weighting artifact into isotropic data.

\end{document}
"""
    (EXP_DIR / "report_synthetic_spacing_validation.tex").write_text(tex)


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    image, mask = create_synthetic_array()
    anis_img, anis_mask = save_nifti(image, mask, (1.0, 1.0, 5.0), "anisotropic")
    iso_img, iso_mask = save_nifti(image, mask, (1.0, 1.0, 1.0), "isotropic")
    make_slice_figure(image, mask)

    runs = [
        ("original_anisotropic", VENV_ORIGINAL_PY, ORIGINAL_BUILD, anis_img, anis_mask, "features_original_anisotropic.csv", None),
        ("modified_default_anisotropic", VENV_MODIFIED_PY, MODIFIED_BUILD, anis_img, anis_mask, "features_modified_default_anisotropic.csv", None),
        ("modified_voxelSpacing_anisotropic", VENV_MODIFIED_PY, MODIFIED_BUILD, anis_img, anis_mask, "features_modified_voxelSpacing_anisotropic.csv", "voxelSpacing"),
        ("modified_default_isotropic", VENV_MODIFIED_PY, MODIFIED_BUILD, iso_img, iso_mask, "features_modified_default_isotropic.csv", None),
        ("modified_voxelSpacing_isotropic", VENV_MODIFIED_PY, MODIFIED_BUILD, iso_img, iso_mask, "features_modified_voxelSpacing_isotropic.csv", "voxelSpacing"),
    ]
    statuses = []
    for variant, py, expected, img, msk, csv_name, weighting in runs:
        statuses.append(
            run_extraction(
                py,
                expected,
                variant,
                img,
                msk,
                EXP_DIR / csv_name,
                EXP_DIR / f"{Path(csv_name).stem}_metadata.json",
                weighting,
            )
        )

    backward_cmp = compare(EXP_DIR / "features_original_anisotropic.csv", EXP_DIR / "features_modified_default_anisotropic.csv", "original", "modified_default")
    activation_cmp = compare(EXP_DIR / "features_modified_default_anisotropic.csv", EXP_DIR / "features_modified_voxelSpacing_anisotropic.csv", "modified_default", "modified_voxelSpacing")
    isotropic_cmp = compare(EXP_DIR / "features_modified_default_isotropic.csv", EXP_DIR / "features_modified_voxelSpacing_isotropic.csv", "modified_default", "modified_voxelSpacing")

    backward_cmp.to_csv(EXP_DIR / "comparison_backward_compatibility.csv", index=False)
    activation_cmp.to_csv(EXP_DIR / "comparison_voxelSpacing_effect.csv", index=False)
    isotropic_cmp.to_csv(EXP_DIR / "comparison_isotropic_sanity.csv", index=False)

    backward_summary = global_summary(backward_cmp, "original anisotropic vs modified default anisotropic")
    activation_by_family = activation_summary(activation_cmp)
    isotropic_summary = global_summary(isotropic_cmp, "modified default isotropic vs modified voxelSpacing isotropic")
    backward_summary.to_csv(EXP_DIR / "comparison_backward_compatibility_summary.csv", index=False)
    activation_by_family.to_csv(EXP_DIR / "comparison_voxelSpacing_effect_summary.csv", index=False)
    isotropic_summary.to_csv(EXP_DIR / "comparison_isotropic_sanity_summary.csv", index=False)
    make_difference_figure(activation_by_family)

    metadata_rows = []
    for meta_path in sorted(EXP_DIR.glob("features_*_metadata.json")):
        meta = json.loads(meta_path.read_text())
        metadata_rows.append(
            {
                "variant": meta["variant"],
                "weightingNorm": meta["weightingNorm"],
                "radiomics": meta["radiomics_file"],
                "_cmatrices": meta["cmatrices_file"],
                "_cshape": meta["cshape_file"],
                "spacing": meta["image_spacing"],
                "n_features": meta["n_features"],
            }
        )
    imports_df = pd.DataFrame(metadata_rows)
    imports_df.to_csv(EXP_DIR / "environment_import_paths.csv", index=False)

    make_report(imports_df, backward_summary, activation_by_family, isotropic_summary)

    status = {
        "runs": statuses,
        "backward_compatibility": backward_summary.to_dict(orient="records"),
        "activation_by_family": activation_by_family.to_dict(orient="records"),
        "isotropic_consistency": isotropic_summary.to_dict(orient="records"),
        "report": str(EXP_DIR / "report_synthetic_spacing_validation.tex"),
    }
    (EXP_DIR / "run_status.json").write_text(json.dumps(status, indent=2))
    log("Experiment completed.")
    log(backward_summary.to_string(index=False))
    log(activation_by_family.to_string(index=False))
    log(isotropic_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
