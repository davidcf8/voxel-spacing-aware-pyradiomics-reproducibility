#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reproduced_results"
OUT.mkdir(exist_ok=True)

checks = []

def add_check(name: str, ok: bool, observed, expected=None, detail: str = "") -> None:
    checks.append({
        "check": name,
        "status": "PASS" if ok else "FAIL",
        "observed": observed,
        "expected": expected if expected is not None else "",
        "detail": detail,
    })

def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol

# Synthetic validation
syn = ROOT / "experiments" / "synthetic_spacing_validation"
back = pd.read_csv(syn / "comparison_backward_compatibility_summary.csv").iloc[0]
add_check("default compatibility n_features", int(back.n_features_compared) == 75, int(back.n_features_compared), 75)
add_check("default compatibility max abs diff", close(back.max_abs_difference, 0.0), float(back.max_abs_difference), 0.0)
iso = pd.read_csv(syn / "comparison_isotropic_sanity_summary.csv").iloc[0]
add_check("isotropic equivalence n_features", int(iso.n_features_compared) == 75, int(iso.n_features_compared), 75)
add_check("isotropic equivalence max abs diff", close(iso.max_abs_difference, 0.0), float(iso.max_abs_difference), 0.0)
act = pd.read_csv(syn / "comparison_voxelSpacing_effect_summary.csv")
expected_activation = {"glcm": 0.101046, "glrlm": 0.260970, "ngtdm": 0.335726, "gldm": 0.491003, "glszm": 0.400000}
for fam, expected in expected_activation.items():
    row = act.loc[act.family == fam].iloc[0]
    add_check(f"anisotropic activation median rel diff {fam}", close(row.median_relative_difference, expected, 5e-4), round(float(row.median_relative_difference), 6), expected)

# Rounding
round_geom = pd.read_csv(ROOT / "experiments" / "finite_volume_rounding_sensitivity" / "results" / "rounding_geometry_table.csv")
row25 = round_geom.loc[round_geom.spacing_label == "z2.5"].iloc[0]
row35 = round_geom.loc[round_geom.spacing_label == "z3.5"].iloc[0]
add_check("rounding z2.5 q_z", int(row25.q_z) == 2, int(row25.q_z), 2)
add_check("rounding z2.5 relative error", close(row25.relative_spacing_error_z, -0.2), float(row25.relative_spacing_error_z), -0.2)
add_check("rounding z3.5 q_z", int(row35.q_z) == 4, int(row35.q_z), 4)
add_check("rounding z3.5 relative error", close(row35.relative_spacing_error_z, 1/7, 1e-6), float(row35.relative_spacing_error_z), 1/7)
round_sum = pd.read_csv(ROOT / "experiments" / "finite_volume_rounding_sensitivity" / "results" / "rounding_summary_by_family.csv")
add_check("rounding max family median rel diff", close(round_sum.median_rel_diff.max(), 0.167, 5e-4), round(float(round_sum.median_rel_diff.max()), 6), 0.167)
add_check("rounding min Spearman", close(round_sum.spearman_corr.min(), 0.900, 5e-4), round(float(round_sum.spearman_corr.min()), 6), 0.900)

# Binning
binning = pd.read_csv(ROOT / "experiments" / "binning_sensitivity" / "results" / "binning_summary.csv")
add_check("binWidth set", set(binning.binWidth.astype(int)) == {5,10,25,50}, sorted(binning.binWidth.astype(int).tolist()), [5,10,25,50])
add_check("binning median rel diff min", close(binning.median_rel_diff_vs_native.min(), 0.145198, 5e-4), round(float(binning.median_rel_diff_vs_native.min()), 6), 0.145198)
add_check("binning median rel diff max", close(binning.median_rel_diff_vs_native.max(), 0.233222, 5e-4), round(float(binning.median_rel_diff_vs_native.max()), 6), 0.233222)
add_check("binning Spearman min", close(binning.spearman_corr_vs_native.min(), 0.972774, 5e-4), round(float(binning.spearman_corr_vs_native.min()), 6), 0.972774)
add_check("binning Spearman max", close(binning.spearman_corr_vs_native.max(), 0.981963, 5e-4), round(float(binning.spearman_corr_vs_native.max()), 6), 0.981963)

# Resampling comparison
res = pd.read_csv(ROOT / "experiments" / "resampling_baseline_comparison" / "results" / "feature_differences_vs_native.csv")
expected_changed = {
    "resample_nearest": 192,
    "resample_linear": 225,
    "resample_bspline": 225,
    "voxel_spacing_aware": 192,
}
for method, expected in expected_changed.items():
    subset = res[res.method == method]
    changed = int(subset.changed.sum())
    add_check(f"resampling changed count {method}", changed == expected, changed, expected)
summary = pd.read_csv(ROOT / "experiments" / "resampling_baseline_comparison" / "results" / "summary_by_family.csv")
med_of_meds = summary[summary.method.isin(expected_changed)].groupby("method").median_rel_diff_vs_native.median().to_dict()
expected_mom = {"resample_nearest":0.333, "resample_linear":0.304, "resample_bspline":0.302, "voxel_spacing_aware":0.307}
for method, expected in expected_mom.items():
    add_check(f"resampling family median-of-medians {method}", close(med_of_meds[method], expected, 5e-4), round(float(med_of_meds[method]), 6), expected)
min_spear = summary[summary.method.isin(expected_changed)].groupby("method").spearman_corr_with_native.min().to_dict()
expected_spear = {"resample_nearest":0.900, "voxel_spacing_aware":0.900, "resample_linear":0.800, "resample_bspline":0.800}
for method, expected in expected_spear.items():
    add_check(f"resampling min Spearman {method}", close(min_spear[method], expected, 5e-4), round(float(min_spear[method]), 6), expected)

# Profiling
prof = pd.read_csv(ROOT / "experiments" / "computational_profiling" / "results" / "runtime_memory_profile.csv")
summary_prof = pd.read_csv(ROOT / "experiments" / "computational_profiling" / "results" / "runtime_summary_by_expansion_factor.csv")
add_check("profiling tested input shapes", set(prof.input_shape) == {"16x64x64","32x96x96","48x128x128"}, sorted(set(prof.input_shape)), ["16x64x64","32x96x96","48x128x128"])
add_check("profiling expansion factors", set(prof.expansion_factor_R.astype(int)) == {1,2,3,5,7}, sorted(set(prof.expansion_factor_R.astype(int))), [1,2,3,5,7])
case = prof[(prof.size_label == "large") & (prof.expansion_factor_R == 7)].iloc[0]
add_check("profiling demanding expanded shape", case.expanded_shape == "48x128x896", case.expanded_shape, "48x128x896")
add_check("profiling demanding expanded ROI", int(case.number_of_mask_voxels_after) == 973196, int(case.number_of_mask_voxels_after), 973196)
add_check("profiling max runtime under 1.75s", float(prof.extraction_time_seconds.max()) < 1.75, round(float(prof.extraction_time_seconds.max()), 6), "<1.75")
row1 = summary_prof[summary_prof.expansion_factor_R == 1].iloc[0]
row7 = summary_prof[summary_prof.expansion_factor_R == 7].iloc[0]
add_check("profiling median memory R=1", close(row1.median_peak_memory_mb, 2.834412, 5e-4), round(float(row1.median_peak_memory_mb), 6), 2.834412)
add_check("profiling median memory R=7", close(row7.median_peak_memory_mb, 13.557629, 5e-4), round(float(row7.median_peak_memory_mb), 6), 13.557629)

report = pd.DataFrame(checks)
report.to_csv(OUT / "reproducibility_checks.csv", index=False)
status = "PASS" if (report.status == "PASS").all() else "FAIL"
lines = ["# Reproducibility report", "", f"Overall status: **{status}**", ""]
for _, row in report.iterrows():
    lines.append(f"- **{row.status}** {row.check}: observed `{row.observed}` expected `{row.expected}`")
(OUT / "REPRODUCIBILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT / "REPRODUCIBILITY_REPORT.md")
raise SystemExit(0 if status == "PASS" else 1)
