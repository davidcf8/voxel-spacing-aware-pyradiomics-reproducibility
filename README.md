# A Voxel-Spacing-Aware Extension of PyRadiomics for Anisotropic Texture Analysis

This repository accompanies the technical implementation manuscript. It reproduces the synthetic phantom and software-validation analyses from already generated local outputs, not from clinical CT/MRI cohorts.

The repository is intentionally separate from the clinical-study material and from the upstream PyRadiomics pull-request preparation. It contains synthetic phantoms, validation result tables, figures, manuscript source, and a clean VS-only PyRadiomics implementation snapshot for inspection.

## What Is Included

- Synthetic spacing validation: backward compatibility, isotropic equivalence, and anisotropic activation.
- Finite-volume rounding sensitivity: integer repeat-factor geometry and feature sensitivity.
- Discretization/binWidth sensitivity.
- Native versus nearest-neighbor, linear, B-spline and voxelSpacing comparison.
- Computational profiling on synthetic volumes.
- Manuscript LaTeX/PDF and figures.
- Clean `implementation/radiomics` snapshot without unrelated experimental preprocessing branches.

## What Is Not Included

- No clinical CT/MRI patient cohort data.
- No private external cohort data.
- No raw patient images.
- No unrelated experimental preprocessing branch material.
- No claim of formal IBSI compliance for voxelSpacing output.
- No claim of physical-ground-truth phantom validation or clinical superiority.

## Quick Reproduction

Create an environment with the packages in `requirements.txt`, then run:

```bash
python reproduce_paper.py
```

This validates the manuscript key numerical results from the included CSV/JSON outputs and writes:

```text
reproduced_results/REPRODUCIBILITY_REPORT.md
reproduced_results/reproducibility_checks.csv
```

## Optional Full Regeneration

The included CSV/JSON/NIfTI files are sufficient for audit reproduction. Full feature re-extraction requires a compiled local PyRadiomics environment. Use the scripts under `experiments/` as reference entry points, but they are not needed for the default offline check.

## Repository Layout

```text
experiments/
  synthetic_spacing_validation/
  finite_volume_rounding_sensitivity/
  binning_sensitivity/
  resampling_baseline_comparison/
  computational_profiling/
implementation/radiomics/
manuscript/
parameter_examples/
results/
tests/
AUDIT/
reproduce_paper.py
```

## Expected Key Results

- Default compatibility: 75 features, max absolute difference 0.
- Isotropic equivalence: 75 features, max absolute difference 0.
- Anisotropic activation medians: GLCM 10.1%, GLRLM 26.1%, NGTDM 33.6%, GLDM 49.1%, GLSZM 40.0%.
- Rounding: 2.5:1:1 gives `q_z=2`, error -20.0%; 3.5:1:1 gives `q_z=4`, error +14.3%.
- Binning: binWidth 5, 10, 25, 50; median relative difference versus native 14.5% to 23.3%.
- Resampling comparison: nearest 192/225 changed, linear 225/225, B-spline 225/225, voxelSpacing 192/225.
- Profiling: volumes 16x64x64, 32x96x96, 48x128x128; expansion factors R=1,2,3,5,7; maximum runtime below 1.75 s in the included run.

## Citation

A Voxel-Spacing-Aware Extension of PyRadiomics for Anisotropic Texture Analysis. arXiv identifier/DOI pending.
