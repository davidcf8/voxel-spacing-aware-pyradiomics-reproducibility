# Voxel-Spacing-Aware PyRadiomics Reproducibility Repository

This repository contains the implementation snapshot and reproducibility material supporting the technical study "A Voxel-Spacing-Aware Extension of PyRadiomics for Anisotropic Texture Analysis".

## Scope

The repository focuses on software validation using synthetic phantoms. It does not contain clinical CT or MRI data, private cohort data, or raw patient images. The voxel-spacing-aware functionality is an opt-in methodological extension to PyRadiomics. Outputs from this mode should not be interpreted as conventional IBSI reference values.

## Implemented Approach

The included implementation snapshot adds spacing-aware behavior to selected texture families:

- GLCM: anisotropy-relative feature-level angular aggregation.
- NGTDM: anisotropy-relative weighted neighborhood mean.
- GLRLM, GLDM, and GLSZM: finite-volume zero-order-hold representation on an expanded lattice.

For the finite-volume paths, discretized gray levels are replicated into isotropic subcells rather than interpolated. This does not generate interpolated gray levels, but it changes the computational lattice and should not be described as preserving native voxel topology. The current spacing-aware implementation uses NumPy array order `(z, y, x)`, SimpleITK spacing order `(x, y, z)`, and maps spacing with `GetSpacing()[::-1]`. The SimpleITK direction matrix is not incorporated in the spacing-aware calculations.

All validation experiments are fully 3D with `force2D=False`.

## Repository Structure

```text
AUDIT/                                  Audit and provenance notes
experiments/                            Synthetic phantoms, scripts, figures, and frozen outputs
implementation/radiomics/               Frozen voxel-spacing-aware PyRadiomics implementation snapshot
parameter_examples/                     Example parameter files
results/                                Frozen machine-readable validation summaries
scripts/                                Supporting scripts, where present
tests/                                  Dedicated synthetic tests
reproduce_paper.py                      Lightweight published-output verification
rerun_experiments.py                    End-to-end local regeneration workflow
requirements.txt                        Lightweight verification dependencies
requirements-repro.txt                  End-to-end build and regeneration dependencies
```

No manuscript source or submission drafting files are included in this release copy.

## Reproduction Levels

### Lightweight Verification

```bash
python reproduce_paper.py
```

This checks the frozen machine-readable outputs and reported validation quantities. It writes local verification artifacts under `reproduced_results/`, which is ignored by Git.

Expected final line:

```text
Overall status: PASS
```

### End-to-End Reproduction

```bash
python -m pip install -r requirements-repro.txt
python rerun_experiments.py
```

The end-to-end workflow:

- reconstructs upstream PyRadiomics at commit `8ed579383b44806651c463d5e691f3b2b57522ab`;
- overlays the included `implementation/radiomics/` snapshot;
- builds the local compiled extension modules;
- verifies that `radiomics` and `_cmatrices` are loaded from the local reproduction build;
- reruns the experiments from the included synthetic NIfTI inputs;
- writes regenerated outputs under `results/rerun/`;
- compares regenerated outputs with the frozen published outputs;
- runs the lightweight verification workflow.

Expected final lines:

```text
End-to-end rerun: PASS
Published-result verification: PASS
Overall status: PASS
```

Useful options:

```bash
python rerun_experiments.py --clean
python rerun_experiments.py --skip-build
python rerun_experiments.py --upstream-source /path/to/pyradiomics
python rerun_experiments.py --run-upstream-tests
python rerun_experiments.py --verbose
```

Generated build and rerun outputs are local artifacts and are intentionally ignored by Git.

## Validation Summary

The frozen outputs support the following validation results:

- Clean upstream suite in the reconstructed release tree: 3563 / 3563 tests passed.
- Default compatibility: 75 texture features, 0 classified as changed, mean absolute difference `8.08e-16`, maximum absolute difference `5.68e-14`.
- Isotropic equivalence: exact equivalence across 75 texture features.
- Anisotropic activation medians: GLCM 10.1%, GLRLM 26.1%, NGTDM 33.6%, GLDM 49.1%, GLSZM 40.0%.
- GLDM and GLSZM changed-feature counts: 11/14 and 8/16, respectively.
- Finite-volume rounding sensitivity: `2.5:1:1` gives `q_z=2` and -20.0% through-plane representation error; `3.5:1:1` gives `q_z=4` and +14.3%.
- Rounding sensitivity summary: largest family-level median relative feature difference 0.167 for GLSZM; lowest Spearman correlation 0.900 for NGTDM.
- Discretization sensitivity: binWidth values 5, 10, 25, and 50; median relative difference versus native 14.5% to 23.3%; Spearman correlation 0.973 to 0.982.
- Resampling comparison: nearest-neighbor 192/225 changed, linear 225/225, B-spline 225/225, voxelSpacing 192/225.
- Resampling median-of-family-medians: nearest-neighbor 0.333, linear 0.304, B-spline 0.302, voxelSpacing 0.307.
- Computational profiling: synthetic native volumes 16x64x64, 32x96x96, and 48x128x128; ROI sizes 12,126, 50,328, and 139,028; expansion factors R=1,2,3,5,7. The most demanding validated geometry uses spacing `0.7:0.7:5.0`, `q=(1,1,7)`, expanded shape `48x128x896`, and expanded ROI 973,196.

Runtime and memory measurements are machine-dependent. The frozen profiling outputs remain included as reference results for the tested environment.

## Requirements

Use `requirements.txt` for the lightweight verification workflow. Use `requirements-repro.txt` for the end-to-end workflow, which rebuilds the local PyRadiomics extension. The validated development environment used Python 3.10.9 with NumPy 2.2.6, SimpleITK 2.5.5, and pytest 8.4.2. Other platforms may produce different runtime or memory measurements.

## Citation

Citation metadata are provided in `CITATION.cff`. A DOI will be added after archival of the `v1.0.0-paper` release.

## License

See `LICENSE`. This repository includes PyRadiomics-derived implementation material and preserves the licensing terms provided in the release copy.

## Limitations

- Finite-volume representation uses integer repeat factors.
- Non-integer spacing ratios are rounded, which introduces a quantified representation error.
- Expansion can increase memory requirements.
- The library-level safeguard limits the expansion factor R, not absolute expanded voxel count.
- The SimpleITK direction matrix is not incorporated in the current spacing-aware paths.
- Validation is synthetic software validation, not clinical validation.
- No independent IBSI benchmark is claimed for voxelSpacing mode.
