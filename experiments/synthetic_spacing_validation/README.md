# Synthetic Spacing Validation

This folder contains a minimal synthetic validation experiment for the voxel-spacing-aware PyRadiomics prototype.

The experiment rebuilds the original and modified PyRadiomics packages from source in two isolated virtual environments:

- `.venv_original`
- `.venv_modified`

It does not reuse pre-existing compiled `.so` files. Stale compiled artifacts are deleted before building.

## Run

```bash
cd <REPOSITORY_ROOT>/experiments/synthetic_spacing_validation
bash run_all.sh
```

The script uses:

```bash
<LOCAL_USER_HOME>/miniconda3/bin/python3.10
```

unless `PYTHON_BIN` is provided.

## What It Does

1. Creates buildable editable package copies under `build_sources/`.
2. Installs dependencies in both environments.
3. Compiles `_cmatrices` and `_cshape` natively for the active macOS architecture.
4. Verifies that `radiomics`, `_cmatrices`, and `_cshape` are imported from the correct editable package.
5. Generates synthetic anisotropic and isotropic NIfTI images.
6. Extracts Original-image texture features only:
   - GLCM
   - GLRLM
   - GLDM
   - NGTDM
   - GLSZM
7. Compares:
   - original anisotropic vs modified default anisotropic;
   - modified default anisotropic vs modified `weightingNorm=voxelSpacing`;
   - modified default isotropic vs modified `weightingNorm=voxelSpacing`.

## Main Outputs

- `synthetic_anisotropic_image.nii.gz`
- `synthetic_anisotropic_mask.nii.gz`
- `synthetic_isotropic_image.nii.gz`
- `synthetic_isotropic_mask.nii.gz`
- `features_original_anisotropic.csv`
- `features_modified_default_anisotropic.csv`
- `features_modified_voxelSpacing_anisotropic.csv`
- `features_modified_default_isotropic.csv`
- `features_modified_voxelSpacing_isotropic.csv`
- `comparison_backward_compatibility.csv`
- `comparison_voxelSpacing_effect.csv`
- `comparison_isotropic_sanity.csv`
- `synthetic_image_slices.pdf`
- `feature_difference_summary.pdf`
- `report_synthetic_spacing_validation.tex`

## Interpretation

This is a software behavior test, not a clinical validation experiment. It verifies whether spacing-aware execution paths activate under controlled anisotropic spacing and whether default behavior remains compatible with original PyRadiomics.
