# Additional Validation Summary

This report summarizes the machine-readable outputs in the synthetic validation folders. Paths are repository-relative.

## Resampling Baseline Comparison

Main output:

```text
experiments/resampling_baseline_comparison/results/summary_by_family.csv
```

The comparison includes native extraction, nearest-neighbor isotropic resampling, linear interpolation, B-spline interpolation, and voxelSpacing extraction across three anisotropic synthetic phantoms.

## Computational Profiling

Main output:

```text
experiments/computational_profiling/results/runtime_summary_by_expansion_factor.csv
```

Profiling was performed on synthetic volumes with expansion factors R=1,2,3,5,7. Runtime and memory measurements are retained as environment-specific reference outputs.

## Finite-Volume Rounding Sensitivity

Main output:

```text
experiments/finite_volume_rounding_sensitivity/results/rounding_geometry_table.csv
```

This table records integer repeat factors, represented spacing, and relative representation error for the tested anisotropic spacing ratios.

## Discretization Sensitivity

Main output:

```text
experiments/binning_sensitivity/results/binning_summary.csv
```

The sensitivity analysis uses binWidth values 5, 10, 25, and 50.

## Figures

Reproducible figure files are stored in the corresponding `figures/` folders under each experiment directory. These files are retained as scientific outputs, not as manuscript source.
