# Additional Validation Report

## Library import paths

```text
 variant                                                                                                                          python                                       python_version machine                     platform                                                                                                                                      radiomics_file                                                                                                                                                           cmatrices_file                                                                                                                                                           cshape_file                                                                                                                         expected_source
original <REPOSITORY_ROOT>/experiments/synthetic_spacing_validation/.venv_original/bin/python 3.10.9 (main, Jan 11 2023, 09:18:18) [Clang 14.0.6 ]   arm64 macOS-14.6.1-arm64-arm-64bit <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/original/radiomics/__init__.py <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/original/radiomics/_cmatrices.cpython-310-darwin.so <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/original/radiomics/_cshape.cpython-310-darwin.so <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/original/radiomics
modified <REPOSITORY_ROOT>/experiments/synthetic_spacing_validation/.venv_modified/bin/python 3.10.9 (main, Jan 11 2023, 09:18:18) [Clang 14.0.6 ]   arm64 macOS-14.6.1-arm64-arm-64bit <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/modified/radiomics/__init__.py <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/modified/radiomics/_cmatrices.cpython-310-darwin.so <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/modified/radiomics/_cshape.cpython-310-darwin.so <REPOSITORY_ROOT>/experiments/additional_validation_runtime/build_sources/modified/radiomics
```


## Source libraries used

The validation runtime was built from the locally available corrected library pair:

- Original source: `<REPOSITORY_ROOT>/librerias/radiomics_original`
- Corrected modified source: `<REPOSITORY_ROOT>/librerias/radiomics_modificado_corregido`

No folder named `radiomics_modificado_corrected` was present at runtime. The master script now accepts both `radiomics_modificado_corrected` and `radiomics_modificado_corregido`, preferring the English `corrected` name if it exists.

## Resampling baseline comparison

Completed. Main CSV: `<REPOSITORY_ROOT>/experiments/resampling_baseline_comparison/results/summary_by_family.csv`

```text
spacing_label           method family  n_features  n_changed_vs_native  median_abs_diff_vs_native  median_rel_diff_vs_native  spearman_corr_with_native  spearman_corr_with_resample_linear  pearson_corr_with_resample_linear
      mild_z2  native_original   glcm          24                    0                   0.000000                   0.000000                   1.000000                            0.951304                           0.999133
      mild_z2  native_original   gldm          14                    0                   0.000000                   0.000000                   1.000000                            0.995604                           0.998182
      mild_z2  native_original  glrlm          16                    0                   0.000000                   0.000000                   1.000000                            1.000000                           0.999685
      mild_z2  native_original  glszm          16                    0                   0.000000                   0.000000                   1.000000                            0.994118                           0.999866
      mild_z2  native_original  ngtdm           5                    0                   0.000000                   0.000000                   1.000000                            0.900000                           0.950388
      mild_z2 resample_bspline   glcm          24                   24                   0.093062                   0.088803                   0.957391                            0.997391                           0.999839
      mild_z2 resample_bspline   gldm          14                   14                   2.343383                   0.351853                   0.986813                            0.995604                           0.999973
      mild_z2 resample_bspline  glrlm          16                   16                   0.519089                   0.309693                   1.000000                            1.000000                           0.999319
      mild_z2 resample_bspline  glszm          16                   16                   2.392076                   0.302420                   0.994118                            1.000000                           1.000000
      mild_z2 resample_bspline  ngtdm           5                    5                   0.019594                   0.310254                   0.900000                            1.000000                           0.988826
      mild_z2  resample_linear   glcm          24                   24                   0.101514                   0.100954                   0.951304                            1.000000                           1.000000
      mild_z2  resample_linear   gldm          14                   14                   1.793160                   0.384217                   0.995604                            1.000000                           1.000000
```

## Computational profiling

Completed. Main CSV: `<REPOSITORY_ROOT>/experiments/computational_profiling/results/runtime_summary_by_expansion_factor.csv`

```text
 expansion_factor_R  n_runs  median_time_seconds  max_time_seconds  median_peak_memory_mb     recommendation
                  1       3             1.205310          1.467667               2.834412 no special warning
                  2       3             1.316726          1.576890               4.325513 no special warning
                  3       3             1.270863          1.630910               6.054606 no special warning
                  5       3             1.360540          1.656242               9.537707 no special warning
                  7       3             1.360040          1.742194              13.557629 no special warning
```

## Finite-volume rounding sensitivity

Completed. Main CSV: `<REPOSITORY_ROOT>/experiments/finite_volume_rounding_sensitivity/results/rounding_geometry_table.csv`

```text
 spacing_z  spacing_y  spacing_x   h  q_z  q_y  q_x  expansion_factor_R  represented_spacing_z  represented_spacing_y  represented_spacing_x  absolute_spacing_error_z  relative_spacing_error_z spacing_label
       2.0        1.0        1.0 1.0    2    1    1                   2                    2.0                    1.0                    1.0                       0.0                  0.000000            z2
       2.5        1.0        1.0 1.0    2    1    1                   2                    2.0                    1.0                    1.0                      -0.5                 -0.200000          z2.5
       3.0        1.0        1.0 1.0    3    1    1                   3                    3.0                    1.0                    1.0                       0.0                  0.000000            z3
       3.5        1.0        1.0 1.0    4    1    1                   4                    4.0                    1.0                    1.0                       0.5                  0.142857          z3.5
       5.0        1.0        1.0 1.0    5    1    1                   5                    5.0                    1.0                    1.0                       0.0                  0.000000            z5
       5.5        1.0        1.0 1.0    6    1    1                   6                    6.0                    1.0                    1.0                       0.5                  0.090909          z5.5
```

## Binning sensitivity

Completed. Main CSV: `<REPOSITORY_ROOT>/experiments/binning_sensitivity/results/binning_summary.csv`

```text
 binWidth  n_valid_features  median_rel_diff_vs_native  median_rel_diff_vs_resample_linear  spearman_corr_vs_native  spearman_corr_vs_resample_linear
        5                75                   0.145198                            0.234671                 0.975135                          0.953030
       10                75                   0.202222                            0.212314                 0.981963                          0.954253
       25                75                   0.233222                            0.178836                 0.978321                          0.967881
       50                75                   0.205816                            0.249258                 0.972774                          0.974680
```

## Figures

- `experiments/resampling_baseline_comparison/figures/heatmap_method_similarity.pdf`
- `experiments/resampling_baseline_comparison/figures/family_relative_difference_barplot.pdf`
- `experiments/computational_profiling/figures/runtime_vs_expansion_factor.pdf`
- `experiments/computational_profiling/figures/memory_vs_expansion_factor.pdf`
- `experiments/finite_volume_rounding_sensitivity/figures/rounding_error_vs_spacing_ratio.pdf`
- `experiments/finite_volume_rounding_sensitivity/figures/feature_sensitivity_to_rounding.pdf`
- `experiments/binning_sensitivity/figures/binwidth_effect_summary.pdf`

## Suggested wording

The experiments support adding a focused software-validation paragraph: voxel-spacing-aware extraction behaves as a distinct interpolation-free alternative to standard isotropic resampling, with runtime governed by finite-volume expansion and non-integer ratios introducing a measurable rounding approximation. These results do not imply clinical superiority.
