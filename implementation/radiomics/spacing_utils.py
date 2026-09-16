import numpy


def validate_spacing(spacing):
  """
  Return spacing as a 1D float array and fail clearly for invalid values.

  The voxel-spacing-aware code paths rely on physical voxel dimensions. Zero,
  negative, NaN or infinite values cannot be interpreted safely.
  """
  spacing = numpy.asarray(spacing, dtype=float).ravel()
  if spacing.size == 0:
    raise ValueError('voxelSpacing requires at least one spacing component')
  if not numpy.all(numpy.isfinite(spacing)):
    raise ValueError('voxelSpacing requires finite spacing components')
  if numpy.any(spacing <= 0):
    raise ValueError('voxelSpacing requires strictly positive spacing components')
  return spacing


def is_isotropic_spacing(spacing, rtol=1e-8, atol=1e-8):
  """
  Return True if all spacing components are equal within numerical tolerance.

  Spacing-aware feature classes use this guard to preserve exact default
  PyRadiomics behaviour in isotropic images.
  """
  spacing = validate_spacing(spacing)
  if spacing.size == 0:
    return True
  return bool(numpy.allclose(spacing, spacing[0], rtol=rtol, atol=atol))


def compute_relative_offset_distance(offset, spacing, epsilon=1e-12):
  """
  Compute the relative physical stretch of a discrete offset.

  Let ``offset`` be a native index-space displacement and ``spacing`` the voxel
  spacing in the same axis order. This function returns

      ||offset * spacing||_2 / ||offset||_2

  rather than the absolute physical distance. The ratio isolates anisotropy:
  for isotropic spacing it is constant across directions, whereas anisotropic
  spacing changes the contribution of offsets that traverse thick-slice axes.
  """
  offset = numpy.asarray(offset, dtype=float).ravel()
  spacing = validate_spacing(spacing)
  index_distance = float(numpy.sqrt(numpy.sum(offset ** 2)))
  if index_distance <= epsilon:
    return 0.0
  physical_distance = float(numpy.sqrt(numpy.sum((offset * spacing) ** 2)))
  return physical_distance / index_distance


def compute_finite_volume_repeat_factors(spacing, force2D=False, force2Ddimension=0, rtol=1e-3, atol=1e-3,
                                         maximumExpansionFactor=1000):
  """
  Compute integer repeat factors for zero-order-hold finite-volume expansion.

  The spacing sequence must be in the same axis order as the numpy image array.
  For PyRadiomics this usually means SimpleITK spacing reversed from
  ``(x, y, z)`` into array order ``(z, y, x)``.

  If force2D is enabled, the excluded dimension is not expanded. Non-integer
  spacing ratios are rounded to the nearest integer repeat factor.
  """
  spacing = validate_spacing(spacing)
  positive_spacing = spacing[spacing > 0]

  effective_spacing = spacing.copy()
  if force2D and 0 <= int(force2Ddimension) < effective_spacing.size:
    effective_spacing[int(force2Ddimension)] = numpy.min(positive_spacing)

  base_spacing = float(numpy.min(effective_spacing[effective_spacing > 0]))
  ratios = effective_spacing / base_spacing
  repeat_factors = numpy.maximum(1, numpy.rint(ratios).astype(numpy.int32))

  if not numpy.allclose(ratios, repeat_factors, rtol=rtol, atol=atol):
    # Feature classes own logging, so this helper remains side-effect free.
    pass

  expansionFactor = int(numpy.prod(repeat_factors))
  if maximumExpansionFactor is not None and expansionFactor > int(maximumExpansionFactor):
    raise ValueError(
      'voxelSpacing finite-volume expansion factor %d exceeds maximumExpansionFactor %d'
      % (expansionFactor, int(maximumExpansionFactor))
    )

  return repeat_factors


def compute_finite_volume_geometry(spacing, repeat_factors, force2D=False, force2Ddimension=0, input_shape=None):
  """
  Return finite-volume representation metadata for reporting and warnings.

  ``represented_spacing`` is the spacing implied by the integer repeat factors,
  and ``relative_spacing_error`` quantifies the approximation introduced when
  non-integer spacing ratios are rounded.
  """
  spacing = validate_spacing(spacing)
  repeat_factors = numpy.asarray(repeat_factors, dtype=numpy.int32).ravel()
  if repeat_factors.size != spacing.size:
    raise ValueError('repeat factors and spacing must have the same length')
  if numpy.any(repeat_factors < 1):
    raise ValueError('repeat factors must be positive integers')

  effective_spacing = spacing.copy()
  if force2D and 0 <= int(force2Ddimension) < effective_spacing.size:
    effective_spacing[int(force2Ddimension)] = numpy.min(spacing)

  base_spacing = float(numpy.min(effective_spacing[effective_spacing > 0]))
  represented_spacing = repeat_factors.astype(float) * base_spacing
  relative_spacing_error = (represented_spacing - effective_spacing) / effective_spacing
  expansion_factor = int(numpy.prod(repeat_factors))

  expanded_shape = None
  if input_shape is not None:
    input_shape = tuple(int(i) for i in input_shape)
    expanded_shape = tuple(int(i * r) for i, r in zip(input_shape, repeat_factors))

  return {
    'baseSpacing': base_spacing,
    'repeatFactors': repeat_factors,
    'representedSpacing': represented_spacing,
    'relativeSpacingError': relative_spacing_error,
    'expansionFactor': expansion_factor,
    'expandedShape': expanded_shape,
  }


def expand_array_zero_order_hold(array, repeat_factors):
  """
  Expand an image or mask by repeating native voxels along each axis.

  This is a signal-preserving finite-volume representation. Native intensity
  values are replicated exactly and no interpolated gray levels are introduced.
  """
  expanded = array
  repeat_factors = numpy.asarray(repeat_factors, dtype=numpy.int32).ravel()
  if numpy.any(repeat_factors < 1):
    raise ValueError('repeat factors must be positive integers')
  for axis, factor in enumerate(repeat_factors):
    if factor > 1:
      expanded = numpy.repeat(expanded, int(factor), axis=axis)
  return expanded
