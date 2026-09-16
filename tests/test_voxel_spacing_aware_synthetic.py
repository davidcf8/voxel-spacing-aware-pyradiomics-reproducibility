import numpy
import SimpleITK as sitk

from radiomics import featureextractor


def _make_synthetic_image(spacing):
  rng = numpy.random.default_rng(12345)
  z, y, x = numpy.indices((12, 24, 24))
  mask = ((x >= 5) & (x <= 18) & (y >= 5) & (y <= 18) & (z >= 3) & (z <= 9)).astype(numpy.uint8)
  image = numpy.zeros_like(x, dtype=numpy.float32)
  texture = 50 + 8 * x + 20 * ((z % 2) == 0) + rng.normal(0, 1, size=x.shape)
  image[mask > 0] = texture[mask > 0]

  sitk_image = sitk.GetImageFromArray(image)
  sitk_mask = sitk.GetImageFromArray(mask)
  sitk_image.SetSpacing(tuple(spacing))
  sitk_mask.SetSpacing(tuple(spacing))
  return sitk_image, sitk_mask


def _extract(image, mask, weighting_norm=None):
  settings = {
    "binWidth": 25,
    "label": 1,
    "force2D": False,
    "resampledPixelSpacing": None,
  }
  if weighting_norm is not None:
    settings["weightingNorm"] = weighting_norm

  extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
  extractor.disableAllImageTypes()
  extractor.enableImageTypeByName("Original")
  extractor.disableAllFeatures()
  for family in ("glcm", "glrlm", "ngtdm", "gldm", "glszm"):
    extractor.enableFeatureClassByName(family)

  result = extractor.execute(image, mask)
  return {
    key: float(value)
    for key, value in result.items()
    if key.startswith("original_")
  }


def _family_values(features, family):
  prefix = "original_%s_" % family
  return numpy.array([value for key, value in sorted(features.items()) if key.startswith(prefix)], dtype=float)


def test_voxel_spacing_isotropic_matches_default():
  image, mask = _make_synthetic_image((1.0, 1.0, 1.0))
  default = _extract(image, mask)
  spacing_aware = _extract(image, mask, "voxelSpacing")

  assert default.keys() == spacing_aware.keys()
  for key in default:
    assert numpy.isclose(default[key], spacing_aware[key], rtol=1e-6, atol=1e-8), key


def test_voxel_spacing_anisotropic_activates_texture_families():
  image, mask = _make_synthetic_image((1.0, 1.0, 5.0))
  default = _extract(image, mask)
  spacing_aware = _extract(image, mask, "voxelSpacing")

  for family in ("glcm", "glrlm", "ngtdm", "gldm", "glszm"):
    before = _family_values(default, family)
    after = _family_values(spacing_aware, family)
    assert before.size > 0
    assert numpy.any(numpy.abs(before - after) > 1e-5), family
