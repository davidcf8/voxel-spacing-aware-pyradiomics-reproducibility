import numpy
from radiomics import base, cMatrices, spacing_utils


class RadiomicsNGTDM(base.RadiomicsFeaturesBase):
    """
    Neighbouring Gray Tone Difference Matrix (NGTDM).

    Standard extraction uses the PyRadiomics neighborhood definition and
    unweighted local mean. With ``weightingNorm='voxelSpacing'`` and
    anisotropic spacing, the same valid neighbors are used, but their local
    mean contribution is weighted by inverse relative physical offset distance.
    """

    def __init__(self, inputImage, inputMask, voxelSpacing=None, **kwargs):
        super(RadiomicsNGTDM, self).__init__(inputImage, inputMask, **kwargs)
        self.voxelSpacing = voxelSpacing if voxelSpacing is not None else inputImage.GetSpacing()[::-1]
        self.P_ngtdm = None
        self.imageArray = self._applyBinning(self.imageArray)

    def _initCalculation(self, voxelCoordinates=None):
        self.P_ngtdm = self._calculateMatrix(voxelCoordinates)
        self._calculateCoefficients()

    def _calculateMatrix(self, voxelCoordinates=None):
      weightingNorm = self.settings.get('weightingNorm', None)
      use_voxel_spacing = weightingNorm == 'voxelSpacing'

      distances_array = numpy.array(self.settings.get('distances', [1]), dtype=float)
      Ng = self.coefficients['Ng']
      force2D = self.settings.get('force2D', False)
      force2Ddim = self.settings.get('force2Ddimension', 0)
      kernelRadius = self.settings.get('kernelRadius', 1) or 1

      spacing_array = None
      spacing_isotropic = True
      if use_voxel_spacing and self.voxelSpacing is not None:
          spacing_array = spacing_utils.validate_spacing(self.voxelSpacing)
          spacing_isotropic = spacing_utils.is_isotropic_spacing(spacing_array)

      if use_voxel_spacing and not spacing_isotropic:
          self.logger.debug('Calling spacing-aware NGTDM backend')
          P_ngtdm = cMatrices.calculate_ngtdm_spacing(
              self.imageArray,
              self.maskArray,
              distances_array,
              Ng,
              force2D,
              force2Ddim,
              kernelRadius,
              voxelCoordinates,
              spacing_array
          )
      else:
          self.logger.debug('Calling standard NGTDM backend')
          P_ngtdm = cMatrices.calculate_ngtdm(
              self.imageArray,
              self.maskArray,
              distances_array,
              Ng,
              force2D,
              force2Ddim,
              kernelRadius,
              voxelCoordinates
          )

      # Delete empty gray levels.
      emptyGrayLevels = numpy.where(numpy.sum(P_ngtdm[:, :, 0], axis=0) == 0)[0]
      if emptyGrayLevels.size > 0:
          P_ngtdm = numpy.delete(P_ngtdm, emptyGrayLevels, axis=1)

      return P_ngtdm

    def _calculateCoefficients(self):
      eps = numpy.spacing(1)
      Nvp = numpy.sum(self.P_ngtdm[:, :, 0], 1)
      Nvp[Nvp == 0] = eps

      self.coefficients['Nvp'] = Nvp
      self.coefficients['p_i'] = self.P_ngtdm[:, :, 0] / Nvp[:, None]
      self.coefficients['s_i'] = self.P_ngtdm[:, :, 1]
      self.coefficients['ivector'] = self.P_ngtdm[:, :, 2]
      self.coefficients['Ngp'] = numpy.sum(self.P_ngtdm[:, :, 0] > 0, 1)
      self.coefficients['p_zero'] = numpy.where(self.coefficients['p_i'] == 0)


    def getCoarsenessFeatureValue(self):
        r"""
        Calculate and return the coarseness.

        :math:`Coarseness = \frac{1}{\sum^{N_g}_{i=1}{p_{i}s_{i}}}`

        Coarseness measures the average difference between a center voxel and
        its neighborhood. A higher value indicates a lower spatial change rate
        and a locally more uniform texture.
        """
        p_i = self.coefficients['p_i']
        s_i = self.coefficients['s_i']
        sum_coarse = numpy.sum(p_i * s_i, 1)
        sum_coarse[sum_coarse != 0] = 1 / sum_coarse[sum_coarse != 0]
        sum_coarse[sum_coarse == 0] = 1e6
        return sum_coarse

    def getContrastFeatureValue(self):
        r"""
        Calculate and return the contrast.

        :math:`Contrast = \left(\frac{1}{N_{g,p}(N_{g,p}-1)}
        \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_i p_j (i-j)^2}\right)
        \left(\frac{1}{N_{v,p}}\sum^{N_g}_{i=1}{s_i}\right)`

        Contrast measures spatial intensity change and also depends on the
        overall gray-level dynamic range.
        """
        Ngp = self.coefficients['Ngp']
        Nvp = self.coefficients['Nvp']
        p_i = self.coefficients['p_i']
        s_i = self.coefficients['s_i']
        i = self.coefficients['ivector']

        div = Ngp * (Ngp - 1)
        contrast = (numpy.sum(p_i[:, :, None] * p_i[:, None, :] * (i[:, :, None] - i[:, None, :]) ** 2, (1, 2)) *
                    numpy.sum(s_i, 1) / Nvp)
        contrast[div != 0] /= div[div != 0]
        contrast[div == 0] = 0
        return contrast

    def getBusynessFeatureValue(self):
        r"""
        Calculate and return the busyness.

        :math:`Busyness = \frac{\sum^{N_g}_{i=1}{p_i s_i}}
        {\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{|i p_i - j p_j|}}`

        Busyness measures the change from a voxel to its neighbors. A high
        value indicates rapid intensity changes in the local neighborhood.
        """
        p_i = self.coefficients['p_i']
        s_i = self.coefficients['s_i']
        i = self.coefficients['ivector']
        p_zero = self.coefficients['p_zero']

        i_pi = i * p_i
        absdiff = numpy.abs(i_pi[:, :, None] - i_pi[:, None, :])
        absdiff[p_zero[0], :, p_zero[1]] = 0
        absdiff[p_zero[0], p_zero[1], :] = 0
        absdiff = numpy.sum(absdiff, (1, 2))

        busyness = numpy.sum(p_i * s_i, 1)
        busyness[absdiff != 0] /= absdiff[absdiff != 0]
        busyness[absdiff == 0] = 0
        return busyness

    def getComplexityFeatureValue(self):
        r"""
        Calculate and return the complexity.

        :math:`Complexity = \frac{1}{N_{v,p}}\sum^{N_g}_{i=1}
        \sum^{N_g}_{j=1}{|i-j|\frac{p_i s_i + p_j s_j}{p_i + p_j}}`

        Complexity is high for non-uniform images with many rapid gray-level
        intensity changes.
        """
        Nvp = self.coefficients['Nvp']
        p_i = self.coefficients['p_i']
        s_i = self.coefficients['s_i']
        i = self.coefficients['ivector']
        p_zero = self.coefficients['p_zero']

        pi_si = p_i * s_i
        numerator = pi_si[:, :, None] + pi_si[:, None, :]
        numerator[p_zero[0], :, p_zero[1]] = 0
        numerator[p_zero[0], p_zero[1], :] = 0

        divisor = p_i[:, :, None] + p_i[:, None, :]
        divisor[divisor == 0] = 1

        complexity = numpy.sum(numpy.abs(i[:, :, None] - i[:, None, :]) * numerator / divisor, (1, 2)) / Nvp
        return complexity

    def getStrengthFeatureValue(self):
        r"""
        Calculate and return the strength.

        :math:`Strength = \frac{\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}
        {(p_i + p_j)(i-j)^2}}{\sum^{N_g}_{i=1}{s_i}}`

        Strength is high when primitives are easily defined and visible, with
        slow intensity change but larger coarse gray-level differences.
        """
        p_i = self.coefficients['p_i']
        s_i = self.coefficients['s_i']
        i = self.coefficients['ivector']
        p_zero = self.coefficients['p_zero']

        sum_s_i = numpy.sum(s_i, 1)
        strength = (p_i[:, :, None] + p_i[:, None, :]) * (i[:, :, None] - i[:, None, :]) ** 2
        strength[p_zero[0], :, p_zero[1]] = 0
        strength[p_zero[0], p_zero[1], :] = 0

        strength = numpy.sum(strength, (1, 2))
        strength[sum_s_i != 0] /= sum_s_i[sum_s_i != 0]
        strength[sum_s_i == 0] = 0
        return strength
