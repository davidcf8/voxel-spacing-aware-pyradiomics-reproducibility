import numpy

from radiomics import base, cMatrices, deprecated
from radiomics import spacing_utils


"""
  A Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in an image.
  A gray level dependency is defined as a the number of connected voxels within distance :math:`\delta` that are
  dependent on the center voxel.
  A neighbouring voxel with gray level :math:`j` is considered dependent on center voxel with gray level :math:`i`
  if :math:`|i-j|\le\alpha`. In a gray level dependence matrix :math:`\textbf{P}(i,j)` the :math:`(i,j)`\ :sup:`th`
  element describes the number of times a voxel with gray level :math:`i` with :math:`j` dependent voxels
  in its neighbourhood appears in image.

  As a two dimensional example, consider the following 5x5 image, with 5 discrete gray levels:

  .. math::
    \textbf{I} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}

  For :math:`\alpha=0` and :math:`\delta = 1`, the GLDM then becomes:

  .. math::
    \textbf{P} = \begin{bmatrix}
    0 & 1 & 2 & 1 \\
    1 & 2 & 3 & 0 \\
    1 & 4 & 4 & 0 \\
    1 & 2 & 0 & 0 \\
    3 & 0 & 0 & 0 \end{bmatrix}

  Let:

  - :math:`N_g` be the number of discrete intensity values in the image
  - :math:`N_d` be the number of discrete dependency sizes in the image
  - :math:`N_z` be the number of dependency zones in the image, which is equal to
    :math:`\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)}`
  - :math:`\textbf{P}(i,j)` be the dependence matrix
  - :math:`p(i,j)` be the normalized dependence matrix, defined as :math:`p(i,j) = \frac{\textbf{P}(i,j)}{N_z}`

  .. note::
    Because incomplete zones are allowed, every voxel in the ROI has a dependency zone. Therefore, :math:`N_z = N_p`,
    where :math:`N_p` is the number of voxels in the image.
    Due to the fact that :math:`Nz = N_p`, the Dependence Percentage and Gray Level Non-Uniformity Normalized (GLNN)
    have been removed. The first because it would always compute to 1, the latter because it is mathematically equal to
    first order - Uniformity (see :py:func:`~radiomics.firstorder.RadiomicsFirstOrder.getUniformityFeatureValue()`). For
    mathematical proofs, see :ref:`here <radiomics-excluded-gldm-label>`.

  The following class specific settings are possible:

  - distances [[1]]: List of integers. This specifies the distances between the center voxel and the neighbor, for which
    angles should be generated.
  - gldm_a [0]: float, :math:`\alpha` cutoff value for dependence. A neighbouring voxel with gray level :math:`j` is
    considered dependent on center voxel with gray level :math:`i` if :math:`|i-j|\le\alpha`

  References:

  - Sun C, Wee WG. Neighboring Gray Level Dependence Matrix for Texture Classification. Comput Vision,
    Graph Image Process. 1983;23:341-352
  """

class RadiomicsGLDM(base.RadiomicsFeaturesBase):
  """
  A Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in an
  image. A neighboring voxel with gray level j is considered dependent on a
  center voxel with gray level i if ``abs(i - j) <= gldm_a``; the matrix counts
  how often gray level i occurs with each integer dependence size.

  This implementation adds optional finite-volume voxel-spacing-aware support.

  When ``weightingNorm`` is set to ``voxelSpacing`` and spacing is anisotropic,
  the native binned image and mask are expanded to an isotropic physical lattice
  using zero-order hold. Standard GLDM dependence counting is then applied on
  that finite-volume representation. Dependence size remains an integer count;
  no fractional dependence axis or rounded weighted dependence is introduced.
  """

  def __init__(self, inputImage, inputMask, voxelSpacing=None, **kwargs):
      super(RadiomicsGLDM, self).__init__(inputImage, inputMask, **kwargs)
      self.voxelSpacing = voxelSpacing if voxelSpacing is not None else inputImage.GetSpacing()[::-1]
      self.gldm_a = kwargs.get('gldm_a', 0)
      self.P_gldm = None
      self.imageArray = self._applyBinning(self.imageArray)


  def _initCalculation(self, voxelCoordinates=None):
    self.P_gldm = self._calculateMatrix(voxelCoordinates)

  def _calculateMatrix(self, voxelCoordinates=None):
      self.logger.debug('Calculating GLDM matrix')

      Ng = self.coefficients['Ng']
      grayLevels = self.coefficients['grayLevels']
      distances = numpy.asarray(self.settings.get('distances', [1]), dtype=numpy.int32)
      force2D = int(self.settings.get('force2D', False))
      force2Ddim = int(self.settings.get('force2Ddimension', 0))
      kernelRadius = self.settings.get('kernelRadius', 1)
      weightingNorm = self.settings.get('weightingNorm', None)

      use_voxel_spacing = weightingNorm == 'voxelSpacing'
      spacing_array = None
      spacing_isotropic = True
      if use_voxel_spacing and self.voxelSpacing is not None:
          spacing_array = spacing_utils.validate_spacing(self.voxelSpacing)
          spacing_isotropic = spacing_utils.is_isotropic_spacing(spacing_array)

      if use_voxel_spacing and not spacing_isotropic:
          if voxelCoordinates is not None:
              raise NotImplementedError('Finite-volume voxelSpacing GLDM is not implemented for voxel-based extraction')

          repeat_factors = spacing_utils.compute_finite_volume_repeat_factors(
              spacing_array, bool(force2D), force2Ddim
          )
          finite_volume_geometry = spacing_utils.compute_finite_volume_geometry(
              spacing_array, repeat_factors, bool(force2D), force2Ddim, self.imageArray.shape
          )
          self.logger.debug(
              'Applying finite-volume zero-order-hold GLDM expansion with spacing %s, repeat factors %s, '
              'represented spacing %s, relative spacing error %s and expanded shape %s',
              spacing_array,
              repeat_factors,
              finite_volume_geometry['representedSpacing'],
              finite_volume_geometry['relativeSpacingError'],
              finite_volume_geometry['expandedShape']
          )
          if numpy.max(numpy.abs(finite_volume_geometry['relativeSpacingError'])) > 0.10:
              self.logger.warning(
                  'GLDM voxelSpacing finite-volume representation has relative spacing error %s',
                  finite_volume_geometry['relativeSpacingError']
              )
          if finite_volume_geometry['expansionFactor'] > 8:
              self.logger.warning(
                  'GLDM voxelSpacing finite-volume expansion factor %d expands image shape from %s to %s',
                  finite_volume_geometry['expansionFactor'],
                  self.imageArray.shape,
                  finite_volume_geometry['expandedShape']
              )

          expanded_image = spacing_utils.expand_array_zero_order_hold(self.imageArray, repeat_factors)
          expanded_mask = spacing_utils.expand_array_zero_order_hold(self.maskArray, repeat_factors)
          expanded_kernel_radius = max(1, int(kernelRadius)) if kernelRadius is not None else 1

          P_gldm = cMatrices.calculate_gldm(
              expanded_image,
              expanded_mask,
              distances,
              Ng,
              self.gldm_a,
              force2D,
              force2Ddim,
              expanded_kernel_radius,
              None
          )

      else:
          self.logger.debug('Calling standard GLDM backend')
          P_gldm = cMatrices.calculate_gldm(
              self.imageArray,
              self.maskArray,
              distances,
              Ng,
              self.gldm_a,
              force2D,
              force2Ddim,
              kernelRadius,
              voxelCoordinates
          )

      self.logger.debug('GLDM matrix shape: %s, dtype: %s', P_gldm.shape, P_gldm.dtype)

      if not isinstance(P_gldm, numpy.ndarray):
          raise TypeError('P_gldm is not an ndarray, got %s' % (type(P_gldm),))
      self.logger.debug('P_gldm.ndim: %d', P_gldm.ndim)

      NgVector = range(1, Ng + 1)
      emptyGrayLevels = numpy.array(list(set(NgVector) - set(grayLevels)), dtype=int)
      if len(emptyGrayLevels) > 0 and P_gldm.shape[1] > 0:
          self.logger.debug('Deleting %d empty gray levels', len(emptyGrayLevels))
          P_gldm = numpy.delete(P_gldm, emptyGrayLevels - 1, axis=1)

      self.logger.debug('P_gldm after empty gray-level deletion: shape=%s, ndim=%d', P_gldm.shape, P_gldm.ndim)
      if P_gldm.ndim != 3:
          raise RuntimeError('GLDM matrix must be 3-dimensional, got shape %s' % (P_gldm.shape,))

      self.logger.debug('GLDM shape before coefficient calculation: %s', P_gldm.shape)
      jvector = numpy.arange(1, P_gldm.shape[2] + 1, dtype='float64')

      pd = numpy.sum(P_gldm, axis=1)  # (Nv, Nd)
      pg = numpy.sum(P_gldm, axis=2)  # (Nv, Ng)

      empty_sizes = numpy.sum(pd, axis=0)
      if numpy.any(empty_sizes == 0):
          self.logger.debug('Deleting empty dependence sizes')
          P_gldm = numpy.delete(P_gldm, numpy.where(empty_sizes == 0), axis=2)
          jvector = numpy.delete(jvector, numpy.where(empty_sizes == 0))
          pd = numpy.delete(pd, numpy.where(empty_sizes == 0), axis=1)

      Nz = numpy.sum(pd, axis=1)
      Nz[Nz == 0] = numpy.spacing(1)

      self.coefficients['Nz'] = Nz
      self.coefficients['pd'] = pd
      self.coefficients['pg'] = pg
      self.coefficients['ivector'] = self.coefficients['grayLevels'].astype(float)
      self.coefficients['jvector'] = jvector

      return P_gldm




  def getSmallDependenceEmphasisFeatureValue(self):
    r"""
    **1. Small Dependence Emphasis (SDE)**

    .. math::
      SDE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}

    A measure of the distribution of small dependencies, with a greater value indicative
    of smaller dependence and less homogeneous textures.
    """
    pd = self.coefficients['pd']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']  # Nz = Np, see class docstring

    sde = numpy.sum(pd / (jvector[None, :] ** 2), 1) / Nz
    return sde

  def getLargeDependenceEmphasisFeatureValue(self):
    r"""
    **2. Large Dependence Emphasis (LDE)**

    .. math::
      LDE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)j^2}}{N_z}

    A measure of the distribution of large dependencies, with a greater value indicative
    of larger dependence and more homogeneous textures.
    """
    pd = self.coefficients['pd']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    lre = numpy.sum(pd * (jvector[None, :] ** 2), 1) / Nz
    return lre

  def getGrayLevelNonUniformityFeatureValue(self):
    r"""
    **3. Gray Level Non-Uniformity (GLN)**

    .. math::
      GLN = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_d}_{j=1}{\textbf{P}(i,j)}\right)^2}{N_z}

    Measures the similarity of gray-level intensity values in the image, where a lower GLN value
    correlates with a greater similarity in intensity values.
    """
    pg = self.coefficients['pg']
    Nz = self.coefficients['Nz']

    gln = numpy.sum(pg ** 2, 1) / Nz
    return gln

  @deprecated
  def getGrayLevelNonUniformityNormalizedFeatureValue(self):
    r"""
    **DEPRECATED. Gray Level Non-Uniformity Normalized (GLNN)**

    :math:`GLNN = \frac{\sum^{N_g}_{i=1}\left(\sum^{N_d}_{j=1}{\textbf{P}(i,j)}\right)^2}{\sum^{N_g}_{i=1}
    \sum^{N_d}_{j=1}{\textbf{P}(i,j)}^2}`

    .. warning::
      This feature has been deprecated, as it is mathematically equal to First Order - Uniformity
      :py:func:`~radiomics.firstorder.RadiomicsFirstOrder.getUniformityFeatureValue()`.
      See :ref:`here <radiomics-excluded-gldm-glnn-label>` for the proof. **Enabling this feature will result in the
      logging of a DeprecationWarning (does not interrupt extraction of other features), no value is calculated for
      this feature**
    """
    raise DeprecationWarning('GLDM - Gray Level Non-Uniformity Normalized is mathematically equal to First Order - '
                             'Uniformity, see http://pyradiomics.readthedocs.io/en/latest/removedfeatures.html for more'
                             'details')

  def getDependenceNonUniformityFeatureValue(self):
    r"""
    **4. Dependence Non-Uniformity (DN)**

    .. math::
      DN = \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z}

    Measures the similarity of dependence throughout the image, with a lower value indicating
    more homogeneity among dependencies in the image.
    """
    pd = self.coefficients['pd']
    Nz = self.coefficients['Nz']

    dn = numpy.sum(pd ** 2, 1) / Nz
    return dn

  def getDependenceNonUniformityNormalizedFeatureValue(self):
    r"""
    **5. Dependence Non-Uniformity Normalized (DNN)**

    .. math::
      DNN = \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z^2}

    Measures the similarity of dependence throughout the image, with a lower value indicating
    more homogeneity among dependencies in the image. This is the normalized version of the DLN formula.
    """
    pd = self.coefficients['pd']
    Nz = self.coefficients['Nz']

    dnn = numpy.sum(pd ** 2, 1) / Nz ** 2
    return dnn

  def getGrayLevelVarianceFeatureValue(self):
    r"""
    **6. Gray Level Variance (GLV)**

    .. math::
      GLV = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{p(i,j)(i - \mu)^2} \text{, where}
      \mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{ip(i,j)}

    Measures the variance in grey level in the image.
    """
    ivector = self.coefficients['ivector']
    Nz = self.coefficients['Nz']
    pg = self.coefficients['pg'] / Nz[:, None]  # divide by Nz to get the normalized matrix

    u_i = numpy.sum(pg * ivector[None, :], 1, keepdims=True)
    glv = numpy.sum(pg * (ivector[None, :] - u_i) ** 2, 1)
    return glv

  def getDependenceVarianceFeatureValue(self):
    r"""
    **7. Dependence Variance (DV)**

    .. math::
      DV = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{p(i,j)(j - \mu)^2} \text{, where}
      \mu = \displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{jp(i,j)}

    Measures the variance in dependence size in the image.
    """
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']
    pd = self.coefficients['pd'] / Nz[:, None]  # divide by Nz to get the normalized matrix

    u_j = numpy.sum(pd * jvector[None, :], 1, keepdims=True)
    dv = numpy.sum(pd * (jvector[None, :] - u_j) ** 2, 1)
    return dv

  def getDependenceEntropyFeatureValue(self):
    r"""
    **8. Dependence Entropy (DE)**

    .. math::
      Dependence Entropy = -\displaystyle\sum^{N_g}_{i=1}\displaystyle\sum^{N_d}_{j=1}{p(i,j)\log_{2}(p(i,j)+\epsilon)}
    """
    eps = numpy.spacing(1)
    Nz = self.coefficients['Nz']
    p_gldm = self.P_gldm / Nz[:, None, None]  # divide by Nz to get the normalized matrix

    return -numpy.sum(p_gldm * numpy.log2(p_gldm + eps), (1, 2))

  @deprecated
  def getDependencePercentageFeatureValue(self):
    r"""
    **DEPRECATED. Dependence Percentage**

    .. math::
      \textit{dependence percentage} = \frac{N_z}{N_p}

    .. warning::
      This feature has been deprecated, as it would always compute 1. See
      :ref:`here <radiomics-excluded-gldm-dependence-percentage-label>` for more details. **Enabling this feature will
      result in the logging of a DeprecationWarning (does not interrupt extraction of other features), no value is
      calculated for this features**
    """
    raise DeprecationWarning('GLDM - Dependence Percentage always computes 1, '
                             'see http://pyradiomics.readthedocs.io/en/latest/removedfeatures.html for more details')

  def getLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **9. Low Gray Level Emphasis (LGLE)**

    .. math::
      LGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}

    Measures the distribution of low gray-level values, with a higher value indicating a greater
    concentration of low gray-level values in the image.
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nz = self.coefficients['Nz']

    lgle = numpy.sum(pg / (ivector[None, :] ** 2), 1) / Nz
    return lgle

  def getHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **10. High Gray Level Emphasis (HGLE)**

    .. math::
      HGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2}}{N_z}

    Measures the distribution of the higher gray-level values, with a higher value indicating
    a greater concentration of high gray-level values in the image.
    """
    pg = self.coefficients['pg']
    ivector = self.coefficients['ivector']
    Nz = self.coefficients['Nz']

    hgle = numpy.sum(pg * (ivector[None, :] ** 2), 1) / Nz
    return hgle

  def getSmallDependenceLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **11. Small Dependence Low Gray Level Emphasis (SDLGLE)**

    .. math::
      SDLGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2j^2}}}{N_z}

    Measures the joint distribution of small dependence with lower gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    sdlgle = numpy.sum(self.P_gldm / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)), (1, 2)) / Nz
    return sdlgle

  def getSmallDependenceHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **12. Small Dependence High Gray Level Emphasis (SDHGLE)**

    .. math:
      SDHGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)i^2}{j^2}}}{N_z}

    Measures the joint distribution of small dependence with higher gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    sdhgle = numpy.sum(self.P_gldm * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2), (1, 2)) / Nz
    return sdhgle

  def getLargeDependenceLowGrayLevelEmphasisFeatureValue(self):
    r"""
    **13. Large Dependence Low Gray Level Emphasis (LDLGLE)**

    .. math::
      LDLGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)j^2}{i^2}}}{N_z}

    Measures the joint distribution of large dependence with lower gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    ldlgle = numpy.sum(self.P_gldm * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2), (1, 2)) / Nz
    return ldlgle

  def getLargeDependenceHighGrayLevelEmphasisFeatureValue(self):
    r"""
    **14. Large Dependence High Gray Level Emphasis (LDHGLE)**

    .. math::
      LDHGLE = \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2j^2}}{N_z}

    Measures the joint distribution of large dependence with higher gray-level values.
    """
    ivector = self.coefficients['ivector']
    jvector = self.coefficients['jvector']
    Nz = self.coefficients['Nz']

    ldhgle = numpy.sum(self.P_gldm * ((jvector[None, None, :] ** 2) * (ivector[None, :, None] ** 2)), (1, 2)) / Nz
    return ldhgle
