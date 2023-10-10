import torch

import numpy as np

from .filters import dfilters, modulate2
from .utils import SLprepareFilters2D, SLgetShearletIdxs2D, SLgetShearlets2D

class ShearletSystem:
    """
    Compute a 2D shearlet system.
    """
    def __init__(self, height, width, scales, device=torch.device('cpu')):
        levels = np.ceil(np.arange(1, scales + 1)/2).astype(int)

        h0, h1 = dfilters('dmaxflat4', 'd')
        h0 /= np.sqrt(2)
        h1 /= np.sqrt(2)

        directionalFilter = modulate2(h0, 'c')

        quadratureMirrorFilter = np.array([0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
                                           0.276348304703363, 0.582566738241592, 0.276348304703363,
                                           -0.0517766952966369, -0.0263483047033631, 0.0104933261758408])
        
        self.preparedFilters = SLprepareFilters2D(height, width, scales, levels, directionalFilter, quadratureMirrorFilter)
        self.shearletIdxs = SLgetShearletIdxs2D(levels, 0)
        self.shearlets, self.RMS, self.dualFrameWeights = SLgetShearlets2D(self.preparedFilters, self.shearletIdxs)
        self.device = device

    def decompose(self, x):
        pass

    def reconstruct(self, coeffs):
        pass
