import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift

import numpy as np

from .filters import dfilters, modulate2
from .utils import SLprepareFilters2D, SLgetShearletIdxs2D, SLgetShearlets2D

class ShearletSystem:
    """
    Compute a 2D shearlet system.
    """
    def __init__(self, height, width, scales, fname='dmaxflat4', device=torch.device('cpu')):
        levels = np.ceil(np.arange(1, scales + 1)/2).astype(int)

        h0, h1 = dfilters(fname, 'd')
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

        self.shearlets = torch.from_numpy(self.shearlets.reshape(1, 1, *self.shearlets.shape)).to(device)
        self.dualFrameWeights = torch.from_numpy(self.dualFrameWeights).to(device)

    def decompose(self, x):
        """
        Shearlet decomposition of 2D data.

        :param x: Input images. Tensor of shape [N, C, H, W].
        :return: Shearlet coefficients. Tensor of shape [N, C, H, W, M].
        """

        # initialize coefficient array
        coeffs = torch.zeros([x.shape[0], x.shape[1], *self.shearlets.shape[2:]], dtype=torch.cfloat).to(self.device)

        # get data in frequency domain
        x_freq = fftshift(fft2(ifftshift(x)))

        # compute shearlet coefficients at each scale
        for j in range(self.shearletIdxs.shape[0]):
            coeffs[:,:,:,:,j] = fftshift(ifft2(ifftshift(x_freq * torch.conj(self.shearlets[:,:,:,:,j]))))

        # return real coefficients
        return torch.real(coeffs).to(self.device)

    def reconstruct(self, coeffs):
        """
        2D reconstruction of shearlet coefficients.

        :param coeffs: Shearlet coefficients. Tensor of shape [N, C, H, W, M].
        :return: Reconstructed images. Tensor of shape [N, C, H, W].
        """
        # initialize image array
        x = torch.zeros(coeffs.shape[:4], dtype=torch.cfloat).to(self.device)

        # compute image values
        for j in range(self.shearletIdxs.shape[0]):
            x = x + fftshift(fft2(ifftshift(coeffs[:,:,:,:,j]))) * self.shearlets[:,:,:,:,j]

        x = fftshift(ifft2(ifftshift((torch.div(x, self.dualFrameWeights)))))

        # return real values
        return torch.real(x)
