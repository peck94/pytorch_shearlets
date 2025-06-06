import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift

import numpy as np

from .filters import dfilters, modulate2, MakeONFilter
from .utils import SLprepareFilters2D, SLgetShearletIdxs2D, SLgetShearlets2D, upAndMergeBands, subsampleBands

class ShearletSystem:
    """
    Compute a 2D shearlet system.
    """
    def __init__(self, height, width, scales, fname='dmaxflat4', qmtype=None, device=torch.device('cpu')):
        levels = np.ceil(np.arange(1, scales + 1)/2).astype(int)

        h0, h1 = dfilters(fname, 'd')
        h0 /= np.sqrt(2)
        h1 /= np.sqrt(2)

        directionalFilter = modulate2(h0, 'c')

        if qmtype is not None:
            if qmtype.lower() =="meyer24":
                quadratureMirrorFilter = MakeONFilter("Meyer",24)
            elif qmtype.lower() == "meyer32":
                quadratureMirrorFilter = MakeONFilter("Meyer",32)
        else:
            quadratureMirrorFilter = np.array([0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
                                           0.276348304703363, 0.582566738241592, 0.276348304703363,
                                           -0.0517766952966369, -0.0263483047033631, 0.0104933261758408])
               
        self.preparedFilters = SLprepareFilters2D(height, width, scales, levels, directionalFilter, quadratureMirrorFilter)
        self.shearletIdxs = SLgetShearletIdxs2D(levels, 0)
        self.shearlets, self.RMS, self.dualFrameWeights = SLgetShearlets2D(self.preparedFilters, self.shearletIdxs)
        self.device = device

        self.shearlets = torch.from_numpy(self.shearlets.reshape(1, 1, *self.shearlets.shape)).to(device)
        self.dualFrameWeights = torch.from_numpy(self.dualFrameWeights).to(device)
        self.RMS = torch.from_numpy(self.RMS).to(device)

    def decompose(self, x):
        """
        Shearlet decomposition of 2D data.

        :param x: Input images. Tensor of shape [N, C, H, W].
        :return: Shearlet coefficients. Tensor of shape [N, C, H, W, M].
        """
        # get data in frequency domain
        x_freq = torch.unsqueeze(fftshift(fft2(ifftshift(x.to(self.device)))), -1)

        # compute shearlet coefficients at each scale
        coeffs = fftshift(
            ifft2(
                ifftshift(
                    x_freq * torch.conj(self.shearlets), dim=[0, 1, 2, 3]),
                    dim=[-3, -2]),
                dim=[0, 1, 2, 3])

        # return real coefficients
        return torch.real(coeffs)

    def reconstruct(self, coeffs):
        """
        2D reconstruction of shearlet coefficients.

        :param coeffs: Shearlet coefficients. Tensor of shape [N, C, H, W, M].
        :return: Reconstructed images. Tensor of shape [N, C, H, W].
        """
        # compute image values
        s = fftshift(
            fft2(
                ifftshift(coeffs.to(self.device), dim=[0, 1, 2, 3]),
                dim=[-3, -2]),
            dim=[0, 1, 2, 3]) * self.shearlets
        x = fftshift(ifft2(ifftshift((torch.div(torch.sum(s, dim=-1), self.dualFrameWeights)))))

        # return real values
        return torch.real(x)

    def decompose_subsample(self, x, decimFactors):
        """
        2D Shearlet decomposition followed by subsampling

        Input
        -----
            x : Input images, tensor of shape [N, C, H, W]
            decimFactors : reverse(decimFactors)[i] gives the decimation factor required for scale i
        
        Output
        -----
            dictionary R where R[d] gives the tensor [N, C, Hd, Wd, Md] of bands that are decimated by factor d
        """
        decomposed = self.decompose(x)
        bandsDict = subsampleBands(decomposed,self.shearletIdxs[:,1],decimFactors)

        return bandsDict


    def upsample_reconstruct(self, x, decimFactors):
        """
        2D reconstruction of downsampled shearlet bands

        Input
        -----
            x : dictionary of bands where x[d] gives the tensor [N, C, Hd, Wd, Md] of the Md bands decimated by factor d
            
            decimFactors : reverse(decimFactors)[i] gives the decimation factor required for scale i

        Output
        -----
            Reconstructed images. Tensor of shape [N, C, H, W]
        
        Warning
        -----
            Might need to clamp to correct range after reconstruction

        """

        merged_bands = upAndMergeBands(x,self.shearletIdxs[:,1],decimFactors)
        reconstructed = self.reconstruct(merged_bands)
    
        return reconstructed