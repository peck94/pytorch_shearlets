import pytest

import matplotlib.pyplot as plt

import numpy as np

import torch

from PIL import Image

from pytorch_shearlets.shearlets import ShearletSystem
from pytorch_shearlets.utils import SLcomputePSNR

@pytest.fixture(scope='module')
def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@pytest.fixture(scope='module')
def shearletSystem(device):
    return ShearletSystem(512, 512, 2, device=device)

def test_inverse(shearletSystem, device):
    """Validate the inverse."""

    # create sample
    X = torch.randn(1, 1, 512, 512).to(device)

    # decomposition
    coeffs = shearletSystem.decompose(X)

    # reconstruction
    Xrec = shearletSystem.reconstruct(coeffs)
    assert Xrec.shape == X.shape

    assert torch.linalg.norm(X - Xrec) < 1e-5 * torch.linalg.norm(X)

def test_call(shearletSystem, device):
    """Validate the regular call."""

    def compute_psnr(img, x_hat):
        mse = np.mean((img.cpu().numpy() - x_hat.cpu().numpy()) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse))
        return psnr

    # load data
    sigma = 25
    img = torch.from_numpy(np.array(Image.open('tests/barbara.jpg')).reshape(1, 1, 512, 512)).to(device)

    # compute mean PSNR
    psnrs = []
    for _ in range(10):
        # create noise
        noise = sigma * torch.randn(512, 512).to(device)

        # decomposition
        coeffs = shearletSystem.decompose(img + noise)

        # thresholding
        thresholdingFactor = 3
        weights = torch.ones_like(coeffs)
        for j in range(len(shearletSystem.RMS)):
            weights[:,:,:,:,j] = shearletSystem.RMS[j] * torch.ones(512, 512)
        zero_indices = (torch.abs(coeffs) / (thresholdingFactor * weights * sigma) < 1)
        new_coeffs = torch.where(zero_indices, torch.zeros_like(coeffs), coeffs)

        # reconstruction
        x_hat = shearletSystem.reconstruct(new_coeffs)

        # compute PSNR
        psnr1 = compute_psnr(img, img + noise)
        psnr2 = compute_psnr(img, x_hat)
        psnrs.append(psnr2)

        assert psnr1 < psnr2, f'PSNR of noisy image ({psnr1:.2f}) must be lower than PSNR of restored image ({psnr2:.2f})'

    # compute result
    mean_psnr = np.mean(psnrs)
    std_psnr = np.std(psnrs)
    print(f'PSNR: {mean_psnr:.2f} dB ({std_psnr:.2f})')

    assert mean_psnr > 25, f'PSNR is too low: {mean_psnr:.2f} < 25 dB'
