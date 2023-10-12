import pytest

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

    # load data
    sigma = 25
    img = np.array(Image.open('tests/barbara.jpg'))
    x = torch.from_numpy(img + sigma*np.random.randn(512, 512)).reshape(1, 1, 512, 512).to(device)

    # decomposition
    coeffs = shearletSystem.decompose(x)

    # thresholding
    thresholdingFactor = 3
    weights = torch.ones_like(coeffs)
    for j in range(len(shearletSystem.RMS)):
        weights[:,:,:,:,j] = shearletSystem.RMS[j] * torch.ones(512, 512)
    zero_indices = torch.abs(coeffs) / (thresholdingFactor * weights * sigma) < 1
    coeffs[zero_indices] = 0

    # reconstruction
    x_hat = shearletSystem.reconstruct(coeffs)

    # compute PSNR
    mse = np.mean((img - x_hat.cpu().numpy()) ** 2)
    max_pixel = img.max()
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    print(f'PSNR: {psnr:.2f}')
