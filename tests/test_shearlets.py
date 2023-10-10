import numpy as np

import torch

from PIL import Image

from pytorch_shearlets.shearlets import ShearletSystem

def test_call():
    """Validate the regular call."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    shearlet_system = ShearletSystem(512, 512, 2, device)

    # load data
    x = torch.from_numpy(np.array(Image.open('tests/barbara.jpg')).reshape(1, 1, 512, 512)).to(device)

    # decomposition
    coeffs = shearlet_system.decompose(x)

    # reconstruction
    x_hat = shearlet_system.reconstruct(coeffs)

    # compute error
    err = torch.mean(torch.square(x - x_hat)).item()
    assert err < 1e-6, f'Error too large: {err}'
