import numpy as np

from pytorch_shearlets.shearlets import ShearletSystem

def test_call():
    """Validate the regular call."""
    shearlet_system = ShearletSystem(512, 512, 2)

    # load data
    x = np.random.randn(512, 512)

    # decomposition
    coeffs = shearlet_system.decompose(x)

    print(coeffs)
