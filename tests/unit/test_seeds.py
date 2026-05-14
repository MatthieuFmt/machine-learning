import numpy as np
from app.core.seeds import set_global_seeds


def test_reproducible():
    set_global_seeds(123)
    a = np.random.rand(10)
    set_global_seeds(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)
