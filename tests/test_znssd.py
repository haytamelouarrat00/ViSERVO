from __future__ import annotations

import numpy as np

from pvs_framework.pvs_math import znssd_cost


def test_znssd_zero_for_identical_images() -> None:
    image = np.linspace(0.0, 255.0, 64, dtype=np.float64).reshape(8, 8)
    assert np.isclose(znssd_cost(image, image), 0.0, atol=1e-10)


def test_znssd_is_symmetric() -> None:
    a = np.arange(100, dtype=np.float64).reshape(10, 10)
    b = np.flipud(a)
    c1 = znssd_cost(a, b)
    c2 = znssd_cost(b, a)
    assert np.isclose(c1, c2, rtol=1e-10, atol=1e-10)


def test_znssd_increases_with_larger_perturbation() -> None:
    base = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float64), (64, 1))
    near = base + 0.01 * np.sin(np.linspace(0.0, np.pi, 64))[None, :]
    far = base + 0.25 * np.sin(np.linspace(0.0, np.pi, 64))[None, :]

    c_near = znssd_cost(base, near)
    c_far = znssd_cost(base, far)
    assert c_far > c_near
