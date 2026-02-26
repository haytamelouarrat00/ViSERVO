from __future__ import annotations

import numpy as np

from pvs_framework.camera import exp_se3, log_se3


def test_exp_log_se3_roundtrip() -> None:
    rng = np.random.default_rng(42)

    for _ in range(20):
        xi = rng.normal(scale=0.05, size=6)
        T = exp_se3(xi)
        xi_back = log_se3(T)
        T_round = exp_se3(xi_back)

        assert np.allclose(T_round, T, atol=1e-6)
