from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_NETWORKX = importlib.util.find_spec("networkx") is not None
if HAS_NETWORKX:
    from pvs_framework.planner import PhotometricAStarPlanner
else:
    PhotometricAStarPlanner = None


class DummyController:
    pass


class DummyRenderer:
    pass


@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx is not installed")
def test_kinggraph_node_count_and_connectivity() -> None:
    planner = PhotometricAStarPlanner(
        controller=DummyController(),
        renderer=DummyRenderer(),
        hd_min=0.1,
        search_space_bounds=((0.0, 1.0), (0.0, 1.0)),
        dof=2,
    )

    n = 4
    G = planner._build_kinggraph(n=n, template_pose=np.eye(4, dtype=np.float64))

    assert G.number_of_nodes() == (n + 1) ** 2
    assert G.degree[(2, 2)] == 8
    assert G.degree[(0, 0)] == 3
    assert G.degree[(0, 2)] == 5
