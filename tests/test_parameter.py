import pytest
from pytest import approx

import numpy as np

from addtree.parameter import Parameter
from addtree.parameter import ParameterNode
from addtree.parameter import clear_state
from addtree.parameter import NodePath


@pytest.fixture
def small_tree1():
    clear_state()
    root = ParameterNode(Parameter("root", 2, np.array([1, 2])))
    x1 = ParameterNode(Parameter("x1", 2, np.array([3.3, 4])))
    x2 = ParameterNode(Parameter("x2", 3, np.array([5, 6.6, 7])))
    root.add_child(x1)
    root.add_child(x2)
    root.finish_add_child()

    return root, x1, x2


@pytest.fixture
def large_tree():
    clear_state()
    root = ParameterNode(Parameter("root", 0))
    x1 = ParameterNode(Parameter("x1", 1))
    x2 = ParameterNode(Parameter("x2", 1))
    root.add_child(x1)
    root.add_child(x2)
    x3 = ParameterNode(Parameter("x3", 1))
    x4 = ParameterNode(Parameter("x4", 1))
    x1.add_child(x3)
    x1.add_child(x4)

    x5 = ParameterNode(Parameter("x5", 1))
    x6 = ParameterNode(Parameter("x6", 1))
    x2.add_child(x5)
    x2.add_child(x6)

    root.finish_add_child()

    return root


def test_bfs_template(small_tree1):
    root = small_tree1[0]
    assert len(root.bfs_template) == 10
    assert root.bfs_template == approx(np.array([-1] * 10))


def test_obs_dim(small_tree1):
    root = small_tree1[0]
    assert root.obs_dim == 10


def test_set_data(small_tree1):
    root, x1, x2 = small_tree1

    p1 = NodePath([root, x1]).set_data(np.array([9, 8, 7, 6], "f"))
    assert p1[0].parameter.data == approx(np.array([9, 8]))
    assert p1[1].parameter.data == approx(np.array([7, 6]))

    p2 = NodePath([root, x2]).set_data(np.array([8, 7, 6, 5, 4], "f"))
    assert p2[0].parameter.data == approx(np.array([8, 7]))
    assert p2[1].parameter.data == approx(np.array([6, 5, 4]))


def test_path2dict(small_tree1):
    root, x1, x2 = small_tree1
    d1 = NodePath([root, x1]).path2dict()
    assert d1["root"] == approx(np.array([1, 2]))
    assert d1["x1"] == approx(np.array([3.3, 4]))

    d2 = NodePath([root, x2]).path2dict()
    assert d2["root"] == approx(np.array([1, 2]))
    assert d2["x2"] == approx(np.array([5, 6.6, 7]))


def test_path2vec(small_tree1):
    root, x1, x2 = small_tree1
    assert NodePath([root, x1]).path2vec(root.obs_dim) == approx(
        np.array([0, 1, 2, 0, 3.3, 4, -1, -1, -1, -1])
    )

    assert NodePath([root, x2]).path2vec(root.obs_dim) == approx(
        np.array([0, 1, 2, -1, -1, -1, 1, 5, 6.6, 7])
    )


def test_bfs_nodes(small_tree1):
    root, x1, x2 = small_tree1
    all_nodes = root.bfs_nodes()
    assert all_nodes[0] == root
    assert all_nodes[1] == x1
    assert all_nodes[2] == x2


def test_select_path(small_tree1):
    root, x1, x2 = small_tree1

    path1 = root.select_path("0")
    assert path1[0] == root
    assert path1[1] == x1

    path2 = root.select_path("1")
    assert path2[0] == root
    assert path2[1] == x2


def test_all_pathids(small_tree1):
    root, x1, x2 = small_tree1

    assert root.all_pathids() == ["0", "1"]


def test_all_pathids_large(large_tree):
    root = large_tree
    assert root.all_pathids() == ["00", "01", "10", "11"]
