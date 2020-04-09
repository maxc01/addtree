import pytest
from pytest import approx

import numpy as np

from addtree.parameter import Parameter
from addtree.parameter import ParameterNode
from addtree.parameter import clear_state


@pytest.fixture
def small_tree1():
    clear_state()
    root = ParameterNode(Parameter("root", 2, np.array([1, 2])))
    x1 = ParameterNode(Parameter("x1", 2, np.array([3, 4])))
    x2 = ParameterNode(Parameter("x2", 3, np.array([5, 6, 7])))
    root.add_child(x1)
    root.add_child(x2)
    root.finish_add_child()

    return root, x1, x2


@pytest.fixture
def large_tree():
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


# total_dim = len(root.bfs_template)


def test_bfs_template(small_tree1):
    root = small_tree1[0]
    assert len(root.bfs_template) == 10
    assert root.bfs_template == [-1] * 10


def test_path2dict(small_tree1):
    root, x1, x2 = small_tree1
    d1 = root.path2dict([root, x1])
    assert d1["root"] == approx(np.array([1, 2]))
    assert d1["x1"] == approx(np.array([3, 4]))

    d2 = root.path2dict([root, x2])
    assert d2["root"] == approx(np.array([1, 2]))
    assert d2["x2"] == approx(np.array([5, 6, 7]))


def test_path2vec(small_tree1):
    root, x1, x2 = small_tree1
    assert root.path2vec([root, x1]) == approx(
        np.array([0, 1, 2, 0, 3, 4, -1, -1, -1, -1])
    )
    assert root.path2vec([root, x2]) == approx(
        np.array([0, 1, 2, -1, -1, -1, 1, 5, 6, 7])
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
