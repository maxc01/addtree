import weakref
import typing as tp
from typing import Dict, Sequence, Optional

import numpy as np


class Parameter:
    def __init__(self, name: str, dim: int, data: tp.Optional[np.ndarray] = None):
        self.name = name
        self.dim = dim
        self.data = data
        if data is None:
            self.set_rand()
        assert data.shape == (dim,)

    def set_rand(self):
        self.data = np.random.rand(self.dim)

    def __repr__(self):
        data_repr = ", ".join(["{:.2f}".format(i) for i in self.data])
        return "{}: [{}]".format(self.name, data_repr)


NAME2NODE = {}


def clear_state():
    global NAME2NODE
    NAME2NODE = {}


class ParameterNode:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter
        self.name = parameter.name
        if self.name in NAME2NODE:
            raise ValueError(f"Node {self.name} already in the tree.")
        NAME2NODE[self.name] = self
        self.parant = None
        # a valid local_id start from 0, a float local_id indicates a missing
        # value, at this case, value takes no effect
        self.local_id = 0 if parameter.dim > 0 else -1
        self.depth = 0
        self.children = []

    def add_child(self, child: "ParameterNode"):
        child.depth = self.depth + 1
        child.parent = self
        child.local_id = len(self.children)
        self.children.append(child)

    def __repr__(self):
        return self.parameter.name

    def __eq__(self, other: "ParameterNode"):
        return self.name == other.name

    def finish_add_child(self):

        bfs_template = []

        def set_index(node):
            bfs_template.append(-1)
            node.bfs_index = len(bfs_template) - 1
            bfs_template.extend([-1] * node.parameter.dim)

        self._bfs(func=set_index)

        self.bfs_template = bfs_template

    def _bfs(self, func=None, *func_args, **func_kwargs):
        bfs_queue = []
        bfs_queue.append(self)
        while len(bfs_queue):
            node = bfs_queue.pop(0)
            # visit node and do something
            if callable(func):
                func(node, *func_args, **func_kwargs)
            for child in node.children:
                bfs_queue.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def bfs_nodes(self):

        all_nodes = []

        def collect_node(node):
            all_nodes.append(node)

        self._bfs(func=collect_node)

        return all_nodes

    def random_path(self) -> Sequence["ParameterNode"]:
        """generate a random path from current node.
        """

        path = []

        def _sample(node):
            path.append(node)
            if node.is_leaf():
                return
            else:
                child = np.random.choice(node.children)
                _sample(child)

        _sample(self)

        return path

    def path2dict(self, path: Sequence["ParameterNode"]) -> Dict[str, np.ndarray]:
        """transform a path to a dict, which is then passed to objective function.
        """
        param_dict = {node.name: node.parameter.data for node in path}
        return param_dict

    def dict2vec(self, param_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """transform a dict to a vector, which will be added to a Storage.
        """
        bfs_repr = self.bfs_template.copy()
        for k, v in param_dict.items():
            node = NAME2NODE[k]
            n_start = node.bfs_index
            n_end = n_start + node.parameter.dim + 1
            bfs_repr[n_start] = node.local_id
            bfs_repr[(n_start + 1) : n_end] = v

        return np.asarray(bfs_repr)

    def path2vec(self, path: Sequence["ParameterNode"]) -> np.ndarray:
        """transform a path to a vector, which will be added to a Storage.
        """
        param_dict = self.path2dict(path)
        return self.dict2vec(param_dict)

    def select_path(self, path_id: str) -> tp.Sequence["ParameterNode"]:
        """select path from a string

        Note: path_id should not contain itself, e.g. '10' means select second
        child of current node,

        """
        cur = self
        path = [cur]
        for local_id in path_id:
            local_id = int(local_id)
            path.append(cur.children[local_id])
            cur = cur.children[local_id]

        return path

    def path_from_keys(self, keys) -> tp.Sequence["ParameterNode"]:
        path = [self]
        for k in keys:
            path.append(NAME2NODE[k])

        return path
