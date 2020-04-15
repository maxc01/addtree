from typing import Dict, Sequence, Optional

import numpy as np

NAME2NODE = {}


def clear_state():
    global NAME2NODE
    NAME2NODE = {}


def get_state():
    return NAME2NODE


class NodePath:
    """ This class represents a list of ParameterNode
    """

    def __init__(self, path: Sequence["ParameterNode"]):
        self.path = path

    def path2dict(self) -> Dict[str, np.ndarray]:
        """transform a path to a dict, which is then passed to objective function.
        """
        param_dict = {node.name: node.parameter.data for node in self.path}
        return param_dict

    def path2vec(self, obs_dim: int) -> np.ndarray:
        """transform a path to a vector, which will be added to a Storage.
        """
        param_dict = self.path2dict()
        return self.dict2vec(param_dict, obs_dim)

    def dict2vec(self, param_dict: Dict[str, np.ndarray], obs_dim: int) -> np.ndarray:
        """transform a dict to a vector, which will be added to a Storage.
        # IMPORTANT: The basic patter is: first set flag, second set data
        """
        bfs_repr = np.array([-1] * obs_dim, dtype="f")
        for k, v in param_dict.items():
            node = NAME2NODE[k]
            bfs_repr[node.bfs_index] = node.local_id
            bfs_repr[node.param_axes] = v

        return bfs_repr

    def axes(self):
        _axes = []
        for node in self.path:
            _axes.extend(node.param_axes)
        return _axes

    def set_data(self, x: np.ndarray) -> "NodePath":
        i = 0
        for node in self.path:
            node.parameter.data = x[i : (i + node.parameter.dim)]
            i += node.parameter.dim

        return self

    def rand(self, n, obs_dim, quasi=False):
        if quasi:
            import ghalton
            a = -np.ones((n, obs_dim))
            for node in self.path:
                sequencer = ghalton.Halton(node.parameter.dim)
                a[:, node.bfs_index] = node.local_id
                a[:, node.param_axes] = sequencer.get(n)
        else:
            a = -np.random.rand(n, obs_dim)
            for node in self.path:
                a[:, node.bfs_index] = node.local_id
                a[:, node.param_axes] *= -1
        return a

    def __getitem__(self, idx):
        return self.path[idx]

    def __repr__(self):
        return "->".join([node.name for node in self.path])


class Parameter:
    def __init__(self, name: str, dim: int, data: Optional[np.ndarray] = None):
        self.name = name
        self.dim = dim
        self.data = data
        if data is None:
            self.set_rand()

    def set_rand(self):
        self.data = np.random.rand(self.dim)

    def __repr__(self):
        data_repr = ", ".join(["{:.2f}".format(i) for i in self.data])
        return "{}: [{}]".format(self.name, data_repr)


class ParameterNode:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter
        self.name = parameter.name
        # TODO: use ID instead of name to index node
        # because name can duplicate, but ID cannot
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
            n_start = len(bfs_template)
            n_end = n_start + node.parameter.dim + 1
            node.bfs_index = n_start
            node.param_axes = list(range(n_start + 1, n_end))
            bfs_template.extend([-1] * (1 + node.parameter.dim))

        self._bfs(func=set_index)
        self.obs_dim = len(bfs_template)

        self.bfs_template = np.asarray(bfs_template, dtype="f")

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

    def _dfs(self, func=None, *func_args, **func_kwargs):
        """pre-order DFS
        """

        def preorder(node):
            # visit node and do something
            if callable(func):
                func(node, *func_args, **func_kwargs)
            for child in self.children:
                preorder(child)

        preorder(self)

    def is_leaf(self):
        return len(self.children) == 0

    def bfs_nodes(self):

        all_nodes = []

        def collect_node(node):
            all_nodes.append(node)

        self._bfs(func=collect_node)

        return all_nodes

    def random_path(self, rand_data=False) -> NodePath:
        """generate a random path from current node.
        """

        path = []

        def _sample(node):
            if rand_data:
                node.parameter.set_rand()
            path.append(node)
            if node.is_leaf():
                return
            else:
                child = np.random.choice(node.children)
                _sample(child)

        _sample(self)

        return NodePath(path)

    def select_path(self, path_id: str) -> NodePath:
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

        return NodePath(path)

    def all_pathids(self) -> Sequence[str]:
        pathids = []

        def preorder(node, cur_ids):
            cur_ids.append(str(node.local_id))
            if node.is_leaf():
                pathids.append("".join(cur_ids[1:]))
            for child in node.children:
                preorder(child, cur_ids)

            cur_ids.pop()

        cur_ids = []
        preorder(self, cur_ids)

        return pathids

    def path_from_keys(self, keys) -> Sequence["ParameterNode"]:
        path = [self]
        for k in keys:
            path.append(NAME2NODE[k])

        return path
