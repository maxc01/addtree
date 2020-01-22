import weakref
import typing as tp

import numpy as np


class Parameter:
    def __init__(self, name: str, dim: int, data: tp.Optional[np.ndarray] = None):
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


NAME2NODE = {}


class ParameterNode:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter
        if parameter.name in NAME2NODE:
            raise ValueError(f"Node {parameter.name} already in the tree.")
        NAME2NODE[parameter.name] = self
        self.parant = None
        # a valid local_id start from 0, a float local_id indicates a missing
        # value, at this case, value takes no effect
        self.local_id = -1
        self.depth = 0
        self.children = []

    def add_child(self, child: ParameterNode):
        child.depth = self.depth + 1
        child.parent = self
        child.local_id = len(self.children)
        self.children.append(child)

    def __repr__(self):
        return self.parameter.name

    def finish_add_child(self):

        bfs_template = []

        def set_index(node):
            bfs_template.append(-1)
            node.data_index_in_template = len(bfs_template)
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

    def _is_leaf(self):
        return len(self.children) == 0

    def bfs_nodes(self):

        all_nodes = []

        def collect_node(node):
            all_nodes.append(node)

        self._bfs(func=collect_node)

        return all_nodes

    def random_path(self, reset_node_data=False) -> tp.Sequence[ParameterNode]:

        path = []

        def _sample(node):
            if reset_node_data:
                node.parameter.set_rand()
            path.append(node)
            if node._is_leaf():
                return
            else:
                child = np.random.choice(node.children)
                _sample(child)

        _sample(self)

        return path

    def path2dict(self, path):
        param_dict = {}
        for node in path:
            param_dict[node.parameter.name] = node.parameter.data

        return param_dict

    def dict2long(self, param_dict) -> np.ndarray:
        bfs_repr = self.bfs_template.copy()
        for k, v in param_dict.items():
            node = NAME2NODE[k]
            start = node.data_index_in_template
            end = start + node.parameter.dim
            bfs_repr[start - 1] = node.local_id
            bfs_repr[start:end] = v

        return np.asarray(bfs_repr)

    def path_from_id(self, path_id: str) -> tp.Sequence[ParameterNode]:
        cur = self
        path = [cur]
        for local_id in path_id:
            local_id = int(local_id)
            path.append(cur.children[local_id])
            cur = cur.children[local_id]

        return path

    def path_from_keys(self, keys) -> tp.Sequence[ParameterNode]:
        path = [self]
        for k in keys:
            path.append(NAME2NODE[k])

        return path


### test code
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
total_dim = len(root.bfs_template)

### test kernel
import george.kernels as K
from functools import reduce

name2ker = {}
ks = []
bfs_nodes = root.bfs_nodes()
for node in bfs_nodes:
    kd = K.DeltaKernel(ndim=total_dim, axes=node.data_index_in_template - 1)
    c_start = node.data_index_in_template
    c_end = c_start + node.parameter.dim
    axes = list(range(c_start, c_end))
    if len(axes) == 0:
        k = kd
    else:
        kc = K.ExpSquaredKernel(
            0.5, ndim=total_dim, axes=axes, metric_bounds=[(-7, 4.0)]
        )
        name2ker[node.parameter.name] = kc
        k = kd * kc
    ks.append(k)

kernel = reduce(lambda x, y: x + y, ks)

for name, ker in name2ker.items():
    print(name)
    print(ker.get_parameter_dict())
    print("-" * 30)

### test objective function
def obj_func(param_dict):
    if "x1" in param_dict and "x3" in param_dict:
        value = param_dict["x3"] ** 2 + param_dict["x1"] + 0.1
    elif "x1" in param_dict and "x4" in param_dict:
        value = param_dict["x4"] ** 2 + param_dict["x1"] + 0.2
    elif "x2" in param_dict and "x5" in param_dict:
        value = param_dict["x5"] ** 2 + param_dict["x2"] + 0.3
    elif "x2" in param_dict and "x6" in param_dict:
        value = param_dict["x6"] ** 2 + param_dict["x2"] + 0.4
    else:
        raise KeyError(f"{param_dict} don't contain the correct keys")

    info = dict()
    info["value"] = value.item()
    info["value_sigma"] = 1e-9
    return info


### generate random data and test kernel
from storage import Storage

ss = Storage(kernel)

for i in range(10):
    param_dict = root.path2dict(root.random_path(reset_node_data=True))
    x = root.dict2long(param_dict)
    res = obj_func(param_dict)
    ss.add(x, res["value"], res["value_sigma"])
ss.optimize(2, True)

for name, ker in name2ker.items():
    print(name)
    print(ker.get_parameter_dict())
    print("-" * 30)

### acquisiton function
def LCB(gp, X_new, Y, kappa=1.0):
    pred, pred_var = gp.predict(Y, X_new, return_var=True)
    pred_sigma = np.sqrt(pred_var)
    return pred - kappa * pred_sigma


acq_func = LCB

### optimize using Algorithm2

def optimize_add_GPUCB():
    # one-step optimization
    # from current GP, get next evaluation point
    for node in bfs_nodes:
        if node.parameter.dim > 0:
            pass



def optimize_acq(gp, Y, paths, total_dim, grid_size=100, nb_seed=2):

    info = []
    for path in paths:
        grid = path.rand_grid(grid_size, total_dim)
        grid_acq = acq_func(gp, grid, Y)
        seeds_idx = np.argsort(grid_acq)[:nb_seed]
        eff_axes = path.effective_axes()
        bounds = [(0, 1)] * len(eff_axes)
        ixgrid = np.ix_(seeds_idx, eff_axes)
        seeds = grid[ixgrid]
        # start optimization using these seeds

        def obj_func_acq(x):
            """x is parameter of compression algorithm, so in this case, it is 2d
            """
            fv = path.populate(total_dim, x)
            fv = np.atleast_2d(fv)
            return acq_func(gp, fv, Y).item()

        # minimization
        _x_best = None
        _y_best = np.inf
        for seed in seeds:
            result = minimize(obj_func_acq, x0=seed, method="L-BFGS-B", bounds=bounds)
            if result.fun < _y_best:
                _y_best = result.fun
                _x_best = result.x
        heapq.heappush(info, (_y_best, _x_best, path))

    # y_best, x_best, path = info[0]
    return info[0]


###
print()
param_dict = {"x1": np.array([0.1, 0.3]), "x4": np.array([0.3, 0.5])}
bfs_repr = root.bfs_repr_from_dict(param_dict)
print(bfs_repr)

param_dict = {"x1": np.array([0.1, 0.3]), "x3": np.array([0.3, 0.5])}
bfs_repr = root.bfs_repr_from_dict(param_dict)
print(bfs_repr)

for i in range(5):
    path = root.random_path()
    for node in path:
        print(node.parameter)
    print()

for path_id in ["00", "01", "10", "11"]:
    path = root.path_from_id(path_id)
    for node in path:
        print(node.parameter)
    print()

param_dict = {"x1": 0.2, "x4": 0.4}
path = root.path_from_keys(param_dict.keys())
for node in path:
    print(node.parameter)
print()


param_dict = {"x1": 0.2, "x4": 0.4}
