import pytest
import numpy as np
from scipy.optimize import minimize
import heapq
from tqdm import tqdm

from addtree.kernel_utils import build_addtree
from addtree.storage import Storage
from addtree.parameter import Parameter
from addtree.parameter import ParameterNode
from addtree.parameter import clear_state
from addtree.parameter import get_state


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


def obj_func(param_dict):
    SHIFT = 0.5
    if "x1" in param_dict and "x3" in param_dict:
        value = (param_dict["x3"] - SHIFT) ** 2 + param_dict["x1"] + 0.1
    elif "x1" in param_dict and "x4" in param_dict:
        value = (param_dict["x4"] - SHIFT) ** 2 + param_dict["x1"] + 0.2
    elif "x2" in param_dict and "x5" in param_dict:
        value = (param_dict["x5"] - SHIFT) ** 2 + param_dict["x2"] + 0.3
    elif "x2" in param_dict and "x6" in param_dict:
        value = (param_dict["x6"] - SHIFT) ** 2 + param_dict["x2"] + 0.4
    else:
        raise KeyError(f"{param_dict} don't contain the correct keys")

    info = dict()
    info["value"] = value.item()
    info["value_sigma"] = 1e-9
    return info


clear_state()
root = large_tree()
ss = Storage()
for i in range(5):
    path = root.random_path(rand_data=True)
    res = obj_func(path.path2dict())
    ss.add(path.path2vec(root.obs_dim), res["value"], res["value_sigma"], path)

ker = build_addtree(root)


def LCB(gp, X_new, Y_train, kappa=1.0):
    pred, pred_var = gp.predict(Y_train, X_new, return_var=True)
    pred_sigma = np.sqrt(pred_var)
    return pred - kappa * pred_sigma


# IMPORTANT: any parameter passed to acq MUST be effective, i.e. other
# dimensions must be set to be invalid
# BUG: the above is just the bug

acq_func = LCB


# BUG: the following is wrong, any parameter passed to acq MUST be effective
def fill_x(x, total_dim, eff_axes):
    vec = np.empty(total_dim)
    vec[...] = -1
    vec[eff_axes] = x
    return vec


def optimize_acq(gp, Y_train, total_dim, grid_size=100, nb_seed=2):

    info = []
    for path_id in ["00", "01", "10", "11"]:
        path = root.select_path(path_id)
        eff_axes = path.axes()
        grid = path.rand(grid_size, total_dim)
        grid_acq = acq_func(gp, grid, Y_train)
        seeds_idx = np.argsort(grid_acq)[:nb_seed]
        bounds = [(0, 1)] * len(eff_axes)
        ixgrid = np.ix_(seeds_idx, eff_axes)
        seeds = grid[ixgrid]

        def obj_func_acq(x):
            # vec = fill_x(x, root.obs_dim, eff_axes)
            vec = path.set_data(x).path2vec(root.obs_dim)
            return acq_func(gp, vec[None], Y_train).item()

        # start minimization using these seeds
        _x_best = None
        _y_best = np.inf
        for seed in seeds:
            result = minimize(obj_func_acq, x0=seed, method="L-BFGS-B", bounds=bounds,)
            if result.fun < _y_best:
                _y_best = result.fun
                _x_best = result.x
        heapq.heappush(info, (_y_best, path_id, _x_best, path))

    return info[0]


for i in tqdm(range(95)):
    gp = ss.optimize(ker, n_restart=2, verbose=False)
    _, _, x_best, path = optimize_acq(gp, ss.Y, root.obs_dim)
    path.set_data(x_best)
    obj_info = obj_func(path.path2dict())
    ss.add(
        path.path2vec(root.obs_dim),
        obj_info["value"],
        obj_info["value_sigma"],
        path=path,
    )
