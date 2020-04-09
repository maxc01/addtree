import pytest
import numpy as np
from scipy.optimize import minimize
import heapq

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


root = large_tree()
ss = Storage()
for i in range(10):
    path = root.random_path(rand_data=True)
    param_dict = root.path2dict(path)
    res = obj_func(param_dict)
    ss.add(root.path2vec(path), res["value"], res["value_sigma"])

ker = build_addtree(root)
gp = ss.optimize(ker, 2, True)


def LCB(gp, X_new, Y_train, kappa=1.0):
    pred, pred_var = gp.predict(Y_train, X_new, return_var=True)
    pred_sigma = np.sqrt(pred_var)
    return pred - kappa * pred_sigma


acq_func = LCB


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
