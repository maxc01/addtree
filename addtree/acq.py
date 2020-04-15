import heapq
import numpy as np
from scipy.optimize import minimize


def UCB(gp, X_new, Y_train, kappa=5.0):
    assert kappa > 0
    pred, pred_var = gp.predict(Y_train, X_new, return_var=True)
    pred_sigma = np.sqrt(pred_var)
    return pred + kappa * pred_sigma


def LCB(gp, X_new, Y_train, kappa=5.0):
    assert kappa > 0
    pred, pred_var = gp.predict(Y_train, X_new, return_var=True)
    pred_sigma = np.sqrt(pred_var)
    return pred - kappa * pred_sigma


def optimize_acq(
    acq_func,
    root,
    gp,
    Y_train,
    total_dim,
    grid_size=100,
    nb_seed=2,
    kappa=-1,
    return_full=False,
    quasi=False,
):

    info = []
    for path_id in root.all_pathids():
        path = root.select_path(path_id)
        eff_axes = path.axes()
        grid = path.rand(grid_size, total_dim, quasi=quasi)
        grid_acq = acq_func(gp, grid, Y_train, kappa=kappa)
        seeds_idx = np.argsort(grid_acq)[:nb_seed]
        bounds = [(0.01, 0.99)] * len(eff_axes)
        ixgrid = np.ix_(seeds_idx, eff_axes)
        seeds = grid[ixgrid]

        def obj_func_acq(x):
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
                _x_best = np.clip(_x_best, *zip(*bounds))
        heapq.heappush(info, (_y_best, path_id, _x_best, path))

    if return_full:
        return info
    else:
        return info[0]
