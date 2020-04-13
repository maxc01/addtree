import heapq
import numpy as np
from scipy.optimize import minimize


def LCB(gp, X_new, Y_train, kappa=5.0):
    pred, pred_var = gp.predict(Y_train, X_new, return_var=True)
    pred_sigma = np.sqrt(pred_var)
    return pred - kappa * pred_sigma


def optimize_acq(acq_func, root, gp, Y_train, total_dim, grid_size=100, nb_seed=2):

    info = []
    for path_id in root.all_pathids():
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
