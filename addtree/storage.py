import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize

import george


class Storage:
    def __init__(self):
        self._X = []
        self._Y = []
        self._Yerr = []
        self._path = []

    def add(self, x, y, yerr=1e-5, path=None):
        self._X.append(x)
        self._Y.append(y)
        self._Yerr.append(yerr)
        self._path.append(path)

    @property
    def X(self):
        return np.asarray(self._X)

    @property
    def Y(self):
        return np.asarray(self._Y)

    @property
    def Yerr(self):
        return np.asarray(self._Yerr)

    def predict(self, gp, X_new):
        pred, pred_var = gp.predict(self.Y, X_new, return_var=True)
        pred_sd = np.sqrt(pred_var)
        return pred, pred_sd

    def optimize(self, kernel, n_restart=1, verbose=False):

        gp = george.GP(kernel, mean=self.Y.mean())

        def _neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(self.Y)

        def _grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(self.Y)

        gp.compute(self.X, self.Yerr)
        if verbose:
            print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(self.Y)))

        bounds = kernel.get_parameter_bounds()
        x_best = None
        y_best = np.inf
        seeds = np.random.uniform(*zip(*bounds), size=(n_restart, len(bounds)))
        for i in range(n_restart):
            result = minimize(
                _neg_ln_like,
                x0=seeds[i],
                jac=_grad_neg_ln_like,
                bounds=bounds,
                method="L-BFGS-B",
            )
            if result.success is False:
                warnings.warn("Gaussian Process optimization is not successful.")
            if result.fun < y_best:
                y_best = result.fun
                x_best = result.x

        if x_best is None:
            raise RuntimeError("All optimizations are not successful.")

        gp.set_parameter_vector(x_best)
        if verbose:
            print("Best parameter of kernel: {}".format(x_best))
            print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(self.Y)))

        return gp

    def best_xy(self):
        idx = np.argmin(self.Y)
        return (self.X[idx], self.Y[idx])


class GroupedStorage:
    def __init__(self):
        self.obs = defaultdict(list)

    def add(self, group, x, full_x, y, yerr):
        self.obs[group].append((x, full_x, y, yerr))

    @property
    def groups(self):
        return self.obs.keys()

    def get_data(self, group_name):
        X, full_X, Y, Yerr = zip(*self.obs[group_name])
        return np.asarray(X), np.asarray(full_X), np.asarray(Y), np.asarray(Yerr)
