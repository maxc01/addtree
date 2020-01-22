import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize

import george


class Storage:
    def __init__(self, kernel):
        self.kernel = kernel
        self._X = []
        self._Y = []
        self._Yerr = []
        self._paths = []

    def add(self, x, y, yerr, path=None):
        self._X.append(x)
        self._Y.append(y)
        self._Yerr.append(yerr)
        if path is not None:
            self._paths.append(path)

    def optimize(self, n_restart=1, verbose=False):

        gp = george.GP(self.kernel, mean=self.Y.mean())

        def _neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(self.Y)

        def _grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(self.Y)

        gp.compute(self.X, self.Yerr)
        if verbose:
            print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(self.Y)))

        bounds = self.kernel.get_parameter_bounds()
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
                warnings.warn("Gaussian Process Optimization is not successful")
            if result.fun < y_best:
                y_best = result.fun
                x_best = result.x

        if x_best is None:
            raise RuntimeError("All Optimization is not successful")

        gp.set_parameter_vector(x_best)
        if verbose:
            # print(result)
            print("Best parameter of kernel: {}".format(x_best))
            print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(self.Y)))

        self.gp = gp

    def bestXY(self, minimization=True):
        if minimization:
            idx = np.argmin(self.Y)
        else:
            idx = np.argmax(self.Y)
        return (self.X[idx], self.Y[idx])

    @property
    def X(self):
        return np.asarray(self._X)

    @property
    def Y(self):
        return np.asarray(self._Y)

    @property
    def Yerr(self):
        return np.asarray(self._Yerr)

    def uniform_grid(self, n):
        return np.random.rand(n, self.X.shape[1])

    def beta_grid(self, n, a=2, b=5):
        return beta(a, b).rvs(size=(n, self.X.shape[1]))


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
