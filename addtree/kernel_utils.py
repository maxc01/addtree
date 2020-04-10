import george.kernels as K
from functools import reduce


def build_addtree(root):
    obs_dim = root.obs_dim
    name2ker = {}
    ks = []
    bfs_nodes = root.bfs_nodes()
    for node in bfs_nodes:
        kd = K.DeltaKernel(ndim=obs_dim, axes=node.bfs_index)
        axes = node.param_axes
        if len(axes) == 0:
            k = kd
        else:
            kc = K.ExpSquaredKernel(
                0.5, ndim=obs_dim, axes=axes, metric_bounds=[(-7, 4.0)]
            )
            name2ker[node.parameter.name] = kc
            k = kd * kc
        ks.append(k)

    kernel = reduce(lambda x, y: x + y, ks)

    return kernel

