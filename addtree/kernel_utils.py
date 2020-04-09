import george.kernels as K
from functools import reduce


def built_addtree(root):
    total_dim = len(root.bfs_template)
    name2ker = {}
    ks = []
    bfs_nodes = root.bfs_nodes()
    for node in bfs_nodes:
        kd = K.DeltaKernel(ndim=total_dim, axes=node.bfs_index)
        c_start = node.bfs_index + 1
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

    return kernel

