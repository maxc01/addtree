import sys
import json
import os
import logging

import numpy as np

from addtree.kernel_utils import build_addtree
from addtree.kernel_utils import get_const_kernel
from addtree.storage import Storage
from addtree.acq import optimize_acq, LCB

from compression_common import setup_logger
from compression_common import setup_and_prune
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id


def main():
    EXP_BASEDIR = "addtree-multiple-new"
    logger = logging.getLogger("addtree-model-compression-vgg16-multiple")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args, _ = get_common_cmd_args()
        expid = get_experiment_id(6)
        output_dir = os.path.join(EXP_BASEDIR, expid)
        os.makedirs(output_dir, exist_ok=False)
        log_path = os.path.join(output_dir, "addtree-model-compression-vgg16.log")
        setup_logger(logger, log_path)

        root = build_tree()
        ss = Storage()
        ker = build_addtree(root)
        const_ker = get_const_kernel(-0.69, root.obs_dim)
        ker = const_ker * ker
        n_init = cmd_args.n_init

        logger.info(f"Experiment {expid} starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

        for i in range(n_init):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Random Design)")
            path = root.random_path(rand_data=True)
            params = path2funcparam(path[1:])
            obj_info = setup_and_prune(cmd_args, params, logger, prune_type="multiple")
            ss.add(
                path.path2vec(root.obs_dim), obj_info["value"], 0.25, path,
            )
            logger.info(f"Finishing BO {i+1} iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {"iteration": i + 1, "params": params, "obj_info": obj_info}
            fn_path = os.path.join(output_dir, f"addtree_iter_{i+1}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

        def get_kappa(t, max_iter):
            ks = np.linspace(1, 3, max_iter)
            return ks[t]

        for i in range(n_init, 300):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Optimization)")
            gp = ss.optimize(ker, n_restart=5, verbose=False)
            _, _, x_best, path = optimize_acq(
                LCB,
                root,
                gp,
                ss.Y,
                root.obs_dim,
                grid_size=2000,
                nb_seed=8,
                kappa=get_kappa(i, 300),
            )
            path.set_data(x_best)
            params = path2funcparam(path[1:])
            obj_info = setup_and_prune(cmd_args, params, logger, prune_type="multiple")
            ss.add(
                path.path2vec(root.obs_dim), obj_info["value"], 0.25, path=path,
            )
            logger.info(f"Finishing BO {i+1} iteration")
            logger.info(params)
            logger.info(obj_info)
            all_info = {"iteration": i + 1, "params": params, "obj_info": obj_info}
            fn_path = os.path.join(output_dir, f"addtree_iter_{i+1}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
