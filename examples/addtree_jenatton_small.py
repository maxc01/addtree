import sys
import json
import os
import logging

import numpy as np

from addtree.kernel_utils import build_addtree
from addtree.kernel_utils import get_const_kernel
from addtree.storage import Storage
from addtree.acq import optimize_acq, LCB

from common_utils import setup_logger
from common_utils import get_experiment_id

from addtree_utils import build_tree_jenatton_small
from addtree_utils import path2funcparam_jenatton_small


def obj_func(params):
    SHIFT = 0.5

    if params["L1"]["cat_value"] == "left" and params["L2"]["cat_value"] == "left":
        value = (
            params["L1"]["cont_value"] + (params["L2"]["cont_value"] - SHIFT) ** 2 + 0.1
        )
    elif params["L1"]["cat_value"] == "left" and params["L2"]["cat_value"] == "right":
        value = (
            params["L1"]["cont_value"] + (params["L2"]["cont_value"] - SHIFT) ** 2 + 0.2
        )
    elif params["L1"]["cat_value"] == "right" and params["L2"]["cat_value"] == "left":
        value = (
            params["L1"]["cont_value"] + (params["L2"]["cont_value"] - SHIFT) ** 2 + 0.3
        )
    elif params["L1"]["cat_value"] == "right" and params["L2"]["cat_value"] == "right":
        value = (
            params["L1"]["cont_value"] + (params["L2"]["cont_value"] - SHIFT) ** 2 + 0.4
        )
    else:
        raise ValueError("parameter names are not correct")

    info = dict()
    info["value"] = value
    info["value_sigma"] = 1e-9
    return info


def main():

    try:
        logger = logging.getLogger("addtree-jenatton-small")
        logger.setLevel(logging.DEBUG)

        expid = get_experiment_id(6)
        output_dir = "./exp_results"
        output_dir = os.path.join(output_dir, "addtree", "jenatton-small", expid)
        os.makedirs(output_dir, exist_ok=False)
        log_path = os.path.join(output_dir, "addtree-jenatton-small.log")
        setup_logger(logger, log_path)

        logger.info(f"Experiment {expid} starts...")

        n_init = 5
        root = build_tree_jenatton_small()
        ss = Storage()
        ker = build_addtree(root)
        for i in range(n_init):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Random Design)")
            path = root.random_path(rand_data=True)
            params = path2funcparam_jenatton_small(path[1:])
            obj_info = obj_func(params)
            ss.add(
                path.path2vec(root.obs_dim),
                obj_info["value"],
                obj_info["value_sigma"],
                path,
            )
            logger.info(f"Finishing BO {i+1} iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {"iteration": i + 1, "params": params, "obj_info": obj_info}
            fn_path = os.path.join(output_dir, f"addtree_iter_{i+1}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

        for i in range(n_init, 100):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Optimization)")
            gp = ss.optimize(ker, n_restart=2, verbose=False)
            _, _, x_best, path = optimize_acq(
                LCB,
                root,
                gp,
                ss.Y,
                root.obs_dim,
                grid_size=100,
                nb_seed=2,
                kappa=1.0,
                pb_L=0.0,
                pb_R=1.0,
            )
            path.set_data(x_best)
            params = path2funcparam_jenatton_small(path[1:])
            obj_info = obj_func(params)
            ss.add(
                path.path2vec(root.obs_dim),
                obj_info["value"],
                obj_info["value_sigma"],
                path=path,
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
