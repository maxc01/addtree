import sys
import json
import os
import argparse
import logging

from addtree.kernel_utils import build_addtree
from addtree.storage import Storage
from addtree.parameter import Parameter
from addtree.parameter import ParameterNode
from addtree.acq import optimize_acq, LCB

from compression_common import setup_logger
from compression_common import setup_and_prune


NAME2METHOD = {
    "x1": "l1",
    "x2": "ln",
    "x3": "l1",
    "x4": "ln",
    "x5": "l1",
    "x6": "ln",
    "x7": "l1",
    "x8": "ln",
    "x9": "l1",
    "x10": "ln",
    "x11": "l1",
    "x12": "ln",
    "x13": "l1",
    "x14": "ln",
}


def build_tree():
    root = ParameterNode(Parameter("root", 0))
    x1 = ParameterNode(Parameter("x1", 1))
    x2 = ParameterNode(Parameter("x2", 1))
    x3 = ParameterNode(Parameter("x3", 1))
    x4 = ParameterNode(Parameter("x4", 1))
    x5 = ParameterNode(Parameter("x5", 1))
    x6 = ParameterNode(Parameter("x6", 1))
    x7 = ParameterNode(Parameter("x7", 1))
    x8 = ParameterNode(Parameter("x8", 1))
    x9 = ParameterNode(Parameter("x9", 1))
    x10 = ParameterNode(Parameter("x10", 1))
    x11 = ParameterNode(Parameter("x11", 1))
    x12 = ParameterNode(Parameter("x12", 1))
    x13 = ParameterNode(Parameter("x13", 1))
    x14 = ParameterNode(Parameter("x14", 1))

    root.add_child(x1)
    root.add_child(x2)

    x1.add_child(x3)
    x1.add_child(x4)

    x2.add_child(x5)
    x2.add_child(x6)

    x3.add_child(x7)
    x3.add_child(x8)

    x4.add_child(x9)
    x4.add_child(x10)

    x5.add_child(x11)
    x5.add_child(x12)

    x6.add_child(x13)
    x6.add_child(x14)

    root.finish_add_child()

    return root


def path2funcparam(path):
    b_names = ["b1", "b2", "b3"]
    params = {}
    for b_name, node in zip(b_names, path):
        params[b_name] = {}
        params[b_name]["prune_method"] = NAME2METHOD[node.name]
        params[b_name]["amount"] = node.parameter.data.item()

    return params


def get_cmd_args():
    """ Get parameters from command line """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_init",
        type=int,
        default=20,
        metavar="N",
        help="number of random design (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--prune_epochs",
        type=int,
        default=5,
        metavar="N",
        help="training epochs for model pruning (default: 5)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="checkpoints directory",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="pretrained model weights"
    )
    parser.add_argument(
        "--multi_gpu", action="store_true", help="Use multiple GPUs for training"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args, _ = parser.parse_known_args()

    return args


def main():
    logger = logging.getLogger("addtree-model-compression-vgg16")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args = get_cmd_args()
        os.makedirs(cmd_args.checkpoints_dir, exist_ok=True)
        log_path = os.path.join(
            cmd_args.checkpoints_dir, "addtree-model-compression-vgg16.log"
        )
        setup_logger(logger, log_path)

        root = build_tree()
        ss = Storage()
        ker = build_addtree(root)
        n_init = cmd_args.n_init

        for i in range(n_init):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Random Design)")
            path = root.random_path(rand_data=True)
            params = path2funcparam(path[1:])
            obj_info = setup_and_prune(cmd_args, params, logger)
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
            fn_path = os.path.join(cmd_args.checkpoints_dir, f"addtree_iter_{i+1}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

        for i in range(n_init, 300):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Optimization)")
            gp = ss.optimize(ker, n_restart=2, verbose=False)
            _, _, x_best, path = optimize_acq(
                LCB, root, gp, ss.Y, root.obs_dim, grid_size=500, nb_seed=5
            )
            path.set_data(x_best)
            params = path2funcparam(path[1:])
            obj_info = setup_and_prune(cmd_args, params, logger)
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
            fn_path = os.path.join(cmd_args.checkpoints_dir, f"addtree_iter_{i+1}.json")
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
