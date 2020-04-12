import sys
import queue
import json
import os
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms

from models.vgg import VGG
from compression_common import testing_params
from compression_common import do_prune
from compression_common import train, test
from compression_common import setup_logger
from compression_common import get_data_loaders
from compression_common import setup_and_prune


# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.random_configuration_design import RandomConfigurations
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario

# from smac.facade.smac_facade import SMAC
from smac.facade.smac_hpo_facade import SMAC4HPO

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

root = CategoricalHyperparameter("root", choices=["l1", "ln"])
x1 = CategoricalHyperparameter("x1", choices=["l1", "ln"])
x2 = CategoricalHyperparameter("x2", choices=["l1", "ln"])
x3 = CategoricalHyperparameter("x3", choices=["l1", "ln"])
x4 = CategoricalHyperparameter("x4", choices=["l1", "ln"])
x5 = CategoricalHyperparameter("x5", choices=["l1", "ln"])
x6 = CategoricalHyperparameter("x6", choices=["l1", "ln"])

# r1 is the data associated in x1
r1 = UniformFloatHyperparameter("r1", lower=0.0, upper=1.0, log=False)
r2 = UniformFloatHyperparameter("r2", lower=0.0, upper=1.0, log=False)
r3 = UniformFloatHyperparameter("r3", lower=0.0, upper=1.0, log=False)
r4 = UniformFloatHyperparameter("r4", lower=0.0, upper=1.0, log=False)
r5 = UniformFloatHyperparameter("r5", lower=0.0, upper=1.0, log=False)
r6 = UniformFloatHyperparameter("r6", lower=0.0, upper=1.0, log=False)
r7 = UniformFloatHyperparameter("r7", lower=0.0, upper=1.0, log=False)
r8 = UniformFloatHyperparameter("r8", lower=0.0, upper=1.0, log=False)
r9 = UniformFloatHyperparameter("r9", lower=0.0, upper=1.0, log=False)
r10 = UniformFloatHyperparameter("r10", lower=0.0, upper=1.0, log=False)
r11 = UniformFloatHyperparameter("r11", lower=0.0, upper=1.0, log=False)
r12 = UniformFloatHyperparameter("r12", lower=0.0, upper=1.0, log=False)
r13 = UniformFloatHyperparameter("r13", lower=0.0, upper=1.0, log=False)
r14 = UniformFloatHyperparameter("r14", lower=0.0, upper=1.0, log=False)

cs.add_hyperparameters(
    [
        root,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        r1,
        r2,
        r3,
        r4,
        r5,
        r6,
        r7,
        r8,
        r9,
        r10,
        r11,
        r12,
        r13,
        r14,
    ]
)


# add condition
cs.add_condition(InCondition(x1, root, ["l1"]))
cs.add_condition(InCondition(x2, root, ["ln"]))
cs.add_condition(InCondition(r1, root, ["l1"]))
cs.add_condition(InCondition(r2, root, ["ln"]))

cs.add_condition(InCondition(x3, x1, ["l1"]))
cs.add_condition(InCondition(x4, x1, ["ln"]))
cs.add_condition(InCondition(r3, x1, ["l1"]))
cs.add_condition(InCondition(r4, x1, ["ln"]))

cs.add_condition(InCondition(x5, x2, ["l1"]))
cs.add_condition(InCondition(x6, x2, ["ln"]))
cs.add_condition(InCondition(r5, x2, ["l1"]))
cs.add_condition(InCondition(r6, x2, ["ln"]))

cs.add_condition(InCondition(r7, x3, ["l1"]))
cs.add_condition(InCondition(r8, x3, ["ln"]))

cs.add_condition(InCondition(r9, x4, ["l1"]))
cs.add_condition(InCondition(r10, x4, ["ln"]))

cs.add_condition(InCondition(r11, x5, ["l1"]))
cs.add_condition(InCondition(r12, x5, ["ln"]))

cs.add_condition(InCondition(r13, x6, ["l1"]))
cs.add_condition(InCondition(r14, x6, ["ln"]))

#


def cfg2funcparams(cfg):
    """
    Configuration:
    r1, Value: 0.5371703233713245
    r10, Value: 0.32063392742881947
    r4, Value: 0.91056232889749
    root, Value: 'l1'
    x1, Value: 'ln'
    x4, Value: 'ln'
    """

    def extract_one_value(keys):
        """this function extract ONE value with key in keys from cfg
        keys: e.g. ["x1","x2"]
        """

        for k in keys:
            if k in cfg.keys():
                return cfg[k]
        raise RuntimeError("key not exist")

    params = {}
    params["b1"] = {}
    params["b1"]["prune_method"] = cfg["root"]
    params["b1"]["amount"] = extract_one_value(["r1", "r2"])

    params["b2"] = {}
    params["b2"]["prune_method"] = "ln"
    params["b2"]["amount"] = extract_one_value(["r3", "r4", "r5", "r6"])

    params["b3"] = {}
    params["b3"]["prune_method"] = "l1"
    params["b3"]["amount"] = extract_one_value(
        ["r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14",]
    )

    return params


def get_cmd_args():
    """ Get parameters from command line """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="checkpoints directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
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
    logger = logging.getLogger("SMAC-model-compression-vgg16")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args = get_cmd_args()
        log_path = os.path.join(
            cmd_args.checkpoints_dir, "SMAC-model-compression-vgg16.log"
        )
        setup_logger(logger, log_path)

        def obj_func(cfg):
            logger.info("Starting BO iteration")
            params = cfg2funcparams(cfg)
            obj_info = setup_and_prune(cmd_args, params, logger)
            logger.info("Finishing BO iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {
                "params": params,
                "obj_info": obj_info,
            }
            fn_path = os.path.join(cmd_args.checkpoints_dir, "smac_iter_hists.txt")
            with open(fn_path, "a") as f:
                json.dump(all_info, f)
                f.write("\n")

            return obj_info["value"]

        # smac default do minimize
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount_limit": 300,  # maximum function evaluations
                "cs": cs,  # configuration space
                "deterministic": "true",
            }
        )

        smac = SMAC4HPO(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=obj_func,
            initial_design=RandomConfigurations,
            initial_design_kwargs={"init_budget": 20,},
        )

        incumbent = smac.optimize()
        print(incumbent)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
