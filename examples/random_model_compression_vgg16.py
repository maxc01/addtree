import sys
import json
import os
import argparse
import logging

from compression_common import setup_logger
from compression_common import setup_and_prune
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id
from smac_utils import cfg2funcparams

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from ConfigSpace.conditions import InCondition

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


def main():
    EXP_BASEDIR = "random-single"
    logger = logging.getLogger("random-model-compression-vgg16")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args, _ = get_common_cmd_args()
        expid = get_experiment_id(6)
        output_dir = os.path.join(EXP_BASEDIR, expid)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "random-model-compression-vgg16.log")
        setup_logger(logger, log_path)

        logger.info(f"Experiment {expid} starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

        def obj_func(cfg, opt_iter):
            logger.info("Starting BO iteration")
            params = cfg2funcparams(cfg)
            obj_info = setup_and_prune(cmd_args, params, logger)
            logger.info("Finishing BO iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {
                "iteration": opt_iter,
                "params": params,
                "obj_info": obj_info,
            }
            fn_path = os.path.join(output_dir, f"random_iter_{opt_iter}.txt")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

            return obj_info["value"]

        for i in range(300):
            cfg = cs.sample_configuration()
            obj_func(cfg, i + 1)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
