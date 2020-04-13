import sys
from datetime import datetime
import json
import os
import argparse
import logging

from compression_common import setup_logger
from compression_common import setup_and_prune
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id
from smac_utils import cfg2funcparams_multiple

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
r1_1 = UniformFloatHyperparameter("r1_1", lower=0.0, upper=1.0, log=False)
r1_2 = UniformFloatHyperparameter("r1_2", lower=0.0, upper=1.0, log=False)
r1_3 = UniformFloatHyperparameter("r1_3", lower=0.0, upper=1.0, log=False)

r2_1 = UniformFloatHyperparameter("r2_1", lower=0.0, upper=1.0, log=False)
r2_2 = UniformFloatHyperparameter("r2_2", lower=0.0, upper=1.0, log=False)
r2_3 = UniformFloatHyperparameter("r2_3", lower=0.0, upper=1.0, log=False)

r3_1 = UniformFloatHyperparameter("r3_1", lower=0.0, upper=1.0, log=False)
r3_2 = UniformFloatHyperparameter("r3_2", lower=0.0, upper=1.0, log=False)
r3_3 = UniformFloatHyperparameter("r3_3", lower=0.0, upper=1.0, log=False)

r4_1 = UniformFloatHyperparameter("r4_1", lower=0.0, upper=1.0, log=False)
r4_2 = UniformFloatHyperparameter("r4_2", lower=0.0, upper=1.0, log=False)
r4_3 = UniformFloatHyperparameter("r4_3", lower=0.0, upper=1.0, log=False)

r5_1 = UniformFloatHyperparameter("r5_1", lower=0.0, upper=1.0, log=False)
r5_2 = UniformFloatHyperparameter("r5_2", lower=0.0, upper=1.0, log=False)
r5_3 = UniformFloatHyperparameter("r5_3", lower=0.0, upper=1.0, log=False)

r6_1 = UniformFloatHyperparameter("r6_1", lower=0.0, upper=1.0, log=False)
r6_2 = UniformFloatHyperparameter("r6_2", lower=0.0, upper=1.0, log=False)
r6_3 = UniformFloatHyperparameter("r6_3", lower=0.0, upper=1.0, log=False)

r7_1 = UniformFloatHyperparameter("r7_1", lower=0.0, upper=1.0, log=False)
r7_2 = UniformFloatHyperparameter("r7_2", lower=0.0, upper=1.0, log=False)
r7_3 = UniformFloatHyperparameter("r7_3", lower=0.0, upper=1.0, log=False)

r8_1 = UniformFloatHyperparameter("r8_1", lower=0.0, upper=1.0, log=False)
r8_2 = UniformFloatHyperparameter("r8_2", lower=0.0, upper=1.0, log=False)
r8_3 = UniformFloatHyperparameter("r8_3", lower=0.0, upper=1.0, log=False)

r9_1 = UniformFloatHyperparameter("r9_1", lower=0.0, upper=1.0, log=False)
r9_2 = UniformFloatHyperparameter("r9_2", lower=0.0, upper=1.0, log=False)
r9_3 = UniformFloatHyperparameter("r9_3", lower=0.0, upper=1.0, log=False)

r10_1 = UniformFloatHyperparameter("r10_1", lower=0.0, upper=1.0, log=False)
r10_2 = UniformFloatHyperparameter("r10_2", lower=0.0, upper=1.0, log=False)
r10_3 = UniformFloatHyperparameter("r10_3", lower=0.0, upper=1.0, log=False)

r11_1 = UniformFloatHyperparameter("r11_1", lower=0.0, upper=1.0, log=False)
r11_2 = UniformFloatHyperparameter("r11_2", lower=0.0, upper=1.0, log=False)
r11_3 = UniformFloatHyperparameter("r11_3", lower=0.0, upper=1.0, log=False)

r12_1 = UniformFloatHyperparameter("r12_1", lower=0.0, upper=1.0, log=False)
r12_2 = UniformFloatHyperparameter("r12_2", lower=0.0, upper=1.0, log=False)
r12_3 = UniformFloatHyperparameter("r12_3", lower=0.0, upper=1.0, log=False)

r13_1 = UniformFloatHyperparameter("r13_1", lower=0.0, upper=1.0, log=False)
r13_2 = UniformFloatHyperparameter("r13_2", lower=0.0, upper=1.0, log=False)
r13_3 = UniformFloatHyperparameter("r13_3", lower=0.0, upper=1.0, log=False)

r14_1 = UniformFloatHyperparameter("r14_1", lower=0.0, upper=1.0, log=False)
r14_2 = UniformFloatHyperparameter("r14_2", lower=0.0, upper=1.0, log=False)
r14_3 = UniformFloatHyperparameter("r14_3", lower=0.0, upper=1.0, log=False)


cs.add_hyperparameters(
    [
        root,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        r1_1,
        r1_2,
        r1_3,
        r2_1,
        r2_2,
        r2_3,
        r3_1,
        r3_2,
        r3_3,
        r4_1,
        r4_2,
        r4_3,
        r5_1,
        r5_2,
        r5_3,
        r6_1,
        r6_2,
        r6_3,
        r7_1,
        r7_2,
        r7_3,
        r8_1,
        r8_2,
        r8_3,
        r9_1,
        r9_2,
        r9_3,
        r10_1,
        r10_2,
        r10_3,
        r11_1,
        r11_2,
        r11_3,
        r12_1,
        r12_2,
        r12_3,
        r13_1,
        r13_2,
        r13_3,
        r14_1,
        r14_2,
        r14_3,
    ]
)


# add condition
cs.add_condition(InCondition(x1, root, ["l1"]))
cs.add_condition(InCondition(x2, root, ["ln"]))
cs.add_condition(InCondition(r1_1, root, ["l1"]))
cs.add_condition(InCondition(r1_2, root, ["l1"]))
cs.add_condition(InCondition(r1_3, root, ["l1"]))
cs.add_condition(InCondition(r2_1, root, ["ln"]))
cs.add_condition(InCondition(r2_2, root, ["ln"]))
cs.add_condition(InCondition(r2_3, root, ["ln"]))

cs.add_condition(InCondition(x3, x1, ["l1"]))
cs.add_condition(InCondition(x4, x1, ["ln"]))
cs.add_condition(InCondition(r3_1, x1, ["l1"]))
cs.add_condition(InCondition(r3_2, x1, ["l1"]))
cs.add_condition(InCondition(r3_3, x1, ["l1"]))
cs.add_condition(InCondition(r4_1, x1, ["ln"]))
cs.add_condition(InCondition(r4_2, x1, ["ln"]))
cs.add_condition(InCondition(r4_3, x1, ["ln"]))

cs.add_condition(InCondition(x5, x2, ["l1"]))
cs.add_condition(InCondition(x6, x2, ["ln"]))
cs.add_condition(InCondition(r5_1, x2, ["l1"]))
cs.add_condition(InCondition(r5_2, x2, ["l1"]))
cs.add_condition(InCondition(r5_3, x2, ["l1"]))
cs.add_condition(InCondition(r6_1, x2, ["ln"]))
cs.add_condition(InCondition(r6_2, x2, ["ln"]))
cs.add_condition(InCondition(r6_3, x2, ["ln"]))

cs.add_condition(InCondition(r7_1, x3, ["l1"]))
cs.add_condition(InCondition(r7_2, x3, ["l1"]))
cs.add_condition(InCondition(r7_3, x3, ["l1"]))
cs.add_condition(InCondition(r8_1, x3, ["ln"]))
cs.add_condition(InCondition(r8_2, x3, ["ln"]))
cs.add_condition(InCondition(r8_3, x3, ["ln"]))

cs.add_condition(InCondition(r9_1, x4, ["l1"]))
cs.add_condition(InCondition(r9_2, x4, ["l1"]))
cs.add_condition(InCondition(r9_3, x4, ["l1"]))
cs.add_condition(InCondition(r10_1, x4, ["ln"]))
cs.add_condition(InCondition(r10_2, x4, ["ln"]))
cs.add_condition(InCondition(r10_3, x4, ["ln"]))

cs.add_condition(InCondition(r11_1, x5, ["l1"]))
cs.add_condition(InCondition(r11_2, x5, ["l1"]))
cs.add_condition(InCondition(r11_3, x5, ["l1"]))
cs.add_condition(InCondition(r12_1, x5, ["ln"]))
cs.add_condition(InCondition(r12_2, x5, ["ln"]))
cs.add_condition(InCondition(r12_3, x5, ["ln"]))

cs.add_condition(InCondition(r13_1, x6, ["l1"]))
cs.add_condition(InCondition(r13_2, x6, ["l1"]))
cs.add_condition(InCondition(r13_3, x6, ["l1"]))
cs.add_condition(InCondition(r14_1, x6, ["ln"]))
cs.add_condition(InCondition(r14_2, x6, ["ln"]))
cs.add_condition(InCondition(r14_3, x6, ["ln"]))
#


def main():
    EXP_BASEDIR = "random-multiple"
    logger = logging.getLogger("random-model-compression-vgg16")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args, _ = get_common_cmd_args()
        expid = get_experiment_id(6)
        output_dir = os.path.join(EXP_BASEDIR, expid)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "random-model-compression-vgg16.log")
        setup_logger(logger, log_path)

        def obj_func(cfg, opt_iter):
            logger.info("Starting BO iteration")
            params = cfg2funcparams_multiple(cfg)
            obj_info = setup_and_prune(cmd_args, params, logger, prune_type="multiple")
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
