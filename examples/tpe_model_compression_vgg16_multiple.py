import sys
from datetime import datetime
import json
import os
import argparse
import logging

import numpy as np

from compression_common import setup_logger
from compression_common import setup_and_prune
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id


# Build Configuration Space which defines all parameters and their ranges
space = hp.choice(
    "root",
    [
        {
            "method": "l1",
            "r1_1": hp.uniform("r1_1", 0.01, 0.9999),
            "r1_2": hp.uniform("r1_2", 0.01, 0.9999),
            "r1_3": hp.uniform("r1_3", 0.01, 0.9999),
            "x1": hp.choice(
                "x1",
                [
                    {
                        "method": "l1",
                        "r3_1": hp.uniform("r3_1", 0.01, 0.9999),
                        "r3_2": hp.uniform("r3_2", 0.01, 0.9999),
                        "r3_3": hp.uniform("r3_3", 0.01, 0.9999),
                        "x3": hp.choice(
                            "x3",
                            [
                                {
                                    "method": "l1",
                                    "r7_1": hp.uniform("r7_1", 0.01, 0.9999),
                                    "r7_2": hp.uniform("r7_2", 0.01, 0.9999),
                                    "r7_3": hp.uniform("r7_3", 0.01, 0.9999),
                                },
                                {
                                    "method": "ln",
                                    "r8_1": hp.uniform("r8_1", 0.01, 0.9999),
                                    "r8_2": hp.uniform("r8_2", 0.01, 0.9999),
                                    "r8_3": hp.uniform("r8_3", 0.01, 0.9999),
                                },
                            ],
                        ),
                    },
                    {
                        "method": "ln",
                        "r4_1": hp.uniform("r4_1", 0.01, 0.9999),
                        "r4_2": hp.uniform("r4_2", 0.01, 0.9999),
                        "r4_3": hp.uniform("r4_3", 0.01, 0.9999),
                        "x4": hp.choice(
                            "x4",
                            [
                                {
                                    "method": "l1",
                                    "r9_1": hp.uniform("r9_1", 0.01, 0.9999),
                                    "r9_2": hp.uniform("r9_2", 0.01, 0.9999),
                                    "r9_3": hp.uniform("r9_3", 0.01, 0.9999),
                                },
                                {
                                    "method": "ln",
                                    "r10_1": hp.uniform("r10_1", 0.01, 0.9999),
                                    "r10_2": hp.uniform("r10_2", 0.01, 0.9999),
                                    "r10_3": hp.uniform("r10_3", 0.01, 0.9999),
                                },
                            ],
                        ),
                    },
                ],
            ),
        },
        {
            "method": "ln",
            "r2_1": hp.uniform("r2_1", 0.01, 0.9999),
            "r2_2": hp.uniform("r2_2", 0.01, 0.9999),
            "r2_3": hp.uniform("r2_3", 0.01, 0.9999),
            "x2": hp.choice(
                "x2",
                [
                    {
                        "method": "l1",
                        "r5_1": hp.uniform("r5_1", 0.01, 0.9999),
                        "r5_2": hp.uniform("r5_2", 0.01, 0.9999),
                        "r5_3": hp.uniform("r5_3", 0.01, 0.9999),
                        "x5": hp.choice(
                            "x5",
                            [
                                {
                                    "method": "l1",
                                    "r11_1": hp.uniform("r11_1", 0.01, 0.9999),
                                    "r11_2": hp.uniform("r11_2", 0.01, 0.9999),
                                    "r11_3": hp.uniform("r11_3", 0.01, 0.9999),
                                },
                                {
                                    "method": "ln",
                                    "r12_1": hp.uniform("r12_1", 0.01, 0.9999),
                                    "r12_2": hp.uniform("r12_2", 0.01, 0.9999),
                                    "r12_3": hp.uniform("r12_3", 0.01, 0.9999),
                                },
                            ],
                        ),
                    },
                    {
                        "method": "ln",
                        "r6_1": hp.uniform("r6_1", 0.01, 0.9999),
                        "r6_2": hp.uniform("r6_2", 0.01, 0.9999),
                        "r6_3": hp.uniform("r6_3", 0.01, 0.9999),
                        "x6": hp.choice(
                            "x6",
                            [
                                {
                                    "method": "l1",
                                    "r13_1": hp.uniform("r13_1", 0.01, 0.9999),
                                    "r13_2": hp.uniform("r13_2", 0.01, 0.9999),
                                    "r13_3": hp.uniform("r13_3", 0.01, 0.9999),
                                },
                                {
                                    "method": "ln",
                                    "r14_1": hp.uniform("r14_1", 0.01, 0.9999),
                                    "r14_2": hp.uniform("r14_2", 0.01, 0.9999),
                                    "r14_3": hp.uniform("r14_3", 0.01, 0.9999),
                                },
                            ],
                        ),
                    },
                ],
            ),
        },
    ],
)

import hyperopt.pyll.stochastic

# print(hyperopt.pyll.stochastic.sample(space))


def testing_cfg():
    return {
        "method": "l1",
        "r1_1": 0.4787187980571228,
        "r1_2": 0.47581041577268224,
        "r1_3": 0.5228518172842436,
        "x1": {
            "method": "ln",
            "r4_1": 0.2990475437036728,
            "r4_2": 0.37493878522605695,
            "r4_3": 0.24950009451280078,
            "x4": {
                "method": "ln",
                "r10_1": 0.14595738097427577,
                "r10_2": 0.2275885307456954,
                "r10_3": 0.7269206516328793,
            },
        },
    }


def cfg2funcparams(cfg):
    """
    """

    def _extract_sufix(ks):
        target_ks = []
        for k in ks:
            if k.endswith("_1") or k.endswith("_2") or k.endswith("_3"):
                target_ks.append(k)
        return sorted(target_ks)

    def extract_one_layer(dd):
        target_ks = _extract_sufix(dd.keys())
        values = [dd[k] for k in target_ks]
        extra = [k for k in dd.keys() if k.startswith("x")]
        return dd["method"], values, extra

    params = {}

    b1_info = extract_one_layer(cfg)
    params["b1"] = {}
    params["b1"]["prune_method"] = b1_info[0]
    params["b1"]["amount"] = b1_info[1]

    dd = cfg[b1_info[2][0]]
    b2_info = extract_one_layer(dd)
    params["b2"] = {}
    params["b2"]["prune_method"] = b2_info[0]
    params["b2"]["amount"] = b2_info[1]

    dd = dd[b2_info[2][0]]
    b3_info = extract_one_layer(dd)
    params["b3"] = {}
    params["b3"]["prune_method"] = b3_info[0]
    params["b3"]["amount"] = b3_info[1]

    return params


def main():
    EXP_BASEDIR = "tpe-multiple"
    logger = logging.getLogger("tpe-model-compression-vgg16-multiple")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args, _ = get_common_cmd_args()
        expid = get_experiment_id(6)
        output_dir = os.path.join(EXP_BASEDIR, expid)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "tpe-model-compression-vgg16.log")
        setup_logger(logger, log_path)

        def obj_func(cfg):
            logger.info("Starting BO iteration")
            params = cfg2funcparams(cfg)
            obj_info = setup_and_prune(cmd_args, params, logger, prune_type="multiple")
            logger.info("Finishing BO iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {
                "params": params,
                "obj_info": obj_info,
            }
            fn_path = os.path.join(output_dir, "tpe_iter_hists.txt")
            with open(fn_path, "a") as f:
                json.dump(all_info, f)
                f.write("\n")

            return {"loss": obj_info["value"], "status": STATUS_OK}

        trials = Trials()
        best = fmin(
            obj_func, space=space, algo=tpe.suggest, max_evals=300, trials=trials
        )
        print(best)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
