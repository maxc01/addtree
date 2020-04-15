import sys
from datetime import datetime
import json
import os
import argparse
import logging

import numpy as np

from compression_common import setup_logger
from compression_common import setup_and_prune
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id

from smac_utils import cfg2funcparams_multiple
from smac_utils import cs_multiple
from smac.configspace import Configuration
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO


def main():
    EXP_BASEDIR = "smac-multiple"
    logger = logging.getLogger("SMAC-model-compression-vgg16-multiple")
    logger.setLevel(logging.DEBUG)

    try:
        cmd_args, _ = get_common_cmd_args()
        expid = get_experiment_id(6)
        output_dir = os.path.join(EXP_BASEDIR, expid)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "SMAC-model-compression-vgg16.log")
        setup_logger(logger, log_path)

        def obj_func(cfg):
            logger.info("Starting BO iteration")
            params = cfg2funcparams_multiple(cfg)
            obj_info = setup_and_prune(cmd_args, params, logger, prune_type="multiple")
            logger.info("Finishing BO iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {
                "params": params,
                "obj_info": obj_info,
            }
            fn_path = os.path.join(output_dir, "smac_iter_hists.txt")
            with open(fn_path, "a") as f:
                json.dump(all_info, f)
                f.write("\n")

            return obj_info["value"]

        # smac default do minimize
        cs = cs_multiple()
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternatively runtime)
                "runcount_limit": 300,  # maximum function evaluations
                "cs": cs,  # configuration space
                "deterministic": "true",
                "initial_incumbent": "LHD",
            }
        )

        # smac = SMAC4HPO(
        #     scenario=scenario,
        #     rng=np.random.RandomState(42),
        #     tae_runner=obj_func,
        #     initial_design=LHDesign,
        #     initial_design_kwargs={"n_configs_x_params": 2},
        # )
        configs = [
            Configuration(configuration_space=cs, values=cs.sample_configuration())
            for i in range(20)
        ]
        import pdb

        pdb.set_trace()
        smac = SMAC4HPO(
            scenario=scenario,
            # initial_design_kwargs={"n_configs_x_params": 10},
            tae_runner=obj_func,
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
