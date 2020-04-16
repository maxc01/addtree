import sys
import json
import os
import logging

from compression_common import setup_logger
from compression_common import setup_and_prune
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id

from smac_utils import cfg2funcparams_multiple
from smac_utils import cs_multiple


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

        logger.info(f"Experiment {expid} starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

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
            fn_path = os.path.join(output_dir, f"random_iter_{opt_iter}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

            return obj_info["value"]

        cs = cs_multiple()
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
