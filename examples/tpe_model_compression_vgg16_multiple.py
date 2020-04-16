import sys
import json
import os
import logging


from compression_common import setup_logger
from compression_common import setup_and_prune
from hyperopt import fmin, tpe, STATUS_OK, Trials
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id

from tpe_utils import cfg2funcparams_multiple
from tpe_utils import get_space_multiple


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

        logger.info(f"Experiment {expid} starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

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
            fn_path = os.path.join(output_dir, "tpe_iter_hists.txt")
            with open(fn_path, "a") as f:
                json.dump(all_info, f)
                f.write("\n")

            return {"loss": obj_info["value"], "status": STATUS_OK}

        space = get_space_multiple()
        trials = Trials()
        best = fmin(
            obj_func,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
            trials=trials,
            show_progressbar=False,
        )
        print(best)
        logger.info("Finish TPE optimization")
        logger.info("Best is: ")
        logger.info(best)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
