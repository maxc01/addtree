import sys
import json
import os
import logging


from compression_common import setup_logger
from compression_common import setup_and_prune
from hyperopt import fmin, tpe, STATUS_OK, Trials
from compression_common import get_common_cmd_args
from compression_common import get_experiment_id

from space_utils.tpe_utils import cfg2funcparams_vgg16
from space_utils.tpe_utils import get_space_vgg16
from space_utils.tpe_utils import cfg2funcparams_resnet50
from space_utils.tpe_utils import get_space_resnet50


def main():

    try:
        cmd_args, _ = get_common_cmd_args()

        output_basedir = cmd_args.output_basedir
        model_name = cmd_args.model_name
        if model_name == "vgg16":
            cfg2funcparams = cfg2funcparams_vgg16
            get_space = get_space_vgg16
        elif model_name == "resnet50":
            cfg2funcparams = cfg2funcparams_resnet50
            get_space = get_space_resnet50
        else:
            raise ValueError(f"model name {model_name} is wrong")

        logger = logging.getLogger(f"TPE-{model_name}")
        logger.setLevel(logging.DEBUG)

        expid = get_experiment_id(6)
        output_dir = os.path.join(output_basedir, expid)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "tpe-model-compression-{model_name}.log")
        setup_logger(logger, log_path)

        logger.info(f"Experiment {expid} starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

        def obj_func(cfg):
            logger.info("Starting BO iteration")
            params = cfg2funcparams(cfg)
            obj_info = setup_and_prune(
                cmd_args, params, logger, prune_type="multiple", model_name=model_name
            )
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

        space = get_space()
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
