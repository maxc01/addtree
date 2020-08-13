import sys
import json
import os
import logging

from compression_common import setup_logger
from compression_common import get_experiment_id
from nas_common import get_common_cmd_args
from nas_common import nas_train_test


from smac_utils import cfg2funcparams_nas_resnet20
from smac_utils import get_cs_nas_resnet20


def main():

    try:
        cmd_args, _ = get_common_cmd_args()

        output_basedir = cmd_args.output_basedir
        model_name = cmd_args.model_name
        if model_name == "resnet20":
            cfg2funcparams = cfg2funcparams_nas_resnet20
            get_cs = get_cs_nas_resnet20
        else:
            raise ValueError(f"model name {model_name} is wrong")

        logger = logging.getLogger(f"RANDOM-NAS-{model_name}")
        logger.setLevel(logging.DEBUG)

        expid = get_experiment_id(6)
        output_dir = os.path.join(output_basedir, "RANDOM", model_name, expid)
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"RANDOM-NAS-{model_name}.log")
        setup_logger(logger, log_path)

        logger.info(f"Experiment {expid} starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

        def obj_func(cfg):
            logger.info("Starting BO iteration")
            params = cfg2funcparams(cfg)
            obj_info = nas_train_test(cmd_args, params, logger, model_name=model_name)
            logger.info("Finishing BO iteration")
            logger.info(params)
            logger.info(obj_info)

            all_info = {
                "params": params,
                "obj_info": obj_info,
            }
            fn_path = os.path.join(output_dir, "random_iter_hists.txt")
            with open(fn_path, "a") as f:
                json.dump(all_info, f)
                f.write("\n")

            return obj_info["value"]

        cs = get_cs()
        for i in range(100):
            cfg = cs.sample_configuration()
            obj_func(cfg)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
