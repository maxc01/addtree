import logging
import os
import sys

from nas_common import get_common_cmd_args
from nas_common import nas_train_test


def main():

    try:
        cmd_args, _ = get_common_cmd_args()

        model_name = cmd_args.model_name

        logger = logging.getLogger(f"ADDTREE-NAS-{model_name}")
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.info("Training Optimized starts...")
        logger.info("Experiment Configuration:")
        logger.info(vars(cmd_args))

        params = {
            "b1": {"method": "elu", "amount": [0.01, 0.748403908117127]},
            "b2": {"method": "leaky", "amount": [0.01, 0.99]},
            "b3": {"method": "elu", "amount": [0.01, 0.01]},
        }
        obj_info = nas_train_test(
            cmd_args,
            params,
            logger,
            model_name=model_name,
            max_epoch=200,
            lr_ms=[100, 150],
        )
        logger.info(obj_info)
    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
