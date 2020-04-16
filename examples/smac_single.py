from smac.configspace import ConfigurationSpace
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.initial_design.latin_hypercube_design import LHDesign
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from ConfigSpace.conditions import InCondition


def testing_cfg_single():
    return {
        "root": "ln",
        "r2": 0.09001820541090022,
        "x2": "ln",
        "r6": 0.7571221577439017,
        "x6": "ln",
        "r14": 0.006530118449628031,
    }


def cfg2funcparams_single(cfg):
    """
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
    params["b2"]["prune_method"] = extract_one_value(["x1", "x2"])
    params["b2"]["amount"] = extract_one_value(["r3", "r4", "r5", "r6"])

    params["b3"] = {}
    params["b3"]["prune_method"] = extract_one_value(["x3", "x4", "x5", "x6"])
    params["b3"]["amount"] = extract_one_value(
        ["r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14",]
    )

    return params


def cs_single():
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
    r1 = UniformFloatHyperparameter("r1", lower=0.01, upper=0.99, log=False)
    r2 = UniformFloatHyperparameter("r2", lower=0.01, upper=0.99, log=False)
    r3 = UniformFloatHyperparameter("r3", lower=0.01, upper=0.99, log=False)
    r4 = UniformFloatHyperparameter("r4", lower=0.01, upper=0.99, log=False)
    r5 = UniformFloatHyperparameter("r5", lower=0.01, upper=0.99, log=False)
    r6 = UniformFloatHyperparameter("r6", lower=0.01, upper=0.99, log=False)
    r7 = UniformFloatHyperparameter("r7", lower=0.01, upper=0.99, log=False)
    r8 = UniformFloatHyperparameter("r8", lower=0.01, upper=0.99, log=False)
    r9 = UniformFloatHyperparameter("r9", lower=0.01, upper=0.99, log=False)
    r10 = UniformFloatHyperparameter("r10", lower=0.01, upper=0.99, log=False)
    r11 = UniformFloatHyperparameter("r11", lower=0.01, upper=0.99, log=False)
    r12 = UniformFloatHyperparameter("r12", lower=0.01, upper=0.99, log=False)
    r13 = UniformFloatHyperparameter("r13", lower=0.01, upper=0.99, log=False)
    r14 = UniformFloatHyperparameter("r14", lower=0.01, upper=0.99, log=False)

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

    return cs
