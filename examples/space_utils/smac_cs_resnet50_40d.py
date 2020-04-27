from smac.configspace import ConfigurationSpace
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.initial_design.latin_hypercube_design import LHDesign
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    Constant,
)
from ConfigSpace.conditions import InCondition


def testing_cfg():
    return {
        "root": "l1",
        "r1_1": 0.5745985057653132,
        "r1_2": 0.3914810073523109,
        "r1_3": 0.2976723864528515,
        "r1_4": 0.2212555286365667,
        "x1": "ln",
        "r4_1": 0.2268537928350621,
        "r4_2": 0.02334693439675707,
        "r4_3": 0.30347993498438,
        "r4_4": 0.6306302847154929,
        "x4": "ln",
        "r10_1": 0.5,
        "r10_2": 0.5,
        "r10_3": 0.5,
        "r10_4": 0.5,
    }


def cfg2funcparams(cfg):
    def extract_one_value(keys):
        """this function extract ONE value with key in keys from cfg
        keys: e.g. ["x1","x2"]
        """

        for k in keys:
            if k in cfg.keys():
                return cfg[k]
        raise RuntimeError("key not exist")

    def extract_using_prefix(keys):
        """this function extract ONE value with key in keys from cfg
        keys: e.g. ["r1","r2"]
        """

        cfg_prefix = set([i.split("_")[0] for i in cfg.keys()])
        # e.g. r1->[r1_1, r1_2, r1_3]
        for k in keys:
            if k in cfg_prefix:
                return [cfg[k + "_1"], cfg[k + "_2"], cfg[k + "_3"], cfg[k + "_4"]]
        raise RuntimeError("key not exist")

    params = {}
    params["layer2"] = {}
    params["layer2"]["prune_method"] = cfg["root"]
    params["layer2"]["amount"] = extract_using_prefix(["r1", "r2"])

    params["layer3"] = {}
    params["layer3"]["prune_method"] = extract_one_value(["x1", "x2"])
    params["layer3"]["amount"] = extract_using_prefix(["r3", "r4", "r5", "r6"])

    params["layer4"] = {}
    params["layer4"]["prune_method"] = extract_one_value(["x3", "x4", "x5", "x6"])
    params["layer4"]["amount"] = extract_using_prefix(
        ["r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14",]
    )

    return params


def get_cs():
    cs = ConfigurationSpace()

    root = CategoricalHyperparameter("root", choices=["l1", "ln"])
    x1 = CategoricalHyperparameter("x1", choices=["l1", "ln"])
    x2 = CategoricalHyperparameter("x2", choices=["l1", "ln"])
    x3 = CategoricalHyperparameter("x3", choices=["l1", "ln"])
    x4 = CategoricalHyperparameter("x4", choices=["l1", "ln"])
    x5 = CategoricalHyperparameter("x5", choices=["l1", "ln"])
    x6 = CategoricalHyperparameter("x6", choices=["l1", "ln"])

    # r1 is the data associated in x1
    r1_1 = UniformFloatHyperparameter("r1_1", lower=0.01, upper=0.99, log=False)
    r1_2 = UniformFloatHyperparameter("r1_2", lower=0.01, upper=0.99, log=False)
    r1_3 = UniformFloatHyperparameter("r1_3", lower=0.01, upper=0.99, log=False)
    r1_4 = UniformFloatHyperparameter("r1_4", lower=0.01, upper=0.99, log=False)

    r2_1 = UniformFloatHyperparameter("r2_1", lower=0.01, upper=0.99, log=False)
    r2_2 = UniformFloatHyperparameter("r2_2", lower=0.01, upper=0.99, log=False)
    r2_3 = UniformFloatHyperparameter("r2_3", lower=0.01, upper=0.99, log=False)
    r2_4 = UniformFloatHyperparameter("r2_4", lower=0.01, upper=0.99, log=False)

    r3_1 = UniformFloatHyperparameter("r3_1", lower=0.01, upper=0.99, log=False)
    r3_2 = UniformFloatHyperparameter("r3_2", lower=0.01, upper=0.99, log=False)
    r3_3 = UniformFloatHyperparameter("r3_3", lower=0.01, upper=0.99, log=False)
    r3_4 = UniformFloatHyperparameter("r3_4", lower=0.01, upper=0.99, log=False)

    r4_1 = UniformFloatHyperparameter("r4_1", lower=0.01, upper=0.99, log=False)
    r4_2 = UniformFloatHyperparameter("r4_2", lower=0.01, upper=0.99, log=False)
    r4_3 = UniformFloatHyperparameter("r4_3", lower=0.01, upper=0.99, log=False)
    r4_4 = UniformFloatHyperparameter("r4_4", lower=0.01, upper=0.99, log=False)

    r5_1 = UniformFloatHyperparameter("r5_1", lower=0.01, upper=0.99, log=False)
    r5_2 = UniformFloatHyperparameter("r5_2", lower=0.01, upper=0.99, log=False)
    r5_3 = UniformFloatHyperparameter("r5_3", lower=0.01, upper=0.99, log=False)
    r5_4 = UniformFloatHyperparameter("r5_4", lower=0.01, upper=0.99, log=False)

    r6_1 = UniformFloatHyperparameter("r6_1", lower=0.01, upper=0.99, log=False)
    r6_2 = UniformFloatHyperparameter("r6_2", lower=0.01, upper=0.99, log=False)
    r6_3 = UniformFloatHyperparameter("r6_3", lower=0.01, upper=0.99, log=False)
    r6_4 = UniformFloatHyperparameter("r6_4", lower=0.01, upper=0.99, log=False)

    r7_1 = Constant("r7_1", 0.5)
    r7_2 = Constant("r7_2", 0.5)
    r7_3 = Constant("r7_3", 0.5)
    r7_4 = Constant("r7_4", 0.5)

    r8_1 = Constant("r8_1", 0.5)
    r8_2 = Constant("r8_2", 0.5)
    r8_3 = Constant("r8_3", 0.5)
    r8_4 = Constant("r8_4", 0.5)

    r9_1 = Constant("r9_1", 0.5)
    r9_2 = Constant("r9_2", 0.5)
    r9_3 = Constant("r9_3", 0.5)
    r9_4 = Constant("r9_4", 0.5)

    r10_1 = Constant("r10_1", 0.5)
    r10_2 = Constant("r10_2", 0.5)
    r10_3 = Constant("r10_3", 0.5)
    r10_4 = Constant("r10_4", 0.5)

    r11_1 = Constant("r11_1", 0.5)
    r11_2 = Constant("r11_2", 0.5)
    r11_3 = Constant("r11_3", 0.5)
    r11_4 = Constant("r11_4", 0.5)

    r12_1 = Constant("r12_1", 0.5)
    r12_2 = Constant("r12_2", 0.5)
    r12_3 = Constant("r12_3", 0.5)
    r12_4 = Constant("r12_4", 0.5)

    r13_1 = UniformFloatHyperparameter("r13_1", lower=0.01, upper=0.99, log=False)
    r13_2 = UniformFloatHyperparameter("r13_2", lower=0.01, upper=0.99, log=False)
    r13_3 = UniformFloatHyperparameter("r13_3", lower=0.01, upper=0.99, log=False)
    r13_4 = UniformFloatHyperparameter("r13_4", lower=0.01, upper=0.99, log=False)

    r14_1 = UniformFloatHyperparameter("r14_1", lower=0.01, upper=0.99, log=False)
    r14_2 = UniformFloatHyperparameter("r14_2", lower=0.01, upper=0.99, log=False)
    r14_3 = UniformFloatHyperparameter("r14_3", lower=0.01, upper=0.99, log=False)
    r14_4 = UniformFloatHyperparameter("r14_4", lower=0.01, upper=0.99, log=False)

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
            r1_4,
            r2_1,
            r2_2,
            r2_3,
            r2_4,
            r3_1,
            r3_2,
            r3_3,
            r3_4,
            r4_1,
            r4_2,
            r4_3,
            r4_4,
            r5_1,
            r5_2,
            r5_3,
            r5_4,
            r6_1,
            r6_2,
            r6_3,
            r6_4,
            r7_1,
            r7_2,
            r7_3,
            r7_4,
            r8_1,
            r8_2,
            r8_3,
            r8_4,
            r9_1,
            r9_2,
            r9_3,
            r9_4,
            r10_1,
            r10_2,
            r10_3,
            r10_4,
            r11_1,
            r11_2,
            r11_3,
            r11_4,
            r12_1,
            r12_2,
            r12_3,
            r12_4,
            r13_1,
            r13_2,
            r13_3,
            r13_4,
            r14_1,
            r14_2,
            r14_3,
            r14_4,
        ]
    )

    # add condition
    cs.add_condition(InCondition(x1, root, ["l1"]))
    cs.add_condition(InCondition(x2, root, ["ln"]))
    cs.add_condition(InCondition(r1_1, root, ["l1"]))
    cs.add_condition(InCondition(r1_2, root, ["l1"]))
    cs.add_condition(InCondition(r1_3, root, ["l1"]))
    cs.add_condition(InCondition(r1_4, root, ["l1"]))
    cs.add_condition(InCondition(r2_1, root, ["ln"]))
    cs.add_condition(InCondition(r2_2, root, ["ln"]))
    cs.add_condition(InCondition(r2_3, root, ["ln"]))
    cs.add_condition(InCondition(r2_4, root, ["ln"]))

    cs.add_condition(InCondition(x3, x1, ["l1"]))
    cs.add_condition(InCondition(x4, x1, ["ln"]))
    cs.add_condition(InCondition(r3_1, x1, ["l1"]))
    cs.add_condition(InCondition(r3_2, x1, ["l1"]))
    cs.add_condition(InCondition(r3_3, x1, ["l1"]))
    cs.add_condition(InCondition(r3_4, x1, ["l1"]))
    cs.add_condition(InCondition(r4_1, x1, ["ln"]))
    cs.add_condition(InCondition(r4_2, x1, ["ln"]))
    cs.add_condition(InCondition(r4_3, x1, ["ln"]))
    cs.add_condition(InCondition(r4_4, x1, ["ln"]))

    cs.add_condition(InCondition(x5, x2, ["l1"]))
    cs.add_condition(InCondition(x6, x2, ["ln"]))
    cs.add_condition(InCondition(r5_1, x2, ["l1"]))
    cs.add_condition(InCondition(r5_2, x2, ["l1"]))
    cs.add_condition(InCondition(r5_3, x2, ["l1"]))
    cs.add_condition(InCondition(r5_4, x2, ["l1"]))
    cs.add_condition(InCondition(r6_1, x2, ["ln"]))
    cs.add_condition(InCondition(r6_2, x2, ["ln"]))
    cs.add_condition(InCondition(r6_3, x2, ["ln"]))
    cs.add_condition(InCondition(r6_4, x2, ["ln"]))

    cs.add_condition(InCondition(r7_1, x3, ["l1"]))
    cs.add_condition(InCondition(r7_2, x3, ["l1"]))
    cs.add_condition(InCondition(r7_3, x3, ["l1"]))
    cs.add_condition(InCondition(r7_4, x3, ["l1"]))
    cs.add_condition(InCondition(r8_1, x3, ["ln"]))
    cs.add_condition(InCondition(r8_2, x3, ["ln"]))
    cs.add_condition(InCondition(r8_3, x3, ["ln"]))
    cs.add_condition(InCondition(r8_4, x3, ["ln"]))

    cs.add_condition(InCondition(r9_1, x4, ["l1"]))
    cs.add_condition(InCondition(r9_2, x4, ["l1"]))
    cs.add_condition(InCondition(r9_3, x4, ["l1"]))
    cs.add_condition(InCondition(r9_4, x4, ["l1"]))
    cs.add_condition(InCondition(r10_1, x4, ["ln"]))
    cs.add_condition(InCondition(r10_2, x4, ["ln"]))
    cs.add_condition(InCondition(r10_3, x4, ["ln"]))
    cs.add_condition(InCondition(r10_4, x4, ["ln"]))

    cs.add_condition(InCondition(r11_1, x5, ["l1"]))
    cs.add_condition(InCondition(r11_2, x5, ["l1"]))
    cs.add_condition(InCondition(r11_3, x5, ["l1"]))
    cs.add_condition(InCondition(r11_4, x5, ["l1"]))
    cs.add_condition(InCondition(r12_1, x5, ["ln"]))
    cs.add_condition(InCondition(r12_2, x5, ["ln"]))
    cs.add_condition(InCondition(r12_3, x5, ["ln"]))
    cs.add_condition(InCondition(r12_4, x5, ["ln"]))

    cs.add_condition(InCondition(r13_1, x6, ["l1"]))
    cs.add_condition(InCondition(r13_2, x6, ["l1"]))
    cs.add_condition(InCondition(r13_3, x6, ["l1"]))
    cs.add_condition(InCondition(r13_4, x6, ["l1"]))
    cs.add_condition(InCondition(r14_1, x6, ["ln"]))
    cs.add_condition(InCondition(r14_2, x6, ["ln"]))
    cs.add_condition(InCondition(r14_3, x6, ["ln"]))
    cs.add_condition(InCondition(r14_4, x6, ["ln"]))

    return cs


print("resnet50")
cs = get_cs()
print(len(cs.get_hyperparameters()))
