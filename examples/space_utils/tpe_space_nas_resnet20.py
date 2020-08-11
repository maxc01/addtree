from hyperopt import hp


def get_space():
    space = hp.choice(
        "root",
        [
            {
                "method": "elu",
                "r1_1": hp.uniform("r1_1", 0.01, 0.99),
                "r1_2": hp.uniform("r1_2", 0.01, 0.99),
                "x1": hp.choice(
                    "x1",
                    [
                        {
                            "method": "elu",
                            "r3_1": hp.uniform("r3_1", 0.01, 0.99),
                            "r3_2": hp.uniform("r3_2", 0.01, 0.99),
                            "x3": hp.choice(
                                "x3",
                                [
                                    {
                                        "method": "elu",
                                        "r7_1": hp.uniform("r7_1", 0.01, 0.99),
                                        "r7_2": hp.uniform("r7_2", 0.01, 0.99),
                                    },
                                    {
                                        "method": "leaky",
                                        "r8_1": hp.uniform("r8_1", 0.01, 0.99),
                                        "r8_2": hp.uniform("r8_2", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                        {
                            "method": "leaky",
                            "r4_1": hp.uniform("r4_1", 0.01, 0.99),
                            "r4_2": hp.uniform("r4_2", 0.01, 0.99),
                            "x4": hp.choice(
                                "x4",
                                [
                                    {
                                        "method": "elu",
                                        "r9_1": hp.uniform("r9_1", 0.01, 0.99),
                                        "r9_2": hp.uniform("r9_2", 0.01, 0.99),
                                    },
                                    {
                                        "method": "leaky",
                                        "r10_1": hp.uniform("r10_1", 0.01, 0.99),
                                        "r10_2": hp.uniform("r10_2", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                    ],
                ),
            },
            {
                "method": "leaky",
                "r2_1": hp.uniform("r2_1", 0.01, 0.99),
                "r2_2": hp.uniform("r2_2", 0.01, 0.99),
                "x2": hp.choice(
                    "x2",
                    [
                        {
                            "method": "elu",
                            "r5_1": hp.uniform("r5_1", 0.01, 0.99),
                            "r5_2": hp.uniform("r5_2", 0.01, 0.99),
                            "x5": hp.choice(
                                "x5",
                                [
                                    {
                                        "method": "elu",
                                        "r11_1": hp.uniform("r11_1", 0.01, 0.99),
                                        "r11_2": hp.uniform("r11_2", 0.01, 0.99),
                                    },
                                    {
                                        "method": "leaky",
                                        "r12_1": hp.uniform("r12_1", 0.01, 0.99),
                                        "r12_2": hp.uniform("r12_2", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                        {
                            "method": "leaky",
                            "r6_1": hp.uniform("r6_1", 0.01, 0.99),
                            "r6_2": hp.uniform("r6_2", 0.01, 0.99),
                            "x6": hp.choice(
                                "x6",
                                [
                                    {
                                        "method": "elu",
                                        "r13_1": hp.uniform("r13_1", 0.01, 0.99),
                                        "r13_2": hp.uniform("r13_2", 0.01, 0.99),
                                    },
                                    {
                                        "method": "leaky",
                                        "r14_1": hp.uniform("r14_1", 0.01, 0.99),
                                        "r14_2": hp.uniform("r14_2", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                    ],
                ),
            },
        ],
    )

    return space


def testing_cfg():
    return {
        "method": "leaky",
        "r2_1": 0.07067686282157104,
        "r2_2": 0.4890780211983336,
        "x2": {
            "method": "leaky",
            "r6_1": 0.5564595351825452,
            "r6_2": 0.9401366848897367,
            "x6": {
                "method": "elu",
                "r13_1": 0.012747480675794905,
                "r13_2": 0.07904304382102956,
            },
        },
    }


def cfg2funcparams(cfg):
    """
    """

    def _extract_sufix(ks):
        target_ks = []
        for k in ks:
            if k.endswith("_1") or k.endswith("_2"):
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
    params["b1"]["method"] = b1_info[0]
    params["b1"]["amount"] = b1_info[1]

    dd = cfg[b1_info[2][0]]
    b2_info = extract_one_layer(dd)
    params["b2"] = {}
    params["b2"]["method"] = b2_info[0]
    params["b2"]["amount"] = b2_info[1]

    dd = dd[b2_info[2][0]]
    b3_info = extract_one_layer(dd)
    params["b3"] = {}
    params["b3"]["method"] = b3_info[0]
    params["b3"]["amount"] = b3_info[1]

    return params


if __name__ == "__main__":
    import hyperopt.pyll.stochastic

    space = get_space()
    print(hyperopt.pyll.stochastic.sample(space))
