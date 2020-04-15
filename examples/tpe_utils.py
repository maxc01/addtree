from hyperopt import hp


def get_space_multiple():
    space = hp.choice(
        "root",
        [
            {
                "method": "l1",
                "r1_1": hp.uniform("r1_1", 0.01, 0.99),
                "r1_2": hp.uniform("r1_2", 0.01, 0.99),
                "r1_3": hp.uniform("r1_3", 0.01, 0.99),
                "x1": hp.choice(
                    "x1",
                    [
                        {
                            "method": "l1",
                            "r3_1": hp.uniform("r3_1", 0.01, 0.99),
                            "r3_2": hp.uniform("r3_2", 0.01, 0.99),
                            "r3_3": hp.uniform("r3_3", 0.01, 0.99),
                            "x3": hp.choice(
                                "x3",
                                [
                                    {
                                        "method": "l1",
                                        "r7_1": hp.uniform("r7_1", 0.01, 0.99),
                                        "r7_2": hp.uniform("r7_2", 0.01, 0.99),
                                        "r7_3": hp.uniform("r7_3", 0.01, 0.99),
                                    },
                                    {
                                        "method": "ln",
                                        "r8_1": hp.uniform("r8_1", 0.01, 0.99),
                                        "r8_2": hp.uniform("r8_2", 0.01, 0.99),
                                        "r8_3": hp.uniform("r8_3", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                        {
                            "method": "ln",
                            "r4_1": hp.uniform("r4_1", 0.01, 0.99),
                            "r4_2": hp.uniform("r4_2", 0.01, 0.99),
                            "r4_3": hp.uniform("r4_3", 0.01, 0.99),
                            "x4": hp.choice(
                                "x4",
                                [
                                    {
                                        "method": "l1",
                                        "r9_1": hp.uniform("r9_1", 0.01, 0.99),
                                        "r9_2": hp.uniform("r9_2", 0.01, 0.99),
                                        "r9_3": hp.uniform("r9_3", 0.01, 0.99),
                                    },
                                    {
                                        "method": "ln",
                                        "r10_1": hp.uniform("r10_1", 0.01, 0.99),
                                        "r10_2": hp.uniform("r10_2", 0.01, 0.99),
                                        "r10_3": hp.uniform("r10_3", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                    ],
                ),
            },
            {
                "method": "ln",
                "r2_1": hp.uniform("r2_1", 0.01, 0.99),
                "r2_2": hp.uniform("r2_2", 0.01, 0.99),
                "r2_3": hp.uniform("r2_3", 0.01, 0.99),
                "x2": hp.choice(
                    "x2",
                    [
                        {
                            "method": "l1",
                            "r5_1": hp.uniform("r5_1", 0.01, 0.99),
                            "r5_2": hp.uniform("r5_2", 0.01, 0.99),
                            "r5_3": hp.uniform("r5_3", 0.01, 0.99),
                            "x5": hp.choice(
                                "x5",
                                [
                                    {
                                        "method": "l1",
                                        "r11_1": hp.uniform("r11_1", 0.01, 0.99),
                                        "r11_2": hp.uniform("r11_2", 0.01, 0.99),
                                        "r11_3": hp.uniform("r11_3", 0.01, 0.99),
                                    },
                                    {
                                        "method": "ln",
                                        "r12_1": hp.uniform("r12_1", 0.01, 0.99),
                                        "r12_2": hp.uniform("r12_2", 0.01, 0.99),
                                        "r12_3": hp.uniform("r12_3", 0.01, 0.99),
                                    },
                                ],
                            ),
                        },
                        {
                            "method": "ln",
                            "r6_1": hp.uniform("r6_1", 0.01, 0.99),
                            "r6_2": hp.uniform("r6_2", 0.01, 0.99),
                            "r6_3": hp.uniform("r6_3", 0.01, 0.99),
                            "x6": hp.choice(
                                "x6",
                                [
                                    {
                                        "method": "l1",
                                        "r13_1": hp.uniform("r13_1", 0.01, 0.99),
                                        "r13_2": hp.uniform("r13_2", 0.01, 0.99),
                                        "r13_3": hp.uniform("r13_3", 0.01, 0.99),
                                    },
                                    {
                                        "method": "ln",
                                        "r14_1": hp.uniform("r14_1", 0.01, 0.99),
                                        "r14_2": hp.uniform("r14_2", 0.01, 0.99),
                                        "r14_3": hp.uniform("r14_3", 0.01, 0.99),
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


def testing_cfg_multiple():
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


def cfg2funcparams_multiple(cfg):
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


if __name__ == "__main__":
    import hyperopt.pyll.stochastic

    print(hyperopt.pyll.stochastic.sample(space))
