from addtree.parameter import Parameter
from addtree.parameter import ParameterNode

NAME2METHOD = {
    "x1": "elu",
    "x2": "leaky",
    "x3": "elu",
    "x4": "leaky",
    "x5": "elu",
    "x6": "leaky",
    "x7": "elu",
    "x8": "leaky",
    "x9": "elu",
    "x10": "leaky",
    "x11": "elu",
    "x12": "leaky",
    "x13": "elu",
    "x14": "leaky",
}


def build_tree():
    root = ParameterNode(Parameter("root", 0))
    x1 = ParameterNode(Parameter("x1", 2))
    x2 = ParameterNode(Parameter("x2", 2))
    x3 = ParameterNode(Parameter("x3", 2))
    x4 = ParameterNode(Parameter("x4", 2))
    x5 = ParameterNode(Parameter("x5", 2))
    x6 = ParameterNode(Parameter("x6", 2))
    x7 = ParameterNode(Parameter("x7", 2))
    x8 = ParameterNode(Parameter("x8", 2))
    x9 = ParameterNode(Parameter("x9", 2))
    x10 = ParameterNode(Parameter("x10", 2))
    x11 = ParameterNode(Parameter("x11", 2))
    x12 = ParameterNode(Parameter("x12", 2))
    x13 = ParameterNode(Parameter("x13", 2))
    x14 = ParameterNode(Parameter("x14", 2))

    root.add_child(x1)
    root.add_child(x2)

    x1.add_child(x3)
    x1.add_child(x4)

    x2.add_child(x5)
    x2.add_child(x6)

    x3.add_child(x7)
    x3.add_child(x8)

    x4.add_child(x9)
    x4.add_child(x10)

    x5.add_child(x11)
    x5.add_child(x12)

    x6.add_child(x13)
    x6.add_child(x14)

    root.finish_add_child()

    return root


def path2funcparam(path):
    b_names = ["b1", "b2", "b3"]
    params = {}
    for b_name, node in zip(b_names, path):
        params[b_name] = {}
        params[b_name]["method"] = NAME2METHOD[node.name]
        if node.parameter.data.shape == (1,):
            params[b_name]["amount"] = node.parameter.data.item()
        else:
            params[b_name]["amount"] = node.parameter.data.tolist()

    return params
