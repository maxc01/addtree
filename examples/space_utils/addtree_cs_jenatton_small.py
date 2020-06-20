from addtree.parameter import Parameter
from addtree.parameter import ParameterNode

NAME2METHOD = {
    "x2": "left",
    "x3": "right",
    "x4": "left",
    "x5": "right",
    "x6": "left",
    "x7": "right",
}


def build_tree():
    root = ParameterNode(Parameter("root", 0))
    x2 = ParameterNode(Parameter("x2", 1))
    x3 = ParameterNode(Parameter("x3", 1))
    x4 = ParameterNode(Parameter("x4", 1))
    x5 = ParameterNode(Parameter("x5", 1))
    x6 = ParameterNode(Parameter("x6", 1))
    x7 = ParameterNode(Parameter("x7", 1))

    root.add_child(x2)
    root.add_child(x3)
    x2.add_child(x4)
    x2.add_child(x5)
    x3.add_child(x6)
    x3.add_child(x7)

    root.finish_add_child()

    return root


def path2funcparam(path):
    layer_names = ["L1", "L2"]
    params = {}
    for layer_name, node in zip(layer_names, path):
        params[layer_name] = {}
        params[layer_name]["cat_value"] = NAME2METHOD[node.name]
        if node.parameter.data.shape == (1,):
            params[layer_name]["cont_value"] = node.parameter.data.item()
        else:
            params[layer_name]["amount"] = node.parameter.data.tolist()

    return params
