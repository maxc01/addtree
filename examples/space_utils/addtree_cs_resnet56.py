from addtree.parameter import Parameter
from addtree.parameter import ParameterNode


NAME2METHOD = {
    "x1": "l1",
    "x2": "ln",
    "x3": "l1",
    "x4": "ln",
    "x5": "l1",
    "x6": "ln",
    "x7": "l1",
    "x8": "ln",
    "x9": "l1",
    "x10": "ln",
    "x11": "l1",
    "x12": "ln",
    "x13": "l1",
    "x14": "ln",
}


def build_tree():
    root = ParameterNode(Parameter("root", 0))
    x1 = ParameterNode(Parameter("x1", 3))
    x2 = ParameterNode(Parameter("x2", 3))
    x3 = ParameterNode(Parameter("x3", 3))
    x4 = ParameterNode(Parameter("x4", 3))
    x5 = ParameterNode(Parameter("x5", 3))
    x6 = ParameterNode(Parameter("x6", 3))
    x7 = ParameterNode(Parameter("x7", 3))
    x8 = ParameterNode(Parameter("x8", 3))
    x9 = ParameterNode(Parameter("x9", 3))
    x10 = ParameterNode(Parameter("x10", 3))
    x11 = ParameterNode(Parameter("x11", 3))
    x12 = ParameterNode(Parameter("x12", 3))
    x13 = ParameterNode(Parameter("x13", 3))
    x14 = ParameterNode(Parameter("x14", 3))

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
    layer_names = ["layer1", "layer2", "layer3"]
    params = {}
    for layer_name, node in zip(layer_names, path):
        params[layer_name] = {}
        params[layer_name]["prune_method"] = NAME2METHOD[node.name]
        if node.parameter.data.shape == (1,):
            params[layer_name]["amount"] = node.parameter.data.item()
        else:
            params[layer_name]["amount"] = node.parameter.data.tolist()

    return params
