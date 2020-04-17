for i in net.layer1:
    print(i)


import torch

for name, module in net.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        print(name, module.weight.nelement())


layers = [net.layer2, net.layer3, net.layer4]
for layer in layers:
    n_conv2d = 0
    for (_, module) in layer.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            n_conv2d += 1
    print(n_conv2d)


for layer in layers:
    print("=" * 80)
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name)

for name, module in net.named_modules():
    print(name)


### get layer-wise information
# net = 0
layer_info = {}
layers = ["layer1", "layer2", "layer3", "layer4"]
for layer_name in layers:
    block = getattr(net, layer_name)
    layer_info[layer_name] = {}
    layer_info[layer_name]["nelement"] = 0
    layer_info[layer_name]["conv2d_info"] = []

    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layer_info[layer_name]["nelement"] += module.weight.nelement()
            layer_info[layer_name]["conv2d_info"].append((name, module.weight.nelement()))


# get per-layer ratio
n_total = 0
for k, v in layer_info.items():
    n_total += v["nelement"]

for k in layer_info.keys():
    layer_info[k]["ratio"] = layer_info[k]["nelement"] / n_total


