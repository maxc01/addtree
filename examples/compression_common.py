import logging
from datetime import datetime
import os
import sys
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune


from models.vgg import VGG
from models.resnet import ResNet50


def testing_params():
    params = {}
    params["b1"] = {}
    params["b1"]["prune_method"] = "l1"
    params["b1"]["amount"] = 0.5
    params["b2"] = {}
    params["b2"]["prune_method"] = "ln"
    params["b2"]["amount"] = 0.5
    params["b3"] = {}
    params["b3"]["prune_method"] = "l1"
    params["b3"]["amount"] = 0.3

    return params


def testing_params_multiple():
    params = {}
    params["b1"] = {}
    params["b1"]["prune_method"] = "l1"
    params["b1"]["amount"] = [0.5, 0.5, 0.5]
    params["b2"] = {}
    params["b2"]["prune_method"] = "ln"
    params["b2"]["amount"] = [0.1, 0.2, 0.3]
    params["b3"] = {}
    params["b3"]["prune_method"] = "l1"
    params["b3"]["amount"] = [0.1, 0.2, 0.3]

    return params


def testing_params_multiple_resnet50():
    params = {}
    params["layer2"] = {}
    params["layer2"]["prune_method"] = "l1"
    params["layer2"]["amount"] = [0.5, 0.5, 0.5, 0.5]
    params["layer3"] = {}
    params["layer3"]["prune_method"] = "ln"
    params["layer3"]["amount"] = [0.1, 0.2, 0.3, 0.5]
    params["layer4"] = {}
    params["layer4"]["prune_method"] = "l1"
    params["layer4"]["amount"] = [0.1, 0.2, 0.3, 0.5]

    return params


def do_prune(model, params):
    """ prune model using params
    params: ref. testing_params()
    """

    def prune_module(module, method, amount):
        if method == "ln":
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        elif method == "l1":
            prune.l1_unstructured(module, name="weight", amount=amount)
        else:
            raise ValueError(f"{method} is wrong")

    fea_idx = {
        "b1": [14, 17, 20],
        "b2": [24, 27, 30],
        "b3": [34, 37, 40],
    }
    n1 = 0
    n2 = 0
    for b_name in ["b1", "b2", "b3"]:
        method = params[b_name]["prune_method"]
        amount = params[b_name]["amount"]
        for fid in fea_idx[b_name]:
            module = model.features[fid]
            prune_module(module, method, amount)
            n1 += torch.sum(module.weight == 0)
            n2 += module.weight.nelement()
    sparsity = float(n1) / n2

    info = {}
    info["sparsity"] = sparsity
    return info


def do_prune_multiple(model, params):
    """ layers in each block have different prune parameters
    params: ref. testing_params_multiple()
    """

    def prune_module(module, method, amount):
        if method == "ln":
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        elif method == "l1":
            prune.l1_unstructured(module, name="weight", amount=amount)
        else:
            raise ValueError(f"{method} is wrong")

    fea_idx = {
        "b1": [14, 17, 20],
        "b2": [24, 27, 30],
        "b3": [34, 37, 40],
    }
    n1 = 0
    n2 = 0
    for b_name in ["b1", "b2", "b3"]:
        method = params[b_name]["prune_method"]
        amount = params[b_name]["amount"]
        for idx, fid in enumerate(fea_idx[b_name]):
            module = model.features[fid]
            prune_module(module, method, amount[idx])
            n1 += torch.sum(module.weight == 0)
            n2 += module.weight.nelement()
    sparsity = float(n1) / n2

    info = {}
    info["sparsity"] = sparsity
    return info


def do_prune_multiple_resnet50(model, params):
    """ layers in each block have different prune parameters
    params: ref. testing_params_multiple()
    """

    def prune_module(module, method, amount):
        """prune a conv2d
        """
        if method == "ln":
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        elif method == "l1":
            prune.l1_unstructured(module, name="weight", amount=amount)
        else:
            raise ValueError(f"{method} is wrong")

    def prune_layer(layer, method, amounts):
        """prune a layer, which has many Conv2d, using amounts dim of amounts must be
        equal to 4, coresponding to conv1,conv2,conv3,and shortcut

        All Conv2d in this layer share the same prune method.

        """
        n1 = 0
        n2 = 0
        for (name, module) in layer.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # four cases
                if "conv1" in name:
                    prune_module(module, method, amounts[0])
                elif "conv2" in name:
                    prune_module(module, method, amounts[1])
                elif "conv3" in name:
                    prune_module(module, method, amounts[2])
                elif "shortcut" in name:
                    prune_module(module, method, amounts[3])
                else:
                    raise ValueError(f"{name} is strange")

                n1 += int(torch.sum(module.weight == 0))
                n2 += int(module.weight.nelement())
        return n1, n2

    # we don't prune layer1, because layer1 has little weights, ref. the
    # following information
    # layer_info = {
    #     "layer1": {"nelement": 212992, "ratio": 0.009084556254367574},
    #     "layer2": {"nelement": 1212416, "ratio": 0.0517120894479385},
    #     "layer3": {"nelement": 7077888, "ratio": 0.3018867924528302},
    #     "layer4": {"nelement": 14942208, "ratio": 0.6373165618448637},
    # }

    n1 = 0
    n2 = 0
    layers = [model.layer2, model.layer3, model.layer4]
    for layer_name, layer in zip(["layer2", "layer3", "layer4"], layers):
        method = params[layer_name]["prune_method"]
        amount = params[layer_name]["amount"]
        _n1, _n2 = prune_layer(layer, method, amount)
        n1 += _n1
        n2 += _n2

    sparsity = float(n1) / n2

    info = {}
    info["sparsity"] = sparsity
    return info


def train(args, model, device, train_loader, optimizer, epoch, main_logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            main_logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader, main_logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    main_logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

    return accuracy


def get_data_loaders(args):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data", train=True, download=True, transform=transform_train
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("../data", train=False, transform=transform_test),
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    return train_loader, test_loader


def setup_logger(main_logger, filename):
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    main_logger.addHandler(ch)
    main_logger.addHandler(fileHandler)


def setup_and_prune(
    cmd_args, params, main_logger, *, prune_type="single", model_name=None
):
    """compress a model

    cmd_args: result from argparse
    params: parameters used to build a pruner

    """
    assert prune_type in ["single", "multiple"]
    assert model_name in ["vgg16", "resnet50"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders(cmd_args)
    if model_name == "vgg16":
        model = VGG("VGG16").to(device=device)
    elif model_name == "resnet50":
        model = ResNet50().to(device=device)
    else:
        raise ValueError(f"Model {model_name} is wrong.")

    main_logger.info("Loading pretrained model {} ...".format(cmd_args.pretrained))
    try:
        model.load_state_dict(torch.load(cmd_args.pretrained))
    except FileNotFoundError:
        print("Pretrained model doesn't exist")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    main_logger.info("Testing pretrained model...")
    test(cmd_args, model, device, test_loader, main_logger)

    main_logger.info("start model pruning...")

    if isinstance(model, nn.DataParallel):
        model = model.module

    optimizer_finetune = torch.optim.SGD(
        model.parameters(), lr=cmd_args.finetune_lr, momentum=0.9, weight_decay=1e-4
    )
    best_top1 = 0

    if model_name == "vgg16":
        if prune_type == "single":
            prune_stats = do_prune(model, params)
        else:
            prune_stats = do_prune_multiple(model, params)
    elif model_name == "resnet50":
        if prune_type == "multiple":
            prune_stats = do_prune_multiple_resnet50(model, params)
        else:
            raise ValueError(f"prune type {prune_type} is not implemented")

    if cmd_args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(cmd_args.prune_epochs):
        main_logger.info("# Finetune Epoch {} #".format(epoch + 1))
        train(
            cmd_args,
            model,
            device,
            train_loader,
            optimizer_finetune,
            epoch,
            main_logger,
        )
        main_logger.info("Testing finetuned model after pruning...")
        top1 = test(cmd_args, model, device, test_loader, main_logger)
        if top1 > best_top1:
            best_top1 = top1
            # export finetuned model

    info = {}
    info["top1"] = best_top1 / 100
    info["sparsity"] = prune_stats["sparsity"]
    info["value"] = -(best_top1 / 100 + prune_stats["sparsity"])
    info["value_sigma"] = 0.25
    return info


def search_constant_params(kernel):
    pnames = []
    for pname in kernel.get_parameter_names():
        if "log_constant" in pname:
            pnames.append(pname)
    return pnames


def freeze_constant_params(kernel):
    for pname in search_constant_params(kernel):
        kernel.freeze_parameter(pname)


def get_common_cmd_args():
    """ Get parameters from command line

    common for all compression, including random, smac, and adtree
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", type=str, help="model name, must be vgg16 or resnet50"
    )
    parser.add_argument(
        "output_basedir", type=str, help="output base directory",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=50,
        metavar="N",
        help="number of random design (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--prune_epochs",
        type=int,
        default=2,
        metavar="N",
        help="training epochs for model pruning (default: 2)",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=0.001,
        metavar="F",
        help="learning rate in finetuning (defualt: 0.001)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="pretrained model weights"
    )
    parser.add_argument(
        "--multi_gpu", action="store_true", help="Use multiple GPUs for training"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args, extra = parser.parse_known_args()

    return args, extra


def char_map(index):
    if index < 26:
        return index + 97
    elif index < 52:
        return index - 26 + 65
    else:
        return index - 52 + 48


def unique_string(n):
    """generate unique n-length string

    n: length of string
    """
    from functools import reduce

    if n == 0:
        return ""
    byte_len = math.ceil(math.log2(52) + math.log2(62) * (n - 1))
    num = reduce(lambda x, y: x * 256 + y, os.urandom(byte_len), 0)
    codes = []
    codes.append(char_map(num % 52))
    num = math.floor(num / 52)
    for i in range(1, n):
        codes.append(char_map(num % 62))
        num = math.floor(num / 62)

    return "".join(map(chr, codes))


def get_experiment_id(n):
    expid = datetime.now().strftime("%Y-%m-%d-%H:%M:%S-") + unique_string(n)
    return expid
