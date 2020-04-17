import sys
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from addtree.kernel_utils import build_addtree
from addtree.storage import Storage
from addtree.parameter import Parameter
from addtree.parameter import ParameterNode
from addtree.acq import optimize_acq, LCB
from models.vgg import VGG
from models.naive import NaiveModel

logger = logging.getLogger("ModelCompression")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler("model-compression.log")
fileHandler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fileHandler.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fileHandler)


def do_prune(model, params):
    names = ["conv1", "conv2", "fc1"]

    def prune_module(module, method, amount):
        if method == "ln":
            logger.addHandler(fileHandler)
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        elif method == "l1":
            prune.l1_unstructured(module, name="weight", amount=amount)
        else:
            raise ValueError(f"{method} is wrong")

    n1 = 0
    n2 = 0
    for name in names:
        module = getattr(model, name)
        prune_module(module, params[name]["prune_method"], params[name]["amount"])
        n1 += torch.sum(module.weight == 0)
        n2 += module.weight.nelement()
    sparsity = float(n1) / n2

    info = {}
    info["sparsity"] = sparsity
    return info


def testing_params():
    params = {}
    params["conv1"] = {}
    params["conv1"]["prune_method"] = "l1"
    params["conv1"]["amount"] = 0.5
    params["conv2"] = {}
    params["conv2"]["prune_method"] = "ln"
    params["conv2"]["amount"] = 0.5
    # NOTE: fc1's method can only be l1
    params["fc1"] = {}
    params["fc1"]["prune_method"] = "l1"
    params["fc1"]["amount"] = 0.3

    return params


NAME2METHOD = {
    "x1": "l1",
    "x2": "ln",
    "x3": "l1",
    "x4": "ln",
    "x5": "l1",
    "x6": "ln",
    "x7": "l1",
    "x8": "l1",
    "x9": "l1",
    "x10": "l1",
}


def build_tree():
    root = ParameterNode(Parameter("root", 0))
    x1 = ParameterNode(Parameter("x1", 1))
    x2 = ParameterNode(Parameter("x2", 1))
    x3 = ParameterNode(Parameter("x3", 1))
    x4 = ParameterNode(Parameter("x4", 1))
    x5 = ParameterNode(Parameter("x5", 1))
    x6 = ParameterNode(Parameter("x6", 1))
    x7 = ParameterNode(Parameter("x7", 1))
    x8 = ParameterNode(Parameter("x8", 1))
    x9 = ParameterNode(Parameter("x9", 1))
    x10 = ParameterNode(Parameter("x10", 1))

    root.add_child(x1)
    root.add_child(x2)
    x1.add_child(x3)
    x1.add_child(x4)

    x2.add_child(x5)
    x2.add_child(x6)

    x3.add_child(x7)
    x4.add_child(x8)
    x5.add_child(x9)
    x6.add_child(x10)

    root.finish_add_child()

    return root


def path2funcparam(path):
    op_names = ["conv1", "conv2", "fc1"]
    params = {}
    for op_name, node in zip(op_names, path):
        params[op_name] = {}
        params[op_name]["prune_method"] = NAME2METHOD[node.name]
        params[op_name]["amount"] = node.parameter.data.item()

    return params


def create_model(model_name="naive"):
    assert model_name in ["naive", "vgg16", "vgg19"]

    if model_name == "naive":
        return NaiveModel()
    elif model_name == "vgg16":
        return VGG("VGG16")
    else:
        return VGG("VGG19")


def get_data_loaders(dataset_name="mnist", batch_size=128):
    assert dataset_name in ["cifar10", "mnist"]

    if dataset_name == "cifar10":
        ds_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.MNIST
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else:
        ds_class = datasets.MNIST
        MEAN, STD = (0.1307,), (0.3081,)

    train_loader = DataLoader(
        ds_class(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ds_class(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
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

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

    return accuracy


def setup_and_prune(cmd_args, params):
    """compress a model

    params: parameters used to build a pruner

    params['conv1']['prune_method'] = 'level'
    params['conv1']['sparsity'] = 0.5
    params['conv2']['prune_method'] = 'level'
    params['conv2']['sparsity'] = 0.5
    params['fc1']['prune_method'] = 'agp'
    params['fc1']['sparsity'] = 0.3

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cmd_args.model_name
    dataset_name = cmd_args.dataset_name
    train_loader, test_loader = get_data_loaders(dataset_name, cmd_args.batch_size)
    model = create_model(model_name).to(device=device)

    if cmd_args.resume_from is not None and os.path.exists(cmd_args.resume_from):
        logger.info("loading checkpoint {} ...".format(cmd_args.resume_from))
        model.load_state_dict(torch.load(cmd_args.resume_from))
        test(cmd_args, model, device, test_loader)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
        )
        if cmd_args.multi_gpu and torch.cuda.device_count():
            model = nn.DataParallel(model)

        logger.info("Model doesn't exist, start training fresh.")
        pretrain_model_path = os.path.join(
            cmd_args.checkpoints_dir,
            "pretrain_{}_{}.pth".format(model_name, dataset_name),
        )
        for epoch in range(cmd_args.pretrain_epochs):
            train(cmd_args, model, device, train_loader, optimizer, epoch)
            test(cmd_args, model, device, test_loader)
        torch.save(model.state_dict(), pretrain_model_path)

    logger.info("start model pruning...")

    # op_names = ["conv1", "conv2", "fc1"]
    # pruner_name = "_".join([params[op_name]["prune_method"] for op_name in op_names])
    # model_path = os.path.join(
    #     cmd_args.checkpoints_dir,
    #     "pruned_{}_{}_{}.pth".format(model_name, dataset_name, pruner_name),
    # )
    # mask_path = os.path.join(
    #     cmd_args.checkpoints_dir,
    #     "mask_{}_{}_{}.pth".format(model_name, dataset_name, pruner_name),
    # )

    if isinstance(model, nn.DataParallel):
        model = model.module

    optimizer_finetune = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )
    best_top1 = 0

    prune_stats = do_prune(model, params)

    if cmd_args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(cmd_args.prune_epochs):
        logger.info("# Finetune Epoch {} #".format(epoch))
        train(cmd_args, model, device, train_loader, optimizer_finetune, epoch)
        top1 = test(cmd_args, model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # export finetuned model

    info = {}
    info["top1"] = best_top1 / 100
    info["sparsity"] = prune_stats["sparsity"]
    info["value"] = -(best_top1 / 100 + prune_stats["sparsity"])
    info["value_sigma"] = 1e-5
    return info


def get_cmd_args():
    """ Get parameters from command line """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="naive",
        help="model type must be naive, vgg16, or vgg19",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mnist",
        help="dataset must be mnist or cifar10",
    )
    parser.add_argument(
        "--n_init", type=int, default=20, help="number of random design"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=10,
        help="training epochs before model pruning",
    )
    parser.add_argument(
        "--prune_epochs", type=int, default=5, help="training epochs for model pruning"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="checkpoints directory",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="pretrained model weights"
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
    args, _ = parser.parse_known_args()

    if args.model_name == "naive":
        assert args.dataset_name == "mnist"
    elif args.model_name in ["vgg16", "vgg19"]:
        assert args.dataset_name == "cifar10"
    return args


def main():
    try:
        cmd_args = get_cmd_args()

        root = build_tree()
        ss = Storage()
        ker = build_addtree(root)
        n_init = cmd_args.n_init
        for i in range(n_init):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1} iteration (Random Design)")
            path = root.random_path(rand_data=True)
            params = path2funcparam(path[1:])
            obj_info = setup_and_prune(cmd_args, params)
            ss.add(
                path.path2vec(root.obs_dim),
                obj_info["value"],
                obj_info["value_sigma"],
                path,
            )
            logger.info(f"Finishing BO {i+1} iteration")
            # TODO: save as json
            logger.info(params)
            logger.info(obj_info)

        for i in range(300):
            logger.info("=" * 50)
            logger.info(f"Starting BO {i+1+n_init} iteration (Optimization)")
            gp = ss.optimize(ker, n_restart=2, verbose=False)
            _, _, x_best, path = optimize_acq(
                LCB, root, gp, ss.Y, root.obs_dim, grid_size=500, nb_seed=5
            )
            path.set_data(x_best)
            params = path2funcparam(path[1:])
            obj_info = setup_and_prune(cmd_args, params)
            ss.add(
                path.path2vec(root.obs_dim),
                obj_info["value"],
                obj_info["value_sigma"],
                path=path,
            )
            logger.info(f"Finishing BO {i+1+n_init} iteration")
            logger.info(params)
            logger.info(obj_info)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
