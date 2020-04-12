import sys
import json
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms

from models.vgg import VGG

logger = logging.getLogger("ModelCompression")
logger.setLevel(logging.DEBUG)


def setup_logger(filename):
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(filename)
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fileHandler)


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
    x11 = ParameterNode(Parameter("x11", 1))
    x12 = ParameterNode(Parameter("x12", 1))
    x13 = ParameterNode(Parameter("x13", 1))
    x14 = ParameterNode(Parameter("x14", 1))

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
        params[b_name]["prune_method"] = NAME2METHOD[node.name]
        params[b_name]["amount"] = node.parameter.data.item()

    return params


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

    cmd_args: result from argparse
    params: parameters used to build a pruner

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cmd_args.checkpoints_dir, exist_ok=True)

    train_loader, test_loader = get_data_loaders(cmd_args)
    model = VGG("VGG16").to(device=device)

    logger.info("loading pretrained model {} ...".format(cmd_args.pretrained))
    try:
        model.load_state_dict(torch.load(cmd_args.pretrained))
    except FileNotFoundError:
        print("pretrained model doesn't exixt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    test(cmd_args, model, device, test_loader)

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
        logger.info("# Finetune Epoch {} #".format(epoch + 1))
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
        "--n_init",
        type=int,
        default=20,
        metavar="N",
        help="number of random design (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--prune_epochs",
        type=int,
        default=5,
        metavar="N",
        help="training epochs for model pruning (default: 5)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="checkpoints directory",
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
    args, _ = parser.parse_known_args()

    return args


def main():
    try:
        cmd_args = get_cmd_args()

        root = build_tree()
        ss = Storage()
        ker = build_addtree(root)
        n_init = cmd_args.n_init

        setup_logger("compression-vgg16-cifar.log")
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
            logger.info(params)
            logger.info(obj_info)

            all_info = {"iteration": i + 1, "params": params, "obj_info": obj_info}
            fn_path = os.path.join(cmd_args.checkpoints_dir, f"iter_{i+1}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

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
            all_info = {"iteration": i + 1, "params": params, "obj_info": obj_info}
            fn_path = os.path.join(cmd_args.checkpoints_dir, f"iter_{i+1+n_init}.json")
            with open(fn_path, "w") as f:
                json.dump(all_info, f)

    except KeyboardInterrupt:
        print("Interrupted. You pressed Ctrl-C!!!")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
