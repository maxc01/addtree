import sys
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nni
from nni.compression.torch import LevelPruner, AGP_Pruner

from addtree.kernel_utils import build_addtree
from addtree.storage import Storage
from addtree.parameter import Parameter
from addtree.parameter import ParameterNode

logger = logging.getLogger("ModelCompression")

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_name="naive"):
    assert model_name in ["naive", "vgg16", "vgg19"]

    if model_name == "naive":
        return NaiveModel()
    elif model_name == "vgg16":
        return VGG(16)
    else:
        return VGG(19)


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


def build_one_pruner(op_name, method, method_param, model):
    assert op_name in ["conv1", "conv2", "fc1"]
    assert method in ["level", "agp"]

    config_list_level = [
        {"sparsity": method_param, "op_names": [op_name]},
    ]
    config_list_agp = [
        {
            "initial_sparsity": 0,
            "final_sparsity": method_param,
            "start_epoch": 0,
            "end_epoch": 3,
            "frequency": 1,
            "op_names": [op_name],
        }
    ]

    prune_config = {
        "level": LevelPruner(model, config_list_level),
        "agp": AGP_Pruner(model, config_list_agp),
    }

    # pruner = prune_config(params["prune_method"]["_name"])
    pruner = prune_config[method]
    return pruner


def chain_pruners(pruners):
    pass


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
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
            test_loss += F.nll_loss(output, target, reduction="sum").item()
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


def compress(cmd_args, params):
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
    os.makedirs(cmd_args.checkpoints_dir, exist_ok=True)

    model_name = "naive"
    dataset_name = "mnist"
    train_loader, test_loader = get_data_loaders(dataset_name, cmd_args.batch_size)
    model = create_model(model_name).to(device=device)

    if cmd_args.resume_from is not None and os.path.exists(cmd_args.resume_from):
        logger.info("loading checkpoint {} ...".format(cmd_args.resume_from))
        model.load_state_dict(torch.load(cmd_args.resume_from, map_location=device))
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

    op_names = ["conv1", "conv2", "fc1"]
    pruner_name = "_".join([params[op_name]["prune_method"] for op_name in op_names])
    model_path = os.path.join(
        cmd_args.checkpoints_dir,
        "pruned_{}_{}_{}.pth".format(model_name, dataset_name, pruner_name),
    )
    mask_path = os.path.join(
        cmd_args.checkpoints_dir,
        "mask_{}_{}_{}.pth".format(model_name, dataset_name, pruner_name),
    )

    # pruner needs to be initialized from a model not wrapped by DataParallel
    if isinstance(model, nn.DataParallel):
        model = model.module

    optimizer_finetune = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )
    best_top1 = 0

    import ipdb; ipdb.set_trace()
    pruners = []
    for op_name in op_names:
        pruner_thisop = build_one_pruner(
            op_name, params[op_name]["prune_method"], params[op_name]["sparsity"], model
        )
        pruners.append(pruner_thisop)
        model = pruner_thisop.compress()

    if cmd_args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(cmd_args.prune_epochs):
        for pruner in pruners:
            pruner.update_epoch(epoch)
        logger.info("# Epoch {} #".format(epoch))
        train(cmd_args, model, device, train_loader, optimizer_finetune, epoch)
        top1 = test(cmd_args, model, device, test_loader)
        if top1 > best_top1:
            best_top1 = top1
            # pruner.export_model(model_path=model_path, mask_path=mask_path)

    info = {}
    info["value"] = best_top1.item()
    info["value_sigma"] = 1e-5
    return info


def get_cmd_args():
    """ Get parameters from command line """
    parser = argparse.ArgumentParser()
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
    return args


def build_tree():
    root = ParameterNode(Parameter("root", 0))
    x1 = ParameterNode(Parameter("x1", 1))
    x2 = ParameterNode(Parameter("x2", 1))
    root.add_child(x1)
    root.add_child(x2)
    x3 = ParameterNode(Parameter("x3", 1))
    x4 = ParameterNode(Parameter("x4", 1))
    x1.add_child(x3)
    x1.add_child(x4)

    x5 = ParameterNode(Parameter("x5", 1))
    x6 = ParameterNode(Parameter("x6", 1))
    x2.add_child(x5)
    x2.add_child(x6)

    root.finish_add_child()

    return root


def main():
    try:
        cmd_args = get_cmd_args()
        info = compress(cmd_args, params)
        import ipdb; ipdb.set_trace()
        print(info)
        print("Done")
    except KeyboardInterrupt:
        print('Interrupted. You pressed Ctrl-C!!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    main()
