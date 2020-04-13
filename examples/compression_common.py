import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune


from models.vgg import VGG


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


def setup_and_prune(cmd_args, params, main_logger, prune_type="single"):
    """compress a model

    cmd_args: result from argparse
    params: parameters used to build a pruner

    """
    assert prune_type in ["single", "multiple"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_data_loaders(cmd_args)
    model = VGG("VGG16").to(device=device)

    main_logger.info("Loading pretrained model {} ...".format(cmd_args.pretrained))
    try:
        model.load_state_dict(torch.load(cmd_args.pretrained))
    except FileNotFoundError:
        print("Pretrained model doesn't exixt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    main_logger.info("Testing pretrained model...")
    test(cmd_args, model, device, test_loader, main_logger)

    main_logger.info("start model pruning...")

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

    if prune_type == "single":
        prune_stats = do_prune(model, params)
    else:
        prune_stats = do_prune_multiple(model, params)

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
        top1 = test(cmd_args, model, device, test_loader, main_logger)
        if top1 > best_top1:
            best_top1 = top1
            # export finetuned model

    info = {}
    info["top1"] = best_top1 / 100
    info["sparsity"] = prune_stats["sparsity"]
    info["value"] = -(best_top1 / 100 + prune_stats["sparsity"])
    info["value_sigma"] = 1e-5
    return info
