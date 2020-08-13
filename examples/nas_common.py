import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from compression_common import get_data_loaders
from compression_common import train, test
from models.resnet_cifar10 import resnet20


def get_common_cmd_args():
    """ Get parameters from command line

    common for all NAS , including random, smac, and adtree
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="model name, must be resnet20")
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
        "--lr",
        type=float,
        default=0.1,
        metavar="F",
        help="learning rate in finetuning (defualt: 0.1)",
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


def nas_train_test(
    cmd_args, params, main_logger, model_name, max_epoch=10, lr_ms=[7, 9]
):
    """construct a net based on params, train the net for several epoches, return
    validation accuracy

    """
    assert model_name in ["resnet20"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(cmd_args)
    if model_name == "resnet20":
        model = resnet20(params).to(device=device)

    if isinstance(model, nn.DataParallel):
        model = model.module

    optimizer = optim.SGD(model.parameters(), lr=cmd_args.lr, momentum=0.9)

    best_top1 = 0
    scheduler = MultiStepLR(optimizer, milestones=lr_ms, gamma=0.1)
    main_logger.info("Training {} started...".format(model_name))
    for epoch in range(max_epoch):
        main_logger.info("# Training Epoch {} #".format(epoch + 1))
        train(cmd_args, model, device, train_loader, optimizer, epoch, main_logger)
        main_logger.info("Testing model...")
        top1 = test(cmd_args, model, device, test_loader, main_logger)
        scheduler.step()
        if top1 > best_top1:
            best_top1 = top1

    info = {}
    info["top1"] = best_top1 / 100
    info["value"] = -(best_top1 / 100)
    info["value_sigma"] = 0.01
    return info
