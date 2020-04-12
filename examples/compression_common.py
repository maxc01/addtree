import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


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
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

    return accuracy
