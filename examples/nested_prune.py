
from models.cifar10.vgg import VGG




# def train(model, device, train_loader, optimizer):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(
#                 "{:2.0f}%  Loss {}".format(
#                     100 * batch_idx / len(train_loader), loss.item()
#                 )
#             )




# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.cross_entropy(output, target, reduction="sum").item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset)
#     acc = 100 * correct / len(test_loader.dataset)

#     print("Loss: {}  Accuracy: {}%)\n".format(test_loss, acc))
#     return acc




if __name__ == "__main__":
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        cmd_args = get_cmd_args()
        print("cmd args:")
        print(cmd_args)
        print("=" * 80)
        print("parameters from automl")
        print(tuner_params)
        main(cmd_args, tuner_params)
    except Exception as exception:
        logger.exception(exception)
        raise
