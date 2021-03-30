import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torch.nn.functional as F


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
batch_size = 100
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)

weights = torch.randn(784, 10, requires_grad=True)

def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view((-1, 28*28))
        outputs = torch.matmul(data, weights)
        softmax = F.softmax(outputs, dim=1)
        pred = softmax.argmax(dim=1, keepdim=True)
        n_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += n_correct

    acc = correct / test_size
    print(" Accuracy on test set", acc)
    return

it = 0
for batch_idx, (data, targets) in enumerate(train_loader):
    # Be sure to start the loop with zeros grad
    if weights.grad is not None:
        weights.grad.zero_()
    data = data.view((-1, 28*28))
    outputs = torch.matmul(data, weights)
    log_softmax = F.log_softmax(outputs, dim=1)
    loss = F.nll_loss(log_softmax, targets)
    print("\rerreur: {}".format(loss), end="")
    loss.backward()
    with torch.no_grad():
        weights -= 0.1*weights.grad
    it += 1
    if it % 100 == 0:
        test(weights, test_loader)
        

