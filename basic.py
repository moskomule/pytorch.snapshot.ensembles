from math import pi
from math import cos

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import visdom
import numpy as np

# variables
cuda = torch.cuda.is_available()
batch_size = 64

# load data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/cifar10', train=True, download=True,
                     transform=transform),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/cifar10', train=False, transform=transform),
    batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=5,
                               stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.dense1 = nn.Linear(in_features=4 * 64, out_features=128)
        self.dense1_bn = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = x.view(-1, 4 * 64)
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.log_softmax(self.dense2(x)) # NLL loss expects log_softmax
        return x


def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2


    """
    during an iteration a batch goes forward and backward  
    while during an epoch every batch of a data set is processed
    """
    snapshots = []
    _lr_list, _loss_list = [], []
    count = 0
    epochs_per_cycle = epochs // cycles
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    for i in range(cycles):

        for j in range(epochs_per_cycle):
            _epoch_loss = 0

            lr = proposed_lr(initial_lr, j, epochs_per_cycle)
            optimizer.state_dict()["param_groups"][0]["lr"] = lr

            for batch_idx, (data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                _epoch_loss += loss.data[0]/len(train_loader)
                loss.backward()
                optimizer.step()

            _lr_list.append(lr)
            _loss_list.append(_epoch_loss)
            count += 1

            if vis is not None and j % 10 == 0:
                vis.line(np.array(_lr_list), np.arange(count), win="lr",
                         opts=dict(title="learning rate",
                                   xlabel="epochs",
                                   ylabel="learning rate (s.e.)"))
                vis.line(np.array(_loss_list), np.arange(count),  win="loss",
                         opts=dict(title="loss",
                                   xlabel="epochs",
                                   ylabel="training loss (s.e.)"))

        snapshots.append(model.state_dict())
    return snapshots


def test_se(Model, snapshots, use_model_num):
    index = len(snapshots) - use_model_num
    snapshots = snapshots[index:]
    model_list = [Model() for _ in snapshots]

    for model, weight in zip(model_list, snapshots):
        model.load_state_dict(weight)
        model.eval()
        if cuda:
            model.cuda()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output_list = [model(data).unsqueeze(0) for model in model_list]
        output = torch.mean(torch.cat(output_list), 0).squeeze()
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))

    return test_loss


def train_normal(model, epochs, vis=None):

    optimizer = optim.Adam(model.parameters())
    _lr_list, _loss_list = [], []
    for epoch in range(epochs):
        _epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            _epoch_loss += loss.data[0] / len(train_loader)
            loss.backward()
            optimizer.step()

        _loss_list.append(_epoch_loss)
        _lr_list.append(optimizer.state_dict()["param_groups"][0]["lr"])

        if vis is not None and epoch % 10 == 0:
            vis.line(np.array(_lr_list), np.arange(epoch+1), win="lr_n",
                     opts=dict(title="learning rate",
                               xlabel="epochs",
                               ylabel="learning rate (normal)"))
            vis.line(np.array(_loss_list), np.arange(epoch+1), win="loss_n",
                     opts=dict(title="loss",
                               xlabel="epochs",
                               ylabel="training loss (normal)"))

    return model


def test_normal(model):

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))

    return test_loss

if __name__ == '__main__':
    vis = visdom.Visdom(port=6006)
    model1, model2 = Net(), Net()
    if cuda:
        model1.cuda()
        model2.cuda()
    print("snapshot ensemble")
    models = train_se(model1, 300, 6, 0.1, vis)
    test_se(Net, models, 5)
    print("---")
    print("normal way")
    normal_model = train_normal(model2, 300, vis)
    test_normal(normal_model)

