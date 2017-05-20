
from math import pi
from math import cos

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

# variables
cuda = torch.cuda.is_available()


def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2


def train_se(model, epochs, cycles, initial_lr, train_loader, vis=None):

    spapshots = []
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

        spapshots.append(model.state_dict())
    return spapshots


def test_se(Model, weights, use_model_num, test_loader):
    index = len(weights) - use_model_num
    weights = weights[index:]
    model_list = [Model() for _ in weights]

    for model, weight in zip(model_list, weights):
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


def train_normal(model, epochs, train_loader, vis=None):

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


def test_normal(model, test_loader):

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