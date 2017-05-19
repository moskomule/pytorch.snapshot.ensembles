from collections import OrderedDict

import math.pi as pi
import math.cos as cos

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# variables
cuda = True if torch.cuda.is_available() else False
batch_size = 64


# load data
transform=transforms.Compose([transforms.ToTensor(),
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
        x = F.softmax(self.dense2(x))
        return x

def train(total_iter, cycle, initial_lr):
    models = []
    cycle_size = total_iter // cycle

    for i in range(cycle):
        model = Net()
        if cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)

        for j in range(cycle_size):
            lr = initial_lr * (cos(pi*j/cycle_size)+1)
            optimizer.state_dict()["param_groups"][0]["lr"] = lr
            for batch_idx, (data, target) in enumerate(train_loader):
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        models.append(model.state_dict())
    return models


def test(models, use_model_num):
    model = Net()

    state_dict = OrderedDict()
    for name in model.state_dict().keys():
        state_dict[name] = torch.mean([x[name] for x in models[len(models)-use_model_num:]])

    model.load_state_dict(state_dict)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)))

    return test_loss

if __name__ == '__main__':
    models = train(100, 10, 0.1)
    test(models, 7)