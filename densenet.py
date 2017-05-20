import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import visdom

from se import train_se, train_normal, test_se, test_normal

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

class CifarDenseNet(nn.Module):
    def __init__(self):
        super(CifarDenseNet, self).__init__()
        densenet = models.densenet121()
        self.dense_head = nn.Sequential(*list(densenet.children())[:-1])
        self.dense1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.dense_head(x)
        x = x.view(-1, 1024)
        x = self.dense1(x)
        return F.softmax(x)

if __name__ == '__main__':
    vis = visdom.Visdom(port=6006)
    print("densenet")

    model1, model2 = CifarDenseNet(), CifarDenseNet()
    if cuda:
        model1.cuda()
        model2.cuda()

    models = train_se(model1, 300, 6, 0.1, train_loader, vis)
    print("snapshot ensemble")
    test_se(CifarDenseNet, models, 5, test_loader)
    print("---")
    print("normal way")
    normal_model = train_normal(model2, 300, train_loader, vis)
    test_normal(normal_model, test_loader)
