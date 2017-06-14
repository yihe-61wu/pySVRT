#!/usr/bin/env python-for-pytorch

import time

import torch

from torch import optim
from torch import FloatTensor as Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as fn
from torchvision import datasets, transforms, utils

from _ext import svrt

######################################################################
# The data

def generate_set(p, n):
    target = torch.LongTensor(n).bernoulli_(0.5)
    input = svrt.generate_vignettes(p, target)
    input = input.view(input.size(0), 1, input.size(1), input.size(2)).float()
    return Variable(input), Variable(target)

######################################################################

# 128x128 --conv(9)-> 120x120 --max(4)-> 30x30 --conv(6)-> 25x25 --max(5)-> 5x5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=9)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=6)
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = fn.relu(fn.max_pool2d(self.conv1(x), kernel_size=4, stride=4))
        x = fn.relu(fn.max_pool2d(self.conv2(x), kernel_size=5, stride=5))
        x = x.view(-1, 500)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_input, train_target):
    model, criterion = Net(), nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    nb_epochs = 25
    optimizer, bs = optim.SGD(model.parameters(), lr = 1e-1), 100

    for k in range(0, nb_epochs):
        for b in range(0, nb_train_samples, bs):
            output = model.forward(train_input.narrow(0, b, bs))
            loss = criterion(output, train_target.narrow(0, b, bs))
            model.zero_grad()
            loss.backward()
            optimizer.step()

    return model

######################################################################

def print_test_error(model, test_input, test_target):
    bs = 100
    nb_test_errors = 0

    for b in range(0, nb_test_samples, bs):
        output = model.forward(test_input.narrow(0, b, bs))
        _, wta = torch.max(output.data, 1)

        for i in range(0, bs):
            if wta[i][0] != test_target.narrow(0, b, bs).data[i]:
                nb_test_errors = nb_test_errors + 1

    print('TEST_ERROR {:.02f}% ({:d}/{:d})'.format(
        100 * nb_test_errors / nb_test_samples,
        nb_test_errors,
        nb_test_samples)
    )

######################################################################

nb_train_samples = 100000
nb_test_samples = 10000

for p in range(1, 24):
    print('-- PROBLEM #{:d} --'.format(p))

    t1 = time.time()
    train_input, train_target = generate_set(p, nb_train_samples)
    test_input, test_target = generate_set(p, nb_test_samples)
    if torch.cuda.is_available():
        train_input, train_target = train_input.cuda(), train_target.cuda()
        test_input, test_target = test_input.cuda(), test_target.cuda()

    mu, std = train_input.data.mean(), train_input.data.std()
    train_input.data.sub_(mu).div_(std)
    test_input.data.sub_(mu).div_(std)

    t2 = time.time()
    print('[data generation {:.02f}s]'.format(t2 - t1))
    model = train_model(train_input, train_target)

    t3 = time.time()
    print('[train {:.02f}s]'.format(t3 - t2))
    print_test_error(model, test_input, test_target)

    t4 = time.time()

    print('[test {:.02f}s]'.format(t4 - t3))
    print()

######################################################################
