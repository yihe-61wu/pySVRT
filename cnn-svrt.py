#!/usr/bin/env python

#  svrt is the ``Synthetic Visual Reasoning Test'', an image
#  generator for evaluating classification performance of machine
#  learning systems, humans and primates.
#
#  Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
#  Written by Francois Fleuret <francois.fleuret@idiap.ch>
#
#  This file is part of svrt.
#
#  svrt is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License version 3 as
#  published by the Free Software Foundation.
#
#  svrt is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with selector.  If not, see <http://www.gnu.org/licenses/>.

import time
import argparse
from colorama import Fore, Back, Style

import torch

from torch import optim
from torch import FloatTensor as Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as fn
from torchvision import datasets, transforms, utils

import svrt

######################################################################

parser = argparse.ArgumentParser(
    description = 'Simple convnet test on the SVRT.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nb_train_samples',
                    type = int, default = 100000,
                    help = 'How many samples for train')

parser.add_argument('--nb_test_samples',
                    type = int, default = 10000,
                    help = 'How many samples for test')

parser.add_argument('--nb_epochs',
                    type = int, default = 25,
                    help = 'How many training epochs')

parser.add_argument('--log_file',
                    type = str, default = 'cnn-svrt.log',
                    help = 'Log file name')

args = parser.parse_args()

######################################################################

log_file = open(args.log_file, 'w')

print(Fore.RED + 'Logging into ' + args.log_file + Style.RESET_ALL)

def log_string(s):
    s = Fore.GREEN + time.ctime() + Style.RESET_ALL + ' ' + \
        str(problem_number) + ' ' + s
    log_file.write(s + '\n')
    log_file.flush()
    print(s)

######################################################################

def generate_set(p, n):
    target = torch.LongTensor(n).bernoulli_(0.5)
    t = time.time()
    input = svrt.generate_vignettes(p, target)
    t = time.time() - t
    log_string('DATA_SET_GENERATION {:.02f} sample/s'.format(n / t))
    input = input.view(input.size(0), 1, input.size(1), input.size(2)).float()
    return Variable(input), Variable(target)

######################################################################

# Afroze's ShallowNet

#                    map size   nb. maps
#                  ----------------------
#                    128x128    1
# -- conv(21x21)  -> 108x108    6
# -- max(2x2)     -> 54x54      6
# -- conv(19x19)  -> 36x36      16
# -- max(2x2)     -> 18x18      16
# -- conv(18x18)  -> 1x1        120
# -- reshape      -> 120        1
# -- full(120x84) -> 84         1
# -- full(84x2)   -> 2          1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=21)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=19)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=18)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 2)

    def forward(self, x):
        x = fn.relu(fn.max_pool2d(self.conv1(x), kernel_size=2))
        x = fn.relu(fn.max_pool2d(self.conv2(x), kernel_size=2))
        x = fn.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_input, train_target):
    model, criterion = Net(), nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    optimizer, bs = optim.SGD(model.parameters(), lr = 1e-2), 100

    for k in range(0, args.nb_epochs):
        acc_loss = 0.0
        for b in range(0, train_input.size(0), bs):
            output = model.forward(train_input.narrow(0, b, bs))
            loss = criterion(output, train_target.narrow(0, b, bs))
            acc_loss = acc_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            optimizer.step()
        log_string('TRAIN_LOSS {:d} {:f}'.format(k, acc_loss))

    return model

######################################################################

def nb_errors(model, data_input, data_target, bs = 100):
    ne = 0

    for b in range(0, data_input.size(0), bs):
        output = model.forward(data_input.narrow(0, b, bs))
        wta_prediction = output.data.max(1)[1].view(-1)

        for i in range(0, bs):
            if wta_prediction[i] != data_target.narrow(0, b, bs).data[i]:
                ne = ne + 1

    return ne

######################################################################

for problem_number in range(1, 24):
    train_input, train_target = generate_set(problem_number, args.nb_train_samples)
    test_input, test_target = generate_set(problem_number, args.nb_test_samples)

    if torch.cuda.is_available():
        train_input, train_target = train_input.cuda(), train_target.cuda()
        test_input, test_target = test_input.cuda(), test_target.cuda()

    mu, std = train_input.data.mean(), train_input.data.std()
    train_input.data.sub_(mu).div_(std)
    test_input.data.sub_(mu).div_(std)

    model = train_model(train_input, train_target)

    nb_train_errors = nb_errors(model, train_input, train_target)

    log_string('TRAIN_ERROR {:.02f}% {:d} {:d}'.format(
        100 * nb_train_errors / train_input.size(0),
        nb_train_errors,
        train_input.size(0))
    )

    nb_test_errors = nb_errors(model, test_input, test_target)

    log_string('TEST_ERROR {:.02f}% {:d} {:d}'.format(
        100 * nb_test_errors / test_input.size(0),
        nb_test_errors,
        test_input.size(0))
    )

######################################################################
