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
import math

from colorama import Fore, Back, Style

# Pytorch

import torch

from torch import optim
from torch import FloatTensor as Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as fn
from torchvision import datasets, transforms, utils

# SVRT

from vignette_set import VignetteSet, CompressedVignetteSet

######################################################################

parser = argparse.ArgumentParser(
    description = 'Simple convnet test on the SVRT.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nb_train_batches',
                    type = int, default = 1000,
                    help = 'How many samples for train')

parser.add_argument('--nb_test_batches',
                    type = int, default = 100,
                    help = 'How many samples for test')

parser.add_argument('--nb_epochs',
                    type = int, default = 50,
                    help = 'How many training epochs')

parser.add_argument('--batch_size',
                    type = int, default = 100,
                    help = 'Mini-batch size')

parser.add_argument('--log_file',
                    type = str, default = 'cnn-svrt.log',
                    help = 'Log file name')

parser.add_argument('--compress_vignettes',
                    action='store_true', default = False,
                    help = 'Use lossless compression to reduce the memory footprint')

args = parser.parse_args()

######################################################################

log_file = open(args.log_file, 'w')

print(Fore.RED + 'Logging into ' + args.log_file + Style.RESET_ALL)

def log_string(s):
    s = Fore.GREEN + time.ctime() + Style.RESET_ALL + ' ' + s
    log_file.write(s + '\n')
    log_file.flush()
    print(s)

######################################################################

# Afroze's ShallowNet

#                       map size   nb. maps
#                     ----------------------
#    input                128x128    1
# -- conv(21x21 x 6)   -> 108x108    6
# -- max(2x2)          -> 54x54      6
# -- conv(19x19 x 16)  -> 36x36      16
# -- max(2x2)          -> 18x18      16
# -- conv(18x18 x 120) -> 1x1        120
# -- reshape           -> 120        1
# -- full(120x84)      -> 84         1
# -- full(84x2)        -> 2          1

class AfrozeShallowNet(nn.Module):
    def __init__(self):
        super(AfrozeShallowNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=21)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=19)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=18)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 2)
        self.name = 'shallownet'

    def forward(self, x):
        x = fn.relu(fn.max_pool2d(self.conv1(x), kernel_size=2))
        x = fn.relu(fn.max_pool2d(self.conv2(x), kernel_size=2))
        x = fn.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

######################################################################

def train_model(model, train_set):
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr = 1e-2)

    for e in range(0, args.nb_epochs):
        acc_loss = 0.0
        for b in range(0, train_set.nb_batches):
            input, target = train_set.get_batch(b)
            output = model.forward(Variable(input))
            loss = criterion(output, Variable(target))
            acc_loss = acc_loss + loss.data[0]
            model.zero_grad()
            loss.backward()
            optimizer.step()
        log_string('train_loss {:d} {:f}'.format(e + 1, acc_loss))

    return model

######################################################################

def nb_errors(model, data_set):
    ne = 0
    for b in range(0, data_set.nb_batches):
        input, target = data_set.get_batch(b)
        output = model.forward(Variable(input))
        wta_prediction = output.data.max(1)[1].view(-1)

        for i in range(0, data_set.batch_size):
            if wta_prediction[i] != target[i]:
                ne = ne + 1

    return ne

######################################################################

for arg in vars(args):
    log_string('argument ' + str(arg) + ' ' + str(getattr(args, arg)))

for problem_number in range(1, 24):
    if args.compress_vignettes:
        train_set = CompressedVignetteSet(problem_number, args.nb_train_batches, args.batch_size,
                                          cuda=torch.cuda.is_available())
        test_set = CompressedVignetteSet(problem_number, args.nb_test_batches, args.batch_size,
                                         cuda=torch.cuda.is_available())
    else:
        train_set = VignetteSet(problem_number, args.nb_train_batches, args.batch_size,
                                          cuda=torch.cuda.is_available())
        test_set = VignetteSet(problem_number, args.nb_test_batches, args.batch_size,
                                          cuda=torch.cuda.is_available())

    model = AfrozeShallowNet()

    if torch.cuda.is_available():
        model.cuda()

    nb_parameters = 0
    for p in model.parameters():
        nb_parameters += p.numel()
    log_string('nb_parameters {:d}'.format(nb_parameters))

    model_filename = model.name + '_' + str(problem_number) + '_' + str(train_set.nb_batches) + '.param'

    try:
        model.load_state_dict(torch.load(model_filename))
        log_string('loaded_model ' + model_filename)
    except:
        log_string('training_model')
        train_model(model, train_set)
        torch.save(model.state_dict(), model_filename)
        log_string('saved_model ' + model_filename)

    nb_train_errors = nb_errors(model, train_set)

    log_string('train_error {:d} {:.02f}% {:d} {:d}'.format(
        problem_number,
        100 * nb_train_errors / train_set.nb_samples,
        nb_train_errors,
        train_set.nb_samples)
    )

    nb_test_errors = nb_errors(model, test_set)

    log_string('test_error {:d} {:.02f}% {:d} {:d}'.format(
        problem_number,
        100 * nb_test_errors / test_set.nb_samples,
        nb_test_errors,
        test_set.nb_samples)
    )

######################################################################
