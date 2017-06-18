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
#  along with svrt.  If not, see <http://www.gnu.org/licenses/>.

import time
import argparse
import math
import distutils.util

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

import svrtset

######################################################################

parser = argparse.ArgumentParser(
    description = "Convolutional networks for the SVRT. Written by Francois Fleuret, (C) Idiap research institute.",
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nb_train_samples',
                    type = int, default = 100000)

parser.add_argument('--nb_test_samples',
                    type = int, default = 10000)

parser.add_argument('--nb_epochs',
                    type = int, default = 50)

parser.add_argument('--batch_size',
                    type = int, default = 100)

parser.add_argument('--log_file',
                    type = str, default = 'default.log')

parser.add_argument('--compress_vignettes',
                    type = distutils.util.strtobool, default = 'True',
                    help = 'Use lossless compression to reduce the memory footprint')

parser.add_argument('--deep_model',
                    type = distutils.util.strtobool, default = 'True',
                    help = 'Use Afroze\'s Alexnet-like deep model')

parser.add_argument('--test_loaded_models',
                    type = distutils.util.strtobool, default = 'False',
                    help = 'Should we compute the test errors of loaded models')

args = parser.parse_args()

######################################################################

log_file = open(args.log_file, 'w')
pred_log_t = None

print(Fore.RED + 'Logging into ' + args.log_file + Style.RESET_ALL)

# Log and prints the string, with a time stamp. Does not log the
# remark
def log_string(s, remark = ''):
    global pred_log_t

    t = time.time()

    if pred_log_t is None:
        elapsed = 'start'
    else:
        elapsed = '+{:.02f}s'.format(t - pred_log_t)

    pred_log_t = t

    log_file.write('[' + time.ctime() + '] ' + elapsed + ' ' + s + '\n')
    log_file.flush()

    print(Fore.BLUE + '[' + time.ctime() + '] ' + Fore.GREEN + elapsed + Style.RESET_ALL + ' ' + s + Fore.CYAN + remark + Style.RESET_ALL)

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

# Afroze's DeepNet

class AfrozeDeepNet(nn.Module):
    def __init__(self):
        super(AfrozeDeepNet, self).__init__()
        self.conv1 = nn.Conv2d(  1,  32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d( 32,  96, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d( 96, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128,  96, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.name = 'deepnet'

    def forward(self, x):
        x = self.conv1(x)
        x = fn.max_pool2d(x, kernel_size=2)
        x = fn.relu(x)

        x = self.conv2(x)
        x = fn.max_pool2d(x, kernel_size=2)
        x = fn.relu(x)

        x = self.conv3(x)
        x = fn.relu(x)

        x = self.conv4(x)
        x = fn.relu(x)

        x = self.conv5(x)
        x = fn.max_pool2d(x, kernel_size=2)
        x = fn.relu(x)

        x = x.view(-1, 1536)

        x = self.fc1(x)
        x = fn.relu(x)

        x = self.fc2(x)
        x = fn.relu(x)

        x = self.fc3(x)

        return x

######################################################################

def train_model(model, train_set):
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr = 1e-2)

    start_t = time.time()

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
        dt = (time.time() - start_t) / (e + 1)
        log_string('train_loss {:d} {:f}'.format(e + 1, acc_loss),
                   ' [ETA ' + time.ctime(time.time() + dt * (args.nb_epochs - e)) + ']')

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

######################################################################

def int_to_suffix(n):
    if n >= 1000000 and n%1000000 == 0:
        return str(n//1000000) + 'M'
    elif n >= 1000 and n%1000 == 0:
        return str(n//1000) + 'K'
    else:
        return str(n)

class vignette_logger():
    def __init__(self, delay_min = 60):
        self.start_t = time.time()
        self.delay_min = delay_min

    def __call__(self, n, m):
        t = time.time()
        if t > self.start_t + self.delay_min:
            dt = (t - self.start_t) / m
            log_string('sample_generation {:d} / {:d}'.format(
                m,
                n), ' [ETA ' + time.ctime(time.time() + dt * (n - m)) + ']'
            )

######################################################################

if args.nb_train_samples%args.batch_size > 0 or args.nb_test_samples%args.batch_size > 0:
    print('The number of samples must be a multiple of the batch size.')
    raise

if args.compress_vignettes:
    log_string('using_compressed_vignettes')
    VignetteSet = svrtset.CompressedVignetteSet
else:
    log_string('using_uncompressed_vignettes')
    VignetteSet = svrtset.VignetteSet

for problem_number in range(1, 24):

    log_string('############### problem ' + str(problem_number) + ' ###############')

    if args.deep_model:
        model = AfrozeDeepNet()
    else:
        model = AfrozeShallowNet()

    if torch.cuda.is_available(): model.cuda()

    model_filename = model.name + '_pb:' + \
                     str(problem_number) + '_ns:' + \
                     int_to_suffix(args.nb_train_samples) + '.param'

    nb_parameters = 0
    for p in model.parameters(): nb_parameters += p.numel()
    log_string('nb_parameters {:d}'.format(nb_parameters))

    ##################################################
    # Tries to load the model

    need_to_train = False
    try:
        model.load_state_dict(torch.load(model_filename))
        log_string('loaded_model ' + model_filename)
    except:
        need_to_train = True

    ##################################################
    # Train if necessary

    if need_to_train:

        log_string('training_model ' + model_filename)

        t = time.time()

        train_set = VignetteSet(problem_number,
                                args.nb_train_samples, args.batch_size,
                                cuda = torch.cuda.is_available(),
                                logger = vignette_logger())

        log_string('data_generation {:0.2f} samples / s'.format(
            train_set.nb_samples / (time.time() - t))
        )

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

    ##################################################
    # Test if necessary

    if need_to_train or args.test_loaded_models:

        t = time.time()

        test_set = VignetteSet(problem_number,
                               args.nb_test_samples, args.batch_size,
                               cuda = torch.cuda.is_available())

        log_string('data_generation {:0.2f} samples / s'.format(
            test_set.nb_samples / (time.time() - t))
        )

        nb_test_errors = nb_errors(model, test_set)

        log_string('test_error {:d} {:.02f}% {:d} {:d}'.format(
            problem_number,
            100 * nb_test_errors / test_set.nb_samples,
            nb_test_errors,
            test_set.nb_samples)
        )

######################################################################
