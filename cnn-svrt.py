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
import re
import signal

from colorama import Fore, Back, Style

# Pytorch

import torch
import torchvision

from torch import optim
from torch import multiprocessing
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

parser.add_argument('--nb_validation_samples',
                    type = int, default = 10000)

parser.add_argument('--validation_error_threshold',
                    type = float, default = 0.0,
                    help = 'Early training termination criterion')

parser.add_argument('--nb_epochs',
                    type = int, default = 50)

parser.add_argument('--batch_size',
                    type = int, default = 100)

parser.add_argument('--log_file',
                    type = str, default = 'default.log')

parser.add_argument('--nb_exemplar_vignettes',
                    type = int, default = 32)

parser.add_argument('--compress_vignettes',
                    type = distutils.util.strtobool, default = 'True',
                    help = 'Use lossless compression to reduce the memory footprint')

parser.add_argument('--save_test_mistakes',
                    type = distutils.util.strtobool, default = 'False')

parser.add_argument('--model',
                    type = str, default = 'deepnet',
                    help = 'What model to use')

parser.add_argument('--test_loaded_models',
                    type = distutils.util.strtobool, default = 'False',
                    help = 'Should we compute the test errors of loaded models')

parser.add_argument('--problems',
                    type = str, default = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                    help = 'What problems to process')

args = parser.parse_args()

######################################################################

log_file = open(args.log_file, 'a')
log_file.write('\n')
log_file.write('@@@@@@@@@@@@@@@@@@@ ' + time.ctime() + ' @@@@@@@@@@@@@@@@@@@\n')
log_file.write('\n')

pred_log_t = None
last_tag_t = time.time()

print(Fore.RED + 'Logging into ' + args.log_file + Style.RESET_ALL)

# Log and prints the string, with a time stamp. Does not log the
# remark

def log_string(s, remark = ''):
    global pred_log_t, last_tag_t

    t = time.time()

    if pred_log_t is None:
        elapsed = 'start'
    else:
        elapsed = '+{:.02f}s'.format(t - pred_log_t)

    pred_log_t = t

    if t > last_tag_t + 3600:
        last_tag_t = t
        print(Fore.RED + time.ctime() + Style.RESET_ALL)

    log_file.write(re.sub(' ', '_', time.ctime()) + ' ' + elapsed + ' ' + s + '\n')
    log_file.flush()

    print(Fore.BLUE + time.ctime() + ' ' + Fore.GREEN + elapsed \
          + Style.RESET_ALL
          + ' ' \
          + s + Fore.CYAN + remark \
          + Style.RESET_ALL)

######################################################################

def handler_sigint(signum, frame):
    log_string('got sigint')
    exit(0)

def handler_sigterm(signum, frame):
    log_string('got sigterm')
    exit(0)

signal.signal(signal.SIGINT, handler_sigint)
signal.signal(signal.SIGTERM, handler_sigterm)

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
    name = 'shallownet'

    def __init__(self):
        super(AfrozeShallowNet, self).__init__()
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

######################################################################

# Afroze's DeepNet

class AfrozeDeepNet(nn.Module):

    name = 'deepnet'

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

class DeepNet2(nn.Module):
    name = 'deepnet2'

    def __init__(self):
        super(DeepNet2, self).__init__()
        self.nb_channels = 512
        self.conv1 = nn.Conv2d(  1,  32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d( 32, nb_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * self.nb_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

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

        x = x.view(-1, 16 * self.nb_channels)

        x = self.fc1(x)
        x = fn.relu(x)

        x = self.fc2(x)
        x = fn.relu(x)

        x = self.fc3(x)

        return x

######################################################################

class DeepNet3(nn.Module):
    name = 'deepnet3'

    def __init__(self):
        super(DeepNet3, self).__init__()
        self.conv1 = nn.Conv2d(  1,  32, kernel_size=7, stride=4, padding=3)
        self.conv2 = nn.Conv2d( 32, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

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

        x = self.conv6(x)
        x = fn.relu(x)

        x = self.conv7(x)
        x = fn.relu(x)

        x = x.view(-1, 2048)

        x = self.fc1(x)
        x = fn.relu(x)

        x = self.fc2(x)
        x = fn.relu(x)

        x = self.fc3(x)

        return x

######################################################################

def nb_errors(model, data_set, mistake_filename_pattern = None):
    ne = 0
    for b in range(0, data_set.nb_batches):
        input, target = data_set.get_batch(b)
        output = model.forward(Variable(input))
        wta_prediction = output.data.max(1)[1].view(-1)

        for i in range(0, data_set.batch_size):
            if wta_prediction[i] != target[i]:
                ne = ne + 1
                if mistake_filename_pattern is not None:
                    img = input[i].clone()
                    img.sub_(img.min())
                    img.div_(img.max())
                    k = b * data_set.batch_size + i
                    filename = mistake_filename_pattern.format(k, target[i])
                    torchvision.utils.save_image(img, filename)
                    print(Fore.RED + 'Wrote ' + filename + Style.RESET_ALL)
    return ne

######################################################################

def train_model(model, model_filename, train_set, validation_set, nb_epochs_done = 0):
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr = 1e-2)

    start_t = time.time()

    for e in range(nb_epochs_done, args.nb_epochs):
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

        torch.save([ model.state_dict(), e + 1 ], model_filename)

        if validation_set is not None:
            nb_validation_errors = nb_errors(model, validation_set)

            log_string('validation_error {:.02f}% {:d} {:d}'.format(
                100 * nb_validation_errors / validation_set.nb_samples,
                nb_validation_errors,
                validation_set.nb_samples)
            )

            if nb_validation_errors / validation_set.nb_samples <= args.validation_error_threshold:
                log_string('below validation_error_threshold')
                break

    return model

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
        self.last_t = self.start_t
        self.delay_min = delay_min

    def __call__(self, n, m):
        t = time.time()
        if t > self.last_t + self.delay_min:
            dt = (t - self.start_t) / m
            log_string('sample_generation {:d} / {:d}'.format(
                m,
                n), ' [ETA ' + time.ctime(time.time() + dt * (n - m)) + ']'
            )
            self.last_t = t

def save_examplar_vignettes(data_set, nb, name):
    n = torch.randperm(data_set.nb_samples).narrow(0, 0, nb)

    for k in range(0, nb):
        b = n[k] // data_set.batch_size
        m = n[k] % data_set.batch_size
        i, t = data_set.get_batch(b)
        i = i[m].float()
        i.sub_(i.min())
        i.div_(i.max())
        if k == 0: patchwork = Tensor(nb, 1, i.size(1), i.size(2))
        patchwork[k].copy_(i)

    torchvision.utils.save_image(patchwork, name)

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

########################################
model_class = None
for m in [ AfrozeShallowNet, AfrozeDeepNet, DeepNet2, DeepNet3 ]:
    if args.model == m.name:
        model_class = m
        break
if model_class is None:
    print('Unknown model ' + args.model)
    raise

log_string('using model class ' + m.name)
########################################

for problem_number in map(int, args.problems.split(',')):

    log_string('############### problem ' + str(problem_number) + ' ###############')

    model = model_class()

    if torch.cuda.is_available(): model.cuda()

    model_filename = model.name + '_pb:' + \
                     str(problem_number) + '_ns:' + \
                     int_to_suffix(args.nb_train_samples) + '.state'

    nb_parameters = 0
    for p in model.parameters(): nb_parameters += p.numel()
    log_string('nb_parameters {:d}'.format(nb_parameters))

    ##################################################
    # Tries to load the model

    try:
        model_state_dict, nb_epochs_done = torch.load(model_filename)
        model.load_state_dict(model_state_dict)
        log_string('loaded_model ' + model_filename)
    except:
        nb_epochs_done = 0


    ##################################################
    # Train if necessary

    if nb_epochs_done < args.nb_epochs:

        log_string('training_model ' + model_filename)

        t = time.time()

        train_set = VignetteSet(problem_number,
                                args.nb_train_samples, args.batch_size,
                                cuda = torch.cuda.is_available(),
                                logger = vignette_logger())

        log_string('data_generation {:0.2f} samples / s'.format(
            train_set.nb_samples / (time.time() - t))
        )

        if args.nb_exemplar_vignettes > 0:
            save_examplar_vignettes(train_set, args.nb_exemplar_vignettes,
                                    'examplar_{:d}.png'.format(problem_number))

        if args.validation_error_threshold > 0.0:
            validation_set = VignetteSet(problem_number,
                                         args.nb_validation_samples, args.batch_size,
                                         cuda = torch.cuda.is_available(),
                                         logger = vignette_logger())
        else:
            validation_set = None

        train_model(model, model_filename, train_set, validation_set, nb_epochs_done = nb_epochs_done)
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

    if nb_epochs_done < args.nb_epochs or args.test_loaded_models:

        t = time.time()

        test_set = VignetteSet(problem_number,
                               args.nb_test_samples, args.batch_size,
                               cuda = torch.cuda.is_available())

        nb_test_errors = nb_errors(model, test_set,
                                   mistake_filename_pattern = 'mistake_{:06d}_{:d}.png')

        log_string('test_error {:d} {:.02f}% {:d} {:d}'.format(
            problem_number,
            100 * nb_test_errors / test_set.nb_samples,
            nb_test_errors,
            test_set.nb_samples)
        )

######################################################################
