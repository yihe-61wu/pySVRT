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

import torch
import torchvision, os

from torch import optim
from torch import FloatTensor as Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as fn

from torchvision import datasets, transforms, utils

import svrt
import svrt.parse

######################################################################
# Parsing arguments
######################################################################

parser = argparse.ArgumentParser(
    description='SVRT sample generator.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--nb_samples',
                    type = int,
                    default = 1000,
                    help='How many samples to generate in total')

parser.add_argument('--batch_size',
                    type = int,
                    default = 1000,
                    help='How many samples to generate at once')

parser.add_argument('--problem',
                    type = int,
                    default = 1,
                    help='Problem to generate samples from')

parser.add_argument('--data_dir',
                    type = str,
                    default = '',
                    help='Where to generate the samples')

parser.add_argument('--parsed_dir',
                    type = str,
                    default = '',
                    help='Where to put parsed output strings')

######################################################################

args = parser.parse_args()

if os.path.isdir(args.data_dir):
    name = 'problem_{:02d}/class_'.format(args.problem)
    os.makedirs(args.data_dir + '/' + name + '0', exist_ok = True)
    os.makedirs(args.data_dir + '/' + name + '1', exist_ok = True)
else:
    # FileNotFoundError does not exist in Python 2, so this is a work-around
    # where we define it as IOError.
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError
    raise FileNotFoundError('Cannot find ' + args.data_dir)

if args.parsed_dir:
    os.makedirs(args.parsed_dir, exist_ok = True)

for n in range(0, args.nb_samples, args.batch_size):
    print(n, '/', args.nb_samples)
    labels = torch.LongTensor(min(args.batch_size, args.nb_samples - n)).zero_()
    labels.narrow(0, 0, labels.size(0)//2).fill_(1)
    x, nb_shapes, shape_list, is_bordering, is_containing = \
        svrt.generate_vignettes_full(args.problem, labels)
    x = x.float()
    x.sub_(128).div_(64)
    for k in range(x.size(0)):
        filename = args.data_dir + '/problem_{:02d}/class_{:d}/img_{:07d}.png'.format(args.problem, labels[k], k + n)
        torchvision.utils.save_image(x[k].view(1, x.size(1), x.size(2)), filename)
        if args.parsed_dir:
            parsed_str = svrt.parse.parse_vignette_to_string(nb_shapes[k],
                                                             shape_list[k],
                                                             is_bordering[k],
                                                             is_containing[k])

            fname = "{:d}_{:d}_{:d}".format(args.problem, labels[k], k + n)
            with open(os.path.join(args.parsed_dir, fname), "w") as f:
                f.write(parsed_str)
