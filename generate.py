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

import h5py

import svrt
import svrt.parse
import svrt.utils

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
                    help='Where to put parsed output strings')

parser.add_argument('--parsed_dir_classic',
                    type = str,
                    help='Where to put classic-style parsed output strings')

parser.add_argument('--symb_h5_raw_dir',
                    type = str,
                    help='Where to put HDF5 files containing raw tensors of'
                         ' symbolic representations of stimuli')

parser.add_argument('--symb_h5_obf_dir',
                    type = str,
                    help='Where to put HDF5 files containing tensors of'
                         ' obfuscated symbolic representations of stimuli')


######################################################################

args = parser.parse_args()

if not os.path.isdir(args.data_dir):
    os.makedirs(args.data_dir)

if not os.path.isdir(args.data_dir):
    # FileNotFoundError does not exist in Python 2, so this is a work-around
    # where we define it as IOError.
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError
    raise FileNotFoundError('Cannot find ' + args.data_dir)

for class_label in [0, 1]:
    dirname = 'problem_{:02d}/class_{:d}'.format(args.problem, class_label)
    dirname = os.path.join(args.data_dir, dirname)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


if args.symb_h5_raw_dir:
    if not os.path.isdir(args.symb_h5_raw_dir):
        os.makedirs(args.symb_h5_raw_dir)
    fname = os.path.join(args.symb_h5_raw_dir,
                         'problem_{:02d}.h5'.format(args.problem))
    hf_raw = h5py.File(fname, 'w')

if args.symb_h5_obf_dir:
    if not os.path.isdir(args.symb_h5_obf_dir):
        os.makedirs(args.symb_h5_obf_dir)
    fname = os.path.join(args.symb_h5_obf_dir,
                         'problem_{:02d}.h5'.format(args.problem))
    hf_obf = h5py.File(fname, 'w')


for n in range(0, args.nb_samples, args.batch_size):
    print('{}/{}'.format(n, args.nb_samples))
    labels = torch.LongTensor(min(args.batch_size, args.nb_samples - n)).zero_()
    labels.narrow(0, 0, labels.size(0)//2).fill_(1)
    x, nb_shapes, shape_list, is_bordering, is_containing = \
        svrt.generate_vignettes_full(args.problem, labels)
    # Save to H5
    if args.symb_h5_raw_dir:
        hf = hf_raw
        for class_label in [0, 1]:
            vg_is_class_member = labels == class_label
            hf.create_dataset(
                'class_{:d}/nb_shapes/{:d}'.format(class_label, n),
                data=nb_shapes[vg_is_class_member])
            hf.create_dataset(
                'class_{:d}/shape_list/{:d}'.format(class_label, n),
                data=shape_list[vg_is_class_member, :, :])
            hf.create_dataset(
                'class_{:d}/is_bordering/{:d}'.format(class_label, n),
                data=is_bordering[vg_is_class_member, :, :])
            hf.create_dataset(
                'class_{:d}/is_containing/{:d}'.format(class_label, n),
                data=is_containing[vg_is_class_member, :, :])
    # Obfuscate shape construction order, and rotation/reflection state
    nb_shapes, shape_list, is_bordering, is_containing = \
        svrt.utils.obfuscate_shape_construction(
            nb_shapes, shape_list, is_bordering, is_containing)
    # Save to H5
    if args.symb_h5_raw_dir:
        hf = hf_obf
        for class_label in [0, 1]:
            vg_is_class_member = labels == class_label
            hf.create_dataset(
                'class_{:d}/nb_shapes/{:d}'.format(class_label, n),
                data=nb_shapes[vg_is_class_member])
            hf.create_dataset(
                'class_{:d}/shape_list/{:d}'.format(class_label, n),
                data=shape_list[vg_is_class_member, :, :])
            hf.create_dataset(
                'class_{:d}/is_bordering/{:d}'.format(class_label, n),
                data=is_bordering[vg_is_class_member, :, :])
            hf.create_dataset(
                'class_{:d}/is_containing/{:d}'.format(class_label, n),
                data=is_containing[vg_is_class_member, :, :])

    x = x.float()
    x.sub_(128).div_(64)

    for k in range(x.size(0)):
        subdir_fname = 'problem_{:02d}/class_{:d}/img_{:07d}.png'.format(
            args.problem, labels[k], k + n)
        filename = os.path.join(args.data_dir, subdir_fname)
        torchvision.utils.save_image(x[k].view(1, x.size(1), x.size(2)), filename)
        # Output parsed strings in classic sasquatch style
        if args.parsed_dir_classic:
            parsed_str = svrt.parse.parse_vignette_to_string_classic(
                nb_shapes[k], shape_list[k], is_bordering[k], is_containing[k])
            fname = os.path.join(args.parsed_dir_classic,
                                 subdir_fname[:-4] + '.txt')
            if not os.path.isdir(os.path.split(fname)[0]):
                os.makedirs(os.path.split(fname)[0])
            with open(fname, "w") as f:
                f.write(parsed_str)
        # Output parsed strings in updated style, with rotation and reflection
        if args.parsed_dir:
            parsed_str = svrt.parse.parse_vignette_to_string(
                nb_shapes[k], shape_list[k], is_bordering[k], is_containing[k])
            fname = os.path.join(args.parsed_dir,
                                 subdir_fname[:-4] + '.txt')
            if not os.path.isdir(os.path.split(fname)[0]):
                os.makedirs(os.path.split(fname)[0])
            with open(fname, "w") as f:
                f.write(parsed_str)

# Close open files
if args.symb_h5_raw_dir:
    hf_raw.close()
if args.symb_h5_obf_dir:
    hf_obf.close()
