#!/usr/bin/env python

import argparse
import os, os.path
import time

import h5py

import svrt
import svrt.parse
import svrt.utils
import svrt.ioutils

######################################################################
# Parsing arguments
######################################################################

parser = argparse.ArgumentParser(
    description='SVRT sample generator.',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('h5_dir',
                    type = str,
                    help='Where to find the HDF5 files containing tensors'
                         ' of symbolic representations of stimuli')

parser.add_argument('--problem',
                    type = int,
                    default = -1,
                    help='Problem to generate samples from')

parser.add_argument('--parsed_dir',
                    type = str,
                    help='Where to put parsed output strings')

parser.add_argument('--parsed_dir_classic',
                    type = str,
                    help='Where to put classic-style parsed output strings')

parser.add_argument('--parse_classic',
                    action = 'store_const',
                    const = 'parsed_classic',
                    dest = 'parsed_dir_classic',
                    help='Whether to put classic parsing in default place')

######################################################################

args = parser.parse_args()
print(args)

def make_dirs_for_file(fname):
    dirname = os.path.split(fname)[0]
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


if args.problem == -1:
    problems = range(1, 24)
else:
    problems = [args.problem]

for problem in problems:
    print('----------------')
    print('Problem {}'.format(problem))
    # Load from HDF5
    n = 0
    r = 0
    for labels, nb_shapes, shape_list, is_bordering, is_containing in \
            svrt.ioutils.load_from_h5(args.h5_dir, problem):
        r += 1
        print('Record {}'.format(r))
        for k in range(len(shape_list)):
            subdir_fname = 'problem_{:02d}/class_{:d}/img_{:07d}.txt'.format(
                problem, labels[k], n)
            n += 1
            # Output parsed strings in classic sasquatch style
            if args.parsed_dir_classic:
                parsed_str = svrt.parse.parse_vignette_to_string_classic(
                    nb_shapes[k], shape_list[k], is_bordering[k], is_containing[k])
                fname = os.path.join(args.parsed_dir_classic, subdir_fname)
                make_dirs_for_file(fname)
                with open(fname, "w") as f:
                    f.write(parsed_str)
            # Output parsed strings in updated style, with rotation and reflection
            if args.parsed_dir:
                parsed_str = svrt.parse.parse_vignette_to_string(
                    nb_shapes[k], shape_list[k], is_bordering[k], is_containing[k])
                fname = os.path.join(args.parsed_dir, subdir_fname)
                make_dirs_for_file(fname)
                with open(fname, "w") as f:
                    f.write(parsed_str)
