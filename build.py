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

from os import path
from torch.utils.ffi import create_extension

abs_path = path.dirname(path.abspath(__file__))

ffi = create_extension(
    'svrt',
    headers = [ 'svrt.h' ],
    sources = [ 'svrt.c' ],
    extra_objects = [ abs_path + '/libsvrt.so' ],
    with_cuda = False
)

ffi.build()

extra_py = """

import numpy as np
import torch

def generate_vignettes_full(problem, labels):

    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(np.array(labels)).type(torch.LongTensor)

    nb_shapes = torch.ByteTensor()
    shape_list = torch.FloatTensor()
    is_containing = torch.FloatTensor()
    intershape_distance = torch.ByteTensor()

    x = generate_vignettes_raw(problem, labels, nb_shapes, shape_list, intershape_distance, is_containing)

    return x, nb_shapes, shape_list, intershape_distance, is_containing

def generate_vignettes(problem, labels):
    return generate_vignettes_full(problem, labels)[0]
"""

with open(path.join('svrt', '__init__.py'), 'a') as f:
    f.write(extra_py)
