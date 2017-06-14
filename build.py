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

import os
from torch.utils.ffi import create_extension

abs_path = os.path.dirname(os.path.abspath(__file__))

ffi = create_extension(
    '_ext.svrt',
    headers = [ 'svrt.h' ],
    sources = [ 'svrt.c' ],
    extra_objects = [ abs_path + '/libsvrt.so' ],
    libraries = [ ],
    library_dirs = [ ],
    define_macros = [ ],
    with_cuda = False
)

if __name__ == '__main__':
    ffi.build()
