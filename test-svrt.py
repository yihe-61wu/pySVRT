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

import torch
import torchvision

from torch import optim
from torch import FloatTensor as Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as fn

from torchvision import datasets, transforms, utils

from _ext import svrt

labels = torch.LongTensor(12).zero_()
labels.narrow(0, 0, labels.size(0)//2).fill_(1)

x = svrt.generate_vignettes(4, labels)

x = x.view(x.size(0), 1, x.size(1), x.size(2))

x.div_(255)

torchvision.utils.save_image(x, 'example.png')
