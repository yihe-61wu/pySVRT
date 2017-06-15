
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

import torch
from math import sqrt

from torch import Tensor
from torch.autograd import Variable

import svrt

######################################################################

class VignetteSet:
    def __init__(self, problem_number, nb_batches, batch_size):
        self.batch_size = batch_size
        self.problem_number = problem_number
        self.nb_batches = nb_batches
        self.nb_samples = self.nb_batches * self.batch_size
        self.targets = []
        self.inputs = []

        acc = 0.0
        acc_sq = 0.0

        for b in range(0, self.nb_batches):
            target = torch.LongTensor(self.batch_size).bernoulli_(0.5)
            input = svrt.generate_vignettes(problem_number, target)
            input = input.float().view(input.size(0), 1, input.size(1), input.size(2))
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            acc += input.sum() / input.numel()
            acc_sq += input.pow(2).sum() /  input.numel()
            self.targets.append(target)
            self.inputs.append(input)

        mean = acc / self.nb_batches
        std = sqrt(acc_sq / self.nb_batches - mean * mean)
        for b in range(0, self.nb_batches):
            self.inputs[b].sub_(mean).div_(std)

    def get_batch(self, b):
        return self.inputs[b], self.targets[b]

######################################################################

class CompressedVignetteSet:
    def __init__(self, problem_number, nb_batches, batch_size):
        self.batch_size = batch_size
        self.problem_number = problem_number
        self.nb_batches = nb_batches
        self.nb_samples = self.nb_batches * self.batch_size
        self.targets = []
        self.input_storages = []

        acc = 0.0
        acc_sq = 0.0
        for b in range(0, self.nb_batches):
            target = torch.LongTensor(self.batch_size).bernoulli_(0.5)
            input = svrt.generate_vignettes(problem_number, target)
            acc += input.float().sum() / input.numel()
            acc_sq += input.float().pow(2).sum() /  input.numel()
            self.targets.append(target)
            self.input_storages.append(svrt.compress(input.storage()))

        self.mean = acc / self.nb_batches
        self.std = sqrt(acc_sq / self.nb_batches - self.mean * self.mean)

    def get_batch(self, b):
        input = torch.ByteTensor(svrt.uncompress(self.input_storages[b])).float()
        input = input.view(self.batch_size, 1, 128, 128).sub_(self.mean).div_(self.std)
        target = self.targets[b]

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        return input, target

######################################################################
