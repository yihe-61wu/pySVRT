
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
from torch import multiprocessing

from torch import Tensor
from torch.autograd import Variable

import svrt

######################################################################

def generate_one_batch(s):
    problem_number, batch_size, random_seed = s
    svrt.seed(random_seed)
    target = torch.LongTensor(batch_size).bernoulli_(0.5)
    input = svrt.generate_vignettes(problem_number, target)
    input = input.float().view(input.size(0), 1, input.size(1), input.size(2))
    return [ input, target ]

class VignetteSet:

    def __init__(self, problem_number, nb_batches, batch_size, cuda = False):
        self.cuda = cuda
        self.batch_size = batch_size
        self.problem_number = problem_number
        self.nb_batches = nb_batches
        self.nb_samples = self.nb_batches * self.batch_size

        seeds = torch.LongTensor(self.nb_batches).random_()
        mp_args = []
        for b in range(0, self.nb_batches):
            mp_args.append( [ problem_number, batch_size, seeds[b] ])

        self.data = []
        for b in range(0, self.nb_batches):
            self.data.append(generate_one_batch(mp_args[b]))

        # Weird thing going on with the multi-processing, waiting for more info

        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # self.data = pool.map(generate_one_batch, mp_args)

        acc = 0.0
        acc_sq = 0.0
        for b in range(0, self.nb_batches):
            input = self.data[b][0]
            acc += input.sum() / input.numel()
            acc_sq += input.pow(2).sum() /  input.numel()

        mean = acc / self.nb_batches
        std = sqrt(acc_sq / self.nb_batches - mean * mean)
        for b in range(0, self.nb_batches):
            self.data[b][0].sub_(mean).div_(std)
            if cuda:
                self.data[b][0] = self.data[b][0].cuda()
                self.data[b][1] = self.data[b][1].cuda()

    def get_batch(self, b):
        return self.data[b]

######################################################################

class CompressedVignetteSet:
    def __init__(self, problem_number, nb_batches, batch_size, cuda = False):
        self.cuda = cuda
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

        if self.cuda:
            input = input.cuda()
            target = target.cuda()

        return input, target

######################################################################
