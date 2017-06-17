/*
 *  svrt is the ``Synthetic Visual Reasoning Test'', an image
 *  generator for evaluating classification performance of machine
 *  learning systems, humans and primates.
 *
 *  Copyright (c) 2009 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Francois Fleuret <francois.fleuret@idiap.ch>
 *
 *  This file is part of svrt.
 *
 *  svrt is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  svrt is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with svrt.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VIGNETTE_GENERATOR_H
#define VIGNETTE_GENERATOR_H

#include "misc.h"
#include "vignette.h"

class VignetteGenerator {
public:
  // We need a virtual destructor since we have virtual methods
  virtual ~VignetteGenerator();

  // Some generators need to do pre-computations that can not be put
  // in the constructor
  virtual void precompute();

  // Generate a vignette
  virtual void generate(int label, Vignette *vignette) = 0;
};

#endif
