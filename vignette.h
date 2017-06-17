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
 *  along with pysvrt.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VIGNETTE_H
#define VIGNETTE_H

#include "misc.h"

#define KEEP_PART_PRESENCE

class Vignette {
public:
  static const int width = 128;
  static const int height = width;
  static const int nb_grayscales = 256;

  int content[width * height];
#ifdef KEEP_PART_PRESENCE
  unsigned int part_presence[width * height];
#endif

  void clear();

  void fill(int x, int y, int v);
  void switch_values(int c1, int c2);
  void replace_value(int from, int to);
  void superpose(Vignette *infront, Vignette *inback);
  int intersection(Vignette *v);
  void grow();

};

#endif
