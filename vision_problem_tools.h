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

#ifndef VISION_PROBLEM_TOOLS_H
#define VISION_PROBLEM_TOOLS_H

#include "misc.h"
#include "random.h"
#include "vignette.h"

int cluttered_shapes(int part_size, int nb_shapes, int *xs, int *ys);

scalar_t dist_point_to_segment(scalar_t xp, scalar_t yp,
                               scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2);

int point_in_band(scalar_t xp, scalar_t yp,
                  scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2,
                  scalar_t width);

#endif
