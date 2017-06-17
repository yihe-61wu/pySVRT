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

#include "vision_problem_tools.h"

int cluttered_shapes(int part_size, int nb_shapes, int *xs, int *ys) {
  for(int n = 0; n < nb_shapes; n++) {
    for(int m = 0; m < n; m++) {
      if(abs(xs[n] - xs[m]) < part_size && abs(ys[n] - ys[m]) < part_size)
        return 1;
    }
  }
  return 0;
}

scalar_t dist_point_to_segment(scalar_t xp, scalar_t yp,
                               scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2) {
  scalar_t s;
  s = (xp - x1) * (x2 - x1) + (yp - y1) * (y2 - y1);
  if(s < 0) {
    return sqrt(sq(xp - x1) + sq(yp - y1));
  } else if(s > sq(x2 - x1) + sq(y2 - y1)) {
    return sqrt(sq(xp - x2) + sq(yp - y2));
  } else {
    return abs((xp - x1) * (y2 - y1) - (yp - y1) * (x2 - x1))/sqrt(sq(x2 - x1) + sq(y2 - y1));
  }
}

int point_in_band(scalar_t xp, scalar_t yp,
                  scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2,
                  scalar_t width) {
  scalar_t s;
  s = (xp - x1) * (x2 - x1) + (yp - y1) * (y2 - y1);
  if(s < 0) {
    return 0;
  } else if(s > sq(x2 - x1) + sq(y2 - y1)) {
    return 0;
  } else {
    return abs((xp - x1) * (y2 - y1) - (yp - y1) * (x2 - x1))/sqrt(sq(x2 - x1) + sq(y2 - y1)) <= width;
  }
}
