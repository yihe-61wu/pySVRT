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

#ifndef SHAPE_H
#define SHAPE_H

#include "misc.h"
#include "random.h"
#include "vignette.h"

class Shape {
  static const int margin = 1;
  static const int nb_max_pixels = Vignette::width * Vignette::height;
  static const scalar_t gap_max = 0.25;
  int n_pixels1, n_pixels2, n_pixels3, n_pixels4;
  int nb_pixels;
  scalar_t xc, yc;
  scalar_t *x_pixels;
  scalar_t *y_pixels;

  int generate_part_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius,
                         scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2);
  void generate_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  int overwrites(Vignette *vignette, scalar_t xc, scalar_t yc, int first, int nb);
  void draw(int part_number, Vignette *vignette, scalar_t xc, scalar_t yc, int first, int nb);

public:
  Shape();
  ~Shape();

  void randomize(scalar_t radius, scalar_t hole_radius);
  void copy(Shape *shape);
  void scale(scalar_t s);
  void rotate(scalar_t alpha);
  void symmetrize(scalar_t axis_x, scalar_t axis_y);

  int overwrites(Vignette *vignette, scalar_t xc, scalar_t yc);
  void draw(int part_number, Vignette *vignette, scalar_t xc, scalar_t yc);
};

#endif
