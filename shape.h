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

#ifndef SHAPE_H
#define SHAPE_H

#include "misc.h"
#include "random.h"
//#include "vignette.h"

class Shape {
  //static const int nb_max_pixels = Vignette::width * Vignette::height;
  static const int nb_max_pixels = 16384;

#if __cplusplus >= 201103L
  static constexpr scalar_t gap_max = 0.25;
#else
  static const scalar_t gap_max = 0.25;
#endif

  int generate_part_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius,
                         scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2);
  int generate_part_part2(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius,
                          scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2);

  void generate_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  void generate_open_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  void generate_tri_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  void generate_circle_part(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  void generate_cross(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  void generate_spiral(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);
  void generate_zigzag(scalar_t *xp, scalar_t *yp, int *nb_pixels, scalar_t radius, scalar_t hole_radius);

public:
  static const int margin = 1;
  int nb_pixels;
  int n_pixels1, n_pixels2, n_pixels3, n_pixels4;
  scalar_t xc, yc;
  scalar_t x_pixels[nb_max_pixels];
  scalar_t y_pixels[nb_max_pixels];

  Shape();
  ~Shape();

  void randomize(scalar_t radius, scalar_t hole_radius);
  void randomize_by_type(scalar_t radius, scalar_t hole_radius, int type);
  void randomize_random_type(scalar_t radius, scalar_t hole_radius);

  void copy(Shape *shape);
  void scale(scalar_t s);
  void rotate(scalar_t alpha);
  void symmetrize(scalar_t axis_x, scalar_t axis_y);

};

#endif
