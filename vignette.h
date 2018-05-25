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

#ifndef VIGNETTE_H
#define VIGNETTE_H

#include "misc.h"
#include "shape.h"

#define KEEP_PART_PRESENCE

class Vignette {
  int overwrites(Shape *shape, scalar_t xc, scalar_t yc, int first, int nb);
  void draw(int part_number, Shape *shape, scalar_t xc, scalar_t yc, int first, int nb);

public:
  static const int width = 128;
  static const int height = width;
  static const int nb_grayscales = 256;
  static const int max_shapes = 8;
  static const int nb_symbolic_outputs = 6;  // x, y, shape_id, rotation, scale, is_mirrored

  int content[width * height];
#ifdef KEEP_PART_PRESENCE
  unsigned int part_presence[width * height];
#endif

  int nb_shapes;
  scalar_t shapes_xs[max_shapes];
  scalar_t shapes_ys[max_shapes];

  float shapes_symb_output[max_shapes * nb_symbolic_outputs];
  unsigned char intershape_distance[max_shapes * max_shapes];
  float shape_is_containing[max_shapes * max_shapes];

  void clear();

  void fill(int x, int y, int v);
  void switch_values(int c1, int c2);
  void replace_value(int from, int to);
  void superpose(Vignette *infront, Vignette *inback);
  int intersection(Vignette *v);
  void grow();
  void extract_part(int part_id, int *output);

  int overwrites(Shape *shape, scalar_t xc, scalar_t yc);
  void draw(int part_number, Shape *shape, scalar_t xc, scalar_t yc);
  void store_and_draw(Shape *shape, scalar_t xc, scalar_t yc,
                      int shapeness, float rot, float scale, int is_mirrored);
  void check_bordering();
  void check_containing();

};

#endif
