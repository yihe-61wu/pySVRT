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

#include "vision_problem_8.h"
#include "shape.h"

VisionProblem_8::VisionProblem_8() { }

void VisionProblem_8::generate(int label, Vignette *vignette) {
  int x_big, y_big, x_small, y_small;
  Shape big_shape, small_shape;

  int error;
  do {

    vignette->clear();

    error = 0;

    big_shape.randomize(big_part_size / 2, big_part_hole_size / 2);
    small_shape.copy(&big_shape);
    small_shape.scale(0.5);
    int small_shapeness = 0;

    do {
      x_big = int(random_uniform_0_1() * Vignette::width);
      y_big = int(random_uniform_0_1() * Vignette::height);
    } while(vignette->overwrites(&big_shape, x_big, y_big));

    if(!error) {
      vignette->store_and_draw(&big_shape, x_big, y_big, 0,
                               0, big_part_size / 2, 0);

      vignette->fill(x_big, y_big, 128);

      if(label) {
        vignette->switch_values(128, 255);
      } else {
        if(random_uniform_0_1() < 0.5) {
          vignette->switch_values(128, 255);
          Shape tmp;
          tmp.randomize(big_part_size / 2, big_part_hole_size / 2);
          small_shape.copy(&tmp);
          small_shape.scale(0.5);
          small_shapeness = 1;
        }
      }

      int nb_attempts = 0;
      do {
        x_small = int(random_uniform_0_1() * Vignette::width);
        y_small = int(random_uniform_0_1() * Vignette::height);
        error = vignette->overwrites(&small_shape, x_small, y_small);
        nb_attempts++;
      } while(error && nb_attempts < 10);

      if(!error) {
        vignette->replace_value(128, 255);
        vignette->store_and_draw(&small_shape, x_small, y_small, small_shapeness,
                                 0, big_part_size / 4, 0);
      }
    }

  } while(error);
}
