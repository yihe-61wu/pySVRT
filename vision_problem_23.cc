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

#include "vision_problem_23.h"
#include "shape.h"


VisionProblem_23::VisionProblem_23() { }

void VisionProblem_23::generate(int label, Vignette *vignette) {
  int x_big, y_big, x_small, y_small;
  Shape big_shape, small_shape;

  int error;
  do {

    vignette->clear();

    error = 0;

    big_shape.randomize(big_part_size / 2, big_part_hole_size / 2);

    do {
      x_big = int(random_uniform_0_1() * Vignette::width);
      y_big = int(random_uniform_0_1() * Vignette::height);
    } while(vignette->overwrites(&big_shape, x_big, y_big));

    if(!error) {
      vignette->store_and_draw(0, &big_shape, x_big, y_big, 0,
                               0, big_part_size / 2, 0);

      if(label) {
        // We fill outside
        vignette->fill(x_big, y_big, 128);
        vignette->switch_values(128, 255);
        // Find a location for a small shape inside
        int nb_attempts = 0;
        do {
          small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
          x_small = int(random_uniform_0_1() * Vignette::width);
          y_small = int(random_uniform_0_1() * Vignette::height);
          error = vignette->overwrites(&small_shape, x_small, y_small);
          nb_attempts++;
        } while(error && nb_attempts < 10);

        if(!error) {
          // Found it, unfill outside, fill inside and draw
          vignette->replace_value(128, 255);
          vignette->fill(x_big, y_big, 128);

          vignette->store_and_draw(1, &small_shape, x_small, y_small, 1,
                                   0, small_part_size / 2, 0);

          int nb_attempts = 0;
          do {
            small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
            x_small = int(random_uniform_0_1() * Vignette::width);
            y_small = int(random_uniform_0_1() * Vignette::height);
            error = vignette->overwrites(&small_shape, x_small, y_small);
            nb_attempts++;
          } while(error && nb_attempts < 10);
          if(!error) {
            // Found it, unfill and draw
            vignette->replace_value(128, 255);
            vignette->store_and_draw(2, &small_shape, x_small, y_small, 2,
                                     0, small_part_size / 2, 0);
          }
        }
      } else {
        vignette->fill(x_big, y_big, 128);

        if(random_uniform_0_1() < 0.5) {
          vignette->switch_values(128, 255);
        }

        int nb_attempts = 0;
        do {
          small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
          x_small = int(random_uniform_0_1() * Vignette::width);
          y_small = int(random_uniform_0_1() * Vignette::height);
          error = vignette->overwrites(&small_shape, x_small, y_small);
          nb_attempts++;
        } while(error && nb_attempts < 10);

        if(!error) {
          vignette->store_and_draw(1, &small_shape, x_small, y_small, 1,
                                   0, small_part_size / 2, 0);
          vignette->fill(x_small, y_small, 128);
          int nb_attempts = 0;
          do {
            small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
            x_small = int(random_uniform_0_1() * Vignette::width);
            y_small = int(random_uniform_0_1() * Vignette::height);
            error = vignette->overwrites(&small_shape, x_small, y_small);
            nb_attempts++;
          } while(error && nb_attempts < 10);

          if(!error) {
            vignette->store_and_draw(2, &small_shape, x_small, y_small, 2,
                                     0, small_part_size / 2, 0);
            vignette->replace_value(128, 255);
          }
        }
      }

    }

  } while(error);
}
