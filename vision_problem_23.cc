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
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
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
    } while(big_shape.overwrites(vignette, x_big, y_big));

    if(!error) {
      big_shape.draw(0, vignette, x_big, y_big);

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
          error = small_shape.overwrites(vignette, x_small, y_small);
          nb_attempts++;
        } while(error && nb_attempts < 10);

        if(!error) {
          // Found it, unfill outside, fill inside and draw
          vignette->replace_value(128, 255);
          vignette->fill(x_big, y_big, 128);

          small_shape.draw(1, vignette, x_small, y_small);

          int nb_attempts = 0;
          do {
            small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
            x_small = int(random_uniform_0_1() * Vignette::width);
            y_small = int(random_uniform_0_1() * Vignette::height);
            error = small_shape.overwrites(vignette, x_small, y_small);
            nb_attempts++;
          } while(error && nb_attempts < 10);
          if(!error) {
            // Found it, unfill and draw
            vignette->replace_value(128, 255);
            small_shape.draw(2, vignette, x_small, y_small);
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
          error = small_shape.overwrites(vignette, x_small, y_small);
          nb_attempts++;
        } while(error && nb_attempts < 10);

        if(!error) {
          small_shape.draw(1, vignette, x_small, y_small);
          vignette->fill(x_small, y_small, 128);
          int nb_attempts = 0;
          do {
            small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
            x_small = int(random_uniform_0_1() * Vignette::width);
            y_small = int(random_uniform_0_1() * Vignette::height);
            error = small_shape.overwrites(vignette, x_small, y_small);
            nb_attempts++;
          } while(error && nb_attempts < 10);

          if(!error) {
            small_shape.draw(2, vignette, x_small, y_small);
            vignette->replace_value(128, 255);
          }
        }
      }

    }

  } while(error);
}
