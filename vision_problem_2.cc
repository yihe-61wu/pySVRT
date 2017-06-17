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

#include "vision_problem_2.h"
#include "shape.h"

VisionProblem_2::VisionProblem_2() { }

void VisionProblem_2::generate(int label, Vignette *vignette) {
  int x_big, y_big, x_small, y_small;
  Shape big_shape, small_shape;
  Vignette mask;
  int nb_attempts, max_nb_attempts = 10;
  int dist_min = Vignette::width/8;

  do {
    vignette->clear();
    mask.clear();

    big_shape.randomize(big_part_size / 2, big_part_hole_size / 2);

    do {
      x_big = int(random_uniform_0_1() * Vignette::width);
      y_big = int(random_uniform_0_1() * Vignette::height);
    } while(big_shape.overwrites(vignette, x_big, y_big));

    // The mask will encode either a thin area the small shape should
    // intersect with (class 1) or a thick one it should not (class 0)

    big_shape.draw(0, &mask, x_big, y_big);

    if(label) {
      mask.grow();
    } else {
      for(int k = 0; k < dist_min; k++) {
        mask.grow();
      }
    }

    big_shape.draw(0, vignette, x_big, y_big);
    vignette->fill(x_big, y_big, 128);
    vignette->switch_values(128, 255);

    nb_attempts = 0;
    do {
      do {
        small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
        x_small = x_big + int((random_uniform_0_1() - 0.5) * big_part_size);
        y_small = y_big + int((random_uniform_0_1() - 0.5) * big_part_size);
      } while(small_shape.overwrites(vignette, x_small, y_small)); // ||
      nb_attempts++;
    } while(nb_attempts < max_nb_attempts &&
            ((label && !small_shape.overwrites(&mask, x_small, y_small)) ||
             (!label && small_shape.overwrites(&mask, x_small, y_small))));

    vignette->replace_value(128, 255);
    small_shape.draw(1, vignette, x_small, y_small);
  } while(nb_attempts >= max_nb_attempts);
}
