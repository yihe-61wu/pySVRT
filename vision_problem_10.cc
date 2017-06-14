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

#include "vision_problem_10.h"
#include "shape.h"

VisionProblem_10::VisionProblem_10() { }

void VisionProblem_10::generate(int label, Vignette *vignette) {
  int nb_shapes = 4;
  int xs[nb_shapes], ys[nb_shapes];

  int error;
  do {
    do {
      if(label == 1) {
        scalar_t alpha = random_uniform_0_1() * 2 * M_PI;
        scalar_t radius = random_uniform_0_1() * Vignette::width / 2;
        scalar_t xc = random_uniform_0_1() * Vignette::width;
        scalar_t yc = random_uniform_0_1() * Vignette::height;
        for(int n = 0; n < nb_shapes; n++) {
          xs[n] = int(xc + part_size/2 +
                      radius * sin(alpha + n * 2 * M_PI / scalar_t(nb_shapes)));
          ys[n] = int(yc + part_size/2 +
                      radius * cos(alpha + n * 2 * M_PI / scalar_t(nb_shapes)));
        }
      } else {
        for(int n = 0; n < nb_shapes; n++) {
          xs[n] = int(random_uniform_0_1() * Vignette::width);
          ys[n] = int(random_uniform_0_1() * Vignette::height);
        }
      }
    } while(cluttered_shapes(part_size, nb_shapes, xs, ys));

    vignette->clear();

    Shape shape;
    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      if(n == 0) {
        shape.randomize(part_size / 2, part_hole_size / 2);
      }
      error |= shape.overwrites(vignette, xs[n], ys[n]);
      if(!error) {
        shape.draw(n, vignette, xs[n], ys[n]);
      }
    }
  } while(error);
}
