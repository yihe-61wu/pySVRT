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

#include "vision_problem_9.h"
#include "shape.h"

VisionProblem_9::VisionProblem_9() { }

void VisionProblem_9::generate(int label, Vignette *vignette) {
  int nb_shapes = 3;
  int xs[nb_shapes], ys[nb_shapes];
  Shape big_shape, small_shape;

  int error;

  do {
    scalar_t x1 = int(random_uniform_0_1() * Vignette::width);
    scalar_t y1 = int(random_uniform_0_1() * Vignette::height);
    scalar_t x2 = int(random_uniform_0_1() * Vignette::width);
    scalar_t y2 = int(random_uniform_0_1() * Vignette::height);
    scalar_t alpha = 0.25 + 0.5 * random_uniform_0_1();

    big_shape.randomize(part_size, hole_size);
    small_shape.copy(&big_shape);
    small_shape.scale(0.5);

    if(label == 0) {
      xs[0] = int(x1); ys[0] = int(y1);
      xs[1] = int(x2); ys[1] = int(y2);
      xs[2] = int(alpha * x1 + (1 - alpha) * x2); ys[2] = int(alpha * y1 + (1 - alpha) * y2);
    } else {
      xs[0] = int(x1); ys[0] = int(y1);
      xs[1] = int(alpha * x1 + (1 - alpha) * x2); ys[1] = int(alpha * y1 + (1 - alpha) * y2);
      xs[2] = int(x2); ys[2] = int(y2);
    }

    vignette->clear();

    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      if(n < 2) {
        error |= small_shape.overwrites(vignette, xs[n], ys[n]);
        if(!error) {
          small_shape.draw(n, vignette, xs[n], ys[n]);
        }
      } else {
        error |= big_shape.overwrites(vignette, xs[n], ys[n]);
        if(!error) {
          big_shape.draw(n, vignette, xs[n], ys[n]);
        }
      }
      vignette->fill(xs[n], ys[n], 128);
    }

    vignette->replace_value(128, 255);
  } while(error);
}
