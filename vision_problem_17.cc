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

#include "vision_problem_17.h"
#include "shape.h"

VisionProblem_17::VisionProblem_17() { }

void VisionProblem_17::generate(int label, Vignette *vignette) {
  const int nb_shapes = 4;
  int xs[nb_shapes], ys[nb_shapes];
  int shape_number[nb_shapes];

  ASSERT(nb_shapes == 4);

  int too_ambiguous;

  int error;

  do {
    Shape shape1, shape2;
    shape1.randomize(part_size/2, hole_size/2);
    shape2.randomize(part_size/2, hole_size/2);

    //////////////////////////////////////////////////////////////////////

    do {
      for(int n = 0; n < nb_shapes; n++) {
        if(n < nb_shapes - 1) {
          shape_number[n] = 0;
        } else {
          shape_number[n] = 1;
        }
        xs[n] = int(random_uniform_0_1() * (Vignette::width - part_size)) + part_size/2;
        ys[n] = int(random_uniform_0_1() * (Vignette::width - part_size)) + part_size/2;
      }

      scalar_t a = scalar_t(xs[1] - xs[0]), b = scalar_t(ys[1] - ys[0]);
      scalar_t c = scalar_t(xs[2] - xs[1]), d = scalar_t(ys[2] - ys[1]);
      scalar_t det = a * d - b * c;
      scalar_t u = scalar_t(xs[1] * xs[1] - xs[0] * xs[0] + ys[1] * ys[1] - ys[0] * ys[0]);
      scalar_t v = scalar_t(xs[2] * xs[2] - xs[1] * xs[1] + ys[2] * ys[2] - ys[1] * ys[1]);
      scalar_t xc = 1/(2 * det) *(  d * u - b * v);
      scalar_t yc = 1/(2 * det) *(- c * u + a * v);

      if(label == 1) {
        xs[nb_shapes - 1] = int(xc);
        ys[nb_shapes - 1] = int(yc);
        too_ambiguous = 0;
      } else {
        too_ambiguous = sqrt(sq(scalar_t(xs[nb_shapes - 1]) - xc) +
                             sq(scalar_t(ys[nb_shapes - 1]) - yc)) < scalar_t(part_size);
      }
    } while(too_ambiguous ||
            cluttered_shapes(part_size, nb_shapes, xs, ys));

    //////////////////////////////////////////////////////////////////////

    vignette->clear();

    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      if(shape_number[n] == 0) {
        error |= shape1.overwrites(vignette, xs[n], ys[n]);
        if(!error) {
          shape1.draw(n, vignette, xs[n], ys[n]);
        }
      } else {
        error |= shape2.overwrites(vignette, xs[n], ys[n]);
        if(!error) {
          shape2.draw(n, vignette, xs[n], ys[n]);
        }
      }
    }
  } while(error);
}
