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

#include "vision_problem_6.h"
#include "shape.h"

VisionProblem_6::VisionProblem_6() { }

void VisionProblem_6::generate(int label, Vignette *vignette) {
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

    scalar_t xc1, yc1, alpha1;
    scalar_t xc2, yc2, alpha2;
    scalar_t r;
    shape_number[0] = 0;
    shape_number[1] = 0;
    shape_number[2] = 1;
    shape_number[3] = 1;
    do {
      if(label == 1) {
        xc1 = random_uniform_0_1() * (Vignette::width - part_size) ;
        yc1 = random_uniform_0_1() * (Vignette::width - part_size) ;
        alpha1 = random_uniform_0_1() * M_PI * 2;
        r = random_uniform_0_1() * (Vignette::width + Vignette::height)/2;

        xc2 = random_uniform_0_1() * (Vignette::width - part_size) ;
        yc2 = random_uniform_0_1() * (Vignette::width - part_size) ;
        alpha2 = random_uniform_0_1() * M_PI * 2;

        xs[0] = int(xc1 + r * cos(alpha1));
        ys[0] = int(yc1 + r * sin(alpha1));
        xs[1] = int(xc1 - r * cos(alpha1));
        ys[1] = int(yc1 - r * sin(alpha1));
        xs[2] = int(xc2 + r * cos(alpha2));
        ys[2] = int(yc2 + r * sin(alpha2));
        xs[3] = int(xc2 - r * cos(alpha2));
        ys[3] = int(yc2 - r * sin(alpha2));
        too_ambiguous = 0;
      } else {
        for(int n = 0; n < nb_shapes; n++) {
          xs[n] = int(random_uniform_0_1() * (Vignette::width - part_size));
          ys[n] = int(random_uniform_0_1() * (Vignette::width - part_size));
        }
        scalar_t d1 = sqrt(sq(xs[0] - xs[1]) + sq(ys[0] - ys[1]));
        scalar_t d2 = sqrt(sq(xs[2] - xs[3]) + sq(ys[2] - ys[3]));
        too_ambiguous = abs(d1 - d2) < scalar_t(part_size);
      }
    } while(too_ambiguous ||
            cluttered_shapes(part_size, nb_shapes, xs, ys));

    vignette->clear();

    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      if(shape_number[n] == 0) {
        error |= vignette->overwrites(&shape1, xs[n], ys[n]);
        if(!error) {
          vignette->store_and_draw(&shape1, xs[n], ys[n], 0,
                                   0, part_size / 2, 0);
        }
      } else {
        error |= vignette->overwrites(&shape2, xs[n], ys[n]);
        if(!error) {
          vignette->store_and_draw(&shape2, xs[n], ys[n], 1,
                                   0, part_size / 2, 0);
        }
      }
    }
  } while(error);
}
