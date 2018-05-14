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

#include "vision_problem_18.h"
#include "shape.h"

VisionProblem_18::VisionProblem_18() { }

void VisionProblem_18::generate(int label, Vignette *vignette) {
  int error;

  int nb_shapes = 6;
  int xs[nb_shapes], ys[nb_shapes];
  Shape shape1, shape2;
  shape1.randomize(part_size / 2, hole_size / 2);
  shape2.copy(&shape1);

  do {
    vignette->clear();
    error = 0;

    // First half of the shapes are random
    for(int n = 0; n < nb_shapes/2; n++) {
      xs[n] = int(random_uniform_0_1() * (Vignette::width - part_size + 1)) + part_size/2;
      ys[n] = int(random_uniform_0_1() * (Vignette::height - part_size + 1)) + part_size/2;
      error |= vignette->overwrites(&shape1, xs[n], ys[n]);
      if(!error) {
        vignette->store_and_draw(n, &shape1, xs[n], ys[n], 0,
                                 0, part_size / 2, 0);
      }
    }

    for(int n = nb_shapes/2; n < nb_shapes; n++) {
      if(label == 1) {
        xs[n] = Vignette::width - xs[n - nb_shapes/2];
        ys[n] = ys[n - nb_shapes/2];
      } else {
        xs[n] = int(random_uniform_0_1() * (Vignette::width - part_size + 1)) + part_size/2;
        ys[n] = ys[n - nb_shapes/2];
      }
      error |= vignette->overwrites(&shape2, xs[n], ys[n]);
      if(!error) {
        vignette->store_and_draw(n, &shape2, xs[n], ys[n], 0,
                                 0, part_size / 2, 0);
      }
    }
  } while(error);
}
