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

#include "vision_problem_401.h"
#include "shape.h"

VisionProblem_401::VisionProblem_401() { }

void VisionProblem_401::generate(int label, Vignette *vignette) {
  int nb_shapes = 2;
  int xs[nb_shapes], ys[nb_shapes];
  scalar_t scales[nb_shapes], angles[nb_shapes];
  Shape shapes[nb_shapes];

  int error;
  do {

    scalar_t max_scale = -1;

    for(int n = 0; n < nb_shapes; n++) {
      xs[n] = int(random_uniform_0_1() * Vignette::width);
      ys[n] = int(random_uniform_0_1() * Vignette::height);

        scales[n] = 2.5;

      if(n == 0 || scales[n] > max_scale) max_scale = scales[n];

      angles[n] = 0;
    }

    for(int n = 0; n < nb_shapes; n++) {
      if(n == 0 || label == 0) {
        shapes[n].randomize_by_type(
          max_scale * part_size / 2,
          max_scale * hole_size / 2,
          4);
      } else {
        shapes[n].copy(&shapes[0]);
      }
    }

    for(int n = 0; n < nb_shapes; n++) {
      shapes[n].scale(scales[n] / max_scale);
      shapes[n].rotate(angles[n]);
    }

    vignette->clear();

    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      error |= vignette->overwrites(&shapes[n], xs[n], ys[n]);
      if(!error) {
        vignette->draw(n, &shapes[n], xs[n], ys[n]);
      }
    }
  } while(error);
}
