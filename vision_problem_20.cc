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

#include "vision_problem_20.h"
#include "shape.h"

VisionProblem_20::VisionProblem_20() { }


void VisionProblem_20::generate(int label, Vignette *vignette) {
  int nb_shapes = 2;
  int xs[nb_shapes], ys[nb_shapes];
  int shapeness[nb_shapes], mirroredness[nb_shapes];
  scalar_t angles[nb_shapes];
  Shape shapes[nb_shapes];

  int error;
  do{
    vignette->clear();
    error = 0;
    for(int n = 0; !error && n < nb_shapes; n++) {
      xs[n] = int(random_uniform_0_1() * Vignette::width);
      ys[n] = int(random_uniform_0_1() * Vignette::height);
      if(!label || n == 0) {
        shapes[n].randomize(part_size / 2, part_hole_size / 2);
        shapeness[n] = 0;
        mirroredness[n] = 0;
        angles[n] = 0;
      } else {
        shapes[n].copy(&shapes[0]);
        shapes[n].symmetrize(ys[n] - ys[0], - xs[n] + xs[0]);
        shapeness[n] = 0;
        // Deconstruct the reflection into a horizontal reflection, followed
        // by a rotation
        mirroredness[n] = 1;
        angles[n] = M_PI - 2 * atan2(ys[n] - ys[0], - xs[n] + xs[0]);
      }
      error |= vignette->overwrites(&shapes[n], xs[n], ys[n]);
      if(!error) {
        vignette->store_and_draw(n, &shapes[n], xs[n], ys[n], shapeness[n],
                                 angles[n], part_size / 2, mirroredness[n]);
      }
    }
  }  while(error);
}
