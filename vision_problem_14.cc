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

#include "vision_problem_14.h"
#include "shape.h"

VisionProblem_14::VisionProblem_14() { }

void VisionProblem_14::sample_shapes_positions_aligned(int nb_shapes, int *xs, int *ys) {
  for(int n = 0; n < nb_shapes - 1; n++) {
    xs[n] = int(random_uniform_0_1() * (Vignette::width - part_size + 1)) + part_size/2;
    ys[n] = int(random_uniform_0_1() * (Vignette::height - part_size + 1)) + part_size/2;
  }
  scalar_t alpha = random_uniform_0_1();
  xs[nb_shapes - 1] = int(alpha * xs[0] + (1 - alpha) * xs[1]);
  ys[nb_shapes - 1] = int(alpha * ys[0] + (1 - alpha) * ys[1]);
}

void VisionProblem_14::sample_shapes_positions_uniformly(int nb_shapes, int *xs, int *ys) {
  for(int n = 0; n < nb_shapes; n++) {
    xs[n] = int(random_uniform_0_1() * (Vignette::width - part_size + 1)) + part_size/2;
    ys[n] = int(random_uniform_0_1() * (Vignette::height - part_size + 1)) + part_size/2;
  }
}

int VisionProblem_14::there_is_an_alignment(scalar_t dist_threshold, int nb_shapes, int *xs, int *ys) {
  for(int a = 0; a < nb_shapes; a++) {
    for(int b = 0; b < nb_shapes; b++) {
      for(int c = 0; c < b; c++) {
        if(a != b && a != c) {
          if(point_in_band(scalar_t(xs[a]), scalar_t(ys[a]),
                           scalar_t(xs[b]), scalar_t(ys[b]),
                           scalar_t(xs[c]), scalar_t(ys[c]),
                           dist_threshold)) return 1;
        }
      }
    }
  }
  return 0;
}


void VisionProblem_14::generate(int label, Vignette *vignette) {
  int nb_shapes = 3;
  int xs[nb_shapes], ys[nb_shapes];
  int error;

  do {
    if(label) {
      sample_shapes_positions_aligned(nb_shapes, xs, ys);
    } else {
      do {
        sample_shapes_positions_uniformly(nb_shapes, xs, ys);
      } while(there_is_an_alignment(Vignette::width/20, 3, xs, ys));
    }

    vignette->clear();

    Shape shape;

    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      if(n == 0) {
        shape.randomize(part_size/2, hole_size/2);
      }
      error |= vignette->overwrites(&shape, xs[n], ys[n]);
      if(!error) {
        vignette->store_and_draw(&shape, xs[n], ys[n], 0,
                                 0, part_size / 2, 0);
      }
    }
  } while(error);
}
