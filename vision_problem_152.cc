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

#include "vision_problem_152.h"
#include "shape.h"

VisionProblem_152::VisionProblem_152() { }

void VisionProblem_152::generate(int label, Vignette *vignette) {
  int max_nb_shapes = 8;
  int min_nb_shapes = 5;
  int nb_shapes, xs[max_nb_shapes], ys[max_nb_shapes];
  scalar_t scales[max_nb_shapes], angles[max_nb_shapes];
  Shape shapes[max_nb_shapes];

  int error;
  do {

    scalar_t max_scale = 2.5;

    // Decide how many shapes we will make.
    nb_shapes = min_nb_shapes;
    nb_shapes += int(random_uniform_0_1() * (max_nb_shapes - min_nb_shapes));

    // First, put a shape in the exclusive zone.
    xs[0] = int(random_uniform_0_1() * exclusive_length);
    ys[0] = int(random_uniform_0_1() * Vignette::height);
    scales[0] = max_scale;
    shapes[0].randomize_random_type(max_scale * part_size / 2, max_scale * hole_size / 2);

    // Now put other shapes around in random positions.
    int num_conseq = 1;
    for(int n = 1; n < nb_shapes; n++) {
      xs[n] = int(exclusive_length + random_uniform_0_1() * (Vignette::width - exclusive_length));
      ys[n] = int(random_uniform_0_1() * Vignette::height);

      scales[n] = max_scale;
      angles[n] = random_uniform_0_1() * M_PI * 2;

      // If this is the first additional shape, we have to take care.
      // To be in class, at least one of the surrounding shapes must be the
      // same as this one, otherwise none of them can be.
      // Otherwise, we play around with repeating or changing the shape.
      if(n == 1) {
        if(label == 1) {
          shapes[n].randomize_random_type(max_scale * part_size / 2, max_scale * hole_size / 2);
          num_conseq = 1;
        } else {
          shapes[n].copy(&shapes[0]);
          //angles[n] = angles[n - 1];
          num_conseq = 2;
        }
      } else if(num_conseq < 2 || (num_conseq < 4 && random_uniform_0_1() > 0.4) ) {
        shapes[n].copy(&shapes[n - 1]);
        //angles[n] = angles[n - 1];
        num_conseq++;
      } else {
        shapes[n].randomize_random_type(max_scale * part_size / 2, max_scale * hole_size/2);
        num_conseq = 1;
      }
    }

    // Scale and rotate all the shapes.
    for(int n = 0; n < nb_shapes; n++) {
      shapes[n].scale(scales[n] / max_scale);
      shapes[n].rotate(angles[n]);
    }

    vignette->clear();

    error = 0;
    for(int n = 0; n < nb_shapes; n++) {
      error |= shapes[n].overwrites(vignette, xs[n], ys[n]);
      if(!error) {
        shapes[n].draw(n, vignette, xs[n], ys[n]);
      }
    }
  } while(error);
}
