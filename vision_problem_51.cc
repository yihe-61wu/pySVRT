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

#include "vision_problem_51.h"
#include "shape.h"

VisionProblem_51::VisionProblem_51() { }

void VisionProblem_51::generate(int label, Vignette *vignette) {
  int max_nb_shapes = 8;
  int min_nb_shapes = 5;
  int nb_shapes, xs[max_nb_shapes], ys[max_nb_shapes];
  scalar_t scales[max_nb_shapes], angles[max_nb_shapes];
  scalar_t r0, r1, t0, t1;
  Shape shapes[max_nb_shapes];

  int error;
  do {

    scalar_t max_scale = 2.5;

    // Decide how many shapes we will make.
    nb_shapes = min_nb_shapes;
    nb_shapes += int(random_uniform_0_1() * (max_nb_shapes - min_nb_shapes));

    // First, put a shape in the centre.
    r0 = random_uniform_0_1() * centre_radius;
    t0 = random_uniform_0_1() * M_PI * 2;
    xs[0] = int(Vignette::width / 2 + r0 * cos(t0));
    ys[0] = int(Vignette::height / 2 + r0 * sin(t0));
    scales[0] = max_scale;
    shapes[0].randomize(max_scale * part_size / 2, max_scale * hole_size/2);

    // Now put shapes around the outside.
    // First, select the radius of the ring to assemble them on
    r1 = ring_inner_radius;
    r1 += random_uniform_0_1() * (ring_outer_radius - ring_inner_radius);

    // Now put these shapes randomly around this circle
    int num_conseq = 1;
    for(int n = 1; n < nb_shapes; n++) {
      t1 = random_uniform_0_1() * M_PI * 2;
      xs[n] = int(xs[0] + r1 * cos(t1));
      ys[n] = int(ys[0] + r1 * sin(t1));

      scales[n] = max_scale;
      angles[n] = random_uniform_0_1() * M_PI * 2;

      // If this is the first surrounding shape, we have to take care.
      // To be in class, the centre shape must be unique, if out of class it
      // must be repeated.
      if(n == 1) {
        if(label == 1) {
          shapes[n].randomize(max_scale * part_size / 2, max_scale * hole_size/2);
          num_conseq = 1;
        } else {
          shapes[n].copy(&shapes[0]);
          angles[n] = angles[n - 1];
          num_conseq = 2;
        }
      } else if(num_conseq < 2 || (num_conseq < 4 && random_uniform_0_1() > 0.4) ) {
        shapes[n].copy(&shapes[n - 1]);
        angles[n] = angles[n - 1];
        num_conseq++;
      } else {
        shapes[n].randomize(max_scale * part_size / 2, max_scale * hole_size/2);
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
      error |= vignette->overwrites(&shapes[n], xs[n], ys[n]);
      if(!error) {
        vignette->draw(n, &shapes[n], xs[n], ys[n]);
      }
    }
  } while(error);
}
