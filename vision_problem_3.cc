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

#include "vision_problem_3.h"
#include "shape.h"

VisionProblem_3::VisionProblem_3() { }

void VisionProblem_3::generate(int label, Vignette *vignette) {
  int nb_shapes = 4;
  Vignette avoid, tmp;
  const int dist_min = Vignette::width / 8;

  int nb_attempts;
  const int max_nb_attempts = 100;

  do {
    avoid.clear();
    vignette->clear();

    nb_attempts = 0;

    for(int s = 0; nb_attempts < max_nb_attempts && s < nb_shapes; s++) {
      Shape shape;

      int xs, ys, i, proper_margin, proper_connection;

      do {
        tmp.clear();
        do {
          do {
            xs = int(random_uniform_0_1() * Vignette::width);
            ys = int(random_uniform_0_1() * Vignette::height);
            shape.randomize(part_size, hole_size);
          } while(tmp.overwrites(&shape, xs, ys)); // check not out-of-vignette

          // omg this is ugly
          if(label && s == 1) {
            proper_margin = 1;
          } else {
            proper_margin = !avoid.overwrites(&shape, xs, ys);
          }

          if((label && (s == 1 || s == 3)) || (!label && (s >= 2))) {
            proper_connection = vignette->overwrites(&shape, xs, ys);
          } else {
            proper_connection = 1;
          }

          nb_attempts++;

        } while(nb_attempts < max_nb_attempts && !proper_margin);

        tmp.draw(s, &shape, xs, ys);
        tmp.fill(xs, ys, 128);

        if(proper_margin && proper_connection) {
          if((label && (s == 1 || s == 3)) || (!label && (s >= 2))) {
            i = vignette->intersection(&tmp);
            proper_connection = (i > 0) && (i < 4);
          } else {
            proper_connection = 1;
          }
        } else {
          proper_connection = 0; // To avoid compilation warning
        }
      } while(nb_attempts < max_nb_attempts && (!proper_margin || !proper_connection));

      if(nb_attempts < max_nb_attempts) {
        vignette->draw(s, &shape, xs, ys);
        vignette->fill(xs, ys, 128);
        if((label && s < 2) || (!label && s < 1)) {
          for(int k = 0; k < dist_min; k++) tmp.grow();
          avoid.superpose(&avoid, &tmp);
        }
      }
    }
  } while(nb_attempts >= max_nb_attempts);

  vignette->replace_value(128, 255);
}
