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

#include "vision_problem_13.h"
#include "shape.h"

VisionProblem_13::VisionProblem_13() { }

void VisionProblem_13::generate(int label, Vignette *vignette) {
  Shape big_shape, small_shape;
  int big_xs1, big_ys1, small_xs1, small_ys1;
  int big_xs2, big_ys2, small_xs2, small_ys2;
  int translated_small_xs = 0, translated_small_ys = 0;
  Vignette tmp;
  const int dist_min = Vignette::width/4;
  int nb_attempts;
  const int max_nb_attempts = 100;

  do {
    nb_attempts = 0;
    do {

      vignette->clear();

      big_shape.randomize(big_part_size / 2, big_part_hole_size / 2);

      tmp.clear();
      do {
        big_xs1 = int(random_uniform_0_1() * Vignette::width);
        big_ys1 = int(random_uniform_0_1() * Vignette::height);
        nb_attempts++;
      } while(nb_attempts < max_nb_attempts &&
              vignette->overwrites(&big_shape, big_xs1, big_ys1));

      if(nb_attempts < max_nb_attempts) {
        vignette->draw(0, &big_shape, big_xs1, big_ys1);
        tmp.draw(0, &big_shape, big_xs1, big_ys1);
        for(int k = 0; k < dist_min; k++) tmp.grow();
      }

      do {
        small_shape.randomize(small_part_size / 2, small_part_hole_size / 2);
        small_xs1 = int(random_uniform_0_1() * Vignette::width);
        small_ys1 = int(random_uniform_0_1() * Vignette::height);
        nb_attempts++;
      } while(nb_attempts < max_nb_attempts &&
              (!tmp.overwrites(&small_shape, small_xs1, small_ys1) ||
               vignette->overwrites(&small_shape, small_xs1, small_ys1)));

      if(nb_attempts < max_nb_attempts) {
        vignette->draw(1, &small_shape, small_xs1, small_ys1);
      }

      tmp.clear();
      do {
        big_xs2 = int(random_uniform_0_1() * Vignette::width);
        big_ys2 = int(random_uniform_0_1() * Vignette::height);
        nb_attempts++;
      } while(nb_attempts < max_nb_attempts &&
              vignette->overwrites(&big_shape, big_xs2, big_ys2));
      if(nb_attempts < max_nb_attempts) {
        vignette->draw(2, &big_shape, big_xs2, big_ys2);
        tmp.draw(0, &big_shape, big_xs2, big_ys2);
        for(int k = 0; k < dist_min; k++) tmp.grow();

        translated_small_xs = small_xs1 + (big_xs2 - big_xs1);
        translated_small_ys = small_ys1 + (big_ys2 - big_ys1);
      }
    } while(nb_attempts < max_nb_attempts &&
            vignette->overwrites(&small_shape,
                                 translated_small_xs,
                                 translated_small_ys));

    if(label) {
      small_xs2 = translated_small_xs;
      small_ys2 = translated_small_ys;
    } else {
      do {
        small_xs2 = int(random_uniform_0_1() * Vignette::width);
        small_ys2 = int(random_uniform_0_1() * Vignette::height);
        nb_attempts++;
      } while(nb_attempts < max_nb_attempts &&
              (sq(small_xs2 - translated_small_xs) + sq(small_ys2 - translated_small_ys) < sq(dist_min) ||
               !tmp.overwrites(&small_shape, small_xs2, small_ys2) ||
               vignette->overwrites(&small_shape, small_xs2, small_ys2)));
    }
  } while(nb_attempts >= max_nb_attempts);
  vignette->draw(3, &small_shape, small_xs2, small_ys2);
}
