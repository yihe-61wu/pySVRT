/*
 *  svrt is the ``Synthetic Visual Reasoning Test'', an image
 *  generator for evaluating classification performance of machine
 *  learning systems, humans and primates.
 *
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
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
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#include "random.h"

#include "vision_problem_1.h"
#include "vision_problem_2.h"
#include "vision_problem_3.h"
#include "vision_problem_4.h"
#include "vision_problem_5.h"
#include "vision_problem_6.h"
#include "vision_problem_7.h"
#include "vision_problem_8.h"
#include "vision_problem_9.h"
#include "vision_problem_10.h"
#include "vision_problem_11.h"
#include "vision_problem_12.h"
#include "vision_problem_13.h"
#include "vision_problem_14.h"
#include "vision_problem_15.h"
#include "vision_problem_16.h"
#include "vision_problem_17.h"
#include "vision_problem_18.h"
#include "vision_problem_19.h"
#include "vision_problem_20.h"
#include "vision_problem_21.h"
#include "vision_problem_22.h"
#include "vision_problem_23.h"

#define NB_PROBLEMS 23

VignetteGenerator *new_generator(int nb) {
  VignetteGenerator *generator;

  switch(nb) {
  case 1:
    generator = new VisionProblem_1();
    break;
  case 2:
    generator = new VisionProblem_2();
    break;
  case 3:
    generator = new VisionProblem_3();
    break;
  case 4:
    generator = new VisionProblem_4();
    break;
  case 5:
    generator = new VisionProblem_5();
    break;
  case 6:
    generator = new VisionProblem_6();
    break;
  case 7:
    generator = new VisionProblem_7();
    break;
  case 8:
    generator = new VisionProblem_8();
    break;
  case 9:
    generator = new VisionProblem_9();
    break;
  case 10:
    generator = new VisionProblem_10();
    break;
  case 11:
    generator = new VisionProblem_11();
    break;
  case 12:
    generator = new VisionProblem_12();
    break;
  case 13:
    generator = new VisionProblem_13();
    break;
  case 14:
    generator = new VisionProblem_14();
    break;
  case 15:
    generator = new VisionProblem_15();
    break;
  case 16:
    generator = new VisionProblem_16();
    break;
  case 17:
    generator = new VisionProblem_17();
    break;
  case 18:
    generator = new VisionProblem_18();
    break;
  case 19:
    generator = new VisionProblem_19();
    break;
  case 20:
    generator = new VisionProblem_20();
    break;
  case 21:
    generator = new VisionProblem_21();
    break;
  case 22:
    generator = new VisionProblem_22();
    break;
  case 23:
    generator = new VisionProblem_23();
    break;
  default:
    cerr << "Can not find problem "
         << nb
         << endl;
    abort();
  }

  generator->precompute();

  return generator;
}

extern "C" {

struct VignetteSet {
  int n_problem;
  int nb_vignettes;
  int width;
  int height;
  unsigned char *data;
};

void svrt_generate_vignettes(int n_problem, int nb_vignettes, VignetteSet *result) {
  Vignette tmp;

  VignetteGenerator *vg = new_generator(n_problem);
  result->n_problem = n_problem;
  result->nb_vignettes = nb_vignettes;
  result->width = Vignette::width;
  result->height = Vignette::height;
  result->data = (unsigned char *) malloc(sizeof(unsigned char) * result->nb_vignettes * result->width * result->height);

  unsigned char *s = result->data;
  for(int i = 0; i < nb_vignettes; i++) {
    vg->generate(drand48() < 0.5 ? 1 : 0, &tmp);
    int *r = tmp.content;
    for(int k = 0; k < Vignette::width * Vignette::height; k++) {
      *s++ = *r++;
    }
  }

  delete vg;
}

}
