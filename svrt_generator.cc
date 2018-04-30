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
 *  along with svrt.  If not, see <http://www.gnu.org/licenses/>.
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
#include "vision_problem_101.h"
#include "vision_problem_201.h"
#include "vision_problem_301.h"
#include "vision_problem_401.h"
#include "vision_problem_501.h"
#include "vision_problem_601.h"
#include "vision_problem_901.h"
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
#include "vision_problem_51.h"
#include "vision_problem_52.h"
#include "vision_problem_151.h"
#include "vision_problem_152.h"

#define NB_PROBLEMS 9999

VignetteGenerator *new_generator(int nb) {
  VignetteGenerator *generator;

  switch(nb) {
  case 1:
    generator = new VisionProblem_1();
    break;
  case 101:
    generator = new VisionProblem_101();
    break;
  case 201:
    generator = new VisionProblem_201();
    break;
  case 301:
    generator = new VisionProblem_301();
    break;
  case 401:
    generator = new VisionProblem_401();
    break;
  case 501:
    generator = new VisionProblem_501();
    break;
  case 601:
    generator = new VisionProblem_601();
    break;
  case 901:
    generator = new VisionProblem_901();
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
  case 51:
    generator = new VisionProblem_51();
    break;
  case 52:
    generator = new VisionProblem_52();
    break;
  case 151:
    generator = new VisionProblem_151();
    break;
  case 152:
    generator = new VisionProblem_152();
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
  int max_shapes;
  int nb_symbolic_outputs;
  unsigned char *nb_shapes_each;
  float *shapes_symb_output;
  uint *shape_is_bordering;
  uint *shape_is_containing;
};

void svrt_generate_vignettes(int n_problem, int nb_vignettes, long *labels,
                             VignetteSet *result) {
  Vignette tmp;

  if(n_problem < 1 || n_problem > NB_PROBLEMS) {
    printf("Problem number should be between 1 and %d. Provided value is %d.\n", NB_PROBLEMS, n_problem);
    exit(1);
  }

  VignetteGenerator *vg = new_generator(n_problem);
  result->n_problem = n_problem;
  result->nb_vignettes = nb_vignettes;
  result->width = Vignette::width;
  result->height = Vignette::height;
  result->data = (unsigned char *) malloc(sizeof(unsigned char) * result->nb_vignettes * result->width * result->height);
  result->max_shapes = Vignette::max_shapes;
  result->nb_symbolic_outputs = Vignette::nb_symbolic_outputs;
  result->nb_shapes_each = (unsigned char *) malloc(sizeof(unsigned char) * result->nb_vignettes);
  result->shapes_symb_output = (float *) malloc(sizeof(float) * result->nb_vignettes * result->max_shapes * result->nb_symbolic_outputs);
  result->shape_is_bordering = (uint *) malloc(sizeof(uint) * result->nb_vignettes * result->max_shapes * result->max_shapes);
  result->shape_is_containing = (uint *) malloc(sizeof(uint) * result->nb_vignettes * result->max_shapes * result->max_shapes);

  unsigned char *s = result->data;
  unsigned char *out_pointer_nb_shapes = result->nb_shapes_each;
  float *out_pointer_shapes_symb_output = result->shapes_symb_output;
  uint *out_pointer_shape_is_bordering = result->shape_is_bordering;
  uint *out_pointer_shape_is_containing = result->shape_is_containing;
  for(int i = 0; i < nb_vignettes; i++) {
    if(labels[i] == 0 || labels[i] == 1) {
      vg->generate(labels[i], &tmp);
    } else {
      printf("Vignette class label has to be 0 or 1. Provided value is %ld.\n", labels[i]);
      exit(1);
    }

    int *r = tmp.content;
    for(int k = 0; k < Vignette::width * Vignette::height; k++) {
      *s++ = *r++;
    }

    *out_pointer_nb_shapes++ = tmp.nb_shapes;

    float *in_pointer_shapes_symb_output = tmp.shapes_symb_output;
    for(int k = 0; k < result->max_shapes * result->nb_symbolic_outputs; k++) {
      *out_pointer_shapes_symb_output++ = in_pointer_shapes_symb_output[k];
    }

    uint *in_pointer_shape_is_bordering = tmp.shape_is_bordering;
    uint *in_pointer_shape_is_containing = tmp.shape_is_containing;
    for(int k = 0; k < result->max_shapes * result->max_shapes; k++) {
      *out_pointer_shape_is_bordering++ = in_pointer_shape_is_bordering[k];
      *out_pointer_shape_is_containing++ = in_pointer_shape_is_containing[k];
    }
  }

  delete vg;
}

}
