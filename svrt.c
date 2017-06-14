
/*
 *  svrt is the ``Synthetic Visual Reasoning Test'', an image
 *  generator for evaluating classification performance of machine
 *  learning systems, humans and primates.
 *
 *  Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
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

#include <TH/TH.h>

#include "svrt_generator.h"

THByteTensor *generate_vignettes(long n_problem, THLongTensor *labels) {
  struct VignetteSet vs;
  long nb_vignettes;
  long st0, st1, st2;
  long v, i, j;
  long *m, *l;
  unsigned char *a, *b;

  if(THLongTensor_nDimension(labels) != 1) {
    printf("Label tensor has to be of dimension 1.\n");
    exit(1);
  }

  nb_vignettes = THLongTensor_size(labels, 0);
  m = THLongTensor_storage(labels)->data + THLongTensor_storageOffset(labels);
  st0 = THLongTensor_stride(labels, 0);
  l = (long *) malloc(sizeof(long) * nb_vignettes);
  for(v = 0; v < nb_vignettes; v++) {
    l[v] = *m;
    m += st0;
  }

  svrt_generate_vignettes(n_problem, nb_vignettes, l, &vs);
  free(l);

  THLongStorage *size = THLongStorage_newWithSize(3);
  size->data[0] = vs.nb_vignettes;
  size->data[1] = vs.height;
  size->data[2] = vs.width;

  THByteTensor *result = THByteTensor_newWithSize(size, NULL);
  THLongStorage_free(size);

  st0 = THByteTensor_stride(result, 0);
  st1 = THByteTensor_stride(result, 1);
  st2 = THByteTensor_stride(result, 2);

  unsigned char *r = vs.data;
  for(v = 0; v < vs.nb_vignettes; v++) {
    a = THByteTensor_storage(result)->data + THByteTensor_storageOffset(result) + v * st0;
    for(i = 0; i < vs.height; i++) {
      b = a + i * st1;
      for(j = 0; j < vs.width; j++) {
        *b = (unsigned char) (*r);
        r++;
        b += st2;
      }
    }
  }

  free(vs.data);

  return result;
}
