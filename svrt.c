
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

THByteTensor *generate_vignettes(long n_problem, long nb_vignettes) {
  struct VignetteSet vs;
  long st0, st1, st2;
  long v, i, j;
  unsigned char *a, *b;

  svrt_generate_vignettes(n_problem, nb_vignettes, &vs);
  printf("SANITY %d %d %d\n", vs.nb_vignettes, vs.width, vs.height);

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

  return result;
}
