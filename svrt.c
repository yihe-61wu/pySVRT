
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
 *  along with svrt.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <TH/TH.h>

#include "svrt.h"

#include "svrt_generator.h"

THByteStorage *compress(THByteStorage *x) {
  long k, g, n;

  k = 0; n = 0;
  while(k < x->size) {
    g = 0;
    while(k < x->size && x->data[k] == 255 && g < 255) { g++; k++; }
    n++;
    if(k < x->size && g < 255) { k++; }
  }

  if(x->data[k-1] == 0) {
    n++;
  }

  THByteStorage *result = THByteStorage_newWithSize(n);

  k = 0; n = 0;
  while(k < x->size) {
    g = 0;
    while(k < x->size && x->data[k] == 255 && g < 255) { g++; k++; }
    result->data[n++] = g;
    if(k < x->size && g < 255) { k++; }
  }
  if(x->data[k-1] == 0) {
    result->data[n++] = 0;
  }

  return result;
}

THByteStorage *uncompress(THByteStorage *x) {
  long k, g, n;

  k = 0;
  for(n = 0; n < x->size - 1; n++) {
    k = k + x->data[n];
    if(x->data[n] < 255) { k++; }
  }
  k = k + x->data[n];

  THByteStorage *result = THByteStorage_newWithSize(k);

  k = 0;
  for(n = 0; n < x->size - 1; n++) {
    for(g = 0; g < x->data[n]; g++) {
      result->data[k++] = 255;
    }
    if(x->data[n] < 255) {
      result->data[k++] = 0;
    }
  }
  for(g = 0; g < x->data[n]; g++) {
    result->data[k++] = 255;
  }

  return result;
}

void seed(long s) {
  srand48(s);
}

THByteTensor *generate_vignettes_raw(
      long n_problem, THLongTensor *labels,
      THByteTensor *nb_shapes, THFloatTensor *shape_list,
      THByteTensor *intershape_distance, THFloatTensor *is_containing) {
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


  // alloc tensor
  THByteTensor_resize1d(nb_shapes, vs.nb_vignettes);
  unsigned char *out_pointer_nb_shapes = THByteTensor_data(nb_shapes);
  // convert tensor data
  for (i=0; i<vs.nb_vignettes; i++) {
    *out_pointer_nb_shapes++ = (unsigned char) vs.nb_shapes_each[i];
  }


  // alloc tensor
  THFloatTensor_resize3d(shape_list, vs.nb_vignettes, vs.max_shapes, vs.nb_symbolic_outputs);

  st0 = THFloatTensor_stride(shape_list, 0);
  st1 = THFloatTensor_stride(shape_list, 1);
  st2 = THFloatTensor_stride(shape_list, 2);

  // convert tensor data
  float *in_pointer_shape_list = vs.shapes_symb_output;
  float *out_pointer_shape_list_base, *out_pointer_shape_list;
  for(v = 0; v < vs.nb_vignettes; v++) {
    out_pointer_shape_list_base = THFloatTensor_storage(shape_list)->data
        + THFloatTensor_storageOffset(shape_list) + v * st0;
    for(i = 0; i < vs.max_shapes; i++) {
      out_pointer_shape_list = out_pointer_shape_list_base + i * st1;
      for(j = 0; j < vs.nb_symbolic_outputs; j++) {
        *out_pointer_shape_list = (float) (*in_pointer_shape_list);
        in_pointer_shape_list++;
        out_pointer_shape_list += st2;
      }
    }
  }

  // alloc tensor
  THByteTensor_resize3d(intershape_distance, vs.nb_vignettes, vs.max_shapes, vs.max_shapes);

  st0 = THByteTensor_stride(intershape_distance, 0);
  st1 = THByteTensor_stride(intershape_distance, 1);
  st2 = THByteTensor_stride(intershape_distance, 2);

  // convert tensor data
  unsigned char *in_pointer_intershape_distance = vs.intershape_distance;
  unsigned char *out_pointer_intershape_distance_base, *out_pointer_intershape_distance;
  for(v = 0; v < vs.nb_vignettes; v++) {
    out_pointer_intershape_distance_base = THByteTensor_storage(intershape_distance)->data
        + THByteTensor_storageOffset(intershape_distance) + v * st0;
    for(i = 0; i < vs.max_shapes; i++) {
      out_pointer_intershape_distance = out_pointer_intershape_distance_base + i * st1;
      for(j = 0; j < vs.max_shapes; j++) {
        *out_pointer_intershape_distance = (unsigned char) (*in_pointer_intershape_distance);
        in_pointer_intershape_distance++;
        out_pointer_intershape_distance += st2;
      }
    }
  }

  // alloc tensor
  THFloatTensor_resize3d(is_containing, vs.nb_vignettes, vs.max_shapes, vs.max_shapes);

  st0 = THFloatTensor_stride(is_containing, 0);
  st1 = THFloatTensor_stride(is_containing, 1);
  st2 = THFloatTensor_stride(is_containing, 2);

  // convert tensor data
  float *in_pointer_shape_is_containing = vs.shape_is_containing;
  float *out_pointer_shape_is_containing_base, *out_pointer_shape_is_containing;
  for(v = 0; v < vs.nb_vignettes; v++) {
    out_pointer_shape_is_containing_base = THFloatTensor_storage(is_containing)->data
        + THFloatTensor_storageOffset(is_containing) + v * st0;
    for(i = 0; i < vs.max_shapes; i++) {
      out_pointer_shape_is_containing = out_pointer_shape_is_containing_base + i * st1;
      for(j = 0; j < vs.max_shapes; j++) {
        *out_pointer_shape_is_containing = (float) (*in_pointer_shape_is_containing);
        in_pointer_shape_is_containing++;
        out_pointer_shape_is_containing += st2;
      }
    }
  }

  free(vs.data);

  return result;
}
