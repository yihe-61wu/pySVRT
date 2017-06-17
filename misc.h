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

#ifndef MISC_H
#define MISC_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <cfloat>
#include <stdlib.h>
#include <string.h>

using namespace std;

typedef double scalar_t;
// typedef float scalar_t;

const int buffer_size = 1024;

using namespace std;

#ifdef DEBUG
#define ASSERT(x) if(!(x)) { \
  std::cerr << "ASSERT FAILED IN " << __FILE__ << ":" << __LINE__ << endl; \
  abort(); \
}
#else
#define ASSERT(x)
#endif

template<class T>
T **allocate_array(int a, int b) {
  T *tmp = new T[a * b];
  T **array = new T *[a];
  for(int k = 0; k < a; k++) {
    array[k] = tmp;
    tmp += b;
  }
  return array;
}

template<class T>
void deallocate_array(T **array) {
  delete[] array[0];
  delete[] array;
}

template <class T>
void write_var(ostream *os, const T *x) { os->write((char *) x, sizeof(T)); }

template <class T>
void read_var(istream *is, T *x) { is->read((char *) x, sizeof(T)); }

template <class T>
inline T sq(T x) {
  return x * x;
}

inline scalar_t log2(scalar_t x) {
  return log(x)/log(2.0);
}

template <class T>
void grow(int *nb_max, int nb, T** current, int factor) {
  ASSERT(*nb_max > 0);
  if(nb == *nb_max) {
    T *tmp = new T[*nb_max * factor];
    memcpy(tmp, *current, *nb_max * sizeof(T));
    delete[] *current;
    *current = tmp;
    *nb_max *= factor;
  }
}

#endif
