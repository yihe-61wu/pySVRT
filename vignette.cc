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
 *  along with pysvrt.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "vignette.h"

void Vignette::clear() {
  for(int k = 0; k < width * height; k++) {
    content[k] = 255;
#ifdef KEEP_PART_PRESENCE
    part_presence[k] = 0;
#endif
  }
}

void Vignette::fill(int x, int y, int v) {
  if(x >= 0 && x < Vignette::width && y >= 0 && y < Vignette::height &&
     content[x + Vignette::width * y] == 255) {
    content[x + Vignette::width * y] = v;
    fill(x + 1, y    , v);
    fill(x - 1, y    , v);
    fill(x    , y + 1, v);
    fill(x    , y - 1, v);
  }
}

void Vignette::switch_values(int v1, int v2) {
  for(int k = 0; k < Vignette::height * Vignette::width; k++) {
    if(content[k] == v1) {
      content[k] = v2;
    } else if(content[k] == v2) {
      content[k] = v1;
    }
  }
}

void Vignette::replace_value(int from, int to) {
  for(int k = 0; k < Vignette::height * Vignette::width; k++) {
    if(content[k] == from) {
      content[k] = to;
    }
  }
}

void Vignette::superpose(Vignette *infront, Vignette *inback) {
  for(int k = 0; k < Vignette::height * Vignette::width; k++) {
    if(infront->content[k] < 255) {
      content[k] = infront->content[k];
    } else {
      content[k] = inback->content[k];
    }
  }
}

int Vignette::intersection(Vignette *v) {
  int n = 0;
  for(int k = 0; k < Vignette::height * Vignette::width; k++) {
    if(content[k] < 255 && v->content[k] < 255) {
      n++;
    }
  }
  return n;
}

void Vignette::grow() {
  int tmp[Vignette::width * Vignette::height];
  for(int k = 0; k < Vignette::height * Vignette::width; k++) {
    tmp[k] = content[k];
  }
  int k;
  for(int y = 1; y < Vignette::height - 1; y++) {
    for(int x = 1; x < Vignette::width - 1; x++) {
      k = x + Vignette::width * y;
      content[k] = min(tmp[k],
                       min(min(tmp[k - Vignette::width], tmp[k - 1]),
                           min(tmp[k + 1], tmp[k + Vignette::width])));
    }
  }
}










































































































