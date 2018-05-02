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

#include "vignette.h"

void Vignette::clear() {
  for(int k = 0; k < width * height; k++) {
    content[k] = 255;
#ifdef KEEP_PART_PRESENCE
    part_presence[k] = 0;
#endif
  }
  nb_shapes = 0;
  for(int i = 0; i < max_shapes * nb_symbolic_outputs; i++) {
    shapes_symb_output[i] = -1.0;
  }
  for(int i = 0; i < max_shapes * max_shapes; i++) {
    shape_is_bordering[i] = -1.0;
    shape_is_containing[i] = -1.0;
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


int Vignette::overwrites(Shape *shape, scalar_t xc, scalar_t yc,
                         int n1, int n2) {
  int x1 = int(shape->x_pixels[n1 % shape->nb_pixels] + xc);
  int y1 = int(shape->y_pixels[n1 % shape->nb_pixels] + yc);
  int x2 = int(shape->x_pixels[n2 % shape->nb_pixels] + xc);
  int y2 = int(shape->y_pixels[n2 % shape->nb_pixels] + yc);
  int n3 = (n1 + n2) / 2;

  if(n1 + 1 < n2 && (abs(x1 - x2) > 1 || abs(y1 - y2) > 1)) {
    return
      overwrites(shape, xc, yc, n1, n3) ||
      overwrites(shape, xc, yc, n3, n2);
  }
  if(x1 < shape->margin || x1 >= Vignette::width - shape->margin ||
      y1 < shape->margin || y1 >= Vignette::height - shape->margin) {
    return 1;
  }
  if(shape->margin <= 0) {
    return 0;
  }
  for(int xx = x1 - shape->margin; xx <= x1 + shape->margin; xx++) {
    for(int yy = y1 - shape->margin; yy <= y1 + shape->margin; yy++) {
      if(content[xx + Vignette::width * yy] != 255) {
        return 1;
      }
    }
  }

  return 0;
}

int Vignette::overwrites(Shape *shape, scalar_t xc, scalar_t yc) {
  return
    overwrites(shape, xc, yc, shape->n_pixels1, shape->n_pixels2) ||
    overwrites(shape, xc, yc, shape->n_pixels2, shape->n_pixels3) ||
    overwrites(shape, xc, yc, shape->n_pixels3, shape->n_pixels4) ||
    overwrites(shape, xc, yc, shape->n_pixels4, shape->nb_pixels);
}

void Vignette::draw(int part_number, Shape *shape, scalar_t xc,
                    scalar_t yc, int n1, int n2) {
  int x1 = int(shape->x_pixels[n1 % shape->nb_pixels] + xc);
  int y1 = int(shape->y_pixels[n1 % shape->nb_pixels] + yc);
  int x2 = int(shape->x_pixels[n2 % shape->nb_pixels] + xc);
  int y2 = int(shape->y_pixels[n2 % shape->nb_pixels] + yc);
  int n3 = (n1 + n2) / 2;

  if(n1 + 1 < n2 && (abs(x1 - x2) > 1 || abs(y1 - y2) > 1)) {
    draw(part_number, shape, xc, yc, n1, n3);
    draw(part_number, shape, xc, yc, n3, n2);
  } else {
    if(x1 >= shape->margin && x1 < Vignette::width-shape->margin &&
       y1 >= shape->margin && y1 < Vignette::height-shape->margin) {
      content[x1 + Vignette::width * y1] = 0;
#ifdef KEEP_PART_PRESENCE
      part_presence[x1 + Vignette::width * y1] |= (1 << part_number);
#endif
    } else {
      abort();
    }
  }
}

void Vignette::draw(int part_number, Shape *shape, scalar_t xc, scalar_t yc) {
  draw(part_number, shape, xc, yc, shape->n_pixels1, shape->n_pixels2);
  draw(part_number, shape, xc, yc, shape->n_pixels2, shape->n_pixels3);
  draw(part_number, shape, xc, yc, shape->n_pixels3, shape->n_pixels4);
  draw(part_number, shape, xc, yc, shape->n_pixels4, shape->nb_pixels);
}

void Vignette::store_and_draw(
      int part_number,
      Shape *shape,
      scalar_t xc,
      scalar_t yc,
      int shapeness,
      float rot,
      float scale,
      int is_mirrored) {
  // x, y, shape_id, scale, rotation, is_mirrored
  int offset = nb_symbolic_outputs * nb_shapes;
  shapes_symb_output[offset + 0] = (float) xc;
  shapes_symb_output[offset + 1] = (float) yc;
  shapes_symb_output[offset + 2] = (float) shapeness;
  shapes_symb_output[offset + 3] = (float) rot;
  shapes_symb_output[offset + 4] = (float) scale;
  shapes_symb_output[offset + 5] = (float) is_mirrored;
  // Create a new shape so we know it won't get overwritten.
  Shape _shape_copy;
  _shape_copy.copy(shape);
  shapes[nb_shapes] = &_shape_copy;
  shapes_xs[nb_shapes] = xc;
  shapes_ys[nb_shapes] = yc;
  // Draw the shape
  draw(part_number, shape, xc, yc);
  nb_shapes++;
}
