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
    intershape_distance[i] = -1.0;
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

void Vignette::extract_part(int part_id, int *output) {
  for(int x; x < Vignette::width * Vignette::height; x++) {
    output[x] = (part_presence[x] & (1 << part_id)) ? 0 : 255;
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
      Shape *shape,
      scalar_t xc,
      scalar_t yc,
      int shapeness,
      float rot,
      float scale,
      int is_mirrored) {
  // x, y, shape_id, rotation, scale, is_mirrored
  int offset = nb_symbolic_outputs * nb_shapes;
  shapes_symb_output[offset + 0] = (float) xc;
  shapes_symb_output[offset + 1] = (float) yc;
  shapes_symb_output[offset + 2] = (float) shapeness;
  shapes_symb_output[offset + 3] = (float) rot;
  shapes_symb_output[offset + 4] = (float) scale;
  shapes_symb_output[offset + 5] = (float) is_mirrored;
  // Store shape center location
  shapes_xs[nb_shapes] = xc;
  shapes_ys[nb_shapes] = yc;
  // Draw the shape
  draw(nb_shapes, shape, xc, yc);
  nb_shapes++;
}

bool any_content_collides(int *content1, int *content2) {
  for(int x; x < Vignette::width * Vignette::height; x++) {
    if(content1[x] < 255 && content2[x] < 255) {
      return true;
    }
  }
  return false;
}

void Vignette::check_bordering() {
  int MAX_DIST = 16;
  Vignette masks[MAX_DIST+1];
  for(int n = 0; n < nb_shapes; n++) {
    masks[0].clear();
    // Extract the relevant shape
    this->extract_part(n, masks[0].content);
    // For mask 0, we leave as is and just check for intersection
    // For the rest, we grow the filled pixels out by 1 in all four cardinal
    // directions, and save after each step, up to the maximum distance we
    // will measure
    for(int k = 1; k <= MAX_DIST; k++) {
      masks[k] = masks[k-1];
      masks[k].grow();
    }
    for(int i = 0; i < nb_shapes; i++) {
      int output = MAX_DIST + 1;
      // Extract the second shape for comarison
      int second_shape_content[width * height];
      this->extract_part(i, second_shape_content);
      // Check which mask first collides with it
      for(int k = 0; k <= MAX_DIST; k++) {
        if(any_content_collides(masks[k].content, second_shape_content)) {
          output = k;
          break;
        }
      }
      this->intershape_distance[n * max_shapes + i] = output;
    }
  }
}

void Vignette::check_containing() {
  bool is_inside;
  bool is_outside;
  Vignette mask_grey_in;
  for(int n = 0; n < nb_shapes; n++) {
    mask_grey_in.clear();
    this->extract_part(n, mask_grey_in.content);
    Vignette mask_grey_out = mask_grey_in;
    // If we fill from the centre of the shape, we'll fill its inside
    // (more if the filling tool escapes the shape boundaries)
    mask_grey_in.fill(shapes_xs[n], shapes_ys[n], 128);
    // If we fill from the corners, we'll definitely fill all the vignette
    // outside the shape, except in a very extreme edge case with a shape
    // that is the size of the entire vignette.
    mask_grey_out.fill(0, 0, 128);
    mask_grey_out.fill(Vignette::width - 1, 0, 128);
    mask_grey_out.fill(0, Vignette::height - 1, 128);
    mask_grey_out.fill(Vignette::width - 1, Vignette::height - 1, 128);
    for(int i = 0; i < nb_shapes; i++) {
      int second_shape_content[width * height];
      this->extract_part(i, second_shape_content);
      is_inside = !any_content_collides(mask_grey_out.content,
                                        second_shape_content);
      is_outside = !any_content_collides(mask_grey_in.content,
                                         second_shape_content);
      float output = -1.0;
      if(is_inside && is_outside) {
        // This means the second shape is nowhere
        output = -0.5;
      } else if(is_inside && !is_outside) {
        // This means the shape is genuinely inside
        output = 1.0;
      } else if(!is_inside && is_outside) {
        // This means the shape is genuinely outside
        output = 0.0;
      } else if(!is_inside && !is_outside) {
        // This means the shapes intersect, or shape n does not form a
        // closed loop
        output = 0.5;
      }
      this->shape_is_containing[n * max_shapes + i] = output;
    }
  }
}
