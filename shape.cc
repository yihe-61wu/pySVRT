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

#include "shape.h"

int Shape::generate_part_part(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                              scalar_t radius, scalar_t hole_radius,
                              scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2) {
  if(abs(x1 - x2) > gap_max || abs(y1 - y2) > gap_max) {

    scalar_t d = sqrt(scalar_t(sq(x1 - x2) + sq(y1 - y2)))/5;
    scalar_t x3, y3, dx, dy;

    do {
      // Isotropic jump
      do {
        dx = (2 * random_uniform_0_1() - 1) * d;
        dy = (2 * random_uniform_0_1() - 1) * d;
      } while(sq(dx) + sq(dy) > sq(d));
      x3 = (x1 + x2) / 2 + dx;
      y3 = (y1 + y2) / 2 + dy;
    } while(sq(x3) + sq(y3) > sq(radius));

    if(generate_part_part(xp, yp, nb_pixels,
                          radius, hole_radius, x1, y1, x3, y3)) {
      return 1;
    }

    if(generate_part_part(xp, yp, nb_pixels,
                          radius, hole_radius, x3, y3, x2, y2)) {
      return 1;
    }

  } else {

    if(sq(x1) + sq(y1) >= sq(radius) || sq(x1) + sq(y1) < sq(hole_radius)) {
      return 1;
    }

    xp[*nb_pixels] = x1;
    yp[*nb_pixels] = y1;
    (*nb_pixels)++;

  }

  return 0;
}

int Shape::generate_part_part2(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                               scalar_t radius, scalar_t hole_radius,
                               scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2) {
  if(abs(x1 - x2) > gap_max || abs(y1 - y2) > gap_max) {

    scalar_t d = sqrt(scalar_t(sq(x1 - x2) + sq(y1 - y2)))/5;
    scalar_t x3, y3, dx, dy;

    do {
      // Isotropic jump
      do {
        dx = (2 * random_uniform_0_1() - 1) * d;
        dy = (2 * random_uniform_0_1() - 1) * d;
      } while(sq(dx) + sq(dy) > sq(d));
      x3 = (x1 + x2) / 2 + dx;
      y3 = (y1 + y2) / 2 + dy;
    } while(abs(x3) > radius || abs(y3) > radius);

    if(generate_part_part2(xp, yp, nb_pixels,
                          radius, hole_radius, x1, y1, x3, y3)) {
      return 1;
    }

    if(generate_part_part2(xp, yp, nb_pixels,
                          radius, hole_radius, x3, y3, x2, y2)) {
      return 1;
    }

  } else {

    if(abs(x1) > radius) {
      cerr << "Fail on x1 radius " << x1 << " outside " << radius << endl;
      return 1;
    }
    if(abs(y1) > radius) {
      cerr << "Fail on y1 radius " << y1 << " outside " << radius << endl;
      return 1;
    }
    if(sq(x1) + sq(y1) < sq(hole_radius)) {
      cerr << "Fail on point " << sq(x1) + sq(y1) << " inside hold radius " << hole_radius << endl;
      return 1;
    }

    xp[*nb_pixels] = x1;
    yp[*nb_pixels] = y1;
    (*nb_pixels)++;

  }

  return 0;
}

void Shape::generate_part(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                          scalar_t radius, scalar_t hole_radius) {
  scalar_t x1, y1, x2, y2, x3, y3, x4, y4;
  int err1, err2, err3, err4;

  do {
    *nb_pixels = 0;

    do {
      x1 = random_uniform_0_1() * radius;
      y1 = random_uniform_0_1() * radius;
    } while(sq(x1) + sq(y1) > sq(radius) || sq(x1) + sq(y1) < sq(hole_radius));

    do {
      x2 = -random_uniform_0_1() * radius;
      y2 = random_uniform_0_1() * radius;
    } while(sq(x2) + sq(y2) > sq(radius) || sq(x2) + sq(y2) < sq(hole_radius));

    do {
      x3 = -random_uniform_0_1() * radius;
      y3 = -random_uniform_0_1() * radius;
    } while(sq(x3) + sq(y3) > sq(radius) || sq(x3) + sq(y3) < sq(hole_radius));

    do {
      x4 = random_uniform_0_1() * radius;
      y4 = -random_uniform_0_1() * radius;
    } while(sq(x4) + sq(y4) > sq(radius) || sq(x4) + sq(y4) < sq(hole_radius));

    n_pixels1 = *nb_pixels;
    err1 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x1, y1, x2, y2);
    n_pixels2 = *nb_pixels;
    err2 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x2, y2, x3, y3);
    n_pixels3 = *nb_pixels;
    err3 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x3, y3, x4, y4);
    n_pixels4 = *nb_pixels;
    err4 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x4, y4, x1, y1);

  } while(err1 || err2 || err3 || err4);
}

void Shape::generate_open_part(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                               scalar_t radius, scalar_t hole_radius) {
  scalar_t x1, y1, x2, y2, x3, y3, x4, y4;
  int err1, err2, err3;

  do {
    *nb_pixels = 0;

    do {
      x1 = random_uniform_0_1() * radius;
      y1 = random_uniform_0_1() * radius;
    } while(sq(x1) + sq(y1) > sq(radius) || sq(x1) + sq(y1) < sq(hole_radius));

    do {
      x2 = -random_uniform_0_1() * radius;
      y2 = random_uniform_0_1() * radius;
    } while(sq(x2) + sq(y2) > sq(radius) || sq(x2) + sq(y2) < sq(hole_radius));

    do {
      x3 = -random_uniform_0_1() * radius;
      y3 = -random_uniform_0_1() * radius;
    } while(sq(x3) + sq(y3) > sq(radius) || sq(x3) + sq(y3) < sq(hole_radius));

    do {
      x4 = random_uniform_0_1() * radius;
      y4 = -random_uniform_0_1() * radius;
    } while(sq(x4) + sq(y4) > sq(radius) || sq(x4) + sq(y4) < sq(hole_radius));

    n_pixels1 = *nb_pixels;
    err1 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x1, y1, x2, y2);
    n_pixels2 = *nb_pixels;
    err2 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x2, y2, x3, y3);
    n_pixels3 = *nb_pixels;
    err3 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x3, y3, x4, y4);
    n_pixels4 = *nb_pixels;

  } while(err1 || err2 || err3);
}

void Shape::generate_tri_part(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                             scalar_t radius, scalar_t hole_radius) {
  scalar_t r1, t1, x1, y1, r2, t2, x2, y2, r3, t3, x3, y3;
  int err1, err2, err3;

  do {
    *nb_pixels = 0;

    r1 = random_uniform_0_1() * (radius - hole_radius) + hole_radius;
    t1 = random_uniform_0_1() * M_PI * 2 / 6;
    x1 = r1 * cos(t1);
    y1 = r1 * sin(t1);

    r2 = random_uniform_0_1() * (radius - hole_radius) + hole_radius;
    t2 = (2 + random_uniform_0_1()) * M_PI * 2 / 6;
    x2 = r2 * cos(t2);
    y2 = r2 * sin(t2);

    r3 = random_uniform_0_1() * (radius - hole_radius) + hole_radius;
    t3 = (4 + random_uniform_0_1()) * M_PI * 2 / 6;
    x3 = r3 * cos(t3);
    y3 = r3 * sin(t3);

    n_pixels1 = *nb_pixels;
    err1 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x1, y1, x2, y2);
    n_pixels2 = *nb_pixels;
    err2 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x2, y2, x3, y3);
    n_pixels3 = *nb_pixels;
    err3 = generate_part_part(xp, yp, nb_pixels, radius, hole_radius, x3, y3, x1, y1);
    n_pixels4 = *nb_pixels;

  } while(err1 || err2 || err3);
}

void Shape::generate_circle_part(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                                 scalar_t radius, scalar_t hole_radius) {
  int num_segments = 8;
  scalar_t r, r_seg, t_seg, xx[num_segments + 1], yy[num_segments + 1];
  int err;

  do {
    *nb_pixels = 0;

    r = hole_radius + random_uniform_0_1() * (radius - hole_radius);

    for(int ss = 0; ss < num_segments + 1; ss++) {
      r_seg = r;
      t_seg = M_PI * 2 * ss / (double)num_segments;
      xx[ss] = r_seg * cos(t_seg);
      yy[ss] = r_seg * sin(t_seg);
    }

    n_pixels1 = *nb_pixels;
    err = 0;
    for(int s = 0; s < num_segments; s++) {
      err |= generate_part_part(xp, yp, nb_pixels, radius, hole_radius,
                                xx[s], yy[s], xx[s+1], yy[s+1]);
      if(s == 0) n_pixels2 = *nb_pixels;
      if(s == 1) n_pixels3 = *nb_pixels;
    }
    n_pixels4 = *nb_pixels;

  } while(err);
}

void Shape::generate_cross(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                           scalar_t radius, scalar_t hole_radius) {
  scalar_t x1, y1, x2, y2, x3, y3, x4, y4;
  int err1, err2;

  //cout << "Trying to make a cross." << endl;

  do {
    *nb_pixels = 0;

    x1 = (1 + random_uniform_0_1()) * radius / 2;
    y1 = (2 * random_uniform_0_1() - 1) * radius / 16;

    x2 = -(1 + random_uniform_0_1()) * radius / 2;
    y2 = (2 * random_uniform_0_1() - 1) * radius / 16;

    x3 = x2 + radius / 8 + random_uniform_0_1() * (x1 - x2 - radius / 4);
    x4 = x3 + (2 * random_uniform_0_1() - 1) * radius / 16;
    x3 = x3 + (2 * random_uniform_0_1() - 1) * radius / 16;
    y3 = (1 + random_uniform_0_1()) * radius / 2;
    y4 = -(1 + random_uniform_0_1()) * radius / 2;

    n_pixels1 = *nb_pixels;
    err1 = generate_part_part2(xp, yp, nb_pixels, radius, 0, x1, y1, x2, y2);
    n_pixels2 = *nb_pixels;
    err2 = generate_part_part2(xp, yp, nb_pixels, radius, 0, x3, y3, x4, y4);
    n_pixels3 = *nb_pixels;
    n_pixels4 = *nb_pixels;

  } while(err1 || err2);
}

void Shape::generate_spiral(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                            scalar_t radius, scalar_t hole_radius) {
  scalar_t r, num_loops, r_seg, t_seg, dir;
  int num_segments = 12;
  scalar_t xx[num_segments + 1], yy[num_segments + 1];
  int err;

  do {
    *nb_pixels = 0;

    r = (1 + random_uniform_0_1()) * radius / 2;
    num_loops = 0.75 + random_uniform_0_1() * 1.75;

    if(random_uniform_0_1() > 0.5) {
      dir = -1;
    } else {
      dir = 1;
    }

    for(int s = 0; s < num_segments + 1; s++) {
      r_seg = r * s / (double)num_segments;
      t_seg = num_loops * M_PI * 2 * s / (double)num_segments;
      xx[s] = r_seg * cos(t_seg);
      yy[s] = r_seg * sin(t_seg) * dir;
    }

    n_pixels1 = *nb_pixels;
    err = 0;
    for(int s = 0; s < num_segments; s++) {
      err |= generate_part_part2(xp, yp, nb_pixels, radius, 0,
                                 xx[s], yy[s], xx[s+1], yy[s+1]);
      if(s == num_segments / 4) n_pixels2 = *nb_pixels;
      if(s == 2 * num_segments / 4) n_pixels3 = *nb_pixels;
      if(s == 3 * num_segments / 4) n_pixels3 = *nb_pixels;
    }

  } while(err);
}

void Shape::generate_zigzag(scalar_t *xp, scalar_t *yp, int *nb_pixels,
                            scalar_t radius, scalar_t hole_radius) {
  scalar_t r, x1, y1, x2, y2, x3, y3, x4, y4;
  int err1, err2, err3;

  do {
    *nb_pixels = 0;

    r = (random_uniform_0_1() + 1) * radius / 2;

    x1 = (random_uniform_0_1() * 2 - 1) * r;
    y1 = random_uniform_0_1() * r / 4;

    x2 = (random_uniform_0_1() * 2 - 1) * r;
    y2 = (random_uniform_0_1() + 1) * r / 4;

    x3 = (random_uniform_0_1() * 2 - 1) * r;
    y3 = (random_uniform_0_1() + 2) * r / 4;

    x4 = (random_uniform_0_1() * 2 - 1) * r;
    y4 = (random_uniform_0_1() + 3) * r / 4;

    n_pixels1 = *nb_pixels;
    err1 = generate_part_part2(xp, yp, nb_pixels, radius, 0, x1, y1, x2, y2);
    n_pixels2 = *nb_pixels;
    err2 = generate_part_part2(xp, yp, nb_pixels, radius, 0, x2, y2, x3, y3);
    n_pixels3 = *nb_pixels;
    err3 = generate_part_part2(xp, yp, nb_pixels, radius, 0, x3, y3, x4, y4);
    n_pixels4 = *nb_pixels;

  } while(err1 || err2 || err3);
}

Shape::Shape() {
  nb_pixels = 0;
  x_pixels = 0;
  y_pixels = 0;
}

Shape::~Shape() {
  delete[] x_pixels;
  delete[] y_pixels;
}

void Shape::randomize(scalar_t radius, scalar_t hole_radius) {
  delete[] x_pixels;
  delete[] y_pixels;
  nb_pixels = 0;
  scalar_t tmp_x_pixels[nb_max_pixels], tmp_y_pixels[nb_max_pixels];
  generate_part(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
  x_pixels = new scalar_t[nb_pixels];
  y_pixels = new scalar_t[nb_pixels];
  for(int p = 0; p < nb_pixels; p++) {
    x_pixels[p] = tmp_x_pixels[p];
    y_pixels[p] = tmp_y_pixels[p];
  }

  rotate(random_uniform_0_1() * M_PI * 2);

  // { // ******************************* START ***************************
// #warning Test code added on 2009 Sep 09 18:15:25
    // for(int p = 0; p < nb_pixels; p++) {
      // cout << x_pixels[p] << " " << y_pixels[p] << endl;
    // }
  // } // ******************************** END ****************************

}

void Shape::randomize_by_type(scalar_t radius, scalar_t hole_radius, int type) {
  delete[] x_pixels;
  delete[] y_pixels;
  nb_pixels = 0;
  scalar_t tmp_x_pixels[nb_max_pixels], tmp_y_pixels[nb_max_pixels];

  switch(type) {
  case 0:
    generate_part(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  case 1:
    generate_open_part(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  case 2:
    generate_tri_part(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  case 3:
    generate_cross(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  case 4:
    generate_spiral(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  case 5:
    generate_circle_part(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  case 6:
    generate_zigzag(tmp_x_pixels, tmp_y_pixels, &nb_pixels, radius, hole_radius);
    break;
  default:
    cerr << "Can not find shape type "
         << type
         << endl;
    abort();
  }

  x_pixels = new scalar_t[nb_pixels];
  y_pixels = new scalar_t[nb_pixels];
  for(int p = 0; p < nb_pixels; p++) {
    x_pixels[p] = tmp_x_pixels[p];
    y_pixels[p] = tmp_y_pixels[p];
  }

  rotate(random_uniform_0_1() * M_PI * 2);
}

void Shape::randomize_random_type(scalar_t radius, scalar_t hole_radius) {
  int NUM_TYPES = 7;
  int type;
  type = int(random_uniform_0_1() * (NUM_TYPES - 1)) + 1;
  randomize_by_type(radius, hole_radius, type);
}

void Shape::copy(Shape *shape) {
  delete[] x_pixels;
  delete[] y_pixels;
  nb_pixels = shape->nb_pixels;
  n_pixels1 = shape->n_pixels1;
  n_pixels2 = shape->n_pixels2;
  n_pixels3 = shape->n_pixels3;
  n_pixels4 = shape->n_pixels4;
  x_pixels = new scalar_t[nb_pixels];
  y_pixels = new scalar_t[nb_pixels];
  for(int p = 0; p < nb_pixels; p++) {
    x_pixels[p] = shape->x_pixels[p];
    y_pixels[p] = shape->y_pixels[p];
  }
}

void Shape::scale(scalar_t s) {
  for(int p = 0; p < nb_pixels; p++) {
    x_pixels[p] *= s;
    y_pixels[p] *= s;
  }
}

void Shape::rotate(scalar_t alpha) {
  scalar_t ux = cos(alpha), uy = -sin(alpha);
  scalar_t vx = sin(alpha), vy = cos(alpha);
  scalar_t x, y;
  for(int p = 0; p < nb_pixels; p++) {
    x = x_pixels[p] * ux + y_pixels[p] * uy;
    y = x_pixels[p] * vx + y_pixels[p] * vy;
    x_pixels[p] = x;
    y_pixels[p] = y;
  }
}

void Shape::symmetrize(scalar_t axis_x, scalar_t axis_y) {
  scalar_t sql = sq(axis_x) + sq(axis_y);
  scalar_t u, v;
  for(int p = 0; p < nb_pixels; p++) {
    u =   x_pixels[p] * axis_y - y_pixels[p] * axis_x;
    v =   x_pixels[p] * axis_x + y_pixels[p] * axis_y;
    u = - u;
    x_pixels[p] = (  u * axis_y + v * axis_x) / sql;
    y_pixels[p] = (- u * axis_x + v * axis_y) / sql;
  }
}


int Shape::overwrites(Vignette *vignette, scalar_t xc, scalar_t yc, int n1, int n2) {
  int x1 = int(x_pixels[n1 % nb_pixels] + xc);
  int y1 = int(y_pixels[n1 % nb_pixels] + yc);
  int x2 = int(x_pixels[n2 % nb_pixels] + xc);
  int y2 = int(y_pixels[n2 % nb_pixels] + yc);
  int n3 = (n1 + n2) / 2;

  if(n1 + 1 < n2 && (abs(x1 - x2) > 1 || abs(y1 - y2) > 1)) {
    return
      overwrites(vignette, xc, yc, n1, n3) ||
      overwrites(vignette, xc, yc, n3, n2);
  } else {

    if(x1 >= margin && x1 < Vignette::width - margin &&
       y1 >= margin && y1 < Vignette::height - margin) {

      if(margin > 0) {
        for(int xx = x1 - margin; xx <= x1 + margin; xx++) {
          for(int yy = y1 - margin; yy <= y1 + margin; yy++) {
            if(vignette->content[xx + Vignette::width * yy] != 255) {
              return 1;
            }
          }
        }
      }

      return 0;
    } else {
      return 1;
    }
  }
}

int Shape::overwrites(Vignette *vignette, scalar_t xc, scalar_t yc) {
  return
    overwrites(vignette, xc, yc, n_pixels1, n_pixels2) ||
    overwrites(vignette, xc, yc, n_pixels2, n_pixels3) ||
    overwrites(vignette, xc, yc, n_pixels3, n_pixels4) ||
    overwrites(vignette, xc, yc, n_pixels4, nb_pixels);
}

void Shape::draw(int part_number, Vignette *vignette, scalar_t xc, scalar_t yc, int n1, int n2) {
  int x1 = int(x_pixels[n1 % nb_pixels] + xc);
  int y1 = int(y_pixels[n1 % nb_pixels] + yc);
  int x2 = int(x_pixels[n2 % nb_pixels] + xc);
  int y2 = int(y_pixels[n2 % nb_pixels] + yc);
  int n3 = (n1 + n2) / 2;

  if(n1 + 1 < n2 && (abs(x1 - x2) > 1 || abs(y1 - y2) > 1)) {
    draw(part_number, vignette, xc, yc, n1, n3);
    draw(part_number, vignette, xc, yc, n3, n2);
  } else {
    if(x1 >= margin && x1 < Vignette::width-margin &&
       y1 >= margin && y1 < Vignette::height-margin) {
      vignette->content[x1 + Vignette::width * y1] = 0;
#ifdef KEEP_PART_PRESENCE
      vignette->part_presence[x1 + Vignette::width * y1] |= (1 << part_number);
#endif
    } else {
      abort();
    }
  }
}

void Shape::draw(int part_number, Vignette *vignette, scalar_t xc, scalar_t yc) {
  draw(part_number, vignette, xc, yc, n_pixels1, n_pixels2);
  draw(part_number, vignette, xc, yc, n_pixels2, n_pixels3);
  draw(part_number, vignette, xc, yc, n_pixels3, n_pixels4);
  draw(part_number, vignette, xc, yc, n_pixels4, nb_pixels);
}
