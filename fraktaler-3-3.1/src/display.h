// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "types.h"

struct image_raw;
struct image_rgb;
struct ppostprocessing;

struct display
{
  coord_t width;
  coord_t height;
  coord_t subsampling;
  display()
  : width(0)
  , height(0)
  , subsampling(0)
  {
  }
  virtual ~display()
  {
  }
  virtual void resize(coord_t widthx, coord_t heightx, coord_t subsamplingx)
  {
    width = widthx;
    height = heightx;
    subsampling = subsamplingx;
  }
  virtual void plot(const image_rgb &img, const ppostprocessing &post) = 0;
  virtual void plot(const image_raw &img, const ppostprocessing &post) = 0;
  virtual void draw(coord_t win_width, coord_t win_height, const mat3 &T, const int srgb_conversion = 0, bool capture = false) = 0;
  virtual void draw_rectangle(coord_t win_width, coord_t win_height, float x0, float y0, float x1, float y1, const int srgb_conversion = 0) = 0;
  virtual void draw_circles(coord_t win_width, coord_t win_height, const std::vector<glm::vec4> &circles, const int srgb_conversion = 0) = 0;
  virtual void image_coord(coord_t win_width, coord_t win_height, double *x, double *y)
  {
    *x -= win_width / 2.0;
    *y -= win_height / 2.0;
    if (width * win_height > height * win_width)
    {
      // image aspect is wider than window
      if (subsampling * width > win_width)
      {
        *x /= win_width;
        *y /= win_width;
        *x *= width;
        *y *= width;
      }
      else
      {
        *x /= subsampling;
        *y /= subsampling;
      }
    }
    else
    {
      // image aspect is narrower than window
      if (subsampling * height > win_height)
      {
        *x /= win_height;
        *y /= win_height;
        *x *= height;
        *y *= height;
      }
      else
      {
        *x /= subsampling;
        *y /= subsampling;
      }
    }
    *x += width / 2.0;
    *y += height / 2.0;
  }
};
