// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifdef HAVE_GUI

#include <vector>

#include "gles2.h"

#include "display.h"

#include "image_raw.h"
#include "image_rgb.h"
#include "histogram.h"

#define HAVE_VAO

struct display_gles : public display
{
  std::vector<unsigned char> pixels;
  histogram3 hist;
  bool have_all_data;
  bool have_some_data;
  GLuint texture;
#ifdef HAVE_VAO
  GLuint vao;
#endif
  GLuint vbo;
  int destination;
  GLuint background[2];
  GLuint fbo;
  GLuint p_display_background;
  GLint u_display_background_transform;
  GLint u_display_background_rgb;
  GLint u_display_background_srgb;
  GLuint p_display;
  GLint u_display_transform;
  GLint u_display_rgb;
  GLint u_display_subframes;
  GLint u_display_srgb;
  GLuint p_display_rectangle;
  GLint u_display_rect;
  GLuint p_display_circles;
  GLint u_display_circles;
  GLint u_display_ncircles;
  GLenum format;

  display_gles();
  virtual ~display_gles();
  virtual void resize(coord_t width, coord_t height, coord_t subsampling);
  virtual void plot(const image_rgb &img, const ppostprocessing &post);
  virtual void plot(const image_raw &img, const ppostprocessing &post);
  virtual void update_histogram();
  virtual void draw(coord_t win_width, coord_t win_height, const mat3 &T, const int srgb_conversion, bool capture);
  virtual void draw_rectangle(coord_t win_width, coord_t win_height, float x0, float y0, float x1, float y1, const int srgb_conversion = 0);
  virtual void draw_circles(coord_t win_width, coord_t win_height, const std::vector<glm::vec4> &circles, const int srgb_conversion = 0);
};

bool is_webgl_1(const char *version);

#endif
