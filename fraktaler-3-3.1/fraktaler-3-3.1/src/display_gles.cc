// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#ifdef HAVE_GUI

#include "f3imconfig.h"

#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/gtx/matrix_transform_2d.hpp>

#include "display_gles.h"
#include "glutil.h"
#include "parallel.h"
#include "param.h"
#include "types.h"

ppostprocessing cook(const ppostprocessing &p) noexcept
{
  ppostprocessing q;
  q.brightness = p.brightness;
  q.contrast = std::exp2(p.contrast);
  q.gamma = 1.0 / p.gamma;
  q.exposure = std::exp2(p.exposure);
  return q;
}

float postprocess(const ppostprocessing &q, float x) noexcept
{
  x += q.brightness;
  x -= 0.5f;
  x *= q.contrast;
  x += 0.5f;
  x = std::pow(std::fmax(0.0f, x), q.gamma);
  x *= q.exposure;
  return x;
}

inline float linear_to_srgb(float c) noexcept
{
  c = glm::clamp(c, 0.0f, 1.0f);
  if (c <= 0.0031308f)
  {
    return 12.92f * c;
  }
  else
  {
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
  }
}

inline float srgb_to_linear(float c) noexcept
{
  c = glm::clamp(c, 0.0f, 1.0f);
  if (c <= 0.04045f)
  {
    return c / 12.92f;
  }
  else
  {
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
  }
}

static const char *version = OPENGL_GLSL_VERSION "\n";

static const char *vert =
  "in vec2 v_position;\n"
  "in vec2 v_texcoord;\n"
  "out vec2 Internal_texcoord;\n"
  "uniform mat3 Internal_transform;\n"
  "void main(void)\n"
  "{\n"
  "  vec3 p = Internal_transform * vec3(v_position, 1.0);\n"
  "  gl_Position = vec4(p.xy / p.z, 0.0, 1.0);\n"
  "  Internal_texcoord = v_texcoord;\n"
  "}\n"
  ;

static const char *vert_simple =
  "in vec2 v_position;\n"
  "in vec2 v_texcoord;\n"
  "out vec2 Internal_texcoord;\n"
  "void main(void)\n"
  "{\n"
  "  vec3 p = vec3(v_position, 1.0);\n"
  "  gl_Position = vec4(p.xy / p.z, 0.0, 1.0);\n"
  "  Internal_texcoord = v_texcoord;\n"
  "}\n"
  ;

static const char *frag_display_background =
  "precision highp float;\n"
  "uniform sampler2D Internal_RGB;\n"
  "uniform int Internal_srgb;\n"
  "in vec2 Internal_texcoord;\n"
  "out vec4 Internal_fragcolor;\n"
  "float srgb_to_linear(float c)\n"
  "{\n"
  "  c = clamp(c, 0.0, 1.0);\n"
  "  if (c <= 0.04045)\n"
  "  {\n"
  "    return c / 12.92;\n"
  "  }\n"
  "  else\n"
  "  {\n"
  "    return pow((c + 0.055) / 1.055, 2.4);\n"
  "  }\n"
  "}\n"
  "float linear_to_srgb(float c)\n"
  "{\n"
  "  c = clamp(c, 0.0, 1.0);\n"
  "  if (c <= 0.0031308)\n"
  "  {\n"
  "    return 12.92 * c;\n"
  "  }\n"
  "  else\n"
  "  {\n"
  "    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;\n"
  "  }\n"
  "}\n"
  "vec3 srgb_to_linear(vec3 c)\n"
  "{\n"
  "  return vec3(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));\n"
  "}\n"
  "vec3 linear_to_srgb(vec3 c)\n"
  "{\n"
  "  return vec3(linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b));\n"
  "}\n"
  "void main(void)\n"
  "{\n"
  "  vec2 t = Internal_texcoord;\n"
  "  vec4 c = texture(Internal_RGB, vec2(t.x, t.y));\n"
  "  if (Internal_srgb > 0)\n"
  "  {\n"
  "    c.rgb = linear_to_srgb(c.rgb);\n"
  "  }\n"
  "  if (Internal_srgb < 0)\n"
  "  {\n"
  "    c.rgb = srgb_to_linear(c.rgb);\n"
  "  }\n"
  "  Internal_fragcolor = c;\n"
  "}\n"
  ;

static const char *frag_display =
  "precision highp float;\n"
  "uniform sampler2D Internal_RGB;\n"
  "uniform int Internal_subframes;\n"
  "uniform int Internal_srgb;\n"
  "in vec2 Internal_texcoord;\n"
  "out vec4 Internal_fragcolor;\n"
  "float srgb_to_linear(float c)\n"
  "{\n"
  "  c = clamp(c, 0.0, 1.0);\n"
  "  if (c <= 0.04045)\n"
  "  {\n"
  "    return c / 12.92;\n"
  "  }\n"
  "  else\n"
  "  {\n"
  "    return pow((c + 0.055) / 1.055, 2.4);\n"
  "  }\n"
  "}\n"
  "float linear_to_srgb(float c)\n"
  "{\n"
  "  c = clamp(c, 0.0, 1.0);\n"
  "  if (c <= 0.0031308)\n"
  "  {\n"
  "    return 12.92 * c;\n"
  "  }\n"
  "  else\n"
  "  {\n"
  "    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;\n"
  "  }\n"
  "}\n"
  "vec3 srgb_to_linear(vec3 c)\n"
  "{\n"
  "  return vec3(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b));\n"
  "}\n"
  "vec3 linear_to_srgb(vec3 c)\n"
  "{\n"
  "  return vec3(linear_to_srgb(c.r), linear_to_srgb(c.g), linear_to_srgb(c.b));\n"
  "}\n"
  "void main(void)\n"
  "{\n"
  "  vec2 t = Internal_texcoord;\n"
  "  vec4 c = texture(Internal_RGB, vec2(t.x, 1.0 - t.y));\n"
  "  if (Internal_srgb > 0)\n"
  "  {\n"
  "    c.rgb = linear_to_srgb(c.rgb);\n"
  "  }\n"
  "  if (Internal_srgb < 0)\n"
  "  {\n"
  "    c.rgb = srgb_to_linear(c.rgb);\n"
  "  }\n"
  "  Internal_fragcolor = c;\n"
  "}\n"
  ;

static const char *frag_display_rectangle =
  "precision highp float;\n"
  "uniform vec4 Internal_rectangle;\n"
  "in vec2 Internal_texcoord;\n"
  "out vec4 Internal_fragcolor;\n"
  "bool in_rectangle(vec2 p, vec4 r)\n"
  "{\n"
  "  return r.x < p.x && p.x < r.z && r.y < p.y && p.y < r.w;\n"
  "}\n"
  "void main(void)\n"
  "{\n"
  "  vec2 t = Internal_texcoord;\n"
  "  vec4 d = vec4(-dFdx(t.x), -dFdy(t.y), dFdx(t.x), dFdy(t.y));\n"
  "  vec4 c = vec4(1.0, 0.8, 0.5, 0.0);\n"
  "  if (in_rectangle(t, Internal_rectangle + d))\n"
  "  {\n"
  "    if (in_rectangle(t, Internal_rectangle - d))\n"
  "    {\n"
  "      c.a = 0.25;\n"
  "    }\n"
  "    else\n"
  "    {\n"
  "      c.a = 0.75;\n"
  "    }\n"
  "  }\n"
  "  else\n"
  "  {\n"
  "    discard;\n"
  "  }\n"
  "  Internal_fragcolor = c;\n"
  "}\n"
  ;

static const char *frag_display_circles =
  "precision highp float;\n"
  "uniform vec4 Internal_circles[16];\n"
  "uniform int Internal_ncircles;\n"
  "in vec2 Internal_texcoord;\n"
  "out vec4 Internal_fragcolor;\n"
  "bool in_circle(vec2 p, vec4 c)\n"
  "{\n"
  "  float x = ((c.x + 1.0) / 2.0 - p.x) / c.z;\n"
  "  float y = ((c.y + 1.0) / 2.0 - p.y) / c.w;\n"
  "  return x * x + y * y < 1.0;\n"
  "}\n"
  "void main(void)\n"
  "{\n"
  "  vec2 t = Internal_texcoord;\n"
  "  vec4 c = vec4(1.0, 0.8, 0.5, 0.0);\n"
  "  for (int circle = 0; circle < 16; ++circle)\n"
  "  {\n"
  "    if (circle < Internal_ncircles)\n"
  "    {\n"
  "      if (in_circle(t, Internal_circles[circle]))\n"
  "      {\n"
  "        c.a = 0.5;\n"
  "      }\n"
  "    }\n"
  "  }\n"
  "  if (c.a == 0.0)\n"
  "  {\n"
  "    discard;\n"
  "  }\n"
  "  Internal_fragcolor = c;\n"
  "}\n"
  ;

display_gles::display_gles()
: display()
, pixels(0)
, hist{{ { 0, 0, false, 0, { }, false }, { 0, 0, false, 0, { }, false }, { 0, 0, false, 0, { }, false } }}
, have_all_data(false)
, have_some_data(false)
, texture(0)
#ifdef HAVE_VAO
, vao(0)
#endif
, vbo(0)
, destination(0)
, background{0, 0}
, fbo(0)
, p_display_background(0)
, u_display_background_rgb(0)
, p_display(0)
, u_display_rgb(0)
, u_display_rect(0)
#ifdef __EMSCRIPTEN__
, format(GL_SRGB8_ALPHA8)
#else
, format(GL_RGBA)
#endif
{
  while (glGetError())
  {
  }
  glGenTextures(2, &background[0]);
  glActiveTexture(GL_TEXTURE1);
  for (int t = 0; t < 2; ++t)
  {
    glBindTexture(GL_TEXTURE_2D, background[t]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
  glGenFramebuffers(1, &fbo);
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  const GLfloat data[] = { -1, -1, 0, 0,  -1, 1, 0, 1,  1, -1, 1, 0,  1, 1, 1, 1 };
  glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
#ifdef HAVE_VAO
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 0 * sizeof(GLfloat)); // vertex
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 2 * sizeof(GLfloat)); // texcoord
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);
#endif
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  p_display_background = vertex_fragment_shader(version, vert, frag_display_background);
  p_display = vertex_fragment_shader(version, vert, frag_display);
  p_display_rectangle = vertex_fragment_shader(version, vert_simple, frag_display_rectangle);
  p_display_circles = vertex_fragment_shader(version, vert_simple, frag_display_circles);
  glUseProgram(p_display_background);
  u_display_background_transform = glGetUniformLocation(p_display_background, "Internal_transform");
  u_display_background_rgb = glGetUniformLocation(p_display_background, "Internal_RGB");
  u_display_background_srgb = glGetUniformLocation(p_display_background, "Internal_srgb");
  glUniform1i(u_display_background_rgb, 1);
  glUseProgram(p_display);
  u_display_transform = glGetUniformLocation(p_display, "Internal_transform");
  u_display_rgb = glGetUniformLocation(p_display, "Internal_RGB");
  u_display_subframes = glGetUniformLocation(p_display, "Internal_subframes");
  u_display_srgb = glGetUniformLocation(p_display, "Internal_srgb");
  glUniform1i(u_display_rgb, 0);
  glUseProgram(p_display_rectangle);
  u_display_rect = glGetUniformLocation(p_display_rectangle, "Internal_rectangle");
  glUseProgram(p_display_circles);
  u_display_circles = glGetUniformLocation(p_display_circles, "Internal_circles");
  u_display_ncircles = glGetUniformLocation(p_display_circles, "Internal_ncircles");
  glUseProgram(0);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d display_gles\n", e);
  }
}

display_gles::~display_gles()
{
  while (glGetError())
  {
  }
  glDeleteProgram(p_display);
#ifdef HAVE_VAO
  glDeleteVertexArrays(1, &vao);
#endif
  glDeleteBuffers(1, &vbo);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glDeleteTextures(1, &texture);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(2, &background[0]);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d ~display_gles\n", e);
  }
}

void display_gles::resize(coord_t width, coord_t height, coord_t subsampling)
{
  while (glGetError())
  {
  }
  display::resize(width, height, subsampling);
  pixels.resize(4 * width * height);
  have_all_data = false;
  have_some_data = false;
  glActiveTexture(GL_TEXTURE1);
  for (int t = 0; t < 2; ++t)
  {
    glBindTexture(GL_TEXTURE_2D, background[t]);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  }
  glActiveTexture(GL_TEXTURE0);
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d resize\n", e);
  }
}

void display_gles::update_histogram()
{
  hist = histogram3
    {{{ 0, 1.0, false, 0, { }, false }
     ,{ 0, 1.0, false, 0, { }, false }
     ,{ 0, 1.0, false, 0, { }, false }}};
  for (coord_t c = 0; c < 3; ++c)
  {
    hist.h[c].data.resize(256);
    std::fill(hist.h[c].data.begin(), hist.h[c].data.end(), 0.0f);
 }
  for (coord_t j = 0; j < height; ++j)
  for (coord_t i = 0; i < width; ++i)
  {
    coord_t k = 4 * (j * width + i);
    float A = pixels[k + 3] / 255.0f;
    for (coord_t c = 0; c < 3; ++c)
    {
      hist.h[c].data[pixels[k + c]] += A;
      hist.h[c].total += A;
    }
  }
}

void display_gles::plot(const image_rgb &out, const ppostprocessing &post)
{
  while (glGetError())
  {
  }
  have_all_data = true;
  const ppostprocessing cooked = cook(post);
#ifndef __EMSCRIPTEN__
  volatile bool running = true;
  parallel2d(std::thread::hardware_concurrency(), 0, width, 32, 0, height, 32, &running, [&](coord_t i, coord_t j) -> void
#else
  for (coord_t j = 0; j < height; ++j)
  for (coord_t i = 0; i < width; ++i)
#endif
  {
    coord_t k = 4 * (j * width + i);
    float A = out.RGBA[k + 3];
    if (A == 0)
    {
      have_all_data = false;
      for (coord_t c = 0; c < 3; ++c)
      {
        pixels[4 * ((height - 1 - j) * width + i) + c] = glm::clamp(255.0f * linear_to_srgb(0.5f), 0.0f, 255.0f);
      }
      pixels[4 * ((height - 1 - j) * width + i) + 3] = 0;
    }
    else
    {
      have_some_data = true;
      for (coord_t c = 0; c < 3; ++c)
      {
        pixels[4 * ((height - 1 - j) * width + i) + c] = glm::clamp(255.0f * linear_to_srgb(postprocess(cooked, out.RGBA[k + c] / A)), 0.0f, 255.0f);
      }
      pixels[4 * ((height - 1 - j) * width + i) + 3] = 255;
    }
  }
#ifndef __EMSCRIPTEN__
  );
#endif
  update_histogram();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, &pixels[0]);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d plot rgb\n", e);
  }
}

void display_gles::plot(const image_raw &out, const ppostprocessing &post)
{
  while (glGetError())
  {
  }
  const ppostprocessing cooked = cook(post);
#ifndef __EMSCRIPTEN__
  volatile bool running = true;
  parallel2d(std::thread::hardware_concurrency(), 0, width, 32, 0, height, 32, &running, [&](coord_t i, coord_t j) -> void
#else
  for (coord_t j = 0; j < height; ++j)
  for (coord_t i = 0; i < width; ++i)
#endif
  {
    coord_t k = j * width + i;
    coord_t w = 4 * ((height - 1 - j) * width + i);
    pixels[w + 0] = glm::clamp(255.0f * linear_to_srgb(out.R ? postprocess(cooked, out.R[k]) : 0.5f), 0.0f, 255.0f);
    pixels[w + 1] = glm::clamp(255.0f * linear_to_srgb(out.G ? postprocess(cooked, out.G[k]) : 0.5f), 0.0f, 255.0f);
    pixels[w + 2] = glm::clamp(255.0f * linear_to_srgb(out.B ? postprocess(cooked, out.B[k]) : 0.5f), 0.0f, 255.0f);
    pixels[w + 3] = 255;
  }
#ifndef __EMSCRIPTEN__
  );
#endif
  update_histogram();
  glActiveTexture(GL_TEXTURE0);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, &pixels[0]);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d plot raw\n", e);
  }
}

void set_viewport(int win_width, int win_height, int width, int height, int subsampling)
{
  if (width * win_height > height * win_width)
  {
    // image is wider than window aspect
    if (subsampling * width > win_width)
    {
      int border = (win_height - win_width * height / width) / 2;
      glViewport(0, border, win_width, win_width * height / width);
    }
    else
    {
      glViewport((win_width - subsampling * width) / 2, (win_height - subsampling * height) / 2, subsampling * width, subsampling * height);
    }
  }
  else
  {
    // image is narrower than window aspect
    if (subsampling * height > win_height)
    {
      int border = (win_width - win_height * width / height) / 2;
      glViewport(border, 0, win_height * width / height, win_height);
    }
    else
    {
      glViewport((win_width - subsampling * width) / 2, (win_height - subsampling * height) / 2, subsampling * width, subsampling * height);
    }
  }
}

void display_gles::draw(coord_t win_width, coord_t win_height, const mat3 &T, const int srgb_conversion, bool capture)
{
  while (glGetError())
  {
  }
  if (capture)
  {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, background[destination], 0);
    glViewport(0, 0, width, height);
    glClearColor(0.5, 0.5, 0.5, 1);
    glClear(GL_COLOR_BUFFER_BIT);
  }
  else
  {
    glViewport(0, 0, win_width, win_height);
    glClear(GL_COLOR_BUFFER_BIT);
    set_viewport(win_width, win_height, width, height, subsampling);
  }
#ifdef HAVE_VAO
  glBindVertexArray(vao);
#else
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 0 * sizeof(GLfloat)); // vertex
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 2 * sizeof(GLfloat)); // texcoord
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
#endif
  mat3 S = mat3(1.0f);
  // [0..w] x [0..h]
  S = glm::scale(S, vec2(float(width), float(height)));
  S = glm::scale(S, vec2(0.5f, 0.5f));
  S = glm::translate(S, vec2(1.0f));
  // [-1..1] x [-1..1]
  S = glm::inverse(S) * T * S;
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, background[! destination]);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);
  glUseProgram(p_display_background);
  glUniformMatrix3fv(u_display_background_transform, 1, false, &S[0][0]);
  glUniform1i(u_display_background_srgb, srgb_conversion);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glUseProgram(p_display);
  glUniformMatrix3fv(u_display_transform, 1, false, &S[0][0]);
  glUniform1i(u_display_subframes, have_some_data);
  glUniform1i(u_display_srgb, srgb_conversion);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glUseProgram(0);
  glDisable(GL_BLEND);
#ifdef HAVE_VAO
  glBindVertexArray(0);
#else
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
  if (capture)
  {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    destination = ! destination;
  }
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d draw\n", e);
  }
}

void display_gles::draw_rectangle(coord_t win_width, coord_t win_height, float x0, float y0, float x1, float y1, const int srgb_conversion)
{
  (void) srgb_conversion;
  while (glGetError())
  {
  }
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  set_viewport(win_width, win_height, width, height, subsampling);
#ifdef HAVE_VAO
  glBindVertexArray(vao);
#else
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 0 * sizeof(GLfloat)); // vertex
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 2 * sizeof(GLfloat)); // texcoord
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
#endif
  glUseProgram(p_display_rectangle);
  glUniform4f(u_display_rect, (x0 + 1) / 2, 1 - (y1 + 1) / 2, (x1 + 1) / 2, 1 - (y0 + 1) / 2);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glUseProgram(0);
#ifdef HAVE_VAO
  glBindVertexArray(0);
#else
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
  glDisable(GL_BLEND);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d draw_rectangle\n", e);
  }
}

void display_gles::draw_circles(coord_t win_width, coord_t win_height, const std::vector<glm::vec4> &circles, const int srgb_conversion)
{
  (void) srgb_conversion;
  while (glGetError())
  {
  }
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glViewport(0, 0, win_width, win_height);
#ifdef HAVE_VAO
  glBindVertexArray(vao);
#else
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 0 * sizeof(GLfloat)); // vertex
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ((char*)0) + 2 * sizeof(GLfloat)); // texcoord
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
#endif
  glUseProgram(p_display_circles);
  glUniform4fv(u_display_circles, circles.size(), &circles[0][0]);
  glUniform1i(u_display_ncircles, circles.size());
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glUseProgram(0);
#ifdef HAVE_VAO
  glBindVertexArray(0);
#else
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
  glDisable(GL_BLEND);
  int e;
  while ((e = glGetError()))
  {
    std::fprintf(stderr, "GL ERROR %d draw_circles\n", e);
  }
}

bool is_webgl_1(const char *version)
{
  const char *webgl1 = "OpenGL ES 2.0 (WebGL 1.0";
  return 0 == std::strncmp(version, webgl1, std::strlen(webgl1));
}

#endif
