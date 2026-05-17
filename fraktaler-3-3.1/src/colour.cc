// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <fstream>
#include <map>
#include <set>
#include <vector>

#include <imgui.h>
#include <imgui_stdlib.h>

#include <SDL.h>

#include "colour.h"

const char *version = OPENGL_GLSL_VERSION "\n";
#include "colour.vert.h"
#include "colour.frag.h"

#include "glutil.h"
#include "param.h"
#include "render.h"
#include "histogram.h"

#include "colour_default.frag.h"
const char *colour_default_glsl = &examples_default_glsl[0];

class invalid_data : public std::exception
{
  const char *msg;
public:
  invalid_data(const char *msg = "invalid data") : msg(msg) { }
  virtual const char *what() const noexcept override
  {
    return msg;
  }
};

struct uniform_type
{
  GLenum type;
  GLint size;
};

union uniform_value
{
  float float_;
  GLint int_;
  bool bool_;
  GLint sampler;
  // GL 3.0
  GLuint uint_;
#if 0
  // GL 4.1
  double double_;
  // GL 4.2
  GLint image;
  GLint atomic_uint;
#endif
};

struct ident
{
  std::string name;
  int index;
  int component;
};

struct fact
{
  ident id;
  GLenum atom_type;
  uniform_value value;
};

struct colour
{
  // the bigger picture
  int image_width, image_height;
  float zoom_log_2, time;
  // current size
  int width, height;
  // textures
  GLuint t_N0, t_N1, t_NF, t_T, t_DEX, t_DEY, t_BLA, t_PTB, t_RGBA, t_CDF;
  // shader
  GLuint program;
  GLint u_N0, u_N1, u_NF, u_T, u_DEX, u_DEY, u_BLA, u_PTB, u_CDF;
  GLint u_have_N0, u_have_N1, u_have_NF, u_have_T, u_have_DEX, u_have_DEY, u_have_BLA, u_have_PTB, u_have_CDF;
  GLint u_tile, u_tile_size, u_image_size, u_zoom_log_2, u_time, u_CDF_range;
  // drawing
  GLuint fbo, vbo, vao;
  // buffer
  float *RGBA;
  unsigned char *RGBA8;
  // histogram
  int cdf_status;
  double cdf_minimum;
  double cdf_maximum;
  double cdf_median;
  // custom colour
  std::string source, editsource;
  std::map<std::string, uniform_type> active;
  std::map<std::string, GLint> location;
  std::vector<fact> db;
  bool export_headerless;
};

colour *colour_new()
{
  colour *c = new colour();
  c->width = -1;
  c->height = -1;
  GLuint t[10];
  glGenTextures(10, &t[0]);
  c->t_N0 = t[0];
  c->t_N1 = t[1];
  c->t_NF = t[2];
  c->t_T  = t[3];
  c->t_DEX = t[4];
  c->t_DEY = t[5];
  c->t_BLA = t[6];
  c->t_PTB = t[7];
  c->t_RGBA = t[8];
  c->t_CDF = t[9];
  c->program = 0;
  colour_set_shader(c, examples_default_glsl);
  glGenFramebuffers(1, &c->fbo);
  glGenBuffers(1, &c->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
  float data[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
  glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
  glGenVertexArrays(1, &c->vao);
  glBindVertexArray(c->vao);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  c->export_headerless = false;
  return c;
}

void colour_delete(struct colour *u)
{
  delete[] u->RGBA;
  delete[] u->RGBA8;
  delete u;
}

void colour_set_image_size(colour *c, int width, int height)
{
  c->image_width = width;
  c->image_height = height;
}

void colour_set_zoom_log_2(colour *c, float zoom_log_2)
{
  c->zoom_log_2 = zoom_log_2;
}

void colour_set_time(colour *c, float time)
{
  c->time = time;
}

void colour_resize(colour *c, int width, int height)
{
  c->width = width;
  c->height = height;
  glActiveTexture(GL_TEXTURE0 + 0);
  glBindTexture(GL_TEXTURE_2D, c->t_N0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
#define TP \
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); \
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); \
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); \
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_N1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_NF);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_T);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_DEX);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_DEY);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_BLA);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_PTB);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, width, height, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_RGBA);
  if (GLAD_GL_EXT_color_buffer_float)
  {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
  }
  else
  {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  }
  TP
  glBindTexture(GL_TEXTURE_2D, c->t_CDF);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, nullptr);
  TP
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#undef TP
  glBindTexture(GL_TEXTURE_2D, 0);
  delete[] c->RGBA;
  delete[] c->RGBA8;
  c->RGBA = new float[width * height * 4];
  c->RGBA8 = new unsigned char[width * height * 4];
}

void colour_tile(colour *c, int x, int y, int subframe, tile *data)
{
  if (! data->RGB)
  {
    return;
  }
  if (data->width != c->width || data->height != c->height)
  {
    colour_resize(c, data->width, data->height);
  }

  glUseProgram(c->program);

  glUniform1i(c->u_N0, 0);
  glUniform1i(c->u_have_N0, !!data->N0);
  if (data->N0)
  {
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, c->t_N0);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED_INTEGER, GL_UNSIGNED_INT, data->N0);
  }

  glUniform1i(c->u_N1, 1);
  glUniform1i(c->u_have_N1, !!data->N1);
  if (data->N1)
  {
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, c->t_N1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED_INTEGER, GL_UNSIGNED_INT, data->N1);
  }

  glUniform1i(c->u_NF, 2);
  glUniform1i(c->u_have_NF, !!data->NF);
  if (data->NF)
  {
    glActiveTexture(GL_TEXTURE0 + 2);
    glBindTexture(GL_TEXTURE_2D, c->t_NF);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED, GL_FLOAT, data->NF);
  }

  glUniform1i(c->u_T, 3);
  glUniform1i(c->u_have_T, !!data->T);
  if (data->T)
  {
    glActiveTexture(GL_TEXTURE0 + 3);
    glBindTexture(GL_TEXTURE_2D, c->t_T);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED, GL_FLOAT, data->T);
  }

  glUniform1i(c->u_DEX, 4);
  glUniform1i(c->u_have_DEX, !!data->DEX);
  if (data->DEX)
  {
    glActiveTexture(GL_TEXTURE0 + 4);
    glBindTexture(GL_TEXTURE_2D, c->t_DEX);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED, GL_FLOAT, data->DEX);
  }

  glUniform1i(c->u_DEY, 5);
  glUniform1i(c->u_have_DEY, !!data->DEY);
  if (data->DEY)
  {
    glActiveTexture(GL_TEXTURE0 + 5);
    glBindTexture(GL_TEXTURE_2D, c->t_DEY);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED, GL_FLOAT, data->DEY);
  }

  glUniform1i(c->u_BLA, 6);
  glUniform1i(c->u_have_BLA, !!data->BLA);
  if (data->BLA)
  {
    glActiveTexture(GL_TEXTURE0 + 6);
    glBindTexture(GL_TEXTURE_2D, c->t_BLA);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED_INTEGER, GL_UNSIGNED_INT, data->BLA);
  }

  glUniform1i(c->u_PTB, 7);
  glUniform1i(c->u_have_PTB, !!data->PTB);
  if (data->PTB)
  {
    glActiveTexture(GL_TEXTURE0 + 7);
    glBindTexture(GL_TEXTURE_2D, c->t_PTB);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->width, data->height, GL_RED_INTEGER, GL_UNSIGNED_INT, data->PTB);
  }

  glUniform2i(c->u_image_size, c->image_width, c->image_height);
  glUniform1f(c->u_zoom_log_2, c->zoom_log_2);
  glUniform1f(c->u_time, c->time);
  glUniform2i(c->u_tile_size, data->width, data->height);
  glUniform3i(c->u_tile, x, y, subframe);

  glUniform1i(c->u_CDF, 8);
  glUniform1i(c->u_have_CDF, c->cdf_status);
  double m = c->cdf_median;
  double d = 2 * m * m - 2 * m;
  glUniform4f(c->u_CDF_range, c->cdf_minimum, c->cdf_maximum - c->cdf_minimum, (1 - 2 * m) / d, (2 * m * m - 1) / d);

  glBindFramebuffer(GL_FRAMEBUFFER, c->fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, c->t_RGBA, 0);
  GLenum buffer = GL_COLOR_ATTACHMENT0;
  glDrawBuffers(1, &buffer);
  glViewport(0, 0, data->width, data->height);

  glBindVertexArray(c->vao);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);

  glReadBuffer(buffer);
  if (GLAD_GL_EXT_color_buffer_float)
  {
    glReadPixels(0, 0, data->width, data->height, GL_RGBA, GL_FLOAT, c->RGBA);
  }
  else
  {
    glReadPixels(0, 0, data->width, data->height, GL_RGBA, GL_UNSIGNED_BYTE, c->RGBA8);
    float convert = 1.0f / 255.0f; // FIXME sRGB to linear?
    for (int y = 0; y < data->height; ++y)
    for (int x = 0; x < data->width; ++x)
    for (int z = 0; z < 4; ++z)
    {
      c->RGBA [(y * data->width + x) * 4 + z] =
      c->RGBA8[(y * data->width + x) * 4 + z] * convert;
    }
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glUseProgram(0);

  for (int y = 0; y < data->height; ++y)
  for (int x = 0; x < data->width; ++x)
  for (int z = 0; z < 3; ++z)
  {
    data->RGB[(y * data->width + x) * 3 + z] =
      c->RGBA[(y * data->width + x) * 4 + z];
  }
}

std::set<std::string> uniform_internal
{ "Internal_N0"
, "Internal_N1"
, "Internal_NF"
, "Internal_T"
, "Internal_DEX"
, "Internal_DEY"
, "Internal_BLA"
, "Internal_PTB"
, "Internal_have_N0"
, "Internal_have_N1"
, "Internal_have_NF"
, "Internal_have_T"
, "Internal_have_DEX"
, "Internal_have_DEY"
, "Internal_have_BLA"
, "Internal_have_PTB"
, "Internal_CDF"
, "Internal_have_CDF"
, "Internal_CDF_range"
, "Internal_tile"
, "Internal_tile_size"
, "Internal_image_size"
, "Internal_frame"
, "Internal_zoom_log_2"
, "Internal_time"
};

int uniform_atoms(const GLenum type)
{
  switch (type)
  {
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT:
    case GL_BOOL:
#if 0
    case GL_DOUBLE:
#endif
      return 1;
    case GL_FLOAT_VEC2:
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
    case GL_BOOL_VEC2:
#if 0
    case GL_DOUBLE_VEC2:
#endif
      return 2;
    case GL_FLOAT_VEC3:
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
    case GL_BOOL_VEC3:
#if 0
    case GL_DOUBLE_VEC3:
#endif
      return 3;
    case GL_FLOAT_VEC4:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
    case GL_BOOL_VEC4:
#if 0
    case GL_DOUBLE_VEC4:
#endif
      return 4;
    case GL_FLOAT_MAT2: return 2 * 2;
    case GL_FLOAT_MAT3: return 3 * 3;
    case GL_FLOAT_MAT4: return 4 * 4;
    case GL_FLOAT_MAT2x3: return 2 * 3;
    case GL_FLOAT_MAT2x4: return 2 * 4;
    case GL_FLOAT_MAT3x2: return 3 * 2;
    case GL_FLOAT_MAT3x4: return 3 * 4;
    case GL_FLOAT_MAT4x2: return 4 * 2;
    case GL_FLOAT_MAT4x3: return 4 * 3;
#if 0
    // double precision requires GL 4.1
    case GL_DOUBLE_MAT2: return 2 * 2;
    case GL_DOUBLE_MAT3: return 3 * 3;
    case GL_DOUBLE_MAT4: return 4 * 4;
    case GL_DOUBLE_MAT2x3: return 2 * 3;
    case GL_DOUBLE_MAT2x4: return 2 * 4;
    case GL_DOUBLE_MAT3x2: return 3 * 2;
    case GL_DOUBLE_MAT3x4: return 3 * 4;
    case GL_DOUBLE_MAT4x2: return 4 * 2;
    case GL_DOUBLE_MAT4x3: return 4 * 3;
#endif
    // samplers
    case GL_SAMPLER_2D:
    case GL_SAMPLER_3D:
    case GL_SAMPLER_CUBE:
    case GL_SAMPLER_2D_SHADOW:
    case GL_SAMPLER_2D_ARRAY:
    case GL_SAMPLER_2D_ARRAY_SHADOW:
    case GL_SAMPLER_CUBE_SHADOW:
#if 0
    case GL_SAMPLER_1D:
    case GL_SAMPLER_1D_SHADOW:
    case GL_SAMPLER_1D_ARRAY:
    case GL_SAMPLER_1D_ARRAY_SHADOW:
    case GL_SAMPLER_2D_MULTISAMPLE:
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_SAMPLER_BUFFER:
    case GL_SAMPLER_2D_RECT:
    case GL_SAMPLER_2D_RECT_SHADOW:
    case GL_INT_SAMPLER_1D:
    case GL_INT_SAMPLER_1D_ARRAY:
    case GL_INT_SAMPLER_BUFFER:
    case GL_INT_SAMPLER_2D_MULTISAMPLE:
    case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_INT_SAMPLER_2D_RECT:
    case GL_UNSIGNED_INT_SAMPLER_1D:
    case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_BUFFER:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
#endif
    case GL_INT_SAMPLER_2D:
    case GL_INT_SAMPLER_3D:
    case GL_INT_SAMPLER_CUBE:
    case GL_INT_SAMPLER_2D_ARRAY:
    case GL_UNSIGNED_INT_SAMPLER_2D:
    case GL_UNSIGNED_INT_SAMPLER_3D:
    case GL_UNSIGNED_INT_SAMPLER_CUBE:
    case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
#if 0
    // images require GL 4.2
    case GL_IMAGE_1D:
    case GL_IMAGE_2D:
    case GL_IMAGE_3D:
    case GL_IMAGE_2D_RECT:
    case GL_IMAGE_CUBE:
    case GL_IMAGE_BUFFER:
    case GL_IMAGE_1D_ARRAY:
    case GL_IMAGE_2D_ARRAY:
    case GL_IMAGE_2D_MULTISAMPLE:
    case GL_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_INT_IMAGE_1D:
    case GL_INT_IMAGE_2D:
    case GL_INT_IMAGE_3D:
    case GL_INT_IMAGE_2D_RECT:
    case GL_INT_IMAGE_CUBE:
    case GL_INT_IMAGE_BUFFER:
    case GL_INT_IMAGE_1D_ARRAY:
    case GL_INT_IMAGE_2D_ARRAY:
    case GL_INT_IMAGE_2D_MULTISAMPLE:
    case GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_1D:
    case GL_UNSIGNED_INT_IMAGE_2D:
    case GL_UNSIGNED_INT_IMAGE_3D:
    case GL_UNSIGNED_INT_IMAGE_2D_RECT:
    case GL_UNSIGNED_INT_IMAGE_CUBE:
    case GL_UNSIGNED_INT_IMAGE_BUFFER:
    case GL_UNSIGNED_INT_IMAGE_1D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE:
    case GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY:
    // counter requires GL 4.2
    case GL_UNSIGNED_INT_ATOMIC_COUNTER:
#endif
      return 1;
    // unsupported type
    default:
      return 1;
  }
}

GLenum uniform_atom_type(const GLenum type)
{
  switch (type)
  {
    case GL_FLOAT:
    case GL_FLOAT_VEC2:
    case GL_FLOAT_VEC3:
    case GL_FLOAT_VEC4:
    case GL_FLOAT_MAT2:
    case GL_FLOAT_MAT3:
    case GL_FLOAT_MAT4:
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT2x4:
    case GL_FLOAT_MAT3x2:
    case GL_FLOAT_MAT3x4:
    case GL_FLOAT_MAT4x2:
    case GL_FLOAT_MAT4x3:
      return GL_FLOAT;
    case GL_INT:
    case GL_INT_VEC2:
    case GL_INT_VEC3:
    case GL_INT_VEC4:
      return GL_INT;
    case GL_UNSIGNED_INT:
    case GL_UNSIGNED_INT_VEC2:
    case GL_UNSIGNED_INT_VEC3:
    case GL_UNSIGNED_INT_VEC4:
      return GL_UNSIGNED_INT;
    case GL_BOOL:
    case GL_BOOL_VEC2:
    case GL_BOOL_VEC3:
    case GL_BOOL_VEC4:
      return GL_BOOL;
#if 0
    case GL_DOUBLE:
    case GL_DOUBLE_VEC2:
    case GL_DOUBLE_VEC3:
    case GL_DOUBLE_VEC4:
    case GL_DOUBLE_MAT2:
    case GL_DOUBLE_MAT3:
    case GL_DOUBLE_MAT4:
    case GL_DOUBLE_MAT2x3:
    case GL_DOUBLE_MAT2x4:
    case GL_DOUBLE_MAT3x2:
    case GL_DOUBLE_MAT3x4:
    case GL_DOUBLE_MAT4x2:
    case GL_DOUBLE_MAT4x3:
      return GL_DOUBLE;
#endif
    default:
      return type;
  }
}

std::map<std::string, uniform_type> uniform_get_active(GLuint program, bool include_internal)
{
  std::map<std::string, uniform_type> result;
  GLint nactive = 0, len = 0;
  glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &nactive);
  glGetProgramiv(program, GL_ACTIVE_UNIFORM_MAX_LENGTH, &len);
  GLchar *buffer = new GLchar[len];
  for (GLint index = 0; index < nactive; ++index)
  {
    uniform_type t;
    glGetActiveUniform(program, index, len, 0, &t.size, &t.type, buffer);
    std::string name(buffer);
    if (include_internal || uniform_internal.find(name) == uniform_internal.end())
    {
      result.insert(std::pair<std::string, uniform_type>(name, t));
    }
  }
  delete[] buffer;
  return result;
}

std::map<std::string, GLint> uniform_get_location(GLuint program, const std::map<std::string, uniform_type> &active)
{
  std::map<std::string, GLint> result;
  for (const auto & [ name, type ] : active)
  {
    GLint loc = glGetUniformLocation(program, name.c_str());
    result.insert(std::pair<std::string, GLint>(name, loc));
  }
  return result;
}

std::pair<uniform_value, uniform_value> uniform_get_initial_range(const uniform_type &type)
{
  uniform_value minimum, maximum;
  std::memset(&minimum, 0, sizeof(minimum));
  std::memset(&maximum, 0, sizeof(maximum));
  switch (uniform_atom_type(type.type))
  {
    case GL_FLOAT:
    {
      maximum.float_ = 1;
      break;
    }
#if 0
    case GL_DOUBLE:
    {
      maximum.double_ = 1;
      break;
    }
#endif
    case GL_INT:
    {
      minimum.int_ = INT_MIN;
      maximum.int_ = INT_MAX;
      break;
    }
    case GL_UNSIGNED_INT:
    {
      maximum.uint_ = UINT_MAX;
      break;
    }
    case GL_BOOL:
    {
      maximum.bool_ = 1;
      break;
    }
    default:
    {
      // TODO
    }
  }
  return std::pair<uniform_value, uniform_value>(minimum, maximum);
}

bool operator<(const ident &a, const ident &b)
{
  return a.name < b.name ||
    (a.name == b.name && (a.index < b.index ||
      (a.index == b.index && a.component < b.component)));
}

bool operator==(const ident &a, const ident &b)
{
  return a.name == b.name && a.index == b.index && a.component == b.component;
}

bool operator!=(const ident &a, const ident &b)
{
  return ! (a == b);
}

bool fact_cmp_ident(const fact &a, const fact &b)
{
  return a.id < b.id;
}

bool operator==(const fact &a, const fact &b)
{
  if (a.id != b.id) return false;
  if (a.atom_type != b.atom_type) return false;
  switch (uniform_atom_type(a.atom_type))
  {
    case GL_FLOAT: return a.value.float_ == b.value.float_;
    case GL_INT: return a.value.int_ == b.value.int_;
    case GL_BOOL: return a.value.bool_ == b.value.bool_;
    case GL_UNSIGNED_INT: return a.value.uint_ == b.value.uint_;
#if 0
    case GL_DOUBLE: return a.value.double_ == b.value.double_;
#endif
    default: return false; // FIXME
  }
}

void facts_sort_ident(std::vector<fact> &db)
{
  std::sort(db.begin(), db.end(), fact_cmp_ident);
}

bool facts_display_table(std::vector<fact> &table)
{
  int cell_id = 0;
  bool modified = false;
  if (table.empty())
  {
    return modified;
  }
  for (auto& f : table)
  {
    ImGui::PushID(cell_id++);
    std::ostringstream label;
    label << f.id.name << "[" << f.id.index << "]" << "[" << f.id.component << "]"; // TODO trim
    switch (f.atom_type)
    {
      case GL_FLOAT:
      {
        label << "##Float";
        modified |= ImGui::InputFloat(label.str().c_str(), &f.value.float_, 0.0f, 0.0f, "%.7g"); // FIXME range
        break;
      }
#if 0
      case GL_DOUBLE:
      {
        label << "##Double";
        modified |= ImGui::InputDouble(label.str().c_str(), &f.value.double_, 0.0, 0.0, "%.17g"); // FIXME range
        break;
      }
#endif
      case GL_INT:
      {
        label << "##Int";
        modified |= ImGui::InputInt(label.str().c_str(), &f.value.int_);
        break;
      }
      case GL_UNSIGNED_INT:
      {
        label << "##UInt";
        int i = f.value.uint_;
        modified |= ImGui::InputInt(label.str().c_str(), &i);
        f.value.uint_ = i;
        break;
      }
      case GL_BOOL:
      {
        label << "##Bool";
        modified |= ImGui::Checkbox(label.str().c_str(), &f.value.bool_);
        break;
      }
      default:
      {
        ImGui::Text("?");
        break;
      }
    }
    ImGui::PopID();
  }
  return modified;
}

void colour_upload(const struct colour *u)
{
  glUseProgram(u->program);
  const bool transpose = false;
  std::map<ident, uniform_value> cooked;
  for (const auto & f : u->db)
  {
    cooked[f.id] = f.value;
  }
  for (const auto & [name, type] : u->active)
  {
    GLint loc = -1;
    try
    {
      loc = u->location.at(name);
    }
    catch (const std::out_of_range &e)
    {
      // ignore
    }
    if (loc == -1)
    {
      continue;
    }
    int nj = uniform_atoms(type.type);
    int atoms = type.size * nj;
    switch (uniform_atom_type(type.type))
    {
      case GL_FLOAT:
      {
        float *data = new float[atoms];
        for (int i = 0; i < type.size; ++i)
        {
          for (int j = 0; j < nj; ++j)
          {
            uniform_value value;
            value.float_ = 0;
            try
            {
              ident id = { name, i, j };
              value = cooked.at(id);
            }
            catch (const std::out_of_range &e)
            {
              // ignore
            }
            data[i * nj + j] = value.float_;
          }
        }
        switch (type.type)
        {
          case GL_FLOAT:        glUniform1fv(loc, type.size, data); break;
          case GL_FLOAT_VEC2:   glUniform2fv(loc, type.size, data); break;
          case GL_FLOAT_VEC3:   glUniform3fv(loc, type.size, data); break;
          case GL_FLOAT_VEC4:   glUniform4fv(loc, type.size, data); break;
          case GL_FLOAT_MAT2:   glUniformMatrix2fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT3:   glUniformMatrix3fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT4:   glUniformMatrix4fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT2x3: glUniformMatrix2x3fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT2x4: glUniformMatrix2x4fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT3x2: glUniformMatrix3x2fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT3x4: glUniformMatrix3x4fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT4x2: glUniformMatrix4x2fv(loc, type.size, transpose, data); break;
          case GL_FLOAT_MAT4x3: glUniformMatrix4x3fv(loc, type.size, transpose, data); break;
        }
        delete[] data;
        break;
      }

#if 0
      case GL_DOUBLE:
      {
        double *data = new double[atoms];
        for (int i = 0; i < type.size; ++i)
        {
          for (int j = 0; j < nj; ++j)
          {
            uniform_value value;
            value.double_ = 0;
            try
            {
              ident id = { name, i, j };
              value = cooked.at(id);
            }
            catch (const std::out_of_range &e)
            {
              // ignore
            }
            data[i * nj + j] = value.double_;
          }
        }
        switch (type.type)
        {
          case GL_DOUBLE:        glUniform1dv(loc, type.size, data); break;
          case GL_DOUBLE_VEC2:   glUniform2dv(loc, type.size, data); break;
          case GL_DOUBLE_VEC3:   glUniform3dv(loc, type.size, data); break;
          case GL_DOUBLE_VEC4:   glUniform4dv(loc, type.size, data); break;
          case GL_DOUBLE_MAT2:   glUniformMatrix2dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT3:   glUniformMatrix3dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT4:   glUniformMatrix4dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT2x3: glUniformMatrix2x3dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT2x4: glUniformMatrix2x4dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT3x2: glUniformMatrix3x2dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT3x4: glUniformMatrix3x4dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT4x2: glUniformMatrix4x2dv(loc, type.size, transpose, data); break;
          case GL_DOUBLE_MAT4x3: glUniformMatrix4x3dv(loc, type.size, transpose, data); break;
        }
        delete[] data;
        break;
      }
#endif

      case GL_BOOL:
      {
        GLint *data = new GLint[atoms];
        for (int i = 0; i < type.size; ++i)
        {
          for (int j = 0; j < nj; ++j)
          {
            uniform_value value;
            value.bool_ = 0;
            try
            {
              ident id = { name, i, j };
              value = cooked.at(id);
            }
            catch (const std::out_of_range &e)
            {
              // ignore
            }
            data[i * nj + j] = value.bool_;
          }
        }
        switch (type.type)
        {
          case GL_BOOL:        glUniform1iv(loc, type.size, data); break;
          case GL_BOOL_VEC2:   glUniform2iv(loc, type.size, data); break;
          case GL_BOOL_VEC3:   glUniform3iv(loc, type.size, data); break;
          case GL_BOOL_VEC4:   glUniform4iv(loc, type.size, data); break;
        }
        delete[] data;
        break;
      }

      case GL_INT:
      {
        GLint *data = new GLint[atoms];
        for (int i = 0; i < type.size; ++i)
        {
          for (int j = 0; j < nj; ++j)
          {
            uniform_value value;
            value.int_ = 0;
            try
            {
              ident id = { name, i, j };
              value = cooked.at(id);
            }
            catch (const std::out_of_range &e)
            {
              // ignore
            }
            data[i * nj + j] = value.int_;
          }
        }
        switch (type.type)
        {
          case GL_INT:        glUniform1iv(loc, type.size, data); break;
          case GL_INT_VEC2:   glUniform2iv(loc, type.size, data); break;
          case GL_INT_VEC3:   glUniform3iv(loc, type.size, data); break;
          case GL_INT_VEC4:   glUniform4iv(loc, type.size, data); break;
        }
        delete[] data;
        break;
      }

      case GL_UNSIGNED_INT:
      {
        GLuint *data = new GLuint[atoms];
        for (int i = 0; i < type.size; ++i)
        {
          for (int j = 0; j < nj; ++j)
          {
            uniform_value value;
            value.int_ = 0;
            try
            {
              ident id = { name, i, j };
              value = cooked.at(id);
            }
            catch (const std::out_of_range &e)
            {
              // ignore
            }
            data[i * nj + j] = value.int_;
          }
        }
        switch (type.type)
        {
          case GL_UNSIGNED_INT:        glUniform1uiv(loc, type.size, data); break;
          case GL_UNSIGNED_INT_VEC2:   glUniform2uiv(loc, type.size, data); break;
          case GL_UNSIGNED_INT_VEC3:   glUniform3uiv(loc, type.size, data); break;
          case GL_UNSIGNED_INT_VEC4:   glUniform4uiv(loc, type.size, data); break;
        }
        delete[] data;
        break;
      }
    }
  }
  glUseProgram(0);
}

void colour_upload_cdf(struct colour *u, const struct histogramcdf &h)
{
  glActiveTexture(GL_TEXTURE0 + 8);
  glBindTexture(GL_TEXTURE_2D, u->t_CDF);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, h.data.size(), 1, 0, GL_RED, GL_FLOAT, &h.data[0]);
  u->cdf_status = 2;
  u->cdf_minimum = h.minimum;
  u->cdf_maximum = h.maximum;
  u->cdf_median = h.median;
}

void colour_stale_cdf(struct colour *u)
{
  u->cdf_status = 1;
}

// preserve any existing values
void colour_set_program(struct colour *u, GLuint program)
{
  if (u->program)
  {
    glDeleteProgram(u->program);
    u->program = 0;
  }
  uniform_value value;
  std::memset(&value, 0, sizeof(value));
  u->program = program;
  u->active = uniform_get_active(u->program, false);
  u->location = uniform_get_location(u->program, u->active);
  std::map<ident, fact> cooked;
  std::set<ident> present;
  for (const auto & f : u->db)
  {
    cooked[f.id] = f;
  }
  for (const auto& [ name, type ] : u->active)
  {
    for (int index = 0; index < type.size; ++index)
    {
      for (int component = 0; component < uniform_atoms(type.type); ++component)
      {
        fact f =
          { { name, index, component }
          , uniform_atom_type(type.type)
          , value
          };
        present.insert(f.id);
        if (cooked.find(f.id) == cooked.end())
        {
          cooked[f.id] = f;
        }
      }
    }
  }
  u->db.clear();
  for (const auto & id : present)
  {
    u->db.push_back(cooked[id]);
  }
  glUseProgram(u->program);
  u->u_N0 = glGetUniformLocation(u->program, "Internal_N0");
  u->u_N1 = glGetUniformLocation(u->program, "Internal_N1");
  u->u_NF = glGetUniformLocation(u->program, "Internal_NF");
  u->u_T  = glGetUniformLocation(u->program, "Internal_T");
  u->u_DEX = glGetUniformLocation(u->program, "Internal_DEX");
  u->u_DEY = glGetUniformLocation(u->program, "Internal_DEY");
  u->u_BLA = glGetUniformLocation(u->program, "Internal_BLA");
  u->u_PTB = glGetUniformLocation(u->program, "Internal_PTB");
  u->u_have_N0 = glGetUniformLocation(u->program, "Internal_have_N0");
  u->u_have_N1 = glGetUniformLocation(u->program, "Internal_have_N1");
  u->u_have_NF = glGetUniformLocation(u->program, "Internal_have_NF");
  u->u_have_T  = glGetUniformLocation(u->program, "Internal_have_T");
  u->u_have_DEX = glGetUniformLocation(u->program, "Internal_have_DEX");
  u->u_have_DEY = glGetUniformLocation(u->program, "Internal_have_DEY");
  u->u_have_BLA = glGetUniformLocation(u->program, "Internal_have_BLA");
  u->u_have_PTB = glGetUniformLocation(u->program, "Internal_have_PTB");
  u->u_tile = glGetUniformLocation(u->program, "Internal_tile");
  u->u_tile_size = glGetUniformLocation(u->program, "Internal_tile_size");
  u->u_image_size = glGetUniformLocation(u->program, "Internal_image_size");
  u->u_zoom_log_2 = glGetUniformLocation(u->program, "Internal_zoom_log_2");
  u->u_time = glGetUniformLocation(u->program, "Internal_time");
  u->u_CDF = glGetUniformLocation(u->program, "Internal_CDF");
  u->u_have_CDF = glGetUniformLocation(u->program, "Internal_have_CDF");
  u->u_CDF_range = glGetUniformLocation(u->program, "Internal_CDF_range");
  glUseProgram(0);
}

bool colour_set_shader(colour *c, std::string source)
{
  c->editsource = source;
  GLuint program = vertex_fragment_shader(version, src_colour_vert_glsl, src_colour_frag_glsl, source.c_str());
  if (program)
  {
    c->source = source;
    colour_set_program(c, program);
    return true;
  }
  else
  {
    return false;
  }
}

std::string colour_get_shader(colour *c)
{
  return c->source;
}

bool colour_reset_unlocked = false;
bool auto_compile = true;
extern bool colour_display(struct colour *u, bool show_gui)
{
  bool modified = false;
  if (show_gui)
  {
    ImGui::Checkbox("##ResetUnlock", &colour_reset_unlocked);
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
      if (colour_reset_unlocked)
      {
        colour_reset_unlocked = false;
        colour_set_shader(u, examples_default_glsl);
        modified = true;
      }
    }
    auto table = u->db;
    modified |= facts_display_table(table);
    if (modified)
    {
      u->db = table;
    }
    if (ImGui::Checkbox("Auto compile", &auto_compile))
    {
      if (auto_compile)
      {
        modified |= colour_set_shader(u, u->editsource);
      }
    }
    if (! auto_compile)
    {
      ImGui::SameLine();
      if (ImGui::Button("Compile"))
      {
        modified |= colour_set_shader(u, u->editsource);
      }
    }
    ImGui::Text("Shader source code:");
    if (ImGui::InputTextMultiline("##Source", &u->editsource, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 16), ImGuiInputTextFlags_AllowTabInput))
    {
      if (auto_compile)
      {
        modified |= colour_set_shader(u, u->editsource);
      }
    }
    ImGui::Text("Shader compilation log:");
    ImGui::TextUnformatted(shader_log.begin(), shader_log.end());
  }
  return modified;
}

std::vector<ident> colour_parse_header(const std::string &s)
{
  std::vector<ident> result;
  std::stringstream ss(s);
  std::string field;
  ident last_id = { "n/a", -1, -1 };
  while (std::getline(ss, field, ','))
  {
    ident id = { "n/a", -1, -1 };
    size_t pos = 0;
    if ((pos = field.find('[')) != std::string::npos)
    {
      id.name = field.substr(0, pos);
      field.erase(0, pos + 1);
      if ((pos = field.find(']')) != std::string::npos)
      {
        id.index = std::stoi(field.substr(0, pos));
        field.erase(0, pos + 1);
        if ((pos = field.find('[')) != std::string::npos)
        {
          field.erase(0, pos + 1);
          if ((pos = field.find(']')) != std::string::npos)
          {
            id.component = std::stoi(field.substr(0, pos));
            result.push_back(id);
            last_id = id;
          }
          else
          {
            throw invalid_data("expected ]");
          }
        }
        else
        {
          throw invalid_data("expected [");
        }
      }
      else
      {
        throw invalid_data("expected ]");
      }
    }
    else
    {
      throw invalid_data("expected [");
    }
  }
  return result;
}

extern void colour_export_csv(std::ostream &out, struct colour *u)
{
  if (! u->export_headerless)
  {
    bool first = true;
    for (const auto & [ name, type ] : u->active)
    {
      for (int i = 0; i < type.size; ++i)
      {
        for (int j = 0; j < uniform_atoms(type.type); ++j)
        {
          if (! first)
          {
            out << ",";
          }
          first = false;
          out << name << "[" << i << "]" << "[" << j << "]";
        }
      }
    }
    out << "\r\n";
  }
  auto row = u->db;
  if (row.empty())
  {
    return;
  }
  std::map<ident, fact> cooked;
  for (const auto & f : row)
  {
    cooked[f.id] = f;
  }
  bool first = true;
  for (const auto & [ name, type ] : u->active)
  {
    for (int i = 0; i < type.size; ++i)
    {
      for (int j = 0; j < uniform_atoms(type.type); ++j)
      {
        if (! first)
        {
          out << ",";
        }
        first = false;
        try
        {
          ident id = { name, i, j };
          fact P = cooked.at(id);
          switch (P.atom_type)
          {
            case GL_FLOAT:
            {
              out << std::defaultfloat << std::setprecision(7) << P.value.float_;
              break;
            }
#if 0
            case GL_DOUBLE:
            {
              out << std::defaultfloat << std::setprecision(17) << P.value.double_;
              break;
            }
#endif
            case GL_INT:
            {
              out << P.value.int_;
              break;
            }
            case GL_UNSIGNED_INT:
            {
              out << P.value.uint_;
              break;
            }
            case GL_BOOL:
            {
              out << (P.value.bool_ ? 1 : 0);
              break;
            }
            default:
            {
              break;
            }
          }
        }
        catch (const std::out_of_range &e)
        {
          // ignore
        }
      }
    }
  }
  out << "\r\n";
}

std::vector<fact> colour_import_csv(std::istream &in, struct colour *u)
{
  std::vector<fact> csv;
  std::string line;
  getline(in, line, '\n');
  std::vector<ident> fields = colour_parse_header(line);
  while (! getline(in, line, '\n').eof())
  {
    size_t cr = line.find('\r');
    if (cr != std::string::npos)
    {
      line = line.substr(0, cr);
    }
    if (line == "")
    {
      continue;
    }
    size_t pos = 0;
    std::string line0 = line;
    for (const auto &field : fields)
    {
        pos = line.find(',');
        if (pos == std::string::npos)
        {
          pos = line.size();
        }
        std::string svalue = line.substr(0, pos);
        line.erase(0, pos + 1);
        if (svalue != "")
        {
          try
          {
            GLenum type = uniform_atom_type(u->active.at(field.name).type);
            fact f = { field, type, {0} };
            switch (type)
            {
              case GL_FLOAT:
              {
                f.value.float_ = std::stof(svalue);
                break;
              }
#if 0
              case GL_DOUBLE:
              {
                f.value.double_ = std::stof(svalue);
                break;
              }
#endif
              case GL_INT:
              {
                f.value.int_ = std::stoi(svalue);
                break;
              }
              case GL_UNSIGNED_INT:
              {
                f.value.uint_ = std::stoi(svalue);
                break;
              }
              case GL_BOOL:
              {
                f.value.bool_ = std::stoi(svalue) != 0;
                break;
              }
              default:
              {
                throw invalid_data("bad uniform type");
              }
            }
            csv.push_back(f);
          }
          catch (const std::out_of_range &e)
          {
            // ignore
          }
      }
      else
      {
        throw invalid_data("expected field");
      }
    }
  }
  return csv;
}

bool colour_save_csv(struct colour *u, const std::string &file)
{
  try
  {
    std::ofstream out(file);
    out.imbue(std::locale::classic());
    colour_export_csv(out, u);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

void colour_load_csv(struct colour *u, const std::string &file)
{
  try
  {
    std::ifstream in(file);
    in.imbue(std::locale::classic());
    u->db = colour_import_csv(in, u);
  }
  catch (...)
  {
    // FIXME ignored
  }
}

std::vector<std::map<std::string, patom>> colour_get_uniforms(colour *u)
{
  std::vector<std::map<std::string, patom>> r;
  for (const auto & f : u->db)
  {
    std::map<std::string, patom> t;
    t["name"] = patom(f.id.name);
    t["index"] = patom(f.id.index);
    t["component"] = patom(f.id.component);
    switch (f.atom_type)
    {
      case GL_FLOAT:
      {
        t["type"] = patom("float");
        t["value"] = patom(f.value.float_);
        break;
      }
#if 0
      case GL_DOUBLE:
      {
        t.insert("type", "double");
        t.insert("value", f.value.double_);
        break;
      }
#endif
      case GL_INT:
      {
        t["type"] = patom("int");
        t["value"] = patom(f.value.int_);
        break;
      }
      case GL_UNSIGNED_INT:
      {
        t["type"] = patom("uint");
        t["value"] = patom(int64_t(f.value.uint_));
        break;
      }
      case GL_BOOL:
      {
        t["type"] = patom("bool");
        t["value"] = patom(f.value.bool_ ? 1 : 0);
        break;
      }
      default:
      {
        // TODO
      }
    }
    r.push_back(t);
  }
  return r;
}

bool colour_set_uniforms(colour *u, std::vector<std::map<std::string, patom>> &r)
{
  try
  {
    std::vector<fact> result;
    for (auto & t : r)
    {
        fact f;
        if (t["name"].tag == t_string)
        {
          f.id.name = std::string(t["name"].string_);
        }
        else
        {
          throw invalid_data("name not string");
        }
        if (t["index"].tag == t_int64)
        {
          f.id.index = int64_t(t["index"].int64_);
        }
        else
        {
          throw invalid_data("index not integer");
        }
        if (t["component"].tag == t_int64)
        {
          f.id.component = int64_t(t["component"].int64_);
        }
        else
        {
          throw invalid_data("component not integer");
        }
        if (t["type"].tag == t_string)
        {
          std::string type = std::string(t["type"].string_);
          if (type == "float")
          {
            f.atom_type = GL_FLOAT;
            if (t["value"].tag == t_double)
            {
              f.value.float_ = t["value"].double_;
            }
            else
            {
              throw invalid_data("value not floating point");
            }
          }
#if 0
          else if (type == "double")
          {
            f.atom_type = GL_DOUBLE;
            if (t["value"].tag == t_double)
            {
              f.value.double_ = t["value"].double_);
            }
            else
            {
              throw invalid_data("value not floating point");
            }
          }
#endif
          else if (type == "int")
          {
            f.atom_type = GL_INT;
            if (t["value"].tag == t_int64)
            {
              f.value.int_ = t["value"].int64_;
            }
            else
            {
              throw invalid_data("value not integer");
            }
          }
          else if (type == "uint")
          {
            f.atom_type = GL_UNSIGNED_INT;
            if (t["value"].tag == t_int64)
            {
              f.value.uint_ = t["value"].int64_;
            }
            else
            {
              throw invalid_data("value not integer");
            }
          }
          else if (type == "bool")
          {
            f.atom_type = GL_BOOL;
            if (t["value"].tag == t_int64)
            {
              f.value.bool_ = t["value"].int64_;
            }
            else
            {
              throw invalid_data("value not integer");
            }
          }
          else
          {
            throw invalid_data("invalid type");
          }
        }
        else
        {
          throw invalid_data("type not string");
        }
        result.push_back(f);
    }
    u->db = result;
    return true;
  }
  catch (const invalid_data &e)
  {
    std::cerr << e.what() << std::endl;
    return false;
  }
}
