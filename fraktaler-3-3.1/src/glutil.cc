// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#ifdef HAVE_GUI

#include <iostream>

#include <imgui.h>

#include "glutil.h"

ImGuiTextBuffer shader_log;

bool debug_program(GLuint program) {
  GLint status = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  GLint length = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
  char *info = nullptr;
  if (length) {
    info = new char[length + 1];
    info[0] = 0;
    glGetProgramInfoLog(program, length, 0, info);
    info[length] = 0;
  }
  if ((info && info[0]) || ! status) {
    shader_log.appendf("\nlink info:\n%s", info ? info : "(no info log)\n");
  }
  delete[] info;
  return status;
}

bool debug_shader(GLuint shader, GLenum type) {
  GLint status = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  GLint length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
  char *info = nullptr;
  if (length) {
    info = new char[length + 1];
    info[0] = 0;
    glGetShaderInfoLog(shader, length, 0, info);
    info[length] = 0;
  }
  if ((info && info[0]) || ! status) {
    const char *type_str = "unknown";
    switch (type) {
      case GL_VERTEX_SHADER: type_str = "vertex"; break;
      case GL_FRAGMENT_SHADER: type_str = "fragment"; break;
    }
    shader_log.appendf("\n%s info:\n%s", type_str, info ? info : "(no info log)\n");
  }
  delete[] info;
  return status;
}

GLuint vertex_fragment_shader(const char *version, const char *vert, const char *frag, const char *frag2)
{
  shader_log.clear();
  bool ok = true;
  GLuint program = glCreateProgram();
  {
    GLuint shader = glCreateShader(GL_VERTEX_SHADER);
    const char *sources[] = { version, vert };
    glShaderSource(shader, 2, sources, 0);
    glCompileShader(shader);
    ok &= debug_shader(shader, GL_VERTEX_SHADER);
    glAttachShader(program, shader);
    glDeleteShader(shader);
  }
  {
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER);
    const char *sources[] = { version, frag, frag2 };
    glShaderSource(shader, frag2 ? 3 : 2, sources, 0);
    glCompileShader(shader);
    ok &= debug_shader(shader, GL_FRAGMENT_SHADER);
    glAttachShader(program, shader);
    glDeleteShader(shader);
  }
  glBindAttribLocation(program, 0, "v_position"); // FIXME hack
  glBindAttribLocation(program, 1, "v_texcoord"); // FIXME hack
  glLinkProgram(program);
  ok &= debug_program(program);
  if (! ok)
  {
    glDeleteProgram(program);
    program = 0;
  }
  return program;
}

#endif
