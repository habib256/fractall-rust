// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifdef HAVE_GUI

#include "types.h"

struct ImGuiTextBuffer;
extern ImGuiTextBuffer shader_log;

bool debug_program(GLuint program);
bool debug_shader(GLuint shader, GLenum type);
GLuint vertex_fragment_shader(const char *version, const char *vert, const char *frag, const char *frag2 = nullptr);

#endif
