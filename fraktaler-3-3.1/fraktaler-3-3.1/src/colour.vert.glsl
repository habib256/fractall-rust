// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

in vec2 v_position;
out vec2 Internal_coord;

void main(void)
{
  Internal_coord = v_position;
  gl_Position = vec4(2.0 * v_position - vec2(1.0), 0.0, 1.0);
}
