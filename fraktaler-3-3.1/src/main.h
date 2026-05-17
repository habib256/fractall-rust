// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "param.h"

extern param par;

int batch(int verbosity, const param &par);
int gui(const char *progname, const char *persistence, bool batchmode);

extern std::string pref_path;
