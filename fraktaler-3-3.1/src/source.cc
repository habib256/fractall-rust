// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstdio>

#include "source.h"

#include "fraktaler-3-source.7z.h"

const char *fraktaler_3_version_string = FRAKTALER_3_VERSION_STRING;
const std::string source_filename = "fraktaler-3-" FRAKTALER_3_VERSION_STRING ".7z";

int write_file(const char *name, const unsigned char *start, const unsigned char *end)
{
  std::fprintf(stderr, "writing '%s'... ", name);
  FILE *file = std::fopen(name, "wxb");
  if (! file)
  {
    std::fprintf(stderr, "FAILED\n");
    return 0;
  }
  int ok = (std::fwrite(start, end - start, 1, file) == 1);
  std::fclose(file);
  if (ok)
  {
    std::fprintf(stderr, "ok\n");
  }
  else
  {
    std::fprintf(stderr, "FAILED\n");
  }
  return ok;
}

bool write_source(const std::string &file)
{
  return write_file(file.c_str(), fraktaler_3_source_7z, fraktaler_3_source_7z + fraktaler_3_source_7z_len);
}
