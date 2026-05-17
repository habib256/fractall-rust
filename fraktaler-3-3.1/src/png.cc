// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

// adapted from my implementation for KF2+

#define _DEFAULT_SOURCE

#include <cstdio>
#include <ctime>
#include <string>
#include <string.h>

#include <png.h>
#include <zlib.h>

#include "png.h"

static void error_handler(png_structp png, png_const_charp msg)
{
  // FIXME make this display in the GUI or something
  std::fprintf(stderr, "PNG ERROR: %s\n", msg);
  longjmp(*static_cast<jmp_buf *>(png_get_error_ptr(png)), 1);
}

static void warning_handler(png_structp png, png_const_charp msg)
{
  (void) png;
  // FIXME make this display in the GUI or something
  std::fprintf(stderr, "PNG WARNING: %s\n", msg);
}

static bool skip_png_image(png_structp png, png_infop info);

bool save_png_rgb8(const std::string &filename, const unsigned char *data, coord_t width, coord_t height, const std::string &comment, int dpi)
{
  jmp_buf jmpbuf;
  FILE *file = std::fopen(filename.c_str(), "wb");
  if (! file)
    return false;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, &jmpbuf, error_handler, warning_handler);
  if (! png)
  {
    std::fclose(file);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (! info)
  {
    png_destroy_write_struct(&png, 0);
    std::fclose(file);
    return false;
  }
  if (setjmp(jmpbuf))
  {
    png_destroy_write_struct(&png, &info);
    fclose(file);
    return false;
  }
  png_init_io(png, file);
  png_set_compression_level(png, Z_BEST_COMPRESSION);
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_ADAM7, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_time mtime;
  png_convert_from_time_t(&mtime, time(0));
  png_set_tIME(png, info, &mtime);
  int dpm = round(dpi * 1000.0 / 25.4);
  png_set_pHYs(png, info, dpm, dpm, PNG_RESOLUTION_METER);
  png_text text;
  text.compression = PNG_TEXT_COMPRESSION_NONE;
  const std::string &key = "Comment";
  text.key = const_cast<char *>(key.c_str());
  text.text = const_cast<char *>(comment.c_str());
  png_set_text(png, info, &text, 1);
  png_write_info(png, info);
  png_bytepp row = new png_bytep[height];
  for (int y = 0; y < height; ++y)
    row[y] = (png_bytep)(data + width * 3 * y);
  png_write_image(png, row);
  png_write_end(png, 0);
  delete [] row;
  std::fclose(file);
  return true;
}

std::string read_png_comment(const std::string &filename)
{
  jmp_buf jmpbuf;
  FILE *file = std::fopen(filename.c_str(), "rb");
  if (! file)
    return "";
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, &jmpbuf, error_handler, warning_handler);
  if (! png)
  {
    std::fclose(file);
    return "";
  }
  png_infop info = png_create_info_struct(png);
  if (! info)
  {
    png_destroy_read_struct(&png, 0, 0);
    std::fclose(file);
    return "";
  }
  png_infop enfo = png_create_info_struct(png);
  if (! enfo)
  {
    png_destroy_read_struct(&png, &info, 0);
    std::fclose(file);
    return "";
  }
  if (setjmp(jmpbuf))
  {
    png_destroy_read_struct(&png, &info, 0);
    std::fclose(file);
    return "";
  }
  png_init_io(png, file);
  png_read_info(png, info);
  png_textp text;
  int count = 0;
  std::string comment = "";
  if (png_get_text(png, info, &text, &count) > 0)
    for (int t = 0; t < count; t++)
      // we save as capitalized, but processing with ImageMagick downcases
      if (0 == strcasecmp("Comment", text[t].key))
        comment = text[t].text; // copy
  if (comment == "")
  {
    if (skip_png_image(png, info))
    {
      png_read_end(png, enfo);
      png_textp etext;
      int ecount = 0;
      if (png_get_text(png, enfo, &etext, &ecount) > 0)
        for (int t = 0; t < ecount; t++)
          // we save as capitalized, but processing with ImageMagick downcases
          if (0 == strcasecmp("Comment", etext[t].key))
            comment = etext[t].text; // copy
    }
  }
  png_destroy_read_struct(&png, &info, &enfo);
  std::fclose(file);
  return comment;
}

static bool skip_png_image(png_structp png, png_infop info)
{
  // this doesn't really skip, it decodes the image
  // hack: use one single row of memory for each row pointer
  // reduces memory usage to O(W + H) from O(W * H)
  bool ok = false;
  png_uint_32 width, height = 0;
  int bit_depth, color_type;
  if (png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, 0, 0, 0))
  {
    png_read_update_info(png, info);
    png_uint_32 bytes = png_get_rowbytes(png, info);
    png_bytep row;
    if ((row = (png_bytep) std::malloc(bytes)))
    {
      png_bytepp rows;
      if ((rows = (png_bytepp) std::malloc(height * sizeof(png_bytep))))
      {
        for (png_uint_32 i = 0; i < height; ++i) rows[i] = row;
        png_read_image(png, rows);
        ok = true;
        std::free(rows);
      }
      std::free(row);
    }
  }
  return ok;
}
