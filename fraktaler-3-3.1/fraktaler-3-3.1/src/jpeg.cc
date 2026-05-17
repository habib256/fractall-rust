// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

// adapted from the implementation in KF2+

/*
Kalles Fraktaler 2
Copyright (C) 2013-2017 Karl Runmo
Copyright (C) 2017-2018 Claude Heiland-Allen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <string.h>
#include <string>

#include <jpeglib.h>

#include "jpeg.h"

bool save_jpeg_8(const std::string &filename, const unsigned char *data, coord_t width, coord_t height, const std::string &comment, int quality, J_COLOR_SPACE colourspace, int dpi);

bool save_jpeg_rgb8(const std::string &filename, const unsigned char *data, coord_t width, coord_t height, const std::string &comment, int quality, int dpi)
{
  return save_jpeg_8(filename, data, width, height, comment, quality, JCS_RGB, dpi);
}

bool save_jpeg_yuv8(const std::string &filename, const unsigned char *data, coord_t width, coord_t height, const std::string &comment, int quality, int dpi)
{
  return save_jpeg_8(filename, data, width, height, comment, quality, JCS_YCbCr, dpi);
}

bool save_jpeg_8(const std::string &filename, const unsigned char *data, coord_t width, coord_t height, const std::string &comment, int quality, J_COLOR_SPACE colourspace, int dpi)
{
  FILE * outfile = fopen(filename.c_str(), "wb");
  if (! outfile)
  {
    return false;;
  }
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_CreateCompress(&cinfo, JPEG_LIB_VERSION, sizeof(jpeg_compress_struct));
  jpeg_stdio_dest(&cinfo, outfile);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.density_unit = 1; // use dots per inch
  cinfo.X_density = dpi;
  cinfo.Y_density = dpi;
  cinfo.input_components = 3;
  cinfo.in_color_space = colourspace;
  jpeg_set_defaults(&cinfo);
  // 444 chroma subsampling
  cinfo.comp_info[0].h_samp_factor = 1;
  cinfo.comp_info[0].v_samp_factor = 1;
  cinfo.comp_info[1].h_samp_factor = 1;
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;
  cinfo.comp_info[2].v_samp_factor = 1;
  jpeg_set_quality(&cinfo, quality, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  size_t length = comment.length();
  const char *comment_str = comment.c_str();
  do
  {
    size_t wlength = length;
    if (wlength > 65533)
      wlength = 65533;
    if (wlength > 0)
      jpeg_write_marker(&cinfo, JPEG_COM, (const unsigned char *) comment_str, wlength);
    comment_str += wlength;
    length -= wlength;
  } while (length > 0);
  JSAMPROW *rows = new JSAMPROW[height];
  for (int y = 0; y < height; ++y)
  {
    rows[y] = const_cast<unsigned char *>(&data[y * width * 3]);
  }
  jpeg_write_scanlines(&cinfo, rows, height);
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  fclose(outfile);
  delete[] rows;
  return true;
}

std::string read_jpeg_comment(const std::string &filename)
{
  std::string comment = "";
  FILE *file = fopen(filename.c_str(), "rb");
  if (! file)
  {
    return "";
  }
  int c = fgetc(file);
  if (c != 0xFF)
  {
    fclose(file);
    return "";
  }
  c = fgetc(file);
  if (c != 0xD8) // SOI
  {
    fclose(file);
    return "";
  }
  do {
    do {
      c = fgetc(file);
    } while (0 <= c && c <= 0xFF && c != 0xFF);
    if (! (0 <= c && c <= 0xFF))
      break;
    // c is 0xFF
    do {
      c = fgetc(file);
    } while (0 <= c && c <= 0xFF && c == 0xFF);
    if (! (0 <= c && c <= 0xFF))
      break;
    // c is now the marker byte
    if (c == 0xFE) // COM
    {
      // read 2 byte length
      int l1 = fgetc(file);
      if (! (0 <= l1 && l1 <= 0xFF))
        break;
      int l2 = fgetc(file);
      if (! (0 <= l2 && l2 <= 0xFF))
        break;
      int length = 0;
      length = ((l1 << 8) | l2) - 2; // length includes itself
      if (length > 0)
      {
        // read comment
        char *buffer = (char *) malloc(length + 1);
        if (! buffer)
        {
          fclose(file);
          return "";
        }
        buffer[length] = 0;
        if (1 != fread(buffer, length, 1, file))
        {
          free(buffer);
          fclose(file);
          return "";
        }
        comment += buffer;
        free(buffer);
      }
      else if (length < 0)
      {
        fclose(file);
        return "";
      }
    }
    else if (c == 0xD9 || c == 0xDA) // EOI SOS
    {
      // end of file, start of compressed data, or so
      break;
    }
    else
    {
      // read 2 byte length
      int l1 = fgetc(file);
      if (! (0 <= l1 && l1 <= 0xFF))
        break;
      int l2 = fgetc(file);
      if (! (0 <= l2 && l2 <= 0xFF))
        break;
      int length = 0;
      length = ((l1 << 8) | l2) - 2;
      if (length >= 0)
      {
        if (0 != fseek(file, length, SEEK_CUR))
        {
          fclose(file);
          return "";
        }
      }
      else
      {
        fclose(file);
        return "";
      }
    }
  } while(true);
  fclose(file);
  return comment;
}
