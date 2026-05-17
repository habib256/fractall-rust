// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <string>

#include "image.h"

struct image_rgb : public image
{
  float *RGBA;
  std::mutex mutex;
  image_rgb(coord_t width, coord_t height);
  image_rgb(image_rgb &source, bool vflip = false);
  virtual ~image_rgb();
  virtual void clear() override;
  virtual void blit(coord_t tx, coord_t ty, const struct tile *t) override;
  virtual bool save_exr(const std::string &filename, int threads, const std::string &metadata, int dpi);
};

struct image_rgb8
{
  coord_t width, height;
  unsigned char *RGB;
  image_rgb8(coord_t width, coord_t height);
  image_rgb8(image_rgb &source, bool vflip = false, bool dither = true);
  virtual ~image_rgb8();
  virtual bool save_png(const std::string &filename, const std::string &metadata, int dpi);
  virtual bool save_jpeg(const std::string &filename, const std::string &metadata, int quality, int dpi);
};

struct image_yuv8
{
  coord_t width, height;
  unsigned char *YUV;
  image_yuv8(coord_t width, coord_t height);
  image_yuv8(image_rgb &source, bool vflip = false, bool dither = true);
  virtual ~image_yuv8();
  virtual bool save_jpeg(const std::string &filename, const std::string &metadata, int quality, int dpi);
};
