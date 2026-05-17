// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <string>

#include "image.h"

struct image_raw : public image
{
  channel_mask_t channels;
  float *R;
  float *G;
  float *B;
  uint32_t *N0;
  uint32_t *N1;
  float *NF;
  float *T;
  float *DEX;
  float *DEY;
  uint32_t *BLA;
  uint32_t *PTB;
  image_raw(coord_t width, coord_t height, channel_mask_t channels);
  image_raw(image_raw &source, bool vflip = false);
  virtual ~image_raw();
  virtual void clear() override;
  virtual void blit(coord_t tx, coord_t ty, const struct tile *t) override;
  virtual bool save_exr(const std::string &filename, channel_mask_t save_channels, count_t maxiters, int threads, const std::string &metadata, int dpi);
};
