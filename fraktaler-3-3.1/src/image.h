// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <mutex>

#include "types.h"

#ifdef RGB
#undef RGB
#endif

constexpr channel_bit_t Channel_R   = 0;
constexpr channel_bit_t Channel_G   = 1;
constexpr channel_bit_t Channel_B   = 2;
constexpr channel_bit_t Channel_N0  = 3;
constexpr channel_bit_t Channel_N1  = 4;
constexpr channel_bit_t Channel_NF  = 5;
constexpr channel_bit_t Channel_T   = 6;
constexpr channel_bit_t Channel_DEX = 7;
constexpr channel_bit_t Channel_DEY = 8;
constexpr channel_bit_t Channel_BLA = 9;
constexpr channel_bit_t Channel_PTB = 10;

constexpr channel_bit_t Channel_Count = 11;

constexpr channel_mask_t Channels_RGB = (1 << Channel_R) | (1 << Channel_G) | (1 << Channel_B);
constexpr channel_mask_t Channels_all = (1 << Channel_Count) - 1;

constexpr count_t Nbias_default = 1024;

struct tile;

struct image
{
  coord_t width;
  coord_t height;
  std::mutex mutex;
  image(coord_t width, coord_t height)
  : width(width), height(height)
  {
  }
  virtual ~image()
  {
  }
  virtual void clear() = 0;
  virtual void blit(coord_t tx, coord_t ty, const tile *t) = 0;
};
