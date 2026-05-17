// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "types.h"

struct tile
{
  coord_t width;
  coord_t height;
  float *RGB;
  uint32_t *N0;
  uint32_t *N1;
  float *NF;
  float *T;
  float *DEX;
  float *DEY;
  uint32_t *BLA;
  uint32_t *PTB;
};

tile *tile_copy(const tile *t);
void tile_delete(tile *t);

struct hooks
{
  hooks()
  {
  }
  virtual ~hooks()
  {
  }
  // before/after image render
  virtual void start()
  {
  }
  virtual void stop()
  {
  }
  // before/after reference calculation
  virtual bool pre_reference(bool needs_recalc)
  {
    (void) needs_recalc;
    return false;
  }
  virtual void post_reference(bool recalculated)
  {
    (void) recalculated;
  }
  // before/after bla calculation
  virtual bool pre_bla(bool needs_recalc)
  {
    (void) needs_recalc;
    return false;
  }
  virtual void post_bla(bool recalculated)
  {
    (void) recalculated;
  }
  // before/after all tiles / all devices
  virtual void pre_render()
  {
  }
  virtual void post_render()
  {
  }
  // before/after each device
  virtual void pre_device(int platform, int device)
  {
    (void) platform;
    (void) device;
  }
  virtual void post_device(int platform, int device)
  {
    (void) platform;
    (void) device;
  }
  // before/after kernel compile
  virtual void pre_compile(int platform, int device)
  {
    (void) platform;
    (void) device;
  }
  virtual void post_compile(int platform, int device)
  {
    (void) platform;
    (void) device;
  }
  // before/after data upload
  virtual void pre_upload(int platform, int device)
  {
    (void) platform;
    (void) device;
  }
  virtual void post_upload(int platform, int device)
  {
    (void) platform;
    (void) device;
  }
  // before/after tile calculation
  virtual void pre_tile(int platform, int device, int x, int y, int subframe)
  {
    (void) platform;
    (void) device;
    (void) x;
    (void) y;
    (void) subframe;
  }
  virtual void post_tile(int platform, int device, int x, int y, int subframe)
  {
    (void) platform;
    (void) device;
    (void) x;
    (void) y;
    (void) subframe;
  }
  // before/after data download
  virtual void pre_download(int platform, int device, int x, int y, int subframe)
  {
    (void) platform;
    (void) device;
    (void) x;
    (void) y;
    (void) subframe;
  }
  virtual void post_download(int platform, int device, int x, int y, int subframe)
  {
    (void) platform;
    (void) device;
    (void) x;
    (void) y;
    (void) subframe;
  }
  // tile data callback
  virtual void tile(int platform, int device, int x, int y, int subframe, const tile *data)
  {
    (void) platform;
    (void) device;
    (void) x;
    (void) y;
    (void) subframe;
    (void) data;
  }
};

void get_required_precision(const param &par, count_t &pixel_spacing_exp, count_t &pixel_precision_exp);
bool reference_can_be_reused(const wlookup &l, const param &par);
void render(const wlookup &l, const param &par, hooks *h, bool first, progress_t *progress, volatile bool *running);
