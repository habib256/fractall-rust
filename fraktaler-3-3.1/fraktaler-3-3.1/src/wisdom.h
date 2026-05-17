// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "types.h"

struct whardware
{
  std::string name;
  int platform;
  int device;
  bool enabled;
};

struct wdevice
{
  int platform;
  int device;
  bool enabled;
  double speed;
};

struct wtype
{
  int mantissa;
  int exponent;
  int bytes;
  std::vector<wdevice> device;
};

struct wisdom
{
  std::map<std::string, std::vector<whardware>> hardware;
  std::map<std::string, wtype> type;
};

wisdom wisdom_load(const std::string &filename, bool &success);
bool wisdom_save(const wisdom &w, const std::string &filename);

wisdom wisdom_enumerate(bool use_opencl);
void wisdom_default(wisdom &w);

struct wlookup
{
  number_type nt;
  int mantissa;
  int exponent;
  int bytes;
  double speed;
  std::vector<wdevice> device;
};

wlookup wisdom_lookup(const wisdom &w, const std::set<number_type> &available, count_t pixel_spacing_exponent, count_t pixel_spacing_precision);

wisdom wisdom_benchmark(const wisdom &w, volatile progress_t *progress, volatile bool *running);

extern wisdom wdom;
