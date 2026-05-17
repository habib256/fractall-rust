// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <thread>
#include <vector>

#include "complex.h"
#include "matrix.h"
#include "types.h"
#include "float128.h"

struct phybrid;

template <typename real>
struct blaR2
{
  mat2<real> A, B;
  real r2;
  count_t l;
};

template <typename real>
blaR2<real> merge(const blaR2<real> &y, const blaR2<real> &x, const real &c)
{
  using std::min;
  using std::max;
  using std::sqrt, ::sqrt;
  const count_t l = x.l + y.l;
  const mat2<real> A = y.A * x.A;
  const mat2<real> B = y.A * x.B + y.B;
  const real xA = sup(x.A);
  const real xB = sup(x.B);
  const real r = min(sqrt(x.r2), max(real(0), (sqrt(y.r2) - xB * c) / xA));
  const real r2 = r * r;
  blaR2<real> b = { A, B, r2, l };
  return b;
}

inline count_t ilog2(count_t n)
{
  if (n <= 0) return 0;
  count_t r = 0;
  while (n >>= 1) ++r;
  return r;
}

template<typename real>
struct blasR2
{
  count_t M;
  count_t L;
  std::vector<std::vector<blaR2<real>>> b;
  const struct blaR2<real> *lookup(const count_t m, const real z2) const noexcept;
};

template <typename real>
struct blasR2calc
{
  const std::vector<complex<real>> &Z;
  const std::vector<std::vector<opcode>> &opss;
  const std::vector<int> &degrees;
  const count_t phase;
  const real c;
  const real e;
  const int skip_levels;
  volatile progress_t *progress;
  volatile bool *running;
  std::atomic<count_t> total;
  count_t concurrency_length;
  count_t concurrency_level;
  blasR2<real> data;
  blasR2calc(const std::vector<complex<real>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, const count_t phase, const real c, const real e, int skip_levels, volatile progress_t *progress, volatile bool *running)
  : Z(Z)
  , opss(opss)
  , degrees(degrees)
  , phase(phase)
  , c(c)
  , e(e)
  , skip_levels(skip_levels)
  , progress(progress)
  , running(running)
  , total({0})
  , concurrency_length(65536)
  {
    data.M = std::max(count_t(Z.size()) - 1, count_t(0));
    data.L = ilog2(data.M - 1) + 1;
    data.b.resize(data.L + 1);
    count_t concurrency = ilog2(std::thread::hardware_concurrency() - 1) + 1; // rounding up, e.g. for 16 threads on 12 cpus
    concurrency_level = data.L + 1 - concurrency;
    count_t m = data.M;
    for (count_t level = 0; level <= data.L; ++level, m = (m + 1) >> 1)
    {
      if (level >= skip_levels)
      {
        data.b[level].resize(m);
      }
    }
    assert(m == 1);
    fill(nullptr, data.L, 0);
  }

  void fill(blaR2<real> *resultp, count_t level, count_t dst);
};
