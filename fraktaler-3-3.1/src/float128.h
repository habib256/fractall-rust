// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "types.h"

#ifdef HAVE_FLOAT128

typedef _Float128 float128;

extern "C"
{
  float128 ldexpq(float128 x, int e) noexcept;
  float128 frexpq(float128 x, int *e) noexcept;
  float128 hypotq(float128 x, float128 y) noexcept;
  float128 sqrtq(float128 x) noexcept;
  float128 expq(float128 x) noexcept;
  float128 logq(float128 x) noexcept;
  float128 sinq(float128 x) noexcept;
  float128 cosq(float128 x) noexcept;
};

inline float128 ldexp(float128 x, int e) noexcept
{
  return ldexpq(x, e);
}

inline float128 frexp(float128 x, int *e) noexcept
{
  return frexpq(x, e);
}

inline float128 sqrt(float128 x) noexcept
{
  return sqrtq(x);
}

inline float128 exp(float128 x) noexcept
{
  return expq(x);
}

inline float128 sin(float128 x) noexcept
{
  return sinq(x);
}

inline float128 cos(float128 x) noexcept
{
  return cosq(x);
}

inline float128 log(float128 x) noexcept
{
  return logq(x);
}

inline float128 hypot(float128 x, float128 y) noexcept
{
  return hypotq(x, y);
}

inline CONSTEXPR bool isnan(float128 x) noexcept
{
  return ! (x == x);
}

inline CONSTEXPR bool isinf(float128 x) noexcept
{
  return 1 / x == 0;
}

inline CONSTEXPR bool signbit(float128 x) noexcept
{
  if (x == 0)
  {
    // handle +/- 0 correctly
    return 1 / x < 0;
  }
  else
  {
    return x < 0;
  }
}

inline CONSTEXPR float128 abs(float128 x) noexcept
{
  if (signbit(x))
  {
    return -x;
  }
  else
  {
    return x;
  }
}

template <> inline float128 convert<float128>(const mpreal &x) noexcept
{
  long e = 0;
  float128 v = mpfr_get_ld_2exp(&e, x.mpfr_srcptr(), MPFR_RNDN); // FIXME
  return ldexp(v, e);
}

inline constexpr float128 sqr(const float128 a) noexcept
{
  return a * a;
}

#endif
