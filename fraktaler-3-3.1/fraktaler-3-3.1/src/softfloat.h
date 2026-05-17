// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cmath>
#include <ctgmath>

#include "float128.h"
#include "floatexp.h"

inline CONSTEXPR uint32_t clz(uint32_t x) noexcept
{
  if (x == 0)
  {
    return 32;
  }
  else
  {
    return __builtin_clz(x);
  }
}

inline CONSTEXPR uint32_t clz(uint64_t x) noexcept
{
  if (x == 0)
  {
    return 64;
  }
  else
  {
    return __builtin_clzll(x);
  }
}

inline CONSTEXPR uint32_t convert_uint_sat(int64_t x) noexcept
{
  if (x < 0) return 0;
  if (x > int64_t(0xFFFFFFFFU)) return 0xFFFFFFFFU;
  return x;
}

inline CONSTEXPR int32_t convert_int_sat(int64_t x) noexcept
{
  if (x < int64_t(int32_t(0x80000000))) return 0x80000000;
  if (x > int64_t(int32_t(0x7FFFFFFF))) return 0x7FFFFFFF;
  return x;
}

inline CONSTEXPR uint32_t convert_uint_rtz(const float x) noexcept
{
  return x;
}

inline CONSTEXPR uint32_t convert_uint_rtz(const double x) noexcept
{
  return x;
}

inline CONSTEXPR uint32_t convert_uint_rtz(const long double x) noexcept
{
  return x;
}

#ifdef HAVE_FLOAT128
inline CONSTEXPR uint32_t convert_uint_rtz(const float128 x) noexcept
{
  return x;
}
#endif

struct softfloat;
inline CONSTEXPR bool sign_bit(const softfloat f) noexcept;
inline CONSTEXPR uint32_t biased_exponent(const softfloat f) noexcept;
inline CONSTEXPR bool is_zero(const softfloat f) noexcept;
inline CONSTEXPR bool is_denormal(const softfloat f) noexcept;
inline CONSTEXPR bool isinf(const softfloat f) noexcept;
inline CONSTEXPR bool isnan(const softfloat f) noexcept;
inline CONSTEXPR bool operator<(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR bool operator>(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR bool operator>(const softfloat a, const int b) noexcept;
inline CONSTEXPR bool operator>=(const softfloat a, const int b) noexcept;
inline CONSTEXPR softfloat ldexp(const softfloat a, int e) noexcept;
inline CONSTEXPR softfloat zero() noexcept;
inline CONSTEXPR softfloat one() noexcept;
inline CONSTEXPR softfloat abs(const softfloat a) noexcept;
inline CONSTEXPR softfloat operator-(const softfloat a) noexcept;
inline CONSTEXPR int sgn(const softfloat a) noexcept;
inline CONSTEXPR softfloat sqr(const softfloat a) noexcept;
inline CONSTEXPR softfloat operator*(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator*(const softfloat a, const int b) noexcept;
inline CONSTEXPR softfloat operator*(const int a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator*(const softfloat a, const double b) noexcept;
inline CONSTEXPR softfloat operator*(const double a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator/(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator/(const softfloat a, const int b) noexcept;
inline CONSTEXPR softfloat operator/(const int a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator/(const softfloat a, const long int b) noexcept;
inline CONSTEXPR softfloat operator/(const long int a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator/(const softfloat a, const long long int b) noexcept;
inline CONSTEXPR softfloat operator/(const long long int a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator<<(const softfloat a, const int b) noexcept;
inline CONSTEXPR softfloat add_a_gt_b_gt_0(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR softfloat add_a_gt_0_gt_b(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator+(const softfloat a) noexcept;
inline CONSTEXPR softfloat operator+(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator+(const softfloat a, const int b) noexcept;
inline CONSTEXPR softfloat operator+(const int a, const softfloat b) noexcept;
inline CONSTEXPR softfloat operator-(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR int cmp(const softfloat a, const softfloat b) noexcept;
inline CONSTEXPR softfloat diffabs(const softfloat c, const softfloat d) noexcept;

struct softfloat
{
  uint32_t se;
  uint32_t m;

  static CONSTEXPR_STATIC uint32_t EXPONENT_BIAS = (1U << 30U) - 1U;
  static CONSTEXPR_STATIC uint32_t MANTISSA_BITS = 32U;

  // POD
  inline ~softfloat() = default;
  inline softfloat() = default;
  inline CONSTEXPR softfloat(const softfloat &fe) = default;
  inline CONSTEXPR softfloat(softfloat &&fe) = default;
  inline CONSTEXPR softfloat &operator=(const softfloat &fe) = default;

  inline CONSTEXPR softfloat(const uint32_t se, const uint32_t m)
  : se(se)
  , m(m)
  {
  }

  inline CONSTEXPR softfloat(const float x) noexcept
  {
    if (std::isnan(x))
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0xFFFFFFFFU;
    }
    else if (std::isinf(x))
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0U;
    }
    else if (x == 0)
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0U;
      m = 0U;
    }
    else
    {
      int e = 0;
      float y = std::frexp(std::abs(x), &e);
      float z = std::ldexp(y, MANTISSA_BITS);
      uint32_t biased_e = convert_uint_sat(e + EXPONENT_BIAS);
      se = ((uint32_t)(!!std::signbit(x)) << 31) | biased_e;
      m = convert_uint_rtz(z);
      assert(0 < biased_e);
      assert(biased_e < 0x7FFFFFFFU);
      assert((m >> (MANTISSA_BITS - 1)) == 1U);
    }
  }

  inline CONSTEXPR softfloat(const double x) noexcept
  {
    if (std::isnan(x))
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0xFFFFFFFFU;
    }
    else if (std::isinf(x))
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0U;
    }
    else if (x == 0)
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0U;
      m = 0U;
    }
    else
    {
      int e;
      double y = std::frexp(std::abs(x), &e);
      double z = std::ldexp(y, MANTISSA_BITS);
      uint32_t biased_e = convert_uint_sat(e + EXPONENT_BIAS);
      se = ((uint32_t)(!!std::signbit(x)) << 31) | biased_e;
      m = convert_uint_rtz(z);
      assert(0 < biased_e);
      assert(biased_e < 0x7FFFFFFFU);
      assert((m >> (MANTISSA_BITS - 1)) == 1U);
    }
  }

  inline CONSTEXPR softfloat(const long double x) noexcept
  {
    if (std::isnan(x))
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0xFFFFFFFFU;
    }
    else if (std::isinf(x))
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0U;
    }
    else if (x == 0)
    {
      se = ((uint32_t)(!!std::signbit(x)) << 31) | 0U;
      m = 0U;
    }
    else
    {
      int e;
      long double y = std::frexp(std::abs(x), &e);
      long double z = std::ldexp(y, MANTISSA_BITS);
      uint32_t biased_e = convert_uint_sat(e + EXPONENT_BIAS);
      se = ((uint32_t)(!!std::signbit(x)) << 31) | biased_e;
      m = convert_uint_rtz(z);
      assert(0 < biased_e);
      assert(biased_e < 0x7FFFFFFFU);
      assert((m >> (MANTISSA_BITS - 1)) == 1U);
    }
  }

#ifdef HAVE_FLOAT128
  inline CONSTEXPR softfloat(const float128 x) noexcept
  {
    if (isnan(x))
    {
      se = ((uint32_t)(!!signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0xFFFFFFFFU;
    }
    else if (isinf(x))
    {
      se = ((uint32_t)(!!signbit(x)) << 31) | 0x7FFFFFFFU;
      m = 0U;
    }
    else if (x == 0)
    {
      se = ((uint32_t)(!!signbit(x)) << 31) | 0U;
      m = 0U;
    }
    else
    {
      int e;
      float128 y = frexp(abs(x), &e);
      float128 z = ldexp(y, MANTISSA_BITS);
      uint32_t biased_e = convert_uint_sat(e + EXPONENT_BIAS);
      se = ((uint32_t)(!!signbit(x)) << 31) | biased_e;
      m = convert_uint_rtz(z);
      assert(0 < biased_e);
      assert(biased_e < 0x7FFFFFFFU);
      assert((m >> (MANTISSA_BITS - 1)) == 1U);
    }
  }
#endif

  template<typename M, typename E>
  inline CONSTEXPR softfloat(const floatexp<M, E> x) noexcept
  : softfloat(ldexp(softfloat(x.val), convert_int_sat(x.exp)))
  {
  }

  inline CONSTEXPR softfloat(const int x) noexcept
  : softfloat(double(x))
  {
  }

  inline CONSTEXPR softfloat(const long int x) noexcept
  : softfloat(double(x))
  {
  }

  inline CONSTEXPR softfloat(const long long int x) noexcept
  : softfloat(double(x))
  {
  }

  inline softfloat(const mpfr::mpreal &x) noexcept
  : softfloat(floatexp<float, int>(x)) // FIXME
  {
  }


  explicit inline CONSTEXPR operator float() const noexcept
  {
    const softfloat &f = *this;
    if (is_zero(f) || is_denormal(f))
    {
      if (sign_bit(f)) return -0.0f; else return 0.0f;
    }
    else if (isinf(f))
    {
      if (sign_bit(f)) return -1.0f/0.0f; else return 1.0f/0.0f;
    }
    else if (isnan(f))
    {
      if (sign_bit(f)) return -(0.0f/0.0f); else return 0.0f/0.0f;
    }
    else
    {
      float x = f.m;
      int e
        = convert_int_sat((int64_t)(biased_exponent(f))
        - int64_t(EXPONENT_BIAS + MANTISSA_BITS));
      if (sign_bit(f)) return -std::ldexp(x, e); else return std::ldexp(x, e);
    }
  }
  
  explicit inline CONSTEXPR operator double() const noexcept
  {
    const softfloat &f = *this;
    if (is_zero(f) || is_denormal(f))
    {
      if (sign_bit(f)) return -0.0; else return 0.0;
    }
    else if (isinf(f))
    {
      if (sign_bit(f)) return -1.0/0.0; else return 1.0/0.0;
    }
    else if (isnan(f))
    {
      if (sign_bit(f)) return -(0.0/0.0); else return 0.0/0.0;
    }
    else
    {
      double x = f.m;
      int e
        = convert_int_sat((int64_t)(biased_exponent(f))
        - (EXPONENT_BIAS + MANTISSA_BITS));
      if (sign_bit(f)) return -std::ldexp(x, e); else return std::ldexp(x, e);
    }
  }

  explicit inline CONSTEXPR operator long double() const noexcept
  {
    const softfloat &f = *this;
    if (is_zero(f) || is_denormal(f))
    {
      if (sign_bit(f)) return -0.0; else return 0.0;
    }
    else if (isinf(f))
    {
      if (sign_bit(f)) return -1.0/0.0; else return 1.0/0.0;
    }
    else if (isnan(f))
    {
      if (sign_bit(f)) return -(0.0/0.0); else return 0.0/0.0;
    }
    else
    {
      long double x = f.m;
      int e
        = convert_int_sat((int64_t)(biased_exponent(f))
        - (EXPONENT_BIAS + MANTISSA_BITS));
      if (sign_bit(f)) return -std::ldexp(x, e); else return std::ldexp(x, e);
    }
  }

#ifdef HAVE_FLOAT128
  explicit inline CONSTEXPR operator float128() const noexcept
  {
    const softfloat &f = *this;
    if (is_zero(f) || is_denormal(f))
    {
      if (sign_bit(f)) return -0.0; else return 0.0;
    }
    else if (isinf(f))
    {
      if (sign_bit(f)) return -1.0/0.0; else return 1.0/0.0;
    }
    else if (isnan(f))
    {
      if (sign_bit(f)) return -(0.0/0.0); else return 0.0/0.0;
    }
    else
    {
      float128 x = f.m;
      int e
        = convert_int_sat((int64_t)(biased_exponent(f))
        - (EXPONENT_BIAS + MANTISSA_BITS));
      if (sign_bit(f)) return -ldexp(x, e); else return ldexp(x, e);
    }
  }
#endif

  template<typename M, typename E>
  explicit inline CONSTEXPR operator floatexp<M, E>() const noexcept
  {
    const softfloat &f = *this;
    if (is_zero(f) || is_denormal(f))
    {
      if (sign_bit(f)) return -0.0; else return 0.0;
    }
    else if (isinf(f))
    {
      if (sign_bit(f)) return -1.0/0.0; else return 1.0/0.0;
    }
    else if (isnan(f))
    {
      if (sign_bit(f)) return -(0.0/0.0); else return 0.0/0.0;
    }
    else
    {
      floatexp<M, E> x = int64_t(f.m);
      int e
        = convert_int_sat((int64_t)(biased_exponent(f))
        - (EXPONENT_BIAS + MANTISSA_BITS));
      if (sign_bit(f)) return -ldexp(x, e); else return ldexp(x, e);
    }
  }
};

inline CONSTEXPR bool sign_bit(const softfloat f) noexcept
{
  return !!(f.se & 0x80000000U);
}

inline CONSTEXPR uint32_t biased_exponent(const softfloat f) noexcept
{
  return f.se & 0x7FFFFFFFU;
}

inline CONSTEXPR bool is_zero(const softfloat f) noexcept
{
  return
    biased_exponent(f) == 0 &&
    f.m == 0;
}

inline CONSTEXPR bool is_denormal(const softfloat f) noexcept
{
  return
    biased_exponent(f) == 0 &&
    f.m != 0;
}

inline CONSTEXPR bool isinf(const softfloat f) noexcept
{
  return
    biased_exponent(f) == 0x7FFFFFFFU &&
    f.m == 0;
}

inline CONSTEXPR bool isnan(const softfloat f) noexcept
{
  return
    biased_exponent(f) == 0x7FFFFFFFU &&
    f.m != 0;
}

inline CONSTEXPR bool operator<(const softfloat a, const softfloat b) noexcept
{
  if (isnan(a) || isnan(b))
  {
    return false;
  }
  else if (sign_bit(a) && ! sign_bit(b))
  {
    return true;
  }
  else if (! sign_bit(a) && sign_bit(b))
  {
    return false;
  }
  else if (biased_exponent(a) > biased_exponent(b))
  {
    return sign_bit(a);
  }
  else if (biased_exponent(a) < biased_exponent(b))
  {
    return ! sign_bit(a);
  }
  else if (a.m > b.m)
  {
    return sign_bit(a);
  }
  else if (a.m < b.m)
  {
    return ! sign_bit(a);
  }
  else
  {
    // equal
    return false;
  }
}

inline CONSTEXPR bool operator<=(const softfloat a, const softfloat b) noexcept
{
  if (isnan(a) || isnan(b))
  {
    return false;
  }
  else if (sign_bit(a) && ! sign_bit(b))
  {
    return true;
  }
  else if (! sign_bit(a) && sign_bit(b))
  {
    return false;
  }
  else if (biased_exponent(a) > biased_exponent(b))
  {
    return sign_bit(a);
  }
  else if (biased_exponent(a) < biased_exponent(b))
  {
    return ! sign_bit(a);
  }
  else if (a.m > b.m)
  {
    return sign_bit(a);
  }
  else if (a.m < b.m)
  {
    return ! sign_bit(a);
  }
  else
  {
    // equal
    return true;
  }
}

inline CONSTEXPR bool operator>(const softfloat a, const softfloat b) noexcept
{
  return b < a;
}

inline CONSTEXPR bool operator>=(const softfloat a, const softfloat b) noexcept
{
  return b <= a;
}

inline CONSTEXPR bool operator>(const softfloat a, const int b) noexcept
{
  return a > softfloat(b);
}

inline CONSTEXPR bool operator>=(const softfloat a, const int b) noexcept
{
  return a >= softfloat(b);
}

inline CONSTEXPR softfloat ldexp(const softfloat a, int e) noexcept
{
  if (is_zero(a) || isinf(a) || isnan(a))
  {
    return a;
  }
  else if (e >= (int32_t)(0x7FFFFFFFU - biased_exponent(a)))
  {
    // overflow to +/-infinity
    softfloat o = { (a.se & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  else if ((int32_t)(biased_exponent(a)) + e <= 0)
  {
    // underfloat to 0
    softfloat o = { (a.se & 0x80000000U) | 0U, 0U };
    return o;
  }
  else
  {
    softfloat o = { (a.se & 0x80000000U) | (biased_exponent(a) + e), a.m };
    return o;
  }
}

inline CONSTEXPR softfloat zero() noexcept
{
  softfloat o = { 0, 0 };
  return o;
}

inline CONSTEXPR softfloat one() noexcept
{
  return softfloat(1.0f);
}

inline CONSTEXPR softfloat abs(const softfloat a) noexcept
{
  softfloat o = { a.se & 0x7FFFFFFFU, a.m };
  return o;
}

inline CONSTEXPR softfloat operator-(const softfloat a) noexcept
{
  softfloat o = { a.se ^ 0x80000000U, a.m };
  return o;
}

inline CONSTEXPR int sgn(const softfloat a) noexcept
{
  return sign_bit(a) ? -1 : 1; // FIXME
}

inline CONSTEXPR softfloat sqr(const softfloat a) noexcept
{
  if (biased_exponent(a) >= ((0x7FFFFFFFU >> 1) + (softfloat::EXPONENT_BIAS >> 1)))
  {
    // overflow to +infinity
    softfloat o = { 0x7FFFFFFFU, isnan(a) ? 0xFFFFFFFFU : 0U };
    return o;
  }
  else if (biased_exponent(a) <= (softfloat::EXPONENT_BIAS >> 1) + 1)
  {
    // underflow to +0
    // FIXME handle denormals
    softfloat o = { 0U, 0U };
    return o;
  }
  else
  {
    uint64_t m = a.m;
    uint32_t mantissa = (m * m) >> softfloat::MANTISSA_BITS;
    uint32_t biased_e = ((a.se & 0x7FFFFFFFU) << 1) - softfloat::EXPONENT_BIAS;
    if ((mantissa & 0x80000000U) == 0)
    {
      mantissa <<= 1;
      biased_e -= 1;
    }
    assert(0 < biased_e);
    assert(biased_e < 0x7FFFFFFFU);
    assert((mantissa >> (softfloat::MANTISSA_BITS - 1)) == 1U);
    softfloat o = { biased_e, mantissa };
    return o;
  }
}

inline CONSTEXPR softfloat operator*(const softfloat a, const softfloat b) noexcept
{
  if ( isnan(a) ||
       isnan(b) ||
       (isinf(a) && is_zero(b)) ||
       (is_zero(a) && isinf(b)) ||
       (isinf(a) && isinf(b) && sign_bit(a) != sign_bit(b))
     )
  {
    // nan
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (is_zero(a) || is_zero(b))
  {
    // zero
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0U, 0U };
    return o;
  }
  else if (isinf(a) || isinf(b) || (biased_exponent(a) + biased_exponent(b)) >= (0x7FFFFFFFU + softfloat::EXPONENT_BIAS))
  {
    // overflow to +/-infinity
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  else if ((biased_exponent(a) + biased_exponent(b)) <= (softfloat::EXPONENT_BIAS + 1))
  {
    // underflow to +/-0
    // FIXME handle denormals
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0U, 0U };
    return o;
  }
  else
  {
    uint64_t ma = a.m;
    uint64_t mb = b.m;
    uint32_t mantissa = (ma * mb) >> softfloat::MANTISSA_BITS;
    uint32_t biased_e = ((a.se & 0x7FFFFFFFU) + (b.se & 0x7FFFFFFFU)) - softfloat::EXPONENT_BIAS;
    if ((mantissa & 0x80000000U) == 0)
    {
      mantissa <<= 1;
      biased_e -= 1;
    }
    assert(0 < biased_e);
    assert(biased_e < 0x7FFFFFFFU);
    assert((mantissa >> (softfloat::MANTISSA_BITS - 1)) == 1U);
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | biased_e, mantissa };
    return o;
  }
}

inline CONSTEXPR softfloat operator*(const softfloat a, const int b) noexcept
{
  return a * softfloat(b);
}

inline CONSTEXPR softfloat operator*(const int a, const softfloat b) noexcept
{
  return softfloat(a) * b;
}

inline CONSTEXPR softfloat operator*(const softfloat a, const double b) noexcept
{
  return a * softfloat(b);
}

inline CONSTEXPR softfloat operator*(const double a, const softfloat b) noexcept
{
  return softfloat(a) * b;
}

inline CONSTEXPR softfloat operator/(const softfloat a, const softfloat b) noexcept
{
  if ( isnan(a) ||
       isnan(b) ||
       (is_zero(a) && is_zero(b)) ||
       (isinf(a) && isinf(b))
     )
  {
    // nan
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (is_zero(b))
  {
    // inf
    softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  else
  {
    return softfloat(floatexp<float, int>(a) / floatexp<float, int>(b)); // FIXME
  }
}

inline CONSTEXPR softfloat operator/(const softfloat a, const int b) noexcept
{
  return a / softfloat(b);
}

inline CONSTEXPR softfloat operator/(const int a, const softfloat b) noexcept
{
  return softfloat(a) / b;
}

inline CONSTEXPR softfloat operator/(const softfloat a, const long int b) noexcept
{
  return a / softfloat(b);
}

inline CONSTEXPR softfloat operator/(const long int a, const softfloat b) noexcept
{
  return softfloat(a) / b;
}

inline CONSTEXPR softfloat operator/(const softfloat a, const long long int b) noexcept
{
  return a / softfloat(b);
}

inline CONSTEXPR softfloat operator/(const long long int a, const softfloat b) noexcept
{
  return softfloat(a) / b;
}

inline CONSTEXPR softfloat operator<<(const softfloat a, const int b) noexcept
{
  return ldexp(a, b);
}

inline CONSTEXPR softfloat add_a_gt_b_gt_0(const softfloat a, const softfloat b) noexcept
{
  // same sign addition, |a| > |b| or same exponent
  uint32_t ea = biased_exponent(a);
  uint32_t eb = biased_exponent(b);
  uint64_t ma = a.m;
  uint64_t mb = b.m;
  assert(ea >= eb);
  assert(sign_bit(a) == sign_bit(b));
  uint64_t mantissa = ma + (mb >> (ea - eb));
  uint32_t biased_e = ea;
  if (!! (mantissa & 0x100000000LU))
  {
    biased_e += 1;
    mantissa >>= 1;
  }
  if (biased_e >= 0x7FFFFFFFU)
  {
    // overflow to +/-infinity
    softfloat o = { (a.se & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  assert(0 < biased_e);
  assert(biased_e < 0x7FFFFFFFU);
  assert((mantissa >> (softfloat::MANTISSA_BITS - 1)) == 1U);
  assert((mantissa >> softfloat::MANTISSA_BITS) == 0U);
  softfloat o = { biased_e, static_cast<uint32_t>(mantissa) };
  if (sign_bit(a)) return -(o); else return o;
}

inline CONSTEXPR softfloat add_a_gt_0_gt_b(const softfloat a, const softfloat b) noexcept
{
  // opposite sign addition, a > 0 > b, |a| > |b|
  uint32_t ea = biased_exponent(a);
  uint32_t eb = biased_exponent(b);
  uint64_t ma = a.m;
  uint64_t mb = b.m;
  assert(ea > eb);
  assert(! sign_bit(a));
  assert(  sign_bit(b));
  // a > 0 > b, |a| > |b|
  int64_t smantissa = (ma << 1) - ((mb << 1) >> (ea - eb));
  assert(smantissa > 0);
  uint64_t mantissa = smantissa;
  int64_t biased_e = ea - 1;
  int shift = ((int)(clz(mantissa))) - softfloat::MANTISSA_BITS;
  if (shift > 0)
  {
    mantissa <<= shift;
    biased_e -= shift;
  }
  else if (shift < 0)
  {
    mantissa >>= -shift;
    biased_e += -shift;
  }
  if (biased_e >= 0x7FFFFFFFU)
  {
    // overflow to +infinity, impossible?
    softfloat o = { 0x7FFFFFFFU, 0U };
    return o;
  }
  else if (biased_e <= 0)
  {
    // underflow to +0
    softfloat o = { 0U, 0U };
    return o;
  }
  assert(0 < biased_e);
  assert(biased_e < 0x7FFFFFFFU);
  assert((mantissa >> (softfloat::MANTISSA_BITS - 1)) == 1U);
  assert((mantissa >> softfloat::MANTISSA_BITS) == 0U);
  softfloat o = { uint32_t(biased_e), (uint32_t)(mantissa) };
  return o;
}

inline CONSTEXPR softfloat operator+(const softfloat a) noexcept
{
  return a;
}

inline CONSTEXPR softfloat operator+(const softfloat a, const softfloat b) noexcept
{
  if ( isnan(a) ||
       isnan(b) ||
       (isinf(a) && isinf(b) && !!((a.se ^ b.se) & 0x80000000U))
     )
  {
    // nan
    softfloat o = { 0x7FFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (is_zero(a))
  {
    return b;
  }
  else if (is_zero(b))
  {
    return a;
  }
  else if (isinf(a))
  {
    return a;
  }
  else if (isinf(b))
  {
    return b;
  }
  else if (((a.se ^ b.se) & 0x80000000U) == 0)
  {
    // same sign addition
    uint32_t ea = biased_exponent(a);
    uint32_t eb = biased_exponent(b);
    if (ea > eb + softfloat::MANTISSA_BITS)
    {
      return a;
    }
    else if (eb > ea + softfloat::MANTISSA_BITS)
    {
      return b;
    }
    else if (ea >= eb)
    {
      return add_a_gt_b_gt_0(a, b);
    }
    else
    {
      return add_a_gt_b_gt_0(b, a);
    }
  }
  else
  {
    // opposite sign addition
    uint32_t ea = biased_exponent(a);
    uint32_t eb = biased_exponent(b);
    if (ea > eb + softfloat::MANTISSA_BITS)
    {
      return a;
    }
    else if (eb > ea + softfloat::MANTISSA_BITS)
    {
      return b;
    }
    else if (ea == eb)
    {
      uint32_t ma = a.m;
      uint32_t mb = b.m;
      if (ma > mb)
      {
        uint32_t mantissa = ma - mb;
        uint32_t shift = clz(mantissa);
        mantissa <<= shift;
        if (ea > shift)
        {
          uint32_t biased_e = ea - shift;
          assert(0 < biased_e);
          assert(biased_e < 0x7FFFFFFFU);
          assert((mantissa >> (softfloat::MANTISSA_BITS - 1)) == 1U);
          softfloat o = { biased_e, mantissa };
          if (sign_bit(a)) return -(o); else return o;
        }
        else
        {
          // FIXME handle denormals
          softfloat o = { 0U, 0U };
          return o;
        }
      }
      else if (mb > ma)
      {
        uint32_t mantissa = mb - ma;
        uint32_t shift = clz(mantissa);
        mantissa <<= shift;
        if (eb > shift)
        {
          uint32_t biased_e = eb - shift;
          assert(0 < biased_e);
          assert(biased_e < 0x7FFFFFFFU);
          assert((mantissa >> (softfloat::MANTISSA_BITS - 1)) == 1U);
          softfloat o = { biased_e, mantissa };
          if (sign_bit(b)) return -(o); else return o;
        }
        else
        {
          // FIXME handle denormals
          softfloat o = { 0U, 0U };
          return o;
        }
      }
      else
      {
        // cancels to 0
        softfloat o = { 0U, 0U };
        return o;
      }
    }
    else if (ea > eb)
    {
      // |a| > |b|
      if (sign_bit(a))
      {
        return -(add_a_gt_0_gt_b(-(a), -(b)));
      }
      else
      {
        return add_a_gt_0_gt_b(a, b);
      }
    }
    else
    {
      // |b| > |a|
      if (sign_bit(b))
      {
        return -(add_a_gt_0_gt_b(-(b), -(a)));
      }
      else
      {
        return add_a_gt_0_gt_b(b, a);
      }
    }
  }
}

inline CONSTEXPR softfloat operator+(const softfloat a, const int b) noexcept
{
  return a + softfloat(b);
}

inline CONSTEXPR softfloat operator+(const int a, const softfloat b) noexcept
{
  return softfloat(a) + b;
}

inline CONSTEXPR softfloat operator-(const softfloat a, const softfloat b) noexcept
{
  return a + (-b);
}

inline CONSTEXPR int cmp(const softfloat a, const softfloat b) noexcept
{
  return ((int)(a > b)) - ((int)(a < b));
}

inline CONSTEXPR softfloat diffabs(const softfloat c, const softfloat d) noexcept
{
  int s = cmp(c, zero());
  if (s > 0)
  {
    int t = cmp(c + d, zero());
    if (t >= 0)
    {
      return d;
    }
    else
    {
      return -(d + (c << 1));
    }
  }
  else if (s < 0)
  {
    int t = cmp(c + d, zero());
    if (t > 0)
    {
      return d + (c << 1);
    }
    else
    {
      return -d;
    }
  }
  return abs(d);
}

inline /*CONSTEXPR*/ softfloat sqrt(const softfloat x) noexcept
{
  return softfloat(sqrt(floatexp<float, int>(x)));
}

inline /*CONSTEXPR*/ softfloat hypot(const softfloat x, const softfloat y) noexcept
{
  return sqrt(sqr(x) + sqr(y));
}

inline /*CONSTEXPR*/ softfloat log(const softfloat x) noexcept
{
  return softfloat(log(floatexp<float, int>(x)));
}

inline /*CONSTEXPR*/ softfloat exp(const softfloat x) noexcept
{
  return softfloat(exp(floatexp<float, int>(x)));
}

inline /*CONSTEXPR*/ softfloat sin(const softfloat x) noexcept
{
  return softfloat(sin(floatexp<float, int>(x)));
}

inline /*CONSTEXPR*/ softfloat cos(const softfloat x) noexcept
{
  return softfloat(cos(floatexp<float, int>(x)));
}

inline CONSTEXPR softfloat& operator+=(softfloat &a, const softfloat b) noexcept
{
  return a = a + b;
}

inline CONSTEXPR bool operator==(const softfloat &a, const softfloat b) noexcept
{
  if (isnan(a) || isnan(b))
  {
    return false;
  }
  return a.se == b.se && a.m == b.m;
}
