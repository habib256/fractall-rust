// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>

#include <mpreal.h>

#include "float128.h"
#include "types.h"

template <typename T>
inline CONSTEXPR int sgn(const T x) noexcept
{
  return (x > 0) - (0 > x);
}

template <typename T>
inline CONSTEXPR int cmp(const T a, const T b) noexcept
{
  return (a > b) - (b > a);
}

template <typename T>
T pow(T x, uint64_t n) noexcept
{
  switch (n)
  {
    case 0: return T(1);
    case 1: return x;
    case 2: return sqr(x);
    case 3: return x * sqr(x);
    case 4: return sqr(sqr(x));
    case 5: return x * sqr(sqr(x));
    case 6: return sqr(x * sqr(x));
    case 7: return x * sqr(x * sqr(x));
    case 8: return sqr(sqr(sqr(x)));
    default:
    {
      T y(1);
      while (n > 1)
      {
        if (n & 1)
          y *= x;
        x = sqr(x);
        n >>= 1;
      }
      return x * y;
    }
  }
}

template <typename T>
inline CONSTEXPR T diffabs(const T &c, const T &d) noexcept
{
  const T cd = c + d;
  const T c2d = 2 * c + d;
  return c >= 0.0 ? cd >= 0.0 ? d : -c2d : cd > 0.0 ? c2d : -d;
}

template<typename mantissa, typename exponent>
struct floatexp
{
  static CONSTEXPR_STATIC exponent LARGE_EXPONENT = sizeof(mantissa) == sizeof(float) ? 126 : 1022;
  static CONSTEXPR_STATIC exponent EXP_MIN = sizeof(exponent) == sizeof(int) ? exponent(-0x00800000) : exponent(-0x0080000000000000L);
  static CONSTEXPR_STATIC exponent EXP_MAX = sizeof(exponent) == sizeof(int) ? exponent( 0x00800000) : exponent( 0x0080000000000000L);

  mantissa val;
  exponent exp;

  // POD
  inline CONSTEXPR ~floatexp() = default;
  inline CONSTEXPR floatexp() = default;
  inline CONSTEXPR floatexp(const floatexp &fe) = default;
  inline CONSTEXPR floatexp(floatexp &&fe) = default;
  inline CONSTEXPR floatexp &operator=(const floatexp &fe) = default;

  inline CONSTEXPR floatexp(const float aval, const exponent aexp = 0) noexcept
  {
    if (aval == 0)
    {
      val = aval;
      exp = EXP_MIN;
    }
    else if (std::isnan(aval))
    {
      val = aval;
      exp = EXP_MIN;
    }
    else if (std::isinf(aval))
    {
      val = aval;
      exp = EXP_MAX;
    }
    else
    {
      int e = 0;
      mantissa f_val = std::frexp(aval, &e);
      exponent f_exp = e + aexp;
      if (f_exp >= EXP_MAX)
      {
        val = f_val / mantissa(0);
        exp = EXP_MAX;
      }
      else if (f_exp <= EXP_MIN)
      {
        val = f_val * mantissa(0);
        exp = EXP_MIN;
      }
      else
      {
        val = f_val;
        exp = f_exp;
      }
    }
  }

  inline CONSTEXPR floatexp(const double aval, const exponent aexp = 0) noexcept
  {
    if (aval == 0)
    {
      val = aval;
      exp = EXP_MIN;
    }
    else if (std::isnan(aval))
    {
      val = aval;
      exp = EXP_MIN;
    }
    else if (std::isinf(aval))
    {
      val = aval;
      exp = EXP_MAX;
    }
    else
    {
      int e = 0;
      mantissa f_val = std::frexp(aval, &e);
      exponent f_exp = e + aexp;
      if (f_exp >= EXP_MAX)
      {
        val = f_val / mantissa(0);
        exp = EXP_MAX;
      }
      else if (f_exp <= EXP_MIN)
      {
        val = f_val * mantissa(0);
        exp = EXP_MIN;
      }
      else
      {
        val = f_val;
        exp = f_exp;
      }
    }
  }

  inline CONSTEXPR floatexp(const long double aval, const exponent aexp = 0) noexcept
  {
    if (aval == 0)
    {
      val = aval;
      exp = EXP_MIN;
    }
    else if (std::isnan(aval))
    {
      val = aval;
      exp = EXP_MIN;
    }
    else if (std::isinf(aval))
    {
      val = aval;
      exp = EXP_MAX;
    }
    else
    {
      int e = 0;
      mantissa f_val = std::frexp(aval, &e);
      exponent f_exp = e + aexp;
      if (f_exp >= EXP_MAX)
      {
        val = f_val / mantissa(0);
        exp = EXP_MAX;
      }
      else if (f_exp <= EXP_MIN)
      {
        val = f_val * mantissa(0);
        exp = EXP_MIN;
      }
      else
      {
        val = f_val;
        exp = f_exp;
      }
    }
  }

#ifdef HAVE_FLOAT128
  inline CONSTEXPR floatexp(const float128 aval, const exponent aexp = 0) noexcept
  {
    if (aval == 0)
    {
      val = mantissa(aval);
      exp = EXP_MIN;
    }
    else if (isnan(aval))
    {
      val = mantissa(aval);
      exp = EXP_MIN;
    }
    else if (isinf(aval))
    {
      val = mantissa(aval);
      exp = EXP_MAX;
    }
    else
    {
      int e = 0;
      mantissa f_val = mantissa(frexp(aval, &e));
      exponent f_exp = e + aexp;
      if (f_exp >= EXP_MAX)
      {
        val = f_val / mantissa(0);
        exp = EXP_MAX;
      }
      else if (f_exp <= EXP_MIN)
      {
        val = f_val * mantissa(0);
        exp = EXP_MIN;
      }
      else
      {
        val = f_val;
        exp = f_exp;
      }
    }
  }
#endif

  template<typename M, typename E>
  inline CONSTEXPR floatexp(const floatexp<M, E> &x)
  : floatexp(x.val, x.exp)
  {
  }

  inline CONSTEXPR floatexp(const int aval, const exponent aexp = 0) noexcept
  : floatexp(mantissa(aval), aexp)
  {
  }

  inline CONSTEXPR floatexp(const long int aval, const exponent aexp = 0) noexcept
  : floatexp(mantissa(aval), aexp)
  {
  }

  inline CONSTEXPR floatexp(const long long int aval, const exponent aexp = 0) noexcept
  : floatexp(mantissa(aval), aexp)
  {
  }

  inline floatexp(const mpreal &x)
  {
    long e = 0;
    double v = mpfr_get_d_2exp(&e, x.mpfr_srcptr(), MPFR_RNDN);
    *this = floatexp(v, e);
  }

  explicit inline CONSTEXPR operator float() const noexcept
  {
    using std::ldexp;
    if (exp < -126)
    {
      return val * float(0);
    }
    if (exp > 126)
    {
      return val / float(0);
    }
    return ldexp(float(val), exp);
  }
  explicit inline CONSTEXPR operator double() const noexcept
  {
    using std::ldexp;
    if (exp < -1022)
    {
      return val * float(0);
    }
    if (exp > 1022)
    {
      return val / float(0);
    }
    return ldexp(double(val), exp);
  }
  explicit inline CONSTEXPR operator long double() const noexcept
  {
    using std::ldexp;
    if (exp < -16382)
    {
      return val * float(0);
    }
    if (exp > 16382)
    {
      return val / float(0);
    }
    return ldexp((long double)(val), exp);
  }
#ifdef HAVE_FLOAT128
  explicit inline CONSTEXPR operator float128() const noexcept
  {
    if (exp < -16382)
    {
      return val * float128(0);
    }
    if (exp > 16382)
    {
      return val / float128(0);
    }
    return ldexp(float128(val), exp);
  }
#endif
};

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> abs(const floatexp<M, E> f) noexcept
{
  floatexp<M, E> fe = { std::abs(f.val), f.exp };
  return fe;
}

template<typename M, typename E>
inline CONSTEXPR int sgn(const floatexp<M, E> f) noexcept
{
  return sgn(f.val);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator-(const floatexp<M, E> f) noexcept
{
  floatexp<M, E> fe = { -f.val, f.exp };
  return fe;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> sqr(const floatexp<M, E> a) noexcept
{
  return floatexp<M, E>(a.val * a.val, a.exp << 1);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a.val * b.val, a.exp + b.exp);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const floatexp<M, E> a, const float b) noexcept
{
  return a * floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const float a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) * b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const floatexp<M, E> a, const double b) noexcept
{
  return a * floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const double a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) * b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const floatexp<M, E> a, const int b) noexcept
{
  return a * floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const floatexp<M, E> a, const long int b) noexcept
{
  return a * floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const floatexp<M, E> a, const long long int b) noexcept
{
  return a * floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator*(const int a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) * b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E>& operator*=(floatexp<M, E> &a, const floatexp<M, E> b) noexcept
{
  return a = a * b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E>& operator*=(floatexp<M, E> &a, const float b) noexcept
{
  return a = a * b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E>& operator*=(floatexp<M, E> &a, const double b) noexcept
{
  return a = a * b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator<<(const floatexp<M, E> a, const E b) noexcept
{
  floatexp<M, E> fe = { a.val, a.exp + b };
  return fe;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> ldexp(const floatexp<M, E> a, const E b) noexcept
{
  floatexp<M, E> fe = { a.val, a.exp + b };
  return fe;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a.val / b.val, a.exp - b.exp);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const floatexp<M, E> a, float b) noexcept
{
  return a / floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const floatexp<M, E> a, double b) noexcept
{
  return a / floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const floatexp<M, E> a, const int b) noexcept
{
  return a / floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const floatexp<M, E> a, const long int b) noexcept
{
  return a / floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const floatexp<M, E> a, const long long int b) noexcept
{
  return a / floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const float a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) / b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const double a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) / b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const int a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) / b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const long int a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) / b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator/(const long long int a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) / b;
}


template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator>>(const floatexp<M, E> a, const E b) noexcept
{
  floatexp<M, E> fe = { a.val, a.exp - b };
  return fe;
}

template<typename M, typename E>
inline floatexp<M, E>& operator>>=(floatexp<M, E> &a, const E b) noexcept
{
  return a = a >> b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator+(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  if (a.exp > b.exp)
  {
    floatexp<M, E> c = { b.val, b.exp - a.exp };
    return floatexp<M, E>(a.val + M(c), a.exp);
  }
  else
  {
    floatexp<M, E> c = { a.val, a.exp - b.exp };
    return floatexp<M, E>(M(c) + b.val, b.exp);
  }
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator+(const floatexp<M, E> a, const int b) noexcept
{
  return a + floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E>& operator+=(floatexp<M, E> &a, const floatexp<M, E> b) noexcept
{
  return a = a + b;
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator-(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  return a + (-b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator-(const int a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) + (-b);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> operator-(const M a, const floatexp<M, E> b) noexcept
{
  return floatexp<M, E>(a) + (-b);
}

template<typename M, typename E>
inline CONSTEXPR int cmp(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  if (a.val > 0)
  {
    if (b.val <= 0)
    {
      return 1;
    }
    else if (a.exp > b.exp)
    {
      return 1;
    }
    else if (a.exp < b.exp)
    {
      return -1;
    }
    else
    {
      return cmp(a.val, b.val);
    }
  }
  else
  {
    if (b.val > 0)
    {
      return -1;
    }
    else if (a.exp > b.exp)
    {
      return -1;
    }
    else if (a.exp < b.exp)
    {
      return 1;
    }
    else
    {
      return cmp(a.val, b.val);
    }
  }
}

template<typename M, typename E>
inline CONSTEXPR bool operator<(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  if (std::isnan(a.val) || std::isnan(b.val)) return false;
  return cmp(a, b) < 0;
}

template<typename M, typename E>
inline CONSTEXPR bool operator<=(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  if (std::isnan(a.val) || std::isnan(b.val)) return false;
  return cmp(a, b) <= 0;
}

template<typename M, typename E>
inline CONSTEXPR bool operator>(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  if (std::isnan(a.val) || std::isnan(b.val)) return false;
  return cmp(a, b) > 0;
}

template<typename M, typename E>
inline CONSTEXPR bool operator>=(const floatexp<M, E> a, const floatexp<M, E> b) noexcept
{
  if (std::isnan(a.val) || std::isnan(b.val)) return false;
  return cmp(a, b) >= 0;
}

template<typename M, typename E>
inline CONSTEXPR bool operator>(const floatexp<M, E> a, const int b) noexcept
{
  return a > floatexp<M, E>(b);
}

template<typename M, typename E>
inline CONSTEXPR bool operator>=(const floatexp<M, E> a, const int b) noexcept
{
  return a >= floatexp<M, E>(b);
}

template<typename M, typename E>
inline /*CONSTEXPR*/ floatexp<M, E> sqrt(const floatexp<M, E> a) noexcept
{
  return floatexp<M, E>
    ( std::sqrt((a.exp & 1) ? 2.0 * a.val : a.val)
    , (a.exp & 1) ? (a.exp - 1) / 2 : a.exp / 2
    );
}

template<typename M, typename E>
inline /*CONSTEXPR*/ floatexp<M, E> log(const floatexp<M, E> a) noexcept
{
  return floatexp<M, E>(std::log(a.val) + std::log(2.0) * a.exp, 0);
}

template<typename M, typename E>
inline /*CONSTEXPR*/ floatexp<M, E> log2(const floatexp<M, E> a) noexcept
{
  return floatexp<M, E>(std::log2(a.val) + a.exp, 0);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> exp(const floatexp<M, E> a) noexcept
{
  using std::exp;
  if (-53 <= a.exp && a.exp <= 8) return floatexp<M, E>(std::exp(double(M(a))), 0);
  if (61 <= a.exp) a.val > 0.0 ? floatexp<M, E>(a.val / 0.0, 0) : floatexp<M, E>(0.0, 0);
  if (a.exp < -53) return floatexp<M, E>(1.0, 0);
  return pow(floatexp<M, E>(std::exp(a.val), 0), int64_t(1) << a.exp);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> sin(const floatexp<M, E> a) noexcept
{
  using std::sin;
  using std::isnan;
  using std::isinf;
  double y = double(a);
  if (isinf(y))
  {
    return 0.0/0.0;
  }
  if (isnan(y))
  {
    return y;
  }
  if (y == 0) // FIXME denormalized numbers lose precision
  {
    return a;
  }
  return floatexp<M, E>(sin(y), 0);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> cos(const floatexp<M, E> a) noexcept
{
  using std::sin;
  using std::isnan;
  using std::isinf;
  double y = double(a);
  if (isinf(y))
  {
    return 0.0/0.0;
  }
  if (isnan(y))
  {
    return y;
  }
  return floatexp<M, E>(cos(y), 0);
}

template<typename M, typename E>
inline CONSTEXPR floatexp<M, E> diffabs(const floatexp<M, E> &c, const floatexp<M, E> &d) noexcept
{
  const floatexp<M, E> cd = c + d;
  const floatexp<M, E> c2d = 2 * c + d;
  return c.val >= 0.0 ? cd.val >= 0.0 ? d : -c2d : cd.val > 0.0 ? c2d : -d;
}

template<typename M, typename E>
inline /*CONSTEXPR*/ floatexp<M, E> hypot(const floatexp<M, E> &x, const floatexp<M, E> &y) noexcept
{
  return sqrt(sqr(x) + sqr(y));
}

template<typename M, typename E>
inline /*CONSTEXPR*/ floatexp<M, E> e10(const M a, const E e) noexcept
{
  return exp(floatexp<M, E>(std::log(a) + std::log(10.0) * e));
}

template<typename mantissa, typename exponent>
inline std::ostream &operator<<(std::ostream &o, const floatexp<mantissa, exponent> f) noexcept
{
  if (std::isnan(f.val)) return o << "nan";
  if (std::isinf(f.val)) return o << (f.val > 0 ? "+inf" : "-inf");
  mantissa lf = std::log10(std::abs(f.val)) + f.exp * std::log10(2.0);
  exponent e10 = exponent(std::floor(lf));
  mantissa d10 = std::pow(10, lf - e10) * ((f.val > 0) - (f.val < 0));
  if (std::abs(d10) == 10)
  {
    d10 /= 10;
    e10 += 1;
  }
  if (f.val == 0) { d10 = 0; e10 = 0; }
  return o
    << std::setprecision(std::numeric_limits<mantissa>::digits10 + 1)
    << std::fixed
    << d10 << 'e' << e10;
}

template<typename M, typename E>
inline CONSTEXPR bool isinf(const floatexp<M, E> &x) noexcept
{
  return std::isinf(x.val);
}

template<typename M, typename E>
inline CONSTEXPR bool operator==(const floatexp<M, E> &a, const floatexp<M, E> &b)
{
  return a.val == b.val && a.exp == b.exp;
}
