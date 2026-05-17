// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cmath>

#include "float128.h"

inline constexpr float sqr(const float a) noexcept
{
  return a * a;
}

inline constexpr double sqr(const double a) noexcept
{
  return a * a;
}

inline constexpr long double sqr(const long double a) noexcept
{
  return a * a;
}

template <typename real>
struct complex
{
  real x;
  real y;

  // POD
  inline ~complex() = default;
  inline constexpr complex() = default;
  inline constexpr complex(const complex &fe) = default;
  inline constexpr complex(complex &&fe) = default;
  inline constexpr complex &operator=(const complex &fe) = default;
  inline constexpr complex(const real &ax, const real &ay) noexcept
  : x(ax)
  , y(ay)
  {
  }
  inline constexpr complex(const real &ax) noexcept
  : complex(ax, real(0))
  {
  }

  inline constexpr complex &operator=(const int i)
  {
    x = i;
    y = 0;
    return *this;
  }

  inline constexpr complex &operator+=(const complex &z)
  {
    x += z.x;
    y += z.y;
    return *this;
  }
};

template <typename real>
inline constexpr real norm(const complex<real> &z) noexcept
{
  return sqr(z.x) + sqr(z.y);
}

template <typename real>
inline /*constexpr*/ real abs(const complex<real> &z) noexcept
{
  using std::sqrt;
  using ::sqrt;
  return sqrt(norm(z));
}

template <typename real>
inline constexpr real arg(const complex<real> &a) noexcept
{
  using std::atan2;
  return atan2(a.y, a.x);
}

template <typename real>
inline constexpr complex<real> conj(const complex<real> &a) noexcept
{
  return complex<real>(a.x, -a.y);
}

template <typename real>
inline constexpr complex<real> operator-(const complex<real> &a) noexcept
{
  return complex<real>(-a.x, -a.y);
}

template <typename real>
inline constexpr complex<real> sqr(const complex<real> &a) noexcept
{
  return complex<real>(sqr(a.x) - sqr(a.y), 2 * a.x * a.y);
}

template <typename real>
inline constexpr complex<real> operator*(const complex<real> &a, const complex<real> &b) noexcept
{
  return complex<real>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

template <typename real>
inline constexpr complex<real> operator*(const real &a, const complex<real> &b) noexcept
{
  return complex<real>(a * b.x, a * b.y);
}

template <typename real>
inline constexpr complex<real> operator*(const int a, const complex<real> &b) noexcept
{
  return complex<real>(a * b.x, a * b.y);
}

template <typename real>
inline constexpr complex<real> operator*(const complex<real> &a, const real &b) noexcept
{
  return complex<real>(a.x * b, a.y * b);
}

template <typename real>
inline constexpr complex<real> operator/(const complex<real> &a, const real &b) noexcept
{
  return complex<real>(a.x / b, a.y / b);
}

template <typename real>
inline constexpr complex<real> operator/(const complex<real> &a, const complex<real> &b) noexcept
{
  return a * conj(b) / norm(b);
}

template <typename real>
inline constexpr complex<real> operator/(const real &a, const complex<real> &b) noexcept
{
  return a * conj(b) / norm(b);
}

template <typename real>
inline constexpr complex<real> operator/(const int &a, const complex<real> &b) noexcept
{
  return real(a) / b;
}

template <typename real>
inline constexpr complex<real> operator+(const complex<real> &a, const complex<real> &b) noexcept
{
  return complex<real>(a.x + b.x, a.y + b.y);
}

template <typename real>
inline constexpr complex<real> operator+(const real &a, const complex<real> &b) noexcept
{
  return complex<real>(a + b.x, b.y);
}

template <typename real>
inline constexpr complex<real> operator+(const complex<real> &a, const real &b) noexcept
{
  return complex<real>(a.x + b, a.y);
}

template <typename real>
inline constexpr complex<real> operator+(const complex<real> &a, const int b) noexcept
{
  return complex<real>(a.x + b, a.y);
}

template <typename real>
inline constexpr complex<real> operator-(const complex<real> &a, const complex<real> &b) noexcept
{
  return complex<real>(a.x - b.x, a.y - b.y);
}

template <typename real>
inline constexpr complex<real> operator-(const real &a, const complex<real> &b) noexcept
{
  return complex<real>(a - b.x, -b.y);
}

template <typename real>
inline constexpr complex<real> operator-(const complex<real> &a, const real &b) noexcept
{
  return complex<real>(a.x - b, a.y);
}

template <typename real>
inline constexpr complex<real> &operator*=(complex<real> &a, const complex<real> &b) noexcept
{
  return a = a * b;
}
