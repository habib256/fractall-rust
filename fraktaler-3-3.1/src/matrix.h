// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>

#include <glm/glm.hpp>

#include "complex.h"

template <typename real>
struct mat2
{
  real x[2][2];

  // POD
  inline ~mat2() = default;
  inline mat2() = default;
  inline constexpr mat2(const mat2 &m) = default;
  inline constexpr mat2(mat2 &&m) = default;
  inline constexpr mat2 &operator=(const mat2 &m) = default;
  
  inline constexpr mat2(const real &s) noexcept
  {
    x[0][0] = s;
    x[0][1] = 0;
    x[1][0] = 0;
    x[1][1] = s;
  }
  inline constexpr mat2(const complex<real> &z) noexcept
  {
    x[0][0] = z.x;
    x[0][1] = -z.y;
    x[1][0] = z.y;
    x[1][1] = z.x;
  }
  inline constexpr mat2(const real &a, const real &b, const real &c, const real &d) noexcept
  {
    x[0][0] = a;
    x[0][1] = b;
    x[1][0] = c;
    x[1][1] = d;
  }
  inline constexpr mat2(const glm::mat3 &m) // FIXME check transpose?
  {
    x[0][0] = m[0][0];
    x[0][1] = m[1][0];
    x[1][0] = m[0][1];
    x[1][1] = m[1][1];
  }

  inline constexpr mat2 &operator+=(const mat2 &b)
  {
    x[0][0] += b.x[0][0];
    x[0][1] += b.x[0][1];
    x[1][0] += b.x[1][0];
    x[1][1] += b.x[1][1];
    return *this;
  }
  inline constexpr mat2 &operator/=(const real &b)
  {
    x[0][0] /= b;
    x[0][1] /= b;
    x[1][0] /= b;
    x[1][1] /= b;
    return *this;
  }
};

template <typename real>
inline constexpr mat2<real> operator+(const mat2<real> &a, const mat2<real> &b)
{
  return mat2<real>
    ( a.x[0][0] + b.x[0][0]
    , a.x[0][1] + b.x[0][1]
    , a.x[1][0] + b.x[1][0]
    , a.x[1][1] + b.x[1][1]
    );
}

template <typename real>
inline constexpr mat2<real> operator*(const mat2<real> &a, const mat2<real> &b)
{
  return mat2<real>
    ( a.x[0][0] * b.x[0][0] + a.x[0][1] * b.x[1][0]
    , a.x[0][0] * b.x[0][1] + a.x[0][1] * b.x[1][1]
    , a.x[1][0] * b.x[0][0] + a.x[1][1] * b.x[1][0]
    , a.x[1][0] * b.x[0][1] + a.x[1][1] * b.x[1][1]
    );
}

template <typename real>
inline constexpr mat2<real> operator*(const mat2<real> &a, const real &b)
{
  return mat2<real>
    ( a.x[0][0] * b
    , a.x[0][1] * b
    , a.x[1][0] * b
    , a.x[1][1] * b
    );
}

template <typename S, typename T>
inline constexpr complex<T> operator*(const mat2<S> &m, const complex<T> &z) noexcept
{
  return complex<T>(m.x[0][0] * z.x + m.x[0][1] * z.y, m.x[1][0] * z.x + m.x[1][1] * z.y);
}

template <typename S, typename T>
inline constexpr complex<T> operator*(const complex<T> &z, const mat2<S> &m) noexcept
{
  return complex<T>(m.x[0][0] * z.x + m.x[1][0] * z.y, m.x[0][1] * z.x + m.x[1][1] * z.y);
}

template <typename S, typename T>
inline constexpr mat2<T> operator/(const mat2<T> &m, const S &b) noexcept
{
  return mat2<T>(m.x[0][0] / b, m.x[0][1] / b, m.x[1][0] / b, m.x[1][1] / b);
}

template <typename real>
inline constexpr mat2<real> transpose(const mat2<real> &a) noexcept
{
  return mat2<real>
    ( a.x[0][0]
    , a.x[1][0]
    , a.x[0][1]
    , a.x[1][1]
    );
}

template <typename real>
inline constexpr real trace(const mat2<real> &a) noexcept
{
  return a.x[0][0] + a.x[1][1];
}

template <typename real>
inline constexpr real determinant(const mat2<real> &a) noexcept
{
  return a.x[0][0] * a.x[1][1] - a.x[0][1] * a.x[1][0];
}

template <typename real>
inline constexpr real sup(const mat2<real> &a) noexcept
{
  using std::max;
  using std::sqrt;
  using ::sqrt;
  const mat2<real> aTa = transpose(a) * a;
  const real T = trace(aTa);
  const real D = determinant(aTa);
  return sqrt(max(real(0), (T + sqrt(max(real(0), sqr(T) - 4 * D))) / 2));
}

template <typename real>
inline constexpr real inf(const mat2<real> &a) noexcept
{
  using std::max;
  using std::sqrt;
  using ::sqrt;
  const mat2<real> aTa = transpose(a) * a;
  const real T = trace(aTa);
  const real D = determinant(aTa);
  return sqrt(max(real(0), (T - sqrt(max(real(0), sqr(T) - 4 * D))) / 2));
}

template <typename real>
inline constexpr mat2<real> inverse(const mat2<real> &a) noexcept
{
  return mat2<real>(a.x[1][1], -a.x[0][1], -a.x[1][0], a.x[0][0]) / determinant(a);
}

template <typename real>
inline constexpr mat2<real> rotation(const real &radians) noexcept
{
  using std::cos;
  using std::sin;
  return mat2<real>(complex<real>(cos(radians), sin(radians)));
}

template <typename real>
struct polar2
{
  double sign, scale, rotate, stretch_factor, stretch_angle;
  polar2(real g, real s, real r, real sf, real sa)
  : sign(g)
  , scale(s)
  , rotate(r)
  , stretch_factor(sf)
  , stretch_angle(sa)
  {
  }
  explicit operator mat2<real>() const noexcept
  {
    using std::cos;
    using std::sin;
    const real r = sign * rotate;
    const real a = stretch_angle;
    const mat2<real> R( cos(r), -sin(r), sin(r), cos(r) );
    const mat2<real> S( stretch_factor, 0, 0, 1/stretch_factor );
    const mat2<real> T( cos(a), -sin(a), sin(a), cos(a) );
    const mat2<real> G( 1, 0, 0, sign );
    return R * T * S * transpose(T) * scale * G;
  }
  polar2(const mat2<real> &M) noexcept
  {
    using std::sqrt;
    using std::abs;
    using std::atan2;
    const real mdet = determinant(M);
    const real sign = mdet > 0 ? 1 : -1;
    const real scale = sqrt(abs(mdet));
    if (scale != 0)
    {
      const mat2<real> A = M / scale;
      const mat2<real> B = A + inverse(transpose(A));
      const real b = sqrt(abs(determinant(B)));
      const mat2<real> V = B / b;
      const real rotate = sign * atan2(V.x[1][0], V.x[0][0]);
      const mat2<real> P = (transpose(A) * A + mat2<real>(1,0,0,1)) / b;
      // [U,D] = eig(P);
      const real pa = P.x[0][0];
      const real pb = P.x[0][1];
      const real pc = P.x[1][0];
      const real pd = P.x[1][1];
      const real ptr2 = (pa + pd) / 2;
      const real pdet = pa * pd - pb * pc;
      const real pdisc = ptr2 * ptr2 - pdet;
      if (pdisc > 0)
      {
        const real d1 = ptr2 + sqrt(pdisc);
        const real d2 = ptr2 - sqrt(pdisc);
        real ua, ub, uc, ud;
        if (pb != 0)
        {
          ua = pb;
          ub = pb;
          uc = d1 - pa;
          ud = d2 - pa;
        }
        else if (pc != 0)
        {
          ua = d1 - pd;
          ub = d2 - pd;
          uc = pc;
          ud = pc;
        }
        else
        {
          ua = 1;
          ub = 0;
          uc = 0;
          ud = 1;
        }
        real stretch_factor = d1;
        real stretch_angle = sign * atan2(uc, ua);
        (void) ub;
        (void) ud;
        if (stretch_factor < 1)
        {
          stretch_factor = 1 / stretch_factor;
          stretch_angle += 1.5707963267948966;
        }
        *this = polar2(sign, scale, rotate, stretch_factor, stretch_angle);
        return;
      }
      *this = polar2(sign, scale, rotate, 1, 0);
      return;
    }
    *this = polar2(1, 1, 0, 1, 0);
    return;
  }
};

// compute eigenvalues
// <http://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html>

inline glm::dvec2 eigenvalues(const glm::dmat2 &m) noexcept
{
  const double t = (m[0][0] + m[1][1]) / 2.0;
  const double d = std::sqrt(std::max(0.0, t * t - glm::determinant(m)));
  const double e_add = t + d;
  const double e_sub = t - d;
  glm::dvec2 r(e_add, e_sub);
  return r;
}

// ratio of eigenvalues of a 2x2 covariance matrix (largest / smallest)

inline double eigenvalue_ratio(const glm::dmat2 &m) noexcept
{
  glm::dvec2 e = eigenvalues(m);
  const double e_max = std::max(std::abs(e[0]), std::abs(e[1]));
  const double e_min = std::min(std::abs(e[0]), std::abs(e[1]));
  return e_max / e_min;
}
