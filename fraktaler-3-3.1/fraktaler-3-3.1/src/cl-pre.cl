// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#ifndef NUMBER_TYPE
#error NUMBER_TYPE not defined
#endif

#define MAKE_CONSTANT_GLOBAL // TODO expose via param
#ifdef MAKE_CONSTANT_GLOBAL // support some OpenCL broken driver versions
#define CONSTANT __global
#else
#define CONSTANT __constant
#endif

#if NUMBER_TYPE == 2 || NUMBER_TYPE == 5
#undef HAVE_DOUBLE
#define HAVE_DOUBLE 1
#endif

#ifdef HAVE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef cl_khr_fp64
#error fp64 required
#endif
#endif

#define assert(x) do{}while(0)

#define MAX_PHASES 64
#define MAX_LEVELS 64

#if NUMBER_TYPE == 4 || NUMBER_TYPE == 5 || NUMBER_TYPE == 6 // floatexp, doubleexp, softfloat

#if NUMBER_TYPE == 5 // doubleexp
typedef double mantissa;
#else
typedef float mantissa;
#endif
typedef int exponent;
struct floatexp
{
  mantissa val;
  exponent exp;
};
__constant const exponent LARGE_EXPONENT = sizeof(mantissa) == sizeof(float) ? 126 : 1022;
__constant const exponent EXP_MIN = sizeof(exponent) == sizeof(int) ? (exponent)(-0x00800000) : (exponent)(-0x0080000000000000L);
__constant const exponent EXP_MAX = sizeof(exponent) == sizeof(int) ? (exponent)( 0x00800000) : (exponent)( 0x0080000000000000L);

struct floatexp floatexp_from_mantissa_exponent(const mantissa aval, const exponent aexp)
{
  struct floatexp r;
  if (aval == 0)
  {
    r.val = aval;
    r.exp = EXP_MIN;
  }
  else if (isnan(aval))
  {
    r.val = aval;
    r.exp = EXP_MIN;
  }
  else if (isinf(aval))
  {
    r.val = aval;
    r.exp = EXP_MAX;
  }
  else
  {
    int e = 0;
    mantissa f_val = frexp(aval, &e);
    exponent f_exp = e + aexp;
    if (f_exp >= EXP_MAX)
    {
      r.val = f_val / (mantissa)(0);
      r.exp = EXP_MAX;
    }
    else if (f_exp <= EXP_MIN)
    {
      r.val = f_val * (mantissa)(0);
      r.exp = EXP_MIN;
    }
    else
    {
      r.val = f_val;
      r.exp = f_exp;
    }
  }
  return r;
}

struct floatexp floatexp_from_float(const float aval)
{
  return floatexp_from_mantissa_exponent(aval, 0);
}

struct floatexp floatexp_from_int(const int aval)
{
  return floatexp_from_mantissa_exponent(aval, 0);
}

struct floatexp floatexp_from_long(const long aval)
{
  return floatexp_from_mantissa_exponent(aval, 0);
}

mantissa mantissa_from_floatexp(const struct floatexp a)
{
  if (a.exp < -126)
  {
    return a.val * (mantissa)(0);
  }
  if (a.exp > 126)
  {
    return a.val / (mantissa)(0);
  }
  return ldexp((mantissa)(a.val), a.exp);
}

float float_from_floatexp(const struct floatexp a)
{
  return mantissa_from_floatexp(a);
}

#ifdef HAVE_DOUBLE
double double_from_floatexp(const struct floatexp a)
{
  if (a.exp < -1022)
  {
    return (double)(a.val) * (double)(0);
  }
  if (a.exp > 1022)
  {
    return (double)(a.val) / (double)(0);
  }
  return ldexp((double)(a.val), a.exp);
}
#endif

struct floatexp floatexp_abs_floatexp(const struct floatexp f)
{
  struct floatexp fe = { fabs(f.val), f.exp };
  return fe;
}

struct floatexp floatexp_neg_floatexp(const struct floatexp f)
{
  struct floatexp fe = { -f.val, f.exp };
  return fe;
}

struct floatexp floatexp_sqr_floatexp(const struct floatexp a)
{
  return floatexp_from_mantissa_exponent(a.val * a.val, a.exp << 1);
}

struct floatexp floatexp_mul_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
{
  return floatexp_from_mantissa_exponent(a.val * b.val, a.exp + b.exp);
}

struct floatexp floatexp_mul2exp_floatexp_exponent(const struct floatexp a, const exponent b)
{
  struct floatexp fe = { a.val, a.exp + b };
  return fe;
}

struct floatexp floatexp_ldexp_floatexp_int(const struct floatexp a, const int b)
{
  return floatexp_mul2exp_floatexp_exponent(a, b);
}

struct floatexp floatexp_div_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
{
  return floatexp_from_mantissa_exponent(a.val / b.val, a.exp - b.exp);
}

struct floatexp floatexp_div2exp_floatexp_exponent(const struct floatexp a, const exponent b)
{
  struct floatexp fe = { a.val, a.exp - b };
  return fe;
}

struct floatexp floatexp_add_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
{
  if (a.exp > b.exp)
  {
    struct floatexp c = { b.val, b.exp - a.exp };
    return floatexp_from_mantissa_exponent(a.val + mantissa_from_floatexp(c), a.exp);
  }
  else
  {
    struct floatexp c = { a.val, a.exp - b.exp };
    return floatexp_from_mantissa_exponent(mantissa_from_floatexp(c) + b.val, b.exp);
  }
}

struct floatexp floatexp_sub_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
{
  return floatexp_add_floatexp_floatexp(a, floatexp_neg_floatexp(b));
}

int int_cmp_mantissa_mantissa(const mantissa a, const mantissa b)
{
  return (a > b) - (b > a);
}

int int_cmp_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
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
      return int_cmp_mantissa_mantissa(a.val, b.val);
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
      return int_cmp_mantissa_mantissa(a.val, b.val);
    }
  }
}

struct floatexp floatexp_min_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
{
  return int_cmp_floatexp_floatexp(a, b) <= 0 ? a : b;
}

bool bool_gezero_floatexp(const struct floatexp a)
{
  return a.val >= 0;
}

bool bool_gtzero_floatexp(const struct floatexp a)
{
  return a.val > 0;
}

bool bool_ltzero_floatexp(const struct floatexp a)
{
  return a.val < 0;
}

bool bool_lt_floatexp_floatexp(const struct floatexp a, const struct floatexp b)
{
  return int_cmp_floatexp_floatexp(a, b) < 0;
}

struct floatexp floatexp_floor_floatexp(const struct floatexp a)
{
  if (a.exp >= LARGE_EXPONENT)
  {
    // already an integer
    return a;
  }
  else if (a.exp <= -LARGE_EXPONENT)
  {
    // very small
    if (a.val < 0)
    {
      return floatexp_from_int(-1);
    }
    else
    {
      return floatexp_from_int(0);
    }
  }
  else
  {
    // won't under/overflow
    return floatexp_from_mantissa_exponent(floor(mantissa_from_floatexp(a)), 0);
  }
}

struct floatexp floatexp_sqrt_floatexp(const struct floatexp a)
{
  return floatexp_from_mantissa_exponent
    ( sqrt((a.exp & 1) ? 2 * a.val : a.val)
    , (a.exp & 1) ? (a.exp - 1) / 2 : a.exp / 2
    );
}

struct floatexp floatexp_log_floatexp(const struct floatexp a)
{
  return floatexp_from_mantissa_exponent(log(a.val) + log((mantissa) 2) * a.exp, 0);
}

struct floatexp floatexp_pow_floatexp_ulong(struct floatexp x, ulong n)
{
  switch (n)
  {
    case 0: return floatexp_from_int(1);
    case 1: return x;
    case 2: return floatexp_sqr_floatexp(x);
    case 3: return floatexp_mul_floatexp_floatexp(x, floatexp_sqr_floatexp(x));
    case 4: return floatexp_sqr_floatexp(floatexp_sqr_floatexp(x));
    case 5: return floatexp_mul_floatexp_floatexp(x, floatexp_sqr_floatexp(floatexp_sqr_floatexp(x)));
    case 6: return floatexp_sqr_floatexp(floatexp_mul_floatexp_floatexp(x, floatexp_sqr_floatexp(x)));
    case 7: return floatexp_mul_floatexp_floatexp(x, floatexp_sqr_floatexp(floatexp_mul_floatexp_floatexp(x, floatexp_sqr_floatexp(x))));
    case 8: return floatexp_sqr_floatexp(floatexp_sqr_floatexp(floatexp_sqr_floatexp(x)));
    default:
    {
      struct floatexp y = floatexp_from_int(1);
      while (n > 1)
      {
        if (n & 1)
          y = floatexp_mul_floatexp_floatexp(y, x);
        x = floatexp_sqr_floatexp(x);
        n >>= 1;
      }
      return floatexp_mul_floatexp_floatexp(x, y);
    }
  }
}

struct floatexp floatexp_exp_floatexp(const struct floatexp a)
{
  if (-53 <= a.exp && a.exp <= 8) return floatexp_from_mantissa_exponent(exp(mantissa_from_floatexp(a)), 0);
  if (61 <= a.exp) return a.val > 0 ? floatexp_from_mantissa_exponent(a.val / 0, 0) : floatexp_from_mantissa_exponent(0, 0);
  if (a.exp < -53) return floatexp_from_mantissa_exponent(1, 0);
  return floatexp_pow_floatexp_ulong(floatexp_from_mantissa_exponent(exp(a.val), 0), ((ulong)(1)) << a.exp);
}

struct floatexp floatexp_sin_floatexp(const struct floatexp a)
{
  mantissa y = mantissa_from_floatexp(a);
  if (isinf(y))
  {
    return floatexp_from_mantissa_exponent(0.0f/0.0f, 0);
  }
  if (isnan(y))
  {
    return floatexp_from_mantissa_exponent(y, 0);
  }
  if (y == 0) // FIXME denormalized numbers lose precision
  {
    return a;
  }
  return floatexp_from_mantissa_exponent(sin(y), 0);
}

struct floatexp floatexp_cos_floatexp(const struct floatexp a)
{
  mantissa y = mantissa_from_floatexp(a);
  if (isinf(y))
  {
    return floatexp_from_mantissa_exponent(0.0f/0.0f, 0);
  }
  if (isnan(y))
  {
    return a;
  }
  return floatexp_from_mantissa_exponent(cos(y), 0);
}

struct floatexp floatexp_diffabs_floatexp_floatexp(const struct floatexp c, const struct floatexp d)
{
  const struct floatexp cd = floatexp_add_floatexp_floatexp(c, d);
  const struct floatexp c2d = floatexp_add_floatexp_floatexp(floatexp_mul2exp_floatexp_exponent(c, 1), d);
  return c.val >= 0 ? cd.val >= 0 ? d : floatexp_neg_floatexp(c2d) : cd.val > 0 ? c2d : floatexp_neg_floatexp(d);
}

struct floatexp floatexp_hypot_floatexp_floatexp(const struct floatexp x, const struct floatexp y)
{
  return floatexp_sqrt_floatexp(floatexp_add_floatexp_floatexp(floatexp_sqr_floatexp(x), floatexp_sqr_floatexp(y)));
}

struct floatexp floatexp_atan2_floatexp_floatexp(struct floatexp y, struct floatexp x)
{
  struct floatexp z = floatexp_hypot_floatexp_floatexp(y, x);
  x = floatexp_div_floatexp_floatexp(x, z);
  y = floatexp_div_floatexp_floatexp(y, z);
  return floatexp_from_mantissa_exponent(atan2(mantissa_from_floatexp(y), mantissa_from_floatexp(x)), 0);
}

struct floatexp floatexp_nextafter_floatexp_floatexp(const struct floatexp x, const struct floatexp y)
{
  return floatexp_from_mantissa_exponent(nextafter(x.val, mantissa_from_floatexp(floatexp_div2exp_floatexp_exponent(y, x.exp))), x.exp);
}

bool bool_isinf_floatexp(const struct floatexp x)
{
  return isinf(x.val);
}

bool bool_isnan_floatexp(const struct floatexp x)
{
  return isnan(x.val);
}

#if NUMBER_TYPE == 6 // softfloat

struct softfloat
{
  uint se;
  uint m;
};
__constant const uint EXPONENT_BIAS = (1U << 30U) - 1U;
__constant const uint MANTISSA_BITS = 32U;

bool bool_signbit_softfloat(const struct softfloat f)
{
  return !!(f.se & 0x80000000U);
}

uint uint_biasedexponent_softfloat(const struct softfloat f)
{
  return f.se & 0x7FFFFFFFU;
}

bool bool_iszero_softfloat(const struct softfloat f)
{
  return
    uint_biasedexponent_softfloat(f) == 0 &&
    f.m == 0;
}

bool bool_isdenormal_softfloat(const struct softfloat f)
{
  return
    uint_biasedexponent_softfloat(f) == 0 &&
    f.m != 0;
}

bool bool_isinf_softfloat(const struct softfloat f)
{
  return
    uint_biasedexponent_softfloat(f) == 0x7FFFFFFFU &&
    f.m == 0;
}

bool bool_isnan_softfloat(const struct softfloat f)
{
  return
    uint_biasedexponent_softfloat(f) == 0x7FFFFFFFU &&
    f.m != 0;
}

struct softfloat softfloat_from_float(const float x)
{
  uint se, m;
  if (isnan(x))
  {
    se = ((uint)(!!signbit(x)) << 31) | 0x7FFFFFFFU;
    m = 0xFFFFFFFFU;
  }
  else if (isinf(x))
  {
    se = ((uint)(!!signbit(x)) << 31) | 0x7FFFFFFFU;
    m = 0U;
  }
  else if (x == 0)
  {
    se = ((uint)(!!signbit(x)) << 31) | 0U;
    m = 0U;
  }
  else
  {
    int e = 0;
    float y = frexp(fabs(x), &e);
    float z = ldexp(y, MANTISSA_BITS);
    uint biased_e = convert_uint_sat(e + EXPONENT_BIAS);
    se = ((uint)(!!signbit(x)) << 31) | biased_e;
    m = convert_uint_rtz(z);
    assert(0 < biased_e);
    assert(biased_e < 0x7FFFFFFFU);
    assert((m >> (MANTISSA_BITS - 1)) == 1U);
  }
  struct softfloat sf = { se, m };
  return sf;
}

#ifdef HAVE_DOUBLE
struct softfloat softfloat_from_double(const double x)
{
  uint se, m;
  if (isnan(x))
  {
    se = ((uint)(!!signbit(x)) << 31) | 0x7FFFFFFFU;
    m = 0xFFFFFFFFU;
  }
  else if (isinf(x))
  {
    se = ((uint)(!!signbit(x)) << 31) | 0x7FFFFFFFU;
    m = 0U;
  }
  else if (x == 0)
  {
    se = ((uint)(!!signbit(x)) << 31) | 0U;
    m = 0U;
  }
  else
  {
    int e = 0;
    float y = frexp(fabs(x), &e);
    float z = ldexp(y, MANTISSA_BITS);
    uint biased_e = convert_uint_sat(e + EXPONENT_BIAS);
    se = ((uint)(!!signbit(x)) << 31) | biased_e;
    m = convert_uint_rtz(z);
    assert(0 < biased_e);
    assert(biased_e < 0x7FFFFFFFU);
    assert((m >> (MANTISSA_BITS - 1)) == 1U);
  }
  struct softfloat sf = { se, m };
  return sf;
}
#else
struct softfloat softfloat_from_double(const float x)
{
  return softfloat_from_float(x);
}
#endif

struct softfloat softfloat_from_int(const int x)
{
  return softfloat_from_double(x);
}

struct softfloat softfloat_from_long(const long x)
{
  return softfloat_from_double(x);
}

float float_from_softfloat(const struct softfloat f)
{
  if (bool_iszero_softfloat(f) || bool_isdenormal_softfloat(f))
  {
    if (bool_signbit_softfloat(f)) return -0.0f; else return 0.0f;
  }
  else if (bool_isinf_softfloat(f))
  {
    if (bool_signbit_softfloat(f)) return -1.0f/0.0f; else return 1.0f/0.0f;
  }
  else if (bool_isnan_softfloat(f))
  {
    if (bool_signbit_softfloat(f)) return -(0.0f/0.0f); else return 0.0f/0.0f;
  }
  else
  {
    float x = f.m;
    int e
      = convert_int_sat((long)(uint_biasedexponent_softfloat(f))
      - (long)(EXPONENT_BIAS + MANTISSA_BITS));
    if (bool_signbit_softfloat(f)) return -ldexp(x, e); else return ldexp(x, e);
  }
}

#ifdef HAVE_DOUBLE
double double_from_softfloat(const struct softfloat f)
{
  if (bool_iszero_softfloat(f) || bool_isdenormal_softfloat(f))
  {
    if (bool_signbit_softfloat(f)) return -0.0; else return 0.0;
  }
  else if (bool_isinf_softfloat(f))
  {
    if (bool_signbit_softfloat(f)) return -1.0/0.0; else return 1.0/0.0;
  }
  else if (bool_isnan_softfloat(f))
  {
    if (bool_signbit_softfloat(f)) return -(0.0/0.0); else return 0.0/0.0;
  }
  else
  {
    double x = f.m;
    int e
      = convert_int_sat((long)(uint_biasedexponent_softfloat(f))
      - (long)(EXPONENT_BIAS + MANTISSA_BITS));
    if (bool_signbit_softfloat(f)) return -ldexp(x, e); else return ldexp(x, e);
  }
}
#else
float double_from_softfloat(const struct softfloat f)
{
  return float_from_softfloat(f);
}
#endif

bool bool_lt_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  if (bool_isnan_softfloat(a) || bool_isnan_softfloat(b))
  {
    return false;
  }
  else if (bool_signbit_softfloat(a) && ! bool_signbit_softfloat(b))
  {
    return true;
  }
  else if (! bool_signbit_softfloat(a) && bool_signbit_softfloat(b))
  {
    return false;
  }
  else if (uint_biasedexponent_softfloat(a) > uint_biasedexponent_softfloat(b))
  {
    return bool_signbit_softfloat(a);
  }
  else if (uint_biasedexponent_softfloat(a) < uint_biasedexponent_softfloat(b))
  {
    return ! bool_signbit_softfloat(a);
  }
  else if (a.m > b.m)
  {
    return bool_signbit_softfloat(a);
  }
  else if (a.m < b.m)
  {
    return ! bool_signbit_softfloat(a);
  }
  else
  {
    // equal
    return false;
  }
}

bool bool_le_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  if (bool_isnan_softfloat(a) || bool_isnan_softfloat(b))
  {
    return false;
  }
  else if (bool_signbit_softfloat(a) && ! bool_signbit_softfloat(b))
  {
    return true;
  }
  else if (! bool_signbit_softfloat(a) && bool_signbit_softfloat(b))
  {
    return false;
  }
  else if (uint_biasedexponent_softfloat(a) > uint_biasedexponent_softfloat(b))
  {
    return bool_signbit_softfloat(a);
  }
  else if (uint_biasedexponent_softfloat(a) < uint_biasedexponent_softfloat(b))
  {
    return ! bool_signbit_softfloat(a);
  }
  else if (a.m > b.m)
  {
    return bool_signbit_softfloat(a);
  }
  else if (a.m < b.m)
  {
    return ! bool_signbit_softfloat(a);
  }
  else
  {
    // equal
    return true;
  }
}

bool bool_gtzero_softfloat(const struct softfloat a)
{
  if (bool_isnan_softfloat(a))
  {
    return false;
  }
  else if (bool_signbit_softfloat(a))
  {
    return false;
  }
  else if (bool_iszero_softfloat(a))
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool bool_gezero_softfloat(const struct softfloat a)
{
  if (bool_isnan_softfloat(a))
  {
    return false;
  }
  else if (bool_signbit_softfloat(a))
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool bool_ltzero_softfloat(const struct softfloat a)
{
  if (bool_isnan_softfloat(a))
  {
    return false;
  }
  else if (bool_signbit_softfloat(a))
  {
    return true;
  }
  else
  {
    return false;
  }
}

struct softfloat softfloat_ldexp_softfloat_int(const struct softfloat a, int e)
{
  if (bool_iszero_softfloat(a) || bool_isinf_softfloat(a) || bool_isnan_softfloat(a))
  {
    return a;
  }
  else if (e >= 0x7FFFFFFFU - (long) uint_biasedexponent_softfloat(a))
  {
    // overflow to +/-infinity
    struct softfloat o = { (a.se & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  else if ((long) uint_biasedexponent_softfloat(a) + e <= 0)
  {
    // underfloat to 0
    struct softfloat o = { (a.se & 0x80000000U) | 0U, 0U };
    return o;
  }
  else
  {
    struct softfloat o = { (a.se & 0x80000000U) | (uint_biasedexponent_softfloat(a) + e), a.m };
    return o;
  }
}

struct softfloat softfloat_from_floatexp(struct floatexp f)
{
  return softfloat_ldexp_softfloat_int(softfloat_from_float(f.val), f.exp);
}

struct floatexp floatexp_from_softfloat(struct softfloat s)
{
  uint e = uint_biasedexponent_softfloat(s);
  int de = convert_int_sat((long) e - EXPONENT_BIAS);
  s = softfloat_ldexp_softfloat_int(s, -de);
  return floatexp_ldexp_floatexp_int(floatexp_from_float(float_from_softfloat(s)), de);
}

struct softfloat softfloat_zero()
{
  struct softfloat o = { 0, 0 };
  return o;
}

struct softfloat softfloat_one()
{
  return softfloat_from_float(1.0f);
}

struct softfloat softfloat_abs_softfloat(const struct softfloat a)
{
  struct softfloat o = { a.se & 0x7FFFFFFFU, a.m };
  return o;
}

struct softfloat softfloat_neg_softfloat(const struct softfloat a)
{
  struct softfloat o = { a.se ^ 0x80000000U, a.m };
  return o;
}

int softfloat_sgn_softfloat(const struct softfloat a)
{
  return bool_signbit_softfloat(a) ? -1 : 1; // FIXME
}

struct softfloat softfloat_sqr_softfloat(const struct softfloat a)
{
  if (uint_biasedexponent_softfloat(a) >= ((0x7FFFFFFFU >> 1) + (EXPONENT_BIAS >> 1)))
  {
    // overflow to +infinity
    struct softfloat o = { 0x7FFFFFFFU, bool_isnan_softfloat(a) ? 0xFFFFFFFFU : 0U };
    return o;
  }
  else if (uint_biasedexponent_softfloat(a) <= (EXPONENT_BIAS >> 1) + 1)
  {
    // underflow to +0
    // FIXME handle denormals
    struct softfloat o = { 0U, 0U };
    return o;
  }
  else
  {
    ulong m = a.m;
    uint mantissa = (m * m) >> MANTISSA_BITS;
    uint biased_e = ((a.se & 0x7FFFFFFFU) << 1) - EXPONENT_BIAS;
    if ((mantissa & 0x80000000U) == 0)
    {
      mantissa <<= 1;
      biased_e -= 1;
    }
    assert(0 < biased_e);
    assert(biased_e < 0x7FFFFFFFU);
    assert((mantissa >> (MANTISSA_BITS - 1)) == 1U);
    struct softfloat o = { biased_e, mantissa };
    return o;
  }
}

struct softfloat softfloat_mul_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  if ( bool_isnan_softfloat(a) ||
       bool_isnan_softfloat(b) ||
       (bool_isinf_softfloat(a) && bool_iszero_softfloat(b)) ||
       (bool_iszero_softfloat(a) && bool_isinf_softfloat(b)) ||
       (bool_isinf_softfloat(a) && bool_isinf_softfloat(b) && bool_signbit_softfloat(a) != bool_signbit_softfloat(b))
     )
  {
    // nan
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (bool_iszero_softfloat(a) || bool_iszero_softfloat(b))
  {
    // zero
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0U, 0U };
    return o;
  }
  else if (bool_isinf_softfloat(a) || bool_isinf_softfloat(b) || ((ulong)uint_biasedexponent_softfloat(a) + uint_biasedexponent_softfloat(b)) >= ((ulong)0x7FFFFFFFU + EXPONENT_BIAS))
  {
    // overflow to +/-infinity
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  else if (((ulong)uint_biasedexponent_softfloat(a) + uint_biasedexponent_softfloat(b)) <= (EXPONENT_BIAS + 1))
  {
    // underflow to +/-0
    // FIXME handle denormals
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0U, 0U };
    return o;
  }
  else
  {
    ulong ma = a.m;
    ulong mb = b.m;
    uint mantissa = (ma * mb) >> MANTISSA_BITS;
    uint biased_e = ((a.se & 0x7FFFFFFFU) + (b.se & 0x7FFFFFFFU)) - EXPONENT_BIAS;
    if ((mantissa & 0x80000000U) == 0)
    {
      mantissa <<= 1;
      biased_e -= 1;
    }
    assert(0 < biased_e);
    assert(biased_e < 0x7FFFFFFFU);
    assert((mantissa >> (MANTISSA_BITS - 1)) == 1U);
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | biased_e, mantissa };
    return o;
  }
}

struct softfloat softfloat_div_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  if ( bool_isnan_softfloat(a) ||
       bool_isnan_softfloat(b) ||
       (bool_iszero_softfloat(a) && bool_iszero_softfloat(b)) ||
       (bool_isinf_softfloat(a) && bool_isinf_softfloat(b))
     )
  {
    // nan
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (bool_iszero_softfloat(b))
  {
    // inf
    struct softfloat o = { ((a.se ^ b.se) & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  else
  {
    return softfloat_from_floatexp(floatexp_div_floatexp_floatexp(floatexp_from_softfloat(a), floatexp_from_softfloat(b))); // FIXME
  }
}

struct softfloat add_a_gt_b_gt_0(const struct softfloat a, const struct softfloat b)
{
  // same sign addition, |a| > |b| or same exponent
  uint ea = uint_biasedexponent_softfloat(a);
  uint eb = uint_biasedexponent_softfloat(b);
  ulong ma = a.m;
  ulong mb = b.m;
  assert(ea >= eb);
  assert(bool_signbit_softfloat(a) == bool_signbit_softfloat(b));
  ulong mantissa = ma + (mb >> (ea - eb));
  uint biased_e = ea;
  if (!! (mantissa & 0x100000000LU))
  {
    biased_e += 1;
    mantissa >>= 1;
  }
  if (biased_e >= 0x7FFFFFFFU)
  {
    // overflow to +/-infinity
    struct softfloat o = { (a.se & 0x80000000U) | 0x7FFFFFFFU, 0U };
    return o;
  }
  assert(0 < biased_e);
  assert(biased_e < 0x7FFFFFFFU);
  assert((mantissa >> (MANTISSA_BITS - 1)) == 1U);
  assert((mantissa >> MANTISSA_BITS) == 0U);
  struct softfloat o = { biased_e, (uint)(mantissa) };
  if (bool_signbit_softfloat(a)) return softfloat_neg_softfloat(o); else return o;
}

struct softfloat add_a_gt_0_gt_b(const struct softfloat a, const struct softfloat b)
{
  // opposite sign addition, a > 0 > b, |a| > |b|
  uint ea = uint_biasedexponent_softfloat(a);
  uint eb = uint_biasedexponent_softfloat(b);
  ulong ma = a.m;
  ulong mb = b.m;
  assert(ea > eb);
  assert(! bool_signbit_softfloat(a));
  assert(  bool_signbit_softfloat(b));
  // a > 0 > b, |a| > |b|
  long smantissa = (ma << 1) - ((mb << 1) >> (ea - eb));
  assert(smantissa > 0);
  ulong mantissa = smantissa;
  long biased_e = ea - 1;
  int shift = ((int)(mantissa == 0 ? 64 : clz(mantissa))) - MANTISSA_BITS;
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
    struct softfloat o = { 0x7FFFFFFFU, 0U };
    return o;
  }
  else if (biased_e <= 0)
  {
    // underflow to +0
    struct softfloat o = { 0U, 0U };
    return o;
  }
  assert(0 < biased_e);
  assert(biased_e < 0x7FFFFFFFU);
  assert((mantissa >> (MANTISSA_BITS - 1)) == 1U);
  assert((mantissa >> MANTISSA_BITS) == 0U);
  struct softfloat o = { (uint)(biased_e), (uint)(mantissa) };
  return o;
}

struct softfloat softfloat_add_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  if ( bool_isnan_softfloat(a) ||
       bool_isnan_softfloat(b) ||
       (bool_isinf_softfloat(a) && bool_isinf_softfloat(b) && !!((a.se ^ b.se) & 0x80000000U))
     )
  {
    // nan
    struct softfloat o = { 0x7FFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (bool_iszero_softfloat(a))
  {
    return b;
  }
  else if (bool_iszero_softfloat(b))
  {
    return a;
  }
  else if (bool_isinf_softfloat(a))
  {
    return a;
  }
  else if (bool_isinf_softfloat(b))
  {
    return b;
  }
  else if (((a.se ^ b.se) & 0x80000000U) == 0)
  {
    // same sign addition
    uint ea = uint_biasedexponent_softfloat(a);
    uint eb = uint_biasedexponent_softfloat(b);
    if (ea > eb + MANTISSA_BITS)
    {
      return a;
    }
    else if (eb > ea + MANTISSA_BITS)
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
    uint ea = uint_biasedexponent_softfloat(a);
    uint eb = uint_biasedexponent_softfloat(b);
    if (ea > eb + MANTISSA_BITS)
    {
      return a;
    }
    else if (eb > ea + MANTISSA_BITS)
    {
      return b;
    }
    else if (ea == eb)
    {
      uint ma = a.m;
      uint mb = b.m;
      if (ma > mb)
      {
        uint mantissa = ma - mb;
        uint shift = mantissa == 0 ? MANTISSA_BITS : clz(mantissa);
        mantissa <<= shift;
        if (ea > shift)
        {
          uint biased_e = ea - shift;
          assert(0 < biased_e);
          assert(biased_e < 0x7FFFFFFFU);
          assert((mantissa >> (MANTISSA_BITS - 1)) == 1U);
          struct softfloat o = { biased_e, mantissa };
          if (bool_signbit_softfloat(a)) return softfloat_neg_softfloat(o); else return o;
        }
        else
        {
          // FIXME handle denormals
          struct softfloat o = { 0U, 0U };
          return o;
        }
      }
      else if (mb > ma)
      {
        uint mantissa = mb - ma;
        uint shift = mantissa == 0 ? MANTISSA_BITS : clz(mantissa);
        mantissa <<= shift;
        if (eb > shift)
        {
          uint biased_e = eb - shift;
          assert(0 < biased_e);
          assert(biased_e < 0x7FFFFFFFU);
          assert((mantissa >> (MANTISSA_BITS - 1)) == 1U);
          struct softfloat o = { biased_e, mantissa };
          if (bool_signbit_softfloat(b)) return softfloat_neg_softfloat(o); else return o;
        }
        else
        {
          // FIXME handle denormals
          struct softfloat o = { 0U, 0U };
          return o;
        }
      }
      else
      {
        // cancels to 0
        struct softfloat o = { 0U, 0U };
        return o;
      }
    }
    else if (ea > eb)
    {
      // |a| > |b|
      if (bool_signbit_softfloat(a))
      {
        return softfloat_neg_softfloat(add_a_gt_0_gt_b(softfloat_neg_softfloat(a), softfloat_neg_softfloat(b)));
      }
      else
      {
        return add_a_gt_0_gt_b(a, b);
      }
    }
    else
    {
      // |b| > |a|
      if (bool_signbit_softfloat(b))
      {
        return softfloat_neg_softfloat(add_a_gt_0_gt_b(softfloat_neg_softfloat(b), softfloat_neg_softfloat(a)));
      }
      else
      {
        return add_a_gt_0_gt_b(b, a);
      }
    }
  }
}

struct softfloat softfloat_sub_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  return softfloat_add_softfloat_softfloat(a, softfloat_neg_softfloat(b));
}

int int_cmp_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  return ((int)(bool_lt_softfloat_softfloat(b, a))) - ((int)(bool_lt_softfloat_softfloat(a, b)));
}

struct softfloat diffabs(const struct softfloat c, const struct softfloat d)
{
  int s = int_cmp_softfloat_softfloat(c, softfloat_zero());
  if (s > 0)
  {
    int t = int_cmp_softfloat_softfloat(softfloat_add_softfloat_softfloat(c, d), softfloat_zero());
    if (t >= 0)
    {
      return d;
    }
    else
    {
      return softfloat_neg_softfloat(softfloat_add_softfloat_softfloat(d, softfloat_ldexp_softfloat_int(c, 1)));
    }
  }
  else if (s < 0)
  {
    int t = int_cmp_softfloat_softfloat(softfloat_add_softfloat_softfloat(c, d), softfloat_zero());
    if (t > 0)
    {
      return softfloat_add_softfloat_softfloat(d, softfloat_ldexp_softfloat_int(c, 1));
    }
    else
    {
      return softfloat_neg_softfloat(d);
    }
  }
  return softfloat_abs_softfloat(d);
}

struct softfloat softfloat_sqrt_softfloat(const struct softfloat x) // FIXME verify
{
  return softfloat_from_floatexp(floatexp_sqrt_floatexp(floatexp_from_softfloat(x))); // FIXME
  if (bool_signbit_softfloat(x) || bool_isnan_softfloat(x))
  {
    // nan
    struct softfloat o = { 0xFFFFFFFFU, 0xFFFFFFFFU };
    return o;
  }
  else if (bool_isinf_softfloat(x))
  {
    // +inf
    return x;
  }
  else
  {
    long e = (long) uint_biasedexponent_softfloat(x) - EXPONENT_BIAS;
    struct softfloat y = softfloat_ldexp_softfloat_int(x, convert_int_sat((e & 1) - e));
    return softfloat_ldexp_softfloat_int(softfloat_from_double(sqrt(double_from_softfloat(y))), convert_int_sat((e - (e & 1)) >> 1));
  }
}

struct softfloat softfloat_hypot_softfloat_softfloat(const struct softfloat x, const struct softfloat y)
{
  return softfloat_sqrt_softfloat(softfloat_add_softfloat_softfloat(softfloat_sqr_softfloat(x), softfloat_sqr_softfloat(y)));
}

struct softfloat softfloat_nextafter_softfloat_softfloat(const struct softfloat x, const struct softfloat y) // FIXME verify
{
  int e = convert_int_sat((long) uint_biasedexponent_softfloat(x) - EXPONENT_BIAS - (MANTISSA_BITS - 2));
  struct softfloat ulp = softfloat_ldexp_softfloat_int(softfloat_from_int(1), e);
  if (bool_lt_softfloat_softfloat(y, x))
  {
    return softfloat_sub_softfloat_softfloat(x, ulp);
  }
  else
  {
    return softfloat_add_softfloat_softfloat(x, ulp);
  }
}

struct softfloat softfloat_min_softfloat_softfloat(const struct softfloat a, const struct softfloat b)
{
  return int_cmp_softfloat_softfloat(a, b) <= 0 ? a : b;
}

struct softfloat softfloat_floor_softfloat(const struct softfloat x) // FIXME implement without double
{
  if (uint_biasedexponent_softfloat(x) >= EXPONENT_BIAS + LARGE_EXPONENT)
  {
    // already an integer
    return x;
  }
  else if (uint_biasedexponent_softfloat(x) <= EXPONENT_BIAS - LARGE_EXPONENT)
  {
    // very small
    if (bool_signbit_softfloat(x))
    {
      return softfloat_from_int(-1);
    }
    else
    {
      return softfloat_from_int(0);
    }
  }
  else
  {
    // won't under/overflow
    return softfloat_from_double(floor(double_from_softfloat(x)));
  }
}

struct softfloat softfloat_log_softfloat(const struct softfloat x)
{
  return softfloat_from_floatexp(floatexp_log_floatexp(floatexp_from_softfloat(x))); // FIXME
}

struct softfloat softfloat_exp_softfloat(const struct softfloat x)
{
  return softfloat_from_floatexp(floatexp_exp_floatexp(floatexp_from_softfloat(x))); // FIXME
}

struct softfloat softfloat_sin_softfloat(const struct softfloat x)
{
  return softfloat_from_floatexp(floatexp_sin_floatexp(floatexp_from_softfloat(x))); // FIXME
}

struct softfloat softfloat_cos_softfloat(const struct softfloat x)
{
  return softfloat_from_floatexp(floatexp_cos_floatexp(floatexp_from_softfloat(x))); // FIXME
}

struct softfloat softfloat_atan2_softfloat_softfloat(const struct softfloat x, const struct softfloat y)
{
  return softfloat_from_floatexp(floatexp_atan2_floatexp_floatexp(floatexp_from_softfloat(x), floatexp_from_softfloat(y))); // FIXME
}

#endif
#endif

#if NUMBER_TYPE == 1 || NUMBER_TYPE == 2

#if NUMBER_TYPE == 1 // float
typedef float real;
#else
#if NUMBER_TYPE == 2 // double
typedef double real;
#endif
#endif

#define float_from_real(x) ((float)(x))
#define real_from_float(x) ((real)(x))
#define real_from_int(x) ((real)(x))
#define real_from_long(x) ((real)(x))
#define real_neg_real(x) (-(x))
#define real_sqr_real(x) ((x)*(x))
#define real_mul2_real(x) ((x)*2.0)
#define real_div2_real(x) ((x)*0.5)
#define real_1div_real(x) (1.0/(x))
#define real_add_real_real(x,y) ((x)+(y))
#define real_sub_real_real(x,y) ((x)-(y))
#define real_mul_real_real(x,y) ((x)*(y))
#define real_div_real_real(x,y) ((x)/(y))
#define real_sqrt_real(x) (sqrt((x)))
#define real_abs_real(x) (fabs((x)))
#define real_exp_real(x) (exp((x)))
#define real_log_real(x) (log((x)))
#define real_sin_real(x) (sin((x)))
#define real_cos_real(x) (cos((x)))
#define real_floor_real(x) (floor((x)))
#define real_atan2_real_real(y,x) (atan2((y),(x)))
#define real_hypot_real_real(x,y) (hypot((x),(y)))
#define real_nextafter_real_real(x,y) (nextafter((x),(y)))
#define real_min_real_real(x,y) (min((x),(y)))
#define bool_gtzero_real(x) ((x)>0.0)
#define bool_gezero_real(x) ((x)>=0.0)
#define bool_ltzero_real(x) ((x)<0.0)
#define bool_lt_real_real(x,y) ((x)<(y))
#define bool_isnan_real(x) (isnan((x)))
#define bool_isinf_real(x) (isinf((x)))
#ifdef HAVE_DOUBLE
#define real_twopi() (6.283185307179586)
#define double_from_real(x) ((double)(x))
#else
#define real_twopi() (6.283185307179586f)
#endif
#else
#if NUMBER_TYPE == 4 || NUMBER_TYPE == 5 // floatexp, doubleexp

typedef struct floatexp real;
#define float_from_real(x) (float_from_floatexp((x)))
#define real_from_float(x) (floatexp_from_float((x)))
#define real_from_int(x) (floatexp_from_int((x)))
#define real_from_long(x) (floatexp_from_long((x)))
#define real_neg_real(x) (floatexp_neg_floatexp((x)))
#define real_sqr_real(x) (floatexp_sqr_floatexp((x)))
#define real_mul2_real(x) (floatexp_mul2exp_floatexp_exponent((x), 1))
#define real_div2_real(x) (floatexp_div2exp_floatexp_exponent((x), 1))
#define real_1div_real(x) (floatexp_div_floatexp_floatexp(floatexp_from_int(1), (x)))
#define real_add_real_real(x,y) (floatexp_add_floatexp_floatexp((x),(y)))
#define real_sub_real_real(x,y) (floatexp_sub_floatexp_floatexp((x),(y)))
#define real_mul_real_real(x,y) (floatexp_mul_floatexp_floatexp((x),(y)))
#define real_div_real_real(x,y) (floatexp_div_floatexp_floatexp((x),(y)))
#define real_sqrt_real(x) (floatexp_sqrt_floatexp((x)))
#define real_abs_real(x) (floatexp_abs_floatexp((x)))
#define real_exp_real(x) (floatexp_exp_floatexp((x)))
#define real_log_real(x) (floatexp_log_floatexp((x)))
#define real_sin_real(x) (floatexp_sin_floatexp((x)))
#define real_cos_real(x) (floatexp_cos_floatexp((x)))
#define real_floor_real(x) (floatexp_floor_floatexp((x)))
#define real_atan2_real_real(y,x) (floatexp_atan2_floatexp_floatexp((y),(x)))
#define real_hypot_real_real(x,y) (floatexp_hypot_floatexp_floatexp((x),(y)))
#define real_nextafter_real_real(x,y) (floatexp_nextafter_floatexp_floatexp((x),(y)))
#define real_min_real_real(x,y) (floatexp_min_floatexp_floatexp((x),(y)))
#define bool_gtzero_real(x) (bool_gtzero_floatexp((x)))
#define bool_gezero_real(x) (bool_gezero_floatexp((x)))
#define bool_ltzero_real(x) (bool_ltzero_floatexp((x)))
#define bool_lt_real_real(x,y) (bool_lt_floatexp_floatexp((x),(y)))
#define bool_isnan_real(x) (bool_isnan_floatexp((x)))
#define bool_isinf_real(x) (bool_isinf_floatexp((x)))
#ifdef HAVE_DOUBLE
#define real_twopi() (floatexp_from_mantissa_exponent(6.283185307179586, 0))
#define double_from_real(x) (double_from_floatexp(x))
#else
#define real_twopi() (floatexp_from_mantissa_exponent(6.283185307179586f, 0))
#endif

#else
#if NUMBER_TYPE == 6

typedef struct softfloat real;
#define float_from_real(x) (float_from_softfloat((x)))
#define real_from_float(x) (softfloat_from_float((x)))
#define real_from_int(x) (softfloat_from_int((x)))
#define real_from_long(x) (softfloat_from_long((x)))
#define real_neg_real(x) (softfloat_neg_softfloat((x)))
#define real_sqr_real(x) (softfloat_sqr_softfloat((x)))
#define real_mul2_real(x) (softfloat_ldexp_softfloat_int((x), 1))
#define real_div2_real(x) (softfloat_ldexp_softfloat_int((x), -1))
#define real_1div_real(x) (softfloat_div_softfloat_softfloat(softfloat_from_int(1), (x)))
#define real_add_real_real(x,y) (softfloat_add_softfloat_softfloat((x),(y)))
#define real_sub_real_real(x,y) (softfloat_sub_softfloat_softfloat((x),(y)))
#define real_mul_real_real(x,y) (softfloat_mul_softfloat_softfloat((x),(y)))
#define real_div_real_real(x,y) (softfloat_div_softfloat_softfloat((x),(y)))
#define real_sqrt_real(x) (softfloat_sqrt_softfloat((x)))
#define real_abs_real(x) (softfloat_abs_softfloat((x)))
#define real_exp_real(x) (softfloat_exp_softfloat((x)))
#define real_log_real(x) (softfloat_log_softfloat((x)))
#define real_sin_real(x) (softfloat_sin_softfloat((x)))
#define real_cos_real(x) (softfloat_cos_softfloat((x)))
#define real_floor_real(x) (softfloat_floor_softfloat((x)))
#define real_atan2_real_real(y,x) (softfloat_atan2_softfloat_softfloat((y),(x)))
#define real_hypot_real_real(x,y) (softfloat_hypot_softfloat_softfloat((x),(y)))
#define real_nextafter_real_real(x,y) (softfloat_nextafter_softfloat_softfloat((x),(y)))
#define real_min_real_real(x,y) (softfloat_min_softfloat_softfloat((x),(y)))
#define bool_gtzero_real(x) (bool_gtzero_softfloat((x)))
#define bool_gezero_real(x) (bool_gezero_softfloat((x)))
#define bool_ltzero_real(x) (bool_ltzero_softfloat((x)))
#define bool_lt_real_real(x,y) (bool_lt_softfloat_softfloat((x),(y)))
#define bool_isnan_real(x) (bool_isnan_softfloat((x)))
#define bool_isinf_real(x) (bool_isinf_softfloat((x)))
#ifdef HAVE_DOUBLE
#define real_twopi() (softfloat_from_double(6.283185307179586))
#define double_from_real(x) (double_from_softfloat(x))
#else
#define real_twopi() (softfloat_from_float(6.283185307179586f))
#endif

#else

#error unsupported NUMBER_TYPE; can handle: 1 (float), 2 (double), 4 (floatexp), 5 (doubleexp), 6 (softfloat)

#endif
#endif
#endif

struct mat2
{
  real a, b, c, d;
};

struct complex
{
  real x, y;
};

struct dual
{
  real x; real dx[2];
};

struct complexdual
{
  struct dual x, y;
};

struct blaR2
{
  struct mat2 A, B;
  real r2;
  long l;
};

struct complex complex_mul2_complex(struct complex a)
{
  struct complex r;
  r.x = real_mul2_real(a.x);
  r.y = real_mul2_real(a.y);
  return r;
}

struct complex complex_sqr_complex(struct complex a)
{
  struct complex r;
  r.x = real_sub_real_real(real_sqr_real(a.x), real_sqr_real(a.y));
  r.y = real_mul2_real(real_mul_real_real(a.x, a.y));
  return r;
}

struct complex complex_add_complex_complex(struct complex a, struct complex b)
{
  struct complex r;
  r.x = real_add_real_real(a.x, b.x);
  r.y = real_add_real_real(a.y, b.y);
  return r;
}

struct complex complex_mul_complex_complex(struct complex a, struct complex b)
{
  struct complex r;
  r.x = real_sub_real_real(real_mul_real_real(a.x, b.x), real_mul_real_real(a.y, b.y));
  r.y = real_add_real_real(real_mul_real_real(a.x, b.y), real_mul_real_real(a.y, b.x));
  return r;
}

struct complex complex_div_real_complex(real a, struct complex b)
{
  struct complex r;
  const real d = real_add_real_real(real_sqr_real(b.x), real_sqr_real(b.y));
  r.x = real_mul_real_real(a, real_div_real_real(b.x, d));
  r.y = real_neg_real(real_mul_real_real(a, real_div_real_real(b.y, d)));
  return r;
}

struct dual dual_neg_dual(struct dual a)
{
  struct dual r;
  r.x = real_neg_real(a.x);
  r.dx[0] = real_neg_real(a.dx[0]);
  r.dx[1] = real_neg_real(a.dx[1]);
  return r;
}

struct dual dual_abs_dual(struct dual a)
{
  return bool_ltzero_real(a.x) ? dual_neg_dual(a) : a;
}

struct dual dual_mul_real_dual(real a, struct dual b)
{
  struct dual r;
  r.x = real_mul_real_real(a, b.x);
  r.dx[0] = real_mul_real_real(a, b.dx[0]);
  r.dx[1] = real_mul_real_real(a, b.dx[1]);
  return r;
}

struct dual dual_mul_dual_real(struct dual b, real a)
{
  struct dual r;
  r.x = real_mul_real_real(a, b.x);
  r.dx[0] = real_mul_real_real(a, b.dx[0]);
  r.dx[1] = real_mul_real_real(a, b.dx[1]);
  return r;
}

struct dual dual_add_dual_real(struct dual a, real b)
{
  struct dual r = a;
  r.x = real_add_real_real(r.x, b);
  return r;
}

struct dual dual_add_real_dual(real b, struct dual a)
{
  struct dual r = a;
  r.x = real_add_real_real(r.x, b);
  return r;
}

struct dual dual_sub_dual_real(struct dual a, real b)
{
  struct dual r = a;
  r.x = real_sub_real_real(r.x, b);
  return r;
}

struct dual dual_add_dual_dual(struct dual a, struct dual b)
{
  struct dual r;
  r.x = real_add_real_real(a.x, b.x);
  r.dx[0] = real_add_real_real(a.dx[0], b.dx[0]);
  r.dx[1] = real_add_real_real(a.dx[1], b.dx[1]);
  return r;
}

struct dual dual_sub_dual_dual(struct dual a, struct dual b)
{
  struct dual r;
  r.x = real_sub_real_real(a.x, b.x);
  r.dx[0] = real_sub_real_real(a.dx[0], b.dx[0]);
  r.dx[1] = real_sub_real_real(a.dx[1], b.dx[1]);
  return r;
}

struct dual dual_mul_dual_dual(struct dual a, struct dual b)
{
  struct dual r;
  r.x = real_mul_real_real(a.x, b.x);
  r.dx[0] = real_add_real_real(real_mul_real_real(a.x, b.dx[0]), real_mul_real_real(a.dx[0], b.x));
  r.dx[1] = real_add_real_real(real_mul_real_real(a.x, b.dx[1]), real_mul_real_real(a.dx[1], b.x));
  return r;
}

struct dual dual_exp_dual(struct dual a)
{
  struct dual r;
  r.x = real_exp_real(a.x);
  r.dx[0] = real_mul_real_real(r.x, a.dx[0]);
  r.dx[1] = real_mul_real_real(r.x, a.dx[1]);
  return r;
}

struct dual dual_cos_dual(struct dual a)
{
  struct dual r;
  r.x = real_cos_real(a.x);
  const real d = real_neg_real(real_sin_real(a.x));
  r.dx[0] = real_mul_real_real(d, a.dx[0]);
  r.dx[1] = real_mul_real_real(d, a.dx[1]);
  return r;
}

struct dual dual_sin_dual(struct dual a)
{
  struct dual r;
  r.x = real_sin_real(a.x);
  const real d = real_cos_real(a.x);
  r.dx[0] = real_mul_real_real(d, a.dx[0]);
  r.dx[1] = real_mul_real_real(d, a.dx[1]);
  return r;
}

struct dual dual_diffabs_real_dual(real c, struct dual d)
{
  const real cd = real_add_real_real(c, d.x);
  const struct dual c2d = dual_add_real_dual(real_mul2_real(c), d);
  return bool_gezero_real(c) ? bool_gezero_real(cd) ? d : dual_neg_dual(c2d) : bool_gtzero_real(cd) ? c2d : dual_neg_dual(d);
}

struct complexdual complexdual_add_complex_complexdual(struct complex a, struct complexdual b)
{
  struct complexdual r;
  r.x = dual_add_real_dual(a.x, b.x);
  r.y = dual_add_real_dual(a.y, b.y);
  return r;
}

struct complexdual complexdual_add_complexdual_complex(struct complexdual a, struct complex b)
{
  struct complexdual r;
  r.x = dual_add_dual_real(a.x, b.x);
  r.y = dual_add_dual_real(a.y, b.y);
  return r;
}

struct complexdual complexdual_add_complexdual_complexdual(struct complexdual a, struct complexdual b)
{
  struct complexdual r;
  r.x = dual_add_dual_dual(a.x, b.x);
  r.y = dual_add_dual_dual(a.y, b.y);
  return r;
}

struct complexdual complexdual_mul_complexdual_complex(struct complexdual a, struct complex b)
{
  struct complexdual r;
  r.x = dual_sub_dual_dual(dual_mul_dual_real(a.x, b.x), dual_mul_dual_real(a.y, b.y));
  r.y = dual_add_dual_dual(dual_mul_dual_real(a.y, b.x), dual_mul_dual_real(a.x, b.y));
  return r;
}

struct complexdual complexdual_mul_complex_complexdual(struct complex a, struct complexdual b)
{
  struct complexdual r;
  r.x = dual_sub_dual_dual(dual_mul_real_dual(a.x, b.x), dual_mul_real_dual(a.y, b.y));
  r.y = dual_add_dual_dual(dual_mul_real_dual(a.y, b.x), dual_mul_real_dual(a.x, b.y));
  return r;
}

struct complexdual complexdual_mul_complexdual_complexdual(struct complexdual a, struct complexdual b)
{
  struct complexdual r;
  r.x = dual_sub_dual_dual(dual_mul_dual_dual(a.x, b.x), dual_mul_dual_dual(a.y, b.y));
  r.y = dual_add_dual_dual(dual_mul_dual_dual(a.y, b.x), dual_mul_dual_dual(a.x, b.y));
  return r;
}

struct complexdual complexdual_mul_mat2_complexdual(struct mat2 a, struct complexdual b)
{
  struct complexdual r;
  r.x = dual_add_dual_dual(dual_mul_real_dual(a.a, b.x), dual_mul_real_dual(a.b, b.y));
  r.y = dual_add_dual_dual(dual_mul_real_dual(a.c, b.x), dual_mul_real_dual(a.d, b.y));
  return r;
}

struct complex complex_mul_complex_mat2(struct complex a, struct mat2 b)
{
  struct complex r;
  r.x = real_add_real_real(real_mul_real_real(a.x, b.a), real_mul_real_real(a.y, b.c));
  r.y = real_add_real_real(real_mul_real_real(a.x, b.b), real_mul_real_real(a.y, b.d));
  return r;
}

struct complex complex_mul_mat2_complex(struct mat2 a, struct complex b)
{
  struct complex r;
  r.x = real_add_real_real(real_mul_real_real(a.a, b.x), real_mul_real_real(a.b, b.y));
  r.y = real_add_real_real(real_mul_real_real(a.c, b.x), real_mul_real_real(a.d, b.y));
  return r;
}

real real_norm_complex(struct complex a)
{
  return real_add_real_real(real_sqr_real(a.x), real_sqr_real(a.y));
}

real real_norm_complexdual(struct complexdual a)
{
  return real_add_real_real(real_sqr_real(a.x.x), real_sqr_real(a.y.x));
}

real real_arg_complex(struct complex a)
{
  return real_atan2_real_real(a.y, a.x);
}

void hsv2rgb(float h, float s, float v, float *r, float *g, float *b)
{
  h -= floor(h);
  h *= 6.0f;
  int i = (int) floor(h);
  float f = h - i;
  if (! (i & 1))
  {
    f = 1.0f - f;
  }
  float m = v * (1.0f - s);
  float n = v * (1.0f - s * f);
  switch (i)
  {
    case 6:
    case 0: *r = v; *g = n; *b = m; break;
    case 1: *r = n; *g = v; *b = m; break;
    case 2: *r = m; *g = v; *b = n; break;
    case 3: *r = m; *g = n; *b = v; break;
    case 4: *r = n; *g = m; *b = v; break;
    case 5: *r = v; *g = m; *b = n; break;
    default: *r = 0.0f; *g = 0.0f; *b = 0.0f; break;
  }
}

float srgb2linear(float c)
{
  if (c <= 0.04045f) return c / 12.92;
  return pow((c + 0.055f) / 1.055f, 2.4f);
}

struct config
{
  long config_size;
  long number_type;
  /* shape */
  long height;
  long width;
  long supersampling;
  long subsampling;
  long tile_height;
  long tile_width;
  long subframes;
  long frame;
  /* bailout */
  long Iterations;
  real ER2;
  long PerturbIterations;
  long BLASteps;
  /* transform */
  int transform_exponential_map;
  int transform_vertical_flip;
  struct mat2 transform_K;
  real pixel_spacing;
  real offset_x;
  real offset_y;
  /* ref layout */
  long number_of_phases;
  long degree[MAX_PHASES];
  long ref_size[MAX_PHASES];
  long ref_start[MAX_PHASES];
  /* bla layout */
  long bla_size[MAX_PHASES];
  long bla_levels[MAX_PHASES];
  long bla_start[MAX_PHASES][MAX_LEVELS];
  long bla_size_level[MAX_PHASES][MAX_LEVELS];
};

// http://www.burtleburtle.net/bob/hash/integer.html
uint burtle_hash(uint a)
{
  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165667b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}

float radical_inverse(long a, const long base)
{
  const float one_minus_epsilon = nextafter(1.0f, 0.0f);
  const float base1 = 1.0f / (float)(base);
  long reversed = 0;
  float base1n = 1.0f;
  while (a)
  {
    const long next  = a / base;
    const long digit = a - base * next;
    reversed = reversed * base + digit;
    base1n *= base1;
    a = next;
  }
  return fmin(reversed * base1n, one_minus_epsilon);
}

float wrap(const float v)
{
  return v - floor(v);
}

// <https://github.com/lycium/FractalTracer/blob/32843cded1c4b89f54535a86e3a0ff8faff7cbeb/src/renderer/Renderer.h#L83-L93>
// Convert uniform distribution into triangle-shaped distribution
// From https://www.shadertoy.com/view/4t2SDh
float sign1(float v) { return (v >= 0) ? (float)1 : (float)-1; }
float triangle(float v)
{
  const float orig = v * 2 - 1;
  v = orig / sqrt(fabs(orig));
  v = fmax((float)(-1), v); // Nerf the NaN generated by 0*rsqrt(0). Thanks @FioraAeterna!
  v = v - sign1(orig);
  return v;
}

void jitter(const long width, const long height, const long frame, const long i, const long j, const long k, float *x, float *y)
{
  long ix = (frame * height + j) * width + i;
  float h = (float)(burtle_hash(ix)) / (float)(0x100000000L);
  *x = triangle(wrap(radical_inverse(k, 2) + h));
  *y = triangle(wrap(radical_inverse(k, 3) + h));
}

__global const struct blaR2 *lookup_bla(CONSTANT const struct config *config, __global const struct blaR2 *bla, long phase, long m, real z2)
{
  if (m <= 0)
  {
    return 0;
  }
  if (! (m < config->bla_size[phase]))
  {
    return 0;
  }
  __global const struct blaR2 *ret = 0;
  long ix = m - 1;
  for (long level = 0; level < config->bla_levels[phase]; ++level)
  {
    long ixm = (ix << level) + 1;
    long start = config->bla_start[phase][level];
    if (ix < config->bla_size_level[phase][level])
    {
      if (m == ixm && bool_lt_real_real(z2, bla[start + ix].r2))
      {
        ret = &bla[start + ix];
      }
      else
      {
        break;
      }
    }
    ix = ix >> 1;
  }
  return ret;
}

__kernel void fraktaler3
( CONSTANT const struct config *config
, __global const real *ref
, __global const struct blaR2 *bla
/* accumulate linear RGB */
, __global float *RGB
/* output raw data */
, __global uint *N0
, __global uint *N1
, __global float *NF
, __global float *T
, __global float *DEX
, __global float *DEY
, __global uint *BLA
, __global uint *PTB
, const long y0
, const long x0
, const long subframe
)
{
  const long j = y0 + get_global_id(0);
  const long i = x0 + get_global_id(1);
  if (config->config_size != sizeof(struct config) || config->number_type != NUMBER_TYPE)
  {
    // sanity check
    return;
  }
  if (! (0 <= i && i < config->width && 0 <= j && j < config->height))
  {
    // sanity check
    return;
  }
  if (! (0 <= i - x0 && i - x0 < config->tile_width && 0 <= j - y0 && j - y0 < config->tile_height))
  {
    // sanity check
    return;
  }
  long last_degree = 2;
  long next_degree = 2;
  {
    float di, dj;
    jitter(config->width, config->height, config->frame, i, j, subframe, &di, &dj);
    struct dual u0 = { real_from_float(i + 0.5f + di), { real_from_int(1), real_from_int(0) } };
    struct dual v0 = { real_from_float(j + 0.5f + dj), { real_from_int(0), real_from_int(1) } };
    if (config->transform_exponential_map)
    {
      struct dual re = dual_mul_real_dual(real_div_real_real(real_neg_real(real_log_real(real_from_int(2))), real_from_long(config->height)), v0);
      struct dual im = dual_mul_real_dual(real_div_real_real(real_twopi(), real_from_long(config->width)), u0);
      real R = real_div2_real(real_hypot_real_real(real_from_long(config->width), real_from_long(config->height)));
      struct dual r = dual_exp_dual(re);
      struct dual c = dual_cos_dual(im);
      struct dual s = dual_sin_dual(im);
      u0 = dual_mul_real_dual(R, dual_mul_dual_dual(r, c));
      v0 = dual_mul_real_dual(R, dual_mul_dual_dual(r, s));
    }
    else
    {
      u0 = dual_sub_dual_real(u0, real_div2_real(real_from_long(config->width)));
      v0 = dual_sub_dual_real(v0, real_div2_real(real_from_long(config->height)));
    }
    const struct complex C = { ref[config->ref_start[0] + 2], ref[config->ref_start[0] + 3] }; // FIXME
    struct dual cx = dual_mul_dual_real(u0, config->pixel_spacing);
    struct dual cy = dual_mul_dual_real(v0, config->pixel_spacing);
    struct complexdual c = { cx, cy };
    struct complex offset = { config->offset_x, config->offset_y };
    c = complexdual_add_complexdual_complex(complexdual_mul_mat2_complexdual(config->transform_K, c), offset);
    long phase = 0;
    long m = 0;
    long n = 0;
    long iters_ptb = 0;
    long steps_bla = 0;
    struct complexdual z = { { real_from_int(0), { real_from_int(0), real_from_int(0) } }, { real_from_int(0), { real_from_int(0), real_from_int(0) } } };
    struct complexdual Zz = z;
    real Zz2 = real_norm_complexdual(Zz);
    real z2 = real_norm_complexdual(z);
    while (n < config->Iterations && bool_lt_real_real(Zz2, config->ER2) && iters_ptb < config->PerturbIterations)
    {
      // bla steps
      __global const struct blaR2 *b = 0;
      while (n < config->Iterations && steps_bla < config->BLASteps && bool_lt_real_real(Zz2, config->ER2) && (b = lookup_bla(config, bla, phase, m, z2)))
      {
        z = complexdual_add_complexdual_complexdual(complexdual_mul_mat2_complexdual(b->A, z), complexdual_mul_mat2_complexdual(b->B, c));
        z2 = real_norm_complexdual(z);
        n += b->l;
        m += b->l;
        steps_bla++;

        // rebase
        if (! (n < config->Iterations && steps_bla < config->BLASteps && bool_lt_real_real(Zz2, config->ER2)))
        {
          break;
        }
        if (! (m < config->ref_size[phase]))
        {
          break;
        }
        struct complex Z = { ref[config->ref_start[phase] + 2 * m], ref[config->ref_start[phase] + 2 * m + 1] };
        Zz = complexdual_add_complex_complexdual(Z, z);
        Zz2 = real_norm_complexdual(Zz);
        next_degree = config->degree[n % config->number_of_phases];
        if (bool_lt_real_real(Zz2, real_mul_real_real(real_from_int(next_degree), z2)) || m + 1 == config->ref_size[phase])
        {
          z = Zz;
          z2 = Zz2;
          phase = (phase + m) % config->number_of_phases;
          m = 0;
        }
      }

      // perturbation iteration
      {
        if (! (n < config->Iterations && iters_ptb < config->PerturbIterations && steps_bla < config->BLASteps && bool_lt_real_real(Zz2, config->ER2)))
        {
          break;
        }
        if (! (m < config->ref_size[phase]))
        {
          break;
        }

        next_degree = config->degree[n % config->number_of_phases];
        // z = f(C, Z, c, z)
{
