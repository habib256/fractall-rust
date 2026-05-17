// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <string>

#include "gles2.h"

#include <glm/glm.hpp>
#include <mpreal.h>

typedef int64_t coord_t;

typedef float smooth_t;
typedef int64_t count_t;

typedef float progress_t;

template <typename mantissa, typename exponent>
struct floatexp;

struct softfloat;

template <typename real> struct complex;

template <int D, typename T> struct dual;

template <typename real> struct mat2;

typedef int channel_bit_t;
typedef int channel_mask_t;
struct map;

struct colour;
struct stats;
struct param;
struct wlookup;

template <typename real> struct blaR2;
template <typename real> struct blasR2;

// update list of names in engine.cc when changing this
// update list of precisions in wisdom.cc when changing this
enum number_type
{ nt_none = 0
, nt_float = 1
, nt_double = 2
, nt_longdouble = 3
, nt_floatexp = 4
, nt_doubleexp = 5
, nt_softfloat = 6
#ifdef HAVE_FLOAT128
, nt_float128 = 7
#endif
};

extern const char *nt_string[
#ifdef HAVE_FLOAT128
  8
#else
  7
#endif
];

using mpreal = mpfr::mpreal;

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using mat3 = glm::mat3;

#if __cplusplus >= 202002L
#define CONSTEXPR constexpr
#define CONSTEXPR_STATIC constexpr
#else
#define CONSTEXPR
#define CONSTEXPR_STATIC const
#endif

template<typename T> T convert(const mpreal &m) noexcept
{
  return T(m);
}

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

inline void syncfs(void)
{
#ifdef __EMSCRIPTEN__
  EM_ASM(
    FS.syncfs(false, function (err) {
      /* ignore error, don't wait for done */
    });
  );
#endif
}

enum opcode_tag
{ op_add = 0
, op_store = 1
, op_mul = 2
, op_sqr = 3
, op_absx = 4
, op_absy = 5
, op_negx = 6
, op_negy = 7
, op_rot = 8
};
#define op_count 9

extern const char * const op_string[op_count];

struct opcode
{
  opcode_tag op;
  union
  {
    struct
    {
      float x;
      float y;
    } rot;
  } u;
};

inline bool operator==(const opcode &a, const opcode &b)
{
  if (a.op != b.op)
  {
    return false;
  }
  switch (a.op)
  {
    case op_add:
    case op_store:
    case op_mul:
    case op_sqr:
    case op_absx:
    case op_absy:
    case op_negx:
    case op_negy:
      return true;
    case op_rot:
      return a.u.rot.x == b.u.rot.x && a.u.rot.y == b.u.rot.y;
  }
  return false;
}

inline bool starts_with(const std::string &haystack, const std::string &needle)
{
#if __cplusplus >= 202002L
  return haystack.starts_with(needle);
#else
  // https://stackoverflow.com/a/40441240
  return haystack.rfind(needle, 0) == 0;
#endif
}

inline bool ends_with(const std::string &haystack, const std::string &needle)
{
#if __cplusplus >= 202002L
  return haystack.ends_with(needle);
#else
  // https://stackoverflow.com/a/42844629
  return haystack.size() >= needle.size() && 0 == haystack.compare(haystack.size() - needle.size(), needle.size(), needle);
#endif
}
