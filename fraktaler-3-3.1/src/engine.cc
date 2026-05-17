// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <chrono>
#include <thread>
#include <vector>

#include <sys/stat.h>

#include <mpreal.h>
//#include <toml.hpp>

#include "bla.h"
//#include "colour.h"
#include "display.h"
#include "engine.h"
#include "floatexp.h"
#include "hybrid.h"
#include "main.h"
//#include "map.h"
#include "param.h"
#include "softfloat.h"
//#include "stats.h"
#include "types.h"

const char *nt_string[
#ifdef HAVE_FLOAT128
  8
#else
  7
#endif
] = { "none", "float", "double", "long double", "floatexp", "doubleexp", "softfloat"
#ifdef HAVE_FLOAT128
  , "float128"
#endif
};

number_type nt_ref = nt_none;
number_type nt_bla = nt_none;


#ifdef HAVE_FLOAT128
std::vector<std::vector<complex<float128>>> Zq;
#endif
std::vector<std::vector<complex<softfloat>>> Zsf;
std::vector<std::vector<complex<floatexp<float, int>>>> Zfe;
std::vector<std::vector<complex<floatexp<double, int>>>> Zde;
std::vector<std::vector<complex<long double>>> Zld;
std::vector<std::vector<complex<double>>> Zd;
std::vector<std::vector<complex<float>>> Zf;

void delete_ref()
{
#ifdef HAVE_FLOAT128
  std::vector<std::vector<complex<float128>>>().swap(Zq);
#endif
  std::vector<std::vector<complex<softfloat>>>().swap(Zsf);
  std::vector<std::vector<complex<floatexp<double, int>>>>().swap(Zde);
  std::vector<std::vector<complex<floatexp<float, int>>>>().swap(Zfe);
  std::vector<std::vector<complex<long double>>>().swap(Zld);
  std::vector<std::vector<complex<double>>>().swap(Zd);
  std::vector<std::vector<complex<float>>>().swap(Zf);
  nt_ref = nt_none;
}

#ifdef HAVE_FLOAT128
std::vector<blasR2<float128>> Bq;
#endif
std::vector<blasR2<softfloat>> Bsf;
std::vector<blasR2<floatexp<double, int>>> Bde;
std::vector<blasR2<floatexp<float, int>>> Bfe;
std::vector<blasR2<long double>> Bld;
std::vector<blasR2<double>> Bd;
std::vector<blasR2<float>> Bf;

void delete_bla()
{
#ifdef HAVE_FLOAT128
  std::vector<blasR2<float128>>().swap(Bq);
#endif
  std::vector<blasR2<softfloat>>().swap(Bsf);
  std::vector<blasR2<floatexp<double, int>>>().swap(Bde);
  std::vector<blasR2<floatexp<float, int>>>().swap(Bfe);
  std::vector<blasR2<long double>>().swap(Bld);
  std::vector<blasR2<double>>().swap(Bd);
  std::vector<blasR2<float>>().swap(Bf);
  nt_bla = nt_none;
}

count_t getM(number_type nt, count_t phase)
{
  switch (nt)
  {
    case nt_none:
      return 0;
    case nt_float:
      return Zf[phase].size();
    case nt_double:
      return Zd[phase].size();
    case nt_longdouble:
      return Zld[phase].size();
    case nt_floatexp:
      return Zfe[phase].size();
    case nt_doubleexp:
      return Zde[phase].size();
    case nt_softfloat:
      return Zsf[phase].size();
#ifdef HAVE_FLOAT128
    case nt_float128:
      return Zq[phase].size();
#endif
  }
  return 0;
}

extern floatexp<float, int> newton_relative_start; // FIXME this is defined in gui.cc
void newton_thread(param &out, bool &ok, const param &par, const complex<floatexp<float, int>> &c, const floatexp<float, int> &r, volatile progress_t *progress, volatile bool *running, volatile bool *ended)
{
  using std::exp;
  using ::exp;
  using std::log;
  using ::log;
  count_t period = par.p.reference.period;
  mpfr_prec_t prec = std::max(mpfr_get_prec(par.reference.x.mpfr_ptr()), mpfr_get_prec(par.reference.y.mpfr_ptr()));
  mpreal::set_default_prec(prec);
  complex<mpreal> center = par.reference;
  const pnewton &newton = par.p.newton;
  if (*running && newton.action >= newton_action_period)
  {
    switch (nt_ref)
    {
      case nt_none: period = 0; break;
      case nt_float: period = hybrid_period(par.opss, Zf, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
      case nt_double: period = hybrid_period(par.opss, Zd, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
      case nt_longdouble: period = hybrid_period(par.opss, Zld, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
      case nt_floatexp: period = hybrid_period(par.opss, Zfe, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
      case nt_doubleexp: period = hybrid_period(par.opss, Zde, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
      case nt_softfloat: period = hybrid_period(par.opss, Zsf, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
#ifdef HAVE_FLOAT128
      case nt_float128: period = hybrid_period(par.opss, Zq, c, par.p.bailout.iterations, r, par.transform, &progress[0], running); break;
#endif
    }
  }
  ok = *running && period > 0;
  if (*running && ok && newton.action >= newton_action_center)
  {
    prec *= 3; // FIXME verify how much overkill this is
    prec += 24;
    mpreal::set_default_prec(prec);
    center.x.set_prec(prec);
    center.y.set_prec(prec);
    center.x += mpreal(c.x.val) << c.x.exp;
    center.y += mpreal(c.y.val) << c.y.exp;
    ok = hybrid_center(par.opss, center, period, &progress[1], running);
  }
  floatexp<float, int> size0 = 1 / par.zoom;
  floatexp<float, int> size = size0;
  mat2<double> transform = par.transform;
  if (*running && ok && newton.action >= newton_action_zoom)
  {
#if 0
    if (newton.domain)
    {
      ok = hybrid_domain_size(size, transform, par.opss, center, period, &progress[3], running);
    }
    else
#endif
    {
      ok = hybrid_size(size, transform, par.opss, par.degrees, center, period, &progress[3], running);
    }
    ok &= 10 * size0 > size && size > r * r * r; // safety check
  }
  out = par;
  ok &= *running;
  if (ok)
  {
    if (newton.action >= newton_action_period)
    {
      out.p.reference.period = period;
    }
    if (newton.action >= newton_action_zoom)
    {
      if (newton.absolute)
      {
        out.zoom = exp(log(1 / size) * newton.power) / newton.factor;
      }
      else
      {
        floatexp<float, int> zoom = newton_relative_start; // if this is 1, formula evaluates same as absolute
        out.zoom = exp(log(zoom) + (log(1 / size) - log(zoom)) * newton.power) / newton.factor;
      }
      prec = 24 + floatexp<float, int>(out.zoom).exp;
      if (prec < 24) prec = 24;
      mpreal::set_default_prec(prec);
      out.reference.x.set_prec(prec);
      out.reference.y.set_prec(prec);
      out.center.x.set_prec(prec);
      out.center.y.set_prec(prec);
    }
    if (newton.action >= newton_action_center)
    {
      out.reference = center;
      out.center = center;
    }
    if (newton.action >= newton_action_transform)
    {
      out.transform = transform;
      unstring_vals(out);
      out.p.transform.rotate = 0; // FIXME figure out why this hack is necessry
      restring_vals(out);
      out.p.reference.period = period;
    }
    restring_locs(out);
    restring_vals(out);
  }
  *ended = true;
}

bool just_did_newton = false;

template <typename T>
bool calculate_reference(std::vector<std::vector<complex<T>>> &Z, const param &par, progress_t *progress, volatile bool *running)
{
  count_t maximum_reference_iterations = par.p.bailout.maximum_reference_iterations;
  if (par.p.algorithm.lock_maximum_reference_iterations_to_period && par.p.reference.period > 0)
  {
    maximum_reference_iterations = par.p.reference.period + 1;
  }
  Z.resize(par.opss.size());
  for (count_t phase = 0; phase < (int) par.opss.size(); ++phase)
  {
    Z[phase].resize(maximum_reference_iterations);
  }
  mat2<T> K(T(par.transform.x[0][0]), T(par.transform.x[0][1]), T(par.transform.x[1][0]), T(par.transform.x[1][1]));
  K = K * T(4 / par.zoom);
  hybrid_references(Z, par.opss, maximum_reference_iterations, par.reference, K, &progress[0], running);
  return *running;
}

void set_reference_to_image_center(param &par)
{
  par.reference.x.set_prec(par.center.x.get_prec());
  par.reference.y.set_prec(par.center.y.get_prec());
  par.reference = par.center;
  delete_ref();
}

bool calculate_reference(number_type nt, const param &par, progress_t *progress, volatile bool *running)
{
  delete_ref();
  switch (nt)
  {
    case nt_float: return calculate_reference(Zf, par, progress, running);
    case nt_double: return calculate_reference(Zd, par, progress, running);
    case nt_longdouble: return calculate_reference(Zld, par, progress, running);
    case nt_floatexp: return calculate_reference(Zfe, par, progress, running);
    case nt_doubleexp: return calculate_reference(Zde, par, progress, running);
    case nt_softfloat: return calculate_reference(Zsf, par, progress, running);
#ifdef HAVE_FLOAT128
    case nt_float128: return calculate_reference(Zq, par, progress, running);
#endif
    default: return false;
  }
}

bool calculate_bla(number_type nt, const param &par, progress_t *progress, volatile bool *running)
{
  using std::max;
  complex<mpreal> offset;
  offset.x.set_prec(par.center.x.get_prec());
  offset.y.set_prec(par.center.y.get_prec());
  offset = par.center - par.reference;
  const floatexp<float, int> pixel_spacing =
    4 / par.zoom / (par.p.image.height / par.p.image.subsampling);
  const floatexp<float, int> pixel_precision = max
    ( max(abs(floatexp<float, int>(offset.x) / pixel_spacing)
        , abs(floatexp<float, int>(offset.y) / pixel_spacing))
    , hypot(floatexp<float, int>(par.p.image.width / par.p.image.subsampling)
          , floatexp<float, int>(par.p.image.height / par.p.image.subsampling))
    );
  const floatexp<float, int> c = pixel_spacing * pixel_precision;
  const floatexp<float, int> e = 1.0 / (count_t(1) << 24); // FIXME hardcoded 24 is not enough in all circumstances, but floatexp is currently using single precision...
  delete_bla();
  switch (nt)
  {
    case nt_float: return hybrid_blas(Bf, Zf, par.opss, par.degrees, float(c), float(e), par.p.algorithm.bla_skip_levels, progress, running);
    case nt_double: return hybrid_blas(Bd, Zd, par.opss, par.degrees,  double(c), double(e), par.p.algorithm.bla_skip_levels, progress, running);
    case nt_longdouble: return hybrid_blas(Bld, Zld, par.opss, par.degrees, (long double)(c), (long double)(e), par.p.algorithm.bla_skip_levels, progress, running);
    case nt_floatexp: return hybrid_blas(Bfe, Zfe, par.opss, par.degrees, floatexp<float, int>(c), floatexp<float, int>(e), par.p.algorithm.bla_skip_levels, progress, running);;
    case nt_doubleexp: return hybrid_blas(Bde, Zde, par.opss, par.degrees, floatexp<double, int>(c), floatexp<double, int>(e), par.p.algorithm.bla_skip_levels, progress, running);;
    case nt_softfloat: return hybrid_blas(Bsf, Zsf, par.opss, par.degrees, softfloat(c), softfloat(e), par.p.algorithm.bla_skip_levels, progress, running);
#ifdef HAVE_FLOAT128
    case nt_float128: return hybrid_blas(Bq, Zq, par.opss, par.degrees, float128(c), float128(e), par.p.algorithm.bla_skip_levels, progress, running);
#endif
    default: return false;
  }
}
