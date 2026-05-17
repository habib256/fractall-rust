// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <cmath>
#include <thread>

#include "bla.h"
#include "floatexp.h"
#include "hybrid.h"
#include "parallel.h"
#include "render.h"
#include "softfloat.h"
//#include "stats.h"

template <typename t>
bool hybrid_blas(std::vector<blasR2<t>> &B, const std::vector<std::vector<complex<t>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, t c, t e, int skip_levels, volatile progress_t *progress, volatile bool *running)
{
  count_t count = opss.size();
  for (count_t phase = 0; phase < count; ++phase)
  {
    B.push_back(std::move(blasR2calc<t>(Z[phase], opss, degrees, phase, c, e, skip_levels, &progress[phase], running).data));
  }
  return *running;
}

template bool hybrid_blas(std::vector<blasR2<float>> &B, const std::vector<std::vector<complex<float>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, float c, float e, int skip_levels, volatile progress_t *progress, volatile bool *running);
template bool hybrid_blas(std::vector<blasR2<double>> &B, const std::vector<std::vector<complex<double>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, double c, double e, int skip_levels, volatile progress_t *progress, volatile bool *running);
template bool hybrid_blas(std::vector<blasR2<long double>> &B, const std::vector<std::vector<complex<long double>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, long double c, long double e, int skip_levels, volatile progress_t *progress, volatile bool *running);
template bool hybrid_blas(std::vector<blasR2<floatexp<float, int>>> &B, const std::vector<std::vector<complex<floatexp<float, int>>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, floatexp<float, int> c, floatexp<float, int> e, int skip_levels, volatile progress_t *progress, volatile bool *running);
template bool hybrid_blas(std::vector<blasR2<floatexp<double, int>>> &B, const std::vector<std::vector<complex<floatexp<double, int>>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, floatexp<double, int> c, floatexp<double, int> e, int skip_levels, volatile progress_t *progress, volatile bool *running);
template bool hybrid_blas(std::vector<blasR2<softfloat>> &B, const std::vector<std::vector<complex<softfloat>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, softfloat c, softfloat e, int skip_levels, volatile progress_t *progress, volatile bool *running);
#ifdef HAVE_FLOAT128
template bool hybrid_blas(std::vector<blasR2<float128>> &B, const std::vector<std::vector<complex<float128>>> &Z, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, float128 c, float128 e, int skip_levels, volatile progress_t *progress, volatile bool *running);
#endif

template<typename t> t mpfr_get(const mpfr_t x, const mpfr_rnd_t rnd);
template<> float mpfr_get<float>(const mpfr_t x, const mpfr_rnd_t rnd) { return mpfr_get_d(x, rnd); }
template<> double mpfr_get<double>(const mpfr_t x, const mpfr_rnd_t rnd) { return mpfr_get_d(x, rnd); }
template<> long double mpfr_get<long double>(const mpfr_t x, const mpfr_rnd_t rnd) { return mpfr_get_ld(x, rnd); }
template<> floatexp<float, int> mpfr_get<floatexp<float, int>>(const mpfr_t x, const mpfr_rnd_t rnd)
{
  signed long e;
  double m = mpfr_get_d_2exp(&e, x, rnd);
  return floatexp<float, int>(m, e);
}
template<> floatexp<double, int> mpfr_get<floatexp<double, int>>(const mpfr_t x, const mpfr_rnd_t rnd)
{
  signed long e;
  double m = mpfr_get_d_2exp(&e, x, rnd);
  return floatexp<double, int>(m, e);
}
template<> softfloat mpfr_get<softfloat>(const mpfr_t x, const mpfr_rnd_t rnd)
{
  signed long e;
  double m = mpfr_get_d_2exp(&e, x, rnd);
  return ldexp(softfloat(m), e);
}
#ifdef HAVE_FLOAT128
template<> float128 mpfr_get<float128>(const mpfr_t x, const mpfr_rnd_t rnd) { return mpfr_get_float128(x, rnd); }
#endif

template <typename t>
count_t hybrid_reference(complex<t> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C0, const mat2<t> &radius, volatile progress_t *progress, volatile bool *running)
{
  mpfr_prec_t prec = C0.x.getPrecision();
  mpfr_t Z_x, Z_y, Z_stored_x, Z_stored_y, T_1, T_2;
  mpfr_init2(Z_x, prec);
  mpfr_init2(Z_y, prec);
  mpfr_init2(Z_stored_x, prec);
  mpfr_init2(Z_stored_y, prec);
  mpfr_init2(T_1, prec);
  mpfr_init2(T_2, prec);
  count_t M = MaxRefIters;
  // calculate reference in high precision
  mpfr_set_ui(Z_x, 0, MPFR_RNDN);
  mpfr_set_ui(Z_y, 0, MPFR_RNDN);
  // calculate derivative in low precison
  mat2<t> dZdC = mat2<t>(0,0,0,0);
  mat2<t> dZdC_stored = dZdC;
  mat2<t> one = mat2<t>(1, 0, 0, 1);
  for (count_t i = 0; i < MaxRefIters; ++i)
  {
    // store low precision orbit
    Zp[i] = complex<t>(mpfr_get<t>(Z_x, MPFR_RNDN), mpfr_get<t>(Z_y, MPFR_RNDN));
    // escape check
    if (! (norm(Zp[i]) < 1e10) || ! *running) // FIXME escape radius
    {
      M = i;
      break;
    }
    // periodicity check
    if (0 < i && abs(inverse(radius * dZdC) * Zp[i]) < 1)
    {
      M = i + 1;
      break;
    }
    // step
    const auto & ops = opss[(phase + i) % opss.size()];
    for (const auto & op : ops)
    {
      switch (op.op)
      {
        case op_add:
          dZdC += one;
          mpfr_add(Z_x, Z_x, C0.x.mpfr_srcptr(), MPFR_RNDN);
          mpfr_add(Z_y, Z_y, C0.y.mpfr_srcptr(), MPFR_RNDN);
          break;
        case op_store:
          dZdC_stored = dZdC;
          mpfr_set(Z_stored_x, Z_x, MPFR_RNDN);
          mpfr_set(Z_stored_y, Z_y, MPFR_RNDN);
          break;
        case op_sqr:
        {
          complex<t> Z(mpfr_get<t>(Z_x, MPFR_RNDN), mpfr_get<t>(Z_y, MPFR_RNDN));
          dZdC = mat2<t>(2 * Z) * dZdC;
          mpfr_add(T_1, Z_x, Z_y, MPFR_RNDN);
          mpfr_sub(T_2, Z_x, Z_y, MPFR_RNDN);
          mpfr_mul(Z_y, Z_x, Z_y, MPFR_RNDN);
          mpfr_mul_2ui(Z_y, Z_y, 1, MPFR_RNDN);
          mpfr_mul(Z_x, T_1, T_2, MPFR_RNDN);
          break;
        }
        case op_mul:
        {
          complex<t> Z(mpfr_get<t>(Z_x, MPFR_RNDN), mpfr_get<t>(Z_y, MPFR_RNDN));
          complex<t> Z_stored(mpfr_get<t>(Z_stored_x, MPFR_RNDN), mpfr_get<t>(Z_stored_y, MPFR_RNDN));
          dZdC = mat2<t>(Z) * dZdC_stored + mat2<t>(Z_stored) * dZdC;
          mpfr_mul(T_1, Z_x, Z_stored_x, MPFR_RNDN);
          mpfr_mul(T_2, Z_y, Z_stored_y, MPFR_RNDN);
          mpfr_mul(Z_x, Z_x, Z_stored_y, MPFR_RNDN);
          mpfr_mul(Z_y, Z_y, Z_stored_x, MPFR_RNDN);
          mpfr_add(Z_y, Z_x, Z_y, MPFR_RNDN);
          mpfr_sub(Z_x, T_1, T_2, MPFR_RNDN);
          break;
        }
        case op_absx:
          if (mpfr_sgn(Z_x) < 0)
          {
            dZdC = mat2<t>(-1,0,0,1) * dZdC;
          }
          mpfr_abs(Z_x, Z_x, MPFR_RNDN);
          break;
        case op_absy:
          if (mpfr_sgn(Z_y) < 0)
          {
            dZdC = mat2<t>(1,0,0,-1) * dZdC;
          }
          mpfr_abs(Z_y, Z_y, MPFR_RNDN);
          break;
        case op_negx:
          dZdC = mat2<t>(-1,0,0,1) * dZdC;
          mpfr_neg(Z_x, Z_x, MPFR_RNDN);
          break;
        case op_negy:
          dZdC = mat2<t>(1,0,0,-1) * dZdC;
          mpfr_neg(Z_y, Z_y, MPFR_RNDN);
          break;
        case op_rot:
          dZdC = mat2<t>(complex<t>(op.u.rot.x, op.u.rot.y)) * dZdC;
          mpfr_mul_d(T_1, Z_x, op.u.rot.x, MPFR_RNDN);
          mpfr_mul_d(T_2, Z_y, op.u.rot.y, MPFR_RNDN);
          mpfr_mul_d(Z_x, Z_x, op.u.rot.y, MPFR_RNDN);
          mpfr_mul_d(Z_y, Z_y, op.u.rot.x, MPFR_RNDN);
          mpfr_add(Z_y, Z_x, Z_y, MPFR_RNDN);
          mpfr_sub(Z_x, T_1, T_2, MPFR_RNDN);
          break;
      }
    }
    *progress = (i + 1) / progress_t(MaxRefIters);
  }
  mpfr_clear(Z_x);
  mpfr_clear(Z_y);
  mpfr_clear(Z_stored_x);
  mpfr_clear(Z_stored_y);
  mpfr_clear(T_1);
  mpfr_clear(T_2);
  return M;
}

template count_t hybrid_reference(complex<float> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<float> &radius, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_reference(complex<double> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<double> &radius, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_reference(complex<long double> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<long double> &radius, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_reference(complex<floatexp<float, int>> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<floatexp<float, int>> &radius, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_reference(complex<softfloat> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<softfloat> &radius, volatile progress_t *progress, volatile bool *running);
#ifdef HAVE_FLOAT128
template count_t hybrid_reference(complex<float128> *Zp, const std::vector<std::vector<opcode>> &opss, const count_t &phase, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<float128> &radius, volatile progress_t *progress, volatile bool *running);
#endif

template <typename t>
void hybrid_references(std::vector<std::vector<complex<t>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<t> &radius, volatile progress_t *progress, volatile bool *running)
{
  parallel1d(std::thread::hardware_concurrency(), 0, opss.size(), 1, running, [&](count_t phase)
  {
    count_t M = hybrid_reference(&Zp[phase][0], opss, phase, MaxRefIters, C, radius, &progress[phase], running);
    Zp[phase].resize(M);
  });
}

template void hybrid_references(std::vector<std::vector<complex<float>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<float> &radius, volatile progress_t *progress, volatile bool *running);
template void hybrid_references(std::vector<std::vector<complex<double>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<double> &radius, volatile progress_t *progress, volatile bool *running);
template void hybrid_references(std::vector<std::vector<complex<long double>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<long double> &radius, volatile progress_t *progress, volatile bool *running);
template void hybrid_references(std::vector<std::vector<complex<floatexp<float, int>>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<floatexp<float, int>> &radius, volatile progress_t *progress, volatile bool *running);
template void hybrid_references(std::vector<std::vector<complex<floatexp<double, int>>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<floatexp<double, int>> &radius, volatile progress_t *progress, volatile bool *running);
template void hybrid_references(std::vector<std::vector<complex<softfloat>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<softfloat> &radius, volatile progress_t *progress, volatile bool *running);
#ifdef HAVE_FLOAT128
template void hybrid_references(std::vector<std::vector<complex<float128>>> &Zp, const std::vector<std::vector<opcode>> &opss, const count_t &MaxRefIters, const complex<mpreal> &C, const mat2<float128> &radius, volatile progress_t *progress, volatile bool *running);
#endif

template <typename real>
bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<real>>> &Zp, const std::vector<blasR2<real>> &bla, volatile bool *running)
{
  const std::vector<std::vector<opcode>> &opss = par.opss;
  complex<mpreal> moffset;
  moffset.x.set_prec(par.center.x.get_prec());
  moffset.y.set_prec(par.center.y.get_prec());
  moffset = par.center - par.reference;
  const complex<real> offset(mpfr_get<real>(moffset.x.mpfr_srcptr(), MPFR_RNDN), mpfr_get<real>(moffset.y.mpfr_srcptr(), MPFR_RNDN));
#define normx(w) norm(complex<real>((w).x.x, (w).y.x))
  using std::isinf;
  using std::isnan;
  using std::log;
  using std::max;
  using std::min;
  const coord_t width  = (par.p.image.width  * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling;
  const coord_t height = (par.p.image.height * par.p.image.supersampling + par.p.image.subsampling - 1) / par.p.image.subsampling;
  const count_t Iterations = par.p.bailout.iterations;
  const count_t PerturbIterations = par.p.bailout.maximum_perturb_iterations;
  const count_t BLASteps = par.p.bailout.maximum_bla_steps;
  const real ER2 = par.p.bailout.escape_radius * par.p.bailout.escape_radius;
  const real IR = par.p.bailout.inscape_radius;
  const real pixel_spacing = real(4 / par.zoom / height);
  const mat2<real> K (real(par.transform.x[0][0]), real(par.transform.x[0][1]), real(par.transform.x[1][0]), real(par.transform.x[1][1]));
  const mat2<float> Kf (float(par.transform.x[0][0]), float(par.transform.x[0][1]), float(par.transform.x[1][0]), float(par.transform.x[1][1]));
  for (coord_t j = y0; j < y1 && *running; ++j)
  for (coord_t i = x0; i < x1 && *running; ++i)
  {
    count_t iters_ptb = 1;
    count_t steps_bla = 0;
    double di, dj;
    jitter(width, height, frame, i, j, subframe, di, dj);
    dual<4, real> u0(real(i + 0.5 + di)); u0.dx[0] = real(1);
    dual<4, real> v0(real(j + 0.5 + dj)); v0.dx[1] = real(1);
    if (par.p.transform.exponential_map)
    {
      auto re = (-0.6931471805599453 / height) * v0; // log 2
      auto im = (6.283185307179586 / width) * u0; // 2 pi
      auto R = 0.5 * std::hypot(width, height);
      auto r = exp(re);
      auto c = cos(im);
      auto s = sin(im);
      u0 = R * r * c;
      v0 = R * r * s;
    }
    else
    {
      u0 -= real(width / 2.0);
      v0 -= real(height / 2.0);
    }
    dual<4, real> cx (u0 * pixel_spacing);
    dual<4, real> cy (v0 * pixel_spacing);
    const complex<real> C (Zp[0][1]); // FIXME
    complex<dual<4, real>> c (cx, cy);
    c = K * c + complex<dual<4, real>>(dual<4, real>(offset.x), dual<4, real>(offset.y));
    count_t phase = 0;
    count_t m = 1;
    count_t n = 1;
    complex<real> Z (Zp[phase][m]);
    complex<dual<4, real>> z (c);
    z.x.dx[2] = real(1);
    z.y.dx[3] = real(1);
    real z2 (normx(z));
    complex<dual<4, real>> Zz (Z + z);
    real Zz2 (normx(Zz));
    real dZ (sup(mat2<real>(Zz.x.dx[2], Zz.x.dx[3], Zz.y.dx[2], Zz.y.dx[3])));
    int last_degree = 2;
    while
      ( n < Iterations &&
        Zz2 < ER2 &&
        IR  < dZ &&
        iters_ptb < PerturbIterations &&
        steps_bla < BLASteps
      )
    {
      // bla steps
      const blaR2<real> *b = 0;
      do
      {
        if (! (n < Iterations)) break;
        if (! (Zz2 < ER2)) break;
        if (! (IR < dZ)) break;
        if (! (iters_ptb < PerturbIterations)) break;
        if (! (steps_bla < BLASteps)) break;
        // rebase
        Z = Zp[phase][m];
        Zz = Z + z;
        dZ = sup(mat2<real>(Zz.x.dx[2], Zz.x.dx[3], Zz.y.dx[2], Zz.y.dx[3]));
        Zz2 = normx(Zz);
        z2 = normx(z);
        if (Zz2 < z2 || m + 1 == count_t(Zp[phase].size()))
        {
          z = Zz;
          z2 = Zz2;
          phase = (phase + m) % Zp.size();
          m = 0;
          Z = 0;
        }
        // bla step
        b = bla[phase].lookup(m, z2);
        if (b)
        {
          const mat2<real> A = b->A;
          const mat2<real> B = b->B;
          count_t l = b->l;
          z = A * z + B * c;
          z2 = normx(z);
          n += l;
          m += l;
          steps_bla++;
        }
      } while (b);
      if (! (n < Iterations)) break;
      if (! (Zz2 < ER2)) break;
      if (! (IR < dZ)) break;
      if (! (iters_ptb < PerturbIterations)) break;
      if (! (steps_bla < BLASteps)) break;
      // already rebased here by bla steps loop
      // perturbation iteration
      // z = f(C, Z, c, z)
      int w = n % opss.size();
      bool rebased = false;
      z = hybrid_perturb(opss[w], C, Zp[phase][m], c, z, rebased);
      last_degree = par.degrees[w];
      if (rebased)
      {
        phase = (phase + m) % Zp.size();
        m = 0;
      }
      n++;
      m++;
      iters_ptb++;
    }

    // compute output
    complex<double> Z1 = complex<double>(double(Zz.x.x), double(Zz.y.x));
    mat2<double> J (double(Zz.x.dx[0]), double(Zz.x.dx[1]), double(Zz.y.dx[0]), double(Zz.y.dx[1]));
    complex<double> dC = Z1 * J;
    complex<double> de = (double(par.p.image.subsampling) / double(par.p.image.supersampling)) * norm(Z1) * log(abs(Z1)) / dC;
    float nf = float(std::min(std::max(1 - log(log(norm(Z1)) / log(double(ER2))) / log(double(last_degree)), 0.), 1.));
    float t = float(arg(Z1)) / (2.0f * 3.141592653f);
    t -= floor(t);
    if (Zz2 < ER2 || isnan(de.x) || isinf(de.x) || isnan(de.y) || isinf(de.y))
    {
      n = Iterations;
      nf = 0;
      t = 0;
      de = 0;
    }
    const coord_t k = (j - y0) * data->width + (i - x0);

    /* output raw */
    const count_t Nbias = 1024;
    uint64_t nn = n + Nbias;
    if (n >= Iterations)
    {
      nn = ~((uint64_t)(0));
    }
    if (data->N0)
    {
      data->N0[k] = nn;
    }
    if (data->N1)
    {
      data->N1[k] = nn >> 32;
    }
    if (data->NF)
    {
      data->NF[k] = nf;
    }
    if (data->T)
    {
      data->T[k] = t;
    }
    if (data->DEX)
    {
      data->DEX[k] = de.x;
    }
    if (data->DEY)
    {
      data->DEY[k] = de.y;
    }
    if (data->BLA)
    {
      data->BLA[k] = steps_bla;
    }
    if (data->PTB)
    {
      data->PTB[k] = iters_ptb;
    }
  }
#undef normx
  return *running;
}

template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<float>>> &Zp, const std::vector<blasR2<float>> &bla, volatile bool *running);
template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<double>>> &Zp, const std::vector<blasR2<double>> &bla, volatile bool *running);
template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<long double>>> &Zp, const std::vector<blasR2<long double>> &bla, volatile bool *running);
template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<floatexp<float, int>>>> &Zp, const std::vector<blasR2<floatexp<float, int>>> &bla, volatile bool *running);
template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<floatexp<double, int>>>> &Zp, const std::vector<blasR2<floatexp<double, int>>> &bla, volatile bool *running);
template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<softfloat>>> &Zp, const std::vector<blasR2<softfloat>> &bla, volatile bool *running);
#ifdef HAVE_FLOAT128
template bool hybrid_render(coord_t frame, coord_t x0, coord_t y0, coord_t x1, coord_t y1, coord_t subframe, tile *data, const param &par, const std::vector<std::vector<complex<float128>>> &Zp, const std::vector<blasR2<float128>> &bla, volatile bool *running);
#endif

template <typename t>
count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<t>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &Iterations, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running)
{
  using floatexp_ = floatexp<float, int>;
  complex<floatexp_> C (floatexp_(Zp[0][1].x), floatexp_(Zp[0][1].y)); // FIXME
  dual<2, floatexp_> cx (c0.x); cx.dx[0] = 1; cx.dx[1] = 0;
  dual<2, floatexp_> cy (c0.y); cy.dx[0] = 0; cy.dx[1] = 1;
  const complex<dual<2, floatexp_>> c (cx, cy);
  complex<dual<2, floatexp_>> z (0);
  const mat2<floatexp_> K1 (inverse(mat2<floatexp_>(floatexp_(K.x[0][0]), floatexp_(K.x[0][1]), floatexp_(K.x[1][0]), floatexp_(K.x[1][1]))));
  floatexp_ Zz2 = 0;
  const floatexp_ r2 = e10(float(1), 10000);
  bool p = true;
  count_t i = 0;
  count_t n = 0;
  count_t m = 0;
  count_t phase = 0;
  while (i < Iterations && Zz2 < r2 && p && *running)
  {
    // progress
    progress[0] = i / progress_t(Iterations);
    // formula
    if (! (m < count_t(Zp[phase].size())))
    {
      break;
    }
    // rebase
    complex<floatexp_> Z(floatexp_(Zp[phase][m].x), floatexp_(Zp[phase][m].y));
    complex<dual<2, floatexp_>> Zz = Z + z;
    Zz2 = norm(complex<floatexp_>(Zz.x.x, Zz.y.x));
    const floatexp_ z2 = norm(complex<floatexp_>(z.x.x, z.y.x));
    if (Zz2 < z2 || m + 1 == count_t(Zp[phase].size()))
    {
      z = Zz;
      phase = (phase + m) % Zp.size();
      m = 0;
      Z = 0;
    }
    bool rebased = false;
    z = hybrid_perturb(opss[n % opss.size()], C, Z, c, z, rebased);
    if (rebased)
    {
      phase = (phase + m) % Zp.size();
      m = 0;
    }
    m++;
    n++;
    // (u1 v1) = s^{-1} K^{-1} J^{-1} (u0 v0)
    if (! (m < count_t(Zp[phase].size())))
    {
      break;
    }
    Z = complex<floatexp_>(floatexp_(Zp[phase][m].x), floatexp_(Zp[phase][m].y));
    Zz = Z + z;
    const mat2<floatexp_> J(z.x.dx[0], z.x.dx[1], z.y.dx[0], z.y.dx[1]);
    complex<floatexp_> w = (K1 * (inverse(J) * complex<floatexp_>(Zz.x.x, Zz.y.x)));
    floatexp_ q = floatexp_(norm(w)) / (s * s);
    p = 1 <= q;
    ++i;
  }
  if (i == Iterations || ! (Zz2 < r2) || p || ! *running)
  {
    return 0;
  }
  return i;
}

template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<float>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<double>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<long double>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<floatexp<float, int>>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<floatexp<double, int>>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<softfloat>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
#ifdef HAVE_FLOAT128
template count_t hybrid_period(const std::vector<std::vector<opcode>> &opss, const std::vector<std::vector<complex<float128>>> &Zp, const complex<floatexp<float, int>> &c0, const count_t &N, const floatexp<float, int> &s, const mat2<double> &K, volatile progress_t *progress, volatile bool *running);
#endif

bool hybrid_center(const std::vector<std::vector<opcode>> &opss, complex<mpreal> &C0, const count_t period, volatile progress_t *progress, volatile bool *running)
{
  mpfr_prec_t prec = std::max(mpfr_get_prec(C0.x.mpfr_srcptr()), mpfr_get_prec(C0.y.mpfr_srcptr()));
  const floatexp<float, int> epsilon2 = floatexp<float, int>(1, 16 - 2 * prec);
  double lepsilon2 = double(log(epsilon2));
  double ldelta0 = 0;
  double ldelta1 = 0;
  progress_t eta = 0;
  bool converged = false;
  const count_t maxsteps = 64; // FIXME
  for (count_t j = 0; j < maxsteps && *running && ! converged; ++j)
  {
    progress[0] = j > 1 ? j / (j + eta) : 0;
    progress[1] = 0;
    dual<2, mpreal> cx(C0.x); cx.dx[0] = 1;
    dual<2, mpreal> cy(C0.y); cy.dx[1] = 1;
    complex<dual<2, mpreal>> c(cx, cy);
    complex<dual<2, mpreal>> z(0, 0);
    // iteration
    for (count_t i = 0; i < period && *running; ++i)
    {
      progress[1] = i / progress_t(period);
      z = hybrid_plain(opss[i % opss.size()], c, z);
    }
    if (*running)
    {
      const mpreal &x = z.x.x;
      const mpreal &y = z.y.x;
      const mpreal &dxa = z.x.dx[0];
      const mpreal &dxb = z.x.dx[1];
      const mpreal &dya = z.y.dx[0];
      const mpreal &dyb = z.y.dx[1];
      // Newton step
      const mpreal det = dxa * dyb - dxb * dya;
      const mpreal u = -( dyb * x - dxb * y) / det;
      const mpreal v = -(-dya * x + dxa * y) / det;
      C0.x += u;
      C0.y += v;
      // check convergence
      floatexp<float, int> uf = floatexp<float, int>(u);
      floatexp<float, int> vf = floatexp<float, int>(v);
      floatexp<float, int> delta = sqr(uf) + sqr(vf);
      converged = delta < epsilon2;
      ldelta0 = ldelta1;
      ldelta1 = double(log(delta));
      eta = log2((lepsilon2 - ldelta0) / (ldelta1 - ldelta0));
    }
  }
  return converged;
}

bool hybrid_size(floatexp<float, int> &s, mat2<double> &K, const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees, const complex<mpreal> &C0, count_t period, volatile progress_t *progress, volatile bool *running)
{
  using std::abs;
  using ::abs;
  using std::exp;
  using ::exp;
  using std::log;
  using ::log;
  using std::sqrt;
  using ::sqrt;
  double log_degree = log(double(degrees[0]));
  complex<dual<2, mpreal>> C(C0.x, C0.y);
  C.x.dx[0] = 0;
  C.y.dx[1] = 0;
  complex<dual<2, mpreal>> Z(C);
  Z.x.dx[0] = 1;
  Z.y.dx[1] = 1;
  mat2<floatexp<float, int>> b (floatexp<float, int>(1));
  count_t j = 1;
  count_t m = 1;
  if (m == period)
  {
    m = 0;
  }
  while (j < period && *running)
  {
    progress[0] = j / progress_t(period);
    Z = hybrid_plain(opss[j % opss.size()], C, Z);
    log_degree += log(double(degrees[j % degrees.size()]));
    const mat2<floatexp<float, int>> l (floatexp<float, int>(Z.x.dx[0]), floatexp<float, int>(Z.x.dx[1]), floatexp<float, int>(Z.y.dx[0]), floatexp<float, int>(Z.y.dx[1]));
    b += inverse(l);
    ++j;
  }
  const double degree = exp(log_degree / period);
  // l^d b
  if (*running)
  {
    double d = degree / (degree - 1);
    const mat2<floatexp<float, int>> l (floatexp<float, int>(Z.x.dx[0]), floatexp<float, int>(Z.x.dx[1]), floatexp<float, int>(Z.y.dx[0]), floatexp<float, int>(Z.y.dx[1]));
    const floatexp<float, int> lambda = sqrt(abs(determinant(l)));
    const floatexp<float, int> beta = sqrt(abs(determinant(b)));
    const floatexp<float, int> llb = exp(log(lambda) * d) * beta;
    s = floatexp<float, int>(1 / llb);
    b = inverse(transpose(b)) / beta; // FIXME should division be inside?
    K = mat2<double>(double(b.x[0][0]), double(b.x[0][1]), double(b.x[1][0]), double(b.x[1][1]));
    return true;
  }
  return false;
}

std::string hybrid_perturb_opencl(const std::vector<std::vector<opcode>> &opss, const std::vector<int> &degrees)
{
  std::ostringstream s;
  s << "{\n";
  s << "  struct complex Z = { ref[config->ref_start[phase] + 2 * m], ref[config->ref_start[phase] + 2 * m + 1] };\n";
  s << "  struct complex Z_stored = Z;\n";
  s << "  struct complexdual z_stored = z;\n";
  s << "  struct complex rot;\n";
  s << "  switch (n % " << opss.size() << ")\n";
  s << "  {\n";
  for (size_t k = 0; k < opss.size(); ++k)
  {
    s << "  case " << k << ":\n";
    s << "    {\n";
    for (const auto & op : opss[k])
    {
      s << "      // rebase\n";
      s << "      {\n";
      s << "        struct complexdual Zz = complexdual_add_complex_complexdual(Z, z);\n";
      s << "        real Zz2 = real_norm_complexdual(Zz);\n";
      s << "        real z2 = real_norm_complexdual(z);\n";
      s << "        if (bool_lt_real_real(Zz2, z2))\n";
      s << "        {\n";
      s << "          z = Zz;\n";
      s << "          phase = (phase + m) % " << opss.size() << ";\n";
      s << "          m = 0;\n";
      s << "          struct complex Zn = { ref[config->ref_start[phase] + 2 * m], ref[config->ref_start[phase] + 2 * m + 1] };\n";
      s << "          Z = Zn;\n";
      s << "        }\n";
      s << "      }\n";
      switch (op.op)
      {
        case op_add:
          s << "      // add\n";
          s << "      z = complexdual_add_complexdual_complexdual(z, c);\n";
          s << "      Z = complex_add_complex_complex(Z, C);\n"; // FIXME
          break;
        case op_store:
          s << "      // store\n";
          s << "      z_stored = z;\n";
          s << "      Z_stored = Z;\n";
          break;
        case op_mul:
          s << "      // mul\n";
          s << "      z = complexdual_add_complexdual_complexdual(complexdual_add_complexdual_complexdual(complexdual_mul_complex_complexdual(Z_stored, z), complexdual_mul_complex_complexdual(Z, z_stored)), complexdual_mul_complexdual_complexdual(z, z_stored));\n";
          s << "      Z = complex_mul_complex_complex(Z, Z_stored);\n";
          break;
        case op_sqr:
          s << "      // sqr\n";
          s << "      z = complexdual_mul_complexdual_complexdual(complexdual_add_complex_complexdual(complex_mul2_complex(Z), z), z);\n";
          s << "      Z = complex_sqr_complex(Z);\n";
          break;
        case op_absx:
          s << "      // absx\n";
          s << "      z.x = dual_diffabs_real_dual(Z.x, z.x);\n";
          s << "      Z.x = real_abs_real(Z.x);\n";
          break;
        case op_absy:
          s << "      // absy\n";
          s << "      z.y = dual_diffabs_real_dual(Z.y, z.y);\n";
          s << "      Z.y = real_abs_real(Z.y);\n";
          break;
        case op_negx:
          s << "      // negx\n";
          s << "      z.x = dual_neg_dual(z.x);\n";
          s << "      Z.x = real_neg_real(Z.x);\n";
          break;
        case op_negy:
          s << "      // negy\n";
          s << "      z.y = dual_neg_dual(z.y);\n";
          s << "      Z.y = real_neg_real(Z.y);\n";
          break;
        case op_rot:
          s << "      // rot\n";
          s << "      rot.x = real_from_float(" << op.u.rot.x << ");\n";
          s << "      rot.y = real_from_float(" << op.u.rot.y << ");\n";
          s << "      z = complexdual_mul_complexdual_complex(z, rot);\n";
          s << "      Z = complex_mul_complex_complex(Z, rot);\n";
          break;
      }
    }
    s << "      // degree\n";
    s << "      last_degree = " << degrees[k] << ";\n";
    s << "    }\n";
    s << "    break;\n";
  }
  s << "  }\n";
  s << "}\n";
  return s.str();
}
