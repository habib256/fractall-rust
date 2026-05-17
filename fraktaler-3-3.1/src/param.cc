// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#include <iostream>
#include <map>

#include <toml.hpp>

#include "exr.h"
#include "jpeg.h"
#include "param.h"
#include "png.h"
#include "source.h"

patom::patom(const toml::value &x)
{
  if (x.is_string())
  {
    tag = t_string;
    string_ = x.as_string();
  }
  else if (x.is_floating())
  {
    tag = t_double;
    double_ = double(x.as_floating());
  }
  else if (x.is_integer())
  {
    tag = t_int64;
    int64_ = int64_t(x.as_integer());
  }
  else
  {
#if TOML11_API >= 4
    throw toml::type_error(toml::format_error("unexpected atom type", x, "expected one of: string, double, integer"), x.location());
#else
    throw toml::type_error(toml::detail::format_underline("unexpected atom type", {{x.location(),"expected one of: string, double, integer"}}), x.location());
#endif
  }
}

toml::value to_toml(const patom &x)
{
  switch (x.tag)
  {
    case t_string: return toml::value(x.string_);
    case t_double: return toml::value(x.double_);
    case t_int64: return toml::value(x.int64_);
  }
  return toml::value(""); // FIXME
}

std::map<std::string, toml::value> to_toml(const std::map<std::string, patom> &x)
{
  std::map<std::string, toml::value> r;
  for (const auto & [ k, v ] : x)
  {
    r[k] = to_toml(v);
  }
  return r;
}

std::vector<std::map<std::string, toml::value>> to_toml(const std::vector<std::map<std::string, patom>> &x)
{
  std::vector<std::map<std::string, toml::value>> r;
  r.reserve(x.size());
  for (const auto & i : x)
  {
    r.push_back(to_toml(i));
  }
  return r;
}


param::param()
: center(0)
, zoom(1)
, reference(0)
{
  exr_channels = Channels_RGB;
  p.render.exr_channels = {"R", "G", "B"};
  post_edit_formula(*this);
  home(*this);
}

void restring_locs(param &par)
{
  par.p.location.real = par.center.x.toString();
  par.p.location.imag = par.center.y.toString();
  par.p.reference.real = par.reference.x.toString();
  par.p.reference.imag = par.reference.y.toString();
}

void restring_vals(param &par)
{
  { std::ostringstream s; s << par.p.reference.period; par.s_period = s.str(); }
  { std::ostringstream s; s << par.zoom; par.p.location.zoom = s.str(); }
  { std::ostringstream s; s << par.p.bailout.iterations; par.s_iterations = s.str(); }
  { std::ostringstream s; s << par.p.bailout.maximum_reference_iterations; par.s_maximum_reference_iterations = s.str(); }
  { std::ostringstream s; s << par.p.bailout.maximum_perturb_iterations; par.s_maximum_perturb_iterations = s.str(); }
  { std::ostringstream s; s << par.p.bailout.maximum_bla_steps; par.s_maximum_bla_steps = s.str(); }
  { std::ostringstream s; s << par.p.bailout.inscape_radius; par.s_inscape_radius = s.str(); }
  par.transform = mat2<double>(polar2<double>(par.p.transform.reflect ? -1 : 1, 1, par.p.transform.rotate * 2 * M_PI / 360, std::exp2(par.p.transform.stretch_amount / 100), par.p.transform.stretch_angle * 2 * M_PI / 360));
  {
    const std::vector<std::string> exr_channels_list = { "R", "G", "B", "N0", "N1", "NF", "T", "DEX", "DEY", "BLA", "PTB" }; // FIXME depends on values in image.h
    channel_mask_t c = 0;
    for (const auto & s : par.p.render.exr_channels)
    {
      bool found = false;
      for (size_t i = 0; i < exr_channels_list.size(); ++i)
      {
        if (s == exr_channels_list[i])
        {
          c |= channel_mask_t(1) << i;
          found = true;
          break;
        }
      }
      if (! found)
      {
        // FIXME ignored
      }
    }
    par.exr_channels = c;
  }
}

void unstring_locs(param &par)
{
  par.zoom = floatexp<float, int>(par.p.location.zoom);
  mpfr_prec_t prec = std::max(24, 24 + (par.zoom * par.p.image.height).exp);
  par.center.x.set_prec(prec);
  par.center.y.set_prec(prec);
  par.center.x = par.p.location.real;
  par.center.y = par.p.location.imag;
  par.reference.x.set_prec(prec); // FIXME
  par.reference.y.set_prec(prec); // FIXME
  par.reference.x = par.p.reference.real;
  par.reference.y = par.p.reference.imag;
}

void unstring_vals(param &par)
{
  par.p.reference.period = std::atoll(par.s_period.c_str());
  par.p.bailout.iterations = std::atoll(par.s_iterations.c_str());
  par.p.bailout.maximum_reference_iterations = std::atoll(par.s_maximum_reference_iterations.c_str());
  par.p.bailout.maximum_perturb_iterations = std::atoll(par.s_maximum_perturb_iterations.c_str());
  par.p.bailout.maximum_bla_steps = std::atoll(par.s_maximum_bla_steps.c_str());
  par.p.bailout.inscape_radius = std::atof(par.s_inscape_radius.c_str());
  polar2<double> P(par.transform);
  par.p.transform.reflect = P.sign < 0 ? true : false;
  par.p.transform.rotate = P.rotate * 360 / (2 * M_PI);
  par.p.transform.stretch_amount = 100 * std::log2(P.stretch_factor);
  par.p.transform.stretch_angle = P.stretch_angle * 360 / (2 * M_PI);
  {
    const std::vector<std::string> exr_channels_list = { "R", "G", "B", "N0", "N1", "NF", "T", "DEX", "DEY", "BLA", "PTB" }; // FIXME depends on values in image.h
    std::vector<std::string> v;
    for (size_t i = 0; i < exr_channels_list.size(); ++i)
    {
      channel_mask_t m = channel_mask_t(1) << i;
      if ((par.exr_channels & m) == m)
      {
        v.push_back(exr_channels_list[i]);
      }
    }
    par.p.render.exr_channels = v;
  }
}

void clamp_escape_radius(param &par)
{
  int min_deg = par.degrees[0];
  int max_deg = par.degrees[0];
  for (size_t i = 0; i < par.degrees.size(); ++i)
  {
    min_deg = std::min(min_deg, par.degrees[i]);
    max_deg = std::min(max_deg, par.degrees[i]);
  }
  double low_bailout = std::pow(2.0, 1.0 / (min_deg - 1));
  double high_bailout = std::pow(1.0e38 / 2.0, 1.0 / (2 * max_deg));
  par.p.bailout.escape_radius = std::min(std::max(par.p.bailout.escape_radius, low_bailout), high_bailout);
}

void post_edit_formula(param &par)
{
  par.opss = compile_formula(par.p.formula);
  assert(validate_opcodess(par.opss));
  par.degrees = opcodes_degrees(par.opss);
  par.s_opss = print_opcodess(par.opss);
  clamp_escape_radius(par);
}

void home(param &par)
{
  par.reference.x.set_prec(24);
  par.reference.y.set_prec(24);
  par.reference = 0;
  par.center.x.set_prec(24);
  par.center.y.set_prec(24);
  par.center = 0;
  par.zoom = 1;
  par.p.bailout.iterations = 1024;
  par.p.bailout.maximum_reference_iterations = par.p.bailout.iterations;
  par.p.bailout.maximum_perturb_iterations = 1024;
  par.p.bailout.maximum_bla_steps = 1024;
  par.p.reference.period = 0;
  par.p.algorithm.lock_maximum_reference_iterations_to_period = false;
  par.p.algorithm.reuse_reference = false;
  par.p.algorithm.reuse_bilinear_approximation = false;
  restring_locs(par);
  restring_vals(par);
  par.transform = mat2<double>(1);
  unstring_vals(par);
}

void zoom(param &par, double x, double y, double g, bool fixed_click)
{
  complex<double> w (x * par.p.image.width / par.p.image.height, -y);
  w = par.transform * w;
  floatexp d = (fixed_click ? 1 - 1 / g : 1) * 2 / par.zoom;
  floatexp u = d * w.x;
  floatexp v = d * w.y;
  par.zoom *= g;
  mpfr_prec_t prec = std::max(24, 24 + (par.zoom * par.p.image.height).exp);
  mpfr_t dx, dy;
  mpfr_init2(dx, 53);
  mpfr_init2(dy, 53);
  mpfr_set_d(dx, u.val, MPFR_RNDN);
  mpfr_set_d(dy, v.val, MPFR_RNDN);
  mpfr_mul_2si(dx, dx, u.exp, MPFR_RNDN);
  mpfr_mul_2si(dy, dy, v.exp, MPFR_RNDN);
  par.center.x.set_prec(prec);
  par.center.y.set_prec(prec);
  mpfr_add(par.center.x.mpfr_ptr(), par.center.x.mpfr_srcptr(), dx, MPFR_RNDN);
  mpfr_add(par.center.y.mpfr_ptr(), par.center.y.mpfr_srcptr(), dy, MPFR_RNDN);
  mpfr_clear(dx);
  mpfr_clear(dy);
  restring_locs(par);
  restring_vals(par);
}

complex<floatexp<float, int>> get_delta_c(const param &par, double x, double y)
{
  complex<double> w (x * par.p.image.width / par.p.image.height, -y);
  w = par.transform * w;
  return complex<floatexp<float, int>>(floatexp<float, int>(w.x / par.zoom), floatexp<float, int>(w.y / par.zoom));
}

void zoom(param &par, const mat3 &T, const mat3 &T0)
{
  // translate
  vec3 t = T * vec3(0.0f, 0.0f, 1.0f);
  vec2 w = vec2(t) / t.z;
  zoom(par, w.x, -w.y, 1, false);
  // zoom
  mat2<double> T2(T0);
  double g = std::sqrt(std::abs(determinant(T2)));
  par.zoom *= g;
  mpfr_prec_t prec = std::max(24, 24 + (par.zoom * par.p.image.height).exp);
  par.center.x.set_prec(prec);
  par.center.y.set_prec(prec);
  // rotate, skew
  T2 /= g;
  T2 = inverse(T2);
  // done
  restring_locs(par);
  restring_vals(par);
  par.transform = par.transform * T2;
  unstring_vals(par);
}

void param::load_toml(const std::string &filename)
{
  std::ifstream ifs(filename, std::ios_base::binary);
  ifs.exceptions(std::ifstream::badbit);
  ifs >> p;
  unstring_locs(*this);
  restring_vals(*this);
  post_edit_formula(*this);
}

void param::from_toml_string(const std::string &str)
{
  std::istringstream ifs(str);
  ifs >> p;
  unstring_locs(*this);
  restring_vals(*this);
  post_edit_formula(*this);
}

void param::from_kfr_string(const std::string &str)
{
  int FractalType = 0;
  int Power = 2;
  double RotateAngle = 0;
  double StretchAngle = 0;
  double StretchAmount = 0;
  bool ImagPointsUp = false;

  // parse fields
  int fields = 0;
  std::ostringstream ofs;
  std::istringstream ifs(str);
  std::string line;
  while (std::getline(ifs, line, '\n'))
  {
    std::istringstream ils(line);
    std::string key;
    if (std::getline(ils, key, ':'))
    {
      std::string val;
      if (std::getline(ils, val, '\r'))
      {
        if (val.c_str()[0] == ' ')
        {
          val = val.substr(1);
        }
        fields++;
        if      ("Re" == key) ofs << "location.real = \"" << val << "\"\n";
        else if ("Im" == key) ofs << "location.imag = \"" << val << "\"\n";
        else if ("Zoom" == key) ofs << "location.zoom = \"" << val << "\"\n";
        else if ("Iterations" == key) ofs << "bailout.iterations = " << val << "\nbailout.maximum_reference_iterations = " << val << "\n";
        else if ("Period" == key) ofs << "reference.period = " << val << "\n";
        else if ("RotateAngle" == key) RotateAngle = std::stod(val);
        else if ("StretchAngle" == key) StretchAngle = std::stod(val);
        else if ("StretchAmount" == key) StretchAmount = std::stod(val);
        else if ("ImagPointsUp" == key) ImagPointsUp = std::stoi(val);
        else if ("Power" == key) Power = std::stoi(val);
        else if ("FractalType" == key) FractalType = std::stoi(val);
      }
    }
  }
  if (fields == 0)
  {
    // probably not a KFR, what do?
    return;
  }

  // transformation: KF2 does skew * rotate * reflect
  polar2<double> reflect(ImagPointsUp ? 1 : -1, 1, 0, 1, 0);
  polar2<double> rotate(1, 1, RotateAngle * 2 * M_PI / 360, 1, 0);
  polar2<double> skew(1, 1, 0, std::exp2(StretchAmount), StretchAngle * 2 * M_PI / 360);
  polar2<double> transform(mat2<double>(skew) * mat2<double>(rotate) * mat2<double>(reflect));
  ofs << "transform.reflect = " << (transform.sign > 0 ? "false" : "true") << "\n";
  ofs << "transform.rotate = " << std::fixed << std::setprecision(16) << (transform.rotate * 360 / (2 * M_PI)) << "\n";
  ofs << "transform.stretch_angle = " << std::fixed << std::setprecision(16) << (transform.stretch_angle * 360 / (2 * M_PI)) << "\n";
  ofs << "transform.stretch_amount = " << std::fixed << std::setprecision(16) << (100 * std::log2(transform.stretch_factor)) << "\n";

  // formulas
  const char *powers[9] =
    { ""
    , ""
    , "sqr"
    , "store sqr mul"
    , "sqr sqr"
    , "store sqr sqr mul"
    , "store sqr mul sqr"
    , "store sqr mul sqr mul"
    , "sqr sqr sqr"
    };
  switch (FractalType)
  {
    // Mandelbrot set
    case  0: ofs << "\n[[formula]]\npower = " << Power << "\n\n"; break;
    // Burning Ship
    case  1: ofs << "\n[[formula]]\nabs_y = true\nabs_x = true\npower = " << Power << "\n\n"; break;
    // Buffalo
    case  2: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"" << powers[Power] << " absx absy add\"\n\n"; } break;
    // Celtic
    case  3: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"" << powers[Power] << " absx add\"\n\n"; } break;
    // Mandelbar
    case  4: ofs << "\n[[formula]]\npower = " << Power << "\nneg_y = true\n\n"; break;
    // Mandelbar Celtic
    case  5: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"negy " << powers[Power] << " absx add\"\n\n"; } break;
    // Perpendicular Mandelbrot
    case  6: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx negy " << powers[Power] << " add\"\n\n"; } break;
    // Perpendicular Burning Ship
    case  7: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absy negy " << powers[Power] << " add\"\n\n"; } break;
    // Perpendicular Celtic
    case  8: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx negy " << powers[Power] << " absx add\"\n\n"; } break;
    // Perpendicular Buffalo
    case  9: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absy negy " << powers[Power] << " absx add\"\n\n"; } break;
    // Cubic Quasi Burning Ship
    case 10: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx " << powers[Power] << " absy negy add\"\n\n"; } break;
    // Cubic Partial BS Real
    case 11: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx " << powers[Power] << " add\"\n\n"; } break;
    // Cubic Partial BS Imag
    case 12: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absy " << powers[Power] << " add\"\n\n"; } break;
    // Cubic Flying Squirrel (Buffalo Imag)
    case 13: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"" << powers[Power] << " absy add\"\n\n"; } break;
    // Cubic Quasi Perpendicular
    case 14: break; // not possible with opcodes?
    // 4th Burning Ship Partial Imag
    case 15: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absy " << powers[Power] << " add\"\n\n"; } break;
    // 4th Burning Ship Partial Real
    case 16: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx " << powers[Power] << " add\"\n\n"; } break;
    // 4th Burning Ship Partial Real Mbar
    case 17: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx negy " << powers[Power] << " add\"\n\n"; } break;
    // 4th Celtic Burning Ship Partial Imag
    case 18: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absy " << powers[Power] << " absx add\"\n\n"; } break;
    // 4th Celtic Burning Ship Partial Real
    case 19: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx " << powers[Power] << " absx add\"\n\n"; } break;
    // 4th Celtic Burning Ship Partial Real Mbar
    case 20: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx negy " << powers[Power] << " absx add\"\n\n"; } break;
    // 4th Buffalo Partial Imag
    case 21: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"" << powers[Power] << " absy add\"\n\n"; } break;
    // 4th Celtic Mbar
    case 22: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"negy " << powers[Power] << " absx add\"\n\n"; } break;
    // 4th False Quasi Perpendicular
    case 23: break; // not possible with opcodes?
    // 4th False Quasi Heart
    case 24: break; // not possible with opcodes?
    // 4th Celtic False Quasi Perpendicular
    case 25: break; // not possible with opcodes?
    // 4th Celtic False Quasi Heart
    case 26: break; // not possible with opcodes?
    // 5th Burning Ship Partial
    case 27: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx " << powers[Power] << " add\"\n\n"; } break;
    // 5th Burning Ship Partial Mbar
    case 28: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx negy " << powers[Power] << " add\"\n\n"; } break;
    // 5th Celtic Mbar
    case 29: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"negy " << powers[Power] << " absx add\"\n\n"; } break;
    // 5th Quasi Burning Ship (BS/Buffalo Hybrid)
    case 30: if (Power < 9) { ofs << "[[formula]]\nopcodes = \"absx " << powers[Power] << " absy negy add\"\n\n"; } break;
    // 5th Quasi Perpendicular
    case 31: break; // not possible with opcodes?
    // 5th Quasi Heart
    case 32: break; // not possible with opcodes?
    // SimonBrot 4th
    case 33: ofs << "[[formula]]\nopcodes = \"store absx absy mul sqr add\"\n\n"; break;
    // 4th Imag Quasi Perpendicular / Heart
    case 34: break; // not possible with opcodes?
    // 4th Real Quasi Perpendicular
    case 35: break; // not possible with opcodes?
    // 4th Real Quasi Heart
    case 36: break; // not possible with opcodes?
    // 4th Celtic Imag Quasi Perpendicular / Heart
    case 37: break; // not possible with opcodes?
    // 4th Celtic Real Quasi Perpendicular
    case 38: break; // not possible with opcodes?
    // 4th Celtic Real Quasi Heart
    case 39: break; // not possible with opcodes?
    // SimonBrot 6th
    case 40: ofs << "[[formula]]\nopcodes = \"store absx absy mul store sqr mul add\"\n\n"; break;
    // HPDZ Buffalo
    case 41: break; // not possible with opcodes
    // TheRedshiftRider 1: a * z^2 + z^3 + c
    case 42: break; // not possible with opcodes
    // TheRedshiftRider 2: a * z^2 - z^3 + c
    case 43: break; // not possible with opcodes
    // TheRedshiftRider 3: 2 * z^2 - z^3 + c
    case 44: break; // not possible with opcodes
    // TheRedshiftRider 4: a * z^2 + z^4 + c
    case 45: break; // not possible with opcodes
    // TheRedshiftRider 5: a * z^2 - z^4 + c
    case 46: break; // not possible with opcodes
    // TheRedshiftRider 6: a * z^2 + z^5 + c
    case 47: break; // not possible with opcodes
    // TheRedshiftRider 7: a * z^2 - z^5 + c
    case 48: break; // not possible with opcodes
    // TheRedshiftRider 8: a * z^2 + z^6 + c
    case 49: break; // not possible with opcodes
    // TheRedshiftRider 9: a * z^2 - z^6 + c
    case 50: break; // not possible with opcodes
    // SimonBrot2 4th
    case 51: ofs << "[[formula]]\nopcodes = \"sqr store absx absy mul add\"\n\n"; break; // FIXME check
    // General Quadratic Minus
    case 52: break; // not possible with opcodes
    // General Quadratic Plus
    case 53: break; // not possible with opcodes
    // Mothbrot 2nd 1x1
    case 54: ofs << "[[formula]]\nopcodes = \"store absx absy mul add\"\n\n"; break;
    // Mothbrot 3rd 1x2
    case 55: ofs << "[[formula]]\nopcodes = \"store absx absy sqr mul add\"\n\n"; break;
    // Mothbrot 3rd 2x1
    case 56: ofs << "[[formula]]\nopcodes = \"store absx absy mul mul add\"\n\n"; break;
    // Mothbrot 4th 1x3
    case 57: break; // not possible with opcodes?
    // Mothbrot 4th 2x2 (aka SimonBrot 4th)
    case 58: ofs << "[[formula]]\nopcodes = \"store absx absy mul sqr add\"\n\n"; break;
    // Mothbrot 4th 3x1
    case 59: ofs << "[[formula]]\nopcodes = \"store absx absy mul mul mul add\"\n\n"; break;
    // Mothbrot 5th 1x4
    case 60: ofs << "[[formula]]\nopcodes = \"store absx absy sqr sqr mul add\"\n\n"; break;
    // Mothbrot 5th 2x3
    case 61: break; // not possible with opcodes?
    // Mothbrot 5th 3x2
    case 62: ofs << "[[formula]]\nopcodes = \"store absx absy sqr mul mul mul add\"\n\n"; break;
    // Mothbrot 5th 4x1
    case 63: ofs << "[[formula]]\nopcodes = \"store absx absy mul mul mul mul add\"\n\n"; break;
    // Mothbrot 6th 1x5
    case 64: break; // not possible with opcodes?
    // Mothbrot 6th 2x4 (Simon's Mothbrot)
    case 65: ofs << "[[formula]]\nopcodes = \"store absx absy sqr sqr mul mul add\"\n\n"; break;
    // Mothbrot 6th 3x3 (aka SimonBrot 6th)
    case 66: ofs << "[[formula]]\nopcodes = \"store absx absy mul store sqr mul add\"\n\n"; break;
    // Mothbrot 6th 4x2
    case 67: ofs << "[[formula]]\nopcodes = \"store absx absy mul mul sqr add\"\n\n"; break;
    // Mothbrot 6th 5x1
    case 68: ofs << "[[formula]]\nopcodes = \"store absx absy mul mul mul mul mul add\"\n\n"; break;
    // fractal types 69 through 102 are not possible with opcodes
    /*
    Abs General Quadratic Minus
    Abs General Quadratic Plus
    Omnibrot
    Hidden Mandelbrot
    Hidden Mandelbrot a la Cos
    Polarbrot
    Mandelbrot Variation 1
    Mandelbrot Variation 2
    Mandelbrot Variation 3
    Mandelbrot Variation 4
    Mandelbrot Variation 5
    Simon 0139-C (plain)
    Simon 0139-C (Burning Ship)
    Simon 0139-C (Celtic)
    Simon 0139-C (Buffalo)
    Cubic Celtic Quasi Perpendicular
    Cubic Quasi Perpendicular Burning Ship
    Cubic Quasi Perpendicular Buffalo
    Quintic Quasi Perpendicular Burning Ship
    Quintic Quasi Perpendicular Buffalo
    Quintic Quasi Perpendicular Celtic
    Quintic Quasi Celtic Heart
    Separated Perpendicular
    Exponential Mandelbrot
    Cosbrot
    Sinbrot
    Talis
    Newton Mandelbrot
    Nova
    Newton Nova Mandelbrot (z^p - 1)
    Halley Nova Mandelbrot (z^p - 1)
    Schroder Nova Mandelbrot (z^p - 1)
    Householder 3 Nova Mandelbrot (z^p - 1)
    Householder Nova Mandelbrot (z^p - 1)
    */
    // unsupported fractal type, what do?
    default: break;
  }

  from_toml_string(ofs.str());
}

void param::from_string(const std::string &str)
{
  try
  {
    from_toml_string(str);
  }
  catch (...)
  {
    from_kfr_string(str);
  }
}

void param::load_kfr(const std::string &filename)
{
  std::ifstream ifs(filename, std::ios_base::binary);
  ifs.exceptions(std::ifstream::badbit);
  std::ostringstream s;
  s << ifs.rdbuf();
  from_kfr_string(s.str());
}

void param::load_exr(const std::string &filename)
{
  MultiPartInputFile in(filename.c_str());
  for (int p = 0; p < in.parts(); ++p)
  {
    const Header &h = in.header(p);
    for (Header::ConstIterator i = h.begin(); i != h.end(); ++i)
    {
      std::string name(i.name());
      if (name == "Fraktaler3")
      {
        const Attribute *a = &i.attribute();
        if (const StringAttribute *s = dynamic_cast<const StringAttribute *>(a))
        {
          from_string(std::string(s->value()));
          return;
        }
      }
      else if (name == "KallesFraktaler2+")
      {
        const Attribute *a = &i.attribute();
        if (const StringAttribute *s = dynamic_cast<const StringAttribute *>(a))
        {
          from_string(std::string(s->value()));
          return;
        }
      }
    }
  }
}

void param::load_png(const std::string &filename)
{
  std::string s = read_png_comment(filename);
  if (s != "")
  {
    from_string(s);
  }
}

void param::load_jpeg(const std::string &filename)
{
  std::string s = read_jpeg_comment(filename);
  if (s != "")
  {
    from_string(s);
  }
}

void param::load_glsl(const std::string &filename)
{
  std::ifstream ifs(filename, std::ios_base::binary);
  ifs.exceptions(std::ifstream::badbit);
  std::ostringstream s;
  s << ifs.rdbuf();
  p.colour.shader = s.str();
}

void param::load_any(const std::string &filename)
{
  if (ends_with(filename, ".exr"))
  {
    load_exr(filename);
  }
  else if (ends_with(filename, ".png"))
  {
    load_png(filename);
  }
  else if (ends_with(filename, ".jpg") || ends_with(filename, ".jpeg"))
  {
    load_jpeg(filename);
  }
  else if (ends_with(filename, ".glsl"))
  {
    load_glsl(filename);
  }
  else if (ends_with(filename, ".kfr"))
  {
    load_kfr(filename);
  }
  else
  {
    load_toml(filename);
  }
}

std::istream &operator>>(std::istream &ifs, pparam &p)
{
  auto t = toml::parse(ifs);
  auto f = toml::find_or(t, "formula", std::vector<toml::table>());
  std::vector<phybrid1> per;
  for (auto f1 : f)
  {
    toml::value g(f1);
    phybrid1 h;
    h.abs_x = toml::find_or(g, "abs_x", h.abs_x);
    h.abs_y = toml::find_or(g, "abs_y", h.abs_y);
    h.neg_x = toml::find_or(g, "neg_x", h.neg_x);
    h.neg_y = toml::find_or(g, "neg_y", h.neg_y);
    h.power = toml::find_or(g, "power", h.power);
    h.opcodes = parse_opcodes(toml::find_or(g, "opcodes", print_opcodes(h.opcodes)));
    per.push_back(h);
  }
  if (! per.empty())
  {
    p.formula.per = per;
  }
#define LOAD(a,b) p.a.b = toml::find_or(t, #a, #b, p.a.b);
#define LOAD3(a,b,c) p.a.b.c = toml::find_or(t, #a, #b, #c, p.a.b.c);
  LOAD(location, real)
  LOAD(location, imag)
  LOAD(location, zoom)
  p.reference.real = p.location.real; // FIXME
  p.reference.imag = p.location.imag; // FIXME
  p.reference.period = 0; // FIXME
  LOAD(reference, real)
  LOAD(reference, imag)
  LOAD(reference, period)
  LOAD(bailout, iterations)
  LOAD(bailout, maximum_reference_iterations)
  LOAD(bailout, maximum_perturb_iterations)
  LOAD(bailout, maximum_bla_steps)
  LOAD(bailout, escape_radius)
  LOAD(bailout, inscape_radius)
  LOAD(algorithm, lock_maximum_reference_iterations_to_period)
  LOAD(algorithm, reuse_reference)
  LOAD(algorithm, reuse_bilinear_approximation)
  LOAD(algorithm, bla_skip_levels)
  LOAD(image, width)
  LOAD(image, height)
  LOAD(image, supersampling)
  LOAD(image, subsampling)
  LOAD(image, subframes)
  LOAD(image, dpi)
  LOAD(transform, reflect)
  LOAD(transform, rotate)
  LOAD(transform, stretch_angle)
  LOAD(transform, stretch_amount)
  LOAD(transform, exponential_map)
  LOAD(transform, vertical_flip)
  LOAD(postprocessing, brightness)
  LOAD(postprocessing, contrast)
  LOAD(postprocessing, gamma)
  LOAD(postprocessing, exposure)
  LOAD(render, filename)
  LOAD(render, zoom_out_sequence);
  LOAD(render, zoom_out_factor);
  LOAD(render, numbering_offset);
  LOAD(render, start_frame);
  LOAD(render, frame_count);
  LOAD(render, save_toml);
  LOAD(render, save_exr);
  LOAD(render, save_png);
  LOAD(render, save_jpg);
  LOAD(render, exr_channels);
  LOAD(newton, action);
  LOAD(newton, domain);
  LOAD(newton, absolute);
  LOAD(newton, power);
  LOAD(newton, factor);
  LOAD(opencl, platform);
  LOAD(opencl, device);
  LOAD(opencl, tile_width);
  LOAD(opencl, tile_height);
  LOAD(colour, shader)
  LOAD(colour, uniforms)
  LOAD3(colour, background, r)
  LOAD3(colour, background, g)
  LOAD3(colour, background, b)
  LOAD(colour, uses_histogram)
#undef LOAD
#undef LOAD3
  return ifs;
}

std::ostream &operator<<(std::ostream &ofs, const pparam &p)
{
  pparam q;
  ofs << "program = " << toml::value("fraktaler-3") << "\n";
  ofs << "version = " << toml::value(fraktaler_3_version_string) << "\n";
#define SAVE(a,b) if (p.a.b != q.a.b) { ofs << #a << "." << #b << " = " << toml::value(p.a.b) << "\n"; }
  SAVE(location, real)
  SAVE(location, imag)
  SAVE(location, zoom)
  if (p.location.real != p.reference.real) SAVE(reference, real)
  if (p.location.imag != p.reference.imag) SAVE(reference, imag)
  SAVE(reference, period)
  SAVE(bailout, iterations)
  SAVE(bailout, maximum_reference_iterations)
  SAVE(bailout, maximum_perturb_iterations)
  SAVE(bailout, maximum_bla_steps)
  SAVE(bailout, escape_radius)
  SAVE(bailout, inscape_radius)
  SAVE(algorithm, lock_maximum_reference_iterations_to_period)
  SAVE(algorithm, reuse_reference)
  SAVE(algorithm, reuse_bilinear_approximation)
  SAVE(algorithm, bla_skip_levels)
  SAVE(image, width)
  SAVE(image, height)
  SAVE(image, supersampling)
  SAVE(image, subsampling)
  SAVE(image, subframes)
  SAVE(image, dpi)
  SAVE(transform, reflect)
  SAVE(transform, rotate)
  SAVE(transform, stretch_angle)
  SAVE(transform, stretch_amount)
  SAVE(transform, exponential_map)
  SAVE(transform, vertical_flip)
  SAVE(postprocessing, brightness)
  SAVE(postprocessing, contrast)
  SAVE(postprocessing, gamma)
  SAVE(postprocessing, exposure)
  SAVE(render, filename)
  SAVE(render, zoom_out_sequence);
  SAVE(render, zoom_out_factor);
  SAVE(render, numbering_offset);
  SAVE(render, start_frame);
  SAVE(render, frame_count);
  SAVE(render, save_toml);
  SAVE(render, save_exr);
  SAVE(render, save_png);
  SAVE(render, save_jpg);
  SAVE(render, exr_channels);
  SAVE(newton, action);
  SAVE(newton, domain);
  SAVE(newton, absolute);
  SAVE(newton, power);
  SAVE(newton, factor);
  SAVE(opencl, platform);
  SAVE(opencl, device);
  SAVE(opencl, tile_width);
  SAVE(opencl, tile_height);
#undef SAVE
  if (! (p.formula == q.formula))
  {
    toml::array per;
    phybrid1 def;
    for (auto h : p.formula.per)
    {
      std::map<std::string, toml::value> f;
      if (h.opcodes.size())
      {
        f["opcodes"] = print_opcodes(h.opcodes);
      }
      else
      {
        if (h.abs_x != def.abs_x) f["abs_x"] = h.abs_x;
        if (h.abs_y != def.abs_y) f["abs_y"] = h.abs_y;
        if (h.neg_x != def.neg_x) f["neg_x"] = h.neg_x;
        if (h.neg_y != def.neg_y) f["neg_y"] = h.neg_y;
        if (h.power != def.power) f["power"] = h.power;
      }
      per.push_back(f);
    }
    std::map<std::string, toml::array> f;
    f["formula"] = per;
    ofs << toml::value(f) << "\n";
  }
  if (! (p.colour == q.colour))
  {
    std::map<std::string, toml::value> g;
    g["shader"] = p.colour.shader;
    g["uniforms"] = to_toml(p.colour.uniforms);
    std::map<std::string, toml::value> bg;
    bg["r"] = p.colour.background.r;
    bg["g"] = p.colour.background.g;
    bg["b"] = p.colour.background.b;
    g["background"] = bg;
    g["uses_histogram"] = p.colour.uses_histogram;
    std::map<std::string, toml::value> f;
    f["colour"] = g;
    ofs << toml::value(f) << "\n";
  }
  return ofs;
}

bool param::save_toml(const std::string &filename) const
{
  try
  {
    std::ofstream ofs;
    ofs.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    ofs.open(filename, std::ios_base::binary);
    ofs << p;
    return true;
  }
  catch (...)
  {
    return false;
  }
}

bool param::save_glsl(const std::string &filename) const
{
  try
  {
    std::ofstream ofs;
    ofs.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    ofs.open(filename, std::ios_base::binary);
    ofs << p.colour.shader;
    return true;
  }
  catch (...)
  {
    return false;
  }
}

std::string param::to_string() const
{
  std::ostringstream ofs;
  ofs << p;
  return ofs.str();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
std::vector<std::vector<opcode>> compile_formula(const phybrid &H)
{
  std::vector<std::vector<opcode>> result;
  for (const auto & h : H.per)
  {
    std::vector<opcode> current;
    if (h.opcodes.size())
    {
      bool need_store = false;
      for (const auto & op : h.opcodes)
      {
        if (op.op == op_mul)
        {
          need_store = true;
          break;
        }
        if (op.op == op_store)
        {
          break;
        }
      }
      if (need_store)
      {
        current.push_back({op_store});
      }
      current.insert(current.end(), h.opcodes.begin(), h.opcodes.end());
    }
    else
    {
      if (h.abs_x) current.push_back({op_absx});
      if (h.abs_y) current.push_back({op_absy});
      if (h.neg_x) current.push_back({op_negx});
      if (h.neg_y) current.push_back({op_negy});
      int p = h.power;
      bool is_power_of_two = (p & (p - 1)) == 0;
      if (! is_power_of_two)
      {
        current.push_back({op_store});
      }
      std::vector<opcode> power;
      while (p > 1)
      {
        if (p & 1)
        {
          assert(! is_power_of_two);
          power.push_back({op_mul});
        }
        power.push_back({op_sqr});
        p >>= 1;
      }
      std::reverse(power.begin(), power.end());
      // FIXME remove this validation code later
      int q = 1;
      for (const auto & op : power)
      {
        switch (op.op)
        {
          case op_mul: q += 1; break;
          case op_sqr: q <<= 1; break;
          case op_add:
          case op_store:
          case op_absx:
          case op_absy:
          case op_negx:
          case op_negy:
          case op_rot:
            assert(! "expected mul or sqr");
            break;
        }
      }
      assert(q == h.power);
      current.insert(current.end(), power.begin(), power.end());
      current.push_back({op_add});
    }
    result.push_back(current);
  }
  return result;
}
#pragma GCC diagnostic pop

int opcodes_degree(const std::vector<opcode> &ops)
{
  int deg_stored = 0;
  int deg = 1;
  for (const auto & op : ops)
  {
    switch (op.op)
    {
      case op_store: deg_stored = deg; break;
      case op_mul: deg += deg_stored; break;
      case op_sqr: deg <<= 1; break;
      // remainder have no effect
      case op_add:
      case op_absx:
      case op_absy:
      case op_negx:
      case op_negy:
      case op_rot:
        break;
    }
  }
  return deg;
}

std::string print_opcodes(const std::vector<opcode> &ops)
{
  std::ostringstream s;
  bool first_word = true;
  for (const auto & op : ops)
  {
    if (! first_word)
    {
      s << " ";
    }
    first_word = false;
    s << op_string[op.op];
    if (op.op == op_rot)
    {
      float degrees = std::atan2(op.u.rot.y, op.u.rot.x) * 180.0f / float(M_PI);
      s << "{" << degrees << "}";
    }
  }
  return s.str();
}

std::string print_opcodess(const std::vector<std::vector<opcode>> &opss)
{
  std::ostringstream s;
  bool first_line = true;
  for (const auto & ops : opss)
  {
    if (! first_line)
    {
      s << "\n";
    }
    first_line = false;
    s << print_opcodes(ops);
  }
  return s.str();
}

std::vector<int> opcodes_degrees(const std::vector<std::vector<opcode>> &opss)
{
  std::vector<int> result;
  for (const auto & ops : opss)
  {
    result.push_back(opcodes_degree(ops));
  }
  return result;
}

std::vector<std::vector<opcode>> parse_opcodess(const std::string &s)
{
  std::istringstream i(s);
  std::vector<std::vector<opcode>> result;
  std::vector<opcode> current;
  while (true)
  {
    std::string word;
    i >> word;
    if (word == "")
    {
      break;
    }
    int j;
    for (j = 0; j < op_count; ++j)
    {
      if (j == op_rot)
      {
        if (starts_with(word, "rot{") && ends_with(word, "}"))
        {
          float degrees = std::atof(word.c_str() + 4);
          float radians = degrees * float(M_PI) / 180.0f;
          opcode op;
          op.op = op_rot;
          op.u.rot.x = std::cos(radians);
          op.u.rot.y = std::sin(radians);
          current.push_back(op);
          break;
        }
      }
      else if (op_string[j] == word)
      {
        opcode_tag op((opcode_tag(j)));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
        current.push_back({op});
#pragma GCC diagnostic pop
        if (op == op_add)
        {
          result.push_back(current);
          current = std::vector<opcode>();
        }
        break;
      }
    }
    if (j == op_count)
    {
      throw std::out_of_range{"unrecognized opcode"};
    }
  }
  if (! current.empty())
  {
    result.push_back(current); // invalid formula, checked later
  }
  return result;
}

std::vector<opcode> parse_opcodes(const std::string &s)
{
  std::vector<std::vector<opcode>> r = parse_opcodess(s);
  if (r.size() == 0)
  {
    return std::vector<opcode>();
  }
  if (r.size() == 1)
  {
    return r[0];
  }
  else
  {
    throw std::out_of_range{"too many opcodes"};
  }
}

bool validate_opcodes(const std::vector<opcode> &ops)
{
  if (ops.empty())
  {
    return false;
  }
  bool have_store = false;
  bool have_add = false;
  for (const auto & op : ops)
  {
    if (have_add)
    {
      return false;
    }
    switch (op.op)
    {
      case op_add: have_add = true; break;
      case op_store: have_store = true; break;
      case op_mul: if (! have_store) return false; break;
      case op_sqr:
      case op_absx:
      case op_absy:
      case op_negx:
      case op_negy:
      case op_rot:
        break;
    }
  }
  if (! have_add)
  {
    return false;
  }
  if (! (opcodes_degree(ops) >= 2))
  {
    // return false; // can cause problems in GUI editor // FIXME
  }
  return true;
}

bool validate_opcodess(const std::vector<std::vector<opcode>> &opss)
{
  if (opss.empty())
  {
    return false;
  }
  for (const auto & ops : opss)
  {
    if (! validate_opcodes(ops))
    {
      return false;
    }
  }
  return true;
}

const char * const op_string[op_count] = { "add", "store", "mul", "sqr", "absx", "absy", "negx", "negy", "rot" };
