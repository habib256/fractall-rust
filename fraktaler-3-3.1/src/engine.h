// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <vector>

#include "types.h"
#include "float128.h"

struct phybrid;

extern const char *nt_string[
#ifdef HAVE_FLOAT128
  8
#else
  7
#endif
];

extern number_type nt_ref, nt_bla;

extern std::string pref_path;

#define newton_action_period 0
#define newton_action_center 1
#define newton_action_zoom 2
#define newton_action_transform 3

#define newton_zoom_minibrot_relative 0
#define newton_zoom_minibrot_absolute 1
#define newton_zoom_domain_relative 2
#define newton_zoom_domain_absolute 3

void populate_number_type_wisdom(void);
void delete_ref();
void delete_bla();
count_t getM(number_type nt, count_t phase);
void reference_thread(stats &sta, param &par, bool just_did_newton, progress_t *progress, volatile bool *running, volatile bool *ended);
void subframe_thread(coord_t frame, map &out, stats &sta, const param &par, const count_t subframe, progress_t *progress, volatile bool *running, volatile bool *ended);
void newton_thread(param &out, bool &ok, const param &par, const complex<floatexp<float, int>> &c, const floatexp<float, int> &r, volatile progress_t *progress, volatile bool *running, volatile bool *ended);

extern bool just_did_newton;
void set_reference_to_image_center(param &par);
bool calculate_reference(number_type nt, const param &par, progress_t *progress, volatile bool *running);
bool calculate_bla(number_type nt, const param &par, progress_t *progress, volatile bool *running);

#ifdef HAVE_FLOAT128
extern std::vector<std::vector<complex<float128>>> Zq;
#endif
extern std::vector<std::vector<complex<softfloat>>> Zsf;
extern std::vector<std::vector<complex<floatexp<double, int>>>> Zde;
extern std::vector<std::vector<complex<floatexp<float, int>>>> Zfe;
extern std::vector<std::vector<complex<long double>>> Zld;
extern std::vector<std::vector<complex<double>>> Zd;
extern std::vector<std::vector<complex<float>>> Zf;

#ifdef HAVE_FLOAT128
extern std::vector<blasR2<float128>> Bq;
#endif
extern std::vector<blasR2<softfloat>> Bsf;
extern std::vector<blasR2<floatexp<double, int>>> Bde;
extern std::vector<blasR2<floatexp<float, int>>> Bfe;
extern std::vector<blasR2<long double>> Bld;
extern std::vector<blasR2<double>> Bd;
extern std::vector<blasR2<float>> Bf;
