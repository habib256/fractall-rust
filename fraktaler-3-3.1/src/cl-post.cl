// Fraktaler 3 -- fast deep escape time fractals
// Copyright (C) 2021-2025 Claude Heiland-Allen
// SPDX-License-Identifier: AGPL-3.0-only

        // z = f(C, Z, c, z)
}

        z2 = real_norm_complexdual(z);
        n++;
        m++;
        iters_ptb++;
        last_degree = next_degree;
      }

      {
        struct complex Z = { ref[config->ref_start[phase] + 2 * m], ref[config->ref_start[phase] + 2 * m + 1] };
        Zz = complexdual_add_complex_complexdual(Z, z);
        Zz2 = real_norm_complexdual(Zz);
        z2 = real_norm_complexdual(z);
        // rebase
        if (! (n < config->Iterations && bool_lt_real_real(Zz2, config->ER2) && iters_ptb < config->PerturbIterations))
        {
          break;
        }
        if (! (m < config->ref_size[phase]))
        {
          break;
        }
        if (bool_lt_real_real(Zz2, z2) || m + 1 == config->ref_size[phase])
        {
          z = Zz;
          z2 = Zz2;
          phase = (phase + m) % config->number_of_phases;
          m = 0;
        }
      }
    }

    // compute output
#ifdef HAVE_DOUBLE
    double Z1x = double_from_real(Zz.x.x);
    double Z1y = double_from_real(Zz.y.x);
    double Z12 = Z1x * Z1x + Z1y * Z1y;
    double Jxx = double_from_real(Zz.x.dx[0]);
    double Jxy = double_from_real(Zz.x.dx[1]);
    double Jyx = double_from_real(Zz.y.dx[0]);
    double Jyy = double_from_real(Zz.y.dx[1]);
    double dCx = Z1x * Jxx + Z1y * Jyx;
    double dCy = Z1x * Jxy + Z1y * Jyy;
    double dC2 = dCx * dCx + dCy * dCy;
    double dex = ((double)(config->subsampling)) / ((double)(config->supersampling)) * Z12 * log(sqrt(Z12)) * ( dCx / dC2);
    double dey = ((double)(config->subsampling)) / ((double)(config->supersampling)) * Z12 * log(sqrt(Z12)) * (-dCy / dC2);
    float nf = (float)(fmin(fmax(1 - log(log(Z12) / double_from_real(real_log_real(config->ER2))) / log((double)(last_degree)), 0.), 1.));
    float t = (float)(atan2(Z1y, Z1x)) / 6.283185307179586f;
    t -= floor(t);
    if (Z12 < double_from_real(config->ER2) || isnan(dex) || isinf(dex) || isnan(dey) || isinf(dey))
#else
    const struct complex Z1 = { Zz.x.x, Zz.y.x };
    const struct mat2 J = { Zz.x.dx[0], Zz.x.dx[1], Zz.y.dx[0], Zz.y.dx[1] };
    const struct complex dC = complex_mul_complex_mat2(Z1, J);
    const real Z1norm = real_norm_complex(Z1);
    struct complex de = complex_div_real_complex(real_mul_real_real(real_mul_real_real(real_div_real_real(real_from_long(config->subsampling), real_from_long(config->supersampling)), Z1norm), real_div2_real(real_log_real(Z1norm))), dC);
    float dex = float_from_real(de.x);
    float dey = float_from_real(de.y);
    float nf = clamp(1.0f - log(float_from_real(real_log_real(Z1norm)) / float_from_real(real_log_real(config->ER2))) / log((float) last_degree), 0.0f, 1.0f);
    float t = atan2(float_from_real(Z1.y), float_from_real(Z1.x)) / 6.283185307179586f;
    t -= floor(t);
    if (bool_lt_real_real(Z1norm, config->ER2) || isnan(dex) || isinf(dex) || isnan(dey) || isinf(dey))
#endif
    {
      n = config->Iterations;
      nf = 0;
      t = 0;
      dex = 0;
      dey = 0;
    }
    const long k = (j - y0) * config->tile_width + (i - x0);
    /* output raw */
    const long Nbias = 1024;
    ulong nn = n + Nbias;
    if (n >= config->Iterations)
    {
      nn = ~((ulong)(0));
    }
    if (N0)
    {
      N0[k] = nn;
    }
    if (N1)
    {
      N1[k] = nn >> 32;
    }
    if (NF)
    {
      NF[k] = nf;
    }
    if (T)
    {
      T[k] = t;
    }
    if (DEX)
    {
      DEX[k] = dex;
    }
    if (DEY)
    {
      DEY[k] = dey;
    }
    if (BLA)
    {
      BLA[k] = steps_bla;
    }
    if (PTB)
    {
      PTB[k] = iters_ptb;
    }
  }
}
