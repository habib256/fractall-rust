//! DIAGNOSTIC TEMPORAIRE : trajectoire d'un pixel du tier exp vs vérité GMP.
//! `DIAG_PNG=… DIAG_I=… DIAG_J=… [DIAG_BLA=1] cargo test --release --lib diag_exp -- --ignored --nocapture`

use rug::{Complex, Float};

use crate::fractal::bytecode::bla_dual_exp::BlaTableUnifiedExp;
use crate::fractal::bytecode::compile::formula_for_params;
use crate::fractal::perturbation::orbit::compute_reference_orbit;
use crate::fractal::perturbation::precision::{
    compute_perturbation_precision_bits, effective_spans_fexp,
};
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

fn fexp_to_gmp(f: FloatExp, prec: u32) -> Float {
    let mut x = Float::with_val(prec, f.mantissa);
    if f.mantissa != 0.0 {
        x <<= f.exponent;
    }
    x
}

#[derive(Clone, Copy, Debug)]
enum Ev {
    Head,
    Rebase,
    Bla(u32),
    Direct,
}

#[test]
#[ignore]
fn diag_exp_pixel_trajectory() {
    let png = std::env::var("DIAG_PNG").expect("DIAG_PNG");
    let pi: u32 = std::env::var("DIAG_I").unwrap().parse().unwrap();
    let pj: u32 = std::env::var("DIAG_J").unwrap().parse().unwrap();
    let use_bla = std::env::var("DIAG_BLA").ok().as_deref() == Some("1");
    let params0 = crate::io::png::load_png_metadata(std::path::Path::new(&png)).unwrap();
    let mut params = params0.clone();
    params.engine.precision_bits = compute_perturbation_precision_bits(&params0);
    let prec = params.engine.precision_bits;
    let (orbit, _, _) = compute_reference_orbit(&params, None, false).expect("orbit");
    eprintln!(
        "prec={prec} ref_len={} atom_truncated={} cycle_period={}",
        orbit.z_ref.len(),
        orbit.atom_truncated,
        orbit.cycle_period
    );
    let (sx, sy) = effective_spans_fexp(&params0);
    let w = params0.width as f64;
    let h = params0.height as f64;
    let dc = ComplexExp {
        re: if std::env::var("DIAG_INV").is_ok() { sx * ((pi as f64 + 0.5) * (1.0 / w) - 0.5) } else { sx * ((pi as f64 + 0.5) / w - 0.5) },
        im: if std::env::var("DIAG_INV").is_ok() { sy * ((pj as f64 + 0.5) * (1.0 / h) - 0.5) } else { sy * ((pj as f64 + 0.5) / h - 0.5) },
    };
    let formula = formula_for_params(&params0).unwrap();
    let bla = if use_bla {
        let px = sx.abs().div(FloatExp::from_f64(w)).max(sy.abs().div(FloatExp::from_f64(h)));
        let c_norm = px * (w * w + h * h).sqrt();
        Some(BlaTableUnifiedExp::build(
            &orbit.z_ref,
            &formula.phases[0],
            c_norm,
            FloatExp::from_f64(1.0 / (1u64 << 53) as f64),
        ))
    } else {
        None
    };

    // --- boucle exp (copie de iterate_pixel_unified_exp_mandelbrot_hp_impl) ---
    let ref_len = orbit.z_ref.len();
    let bailout_sqr = FloatExp::from_f64(params0.bailout * params0.bailout);
    let mut delta = ComplexExp::zero();
    let mut n: u32 = 0;
    let mut m: usize = 0;
    // (n, m, z au début de l'itération n, événement)
    let mut trace: Vec<(u32, usize, ComplexExp, Ev)> = Vec::new();
    let itmax = params0.iteration_max;
    let mut exp_iter = itmax;
    // Référence GMP exacte (pour DIAG_LO=1 : rebase avec Z exact au lieu de Z arrondi 53 b).
    let use_lo = std::env::var("DIAG_LO").ok().as_deref() == Some("1");
    let zref_true: Vec<Complex> = if use_lo {
        let gp = prec + 64;
        let mut zz = Complex::with_val(gp, (0, 0));
        let cr = Complex::with_val(gp, &orbit.cref_gmp);
        let mut v = Vec::with_capacity(ref_len);
        for _ in 0..ref_len {
            v.push(zz.clone());
            zz.square_mut();
            zz += &cr;
        }
        v
    } else {
        Vec::new()
    };
    let mut rebases: Vec<(u32, usize, ComplexExp)> = Vec::new();
    use crate::fractal::bytecode::reliability::{ShadowTrackerExp, StepObserverExp};
    let mut tr = ShadowTrackerExp::new(dc, 1.0 / (1u64 << 53) as f64);
    let mut ebound: std::collections::HashMap<u32, FloatExp> = Default::default();
    while n < itmax {
        let end_of_ref = m + 1 >= ref_len;
        let mut z_curr = orbit.z_ref[m].add(delta);
        if use_lo {
            let gp = prec + 64;
            let d = Complex::with_val(
                gp,
                (fexp_to_gmp(delta.re, gp), fexp_to_gmp(delta.im, gp)),
            );
            let s = Complex::with_val(gp, &zref_true[m] + &d);
            z_curr = ComplexExp::from_gmp(&s);
        }
        if end_of_ref || z_curr.norm_sqr_fexp() < delta.norm_sqr_fexp() {
            rebases.push((n, m, delta));
            tr.rebase(orbit.z_ref[m], z_curr);
            delta = z_curr;
            m = 0;
            trace.push((n, m, z_curr, Ev::Rebase));
        }
        let z_abs = orbit.z_ref[m].add(delta);
        trace.push((n, m, z_abs, Ev::Head));
        ebound.insert(n, tr.err);
        if !(z_abs.norm_sqr_fexp() < bailout_sqr) {
            exp_iter = n;
            break;
        }
        if let Some(t) = &bla {
            if let Some(node) = t.lookup_fexp(m, delta.norm_sqr_fexp()) {
                if n + node.l <= itmax && m + (node.l as usize) < ref_len {
                    let (a, b) = (node.a, node.b);
                    let cand = ComplexExp {
                        re: a.m00 * delta.re + a.m01 * delta.im + b.m00 * dc.re + b.m01 * dc.im,
                        im: a.m10 * delta.re + a.m11 * delta.im + b.m10 * dc.re + b.m11 * dc.im,
                    };
                    let over = node.l >= 2 && {
                        let ze = orbit.z_ref[m + node.l as usize].add(cand);
                        !(ze.norm_sqr_fexp() < bailout_sqr)
                    };
                    if !over {
                        if std::env::var("DIAG_LSCALE").is_ok() {
                            let mut t2 = ShadowTrackerExp::new(
                                dc,
                                node.l as f64 / (1u64 << 53) as f64,
                            );
                            t2.err = tr.err;
                            t2.bla(&a, &b, delta, dc, 1);
                            tr.err = t2.err;
                        } else {
                            tr.bla(&a, &b, delta, dc, node.l);
                        }
                        trace.push((n, m, z_abs, Ev::Bla(node.l)));
                        delta = cand;
                        n += node.l;
                        m += node.l as usize;
                        continue;
                    }
                }
            }
        }
        let z_m = orbit.z_ref[m];
        let two_z = ComplexExp {
            re: z_m.re + z_m.re,
            im: z_m.im + z_m.im,
        };
        tr.direct(z_abs, delta.mul(two_z), delta);
        delta = delta.mul(two_z).add(delta.mul(delta)).add(dc);
        trace.push((n, m, z_abs, Ev::Direct));
        n += 1;
        m += 1;
        if n % 250 == 0 {
            delta.reduce();
        }
    }
    eprintln!("exp iter = {exp_iter}  (trace {} events)", trace.len());
    eprintln!("rebases: {} (au bout de réf : {})", rebases.len(),
        rebases.iter().filter(|r| r.1 + 1 >= ref_len).count());
    for &(nn, mm, d) in rebases.iter().filter(|r| r.1 + 1 >= ref_len).take(6) {
        let dn = d.norm_sqr_fexp();
        let zr = orbit.z_ref[mm].norm_sqr_fexp();
        let zc = orbit.z_ref[mm].add(d).norm_sqr_fexp();
        eprintln!(
            "  end-rebase n={nn} m={mm} |δ|²=2^{} |Z|²=2^{} |Z+δ|²=2^{}",
            dn.exponent, zr.exponent, zc.exponent
        );
    }

    // --- vérité GMP ---
    let gp = prec + 64;
    let cx = Float::with_val(gp, Float::parse(params0.center_x_hp.as_ref().unwrap()).unwrap());
    let cy = Float::with_val(gp, Float::parse(params0.center_y_hp.as_ref().unwrap()).unwrap());
    let spx = Float::with_val(gp, Float::parse(params0.span_x_hp.as_ref().unwrap()).unwrap());
    let spy = Float::with_val(gp, Float::parse(params0.span_y_hp.as_ref().unwrap()).unwrap());
    let fx = Float::with_val(gp, (pi as f64 + 0.5) / w - 0.5); // exact (dyadique / 96 -> non) : ok à 1e-16 px
    let fy = Float::with_val(gp, (pj as f64 + 0.5) / h - 0.5);
    let c = if std::env::var("DIAG_CUSED").is_ok() {
        // c EFFECTIVEMENT utilisé par la boucle exp : cref + dc (FloatExp) exact.
        let d = Complex::with_val(gp, (fexp_to_gmp(dc.re, gp), fexp_to_gmp(dc.im, gp)));
        let cc = Complex::with_val(gp, &orbit.cref_gmp + &d);
        let c_ex = Complex::with_val(gp, (cx.clone() + spx.clone() * fx.clone(), cy.clone() + spy.clone() * fy.clone()));
        let diff = Complex::with_val(gp, &cc - &c_ex);
        let px_size = Float::with_val(gp, &spx / w);
        let dd = Float::with_val(gp, diff.abs_ref()) / px_size;
        eprintln!("c_used − c_exact = {:.3e} px", dd.to_f64());
        cc
    } else {
        Complex::with_val(gp, (cx + spx * fx, cy + spy * fy))
    };
    let mut z = Complex::with_val(gp, (0, 0));
    let mut true_iter = itmax;
    let mut zs_true: Vec<ComplexExp> = Vec::with_capacity(exp_iter as usize + 2);
    for k in 0..itmax {
        zs_true.push(ComplexExp::from_gmp(&z));
        let nrm = Float::with_val(64, z.real().clone().square() + z.imag().clone().square());
        if nrm >= 625.0 {
            true_iter = k;
            break;
        }
        z.square_mut();
        z += &c;
        if k as usize > exp_iter as usize + 10 {
            break;
        }
    }
    eprintln!("gmp iter = {true_iter}");

    // --- comparaison ---
    let rel = |a: ComplexExp, b: ComplexExp| -> f64 {
        let ga = Complex::with_val(gp, (fexp_to_gmp(a.re, gp), fexp_to_gmp(a.im, gp)));
        let gb = Complex::with_val(gp, (fexp_to_gmp(b.re, gp), fexp_to_gmp(b.im, gp)));
        let d = Complex::with_val(gp, &ga - &gb);
        let dn = Float::with_val(gp, d.abs_ref()).to_f64_exp();
        let bn = Float::with_val(gp, gb.abs_ref()).to_f64_exp();
        if bn.0 == 0.0 {
            return f64::INFINITY;
        }
        (dn.0 / bn.0) * 2f64.powi(dn.1 - bn.1)
    };
    let mut first_bad: Option<usize> = None;
    let mut worst = 0.0f64;
    let mut next_report = 1e-14;
    for (k, &(nn, mm, zz, ev)) in trace.iter().enumerate() {
        if !matches!(ev, Ev::Head) {
            continue;
        }
        if (nn as usize) >= zs_true.len() {
            break;
        }
        let r = rel(zz, zs_true[nn as usize]);
        if !r.is_finite() {
            continue;
        }
        if r > worst {
            worst = r;
        }
        if r > next_report {
            let zn = zs_true[nn as usize].norm_sqr_fexp();
            let eb = ebound.get(&nn).copied().unwrap_or(FloatExp::zero());
            let zabs = zs_true[nn as usize].norm_sqr_fexp().sqrt();
            let eb_rel = if zabs.mantissa != 0.0 { eb.div(zabs).to_f64() } else { f64::NAN };
            eprintln!(
                "  rel err {r:.3e} at n={nn} m={mm} |z|^2=({:.3}e2^{})  tracker E/|z|={eb_rel:.3e}",
                zn.mantissa, zn.exponent
            );
            next_report = r * 10.0;
        }
        if r > 1e-3 && first_bad.is_none() {
            first_bad = Some(k);
        }
    }
    eprintln!("worst rel = {worst:.3e}");
    if let Some(k) = first_bad {
        let lo = k.saturating_sub(40);
        for &(nn, mm, zz, ev) in &trace[lo..(k + 5).min(trace.len())] {
            let r = if (nn as usize) < zs_true.len() {
                rel(zz, zs_true[nn as usize])
            } else {
                f64::NAN
            };
            let zn = zz.norm_sqr_fexp();
            eprintln!(
                "    n={nn} m={mm} {ev:?} |z|^2=({:.4} 2^{}) rel={r:.3e}",
                zn.mantissa, zn.exponent
            );
        }
    }
}
