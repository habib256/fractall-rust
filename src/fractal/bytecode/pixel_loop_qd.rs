//! Boucle pixel perturbation **quad-double** (~212 b) — tier qd de la
//! correction de fiabilité (TODO G14.1 h), entre le dd (~106 b) et le GMP
//! par pixel.
//!
//! Chaîne : f64 signalé → dd suivi → dd sans BLA → **qd** → GMP par pixel.
//! Le qd n'est pas la vérité non plus : même garde-fou que le dd (borne
//! d'erreur absolue sur z, emballement collant, unité d'arrondi `U_QD`) ;
//! un pixel dont la borne qd s'emballe part au GMP sous le plafond existant.
//!
//! Mandelbrot mono-phase, pas DIRECTS uniquement (pas de BLA : une table à
//! coefficients qd n'est pas bâtie), rebasing F3 strict (`|Z[m]+δ|² < |δ|²`
//! OU bout de référence ; `wrap_periodic` pour les références cyclées).
//!
//! **Cadre mis à l'échelle** : δ et dc sont stockés multipliés par `2^s`
//! (`s` pair, `|dc|·2^s ≈ 1`). Le pas `δ' = (2Z + δ)·δ + dc` est formé comme
//! `A·B + dc_s` avec `A = (2Z + δ)·2^h`, `B = δ·2^{s−h}`, `h = s/2` : aucun
//! produit ne sort de la plage f64 tant que `s ≤ QD_MAX_SCALE` (au-delà le
//! pixel n'est pas traité en qd → GMP, comme avant). Les multiplications par
//! 2^k sont exactes hors sous-normal.

use num_complex::Complex64;

use super::pixel_loop::PixelLoopLimits;
use super::pixel_loop_exp::UnifiedPixelResultExp;
use crate::fractal::perturbation::counter::PixelCounter;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::qd::{ComplexQd, Qd};

/// Unité d'arrondi du qd pour le traqueur : erreur relative mesurée des
/// opérations sloppy ≤ 2⁻²¹⁰ (test `ops_match_rug_within_qd_unit`), marge ×16.
pub const U_QD: f64 = 1.214_306_433_183_765e-62; // 2^-206
/// Constante d'arrondi locale d'un pas complexe qd.
pub const GAMMA_QD: f64 = 4.0 * U_QD;

/// Échelle maximale `s` : |dc| ≥ 2⁻⁸⁴⁰ (~1e-253). Au-delà, les composantes
/// basses de δ et de Z (Z en qd SANS exposant : erreur absolue 2⁻¹⁰⁷⁴)
/// approcheraient la précision requise — le pixel reste au GMP.
pub const QD_MAX_SCALE: i32 = 840;

/// `FRACTALL_QD=0` coupe le tier qd (les pixels au-delà du dd vont au GMP
/// comme avant). Lu UNE fois.
pub fn qd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRACTALL_QD").ok().as_deref().map(str::trim),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// Fraction par défaut du plafond de l'escalade dd → qd (cf.
/// [`qd_escalation_allowed`]) : 1 % des pixels de la frame.
pub const DEFAULT_QD_MAX_FRACTION: f64 = 0.01;

/// Plafond de l'escalade dd → qd en fraction de pixels
/// (`FRACTALL_QD_MAX_FRACTION`, ≥ 1 = illimité). Lu une fois.
pub fn qd_max_fraction() -> f64 {
    static F: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("FRACTALL_QD_MAX_FRACTION")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|f| f.is_finite() && *f >= 0.0)
            .unwrap_or(DEFAULT_QD_MAX_FRACTION)
    })
}

/// Décision TOUT OU RIEN, déterministe (jamais en temps), de l'escalade
/// dd → qd : `n ≤ max(16, ⌊f·N⌋)` ou `f ≥ 1`. Même forme que le plafond
/// GMP (`reliability::gmp_escalation_allowed`).
pub fn qd_escalation_allowed(n_unreliable: usize, frame_pixels: usize, max_fraction: f64) -> bool {
    crate::fractal::bytecode::reliability::gmp_escalation_allowed(
        n_unreliable,
        frame_pixels,
        max_fraction,
    )
}

/// Échelle `s` (paire) pour un dc d'exposant binaire `dc_log2`
/// (|dc| < 2^dc_log2), ou `None` hors du domaine du tier qd.
pub fn qd_scale_for(dc_log2: i64) -> Option<i32> {
    let s = (-dc_log2).max(0);
    let s = (s + (s & 1)) as i32;
    (s <= QD_MAX_SCALE).then_some(s)
}

/// Traqueur de fiabilité qd, dans le cadre ×2^s : `e` = borne ABSOLUE sur z
/// multipliée par 2^s. Même récurrence que `ShadowTrackerDd` (pas direct
/// `E' = E·(2|z| + E) + γ·(|2Zδ| + |δ|² + |dc|)`, rebase `+u·(|Z| + |z|)`),
/// plus l'arrondi de la RÉFÉRENCE quand son orbite GMP est moins précise que
/// le qd (`u_ref·|Z[m+1]|` par pas : le pixel hérite de l'arrondi de la
/// référence à chaque itération).
#[derive(Clone, Copy, Debug)]
struct TrackerQd {
    e: f64,
    inv_scale: f64,
    scale: f64,
    dc_abs_s: f64,
    u_ref: f64,
    cap_s: f64,
}

impl TrackerQd {
    #[inline(always)]
    fn stick(&mut self) {
        if !(self.e < self.cap_s) {
            self.e = f64::INFINITY;
        }
    }

    /// `z_abs` = |Z+δ|, `two_z` = |2Z|, `d_s` = |δ|·2^s, `zn` = |Z[m+1]|.
    #[inline(always)]
    fn direct(&mut self, z_abs: f64, two_z: f64, d_s: f64, zn: f64) {
        let d = d_s * self.inv_scale;
        let local = GAMMA_QD * (two_z * d_s + d * d_s + self.dc_abs_s) + self.u_ref * zn * self.scale;
        self.e = self.e * (2.0 * z_abs + self.e * self.inv_scale) + local;
        self.stick();
    }

    #[inline(always)]
    fn rebase(&mut self, z_ref: f64, z_new: f64) {
        self.e += U_QD * (z_ref + z_new) * self.scale;
        self.stick();
    }

    fn bound(&self) -> f32 {
        if self.e.is_nan() {
            f32::INFINITY
        } else {
            (self.e * self.inv_scale) as f32
        }
    }
}

#[inline(always)]
fn l1(z: ComplexQd) -> f64 {
    z.re.hi().abs() + z.im.hi().abs()
}

#[inline(always)]
fn norm_hi(z: ComplexQd) -> f64 {
    let (r, i) = (z.re.hi(), z.im.hi());
    r * r + i * i
}

/// Pixel Mandelbrot en qd. `zq` = orbite qd (même longueur et mêmes champs
/// de cyclage que `ref_orbit`), `dc_s` = dc·2^s (exact depuis le GMP),
/// `ref_prec` = précision de l'orbite GMP source. Renvoie le résultat et la
/// borne d'erreur finale (`+∞` si emballée — le pixel n'est alors PAS
/// exploitable : la boucle s'arrête à l'emballement quand `track`).
pub fn iterate_pixel_qd_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    zq: &[ComplexQd],
    dc_s: ComplexQd,
    s: i32,
    ref_prec: u32,
    limits: PixelLoopLimits,
    track: bool,
) -> (UnifiedPixelResultExp, f32) {
    let ref_len = zq.len();
    let ret = |n: u32, z: Complex64, rebase_count: u32| UnifiedPixelResultExp {
        iteration: n,
        z_final: z,
        rebase_count,
        bla_steps: 0,
        ref_exhausted: false,
    };
    if ref_len < 2 {
        return (ret(0, Complex64::new(0.0, 0.0), 0), f32::INFINITY);
    }
    debug_assert!(s % 2 == 0 && (0..=QD_MAX_SCALE).contains(&s));
    let scale = 2f64.powi(s);
    let inv = 2f64.powi(-s);
    let sh = 2f64.powi(s / 2);
    let ish = 2f64.powi(-(s / 2));
    let two_sh = 2.0 * sh;
    let bailout_sqr = limits.bailout * limits.bailout;
    // Bande où la norme f64 ne tranche pas l'évasion : test qd exact.
    let (lo_b, hi_b) = (bailout_sqr * (1.0 - 1e-12), bailout_sqr * (1.0 + 1e-12));
    let runaway = crate::fractal::bytecode::reliability::reliability_runaway();
    // Arrondi de la RÉFÉRENCE (pseudo-orbite GMP à `ref_prec` bits) : compté
    // seulement si elle est moins précise que le qd. Au-delà, il est COMMUN
    // au qd et au GMP par pixel de repli (même précision de formule) : le
    // compter enverrait au GMP des pixels qu'il ne rendrait pas mieux.
    // Mesuré (golden_spider 256², 16 px) : compté, la borne vaut 0,1 à 7e7
    // à l'évasion — le z final n'est déterminé NI par le qd NI par le GMP à
    // 605 b (GMP 1210 b == 2420 b ≠ les deux, écart relatif ~1e-3), alors que
    // les ITÉRATIONS sont identiques aux trois précisions.
    let u_ref = if ref_prec >= 212 { 0.0 } else { 2f64.powi(-(ref_prec as i32 - 2)) };
    let mut tr = track.then(|| TrackerQd {
        e: 0.0,
        inv_scale: inv,
        scale,
        dc_abs_s: l1(dc_s),
        u_ref,
        cap_s: runaway * scale,
    });
    let mut delta_s = ComplexQd::ZERO;
    let mut c = PixelCounter::new();
    let mut rebase_count = 0u32;
    let iteration_max = limits.iteration_max;

    while c.keep_iterating(iteration_max) {
        let z_m0 = zq[c.m_usize()];
        let dlt = delta_s.mul_pow2(inv);
        // Rebase F3 avant l'évasion.
        {
            let end_of_ref = c.m_usize() + 1 >= ref_len;
            let zf = Complex64::new(z_m0.re.hi() + dlt.re.hi(), z_m0.im.hi() + dlt.im.hi());
            let rebase_now = if end_of_ref {
                match ref_orbit.wrap_periodic(c.m()) {
                    Some(mw) => {
                        c.set_ref_index(mw);
                        false
                    }
                    None => true,
                }
            } else {
                zf.norm_sqr() < norm_hi(dlt)
            };
            if rebase_now {
                delta_s = z_m0.mul_pow2(scale).add(delta_s);
                c.rebase();
                rebase_count += 1;
                if let Some(t) = tr.as_mut() {
                    t.rebase(l1(z_m0), l1(delta_s) * inv);
                }
            }
        }
        let z_m = zq[c.m_usize()];
        let dlt = delta_s.mul_pow2(inv);
        let zf = Complex64::new(z_m.re.hi() + dlt.re.hi(), z_m.im.hi() + dlt.im.hi());
        let nz = zf.norm_sqr();
        let escaped = if nz < lo_b {
            false
        } else if nz > hi_b || !nz.is_finite() {
            true
        } else {
            let z = z_m.add(dlt);
            let (re, im) = (z.re, z.im);
            let n2 = re.mul(re).add(im.mul(im));
            !(n2.sub(Qd::from_f64(bailout_sqr)).hi() < 0.0)
        };
        if escaped {
            let z = z_m.add(dlt);
            return (ret(c.n(), z.hi(), rebase_count), tr.map_or(0.0, |t| t.bound()));
        }
        // δ' = (2Z + δ)·δ + dc dans le cadre ×2^s.
        let a = z_m.mul_pow2(two_sh).add(delta_s.mul_pow2(ish));
        let b = delta_s.mul_pow2(ish);
        if let Some(t) = tr.as_mut() {
            let zn = if c.m_usize() + 1 < ref_len { l1(zq[c.m_usize() + 1]) } else { 0.0 };
            t.direct(zf.norm(), 2.0 * l1(z_m), l1(delta_s), zn);
        }
        delta_s = a.mul(b).add(dc_s);
        c.step();
        if let Some(t) = tr.as_ref() {
            if !(t.e < f64::INFINITY) {
                let m = c.m_usize().min(ref_len - 1);
                let z = zq[m].add(delta_s.mul_pow2(inv));
                return (ret(c.n(), z.hi(), rebase_count), f32::INFINITY);
            }
        }
        if !delta_s.re.hi().is_finite() || !delta_s.im.hi().is_finite() {
            let m = c.m_usize().min(ref_len - 1);
            return (ret(c.n(), zq[m].hi(), rebase_count), f32::INFINITY);
        }
    }
    let m = c.m_usize().min(ref_len - 1);
    let z = zq[m].add(delta_s.mul_pow2(inv));
    (ret(c.n(), z.hi(), rebase_count), tr.map_or(0.0, |t| t.bound()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::perturbation::qd::ComplexQd;
    use crate::fractal::perturbation::dd::{ComplexDDExp, DoubleDoubleExp};
    use rug::{Complex, Float};
    use std::time::Instant;

    /// Micro-banc de la porte de décision G14.1 (h) : coût par itération du
    /// GMP pur par pixel (`iterate_point_mpc`) à 605 et 857 b, du pas qd
    /// direct et du pas dd. `cargo test --release --lib qd_gate_bench --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn qd_gate_bench() {
        use crate::fractal::gmp::{iterate_point_mpc, MpcParams};
        use crate::fractal::FractalType;
        let n_gmp = 200_000u32;
        for prec in [605u32, 857] {
            let mut p = crate::fractal::default_params_for_type(FractalType::Mandelbrot, 64, 64);
            p.iteration_max = n_gmp;
            p.engine.precision_bits = prec;
            let mut g = MpcParams::from_params(&p);
            g.prec = prec;
            g.bailout_sqr = Float::with_val(prec, 625);
            g.seed = Complex::with_val(prec, (0, 0));
            // Point intérieur (cardioïde) : toutes les itérations sont faites.
            let c = Complex::with_val(prec, (-0.1, 0.1));
            let mut ns = f64::INFINITY;
            let mut it = 0;
            for _ in 0..5 {
                let t = Instant::now();
                it = iterate_point_mpc(&g, &c).0;
                ns = ns.min(t.elapsed().as_nanos() as f64 / it as f64);
            }
            eprintln!("[QD-GATE] gmp iterate_point_mpc prec={prec} : {ns:.0} ns/iter ({it} iters)");
        }
        // Orbite de référence intérieure (pas d'évasion), longue.
        let n = 500_000usize;
        let prec = 857;
        let cref = Complex::with_val(prec, (-0.1, 0.1));
        let mut zg = Complex::with_val(prec, (0, 0));
        let mut sc = Float::new(prec);
        let mut zq = Vec::with_capacity(n);
        let mut zdd = Vec::with_capacity(n);
        let (mut t1, mut t2) = (Float::new(prec), Float::new(prec));
        let t = Instant::now();
        for _ in 0..n {
            zq.push(ComplexQd::from_complex(&zg, &mut sc));
            crate::fractal::perturbation::orbit::fast_sqr_step(&mut zg, &cref, &mut t1, &mut t2);
        }
        eprintln!(
            "[QD-GATE] orbite GMP fast_sqr + conversion qd prec={prec} : {:.0} ns/iter (séquentiel)",
            t.elapsed().as_nanos() as f64 / n as f64
        );
        let mut zg = Complex::with_val(prec, (0, 0));
        for _ in 0..n {
            zdd.push(ComplexDDExp {
                re: DoubleDoubleExp::from_gmp(zg.real()),
                im: DoubleDoubleExp::from_gmp(zg.imag()),
            });
            crate::fractal::perturbation::orbit::fast_sqr_step(&mut zg, &cref, &mut t1, &mut t2);
        }
        let mut orbit = lean_orbit();
        orbit.z_ref_dd = zdd;
        let dc = Complex::with_val(prec, (1e-230, 1e-230));
        let s = qd_scale_for(-764).unwrap();
        let dc_s = Complex::with_val(prec, &dc << s as u32);
        let dc_s = ComplexQd::from_complex(&dc_s, &mut sc);
        for track in [false, true] {
            let mut ns = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                let (r, b) = iterate_pixel_qd_mandelbrot(
                    &orbit,
                    &zq,
                    dc_s,
                    s,
                    prec,
                    PixelLoopLimits::uncapped(n as u32 - 1, 25.0),
                    track,
                );
                ns = ns.min(t.elapsed().as_nanos() as f64 / r.iteration as f64);
                assert!(b < 1.0 && r.iteration == n as u32 - 1);
            }
            eprintln!("[QD-GATE] pas qd direct track={track} : {ns:.1} ns/iter (min de 5)");
        }
        let dc_dd = ComplexDDExp {
            re: DoubleDoubleExp::from_gmp(dc.real()),
            im: DoubleDoubleExp::from_gmp(dc.imag()),
        };
        let mut ns = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let r = crate::fractal::bytecode::pixel_loop_dd::iterate_pixel_unified_ddexp_mandelbrot(
                &orbit,
                None,
                dc_dd,
                ComplexDDExp::ZERO,
                PixelLoopLimits::uncapped(n as u32 - 1, 25.0),
            );
            ns = ns.min(t.elapsed().as_nanos() as f64 / r.iteration as f64);
        }
        eprintln!("[QD-GATE] pas dd direct (sans BLA) : {ns:.1} ns/iter (min de 5)");
    }

    /// Vérité : la boucle qd rend les MÊMES itérations que le GMP pur par
    /// pixel (`iterate_point_mpc`, 512 b) sur des grilles 8×8 (seahorse 1e6,
    /// e50 à 1e20 et 1e30, seahorse 1e200 — échelle s jusqu'à ~680), et
    /// aucune borne ne s'emballe sur ces scènes.
    #[test]
    fn qd_loop_matches_gmp_per_pixel() {
        use crate::fractal::gmp::{iterate_point_mpc, MpcParams};
        use crate::fractal::perturbation::orbit_qd::build_qd_orbit;
        use crate::fractal::FractalType;
        use rug::ops::Pow;
        let prec = 512u32;
        let seahorse = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139");
        let e50 = (
            "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000",
            "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880",
        );
        let iter_max = 20000u32;
        for (center, log10_zoom, varied) in [(seahorse, 6i32, true), (e50, 20, true), (e50, 30, false), (seahorse, 200, false)] {
            let cref = Complex::with_val(
                prec,
                (Float::parse(center.0).unwrap(), Float::parse(center.1).unwrap()),
            );
            // Orbite référence : jusqu'à l'évasion (|Z|² > 1e10) ou iter_max.
            let mut n = 1usize;
            {
                let mut z = Complex::new(prec);
                while n <= iter_max as usize {
                    let nz = Float::with_val(prec, z.real().square_ref()) + z.imag().clone().square();
                    if nz > 1e10 {
                        break;
                    }
                    z.square_mut();
                    z += &cref;
                    n += 1;
                }
            }
            let mut orbit = lean_orbit();
            orbit.cref_gmp = cref.clone();
            let zq = build_qd_orbit(&orbit, n).unwrap();
            let mut p = crate::fractal::default_params_for_type(FractalType::Mandelbrot, 16, 16);
            p.iteration_max = iter_max;
            p.engine.precision_bits = prec;
            let mut g = MpcParams::from_params(&p);
            g.prec = prec;
            g.bailout_sqr = Float::with_val(prec, 625);
            g.seed = Complex::new(prec);
            let mut sc = Float::new(prec);
            let limits = PixelLoopLimits::uncapped(iter_max, 25.0);
            let span = Float::with_val(prec, 4) / Float::with_val(prec, 10).pow(log10_zoom);
            let mut seen = std::collections::BTreeSet::new();
            for j in 0..8 {
                for i in 0..8 {
                    let fx = Float::with_val(prec, (i as f64 + 0.5) / 8.0 - 0.5) * &span;
                    let fy = Float::with_val(prec, (j as f64 + 0.5) / 8.0 - 0.5) * &span;
                    let dc = Complex::with_val(prec, (fx, fy));
                    let e = dc.real().get_exp().max(dc.imag().get_exp()).unwrap();
                    let s = qd_scale_for(e as i64).expect("échelle qd");
                    let dc_s = Complex::with_val(prec, &dc << s as u32);
                    let dq = ComplexQd::from_complex(&dc_s, &mut sc);
                    let (r, b) =
                        iterate_pixel_qd_mandelbrot(&orbit, &zq.z, dq, s, zq.prec, limits, true);
                    assert!(b < 1e30, "zoom 1e{log10_zoom} ({i},{j}) borne emballée {b:e}");
                    let c = Complex::with_val(prec, &cref + &dc);
                    let (it, _) = iterate_point_mpc(&g, &c);
                    assert_eq!(r.iteration, it, "zoom 1e{log10_zoom} pixel ({i},{j})");
                    seen.insert(it);
                }
            }
            // Anti-vacuité : la grille n'est pas uniforme (les niveaux
            // profonds — intérieur à 1e30, bande lisse à 1e200 — n'exercent
            // que l'échelle s, jusqu'à ~680).
            assert!(!varied || seen.len() >= 4, "scène triviale à 1e{log10_zoom} : {seen:?}");
        }
    }

    /// Même règle que `hot_path_sources_only_read_env_through_once_lock`
    /// (bytecode/tests.rs) pour les fichiers du tier qd : aucune lecture
    /// d'environnement hors `OnceLock` dans le chemin par pixel.
    #[test]
    fn qd_sources_only_read_env_through_once_lock() {
        let sources = [
            ("bytecode/pixel_loop_qd.rs", include_str!("pixel_loop_qd.rs")),
            ("perturbation/qd.rs", include_str!("../perturbation/qd.rs")),
            ("perturbation/orbit_qd.rs", include_str!("../perturbation/orbit_qd.rs")),
            ("perturbation/correct_qd.rs", include_str!("../perturbation/correct_qd.rs")),
        ];
        let mut offenders = Vec::new();
        for (name, text) in sources {
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let code = line.trim_start();
                if code.starts_with("//") || !code.contains(concat!("std::env::", "var")) {
                    continue;
                }
                let from = i.saturating_sub(8);
                if !lines[from..=i].iter().any(|l| l.contains("get_or_init")) {
                    offenders.push(format!("{name}:{} : {}", i + 1, code));
                }
            }
        }
        assert!(offenders.is_empty(), "lecture d'environnement hors OnceLock :\n{}", offenders.join("\n"));
    }

    /// Le traqueur est COLLANT et la boucle s'arrête à l'emballement : avec
    /// un dc à l'échelle s mais une orbite garbage (Z hors de l'ensemble,
    /// amplification maximale) la borne finit à +∞, jamais redescendue.
    #[test]
    fn qd_tracker_runaway_is_sticky() {
        let n = 2000;
        let mut zq = vec![ComplexQd::ZERO; n];
        let mut z = num_complex::Complex64::new(0.0, 0.0);
        let c = num_complex::Complex64::new(-1.9999, 0.0);
        for k in 1..n {
            z = z * z + c;
            zq[k] = ComplexQd { re: Qd::from_f64(z.re), im: Qd::from_f64(z.im) };
        }
        let orbit = lean_orbit();
        let s = qd_scale_for(-100).unwrap();
        let dq = ComplexQd { re: Qd::from_f64(0.5), im: Qd::ZERO };
        let (_, b) = iterate_pixel_qd_mandelbrot(
            &orbit,
            &zq,
            dq,
            s,
            1000,
            PixelLoopLimits::uncapped(n as u32 - 1, 25.0),
            true,
        );
        assert!(b.is_infinite() || b >= 1e30, "borne {b:e}");
    }

    pub(crate) fn lean_orbit() -> ReferenceOrbit {
        ReferenceOrbit {
            cref: Complex64::new(0.0, 0.0),
            z_ref: Vec::new(),
            z_ref_f64: Vec::new(),
            z_ref_gmp: Vec::new(),
            cref_gmp: Complex::new(64),
            phase_offset: 0,
            extended_iterations: Vec::new(),
            high_precision_data: Vec::new(),
            data_storage_interval: 1,
            cycle_period: 0,
            cycle_start: 0,
            z_ref_dd: Vec::new(),
            z_ref_lo: Vec::new(),
            atom_truncated: false,
            compressed_f64: None,
            hybrid_phase_refs: Vec::new(),
        }
    }
}
