//! Boucle pixel « double mis à l'échelle » (scaled double) pour le tier exp
//! (zoom > 1e280), Mandelbrot atom-tronqué — **par défaut** depuis le
//! 2026-11-06 (`FRACTALL_EXP_SCALED=0` revient à la boucle FloatExp).
//!
//! Activation décidée sur preuve d'identité : PNG identiques à l'octet à la
//! boucle FloatExp sur les 37 cas du tier exp du corpus `toml/` à 192² (tous
//! sauf e52465, hors budget temps), 17 à 256², 5 à 1024² — pour un temps de
//! phase pixel cumulé de 25,6 → 7,1 s (×2 à ×7 par cas).
//!
//! Le path HP de référence (`pixel_loop_exp::iterate_pixel_unified_exp_
//! mandelbrot_hp_impl`) itère `δ` en `ComplexExp` : chaque opération `FloatExp`
//! renormalise (`frexp`) et chaque addition aligne les exposants. Ici
//! `δ = d · 2^e` avec `d: Complex64` dans une plage confortable et UN exposant
//! commun, `dc = dcm · 2^ec` de même ; un pas direct devient de l'arithmétique
//! f64 pure (`d' = 2·Z·d + (d·d)·2^e + dcm·2^(ec−e)`), renormalisée seulement
//! quand `d` sort de `[2⁻⁶⁴, 2⁶⁴]`.
//!
//! **Pourquoi c'est le MÊME calcul** : `FloatExp` arrondit chaque opération à
//! 53 bits avec des mises à l'échelle par puissances de 2 exactes — c'est de la
//! f64 à plage d'exposant illimitée. La f64 mise à l'échelle fait les mêmes
//! arrondis tant qu'aucune valeur SIGNIFICATIVE n'est sous-normale. Chaque pas
//! rapide vérifie, pour chaque addition `x + y`, que `max(|x|,|y|) ≥ T = 2⁻⁸⁰⁰`
//! (partenaire « grand ») : un opérande sous-débordé (vraie valeur < 2⁻¹⁰²²) est
//! alors négligé à l'identique par `FloatExp` (écart d'exposant ≥ 54 → abandon)
//! et par l'arrondi f64, et le résultat est normal. Sinon le pas est REFAIT par
//! le chemin lent (copie exacte du pas `ComplexExp`). Les décisions de rebase et
//! de bailout ne sont prises en rapide que lorsqu'elles sont triviales avec une
//! marge (|δ| ≤ |Z|/4 ; `2(|Z|²+|δ|²) < ER²`) ; sinon tout le tour de boucle
//! passe par le chemin lent — c'est le cas près de la fermeture de période
//! atom-domain (|Z| ~ 1e-1200, `z_ref_f64` = 0) où ont lieu les rebases.
//!
//! Seul écart connu à l'arithmétique `FloatExp` : `FloatExp::add` ABANDONNE un
//! opérande d'exposant inférieur de ≥ 54 là où l'arrondi f64 correct, si l'autre
//! opérande est une puissance de 2 EXACTE de signe opposé, descend d'un ulp
//! (probabilité ~2⁻⁵² par addition sur des données génériques).

use std::sync::OnceLock;

use num_complex::Complex64;

use super::bla_dual_exp::{BlaTableUnifiedExp, Mat2Exp};
use super::pixel_loop_exp::UnifiedPixelResultExp;
use super::reliability::{NoObserve, ShadowTrackerExp, StepObserverExp};
use super::{Formula, Op};
use crate::fractal::perturbation::counter::PixelCounter;
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

/// Actif par défaut ; `FRACTALL_EXP_SCALED=0` (lu une fois) le coupe (A/B,
/// diagnostic — la boucle FloatExp reste la référence du verrou d'identité).
pub fn scaled_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("FRACTALL_EXP_SCALED").ok().as_deref() != Some("0"))
}

/// Diagnostic `FRACTALL_EXP_SCALED_STATS=1` : ventilation des tours de boucle.
fn stats_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("FRACTALL_EXP_SCALED_STATS").ok().as_deref() == Some("1"))
}

/// Mirror de `pixel_loop_exp::atom_nobla_debug` (privé là-bas).
fn nobla_debug() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("FRACTALL_ATOM_NOBLA").ok().as_deref() == Some("1"))
}

/// Formule exactement `[Sqr, Add]` mono-phase (mêmes conditions que le path HP).
pub fn is_plain_mandelbrot(formula: &Formula) -> bool {
    formula.phases.len() == 1
        && formula.phases[0].ops.len() == 2
        && matches!(formula.phases[0].ops[0], Op::Sqr)
        && matches!(formula.phases[0].ops[1], Op::Add)
}

/// 2^k en f64, 0 sous 2⁻¹⁰²² (jamais de sous-normal), +∞ au-dessus de 2¹⁰²³.
#[inline(always)]
fn p2(k: i32) -> f64 {
    if k < -1022 {
        0.0
    } else if k > 1023 {
        f64::INFINITY
    } else {
        f64::from_bits(((k + 1023) as u64) << 52)
    }
}

/// Seuil « partenaire grand » des additions (cf. doc du module).
const T: f64 = f64::from_bits(((-800 + 1023) as u64) << 52);
/// Plage de la mantisse commune (composante max).
const D_LO: f64 = f64::from_bits(((-64 + 1023) as u64) << 52);
const D_HI: f64 = f64::from_bits(((64 + 1023) as u64) << 52);
/// Plancher d'une composante non nulle après renormalisation (normale, marge).
const MIN_COMP: f64 = f64::from_bits(((-1000 + 1023) as u64) << 52);
/// |Z|² minimal pour décider « pas de rebase » en rapide.
const NZ_MIN: f64 = f64::from_bits(((-850 + 1023) as u64) << 52);
/// Marge relative des décisions conservatrices (bailout).
const MARGIN: f64 = 1.0 + 1.0 / (1u64 << 30) as f64;
/// Exposant sentinelle d'un coefficient nul (→ terme 0 exact).
const NEG: i32 = -(1 << 28);
/// Exposant de cadre maximal toléré : évite tout débordement de d²·2^e ET
/// garantit qu'un produit `d.re·d.im` sous-débordé (< 2⁻¹⁰²²) reste, après
/// ×2^e, < 2⁻⁹²⁰ — négligeable devant un partenaire ≥ T (en pratique e ≤ ~70 :
/// |δ| < ER et composante max ≥ 2⁻⁶⁴).
const E_MAX: i32 = 100;

/// `δ = d · 2^e`, exposant commun aux deux composantes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaled {
    pub d: Complex64,
    pub e: i32,
}

impl Scaled {
    /// Conversion EXACTE depuis `ComplexExp` ; `None` si δ = 0 ou si une
    /// composante non nulle est > 2⁹⁰⁰ fois plus petite que l'autre (elle
    /// deviendrait sous-normale sous l'exposant commun).
    #[inline]
    pub fn from_cexp(x: ComplexExp) -> Option<Self> {
        let (mr, er) = (x.re.mantissa, x.re.exponent);
        let (mi, ei) = (x.im.mantissa, x.im.exponent);
        let rz = mr == 0.0;
        let iz = mi == 0.0;
        if rz && iz {
            return None;
        }
        let e = if rz {
            ei
        } else if iz {
            er
        } else {
            er.max(ei)
        };
        let (kr, ki) = (er - e, ei - e);
        if (!rz && kr < -900) || (!iz && ki < -900) {
            return None;
        }
        let re = if rz { 0.0 } else { mr * p2(kr) };
        let im = if iz { 0.0 } else { mi * p2(ki) };
        Some(Self {
            d: Complex64::new(re, im),
            e,
        })
    }

    /// Conversion EXACTE vers `ComplexExp` (composantes normales ou nulles).
    #[inline]
    pub fn to_cexp(self) -> ComplexExp {
        ComplexExp {
            re: FloatExp::new(self.d.re, self.e),
            im: FloatExp::new(self.d.im, self.e),
        }
    }
}

/// Ramène la composante max de `d` dans `[2⁻⁶⁴, 2⁶⁴]` (mise à l'échelle exacte).
/// `None` (⇒ le caller refait le tour en lent) si `d` = 0 ou si la petite
/// composante tomberait sous `MIN_COMP`.
#[inline(always)]
fn renorm(d: Complex64, e: i32) -> Option<Scaled> {
    let mx = d.re.abs().max(d.im.abs());
    if (D_LO..=D_HI).contains(&mx) {
        return Some(Scaled { d, e });
    }
    let biased = ((mx.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 || biased == 0x7ff {
        return None; // zéro, sous-normal, inf ou NaN
    }
    let k = biased - 1023;
    let s = p2(-k);
    let nd = Complex64::new(d.re * s, d.im * s);
    let ok = (nd.re == 0.0 || nd.re.abs() >= MIN_COMP) && (nd.im == 0.0 || nd.im.abs() >= MIN_COMP);
    if ok {
        Some(Scaled { d: nd, e: e + k })
    } else {
        None
    }
}

#[inline(always)]
fn big(a: f64, b: f64) -> bool {
    a.abs() >= T || b.abs() >= T
}

/// Seuil abaissé (2⁻⁹⁶⁰) : un opérande sous-débordé (< 2⁻¹⁰²²) reste à ≥ 62
/// binades du partenaire (abandon FloatExp ET arrondi f64 à l'identique), et
/// le résultat reste normal (≥ 2⁻¹⁰¹³).
const T2: f64 = f64::from_bits(((-960 + 1023) as u64) << 52);

#[inline(always)]
fn big2(a: f64, b: f64) -> bool {
    a.abs() >= T2 || b.abs() >= T2
}

/// Pas direct quand Z est EXACTEMENT nul (m = 0, juste après un rebase) :
/// FloatExp calcule `2·Z·δ = 0` exact puis `0 + δ² = δ²` exact, donc
/// `δ' = δ² + dc`. Gardes : δ² sans perte (normal, ou zéro vrai) puis
/// partenaire de l'addition de `dc`.
#[inline(always)]
fn direct_fast_zero_z(s: Scaled, dcs: Scaled) -> Option<(Scaled, f64, f64)> {
    let kc = dcs.e - s.e;
    if s.e > E_MAX || kc > 900 {
        return None;
    }
    let d = s.d;
    let sc = p2(s.e);
    let diff = d.re * d.re - d.im * d.im;
    let q_re = diff * sc;
    let pr = d.re * d.im;
    let q_im = (pr + d.im * d.re) * sc;
    let t = p2(kc);
    let c_re = dcs.d.re * t;
    let c_im = dcs.d.im * t;
    let ok = (diff == 0.0 || q_re.abs() >= MIN_COMP)
        & (pr == 0.0 || (pr.abs() >= MIN_COMP && q_im.abs() >= MIN_COMP))
        & big(q_re, c_re)
        & big(q_im, c_im);
    if !ok {
        return None;
    }
    Some((
        renorm(Complex64::new(q_re + c_re, q_im + c_im), s.e)?,
        0.0,
        0.0,
    ))
}

/// Pas direct `δ' = 2·Z·δ + δ² + dc` dans le cadre de `s`, même ordre
/// d'opérations que `ComplexExp::mul/add`. `None` si une garde échoue.
#[inline(always)]
/// Renvoie aussi `2·Z·δ` dans le cadre de `s` (exact sous les gardes : entrée
/// du hook `direct` du traqueur).
fn direct_fast(s: Scaled, z: Complex64, dcs: Scaled) -> Option<(Scaled, f64, f64)> {
    let kc = dcs.e - s.e;
    if s.e > E_MAX || kc > 900 {
        return None;
    }
    let d = s.d;
    let (tzr, tzi) = (z.re + z.re, z.im + z.im);
    let p1 = d.re * tzr;
    let p2_ = d.im * tzi;
    let p3 = d.re * tzi;
    let p4 = d.im * tzr;
    let tzd_re = p1 - p2_;
    let tzd_im = p3 + p4;
    let sc = p2(s.e);
    let q_re = (d.re * d.re - d.im * d.im) * sc;
    let q_im = (d.re * d.im + d.im * d.re) * sc;
    let u_re = tzd_re + q_re;
    let u_im = tzd_im + q_im;
    let t = p2(kc);
    let c_re = dcs.d.re * t;
    let c_im = dcs.d.im * t;
    let r_re = u_re + c_re;
    let r_im = u_im + c_im;
    let ok = big(p1, p2_)
        & big(p3, p4)
        & big(tzd_re, q_re)
        & big(tzd_im, q_im)
        & big(u_re, c_re)
        & big(u_im, c_im);
    if !ok {
        return None;
    }
    Some((renorm(Complex64::new(r_re, r_im), s.e)?, tzd_re, tzd_im))
}

#[inline(always)]
fn ex(x: FloatExp) -> i32 {
    if x.mantissa == 0.0 {
        NEG
    } else {
        x.exponent
    }
}

/// Termes du saut pour le hook `bla_scaled` du traqueur (cadre 2^ep).
#[derive(Clone, Copy)]
struct JumpTerms {
    ad: Complex64,
    bt: [f64; 4],
    ep: i32,
}

/// Saut BLA `δ' = A·δ + B·dc` (mêmes produits et même ordre d'additions que
/// `pixel_loop_exp`), cadre de sortie = plus grand exposant de terme.
#[inline(always)]
fn jump_fast(s: Scaled, dcs: Scaled, a: &Mat2Exp, b: &Mat2Exp) -> Option<(Scaled, JumpTerms)> {
    let ea = ex(a.m00).max(ex(a.m01)).max(ex(a.m10)).max(ex(a.m11));
    let eb = ex(b.m00).max(ex(b.m01)).max(ex(b.m10)).max(ex(b.m11));
    if ea == NEG && eb == NEG {
        return None;
    }
    let ep = (ea + s.e).max(eb + dcs.e);
    let term = |coef: FloatExp, x: f64, xe: i32| (coef.mantissa * x) * p2(ex(coef) + xe - ep);
    let (dr, di, cr, ci) = (s.d.re, s.d.im, dcs.d.re, dcs.d.im);
    let t1 = term(a.m00, dr, s.e);
    let t2 = term(a.m01, di, s.e);
    let t3 = term(b.m00, cr, dcs.e);
    let t4 = term(b.m01, ci, dcs.e);
    let s1 = t1 + t2;
    let s2 = s1 + t3;
    let re = s2 + t4;
    let v1 = term(a.m10, dr, s.e);
    let v2 = term(a.m11, di, s.e);
    let v3 = term(b.m10, cr, dcs.e);
    let v4 = term(b.m11, ci, dcs.e);
    let w1 = v1 + v2;
    let w2 = w1 + v3;
    let im = w2 + v4;
    let ok = big(t1, t2) & big(s1, t3) & big(s2, t4) & big(v1, v2) & big(w1, v3) & big(w2, v4);
    if !ok {
        return None;
    }
    let terms = JumpTerms {
        ad: Complex64::new(s1, w1),
        bt: [t3, t4, v3, v4],
        ep,
    };
    Some((renorm(Complex64::new(re, im), ep)?, terms))
}

#[path = "pixel_loop_scaled_track.rs"]
mod track;
pub use track::ScaledTracker;
#[path = "pixel_loop_scaled_stats.rs"]
mod stats;
pub use stats::ScaledStats;
use stats::{fallback_reason, flush_stats};

/// Observateur de la boucle scaled : les hooks de [`StepObserverExp`] (même
/// traqueur, mêmes entrées ComplexExp que la boucle FloatExp) + `TRACKS` pour
/// ne PAS calculer ces entrées sans traqueur.
pub trait ScaledObs: StepObserverExp {
    const TRACKS: bool;
    /// Hook `direct` en f64 dans le cadre 2^e ; `false` ⇒ hook ComplexExp.
    #[inline(always)]
    fn direct_scaled(&mut self, _z: Complex64, _d: Complex64, _nd: f64, _e: i32, _tzd: Complex64) -> bool {
        false
    }
    /// Hook `bla` en f64 (cadre 2^ep) ; `sigma` = σ₁(A) précalculé du nœud
    /// (`BlaMultiStepExp::sigma_m · 2^sigma_e`) ; `false` ⇒ hook ComplexExp.
    #[inline(always)]
    fn bla_scaled(
        &mut self,
        _sigma: (f64, i32),
        _ad: Complex64,
        _bt: [f64; 4],
        _ep: i32,
        _l: u32,
    ) -> bool {
        false
    }
}

impl ScaledObs for NoObserve {
    const TRACKS: bool = false;
}

impl ScaledObs for ShadowTrackerExp {
    const TRACKS: bool = true;
    #[inline(always)]
    fn direct_scaled(&mut self, z: Complex64, d: Complex64, _nd: f64, e: i32, tzd: Complex64) -> bool {
        ShadowTrackerExp::direct_scaled(self, z, d, e, tzd)
    }
    #[inline(always)]
    fn bla_scaled(&mut self, sigma: (f64, i32), ad: Complex64, bt: [f64; 4], ep: i32, l: u32) -> bool {
        ShadowTrackerExp::bla_scaled(self, sigma, ad, bt, ep, l)
    }
}

enum Slow {
    Continue,
    Escaped(UnifiedPixelResultExp),
}

/// UN tour de boucle du path HP `ComplexExp`, copie EXACTE de
/// `iterate_pixel_unified_exp_mandelbrot_hp_impl` (observateur `NoObserve`).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn slow_iter<O: ScaledObs>(
    obs: &mut O,
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta: &mut ComplexExp,
    c: &mut PixelCounter,
    rebase_count: &mut u32,
    bla_steps: &mut u32,
    iters_ptb: &mut u32,
    iteration_max: u32,
    bailout_sqr_fexp: FloatExp,
    use_bla: bool,
) -> Slow {
    let ref_len = ref_orbit.z_ref.len();
    let end_of_ref = c.m_usize() + 1 >= ref_len;
    let z_m0 = ref_orbit.z_ref[c.m_usize()];
    let z_curr = z_m0.add(*delta);
    let rebase_mid = !end_of_ref && z_curr.norm_sqr_fexp() < delta.norm_sqr_fexp();
    if end_of_ref {
        if let Some(m_wrapped) = ref_orbit.wrap_periodic(c.m()) {
            c.set_ref_index(m_wrapped);
        } else {
            obs.rebase(z_m0, z_curr);
            *delta = z_curr;
            c.rebase();
            *rebase_count += 1;
        }
    } else if rebase_mid {
        obs.rebase(z_m0, z_curr);
        *delta = z_curr;
        c.rebase();
        *rebase_count += 1;
    }

    let z_m = ref_orbit.z_ref[c.m_usize()];
    let z_abs = z_m.add(*delta);
    if !(z_abs.norm_sqr_fexp() < bailout_sqr_fexp) {
        let delta_approx = delta.to_complex64_approx();
        return Slow::Escaped(UnifiedPixelResultExp {
            iteration: c.n(),
            z_final: ref_orbit.z_ref_f64[c.m_usize()] + delta_approx,
            rebase_count: *rebase_count,
            bla_steps: *bla_steps,
            ref_exhausted: false,
        });
    }

    if use_bla {
        if let Some(bla) = bla_exp {
            if let Some(node) = bla.lookup_fexp(c.m_usize(), delta.norm_sqr_fexp()) {
                let cand_c = c.after_jump(node.l);
                if cand_c.n() <= iteration_max && cand_c.m_usize() < ref_len {
                    let a = node.a;
                    let b = node.b;
                    let new_re =
                        a.m00 * delta.re + a.m01 * delta.im + b.m00 * dc.re + b.m01 * dc.im;
                    let new_im =
                        a.m10 * delta.re + a.m11 * delta.im + b.m10 * dc.re + b.m11 * dc.im;
                    let cand = ComplexExp {
                        re: new_re,
                        im: new_im,
                    };
                    let overshoots_escape = node.l >= 2 && {
                        let z_end = ref_orbit.z_ref[cand_c.m_usize()].add(cand);
                        !(z_end.norm_sqr_fexp() < bailout_sqr_fexp)
                    };
                    if !overshoots_escape {
                        obs.bla(&a, &b, *delta, dc, node.l);
                        *delta = cand;
                        *c = cand_c;
                        *bla_steps += 1;
                        // `delta.reduce()` du path HP : no-op sur des FloatExp
                        // normalisés (toutes les opérations normalisent).
                        return Slow::Continue;
                    }
                }
            }
        }
    }

    let two_z = ComplexExp {
        re: z_m.re + z_m.re,
        im: z_m.im + z_m.im,
    };
    let two_z_delta = delta.mul(two_z);
    obs.direct(z_abs, two_z_delta, *delta);
    let delta_sq = delta.mul(*delta);
    *delta = two_z_delta.add(delta_sq).add(dc);
    c.step();
    *iters_ptb += 1;
    Slow::Continue
}

/// Boucle pixel scaled-double. Même signature logique et MÊME résultat que
/// `iterate_pixel_unified_exp_mandelbrot_hp_impl` (Mandelbrot `[Sqr, Add]`,
/// non-Julia, atom-tronqué, sans traqueur de fiabilité).
pub fn iterate_pixel_scaled_exp_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
) -> UnifiedPixelResultExp {
    let (res, st) = scaled_impl(
        ref_orbit,
        bla_exp,
        dc,
        delta_initial,
        limits,
        &mut NoObserve,
    );
    if stats_enabled() {
        flush_stats(&st);
    }
    res
}

/// Variante avec **détecteur de fiabilité** (tier exp) : MÊME résultat et MÊME
/// borne que `pixel_loop_exp::iterate_pixel_unified_exp_mandelbrot_hp_tracked`
/// — mêmes hooks, mêmes événements, mêmes valeurs (f64 exact sous gardes).
pub fn iterate_pixel_scaled_exp_mandelbrot_tracked(
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
    bla_epsilon: f64,
) -> (UnifiedPixelResultExp, ShadowTrackerExp) {
    // Traqueur léger (borne em·2^ee, cf. `pixel_loop_scaled_track`) : mêmes
    // décisions, pas la même borne au bit.
    let mut tracker = ScaledTracker::new(ShadowTrackerExp::new(dc, bla_epsilon));
    let (res, st) = scaled_impl(ref_orbit, bla_exp, dc, delta_initial, limits, &mut tracker);
    if stats_enabled() {
        flush_stats(&st);
    }
    tracker.sync();
    (res, tracker.trk)
}

/// Corps de la boucle, avec la ventilation des tours (rapide/lent).
pub fn iterate_pixel_scaled_with_stats(
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
) -> (UnifiedPixelResultExp, ScaledStats) {
    scaled_impl(
        ref_orbit,
        bla_exp,
        dc,
        delta_initial,
        limits,
        &mut NoObserve,
    )
}

/// Corps générique (observateur copié en local, réécrit à la sortie).
pub fn scaled_impl<O: ScaledObs>(
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
    obs_out: &mut O,
) -> (UnifiedPixelResultExp, ScaledStats) {
    let mut obs = *obs_out;
    let r = scaled_body(ref_orbit, bla_exp, dc, delta_initial, limits, &mut obs);
    *obs_out = obs;
    r
}

#[inline(always)]
fn scaled_body<O: ScaledObs>(
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
    obs: &mut O,
) -> (UnifiedPixelResultExp, ScaledStats) {
    let crate::fractal::bytecode::pixel_loop::PixelLoopLimits {
        iteration_max,
        bailout,
        max_perturb_iterations,
        max_bla_steps,
    } = limits;
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref.len();
    if ref_len < 2 || ref_orbit.z_ref_f64.len() != ref_len {
        let res = UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
        return (res, ScaledStats::default());
    }
    let bailout_sqr_fexp = FloatExp::from_f64(bailout_sqr);
    let bailout_fast = bailout_sqr / MARGIN;
    let use_bla = !nobla_debug();
    let bla = if use_bla { bla_exp } else { None };
    let z_ref_f64 = &ref_orbit.z_ref_f64[..];
    let dcs = Scaled::from_cexp(dc);
    let mut st = ScaledStats::default();

    let mut delta = delta_initial; // valide en mode lent
    let mut fast: Option<Scaled> = dcs.and_then(|_| Scaled::from_cexp(delta));
    let mut c = PixelCounter::new();
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;

    while c.keep_iterating(iteration_max)
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        if let (Some(s), Some(dcs)) = (fast, dcs) {
            if let Some((next, next_c, jumped, rebased)) = fast_iter(
                obs,
                s,
                dcs,
                dc,
                c,
                ref_orbit,
                z_ref_f64,
                bla,
                iteration_max,
                bailout_fast,
                bailout_sqr_fexp,
            ) {
                fast = Some(next);
                c = next_c;
                if rebased {
                    rebase_count += 1;
                    st.fast_rebase += 1;
                }
                if jumped {
                    bla_steps += 1;
                    st.fast_jump += 1;
                } else {
                    iters_ptb += 1;
                    st.fast_direct += 1;
                }
                continue;
            }
            st.fallback += 1;
            if stats_enabled() {
                st.reasons[fallback_reason(s, c, z_ref_f64, bailout_fast)] += 1;
            }
            delta = s.to_cexp();
            fast = None;
        }
        st.slow_iter += 1;
        match slow_iter(
            obs,
            ref_orbit,
            bla,
            dc,
            &mut delta,
            &mut c,
            &mut rebase_count,
            &mut bla_steps,
            &mut iters_ptb,
            iteration_max,
            bailout_sqr_fexp,
            use_bla,
        ) {
            Slow::Continue => {
                if dcs.is_some() {
                    fast = Scaled::from_cexp(delta);
                }
            }
            Slow::Escaped(res) => return (res, st),
        }
    }

    if let Some(s) = fast {
        delta = s.to_cexp();
    }
    let final_m = c.m().min((ref_len - 1) as u32);
    let res = UnifiedPixelResultExp {
        iteration: c.n(),
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta.to_complex64_approx(),
        rebase_count,
        bla_steps,
        ref_exhausted: false,
    };
    (res, st)
}

/// Un tour de boucle en rapide : `Some((δ', c', sauté))`, ou `None` si une
/// décision n'est pas triviale / une garde échoue (le tour est alors refait en
/// lent depuis l'état INCHANGÉ).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fast_iter<O: ScaledObs>(
    obs: &mut O,
    s: Scaled,
    dcs: Scaled,
    dc: ComplexExp,
    c: PixelCounter,
    ref_orbit: &ReferenceOrbit,
    z_ref_f64: &[Complex64],
    bla: Option<&BlaTableUnifiedExp>,
    iteration_max: u32,
    bailout_fast: f64,
    bailout_sqr_fexp: FloatExp,
) -> Option<(Scaled, PixelCounter, bool, bool)> {
    let ref_len = z_ref_f64.len();
    if c.m_usize() + 1 >= ref_len {
        return None; // bout de référence : wrap/rebase-at-end en lent
    }
    // Hooks appliqués seulement si le tour aboutit (un repli les rappelle) :
    // le rebase est différé jusqu'au commit (pas de copie de l'observateur).
    let mut pending_rebase: Option<(ComplexExp, ComplexExp)> = None;
    let mut s = s;
    let mut c = c;
    let mut rebased = false;
    let mut z = z_ref_f64[c.m_usize()];
    let mut nz = z.re * z.re + z.im * z.im;
    let mut nd = s.d.re * s.d.re + s.d.im * s.d.im;
    // 1. Rebase. Trivialement faux si |δ| ≤ |Z|/4 ; sinon test EXACT en f64
    //    (δ ramené en f64 absolu ; gardes « partenaire ≥ 2⁻⁹⁶⁰ » sur chaque
    //    addition de `Z+δ` et des deux normes ⇒ mêmes arrondis que FloatExp,
    //    comparaison de deux valeurs normales exactes).
    if !(nz >= NZ_MIN && nd * p2(2 * s.e + 4) < nz) {
        if s.e < -900 || s.e > E_MAX {
            return None;
        }
        let se = p2(s.e);
        let (dr, di) = (s.d.re * se, s.d.im * se);
        let (zr, zi) = (z.re + dr, z.im + di);
        let (zr2, zi2, dr2, di2) = (zr * zr, zi * zi, dr * dr, di * di);
        let ok = big2(z.re, dr) & big2(z.im, di) & big2(zr2, zi2) & big2(dr2, di2);
        if !ok {
            return None;
        }
        if zr2 + zi2 < dr2 + di2 {
            // δ := Z + δ (valeurs f64 exactes), m := 0.
            if O::TRACKS {
                let z_new = ComplexExp {
                    re: FloatExp::from_f64(zr),
                    im: FloatExp::from_f64(zi),
                };
                pending_rebase = Some((ref_orbit.z_ref[c.m_usize()], z_new));
            }
            s = renorm(Complex64::new(zr, zi), 0)?;
            c.rebase();
            rebased = true;
            z = z_ref_f64[0];
            nz = z.re * z.re + z.im * z.im;
            nd = s.d.re * s.d.re + s.d.im * s.d.im;
        }
    }
    // 2. Bailout trivialement non atteint si 2(|Z|²+|δ|²) < ER² (marge).
    let n_delta = nd * p2(2 * s.e);
    if !(2.0 * (nz + n_delta) < bailout_fast) {
        return None;
    }
    let m = c.m_usize();
    if let Some(bla) = bla {
        // Norme FloatExp identique à `delta.norm_sqr_fexp()` (re²+im², même
        // ordre ; la composante max ≥ 2⁻⁶⁴ garantit un partenaire normal).
        let nfx = FloatExp::new(nd, 2 * s.e);
        if let Some(node) = bla.lookup_fexp(m, nfx) {
            let cand_c = c.after_jump(node.l);
            if cand_c.n() <= iteration_max && cand_c.m_usize() < ref_len {
                let (cand, jt) = jump_fast(s, dcs, &node.a, &node.b)?;
                let overshoots = node.l >= 2 && {
                    let ze = z_ref_f64[cand_c.m_usize()];
                    let nze = ze.re * ze.re + ze.im * ze.im;
                    let ndc = (cand.d.re * cand.d.re + cand.d.im * cand.d.im) * p2(2 * cand.e);
                    if 2.0 * (nze + ndc) < bailout_fast {
                        false
                    } else {
                        let z_end = ref_orbit.z_ref[cand_c.m_usize()].add(cand.to_cexp());
                        !(z_end.norm_sqr_fexp() < bailout_sqr_fexp)
                    }
                };
                if !overshoots {
                    if O::TRACKS {
                        if let Some((zr, zn)) = pending_rebase {
                            obs.rebase(zr, zn);
                        }
                        let sigma = (node.sigma_m, node.sigma_e);
                        if !obs.bla_scaled(sigma, jt.ad, jt.bt, jt.ep, node.l) {
                            obs.bla(&node.a, &node.b, s.to_cexp(), dc, node.l);
                        }
                    }
                    return Some((cand, cand_c, true, rebased));
                }
            }
        }
    }
    let (next, tzd_re, tzd_im) = if z.re == 0.0 && z.im == 0.0 {
        // Z f64 nul : chemin dédié seulement si Z est EXACTEMENT nul (Z₀).
        let zx = ref_orbit.z_ref[m];
        if zx.re.mantissa != 0.0 || zx.im.mantissa != 0.0 {
            return None;
        }
        direct_fast_zero_z(s, dcs)?
    } else {
        direct_fast(s, z, dcs)?
    };
    if O::TRACKS {
        // Entrées du hook EXACTEMENT celles de la boucle FloatExp : δ_old,
        // `2·Z·δ_old` (exact sous les gardes de `direct_fast`, 0 exact si Z₀ = 0)
        // et `Z[m] + δ_old` calculé en FloatExp. Chemin f64 d'abord (même
        // borne au bit près sous gardes), ComplexExp sinon.
        if let Some((zr, zn)) = pending_rebase {
            obs.rebase(zr, zn);
        }
        if !obs.direct_scaled(z, s.d, nd, s.e, Complex64::new(tzd_re, tzd_im)) {
            let delta_old = s.to_cexp();
            let two_z_delta = ComplexExp {
                re: FloatExp::new(tzd_re, s.e),
                im: FloatExp::new(tzd_im, s.e),
            };
            let z_abs = ref_orbit.z_ref[m].add(delta_old);
            obs.direct(z_abs, two_z_delta, delta_old);
        }
    }
    c.step();
    Some((next, c, false, rebased))
}

#[cfg(test)]
#[path = "pixel_loop_scaled_tests.rs"]
mod tests;
