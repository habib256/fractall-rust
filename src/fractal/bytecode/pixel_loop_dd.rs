//! Pixel loop perturbation **double-double** (~106 bits de mantisse) pour le
//! tier haute précision deep-zoom Mandelbrot (`params.use_dd_tier`).
//!
//! Équivalent pur-Rust du path `float128` de Fraktaler-3 (sélectionné par son
//! wisdom file). Mirror de `pixel_loop_exp.rs::iterate_pixel_unified_exp_mandelbrot`
//! mais :
//! - `delta` et `dc` : `ComplexDDExp` (mantisse double-double + exposant `i32`)
//!   au lieu de `ComplexExp` (mantisse f64 53 bits).
//! - la **référence** est lue depuis `ref_orbit.z_ref_dd` (Z à 106 bits) et non
//!   `z_ref_f64` : indispensable car `Z` entre non-arrondi dans `2·Z·δ` — un `Z`
//!   f64 (2⁻⁵²·|Z|) capperait la précision quel que soit le delta.
//! - **pas de BLA** : un pas BLA linéarisé avec des coefficients `A` f64
//!   réintroduirait 2⁻⁵² tôt dans l'orbite (les pixels sensibles BLA-ent quand
//!   δ est minuscule), annulant le gain dd. On itère donc en pas directs purs.
//!   (Une dd-BLA — coefficients `A` double-double — est le chantier perf suivant.)
//!
//! Motivation : les spirales ultra-sensibles profondes (e30/e50) escapent après
//! ~25000 itérations ; l'amplification de Lyapunov transforme le 2⁻⁵² d'arrondi
//! f64 en écart d'itération O(50-200) vs GMP (cf. TODO G2). Le tier dd repousse
//! ce plancher à ~2⁻¹⁰⁵.

use num_complex::Complex64;

use super::bla_dd::BlaTableDd;
use super::pixel_loop_exp::UnifiedPixelResultExp;
use crate::fractal::perturbation::counter::PixelCounter;
use crate::fractal::perturbation::dd::{ComplexDDExp, DoubleDoubleExp};
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::FloatExp;

/// Norme² d'un delta dd en `FloatExp` (mantisse 53 b + exposant) pour comparer
/// au rayon de validité BLA `r²` (magnitude, la précision dd y est superflue).
#[inline]
fn delta_norm_sqr_fexp(delta: ComplexDDExp) -> FloatExp {
    let n = delta.norm_sqr();
    FloatExp::new(n.mantissa.to_f64(), n.exponent)
}

/// Pixel loop Mandelbrot double-double : pas directs `δ' = 2·Z·δ + δ² + dc` en
/// ~106 bits + **BLA dd** (skip d'itérations sans perte de précision) + rebasing
/// F3. Utilise `ref_orbit.z_ref_dd`.
///
/// Structure F3 (`hybrid.cc:286-343`) : à chaque tour, rebase AVANT le lookup
/// BLA, puis bailout, puis pas BLA si valide, sinon pas direct. `bla = None`
/// retombe sur les pas directs purs (comportement historique sans BLA).
///
/// `dc` et `delta_initial` sont en `ComplexDDExp`. Renvoie la même shape que le
/// path exp (`UnifiedPixelResultExp`) pour un câblage transparent au dispatch.
#[allow(clippy::too_many_arguments)]
pub fn iterate_pixel_unified_ddexp_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    bla: Option<&BlaTableDd>,
    dc: ComplexDDExp,
    delta_initial: ComplexDDExp,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
) -> UnifiedPixelResultExp {
    let ref_len = ref_orbit.z_ref_dd.len();
    // Garde : le tier dd exige une orbite dd non vide et de longueur cohérente.
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
    }
    let mut st = DdLoopState {
        delta: delta_initial,
        c: PixelCounter::new(),
        rebase_count: 0,
        bla_steps: 0,
        iters_ptb: 0,
    };
    if fast_dd_enabled() {
        if let Some(r) = fast::run(ref_orbit, bla, dc, &mut st, limits) {
            return r;
        }
    }
    ddexp_loop(ref_orbit, bla, dc, st, limits)
}

/// État de la boucle dd, transférable du chemin rapide (`fast`) vers la
/// boucle `DoubleDoubleExp` au point EXACT où le premier s'arrête.
#[derive(Clone, Copy)]
struct DdLoopState {
    delta: ComplexDDExp,
    c: PixelCounter,
    rebase_count: u32,
    bla_steps: u32,
    iters_ptb: u32,
}

/// Chemin rapide dd sans exposant (`fast`) actif ? Défaut oui ;
/// `FRACTALL_DD_FAST=0` force la boucle `DoubleDoubleExp` seule (A/B, et
/// verrou d'identité).
fn fast_dd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("FRACTALL_DD_FAST").as_deref(), Ok("0") | Ok("off")))
}

/// Boucle dd de référence en `DoubleDoubleExp` (mantisse dd + exposant
/// `i32`, normalisée à chaque opération) — la sémantique que le chemin
/// rapide reproduit au bit près. Reprend depuis `st` (début de tour).
fn ddexp_loop(
    ref_orbit: &ReferenceOrbit,
    bla: Option<&BlaTableDd>,
    dc: ComplexDDExp,
    st: DdLoopState,
    limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
) -> UnifiedPixelResultExp {
    let crate::fractal::bytecode::pixel_loop::PixelLoopLimits {
        iteration_max,
        bailout,
        max_perturb_iterations,
        max_bla_steps,
    } = limits;
    let ref_len = ref_orbit.z_ref_dd.len();
    let z_ref_dd = &ref_orbit.z_ref_dd;
    let bailout_sqr = DoubleDoubleExp::from_f64(bailout * bailout);

    let DdLoopState {
        mut delta,
        mut c,
        mut rebase_count,
        mut bla_steps,
        mut iters_ptb,
    } = st;
    const REDUCE_INTERVAL: u32 = 250;

    while c.keep_iterating(iteration_max)
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        // Rebase F3 AVANT bailout & BLA (`hybrid.cc:295-308`) : `|Z[m]+δ|² <
        // |δ|²` OU bout de référence ⇒ z := Z[m]+δ, m := 0. Indispensable ici :
        // un pas BLA saute plusieurs itérations et pourrait passer par-dessus le
        // point de rebase. Périodique : `wrap_periodic`.
        {
            let z_m0 = z_ref_dd[c.m_usize()];
            let zz = z_m0.add(delta);
            let end_of_ref = c.m_usize() + 1 >= ref_len;
            if end_of_ref {
                if let Some(m_wrapped) = ref_orbit.wrap_periodic(c.m()) {
                    c.set_ref_index(m_wrapped);
                } else {
                    delta = zz;
                    c.rebase();
                    rebase_count += 1;
                }
            } else if zz.norm_sqr() < delta.norm_sqr() {
                delta = zz;
                c.rebase();
                rebase_count += 1;
            }
        }

        let z_m = z_ref_dd[c.m_usize()];
        // Bailout : |Z[m] + δ|² ≥ bailout² (tout en dd, pas de saturation f64).
        let z_abs = z_m.add(delta);
        if !(z_abs.norm_sqr() < bailout_sqr) {
            return UnifiedPixelResultExp {
                iteration: c.n(),
                z_final: z_abs.to_complex64_approx(),
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        // Pas BLA dd si disponible et valide : δ := A·δ + B·dc, saute `l` iters.
        if let Some(table) = bla {
            if let Some(node) = table.lookup_fexp(c.m_usize(), delta_norm_sqr_fexp(delta)) {
                let cand_c = c.after_jump(node.l);
                if cand_c.n() <= iteration_max && cand_c.m_usize() < ref_len {
                    let a = node.a;
                    let b = node.b;
                    // A·δ (Mat2Dd × vecteur dd) + B·dc.
                    let new_re = delta
                        .re
                        .mul_dd(a.m00)
                        .add(delta.im.mul_dd(a.m01))
                        .add(dc.re.mul_dd(b.m00))
                        .add(dc.im.mul_dd(b.m01));
                    let new_im = delta
                        .re
                        .mul_dd(a.m10)
                        .add(delta.im.mul_dd(a.m11))
                        .add(dc.re.mul_dd(b.m10))
                        .add(dc.im.mul_dd(b.m11));
                    delta = ComplexDDExp {
                        re: new_re,
                        im: new_im,
                    };
                    c = cand_c;
                    bla_steps += 1;
                    if c.n() % REDUCE_INTERVAL == 0 {
                        delta.reduce();
                    }
                    continue;
                }
            }
        }

        // Pas perturbation Mandelbrot en dd : δ' = 2·Z·δ + δ² + dc.
        let two_z = ComplexDDExp {
            re: z_m.re.mul_f64(2.0),
            im: z_m.im.mul_f64(2.0),
        };
        delta = delta.mul(two_z).add(delta.sqr()).add(dc);
        c.step();
        iters_ptb += 1;

        if !delta.is_finite() {
            let m_read = c.m_usize().min(ref_len - 1);
            return UnifiedPixelResultExp {
                iteration: c.n(),
                z_final: z_ref_dd[m_read].add(delta).to_complex64_approx(),
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        if c.n() % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }
    }

    let final_m = c.m_usize().min(ref_len - 1);
    UnifiedPixelResultExp {
        iteration: c.n(),
        z_final: z_ref_dd[final_m].add(delta).to_complex64_approx(),
        rebase_count,
        bla_steps,
        ref_exhausted: false,
    }
}

/// Chemin rapide de la boucle dd : **double-double SANS exposant** (valeurs
/// réelles dans la plage f64), qui reproduit au bit près la boucle
/// `DoubleDoubleExp` tant que toutes les grandeurs restent loin du sous-normal.
///
/// Pourquoi c'est la même arithmétique : `DoubleDoubleExp` = mantisse dd
/// normalisée (`hi ∈ [0.5, 1)`) × `2^e`. Ses opérations calculent sur les
/// mantisses puis renormalisent par une puissance de 2 ; or les opérations dd
/// (`two_sum`, `two_prod` par FMA, produits croisés) commutent avec un
/// changement d'échelle par 2^k tant qu'aucun résultat de MULTIPLICATION
/// n'est sous-normal (les additions restent exactes même en sous-normal).
/// Une seule règle n'est pas une pure mise à l'échelle :
/// `DoubleDoubleExp::add` ABANDONNE l'opérande le plus petit quand les
/// exposants diffèrent de ≥ 108 — reproduite telle quelle (`add_e`, exposants
/// lus dans les bits, `frexp` exact en sous-normal).
///
/// Gardes (conservatrices) qui tiennent tout produit loin du sous-normal ;
/// échec ⇒ on rend la main à `ddexp_loop`, état converti exactement :
/// - composantes de δ et de dc : 0 ou |·| ≥ 2⁻³⁵⁰ (produits ≥ 2⁻⁷⁰⁰, termes
///   d'erreur ≥ 2⁻⁸⁰⁶) ;
/// - composantes de Z : 0 ou |·| ≥ 2⁻³⁰⁰ ;
/// - pas BLA : chaque produit coefficient × opérande entre 2⁻⁷⁰⁰ et 2⁹⁰⁰,
///   coefficient ≥ 2⁻⁶⁰⁰ (la boucle `DoubleDoubleExp` multiplie la MANTISSE
///   de δ par le coefficient : un coefficient minuscule y sous-déborde
///   autrement) ;
/// - norme de comparaison du rebase : si `Z+δ` a une composante minuscule,
///   la comparaison est faite en `DoubleDoubleExp` (conversion exacte).
/// Points de reprise : début de tour, ou juste après un rebase (alors
/// `m = 0`, `Z[0] = 0` : la boucle `DoubleDoubleExp` refait le test de rebase
/// sans effet et le test d'évasion à l'identique). D'où les conditions
/// d'entrée : `Z[0] = 0` et pas de `cycle_period` (Brent, opt-in).
///
/// Reste hors preuve : les bits d'une partie basse `lo` plus de ~2⁻⁷⁰⁰ sous
/// sa partie haute, tronqués par le sous-normal ici et pas en
/// `DoubleDoubleExp` — 2⁻⁶⁰⁰ sous la précision dd, ils ne peuvent changer un
/// arrondi qu'à égalité exacte. Vérifié bit à bit (verrou
/// `fast_dd_loop_matches_ddexp_loop_bitwise`, cadres dd complets en `.fmap`).
mod fast {
    use super::{DdLoopState, UnifiedPixelResultExp};
    use crate::fractal::bytecode::bla_dd::BlaTableDd;
    use crate::fractal::perturbation::dd::{ComplexDDExp, DoubleDouble as Dd, DoubleDoubleExp};
    use crate::fractal::perturbation::orbit::ReferenceOrbit;
    use crate::fractal::perturbation::types::FloatExp;
    use num_complex::Complex64;

    /// 2⁻³⁵⁰.
    const MIN_DELTA: f64 = 4.258_000_478_242_964e-106;
    /// 2⁻³⁰⁰.
    const MIN_Z: f64 = 4.909_093_465_297_727e-91;
    /// 2⁻⁶⁰⁰.
    const MIN_COEF: f64 = 2.409_919_865_102_884e-181;
    /// Demi-largeur relative de la bande où une norme f64 ne suffit pas.
    const BAND: f64 = 2.842_170_943_040_401e-14; // 2^-45
    /// 2⁻¹⁰²¹.
    const MIN_TO_F64: f64 = 4.450_147_717_014_403e-308;

    /// Exposant `frexp` (valeur ∈ [2^(e-1), 2^e)) — celui de la
    /// normalisation de `DoubleDoubleExp`.
    #[inline(always)]
    fn fexp(x: f64) -> i32 {
        let e = ((x.to_bits() >> 52) & 0x7ff) as i32;
        if e != 0 {
            e - 1022
        } else {
            crate::fractal::perturbation::types::frexp(x).1
        }
    }

    /// `DoubleDoubleExp::add` : abandon de l'opérande à ≥ 108 binades.
    #[inline(always)]
    fn add_e(a: Dd, b: Dd) -> Dd {
        if a.hi == 0.0 {
            return b;
        }
        if b.hi == 0.0 {
            return a;
        }
        let d = fexp(a.hi) - fexp(b.hi);
        if d >= 108 {
            return a;
        }
        if d <= -108 {
            return b;
        }
        a.add(b)
    }

    #[inline(always)]
    fn sub_e(a: Dd, b: Dd) -> Dd {
        add_e(a, b.neg())
    }

    #[inline(always)]
    fn ok(x: Dd, min: f64) -> bool {
        x.hi == 0.0 || x.hi.abs() >= min
    }

    /// `DoubleDoubleExp` → dd plage f64, si la garde `min` tient.
    #[inline(always)]
    fn plain(x: DoubleDoubleExp, min: f64) -> Option<Dd> {
        if x.mantissa.hi == 0.0 {
            return Some(Dd::ZERO);
        }
        let d = x.to_dd();
        (d.hi.abs() >= min && d.hi.is_finite()).then_some(d)
    }

    /// dd plage f64 → `DoubleDoubleExp`, EXACT y compris en sous-normal
    /// (`normalized` ne sait pas remettre à l'échelle une mantisse
    /// sous-normale : pré-multiplication exacte par 2⁶⁰⁰).
    #[inline]
    fn to_ddexp(d: Dd) -> DoubleDoubleExp {
        if d.hi == 0.0 {
            return DoubleDoubleExp::ZERO;
        }
        if d.hi.abs() < 1e-250 {
            let s = crate::fractal::perturbation::types::pow2i(600);
            DoubleDoubleExp::normalized(
                Dd {
                    hi: d.hi * s,
                    lo: d.lo * s,
                },
                -600,
            )
        } else {
            DoubleDoubleExp::normalized(d, 0)
        }
    }

    /// `DoubleDoubleExp::to_f64` (qui rend 0 sous 2⁻¹⁰²³ et arrondit la
    /// mantisse AVANT la mise à l'échelle).
    #[inline]
    fn to_f64_e(d: Dd) -> f64 {
        if d.hi == 0.0 {
            0.0
        } else if d.hi.abs() >= MIN_TO_F64 {
            d.hi + d.lo
        } else {
            to_ddexp(d).to_f64()
        }
    }

    #[inline(always)]
    fn norm(re: Dd, im: Dd) -> Dd {
        add_e(re.sqr(), im.sqr())
    }

    /// `<` de `DoubleDoubleExp` entre deux normes (≥ 0).
    #[inline(always)]
    fn less(a: Dd, b: Dd) -> bool {
        a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo)
    }

    /// Produit coefficient BLA × opérande loin du sous-normal et de
    /// l'overflow.
    #[inline(always)]
    fn prod_ok(x: Dd, c: Dd) -> bool {
        if !(c.hi.is_finite() && c.lo.is_finite()) {
            return false;
        }
        if x.hi == 0.0 || c.hi == 0.0 {
            return true;
        }
        c.hi.abs() >= MIN_COEF && (-700..=900).contains(&(fexp(x.hi) + fexp(c.hi)))
    }

    /// Exécute le pixel en dd plage f64. `Some(résultat)` si terminé ; `None`
    /// si une garde échoue — `st` contient alors l'état exact à un point de
    /// reprise de `ddexp_loop` (cf. doc du module).
    pub(super) fn run(
        ref_orbit: &ReferenceOrbit,
        bla: Option<&BlaTableDd>,
        dc: ComplexDDExp,
        st: &mut DdLoopState,
        limits: crate::fractal::bytecode::pixel_loop::PixelLoopLimits,
    ) -> Option<UnifiedPixelResultExp> {
        let crate::fractal::bytecode::pixel_loop::PixelLoopLimits {
            iteration_max,
            bailout,
            max_perturb_iterations,
            max_bla_steps,
        } = limits;
        let z_ref_dd = &ref_orbit.z_ref_dd;
        let ref_len = z_ref_dd.len();
        if ref_orbit.cycle_period != 0
            || z_ref_dd[0].re.mantissa.hi != 0.0
            || z_ref_dd[0].im.mantissa.hi != 0.0
        {
            return None;
        }
        let dcr = plain(dc.re, MIN_DELTA)?;
        let dci = plain(dc.im, MIN_DELTA)?;
        let mut dr = plain(st.delta.re, MIN_DELTA)?;
        let mut di = plain(st.delta.im, MIN_DELTA)?;
        let bailout_sqr = DoubleDoubleExp::from_f64(bailout * bailout).to_dd();
        let bailout_f = bailout_sqr.hi;
        let mut c = st.c;
        let mut rebase_count = st.rebase_count;
        let mut bla_steps = st.bla_steps;
        let mut iters_ptb = st.iters_ptb;

        macro_rules! bail {
            () => {{
                st.delta = ComplexDDExp {
                    re: to_ddexp(dr),
                    im: to_ddexp(di),
                };
                st.c = c;
                st.rebase_count = rebase_count;
                st.bla_steps = bla_steps;
                st.iters_ptb = iters_ptb;
                return None;
            }};
        }

        while c.keep_iterating(iteration_max)
            && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
            && (max_bla_steps == 0 || bla_steps < max_bla_steps)
        {
            if !ok(dr, MIN_DELTA) || !ok(di, MIN_DELTA) {
                bail!();
            }
            let z0 = z_ref_dd[c.m_usize()];
            let (Some(zr0), Some(zi0)) = (plain(z0.re, MIN_Z), plain(z0.im, MIN_Z)) else {
                bail!();
            };

            // Rebase F3 (cf. `ddexp_loop`). `cycle_period == 0` : pas de wrap.
            // `zz = Z[m] + δ` est AUSSI le `Z + δ` du test d'évasion : sans
            // rebase m est inchangé ; avec, δ := zz et Z[0] = 0.
            let zzr = add_e(zr0, dr);
            let zzi = add_e(zi0, di);
            // Normes en f64 sur les parties hautes (erreur relative < 2⁻⁵⁰,
            // |lo| ≤ 2⁻⁵³|hi|) : décident seules hors d'une bande ±2⁻⁴⁵ — la
            // comparaison dd exacte n'est faite que dans la bande, donc même
            // décision que `DoubleDoubleExp`. Un sous-dépassement des carrés
            // ne touche que des normes < 2⁻¹⁰²¹, déjà « plus petites » que
            // tout ce à quoi elles sont comparées (|δ|² ≥ 2⁻⁷⁰⁰, 625).
            let nz = zzr.hi * zzr.hi + zzi.hi * zzi.hi;
            let rebase = if c.m_usize() + 1 >= ref_len {
                true
            } else {
                let nd = dr.hi * dr.hi + di.hi * di.hi;
                if nz < nd * (1.0 - BAND) {
                    true
                } else if nz > nd * (1.0 + BAND) {
                    false
                } else if ok(zzr, MIN_DELTA) && ok(zzi, MIN_DELTA) {
                    less(norm(zzr, zzi), norm(dr, di))
                } else {
                    // Composante minuscule : comparaison en DoubleDoubleExp.
                    let zz = ComplexDDExp {
                        re: to_ddexp(zzr),
                        im: to_ddexp(zzi),
                    };
                    let d = ComplexDDExp {
                        re: to_ddexp(dr),
                        im: to_ddexp(di),
                    };
                    zz.norm_sqr() < d.norm_sqr()
                }
            };
            if rebase {
                dr = zzr;
                di = zzi;
                c.rebase();
                rebase_count += 1;
            }

            // Évasion `|Z+δ|² < bailout²` (même bande). Une composante
            // minuscule de Z+δ ne peut pas changer l'issue (abandonnée à 108
            // binades d'une composante ~bailout, ou les deux minuscules).
            let escaped = if nz < bailout_f * (1.0 - BAND) {
                false
            } else if nz > bailout_f * (1.0 + BAND) {
                true
            } else {
                !less(norm(zzr, zzi), bailout_sqr)
            };
            if escaped {
                return Some(UnifiedPixelResultExp {
                    iteration: c.n(),
                    z_final: Complex64::new(to_f64_e(zzr), to_f64_e(zzi)),
                    rebase_count,
                    bla_steps,
                    ref_exhausted: false,
                });
            }
            let (zr, zi) = if rebase { (Dd::ZERO, Dd::ZERO) } else { (zr0, zi0) };

            // δ issu d'un rebase : composantes éventuellement minuscules.
            if !ok(dr, MIN_DELTA) || !ok(di, MIN_DELTA) {
                bail!();
            }

            // Pas BLA dd. `lookup_fexp` ne consulte que les niveaux
            // `BLA_SKIP_LEVELS..=trailing_zeros(m)` (tous si m = 0) : ailleurs
            // il rend `None` quoi qu'il arrive — on s'épargne alors la norme.
            let m_now = c.m_usize();
            let bla = bla.filter(|_| {
                m_now == 0
                    || m_now.trailing_zeros() as usize
                        >= crate::fractal::bytecode::bla_dual::BLA_SKIP_LEVELS
            });
            if let Some(table) = bla {
                let n = norm(dr, di);
                if let Some(node) = table.lookup_fexp(c.m_usize(), FloatExp::new(n.hi + n.lo, 0)) {
                    let cand_c = c.after_jump(node.l);
                    if cand_c.n() <= iteration_max && cand_c.m_usize() < ref_len {
                        let (a, b) = (node.a, node.b);
                        if !(prod_ok(dr, a.m00)
                            && prod_ok(di, a.m01)
                            && prod_ok(dcr, b.m00)
                            && prod_ok(dci, b.m01)
                            && prod_ok(dr, a.m10)
                            && prod_ok(di, a.m11)
                            && prod_ok(dcr, b.m10)
                            && prod_ok(dci, b.m11))
                        {
                            bail!();
                        }
                        let nr = add_e(
                            add_e(add_e(dr.mul(a.m00), di.mul(a.m01)), dcr.mul(b.m00)),
                            dci.mul(b.m01),
                        );
                        let ni = add_e(
                            add_e(add_e(dr.mul(a.m10), di.mul(a.m11)), dcr.mul(b.m10)),
                            dci.mul(b.m11),
                        );
                        dr = nr;
                        di = ni;
                        c = cand_c;
                        bla_steps += 1;
                        continue;
                    }
                }
            }

            // Pas direct δ' = 2·Z·δ + δ² + dc, même ordre d'opérations que
            // `ComplexDDExp::{mul, sqr, add}`.
            let tzr = zr.mul_f64(2.0);
            let tzi = zi.mul_f64(2.0);
            let pr = sub_e(dr.mul(tzr), di.mul(tzi));
            let pi = add_e(dr.mul(tzi), di.mul(tzr));
            let sr = sub_e(dr.sqr(), di.sqr());
            let si = dr.mul(di).mul_f64(2.0);
            dr = add_e(add_e(pr, sr), dcr);
            di = add_e(add_e(pi, si), dci);
            c.step();
            iters_ptb += 1;
            // |δ| est borné par l'évasion du tour (|Z| < 1e5, |Z+δ| < bailout) :
            // pas de non-fini possible ici, contrairement à `ddexp_loop` dont
            // ce chemin n'a donc pas besoin de la sortie `!is_finite`.
            debug_assert!(dr.hi.is_finite() && di.hi.is_finite());
        }

        // Sortie par les plafonds (cf. fin de `ddexp_loop`).
        let zm = z_ref_dd[c.m_usize().min(ref_len - 1)];
        let z_final = match (plain(zm.re, MIN_Z), plain(zm.im, MIN_Z)) {
            (Some(zr), Some(zi)) => Complex64::new(to_f64_e(add_e(zr, dr)), to_f64_e(add_e(zi, di))),
            _ => zm
                .add(ComplexDDExp {
                    re: to_ddexp(dr),
                    im: to_ddexp(di),
                })
                .to_complex64_approx(),
        };
        Some(UnifiedPixelResultExp {
            iteration: c.n(),
            z_final,
            rebase_count,
            bla_steps,
            ref_exhausted: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::perturbation::dd::DoubleDouble;

    /// Le pixel loop dd doit rester fini et cohérent sur une orbite courte
    /// synthétique (référence = quelques Z bornés). Sanity de vivacité.
    #[test]
    fn ddexp_loop_terminates_finite() {
        // Orbite référence dd triviale : Z borné O(1).
        let z_ref_dd: Vec<ComplexDDExp> = (0..64)
            .map(|k| {
                let v = 0.1 * (k as f64 % 5.0);
                ComplexDDExp {
                    re: DoubleDoubleExp::from_dd(DoubleDouble::from_f64(v)),
                    im: DoubleDoubleExp::from_dd(DoubleDouble::from_f64(-v)),
                }
            })
            .collect();
        let orbit = ReferenceOrbit {
            cref: Complex64::new(0.0, 0.0),
            z_ref: Vec::new(),
            z_ref_f64: vec![Complex64::new(0.0, 0.0); z_ref_dd.len()],
            z_ref_lo: Vec::new(),
            z_ref_gmp: Vec::new(),
            cref_gmp: rug::Complex::with_val(53, (0, 0)),
            phase_offset: 0,
            extended_iterations: Vec::new(),
            high_precision_data: Vec::new(),
            data_storage_interval: 1,
            cycle_period: 0,
            cycle_start: 0,
            z_ref_dd,
            atom_truncated: false,
            compressed_f64: None,
            hybrid_phase_refs: Vec::new(),
        };
        let dc = ComplexDDExp::from_complex64(Complex64::new(1e-30, 1e-30));
        let res = iterate_pixel_unified_ddexp_mandelbrot(
            &orbit,
            None,
            dc,
            ComplexDDExp::ZERO,
            crate::fractal::bytecode::pixel_loop::PixelLoopLimits {
                iteration_max: 500,
                bailout: 25.0,
                max_perturb_iterations: 0,
                max_bla_steps: 0,
            },
        );
        assert!(res.z_final.re.is_finite() && res.z_final.im.is_finite());
        assert!(res.iteration <= 500);
    }

    /// Verrou du chemin rapide (`fast`) : sur une vraie orbite dd profonde
    /// (rebases, sauts BLA dd, pixels qui s'échappent et pixels intérieurs),
    /// le résultat est BIT-IDENTIQUE à la boucle `DoubleDoubleExp` seule —
    /// itération, `z_final`, compteurs. Garde anti-vacuité : le chemin rapide
    /// doit terminer lui-même la grande majorité des pixels, avec des sauts
    /// BLA et des rebases.
    #[test]
    fn fast_dd_loop_matches_ddexp_loop_bitwise() {
        use crate::fractal::bytecode::bla_dd::{BlaTableDd, DD_BLA_EPSILON};
        use crate::fractal::perturbation::effective_spans_dd;
        use crate::fractal::{default_params_for_type, FractalType};
        let (mut tot_bla, mut tot_rebases, mut tot_escaped) = (0u64, 0u64, 0u32);

        for (cx, cy, zoom, iters) in [
            // Seahorse, profondeur f64-perturbation classique.
            (
                "-0.743643887037158704752191506114774",
                "0.131825904205311970493132056385139",
                "1e25",
                6000u32,
            ),
            // Spirale e30 (preset quality `mandelbrot-e30`, où l'escalade dd
            // agit ; évasions vers ~25-37 k itérations).
            (
                "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000",
                "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880",
                "1e30",
                40000u32,
            ),
        ] {
            let (w, h) = (24u32, 16u32);
            let mut params = default_params_for_type(FractalType::Mandelbrot, w, h);
            params.center_x = cx.parse().unwrap();
            params.center_y = cy.parse().unwrap();
            params.center_x_hp = Some(cx.into());
            params.center_y_hp = Some(cy.into());
            let span = 4.0 / zoom.parse::<f64>().unwrap();
            params.span_x = span * w as f64 / h as f64;
            params.span_y = span;
            params.span_x_hp = None;
            params.span_y_hp = None;
            params.iteration_max = iters;
            params.engine.use_dd_tier = true;
            params.engine.precision_bits =
                crate::fractal::perturbation::compute_perturbation_precision_bits(&params);
            let (orbit, _, _) =
                crate::fractal::perturbation::orbit::compute_reference_orbit(&params, None, false)
                    .expect("orbite");
            assert!(orbit.has_dd());
            let pixel = span / h as f64;
            let c_norm = pixel * ((w * w + h * h) as f64).sqrt();
            let table = BlaTableDd::build_mandelbrot(&orbit.z_ref_dd, c_norm, DD_BLA_EPSILON);
            let limits = crate::fractal::bytecode::pixel_loop::PixelLoopLimits::uncapped(
                params.iteration_max,
                25.0,
            );
            let (xr, yr) = effective_spans_dd(&params);
            let (mut fast_done, mut bla_steps, mut rebases, mut escaped) = (0, 0u64, 0u64, 0);
            for j in 0..h {
                for i in 0..w {
                    let fx = DoubleDoubleExp::from_f64((i as f64 + 0.5) / w as f64 - 0.5);
                    let fy = DoubleDoubleExp::from_f64((j as f64 + 0.5) / h as f64 - 0.5);
                    let dc = ComplexDDExp {
                        re: xr.mul(fx),
                        im: yr.mul(fy),
                    };
                    let st0 = DdLoopState {
                        delta: ComplexDDExp::ZERO,
                        c: PixelCounter::new(),
                        rebase_count: 0,
                        bla_steps: 0,
                        iters_ptb: 0,
                    };
                    let reference = ddexp_loop(&orbit, Some(&table), dc, st0, limits);
                    let mut st = st0;
                    let got = match fast::run(&orbit, Some(&table), dc, &mut st, limits) {
                        Some(r) => {
                            fast_done += 1;
                            r
                        }
                        None => ddexp_loop(&orbit, Some(&table), dc, st, limits),
                    };
                    assert_eq!(got.iteration, reference.iteration, "{zoom} px ({i},{j})");
                    assert_eq!(
                        (got.z_final.re.to_bits(), got.z_final.im.to_bits()),
                        (reference.z_final.re.to_bits(), reference.z_final.im.to_bits()),
                        "{zoom} px ({i},{j})"
                    );
                    assert_eq!(got.rebase_count, reference.rebase_count, "{zoom} ({i},{j})");
                    assert_eq!(got.bla_steps, reference.bla_steps, "{zoom} ({i},{j})");
                    bla_steps += reference.bla_steps as u64;
                    rebases += reference.rebase_count as u64;
                    escaped += (reference.iteration < iters) as u32;
                }
            }
            let n = w * h;
            assert!(fast_done * 10 >= n * 9, "{zoom}: chemin rapide {fast_done}/{n}");
            eprintln!("{zoom}: rapide {fast_done}/{n} bla={bla_steps} rebases={rebases} évadés={escaped}");
            tot_bla += bla_steps;
            tot_rebases += rebases;
            tot_escaped += escaped;
        }
        assert!(tot_bla > 0 && tot_rebases > 0, "bla={tot_bla} rebases={tot_rebases}");
        assert!(tot_escaped > 0, "aucun pixel évadé");
    }
}
