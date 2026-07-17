//! Pixel loop unifié extended-precision (ComplexExp) pour deep zoom > 1e15.
//!
//! Mirror de `pixel_loop.rs::iterate_pixel_unified` mais avec `delta` en
//! `ComplexExp` (mantissa + exponent) au lieu de `Complex64`. La reference
//! orbit reste en `Complex64` (les valeurs sont O(1), pas d'underflow).
//!
//! Choix d'implémentation pour la précision :
//! - `delta` et `dc` : `ComplexExp` (préservent les magnitudes < 1e-308)
//! - `z_ref` : `Complex64` (valeurs bornées, suffit)
//! - `BLA mat2 A, B` : `Mat2<f64>` (coefficients O(1) à profondeur raisonnable)
//! - Bailout check : convertit `z_ref + delta_approx` en `f64` (Z domine)
//! - Rebase check : compare via `f64` (les magnitudes après cancellation
//!   restent dans la range f64 normale dès qu'on est en zone d'évasion)
//!
//! Limitation : la BLA est construite en f64 ; à très deep zoom (> ~1e150),
//! les coefficients A peuvent devenir mal conditionnés. Pour ces zooms,
//! l'orbite référence f64 underflow aussi → fallback sur le path GMP legacy
//! reste actif (gated par `pixel_size < 1e-150` dans le dispatcher).

use num_complex::Complex64;

use super::bla_dual::BlaTableUnified;
use super::bla_dual_exp::BlaTableUnifiedExp;
use super::delta_form::DeltaStateExp;
use super::{Formula, Op, Phase};
use crate::fractal::perturbation::orbit::ReferenceOrbit;
use crate::fractal::perturbation::types::{ComplexExp, FloatExp};

/// Path atom-tronqué (troncature réf + variant HP ComplexExp) : **ON par défaut**
/// (`FRACTALL_ATOM_PERIOD=0` force OFF). Délègue au flag canonique de `delta`
/// pour rester cohérent avec les 2 autres portes (delta::atom_hp_enabled +
/// orbit::atom_period_enabled). Cf. `delta::atom_hp_enabled` pour le pourquoi.
fn atom_hp_enabled() -> bool {
    crate::fractal::perturbation::delta::atom_hp_enabled()
}

/// DEBUG uniquement : `FRACTALL_ATOM_NOBLA=1` désactive le skip BLA dans le path
/// HP (pas directs purs) pour isoler la correction BLA de la boucle directe.
fn atom_nobla_debug() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var("FRACTALL_ATOM_NOBLA").ok().as_deref() == Some("1"))
}

/// Phase Mandelbrot = exactement [Sqr, Add]. Détection au runtime pour
/// activer le path spécialisé qui évite le dispatch opcode et l'update
/// wasted de `z_ref` dans DeltaStateExp.
#[inline]
fn phase_is_mandelbrot(phase: &Phase) -> bool {
    phase.ops.len() == 2
        && matches!(phase.ops[0], Op::Sqr)
        && matches!(phase.ops[1], Op::Add)
}

/// Résultat du pixel loop extended-precision.
pub struct UnifiedPixelResultExp {
    pub iteration: u32,
    pub z_final: Complex64,
    #[allow(dead_code)]
    pub rebase_count: u32,
    #[allow(dead_code)]
    pub bla_steps: u32,
    /// `true` si la boucle a quitté parce que l'orbite référence s'est
    /// épuisée avant `iteration_max` (centre escape-time, non-périodique).
    /// L'iteration count retourné est non fiable : le caller doit
    /// re-rendre via `iterate_pixel_gmp` pour obtenir le vrai iter
    /// d'évasion par pixel (cf. fix e113.toml uniforme).
    pub ref_exhausted: bool,
}

/// Pixel loop unifié extended-precision. Signature mirror de
/// `iterate_pixel_unified` (pixel_loop.rs) mais accepte `dc` et
/// `delta_initial` en `ComplexExp`.
///
/// `bla_exp` : table BLA FloatExp, `Some` uniquement quand
/// `FRACTALL_ATOM_PERIOD=1` (path atom-tronqué HP). `None` sinon — le path
/// f64 par défaut n'y touche jamais.
#[allow(clippy::too_many_arguments)]
pub fn iterate_pixel_unified_exp(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    bla_exp: Option<&BlaTableUnifiedExp>,
    formula: &Formula,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResultExp {
    // G4 jalon 4 : hybrides multi-phase deep (> 1e13, ComplexExp). Pas de BLA
    // (table construite pour UNE phase précise, pas applicable au mélange —
    // mirror du path f64 `iterate_pixel_unified_multi_phase`), rebasing F3 en
    // FloatExp. Pour une formule mono-phase [M] ce chemin n'est PAS pris (fast
    // path Mandelbrot ci-dessous), donc l'invariant [M,M]==[M] compare bien le
    // multi-phase au fast path.
    if formula.phases.len() > 1 {
        return iterate_pixel_unified_exp_multi_phase(
            ref_orbit,
            &[],
            formula,
            c_ref,
            dc,
            delta_initial,
            iteration_max,
            bailout,
            max_perturb_iterations,
            max_bla_steps,
        );
    }
    let phase = &formula.phases[0];
    iterate_pixel_unified_exp_single_phase(
        ref_orbit,
        bla,
        bla_exp,
        phase,
        c_ref,
        dc,
        delta_initial,
        iteration_max,
        bailout,
        max_perturb_iterations,
        max_bla_steps,
    )
}

#[allow(clippy::too_many_arguments)]
fn iterate_pixel_unified_exp_single_phase(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    bla_exp: Option<&BlaTableUnifiedExp>,
    phase: &Phase,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResultExp {
    // Hot path Mandelbrot : évite DeltaStateExp::step (dispatch opcode + update
    // wasted de z_ref) en inlinant `δ' = 2·Z·δ + δ² + dc`.
    if phase_is_mandelbrot(phase) {
        // Path atom-tronqué (`FRACTALL_ATOM_PERIOD`) : variant HP ComplexExp
        // ref + BLA FloatExp (réf f64/BLA f64 zéroent les grazes ~1e-8000).
        if atom_hp_enabled() {
            return iterate_pixel_unified_exp_mandelbrot_hp(
                ref_orbit,
                bla_exp,
                dc,
                delta_initial,
                iteration_max,
                bailout,
                max_perturb_iterations,
                max_bla_steps,
            );
        }
        return iterate_pixel_unified_exp_mandelbrot(
            ref_orbit,
            bla,
            dc,
            delta_initial,
            iteration_max,
            bailout,
            max_perturb_iterations,
            max_bla_steps,
        );
    }
    let _ = c_ref;
    iterate_pixel_unified_exp_generic(
        ref_orbit,
        bla,
        phase,
        c_ref,
        dc,
        delta_initial,
        iteration_max,
        bailout,
        max_perturb_iterations,
        max_bla_steps,
    )
}

/// Mirror exact de la boucle générique mais avec le pas perturbation
/// hardcoded Mandelbrot. Reste à l'identique : BLA lookup, bailout check,
/// rebase F3 strict, period clamp.
fn iterate_pixel_unified_exp_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResultExp {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
    }
    // Référence escape-time (non-périodique) tronquée à l'évasion : PAS de
    // fallback GMP. On rebase au bout de la référence (F3 `hybrid.cc:301` :
    // `m + 1 == size` ⇒ rebase z=Z+δ, m=0), ce qui permet aux pixels qui
    // survivent à la référence de continuer sur le path perturbation rapide.
    // (Ancien gate `ref_truncated` → exhausted → GMP par-pixel = ~50× plus lent,
    // cf. G2.) L'image uniforme e113 d'avant venait de l'ABSENCE de rebase au
    // bout, pas de la troncature elle-même.

    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;
    let ref_exhausted_flag = false;
    const REDUCE_INTERVAL: u32 = 250;

    let bailout_sqr_fexp = FloatExp::from_f64(bailout_sqr);
    // Cache tête-de-boucle (mirror `pixel_loop.rs`, style chaînage MipLA) : le
    // bas de boucle et la garde anti-over-skip calculent déjà `Z[m']+δ'` et sa
    // norme FloatExp — les réutiliser évite load + 2 from_f64 + 2 add + 2 sqr
    // redondants par itération. Bit-identique SAUF si `delta.reduce()` vient de
    // tirer (représentation renormalisée → décisions d'exposant possiblement
    // différentes) : dans ce cas on recalcule depuis le δ réduit, comme la
    // tête d'origine.
    let mut z_m = ref_orbit.z_ref_f64[0];
    let mut z_abs_re = FloatExp::from_f64(z_m.re) + delta.re;
    let mut z_abs_im = FloatExp::from_f64(z_m.im) + delta.im;
    let mut z_abs_norm_sqr = z_abs_re.sqr() + z_abs_im.sqr();
    let mut delta_norm_sqr_fexp = delta.norm_sqr_fexp();

    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        // Bailout en FloatExp pour préserver la différentiation pixel-à-pixel
        // au-delà de 2^1023 (où to_complex64_approx sature à inf et tous les
        // pixels bailent au même iter, cf. floral_fantasy).
        if !(z_abs_norm_sqr < bailout_sqr_fexp) {
            let delta_approx = delta.to_complex64_approx();
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_m + delta_approx,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        if let Some(node) = bla.lookup_fexp(m as usize, delta_norm_sqr_fexp) {
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            if new_n <= iteration_max && (new_m as usize) < ref_len {
                let a = node.a;
                let b = node.b;
                let new_re = (a.m00 * delta.re) + (a.m01 * delta.im)
                    + (b.m00 * dc.re) + (b.m01 * dc.im);
                let new_im = (a.m10 * delta.re) + (a.m11 * delta.im)
                    + (b.m10 * dc.re) + (b.m11 * dc.im);
                let cand = ComplexExp { re: new_re, im: new_im };
                // Garde anti-over-skip BLA (mirror `pixel_loop.rs`) : un saut
                // l ≥ 2 dont le point d'arrivée `Z[m']+δ'` est déjà échappé
                // rapporterait l'iter d'évasion trop tard. Test en FloatExp
                // (cohérent avec le bailout ci-dessus, survit à > 2^1023).
                // z_end calculé sans condition : cache de la tête suivante.
                let ze_m = ref_orbit.z_ref_f64[new_m as usize];
                let z_end_re = FloatExp::from_f64(ze_m.re) + cand.re;
                let z_end_im = FloatExp::from_f64(ze_m.im) + cand.im;
                let z_end_norm_sqr = z_end_re.sqr() + z_end_im.sqr();
                let overshoots_escape = node.l >= 2 && !(z_end_norm_sqr < bailout_sqr_fexp);
                if !overshoots_escape {
                    delta = cand;
                    n = new_n;
                    m = new_m;
                    bla_steps += 1;
                    z_m = ze_m;
                    if n % REDUCE_INTERVAL == 0 {
                        delta.reduce();
                        // δ renormalisé : recalcul de tête sur la représentation
                        // réduite (bit-identité avec la tête d'origine).
                        z_abs_re = FloatExp::from_f64(z_m.re) + delta.re;
                        z_abs_im = FloatExp::from_f64(z_m.im) + delta.im;
                        z_abs_norm_sqr = z_abs_re.sqr() + z_abs_im.sqr();
                    } else {
                        z_abs_re = z_end_re;
                        z_abs_im = z_end_im;
                        z_abs_norm_sqr = z_end_norm_sqr;
                    }
                    delta_norm_sqr_fexp = delta.norm_sqr_fexp();
                    continue;
                }
            }
        }

        // Pas perturbation Mandelbrot : δ' = 2·Z·δ + δ² + dc
        // Inline complet, pas de DeltaStateExp à allouer/lire/écrire.
        let two_z = z_m + z_m;
        let two_z_delta = delta.mul_complex64(two_z);
        // δ² est droppé par l'add dès que |2Zδ| ≫ |δ²| (l'add FloatExp renvoie
        // `self` inchangé quand `exp(self) - exp(rhs) >= 54`). À deep zoom δ est
        // minuscule ⇒ vrai sur la quasi-totalité des pas directs ; on saute alors
        // le calcul de δ² (6 frexp du `mul` + 2 des adds), **bit-identique**.
        // Condition conservatrice SANS calculer δ² : borne sup de son exposant =
        // `2·max(exp δ.re, exp δ.im) + 1` (produit + report), comparée aux deux
        // composantes NON NULLES de 2Zδ. Si l'une est nulle (Z=0 à l'itér. 0),
        // l'add ne dropperait pas δ² → on ne saute pas.
        let dsq_exp_ub = 2 * delta.re.exponent.max(delta.im.exponent) + 1;
        let skip_dsq = two_z_delta.re.mantissa != 0.0
            && two_z_delta.im.mantissa != 0.0
            && two_z_delta.re.exponent.min(two_z_delta.im.exponent) - dsq_exp_ub >= 54;
        delta = if skip_dsq {
            two_z_delta.add(dc)
        } else {
            let delta_sq = delta.mul(delta);
            two_z_delta.add(delta_sq).add(dc)
        };
        n += 1;
        m += 1;
        iters_ptb += 1;

        let delta_approx = delta.to_complex64_approx();
        if !delta_approx.re.is_finite() || !delta_approx.im.is_finite() {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta_approx,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }

        // Rebase F3 : |Z[m+1] + δ|² < |δ|² OR fin d'orbite. Pour les orbites
        // périodiques (centre intérieur), on cycle m via modulo (la cycle
        // commence à cycle_start > 0, donc rebaser vers m=0 re-traverserait
        // la queue pré-cycle → uniformisation). Sinon (orbite tronquée par
        // escape), on flag exhaustion : rebaser blind collerait tous les
        // pixels au même état (cf. e113.toml uniforme).
        // Rebase F3 (`hybrid.cc:296-307`) : `|Z[m]+δ|² < |δ|²` OU bout de
        // référence ⇒ z := Z[m]+δ, m := 0. En FloatExp pour ne pas saturer en
        // f64 deep zoom (cf. floral_fantasy). Pour les orbites PÉRIODIQUES on
        // garde l'optimisation `wrap_periodic` (cycle m sans recomputer la
        // queue) au lieu de rebaser à 0.
        // ⚠️ Après le pas, `m` peut valoir `ref_len` (un cran après le dernier
        // index) : NE PAS lire `z_ref[m]` sans garde (panic OOB). On clamp à
        // `ref_len-1` ; la branche périodique passe par `wrap_periodic` (ignore
        // z_curr), l'escape-time au bout rebase avec la dernière valeur valide.
        let end_of_ref = (m as usize) + 1 >= ref_len;
        let m_read = (m as usize).min(ref_len - 1);
        let z_m_new = ref_orbit.z_ref_f64[m_read];
        let z_curr_re = FloatExp::from_f64(z_m_new.re) + delta.re;
        let z_curr_im = FloatExp::from_f64(z_m_new.im) + delta.im;
        let z_curr_norm_sqr_fexp = z_curr_re.sqr() + z_curr_im.sqr();
        delta_norm_sqr_fexp = delta.norm_sqr_fexp();
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else {
                // Escape-time : rebase au bout (F3) au lieu d'abandonner.
                delta = ComplexExp { re: z_curr_re, im: z_curr_im };
                m = 0;
                rebase_count += 1;
                delta_norm_sqr_fexp = z_curr_norm_sqr_fexp;
            }
            // m (et éventuellement δ) a changé : recalcul de tête (froid).
            z_m = ref_orbit.z_ref_f64[m as usize];
            z_abs_re = FloatExp::from_f64(z_m.re) + delta.re;
            z_abs_im = FloatExp::from_f64(z_m.im) + delta.im;
            z_abs_norm_sqr = z_abs_re.sqr() + z_abs_im.sqr();
        } else if z_curr_norm_sqr_fexp < delta_norm_sqr_fexp {
            delta = ComplexExp { re: z_curr_re, im: z_curr_im };
            m = 0;
            rebase_count += 1;
            // |δ_nouveau|² = |z_curr|², déjà calculé.
            delta_norm_sqr_fexp = z_curr_norm_sqr_fexp;
            z_m = ref_orbit.z_ref_f64[0];
            z_abs_re = FloatExp::from_f64(z_m.re) + delta.re;
            z_abs_im = FloatExp::from_f64(z_m.im) + delta.im;
            z_abs_norm_sqr = z_abs_re.sqr() + z_abs_im.sqr();
        } else {
            // Pas de rebase : la tête suivante lit exactement z_curr.
            z_m = z_m_new;
            z_abs_re = z_curr_re;
            z_abs_im = z_curr_im;
            z_abs_norm_sqr = z_curr_norm_sqr_fexp;
        }
    }

    let final_m = m.min((ref_len - 1) as u32);
    let delta_approx = delta.to_complex64_approx();
    UnifiedPixelResultExp {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta_approx,
        rebase_count,
        bla_steps,
        ref_exhausted: ref_exhausted_flag,
    }
}

/// Variant HAUTE PRÉCISION du pixel loop Mandelbrot pour le path atom-tronqué
/// (référence périodique / graze deep zoom). Diffère du path f64 rapide :
/// - **référence lue en `ComplexExp`** (`ref_orbit.z_ref`) : les valeurs de
///   graze ~1e-8000 (où l'orbite frôle 0 à la fermeture de période) underflow
///   à 0 en `Complex64` → tue la reconstruction de l'évasion. F3 garde la réf
///   en `floatexp` pour la même raison.
/// - **pas de BLA** : la BLA `mat2<f64>` overflow (`A=∏2Z` ~1e444/période) sur
///   ces réfs. On fait des pas directs `δ'=2Zδ+δ²+dc` en `ComplexExp`.
/// Le rebase-at-end / `|Z+δ|<|δ|` est identique (mais en ComplexExp).
///
/// Avec un `bla_exp: Some(&BlaTableUnifiedExp)` (FloatExp), on skip des
/// itérations exactement comme le sibling f64 `iterate_pixel_unified_exp_mandelbrot`
/// (lookup+skip `δ := A·δ + B·dc`), mais les coefficients A/B sont en FloatExp
/// donc ne under/overflow pas sur les réfs graze deep zoom. `None` ⇒ pas de
/// skip (mode correctness-reference lent).
#[allow(clippy::too_many_arguments)]
fn iterate_pixel_unified_exp_mandelbrot_hp(
    ref_orbit: &ReferenceOrbit,
    bla_exp: Option<&BlaTableUnifiedExp>,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResultExp {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref.len();
    // `z_ref` (ComplexExp) et `z_ref_f64` sont poussés ensemble → même longueur.
    // Garde défensive : réf trop courte ou incohérente ⇒ résultat trivial.
    if ref_len < 2 || ref_orbit.z_ref_f64.len() != ref_len {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
    }
    let bailout_sqr_fexp = FloatExp::from_f64(bailout_sqr);
    let use_bla = !atom_nobla_debug();
    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;
    const REDUCE_INTERVAL: u32 = 250;

    // Structure de boucle F3 (`hybrid.cc:287-345`) : le **rebase** (`|Z+δ|²<|δ|²`
    // OU bout de référence) est testé au DÉBUT de chaque itération, AVANT le
    // lookup BLA. Un skip BLA repasse alors par le rebase avant le prochain
    // lookup. ⚠️ Sans ce placement (rebase seulement APRÈS un pas direct, `continue`
    // après un skip), un skip qui atterrit exactement à `ref_len-1` saute le
    // rebase-at-end → un pas direct parasite par-dessus le graze (δ→δ²) → off-by-one
    // à la fermeture de période → escape décalé de ~1 période (cf. e8000 : Δ~20000).
    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        // 1. Rebase (F3 hybrid.cc:296) — AVANT le lookup. Pour les orbites
        //    périodiques (cycle_period>0) on garde `wrap_periodic` (cycle m sans
        //    toucher δ) ; sinon rebase-to-0 (`δ := Z[m]+δ, m := 0`).
        let end_of_ref = (m as usize) + 1 >= ref_len;
        let z_m0 = ref_orbit.z_ref[m as usize];
        let z_curr = z_m0.add(delta);
        let rebase_mid = !end_of_ref && z_curr.norm_sqr_fexp() < delta.norm_sqr_fexp();
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else {
                delta = z_curr;
                m = 0;
                rebase_count += 1;
            }
        } else if rebase_mid {
            delta = z_curr;
            m = 0;
            rebase_count += 1;
        }

        // 2. Bailout : |Z[m] + δ|² (état POST-rebase, F3 `Zz2 < ER2`).
        let z_m = ref_orbit.z_ref[m as usize];
        let z_abs = z_m.add(delta);
        if !(z_abs.norm_sqr_fexp() < bailout_sqr_fexp) {
            let delta_approx = delta.to_complex64_approx();
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m as usize] + delta_approx,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        // 3. BLA lookup+skip FloatExp (mirror du sibling f64). δ := A·δ + B·dc,
        //    A/B mat2 FloatExp → pas d'under/overflow des coefs sur réf graze.
        //    DEBUG: `FRACTALL_ATOM_NOBLA=1` désactive le skip (path direct pur).
        if use_bla {
            if let Some(bla) = bla_exp {
                let delta_norm_sqr_fexp = delta.norm_sqr_fexp();
                if let Some(node) = bla.lookup_fexp(m as usize, delta_norm_sqr_fexp) {
                    let new_n = n.saturating_add(node.l);
                    let new_m = m.saturating_add(node.l);
                    if new_n <= iteration_max && (new_m as usize) < ref_len {
                        let a = node.a;
                        let b = node.b;
                        let new_re = a.m00 * delta.re + a.m01 * delta.im
                            + b.m00 * dc.re + b.m01 * dc.im;
                        let new_im = a.m10 * delta.re + a.m11 * delta.im
                            + b.m10 * dc.re + b.m11 * dc.im;
                        let cand = ComplexExp { re: new_re, im: new_im };
                        // Garde anti-over-skip BLA (mirror `pixel_loop.rs`) : rejeter
                        // un saut l ≥ 2 dont l'endpoint `Z[m']+δ'` est déjà échappé.
                        // Test en ComplexExp/FloatExp (réf `z_ref` HP, cohérent avec
                        // le bailout POST-rebase ci-dessus).
                        let overshoots_escape = node.l >= 2 && {
                            let z_end = ref_orbit.z_ref[new_m as usize].add(cand);
                            !(z_end.norm_sqr_fexp() < bailout_sqr_fexp)
                        };
                        if !overshoots_escape {
                            delta = cand;
                            n = new_n;
                            m = new_m;
                            bla_steps += 1;
                            if n % REDUCE_INTERVAL == 0 {
                                delta.reduce();
                            }
                            continue;
                        }
                    }
                }
            }
        }

        // 4. Pas perturbation Mandelbrot direct : δ' = 2·Z·δ + δ² + dc (ComplexExp).
        //    (Le rebase du prochain tour de boucle traite `|Z+δ|<|δ|`/bout de réf.)
        let two_z = ComplexExp { re: z_m.re + z_m.re, im: z_m.im + z_m.im };
        let two_z_delta = delta.mul(two_z);
        let delta_sq = delta.mul(delta);
        delta = two_z_delta.add(delta_sq).add(dc);
        n += 1;
        m += 1;
        iters_ptb += 1;

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }
    }

    let final_m = m.min((ref_len - 1) as u32);
    let delta_approx = delta.to_complex64_approx();
    UnifiedPixelResultExp {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta_approx,
        rebase_count,
        bla_steps,
        ref_exhausted: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn iterate_pixel_unified_exp_generic(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    phase: &Phase,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResultExp {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
    }
    // Référence escape-time tronquée : pas de fallback GMP — on rebase au bout
    // (F3 `hybrid.cc:301`). Cf. variante Mandelbrot pour le détail (G2).

    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;
    let ref_exhausted_flag = false;

    // Reduce périodique (cf. rust-fractal-core) pour éviter la perte de
    // précision graduelle sur les mantissas après beaucoup d'itérations.
    const REDUCE_INTERVAL: u32 = 250;

    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        let z_m = ref_orbit.z_ref_f64[m as usize];
        // Bailout : |Z + δ|² ≥ bailout² (les magnitudes après évasion sont
        // dans la range f64 normale donc to_complex64_approx est OK).
        let delta_approx = delta.to_complex64_approx();
        let z_abs = z_m + delta_approx;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        // BLA lookup. La norme du delta est calculée en FloatExp pour ne
        // pas underflow ; on prend l'approx f64 SEULEMENT pour comparer au
        // r² (qui est f64). Si delta est minuscule (exp très négatif), son
        // norm_sqr_approx vaudra 0.0 → toujours < r² → BLA toujours valide.
        // C'est cohérent : un delta très petit DOIT pouvoir être absorbé
        // par n'importe quel BLA.
        let delta_norm_sqr_fexp = delta.norm_sqr_fexp();
        if let Some(node) = bla.lookup_fexp(m as usize, delta_norm_sqr_fexp) {
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            if new_n <= iteration_max && (new_m as usize) < ref_len {
                // δ := A·δ + B·dc
                // A est mat2<f64>. δ peut être ComplexExp (très petit). On
                // multiplie chaque composante FloatExp par les coefs f64.
                let a = node.a;
                let b = node.b;
                let new_re = (a.m00 * delta.re) + (a.m01 * delta.im)
                    + (b.m00 * dc.re) + (b.m01 * dc.im);
                let new_im = (a.m10 * delta.re) + (a.m11 * delta.im)
                    + (b.m10 * dc.re) + (b.m11 * dc.im);
                let cand = ComplexExp {
                    re: new_re,
                    im: new_im,
                };
                // Garde anti-over-skip BLA (mirror `pixel_loop.rs`) : un saut
                // multi-pas (l ≥ 2) est linéarisé autour de la RÉFÉRENCE, aveugle
                // à la divergence propre du pixel. Si le point d'arrivée
                // `Z[m']+δ'` est déjà échappé, l'escape a eu lieu PENDANT le saut
                // → le pas BLA le rapporterait jusqu'à l-1 itérations trop tard
                // (bug over-skip, cf. julia-siegel côté f64 déjà corrigé ; le
                // path exp partageait le même bloc SANS ce guard). Escape
                // irréversible (|z| > ER ≫ |c|) → tester le seul endpoint suffit.
                let overshoots_escape = node.l >= 2 && {
                    let z_end =
                        ref_orbit.z_ref_f64[new_m as usize] + cand.to_complex64_approx();
                    z_end.norm_sqr() >= bailout_sqr
                };
                if !overshoots_escape {
                    delta = cand;
                    n = new_n;
                    m = new_m;
                    bla_steps += 1;
                    if n % REDUCE_INTERVAL == 0 {
                        delta.reduce();
                    }
                    continue;
                }
            }
        }

        // Pas perturbation via DeltaStateExp.
        let mut state = DeltaStateExp::new(z_m, delta);
        state.step(phase, c_ref, dc);
        delta = state.delta;
        n += 1;
        m += 1;
        iters_ptb += 1;

        // NaN / Inf check sur l'approx f64 (suffit pour détecter explosion).
        let delta_approx = delta.to_complex64_approx();
        if !delta_approx.re.is_finite() || !delta_approx.im.is_finite() {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta_approx,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }

        // Rebase F3 (`hybrid.cc:296-307`) : `|Z[m]+δ|² < |δ|²` OU bout de
        // référence ⇒ z := Z[m]+δ, m := 0. Orbites périodiques : `wrap_periodic`
        // (cycle m). Escape-time au bout : rebase à 0 (F3) au lieu d'abandonner
        // → la perturbation reste utilisable (pas de fallback GMP), cf. G2.
        // ⚠️ `m` peut valoir `ref_len` après le pas → clamp pour éviter le panic
        // OOB (la branche périodique passe par wrap_periodic, ignore z_curr).
        let end_of_ref = (m as usize) + 1 >= ref_len;
        let m_read = (m as usize).min(ref_len - 1);
        let z_m_new = ref_orbit.z_ref_f64[m_read];
        let z_curr_re = FloatExp::from_f64(z_m_new.re) + delta.re;
        let z_curr_im = FloatExp::from_f64(z_m_new.im) + delta.im;
        let z_curr_norm_sqr_fexp = z_curr_re.sqr() + z_curr_im.sqr();
        let delta_norm_sqr_fexp = delta.norm_sqr_fexp();
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else {
                delta = ComplexExp { re: z_curr_re, im: z_curr_im };
                m = 0;
                rebase_count += 1;
            }
        } else if z_curr_norm_sqr_fexp < delta_norm_sqr_fexp {
            delta = ComplexExp { re: z_curr_re, im: z_curr_im };
            m = 0;
            rebase_count += 1;
        }
    }

    let final_m = m.min((ref_len - 1) as u32);
    let delta_approx = delta.to_complex64_approx();
    UnifiedPixelResultExp {
        iteration: n,
        z_final: ref_orbit.z_ref_f64[final_m as usize] + delta_approx,
        rebase_count,
        bla_steps,
        ref_exhausted: ref_exhausted_flag,
    }
}

/// Hybride multi-phase deep (> 1e13) en ComplexExp. Mirror de
/// `iterate_pixel_unified_exp_generic` SANS BLA (la table est bâtie pour une
/// phase, inapplicable au mélange — cf. `pixel_loop.rs::iterate_pixel_unified_
/// multi_phase`, la version f64), avec cyclage de phase `phases[n % n_phases]`
/// et rebasing F3 en FloatExp. `ref_exhausted` reste `false` : comme le path
/// exp single-phase, on rebase inconditionnellement au bout de la référence
/// (F3 `hybrid.cc:301`), jamais de fallback GMP par-pixel.
#[allow(clippy::too_many_arguments)]
pub fn iterate_pixel_unified_exp_multi_phase(
    ref_orbit: &ReferenceOrbit,
    tables: &[BlaTableUnified],
    formula: &Formula,
    c_ref: Complex64,
    dc: ComplexExp,
    delta_initial: ComplexExp,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResultExp {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResultExp {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            ref_exhausted: false,
        };
    }
    let n_phases = formula.phases.len();
    // G4 jalon 5 : références PAR PHASE + tracking de phase (F3
    // `hybrid_references` + `hybrid.cc:266-341`), mirror du loop f64.
    // Invariant : `(phase + m) ≡ n (mod N)` ; chaque rebase (norme OU bout de
    // réf) fait `phase := (phase + m) % N` AVANT `m := 0`. Pas de wrap_periodic
    // (Brent gaté OFF pour les hybrides) ni d'exhaustion GMP (z²+c hardcodé).
    let refs = |p: usize| -> &ReferenceOrbit {
        if p == 0 { ref_orbit } else { &ref_orbit.hybrid_phase_refs[p - 1] }
    };
    // BLA par phase (jalon 5b) : active seulement avec une table par phase.
    let use_bla = tables.len() == n_phases;
    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut phase: usize = 0;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;
    const REDUCE_INTERVAL: u32 = 250;

    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        let z_m = refs(phase).z_ref_f64[m as usize];
        let delta_approx = delta.to_complex64_approx();
        let z_abs = z_m + delta_approx;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        // Saut BLA (jalon 5b) : table de la phase COURANTE, lookup FloatExp
        // (mirror du bloc exp générique single-phase : coefs f64, δ ComplexExp).
        // n et m avancent du même l → invariant (phase+m) ≡ n préservé.
        if use_bla {
            let ref_len_cur = refs(phase).z_ref_f64.len();
            let delta_norm_sqr_fexp = delta.norm_sqr_fexp();
            if let Some(node) = tables[phase].lookup_fexp(m as usize, delta_norm_sqr_fexp) {
                let new_n = n.saturating_add(node.l);
                let new_m = m.saturating_add(node.l);
                let lands_on_ref_end =
                    refs(phase).atom_truncated && (new_m as usize) + 1 >= ref_len_cur;
                if new_n <= iteration_max
                    && (new_m as usize) < ref_len_cur
                    && !lands_on_ref_end
                {
                    let a = node.a;
                    let b = node.b;
                    let new_re = (a.m00 * delta.re) + (a.m01 * delta.im)
                        + (b.m00 * dc.re) + (b.m01 * dc.im);
                    let new_im = (a.m10 * delta.re) + (a.m11 * delta.im)
                        + (b.m10 * dc.re) + (b.m11 * dc.im);
                    let cand = ComplexExp { re: new_re, im: new_im };
                    let overshoots_escape = node.l >= 2 && {
                        let z_end = refs(phase).z_ref_f64[new_m as usize]
                            + cand.to_complex64_approx();
                        z_end.norm_sqr() >= bailout_sqr
                    };
                    if !overshoots_escape {
                        delta = cand;
                        n = new_n;
                        m = new_m;
                        bla_steps += 1;
                        if n % REDUCE_INTERVAL == 0 {
                            delta.reduce();
                        }
                        // PAS de rebase-check post-saut : mirror du single-phase
                        // et du loop f64 multi-phase (check après pas DIRECT
                        // seulement — un rebase mid-chaîne casse la rampe
                        // géométrique des skips). Bout de réf sûr : saut
                        // dépassant rejeté par les bornes → pas direct → rebase
                        // au bas de boucle (avec màj de phase).
                        continue;
                    }
                }
            }
        }

        // Pas perturbation : phase courante = phases[n % n_phases].
        let ph = &formula.phases[(n as usize) % n_phases];
        let mut state = DeltaStateExp::new(z_m, delta);
        state.step(ph, c_ref, dc);
        delta = state.delta;
        n += 1;
        m += 1;
        iters_ptb += 1;

        let ref_len_cur = refs(phase).z_ref_f64.len();
        let delta_approx = delta.to_complex64_approx();
        if !delta_approx.re.is_finite() || !delta_approx.im.is_finite() {
            return UnifiedPixelResultExp {
                iteration: n,
                z_final: refs(phase).z_ref_f64[(m as usize).min(ref_len_cur - 1)]
                    + delta_approx,
                rebase_count,
                bla_steps,
                ref_exhausted: false,
            };
        }

        if n % REDUCE_INTERVAL == 0 {
            delta.reduce();
        }

        // Rebase F3 (`hybrid.cc:296-307`) : `|Z[m]+δ|² < |δ|²` OU bout de réf ⇒
        // δ := Z[m]+δ ; phase := (phase+m) % N ; m := 0.
        // ⚠️ `m` peut valoir `ref_len` après le pas → clamp anti-OOB (lecture
        // seulement ; la màj de phase utilise le `m` VRAI pour l'invariant).
        let end_of_ref = (m as usize) + 1 >= ref_len_cur;
        let m_read = (m as usize).min(ref_len_cur - 1);
        let z_m_new = refs(phase).z_ref_f64[m_read];
        let z_curr_re = FloatExp::from_f64(z_m_new.re) + delta.re;
        let z_curr_im = FloatExp::from_f64(z_m_new.im) + delta.im;
        let z_curr_norm_sqr_fexp = z_curr_re.sqr() + z_curr_im.sqr();
        let delta_norm_sqr_fexp = delta.norm_sqr_fexp();
        if end_of_ref || z_curr_norm_sqr_fexp < delta_norm_sqr_fexp {
            delta = ComplexExp { re: z_curr_re, im: z_curr_im };
            phase = (phase + m as usize) % n_phases;
            m = 0;
            rebase_count += 1;
        }
    }

    let ref_len_cur = refs(phase).z_ref_f64.len();
    let final_m = (m as usize).min(ref_len_cur - 1);
    let delta_approx = delta.to_complex64_approx();
    UnifiedPixelResultExp {
        iteration: n,
        z_final: refs(phase).z_ref_f64[final_m] + delta_approx,
        rebase_count,
        bla_steps,
        ref_exhausted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::{build_bla_table_for_formula, compile_formula};
    use crate::fractal::perturbation::orbit::compute_reference_orbit;
    use crate::fractal::{default_params_for_type, AlgorithmMode, FractalType};

    fn make_ref_orbit(
        center_x: f64,
        center_y: f64,
        zoom: f64,
        iter_max: u32,
    ) -> ReferenceOrbit {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x = center_x;
        params.center_y = center_y;
        let span_x = 4.0 / zoom;
        params.span_x = span_x;
        params.span_y = span_x * 100.0 / 160.0;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None, true).expect("ref orbit");
        orbit
    }

    /// À zoom 1e6, le pixel loop ComplexExp doit donner les mêmes
    /// classifications que la version f64 (iterate_pixel_unified) et que
    /// l'itération directe.
    #[test]
    fn unified_exp_matches_f64_at_zoom_1e6() {
        let iter_max = 1500u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;
        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables =
            build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8).expect("BLA");
        let bla = &tables[0];

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc_exp = ComplexExp::from_complex64(Complex64::new(dx, dy));
                let res = iterate_pixel_unified_exp(
                    &orbit,
                    bla,
                    None,
                    &formula,
                    orbit.cref,
                    dc_exp,
                    ComplexExp::zero(),
                    iter_max,
                    4.0,
                    0,
                    0,
                );
                let escaped_exp = res.iteration < iter_max;

                let c_abs = Complex64::new(cx + dx, cy + dy);
                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    z = z * z + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;
                if escaped_exp != escaped_direct {
                    mismatches += 1;
                }
                total += 1;
            }
        }
        assert_eq!(mismatches, 0, "exp doit classifier comme direct à zoom 1e6");
        let _ = total;
    }

    /// Test deep zoom : zoom 1e20. À cette profondeur, dc_f64 underflowerait
    /// (~ 1e-20 / 160 = 1e-22, encore OK f64 mais le delta après quelques
    /// itérations en δ² serait ~1e-44 puis 1e-88 etc.). L'exp version
    /// préserve la magnitude via l'exponent séparé.
    ///
    /// On ne peut pas comparer à f64 direct (qui pète à ce zoom), donc on
    /// vérifie juste que le pixel loop termine sans NaN/Inf et produit un
    /// résultat (escape ou intérieur) plausible.
    #[test]
    fn unified_exp_zoom_1e20_terminates() {
        let iter_max = 2000u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        // Pour zoom 1e20, on a besoin d'une vraie référence HP. Sans ça
        // le ref center fait perdre toute la précision avant même de
        // commencer. Skip le test si la fonction make_ref_orbit ne peut
        // pas générer une orbite suffisamment précise.
        let zoom = 1e20;
        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        if orbit.z_ref_f64.len() < 10 {
            eprintln!("orbite trop courte à zoom 1e20, skip");
            return;
        }
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA");
        let bla = &tables[0];

        let dc_exp = ComplexExp {
            re: FloatExp::new(1.0, -70),
            im: FloatExp::new(2.0, -70),
        };
        let res = iterate_pixel_unified_exp(
            &orbit,
            bla,
            None,
            &formula,
            orbit.cref,
            dc_exp,
            ComplexExp::zero(),
            iter_max,
            4.0,
            0,
            0,
        );
        // Pas de NaN, pas d'Inf.
        assert!(res.z_final.re.is_finite() && res.z_final.im.is_finite());
        // Le résultat est soit dans le set (n == iter_max) soit échappé
        // à un n < iter_max. Pas besoin de valeur précise — juste vivacité.
        eprintln!(
            "exp zoom 1e20 : n={} (max {}), z={:?}, rebases={}",
            res.iteration, iter_max, res.z_final, res.rebase_count
        );
    }

    /// Verrou over-skip BLA côté PATH EXP (mirror du sibling f64
    /// `bla_no_overskip_past_escape_julia_siegel`). Julia c=-0.8+0.156i : la
    /// référence (critique, centre 0,0) est BORNÉE (siegel) mais les pixels
    /// s'échappent → un saut BLA `l≥2` linéarisé autour de la réf bornée peut
    /// sauter par-dessus l'évasion propre du pixel et rapporter l'iter d'escape
    /// jusqu'à `l-1` trop tard. Le path exp partageait ce bloc SANS le guard
    /// (bug latent : exp seulement > 1e278 en prod, mais reproductible en
    /// forçant le path ; div_ratio 1.0, +2 uniforme sans le guard). Avec le
    /// guard, le path BLA suit le f64 direct au pixel près (biais ~0).
    #[test]
    fn exp_bla_no_overskip_past_escape_julia_siegel() {
        let width = 96u32;
        let height = 96u32;
        let iter_max = 3000u32;
        let (sx, sy) = (-0.8f64, 0.156f64);
        let zoom = 1e10f64;
        let bailout = 25.0f64;
        let bailout_sqr = bailout * bailout;

        let mut params = default_params_for_type(FractalType::Julia, width, height);
        params.center_x = 0.0;
        params.center_y = 0.0;
        params.seed = Complex64::new(sx, sy);
        let span_x = 4.0 / zoom;
        let span_y = span_x * height as f64 / width as f64;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.bailout = bailout;
        params.algorithm_mode = AlgorithmMode::Perturbation;

        let (orbit, _, _) =
            compute_reference_orbit(&params, None, true).expect("ref orbit (Julia)");
        let formula = compile_formula(FractalType::Julia, 2.0).unwrap();
        let c_norm = crate::fractal::perturbation::effective_pixel_size(&params)
            * ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA (Julia)");
        let bla = &tables[0];

        let seed = Complex64::new(sx, sy);
        let plain_escape = |z0x: f64, z0y: f64| -> Option<u32> {
            let (mut zx, mut zy) = (z0x, z0y);
            let mut i = 0u32;
            while i < iter_max {
                if zx * zx + zy * zy >= bailout_sqr {
                    return Some(i);
                }
                let nzx = zx * zx - zy * zy + sx;
                let nzy = 2.0 * zx * zy + sy;
                zx = nzx;
                zy = nzy;
                i += 1;
            }
            None
        };

        let mut sum_signed = 0i64;
        let mut sum_abs = 0i64;
        let mut escaped = 0i64;
        for j in 0..height {
            for i in 0..width {
                let dx = ((i as f64 + 0.5) / width as f64 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / height as f64 - 0.5) * span_y;
                // Julia : delta_init = pixel, c_ref = seed, dc = 0.
                let delta_init = ComplexExp::from_complex64(Complex64::new(dx, dy));
                let res = iterate_pixel_unified_exp(
                    &orbit,
                    bla,
                    None,
                    &formula,
                    seed,
                    ComplexExp::zero(),
                    delta_init,
                    iter_max,
                    bailout,
                    0,
                    0,
                );
                if let Some(plain) = plain_escape(dx, dy) {
                    if res.iteration < iter_max {
                        let d = res.iteration as i64 - plain as i64;
                        sum_signed += d;
                        sum_abs += d.abs();
                        escaped += 1;
                    }
                }
            }
        }

        assert!(escaped > 500, "trop peu de pixels échappés ({escaped})");
        let mean_signed = sum_signed as f64 / escaped as f64;
        let mean_abs = sum_abs as f64 / escaped as f64;
        assert!(
            mean_signed.abs() < 0.5,
            "biais d'iter escape BLA exp vs f64 direct = {mean_signed:.4} (over-skip non gardé côté exp ?)"
        );
        assert!(
            mean_abs < 0.5,
            "erreur moyenne iter BLA exp vs f64 direct = {mean_abs:.4} (over-skip BLA exp)"
        );
    }
}
