//! Pixel loop perturbation F3-pur avec BLA mat2 unifiée.
//!
//! Mirror direct de `hybrid_render` de Fraktaler-3 (cf.
//! `docs/fraktaler-3-analysis.md` §3). Architecture :
//! 1. BLA lookup : `bla.lookup(m, |δ|²)`. Si trouvé : `δ := A·δ + B·dc`,
//!    skip `l` itérations.
//! 2. Sinon : pas perturbation via la forme delta de l'opcode.
//! 3. Check rebasing F3 strict : `|Z[m+1] + δ|² < |δ|²` OR fin d'orbite
//!    → reseat `δ := Z[m+1] + δ`, `m := 0`.
//!
//! Pas de glitch detection (Pauldelbrot). Pas de secondary refs. C'est
//! l'objectif F3 : remplacer toute la machinerie glitch par le rebasing
//! proactif.
//!
//! Limitations actuelles (incrémental) :
//! - Mandelbrot uniquement pour le pas perturbation (formule
//!   `δ' = 2·Z·δ + δ² + dc` hardcodée). Les autres types appellent le
//!   path existant via fallback. Extension à BS/Tricorn/Celtic/Multibrot
//!   = chantier suivant (utilise la forme delta du bytecode).
//! - f64 path (pas de ComplexExp pour très deep zoom).
//! - Mono-phase seulement (la `BlaTableUnified` est passée seule, pas
//!   `Vec<BlaTableUnified>`).

use num_complex::Complex64;

use super::bla_dual::BlaTableUnified;
use super::delta_form::DeltaState;
use super::{Formula, Phase};
use crate::fractal::orbit_traps::{OrbitData, OrbitTrapType};
use crate::fractal::perturbation::orbit::ReferenceOrbit;

/// Résultat d'un pixel via le pixel loop unifié.
pub struct UnifiedPixelResult {
    pub iteration: u32,
    pub z_final: Complex64,
    /// Nombre de rebases effectués (utile pour stats/debug).
    #[allow(dead_code)]
    pub rebase_count: u32,
    /// Nombre de pas BLA appliqués (utile pour stats/debug).
    #[allow(dead_code)]
    pub bla_steps: u32,
    /// Orbit data (uniquement si demandé via param). Stocke les z_abs
    /// traversés pour le calcul de orbit traps (distance min à un point/
    /// ligne/croix/cercle).
    #[allow(dead_code)]
    pub orbit: Option<OrbitData>,
    /// Distance estimate à la frontière du set (formule 2|z|·ln|z|/|dz|).
    /// `None` si non demandé ou pixel intérieur.
    pub distance: Option<f64>,
    /// `true` si le pixel est détecté comme intérieur (|dz| < threshold à
    /// `iter_max`). Encoder éventuellement via signe z.im pour le pipeline
    /// coloring (cf. convention legacy).
    pub is_interior: bool,
    /// `true` si la boucle a quitté parce que l'orbite référence s'est
    /// épuisée avant `iteration_max` (centre escape-time, non-périodique).
    /// L'iteration count retourné est non fiable : le caller doit
    /// re-rendre via `iterate_pixel_gmp` (per-pixel GMP) pour obtenir le
    /// vrai iter d'évasion, sinon tous les pixels se collent à `ref_len`
    /// (cf. e113.toml uniforme avant ce flag).
    pub ref_exhausted: bool,
}

/// Options pour le pixel loop unifié (dual-numbers features).
#[derive(Clone, Copy, Debug, Default)]
pub struct UnifiedOptions {
    pub orbit_trap: Option<OrbitTrapType>,
    pub enable_distance: bool,
    pub enable_interior: bool,
    pub interior_threshold: f64,
    /// `true` si Julia-like : ddelta init=1, dc=0, distance factor=1.
    /// `false` si Mandelbrot-like : ddelta init=0, dc≠0, distance factor=2.
    pub is_julia: bool,
    /// Cap "raw perturbation steps" par pixel (0 = illimité). Aligné F3
    /// `bailout.maximum_perturb_iterations` (`param.h:39`). Les pas BLA ne
    /// comptent pas. Quand atteint, la boucle sort avec `iteration = n`
    /// (smooth coloring color le pixel comme "échappé tard" plutôt
    /// qu'intérieur, cf. `cl-post.cl:21`).
    pub max_perturb_iterations: u32,
    /// Cap "BLA jumps" par pixel (0 = illimité). Aligné F3
    /// `bailout.maximum_bla_steps`.
    pub max_bla_steps: u32,
}

/// Pixel loop unifié pour TOUS les types escape-time supportés par le bytecode :
/// BLA mat2 + interpréteur delta-form + rebasing F3.
///
/// Généralise `iterate_pixel_unified_mandelbrot` à tous les types via
/// `DeltaState::step` qui propage (Z, δ) selon la règle de chaîne propre
/// à chaque opcode.
///
/// - `ref_orbit` : orbite référence (f64 path).
/// - `bla` : table BLA unifiée pour la phase.
/// - `formula` : compile_formula(type, multibrot_power) — défini la phase
///   appliquée à chaque itération.
/// - `c_ref` : constante ajoutée à la reference (cref pour Mandelbrot-like,
///   seed pour Julia-like).
/// - `dc` : offset = c_pixel - c_ref.
#[allow(dead_code)]
pub fn iterate_pixel_unified(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    formula: &Formula,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
) -> UnifiedPixelResult {
    iterate_pixel_unified_with_options(
        ref_orbit, bla, formula, c_ref, dc, delta_initial, iteration_max, bailout, None,
    )
}

/// Variante avec orbit_trap_type optionnel. Quand `Some`, désactive la BLA
/// (pour tracker chaque z_abs individuel) et construit un `OrbitData` qu'on
/// retourne dans le résultat. Hook pour `OutColoringMode::OrbitTraps` et
/// `Wings` qui ont besoin de l'orbite complète.
#[allow(dead_code)]
pub fn iterate_pixel_unified_with_options(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    formula: &Formula,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
    orbit_trap_type: Option<OrbitTrapType>,
) -> UnifiedPixelResult {
    iterate_pixel_unified_full(
        ref_orbit,
        bla,
        formula,
        c_ref,
        dc,
        delta_initial,
        iteration_max,
        bailout,
        UnifiedOptions {
            orbit_trap: orbit_trap_type,
            ..Default::default()
        },
    )
}

/// Version complète avec toutes les options dual-numbers (distance,
/// interior, orbit_traps). Propage `ddelta = ∂δ/∂dc` à travers la boucle
/// pour permettre :
/// - Distance estimation : `d = factor·|z|·ln|z|/|dz|` à l'évasion
/// - Interior detection : `|dz| < threshold` à `iter_max`
///
/// Le caller doit fournir `options.is_julia` correctement (Julia-like
/// utilise ddelta init = 1 et Add n'incrémente pas ddelta ; Mandelbrot-like
/// utilise ddelta init = 0 et Add fait ddelta += 1).
pub fn iterate_pixel_unified_full(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    formula: &Formula,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
    options: UnifiedOptions,
) -> UnifiedPixelResult {
    // Mono-phase : path optimisé direct.
    if formula.phases.len() == 1 {
        return iterate_pixel_unified_single_phase(
            ref_orbit,
            bla,
            &formula.phases[0],
            c_ref,
            dc,
            delta_initial,
            iteration_max,
            bailout,
            options,
        );
    }
    // Multi-phase : cycle de phases. Pour l'instant on partage la même BLA
    // table (mono-orbit) et c_ref pour toutes les phases ; ça suppose que
    // l'orbite référence a été itérée avec une formule cyclant les phases.
    // Une vraie implémentation hybride F3 a un Vec<ReferenceOrbit> et un
    // Vec<BlaTableUnified> (une par phase). À implémenter quand on aura
    // une vraie formule multi-phase dans compile_formula (actuellement
    // mono-phase seulement).
    iterate_pixel_unified_multi_phase(
        ref_orbit,
        bla,
        formula,
        c_ref,
        dc,
        delta_initial,
        iteration_max,
        bailout,
        options.max_perturb_iterations,
    )
}

/// Multi-phase mono-orbit. Conservé pour Vec<Phase> ; cycle l'index de
/// phase à chaque itération via le compteur n (pas via la position dans
/// la formule, pour rester cohérent quand un BLA step saute `l` itérations).
fn iterate_pixel_unified_multi_phase(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    formula: &Formula,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
) -> UnifiedPixelResult {
    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResult {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
        };
    }
    let n_phases = formula.phases.len();
    // G4 jalon 5 : références PAR PHASE (F3 `hybrid_references`). `refs(p)` =
    // orbite itérée avec `phases[(p+i) % N]`. Invariant de la boucle (F3
    // `hybrid.cc:266-341`) : `(phase + m) ≡ n (mod N)` — le pas pixel
    // `phases[n % N]` correspond TOUJOURS au pas de référence `Z[m]→Z[m+1]`.
    // Chaque rebase (norme OU bout de réf) fait `phase := (phase + m) % N`
    // AVANT `m := 0`, préservant l'invariant. Sans ça, tout rebase avec
    // `n % N ≠ 0` désynchronise → garbage (prouvé [M,BS] @3e10 uniforme).
    let refs = |p: usize| -> &ReferenceOrbit {
        if p == 0 { ref_orbit } else { &ref_orbit.hybrid_phase_refs[p - 1] }
    };
    let mut delta = delta_initial;
    let mut n = 0u32;
    let mut m = 0u32;
    let mut phase: usize = 0;
    let mut rebase_count = 0u32;
    let bla_steps = 0u32;
    let mut iters_ptb = 0u32;

    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
    {
        let z_m = refs(phase).z_ref_f64[m as usize];
        let z_abs = z_m + delta;
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResult {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
                orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
            };
        }
        // En multi-phase, on n'utilise pas la BLA pour simplifier
        // (la BLA construite l'a été pour une phase précise, pas applicable
        // au mélange). Une vraie BLA multi-phase est un travail séparé.
        let _ = bla;

        // Pas perturbation : phase courante = phases[n % n_phases] — cohérent
        // avec le pas de référence grâce à l'invariant (phase + m) ≡ n.
        let ph = &formula.phases[(n as usize) % n_phases];
        let mut state = super::delta_form::DeltaState::new(z_m, delta);
        state.step(ph, c_ref, dc);
        delta = state.delta;
        n += 1;
        m += 1;
        iters_ptb += 1;

        let ref_len_cur = refs(phase).z_ref_f64.len();
        if !delta.re.is_finite() || !delta.im.is_finite() {
            return UnifiedPixelResult {
                iteration: n,
                z_final: refs(phase).z_ref_f64[(m as usize).min(ref_len_cur - 1)] + delta,
                rebase_count,
                bla_steps,
                orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
            };
        }

        // Rebase F3 (`hybrid.cc:296-307`) : norme OU bout de référence ⇒
        // `δ := Z[m]+δ ; phase := (phase+m) % N ; m := 0`. Pas de wrap_periodic
        // ni de fallback GMP par-pixel en multi-phase : le Brent est gaté OFF
        // pour les hybrides (cycle_period=0) et `iterate_pixel_gmp` est z²+c
        // hardcodé (ne cycle pas) — le rebase-at-end F3 couvre les réfs
        // tronquées par escape (chaque réf de phase repart de Z[0]=0).
        let m_read = (m as usize).min(ref_len_cur - 1);
        let z_m_new = refs(phase).z_ref_f64[m_read];
        let z_curr = z_m_new + delta;
        let end_of_ref = (m as usize) + 1 >= ref_len_cur;
        if end_of_ref || z_curr.norm_sqr() < delta.norm_sqr() {
            delta = z_curr;
            phase = (phase + m as usize) % n_phases;
            m = 0;
            rebase_count += 1;
        }
    }
    let ref_len_cur = refs(phase).z_ref_f64.len();
    let final_m = (m as usize).min(ref_len_cur - 1);
    UnifiedPixelResult {
        iteration: n,
        z_final: refs(phase).z_ref_f64[final_m] + delta,
        rebase_count,
        bla_steps,
                orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
    }
}

fn iterate_pixel_unified_single_phase(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    phase: &Phase,
    c_ref: Complex64,
    dc: Complex64,
    delta_initial: Complex64,
    iteration_max: u32,
    bailout: f64,
    options: UnifiedOptions,
) -> UnifiedPixelResult {
    // Hot path Mandelbrot : phase exactement [Sqr, Add] + aucune feature
    // dual-numbers (distance/interior/orbit_traps) + Mandelbrot-like (not
    // Julia, delta_initial ≈ 0). Dispatch vers `iterate_pixel_unified_mandelbrot`
    // qui inline `δ' = 2·Z·δ + δ² + dc` sans DeltaState et sans dispatch opcode.
    let is_mandelbrot_phase = phase.ops.len() == 2
        && matches!(phase.ops[0], super::Op::Sqr)
        && matches!(phase.ops[1], super::Op::Add);
    let no_dual_features = options.orbit_trap.is_none()
        && !options.enable_distance
        && !options.enable_interior;
    if is_mandelbrot_phase && no_dual_features && !options.is_julia
        && delta_initial.norm_sqr() == 0.0
    {
        let _ = c_ref;
        return iterate_pixel_unified_mandelbrot(
            ref_orbit,
            bla,
            dc,
            iteration_max,
            bailout,
            options.max_perturb_iterations,
            options.max_bla_steps,
        );
    }

    let bailout_sqr = bailout * bailout;
    let ref_len = ref_orbit.z_ref_f64.len();
    if ref_len < 2 {
        return UnifiedPixelResult {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            orbit: options.orbit_trap.map(OrbitData::new),
            distance: None,
            is_interior: false,
            ref_exhausted: false,
        };
    }

    // Quand orbit traps demandé, on tracke chaque z_abs → désactive la BLA.
    let mut orbit_data = options.orbit_trap.map(OrbitData::new);
    let bla_enabled = orbit_data.is_none();
    let track_ddelta = options.enable_distance || options.enable_interior;

    let mut delta = delta_initial;
    // ddelta init : 0 pour Mandelbrot-like (δ₀=0 indépendant de dc),
    //               1 pour Julia-like (δ₀ = dc, ∂/∂dc = I).
    let mut ddelta = if options.is_julia {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    };
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;
    let max_ptb = options.max_perturb_iterations;
    let max_bla = options.max_bla_steps;
    let mut ref_exhausted_flag = false;

    // Point initial pour orbit traps.
    if let Some(ref mut od) = orbit_data {
        od.add_point(ref_orbit.z_ref_f64[0] + delta, 0);
    }

    while n < iteration_max
        && (max_ptb == 0 || iters_ptb < max_ptb)
        && (max_bla == 0 || bla_steps < max_bla)
    {
        let z_m = ref_orbit.z_ref_f64[m as usize];
        let z_abs = z_m + delta;
        if z_abs.norm_sqr() >= bailout_sqr {
            // Distance estimation à l'évasion.
            let distance = if options.enable_distance && n > 0 {
                compute_distance_from_dz(z_abs, ddelta, options.is_julia)
            } else {
                None
            };
            return UnifiedPixelResult {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
                orbit: orbit_data,
                distance,
                is_interior: false,
                ref_exhausted: false,
            };
        }

        // BLA lookup (skipped when orbit traps demande tracking par itération).
        let delta_norm_sqr = delta.norm_sqr();
        if bla_enabled {
            if let Some(node) = bla.lookup(m as usize, delta_norm_sqr) {
                let new_n = n.saturating_add(node.l);
                let new_m = m.saturating_add(node.l);
                // Réf atom-tronquée : interdire au BLA d'atterrir SUR la dernière
                // entrée (le graze |Z[end]| ~ atome). Le `continue` BLA saute le
                // check end-of-ref du bas de boucle → le pas direct suivant
                // partirait du graze (δ' ≈ δ²+dc, minuscule) puis le rebase
                // ajouterait Z[end] ≫ δ' qui ABSORBE δ en f64 → tous les pixels
                // identiques → image intérieure uniforme (cf. G2 mid-range atom).
                // En forçant l'arrivée en fin de réf par pas direct, le check du
                // bas rebase AVANT le pas de graze (ordre F3 `hybrid.cc:295-308`).
                let lands_on_ref_end =
                    ref_orbit.atom_truncated && (new_m as usize) + 1 >= ref_len;
                if new_n <= iteration_max && (new_m as usize) < ref_len && !lands_on_ref_end {
                    let a = node.a;
                    let b = node.b;
                    let (a_re, a_im) = (
                        a.m00 * delta.re + a.m01 * delta.im,
                        a.m10 * delta.re + a.m11 * delta.im,
                    );
                    let (b_re, b_im) = (
                        b.m00 * dc.re + b.m01 * dc.im,
                        b.m10 * dc.re + b.m11 * dc.im,
                    );
                    let cand = Complex64::new(a_re + b_re, a_im + b_im);
                    // Garde anti-over-skip BLA : un saut multi-pas (l ≥ 2) est
                    // linéarisé autour de la RÉFÉRENCE, aveugle à la divergence
                    // propre du pixel. Si le point d'arrivée `Z[m']+δ'` est déjà
                    // échappé, l'escape a eu lieu PENDANT le saut → le pas BLA le
                    // rapporterait jusqu'à l-1 itérations trop tard (F3 fait cet
                    // over-skip ; le GMP pur non — cf. julia-siegel FAIL max_diff=2).
                    // L'escape étant irréversible (|z| > ER ≫ |c|), tester le seul
                    // point d'arrivée suffit. On rejette alors le saut et on
                    // single-steppe pour trouver l'itér d'évasion exacte.
                    let overshoots_escape = node.l >= 2 && {
                        let z_end = ref_orbit.z_ref_f64[new_m as usize] + cand;
                        z_end.norm_sqr() >= bailout_sqr
                    };
                    if !overshoots_escape {
                        // ddelta : applique aussi le BLA. ddelta' = A·ddelta + B (col0 pour dc).
                        if track_ddelta {
                            let (dd_re, dd_im) = (
                                a.m00 * ddelta.re + a.m01 * ddelta.im + b.m00,
                                a.m10 * ddelta.re + a.m11 * ddelta.im + b.m10,
                            );
                            ddelta = Complex64::new(dd_re, dd_im);
                        }
                        delta = cand;
                        n = new_n;
                        m = new_m;
                        bla_steps += 1;

                        if !delta.re.is_finite() || !delta.im.is_finite() {
                            return UnifiedPixelResult {
                                iteration: n,
                                z_final: ref_orbit.z_ref_f64[m as usize] + delta,
                                rebase_count,
                                bla_steps,
                                orbit: orbit_data,
                                distance: None,
                                is_interior: false,
                                ref_exhausted: false,
                            };
                        }
                        continue;
                    }
                }
            }
        }

        // Pas perturbation via delta-form interpreter (avec tracking ddelta
        // si demandé via track_ddelta).
        let mut state = if track_ddelta {
            DeltaState::with_ddelta(z_m, delta, ddelta)
        } else {
            DeltaState::new(z_m, delta)
        };
        if track_ddelta {
            state.step_with_julia(phase, c_ref, dc, options.is_julia);
        } else {
            state.step(phase, c_ref, dc);
        }
        delta = state.delta;
        if track_ddelta {
            ddelta = state.ddelta;
        }
        n += 1;
        m += 1;
        iters_ptb += 1;

        if !delta.re.is_finite() || !delta.im.is_finite() {
            return UnifiedPixelResult {
                iteration: n,
                z_final: ref_orbit.z_ref_f64[m.min((ref_len - 1) as u32) as usize] + delta,
                rebase_count,
                bla_steps,
                orbit: orbit_data,
                distance: None,
                is_interior: false,
                ref_exhausted: false,
            };
        }

        // Hook orbit traps : ajouter z_abs après le pas perturbation.
        if let Some(ref mut od) = orbit_data {
            let z_m_new = if (m as usize) < ref_len {
                ref_orbit.z_ref_f64[m as usize]
            } else {
                ref_orbit.z_ref_f64[ref_len - 1]
            };
            od.add_point(z_m_new + delta, n);
        }

        // Rebase F3 (ddelta inchangé par rebase : δ_new = Z+δ, d(Z+δ)/d(dc) = d(δ)/d(dc)).
        // Quand l'orbite référence est tronquée par période détectée (interior
        // center), on cycle m via modulo (orbite cyclique, valeurs identiques
        // à epsilon près) — évite l'uniformisation observée sur glitch_test_1.
        // Réf tronquée atom-domain : rebase-at-end F3 (cf. site homologue de
        // `iterate_pixel_unified_mandelbrot` plus bas). Sinon (orbite échappée
        // ou non-périodique), on flag exhaustion : le rebase blind sur
        // z_ref[end] uniformiserait tous les pixels post-escape
        // (cf. e113.toml). Le caller route ces pixels vers `iterate_pixel_gmp`.
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
            } else if ref_orbit.atom_truncated {
                let z_m_end = ref_orbit.z_ref_f64[(m as usize).min(ref_len - 1)];
                delta = z_m_end + delta;
                m = 0;
                rebase_count += 1;
            } else {
                ref_exhausted_flag = true;
                break;
            }
        } else {
            let z_m_new = ref_orbit.z_ref_f64[m as usize];
            let z_curr = z_m_new + delta;
            let z_curr_norm_sqr = z_curr.norm_sqr();
            let delta_norm_sqr = delta.norm_sqr();
            if z_curr_norm_sqr < delta_norm_sqr {
                delta = z_curr;
                m = 0;
                rebase_count += 1;
            }
        }
    }

    // Pixel intérieur (n == iteration_max) : check interior.
    let final_m = m.min((ref_len - 1) as u32);
    let z_final = ref_orbit.z_ref_f64[final_m as usize] + delta;
    let is_interior = if options.enable_interior {
        let dd_norm = ddelta.norm();
        dd_norm.is_finite() && dd_norm > 0.0 && dd_norm < options.interior_threshold
    } else {
        false
    };
    // Encoder is_interior via le signe de z.im (convention legacy).
    let z_out = if is_interior {
        Complex64::new(z_final.re, -z_final.im.abs())
    } else {
        z_final
    };
    UnifiedPixelResult {
        iteration: n,
        z_final: z_out,
        rebase_count,
        bla_steps,
        orbit: orbit_data,
        distance: None,
        is_interior,
        ref_exhausted: ref_exhausted_flag,
    }
}

/// Formule distance estimation à l'évasion. `z_abs` = Z + δ au moment de
/// bailout, `dz` = ddelta = ∂δ/∂dc. Factor 2 pour Mandelbrot, 1 pour Julia.
fn compute_distance_from_dz(z_abs: Complex64, dz: Complex64, is_julia: bool) -> Option<f64> {
    let z_norm = z_abs.norm().max(2.0);
    let dz_norm = dz.norm();
    if dz_norm > 1e-300 && z_norm > 1.0 {
        let factor = if is_julia { 1.0 } else { 2.0 };
        Some(factor * z_norm * z_norm.ln() / dz_norm)
    } else {
        None
    }
}

/// Source de lecture de la référence f64 pour le fast-path Mandelbrot
/// (PTWithCompression G8.2 phase 2). Deux impls :
/// - [`SliceSource`] : indexation directe de `z_ref_f64` (path par défaut,
///   inline zéro coût, BIT-IDENTIQUE à l'ancienne indexation).
/// - [`CompressedSource`] : décompresseur à waypoints Imagina
///   (`FRACTALL_COMPRESS_REF=1`), mémoire O(waypoints).
///
/// La boucle n'accède la référence QUE séquentiellement (`advance`), aux
/// rebases (`reset`), à l'atterrissage d'un saut BLA accepté (`teleport`,
/// valeur exacte = `BlaMultiStep::z_land`) et au rebase-at-end (`end_value`).
trait RefF64Source {
    /// État → itération 0, rend `Z[0]` (= 0 pour Mandelbrot seed 0).
    fn reset(&mut self) -> Complex64;
    /// Itération += 1, rend `Z[iter]`. Précondition : le nouvel index est
    /// dans l'orbite (le caller garde `m < ref_len`).
    fn advance(&mut self) -> Complex64;
    /// État → itération `m` (atterrissage BLA) avec la valeur EXACTE fournie
    /// (`node.z_land`, bit-copie de l'orbite). Rend `Z[m]`. À n'appeler QUE si
    /// le saut est ACCEPTÉ (la garde anti-over-skip lit `z_land` sans état).
    fn teleport(&mut self, m: u32, z_exact: Complex64) -> Complex64;
    /// `Z[ref_len-1]` (rebase-at-end F3). Ne modifie pas l'état.
    fn end_value(&self) -> Complex64;
    /// Accès arbitraire `Z[m]` SANS valeur exacte fournie (wrap_periodic
    /// Brent, `cycle_period > 0`). Jamais atteint sur la source compressée
    /// (routage gaté `cycle_period == 0`, cf. delta.rs) — l'impl compressée
    /// reste correcte (replay) mais lente.
    fn wrap(&mut self, m: u32) -> Complex64;
}

/// Indexation directe du tableau `z_ref_f64` (path par défaut).
struct SliceSource<'a> {
    z: &'a [Complex64],
    cursor: usize,
}

impl RefF64Source for SliceSource<'_> {
    #[inline(always)]
    fn reset(&mut self) -> Complex64 {
        self.cursor = 0;
        self.z[0]
    }
    #[inline(always)]
    fn advance(&mut self) -> Complex64 {
        self.cursor += 1;
        self.z[self.cursor]
    }
    #[inline(always)]
    fn teleport(&mut self, m: u32, _z_exact: Complex64) -> Complex64 {
        self.cursor = m as usize;
        self.z[self.cursor]
    }
    #[inline(always)]
    fn end_value(&self) -> Complex64 {
        self.z[self.z.len() - 1]
    }
    #[inline(always)]
    fn wrap(&mut self, m: u32) -> Complex64 {
        self.cursor = m as usize;
        self.z[self.cursor]
    }
}

/// Lecture via le décompresseur à waypoints (`FRACTALL_COMPRESS_REF=1`).
struct CompressedSource<'a> {
    dec: crate::fractal::perturbation::compress::ReferenceDecompressor<'a>,
}

impl RefF64Source for CompressedSource<'_> {
    #[inline]
    fn reset(&mut self) -> Complex64 {
        self.dec.reset()
    }
    #[inline]
    fn advance(&mut self) -> Complex64 {
        self.dec.next()
    }
    #[inline]
    fn teleport(&mut self, m: u32, z_exact: Complex64) -> Complex64 {
        // `z_exact` = `node.z_land` (valeur f64 exacte de l'orbite) : le seek
        // repart d'un point exact → erreur de replay ≤ fantôme canonique.
        self.dec.seek(m, z_exact);
        z_exact
    }
    #[inline]
    fn end_value(&self) -> Complex64 {
        self.dec.end_value()
    }
    fn wrap(&mut self, m: u32) -> Complex64 {
        // Jamais atteint (routage gaté `cycle_period == 0`) : replay complet
        // correct-mais-lent, par sûreté.
        debug_assert!(false, "wrap_periodic sur source compressée (gate cycle_period)");
        let mut z = self.dec.reset();
        for _ in 0..m {
            z = self.dec.next();
        }
        z
    }
}

/// Pixel loop Mandelbrot spécialisé : BLA mat2 + perturbation delta + rebasing F3.
/// Inline `δ' = 2·Z·δ + δ² + dc` sans DeltaState ni dispatch opcode, ~2× plus
/// rapide que `iterate_pixel_unified_full` pour le cas Mandelbrot pur.
///
/// Dispatché automatiquement depuis `iterate_pixel_unified_single_phase` quand :
/// - Phase = [Sqr, Add] (Mandelbrot)
/// - No orbit_trap / distance / interior
/// - !is_julia
/// - delta_initial = 0
///
/// - `ref_orbit` : orbite référence (f64 path).
/// - `bla` : table BLA unifiée pour la phase.
/// - `dc` : offset du pixel par rapport au centre de la référence (Complex64).
/// - `iteration_max`, `bailout` : caps standards.
pub fn iterate_pixel_unified_mandelbrot(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    dc: Complex64,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResult {
    let ref_len = ref_orbit.z_ref_f64.len();
    let mut src = SliceSource { z: &ref_orbit.z_ref_f64, cursor: 0 };
    iterate_pixel_unified_mandelbrot_impl(
        ref_orbit,
        &mut src,
        ref_len,
        bla,
        dc,
        iteration_max,
        bailout,
        max_perturb_iterations,
        max_bla_steps,
    )
}

/// Variante COMPRESSÉE (`FRACTALL_COMPRESS_REF=1`, G8.2 phase 2) : lit la
/// référence via `ReferenceOrbit::compressed_f64` (waypoints + replay f64) au
/// lieu de `z_ref_f64` — qui peut avoir été LIBÉRÉ (cf. mod.rs
/// `strip_orbit_arrays_for_compress`). `ref_len` = longueur LOGIQUE
/// (`CompressedReference::len`). Routage : `delta.rs`
/// (`compressed_ref_route_active`), Mandelbrot f64 pur uniquement.
pub fn iterate_pixel_unified_mandelbrot_compressed(
    ref_orbit: &ReferenceOrbit,
    bla: &BlaTableUnified,
    dc: Complex64,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResult {
    let Some(compressed) = ref_orbit.compressed_f64.as_ref() else {
        // Garde défensive : sans réf compressée, path plein classique.
        return iterate_pixel_unified_mandelbrot(
            ref_orbit,
            bla,
            dc,
            iteration_max,
            bailout,
            max_perturb_iterations,
            max_bla_steps,
        );
    };
    let ref_len = compressed.len as usize;
    let mut src = CompressedSource {
        dec: crate::fractal::perturbation::compress::ReferenceDecompressor::new(compressed),
    };
    iterate_pixel_unified_mandelbrot_impl(
        ref_orbit,
        &mut src,
        ref_len,
        bla,
        dc,
        iteration_max,
        bailout,
        max_perturb_iterations,
        max_bla_steps,
    )
}

/// Cœur générique du fast-path Mandelbrot, paramétré par la source de
/// référence (cf. [`RefF64Source`]). Avec [`SliceSource`], monomorphise en un
/// code logiquement identique à l'ancienne indexation directe (bit-identique,
/// verrouillé par les goldens). `ref_len` est passé explicitement (longueur
/// logique pour la source compressée ; `z_ref_f64` peut être vide).
#[allow(clippy::too_many_arguments)]
fn iterate_pixel_unified_mandelbrot_impl<S: RefF64Source>(
    ref_orbit: &ReferenceOrbit,
    src: &mut S,
    ref_len: usize,
    bla: &BlaTableUnified,
    dc: Complex64,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
    max_bla_steps: u32,
) -> UnifiedPixelResult {
    let bailout_sqr = bailout * bailout;
    if ref_len < 2 {
        return UnifiedPixelResult {
            iteration: 0,
            z_final: Complex64::new(0.0, 0.0),
            rebase_count: 0,
            bla_steps: 0,
            orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
        };
    }

    let mut delta = Complex64::new(0.0, 0.0);
    let mut n = 0u32;
    let mut m = 0u32;
    let mut rebase_count = 0u32;
    let mut bla_steps = 0u32;
    let mut iters_ptb = 0u32;
    let mut ref_exhausted_flag = false;

    // Cache tête-de-boucle : le bas de boucle (check de rebase) et la garde
    // anti-over-skip BLA calculent déjà `Z[m']+δ'` et sa norme pour l'état
    // suivant — les réutiliser évite un load + add + norm_sqr redondants par
    // itération (bit-identique : mêmes opérandes f64, style chaînage MipLA,
    // cf. docs/imagina-algorithms-analysis.md). Invariant : en tête de boucle,
    // `z_m == Z[m]` et `m ≤ ref_len-1`.
    let mut z_m = src.reset();
    let mut z_abs = z_m + delta;
    let mut z_abs_norm_sqr = z_abs.norm_sqr();
    let mut delta_norm_sqr = delta.norm_sqr();

    while n < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
        && (max_bla_steps == 0 || bla_steps < max_bla_steps)
    {
        // Bailout absolu : |Z[m] + δ|² ≥ bailout²
        if z_abs_norm_sqr >= bailout_sqr {
            return UnifiedPixelResult {
                iteration: n,
                z_final: z_abs,
                rebase_count,
                bla_steps,
                orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
            };
        }

        // Étape 1 : essai BLA
        if let Some(node) = bla.lookup(m as usize, delta_norm_sqr) {
            // Vérifier qu'on ne dépasse pas iteration_max ni ref_len. Réf
            // atom-tronquée : le BLA ne doit pas atterrir SUR la dernière entrée
            // (graze) — le `continue` sauterait le check end-of-ref du bas et le
            // pas direct partirait du graze, puis le rebase absorberait δ en f64
            // (image uniforme). Arrivée en fin de réf par pas direct uniquement,
            // pour rebaser AVANT le pas de graze (ordre F3 `hybrid.cc:295-308`).
            let new_n = n.saturating_add(node.l);
            let new_m = m.saturating_add(node.l);
            let lands_on_ref_end =
                ref_orbit.atom_truncated && (new_m as usize) + 1 >= ref_len;
            if new_n <= iteration_max && (new_m as usize) < ref_len && !lands_on_ref_end {
                // δ := A·δ + B·dc
                let a = node.a;
                let b = node.b;
                let (a_re, a_im) = (
                    a.m00 * delta.re + a.m01 * delta.im,
                    a.m10 * delta.re + a.m11 * delta.im,
                );
                let (b_re, b_im) = (
                    b.m00 * dc.re + b.m01 * dc.im,
                    b.m10 * dc.re + b.m11 * dc.im,
                );
                let cand = Complex64::new(a_re + b_re, a_im + b_im);
                // Garde anti-over-skip BLA : un saut multi-pas (l ≥ 2) linéarisé
                // autour de la référence peut sauter par-dessus l'évasion propre
                // du pixel et rapporter l'iter d'escape jusqu'à l-1 trop tard
                // (cf. julia-siegel). Escape irréversible (|z| > ER ≫ |c|) → le
                // seul point d'arrivée suffit ; si échappé, on rejette le saut et
                // on single-steppe pour l'iter exacte.
                // `Z[m']` = `node.z_land` (bit-copie de l'orbite, remplie au
                // build BLA) : la garde n'accède PAS la source — le saut peut
                // être REJETÉ sans toucher l'état du décompresseur.
                // (z_end calculé sans condition : c'est la valeur de cache de la
                // tête suivante ; pour l == 1 — dernier nœud d'un niveau — la
                // garde n'est juste pas appliquée, comme avant.)
                let z_new_m = node.z_land;
                let z_end = z_new_m + cand;
                let z_end_norm_sqr = z_end.norm_sqr();
                let overshoots_escape = node.l >= 2 && z_end_norm_sqr >= bailout_sqr;
                if !overshoots_escape {
                    delta = cand;
                    n = new_n;
                    m = new_m;
                    bla_steps += 1;

                    // NaN / Inf protection
                    if !delta.re.is_finite() || !delta.im.is_finite() {
                        return UnifiedPixelResult {
                            iteration: n,
                            z_final: z_end,
                            rebase_count,
                            bla_steps,
                            orbit: None,
                            distance: None,
                            is_interior: false,
                            ref_exhausted: false,
                        };
                    }
                    // Saut ACCEPTÉ : téléporter la source au point
                    // d'atterrissage (valeur exacte = z_land).
                    z_m = src.teleport(new_m, z_new_m);
                    z_abs = z_end;
                    z_abs_norm_sqr = z_end_norm_sqr;
                    delta_norm_sqr = delta.norm_sqr();
                    continue;
                }
            }
            // BLA voulait sauter trop loin → on tombe sur le pas perturbation.
        }

        // Étape 2 : pas perturbation Mandelbrot
        // δ_{n+1} = 2·Z[m]·δ + δ² + dc
        let two_zm = z_m * 2.0;
        delta = two_zm * delta + delta * delta + dc;
        n += 1;
        m += 1;
        iters_ptb += 1;

        // Avance de la source : `Z[m]` si m est dans l'orbite. `m == ref_len`
        // (possible quand un BLA a atterri sur ref_len-1) ne correspond à
        // aucune valeur → clamp historique `Z[min(m, ref_len-1)]` = end_value.
        let z_after = if (m as usize) < ref_len {
            src.advance()
        } else {
            src.end_value()
        };

        if !delta.re.is_finite() || !delta.im.is_finite() {
            return UnifiedPixelResult {
                iteration: n,
                z_final: z_after + delta,
                rebase_count,
                bla_steps,
                orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
            };
        }

        // Étape 3 : cyclage si périodique (intérieur) ; rebase-at-end F3 si réf
        // tronquée atom-domain (quasi-périodique, `δ := Z[end]+δ, m := 0`,
        // cf. `hybrid.cc:301` + diag G2) ; sinon flag exhaustion
        // (cf. e113.toml : rebase blind = uniformise tous les pixels post-escape).
        let end_of_ref = (m as usize) + 1 >= ref_len;
        if end_of_ref {
            if let Some(m_wrapped) = ref_orbit.wrap_periodic(m) {
                m = m_wrapped;
                z_m = src.wrap(m);
            } else if ref_orbit.atom_truncated {
                // `z_after` == Z[min(m, ref_len-1)] == Z[end] ici (m ≥ ref_len-1).
                delta = z_after + delta;
                m = 0;
                rebase_count += 1;
                z_m = src.reset();
            } else {
                ref_exhausted_flag = true;
                // Pour le z_final du bas : Z[min(m, ref_len-1)] = Z[end].
                z_m = z_after;
                break;
            }
            // m (et éventuellement δ) a changé : recalculer le cache de tête
            // (froid — au plus 1×/tour de référence).
            z_abs = z_m + delta;
            z_abs_norm_sqr = z_abs.norm_sqr();
            delta_norm_sqr = delta.norm_sqr();
        } else {
            let z_curr = z_after + delta;
            let z_curr_norm_sqr = z_curr.norm_sqr();
            delta_norm_sqr = delta.norm_sqr();
            if z_curr_norm_sqr < delta_norm_sqr {
                delta = z_curr;
                m = 0;
                rebase_count += 1;
                z_m = src.reset();
                z_abs = z_m + delta;
                z_abs_norm_sqr = z_abs.norm_sqr();
                // |δ_nouveau|² = |z_curr|², déjà calculé.
                delta_norm_sqr = z_curr_norm_sqr;
            } else {
                // Pas de rebase : la tête suivante lit exactement z_curr.
                z_m = z_after;
                z_abs = z_curr;
                z_abs_norm_sqr = z_curr_norm_sqr;
            }
        }
    }

    // Sortie de boucle (iteration_max / caps / exhaustion) : l'invariant de
    // tête garantit `z_m == Z[min(m, ref_len-1)]` (cf. branche exhaustion).
    UnifiedPixelResult {
        iteration: n,
        z_final: z_m + delta,
        rebase_count,
        bla_steps,
                orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: ref_exhausted_flag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::{build_bla_table_for_formula, compile_formula};
    use crate::fractal::perturbation::orbit::compute_reference_orbit;
    use crate::fractal::{default_params_for_type, AlgorithmMode, FractalType};
    use std::path::PathBuf;

    fn make_ref_orbit(
        center_x: f64,
        center_y: f64,
        zoom: f64,
        iter_max: u32,
    ) -> crate::fractal::perturbation::orbit::ReferenceOrbit {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x = center_x;
        params.center_y = center_y;
        let span_x = 4.0 / zoom;
        params.span_x = span_x;
        params.span_y = span_x * 100.0 / 160.0;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None, true)
            .expect("compute_reference_orbit failed");
        orbit
    }

    /// **G4 jalon 5 — verrou GMP des hybrides genuine en perturbation.**
    ///
    /// Le SEUL ground truth valable pour un hybride multi-phase est le GMP
    /// par-pixel qui CYCLE les phases (`GmpInterpState`) : le f64-std diverge
    /// par chaos sur bord hirsute (mesuré : 4 %→85 % de désaccord entre 300 et
    /// 2000 iters, sans qu'aucun des deux côtés soit « vrai »), et
    /// `iterate_point_mpc` est z²+c hardcodé. Ici : grille de dc autour d'un
    /// centre [M,BS] à 3e10 (structure réelle, trouvée par zoom-hunt),
    /// perturbation multi-phase (réfs par phase + tracking F3) vs itération
    /// GMP-256 par point. Avant le fix jalon 5 (désync phase/référence au
    /// rebase + réf dd z²+c), l'image était UNIFORME (100 % faux).
    /// Grille de dc → (exact, off1, worse, max_diff) : perturbation bytecode
    /// vs GMP par-point cyclant les phases (`GmpInterpState`, même interpréteur
    /// que l'orbite référence). Instrument des verrous hybrides jalon 5.
    fn grid_vs_gmp_cycling(
        label: &str,
        phases: Option<Vec<FractalType>>,
        cx: &str,
        cy: &str,
        zoom: f64,
        iter_max: u32,
    ) -> (u32, u32, u32, u32, i64) {
        use crate::fractal::bytecode::formula_for_params;
        use rug::{Assign, Complex as GmpComplex};

        let width = 160u32;
        let height = 100u32;
        let span_x = 4.0 / zoom;
        let span_y = span_x * height as f64 / width as f64;

        let mut params = default_params_for_type(FractalType::Mandelbrot, width, height);
        params.center_x = cx.parse().unwrap();
        params.center_y = cy.parse().unwrap();
        params.center_x_hp = Some(cx.into());
        params.center_y_hp = Some(cy.into());
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        params.hybrid_phases = phases;

        let (orbit, _, _) = compute_reference_orbit(&params, None, false)
            .expect("compute_reference_orbit failed");
        let formula = formula_for_params(&params).expect("formula");
        assert_eq!(
            orbit.hybrid_phase_refs.len() + 1,
            formula.phases.len().max(1),
            "réfs par phase manquantes"
        );
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table build");
        let bla = &tables[0];

        let prec = crate::fractal::perturbation::compute_perturbation_precision_bits(&params);
        let bailout_sqr = params.bailout * params.bailout;

        let (mut total, mut exact, mut off_by_1, mut worse) = (0u32, 0u32, 0u32, 0u32);
        let mut max_diff = 0i64;
        for gj in 0..10 {
            for gi in 0..16 {
                let dx = ((gi as f64 + 0.5) / 16.0 - 0.5) * span_x;
                let dy = ((gj as f64 + 0.5) / 10.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);

                let res = iterate_pixel_unified_full(
                    &orbit,
                    bla,
                    &formula,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    params.bailout,
                    UnifiedOptions::default(),
                );

                let mut c_px = GmpComplex::with_val(prec, (dx, dy));
                c_px += &orbit.cref_gmp;
                let mut st = crate::fractal::bytecode::GmpInterpState::new(
                    prec,
                    GmpComplex::with_val(prec, (0.0, 0.0)),
                );
                let mut n_gmp = iter_max;
                let mut n = 0u32;
                let mut norm = rug::Float::with_val(prec, 0);
                while n < iter_max {
                    // |z|² >= bailout² → escape à n (convention tête-de-boucle
                    // du pixel loop).
                    norm.assign(st.z.real() * st.z.real());
                    norm += GmpComplex::with_val(prec, (st.z.imag() * st.z.imag(), 0)).real();
                    if norm.to_f64() >= bailout_sqr {
                        n_gmp = n;
                        break;
                    }
                    st.step(&formula, &c_px);
                    n += 1;
                }

                let diff = (res.iteration as i64 - n_gmp as i64).abs();
                total += 1;
                if diff == 0 {
                    exact += 1;
                } else if diff == 1 {
                    off_by_1 += 1;
                } else {
                    worse += 1;
                }
                max_diff = max_diff.max(diff);
            }
        }
        eprintln!(
            "[GRID-GMP {label}] total={total} exact={exact} off1={off_by_1} worse={worse} max_diff={max_diff}"
        );
        (total, exact, off_by_1, worse, max_diff)
    }

    /// **G4 jalon 5 — verrou GMP des hybrides genuine en perturbation.**
    ///
    /// Le SEUL ground truth valable pour un hybride multi-phase est le GMP
    /// par-pixel qui CYCLE les phases (le f64-std diverge par chaos sur bord
    /// hirsute : 4 %→85 % de désaccord entre 300 et 2000 iters, et
    /// `iterate_point_mpc` est z²+c hardcodé). Contrôles imprimés : [M,M] à un
    /// bord M (mécanique multi-phase, dynamique douce) et [M,BS] (hérite la
    /// sensibilité hirsute BS, cf. G3 : presets BS-famille sur frontières
    /// LISSES uniquement). Avant le fix jalon 5 (désync phase/réf au rebase +
    /// réf dd z²+c), [M,BS] rendait UNIFORME (0 % de structure).
    #[test]
    fn multi_phase_perturbation_matches_gmp_per_pixel() {
        // Contrôle : [M,M] au bord seahorse — multi-phase mécanique, chaos M.
        let (t_mm, e_mm, o_mm, _w_mm, _m_mm) = grid_vs_gmp_cycling(
            "MM@3e10",
            Some(vec![FractalType::Mandelbrot, FractalType::Mandelbrot]),
            "-0.743643887037158704752191506114774",
            "0.131825904205311970493132056385139",
            3e10,
            800,
        );
        // Hybride genuine : [M,BS] (bord hirsute BS-famille).
        let (t_bs, e_bs, o_bs, _w_bs, _m_bs) = grid_vs_gmp_cycling(
            "MBS@3e10",
            Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]),
            "-0.4744047619051931",
            "-0.6327380952385265",
            3e10,
            800,
        );
        // [M,M] : dynamique M douce → quasi-exactitude exigée.
        assert!(
            (e_mm + o_mm) as f64 >= t_mm as f64 * 0.97,
            "[M,M] vs GMP cyclant : exact+off1={}/{}",
            e_mm + o_mm,
            t_mm
        );
        // [M,BS] : bord hirsute — la MAJORITÉ doit être exacte (avant fix :
        // 0 % de structure, uniforme). Le résidu est le plancher chaos f64
        // (même classe que les frontières hirsutes single-phase, cf. G3).
        assert!(
            e_bs as f64 >= t_bs as f64 * 0.50,
            "[M,BS] vs GMP cyclant : exact={}/{} off1={}",
            e_bs,
            t_bs,
            o_bs
        );
    }

    /// Diagnostic (--ignored) : stabilité du ground truth GMP sur la grille
    /// [M,BS] — itère chaque point à 256 ET 512 bits et compte les désaccords
    /// ENTRE LES DEUX VÉRITÉS. Des points truth-instables = scène au-delà de la
    /// sensibilité de précision (classe « hirsute » G3) : le résidu du verrou
    /// principal sur ces points n'est PAS attribuable au pixel loop.
    #[test]
    #[ignore]
    fn multi_phase_truth_stability_diagnostic() {
        use crate::fractal::bytecode::formula_for_params;
        use rug::{Assign, Complex as GmpComplex};

        let iter_max = 800u32;
        let zoom = 3e10;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;
        let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
        params.center_x_hp = Some("-0.4744047619051931".into());
        params.center_y_hp = Some("-0.6327380952385265".into());
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
        let formula = formula_for_params(&params).expect("formula");
        let bailout_sqr = params.bailout * params.bailout;

        let truth = |prec: u32, dx: f64, dy: f64| -> u32 {
            let cx = rug::Float::parse(params.center_x_hp.as_deref().unwrap()).unwrap();
            let cy = rug::Float::parse(params.center_y_hp.as_deref().unwrap()).unwrap();
            let mut c_px = GmpComplex::with_val(prec, (dx, dy));
            c_px += GmpComplex::with_val(
                prec,
                (rug::Float::with_val(prec, cx), rug::Float::with_val(prec, cy)),
            );
            let mut st = crate::fractal::bytecode::GmpInterpState::new(
                prec,
                GmpComplex::with_val(prec, (0.0, 0.0)),
            );
            let mut n = 0u32;
            let mut norm = rug::Float::with_val(prec, 0);
            while n < iter_max {
                norm.assign(st.z.real() * st.z.real());
                norm += GmpComplex::with_val(prec, (st.z.imag() * st.z.imag(), 0)).real();
                if norm.to_f64() >= bailout_sqr {
                    return n;
                }
                st.step(&formula, &c_px);
                n += 1;
            }
            iter_max
        };

        let (mut unstable, mut total) = (0u32, 0u32);
        for gj in 0..10 {
            for gi in 0..16 {
                let dx = ((gi as f64 + 0.5) / 16.0 - 0.5) * span_x;
                let dy = ((gj as f64 + 0.5) / 10.0 - 0.5) * span_y;
                let a = truth(256, dx, dy);
                let b = truth(512, dx, dy);
                total += 1;
                if a != b {
                    unstable += 1;
                }
            }
        }
        eprintln!("[TRUTH-STAB MBS@3e10] truth256≠truth512 : {unstable}/{total}");
    }

    /// Diagnostic (--ignored) : attribution du résidu [M,BS] — bruit inhérent
    /// vs bug systématique. Rend la MÊME grille absolue avec deux références
    /// différentes (centre décalé de ~span/3). Points dont la réponse CHANGE
    /// avec la référence = dominés par le bruit f64 du delta (chaque réf = une
    /// réalisation de bruit) ; stables mais ≠ vérité = systématique.
    #[test]
    #[ignore]
    fn multi_phase_reference_sensitivity_diagnostic() {
        use crate::fractal::bytecode::formula_for_params;
        use rug::{Assign, Complex as GmpComplex, Float};

        let iter_max = 800u32;
        let zoom = 3e10;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;
        let cx_hp = "-0.4744047619051931";
        let cy_hp = "-0.6327380952385265";

        let build = |cx_hp: &str, cy_hp: &str| {
            let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
            params.center_x = cx_hp.parse().unwrap();
            params.center_y = cy_hp.parse().unwrap();
            params.center_x_hp = Some(cx_hp.into());
            params.center_y_hp = Some(cy_hp.into());
            params.span_x = span_x;
            params.span_y = span_y;
            params.iteration_max = iter_max;
            params.algorithm_mode = AlgorithmMode::Perturbation;
            params.hybrid_phases =
                Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
            let (orbit, _, _) =
                compute_reference_orbit(&params, None, false).expect("orbit");
            (params, orbit)
        };

        // Réf 1 : centre nominal. Réf 2 : centre décalé (span/3, span/7).
        let shift_x = span_x / 3.0;
        let shift_y = span_y / 7.0;
        let prec = 256u32;
        let cx2 = {
            let v = Float::with_val(prec, Float::parse(cx_hp).unwrap()) + shift_x;
            v.to_string_radix(10, None)
        };
        let cy2 = {
            let v = Float::with_val(prec, Float::parse(cy_hp).unwrap()) + shift_y;
            v.to_string_radix(10, None)
        };
        let (params1, orbit1) = build(cx_hp, cy_hp);
        let (_params2, orbit2) = build(&cx2, &cy2);
        let formula = formula_for_params(&params1).expect("formula");
        let c_norm =
            (orbit1.cref.re * orbit1.cref.re + orbit1.cref.im * orbit1.cref.im).sqrt();
        let t1 = build_bla_table_for_formula(&formula, &orbit1.z_ref_f64, c_norm, 6e-8)
            .expect("bla1");
        let t2 = build_bla_table_for_formula(&formula, &orbit2.z_ref_f64, c_norm, 6e-8)
            .expect("bla2");

        let bailout_sqr = params1.bailout * params1.bailout;
        let truth = |dx: f64, dy: f64| -> u32 {
            let mut c_px = GmpComplex::with_val(prec, (dx, dy));
            c_px += &orbit1.cref_gmp;
            let mut st = crate::fractal::bytecode::GmpInterpState::new(
                prec,
                GmpComplex::with_val(prec, (0.0, 0.0)),
            );
            let mut n = 0u32;
            let mut norm = Float::with_val(prec, 0);
            while n < iter_max {
                norm.assign(st.z.real() * st.z.real());
                norm += GmpComplex::with_val(prec, (st.z.imag() * st.z.imag(), 0)).real();
                if norm.to_f64() >= bailout_sqr {
                    return n;
                }
                st.step(&formula, &c_px);
                n += 1;
            }
            iter_max
        };

        let (mut both_ok, mut noise, mut systematic, mut total_bad) = (0u32, 0u32, 0u32, 0u32);
        for gj in 0..10 {
            for gi in 0..16 {
                let dx = ((gi as f64 + 0.5) / 16.0 - 0.5) * span_x;
                let dy = ((gj as f64 + 0.5) / 10.0 - 0.5) * span_y;
                let run = |orbit: &crate::fractal::perturbation::orbit::ReferenceOrbit,
                           bla: &BlaTableUnified,
                           dcx: f64,
                           dcy: f64| {
                    iterate_pixel_unified_full(
                        orbit,
                        bla,
                        &formula,
                        orbit.cref,
                        Complex64::new(dcx, dcy),
                        Complex64::new(0.0, 0.0),
                        iter_max,
                        params1.bailout,
                        UnifiedOptions::default(),
                    )
                    .iteration
                };
                let n1 = run(&orbit1, &t1[0], dx, dy);
                let n2 = run(&orbit2, &t2[0], dx - shift_x, dy - shift_y);
                let nt = truth(dx, dy);
                if n1 == nt && n2 == nt {
                    both_ok += 1;
                } else {
                    total_bad += 1;
                    if n1 != n2 {
                        noise += 1; // réponse dépend de la réf → plancher bruit
                    } else {
                        systematic += 1; // stable mais faux → suspect
                    }
                }
            }
        }
        eprintln!(
            "[REF-SENS MBS@3e10] both_exact={both_ok} bad={total_bad} (noise={noise} systematic={systematic})"
        );
    }

    /// Render une image complète Mandelbrot zoom 1e6 via le pixel loop
    /// unifié, sauvegarde-la, et diffe vs la golden legacy
    /// `tests/golden/mandelbrot_perturb_1e6.png`.
    ///
    /// Cette comparaison est *le* test de Session C : si le diff est faible
    /// (<1% pixels), l'unified path produit une image visuellement
    /// équivalente à legacy. Si gros, on a quantifié le travail restant.
    ///
    /// Le test ne fait PAS échouer le build sur diff > 0 — il rapporte.
    #[test]
    fn render_full_image_vs_legacy_golden() {
        use crate::color::palettes::{color_for_pixel_with_lut, PaletteLut};
        use crate::fractal::OutColoringMode;

        let width = 160u32;
        let height = 100u32;
        let iter_max = 1500u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * height as f64 / width as f64;

        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table build");
        let bla = &tables[0];

        // Render
        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        let palette = PaletteLut::new(
            6, // Plasma
            crate::fractal::ColorSpace::Rgb,
        );

        for j in 0..height {
            for i in 0..width {
                let dx = ((i as f64 + 0.5) / width as f64 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / height as f64 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);

                let res = iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 4.0, 0, 0);

                // Coloriage via la fonction de production (mêmes constantes
                // que default_params_for_type pour Mandelbrot : palette 6 (Plasma),
                // color_repeat 40, OutColoringMode::Smooth, ColorSpace::Rgb).
                let (r, g, b) = color_for_pixel_with_lut(
                    res.iteration,
                    res.z_final,
                    iter_max,
                    6,  // palette index Plasma
                    40, // color_repeat
                    OutColoringMode::Smooth,
                    crate::fractal::ColorSpace::Rgb,
                    None, // orbit
                    None, // distance
                    false, // interior_flag_encoded
                    Some(&palette),
                );
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
        }

        // Save the unified-render output
        let unified_path = std::env::temp_dir().join("mandelbrot_perturb_1e6_unified.png");
        let buf = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, pixels.clone())
            .expect("buffer build");
        buf.save(&unified_path).expect("save unified png");

        // Diff vs legacy golden
        let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden/mandelbrot_perturb_1e6.png");
        let golden = image::open(&golden_path)
            .expect("open golden")
            .to_rgba8()
            .into_raw();

        assert_eq!(
            golden.len(),
            pixels.len(),
            "Tailles différentes : golden {} vs unified {}",
            golden.len(),
            pixels.len()
        );

        // Stats supplémentaires : combien de pixels utilisent la BLA ?
        let mut total_bla_steps = 0u64;
        let mut total_rebase = 0u64;
        let mut total_iter = 0u64;
        for j in 0..height {
            for i in 0..width {
                let dx = ((i as f64 + 0.5) / width as f64 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / height as f64 - 0.5) * span_y;
                let res = iterate_pixel_unified_mandelbrot(
                    &orbit,
                    bla,
                    Complex64::new(dx, dy),
                    iter_max,
                    4.0,
                    0,
                    0,
                );
                total_bla_steps += res.bla_steps as u64;
                total_rebase += res.rebase_count as u64;
                total_iter += res.iteration as u64;
            }
        }

        let mut diffs = 0usize;
        let mut total_abs_diff = 0u64;
        let mut heavy_diff = 0usize;
        for (g, u) in golden.chunks_exact(4).zip(pixels.chunks_exact(4)) {
            if g != u {
                diffs += 1;
            }
            let mut max_chan = 0u8;
            for k in 0..3 {
                let d = (g[k] as i64 - u[k] as i64).unsigned_abs() as u8;
                total_abs_diff += d as u64;
                if d > max_chan {
                    max_chan = d;
                }
            }
            if max_chan > 50 {
                heavy_diff += 1;
            }
        }
        let total_pixels = (width * height) as usize;
        let mean_abs_diff = total_abs_diff as f64 / (total_pixels as f64 * 3.0);
        eprintln!(
            "Session C unified vs legacy golden (mandelbrot zoom 1e6) :\n  \
             {}/{} pixels diffèrent ({:.2}%), mean |diff|/canal = {:.2}/255\n  \
             pixels lourdement diff (>50/canal) : {}/{} ({:.2}%)\n  \
             stats unified : BLA steps total {}, rebases {}, iter sum {}\n  \
             Output: {}",
            diffs,
            total_pixels,
            100.0 * diffs as f64 / total_pixels as f64,
            mean_abs_diff,
            heavy_diff,
            total_pixels,
            100.0 * heavy_diff as f64 / total_pixels as f64,
            total_bla_steps,
            total_rebase,
            total_iter,
            unified_path.display()
        );
    }

    /// P1.3 — caps `max_perturb_iterations` / `max_bla_steps`. Quand un cap
    /// est atteint, la boucle sort avec `iteration < iteration_max` (le
    /// smooth coloring du caller le rend "échappé tard" plutôt qu'intérieur).
    /// Pour un pixel intérieur (centre de la cardioïde), avec caps=0 la boucle
    /// va jusqu'à iter_max ; avec un cap petit, elle s'arrête au cap.
    #[test]
    fn caps_max_perturb_iterations_truncates_interior_pixel() {
        let iter_max = 2000u32;
        let cx = 0.0;
        let cy = 0.0;
        let zoom = 1.0;
        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table build");
        let bla = &tables[0];
        // Pixel intérieur (cardioïde principale) — sans cap, devrait atteindre iter_max.
        let dc = Complex64::new(0.0, 0.0);
        let res_uncapped = iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 4.0, 0, 0);
        assert_eq!(
            res_uncapped.iteration, iter_max,
            "expected interior pixel to reach iter_max without cap"
        );
        // Avec cap=200 raw perturbations, la boucle sort dès qu'on en a fait
        // 200 (les pas BLA ne comptent pas, mais à zoom 1.0 et centre 0 la BLA
        // ne sera pas particulièrement active non plus).
        let cap = 200u32;
        let res_capped = iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 4.0, cap, 0);
        assert!(
            res_capped.iteration < iter_max,
            "expected cap to truncate before iter_max, got {} >= {}",
            res_capped.iteration,
            iter_max
        );
    }

    /// Test le pixel loop unifié généralisé sur Burning Ship.
    /// Le delta-form gère les ops AbsX/AbsY via diffabs.
    #[test]
    fn unified_pixel_loop_burning_ship_zoom_100() {
        let iter_max = 800u32;
        let cx = -1.762f64;
        let cy = -0.028f64;
        let zoom = 100.0f64;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        // ReferenceOrbit pour BurningShip
        let mut params = default_params_for_type(FractalType::BurningShip, 160, 100);
        params.center_x = cx;
        params.center_y = cy;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None, true).expect("ref orbit");

        let formula = compile_formula(FractalType::BurningShip, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table");
        let bla = &tables[0];

        // Itération directe BurningShip pour ref classification.
        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);
                let c_abs = Complex64::new(cx + dx, cy + dy);

                let res = iterate_pixel_unified(
                    &orbit,
                    bla,
                    &formula,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    4.0,
                );
                let escaped_unif = res.iteration < iter_max;

                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    let z_abs = Complex64::new(z.re.abs(), z.im.abs());
                    z = z_abs * z_abs + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;
                if escaped_unif != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "BS ({},{}) unif n={} esc={} | direct n={} esc={}",
                            i, j, res.iteration, escaped_unif, nn, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Unified loop BurningShip zoom 100 : {}/{} classifications matchent",
            total - mismatches,
            total
        );
        // BurningShip est non-conformal et hautement chaotique : tolérer
        // 1-2 pixels frontière sur 100 (cas typique : pixel à l'extrême
        // limite d'un minibrot).
        let mismatch_pct = 100.0 * mismatches as f64 / total as f64;
        assert!(
            mismatch_pct <= 5.0,
            "Trop de mismatches BS unified vs direct : {:.1}%",
            mismatch_pct
        );
    }

    /// Idem sur Tricorn.
    #[test]
    fn unified_pixel_loop_tricorn_zoom_50() {
        let iter_max = 500u32;
        let cx = -0.5f64;
        let cy = 0.1f64;
        let zoom = 50.0f64;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        let mut params = default_params_for_type(FractalType::Tricorn, 160, 100);
        params.center_x = cx;
        params.center_y = cy;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) = compute_reference_orbit(&params, None, true).expect("ref orbit");

        let formula = compile_formula(FractalType::Tricorn, 2.0).unwrap();
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table");
        let bla = &tables[0];

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);
                let c_abs = Complex64::new(cx + dx, cy + dy);

                let res = iterate_pixel_unified(
                    &orbit,
                    bla,
                    &formula,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    4.0,
                );
                let escaped_unif = res.iteration < iter_max;

                // Tricorn direct : (X - iY)² + C
                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    let conj = Complex64::new(z.re, -z.im);
                    z = conj * conj + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;
                if escaped_unif != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "Tricorn ({},{}) unif n={} esc={} | direct n={} esc={}",
                            i, j, res.iteration, escaped_unif, nn, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Unified loop Tricorn zoom 50 : {}/{} classifications matchent",
            total - mismatches,
            total
        );
        assert_eq!(mismatches, 0);
    }

    /// Sanity-check multi-phase : Formula::hybrid(vec![mandel_phase,
    /// mandel_phase]) = formula avec 2 phases identiques Mandelbrot. Le
    /// pixel loop doit produire le même résultat que la formule mono-phase
    /// Mandelbrot (modulo le path multi-phase qui n'utilise pas la BLA).
    /// Jalon 5 : l'orbite est bâtie avec `hybrid_phases` — le contrat du
    /// loop multi-phase exige les références PAR PHASE (`hybrid_phase_refs`).
    #[test]
    fn unified_multi_phase_identical_phases_eq_mono() {
        let iter_max = 300u32;
        let cx = -0.5f64;
        let cy = 0.0f64;
        let zoom = 1.0f64;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        let orbit = {
            let mut params = default_params_for_type(FractalType::Mandelbrot, 160, 100);
            params.center_x = cx;
            params.center_y = cy;
            params.span_x = span_x;
            params.span_y = span_y;
            params.iteration_max = iter_max;
            params.algorithm_mode = AlgorithmMode::Perturbation;
            params.hybrid_phases =
                Some(vec![FractalType::Mandelbrot, FractalType::Mandelbrot]);
            let (orbit, _, _) = compute_reference_orbit(&params, None, true)
                .expect("compute_reference_orbit failed");
            orbit
        };
        let mono = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let hybrid = Formula::hybrid(vec![mono.phases[0].clone(), mono.phases[0].clone()]);

        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&mono, &orbit.z_ref_f64, c_norm, 6e-8).unwrap();
        let bla = &tables[0];

        // Mono-phase et hybrid devraient donner la même classification.
        // Iter counts peuvent différer parce que multi-phase ne fait pas BLA.
        let mut classif_match = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(20) {
            for j in (0..100).step_by(20) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);

                let r_mono = iterate_pixel_unified(
                    &orbit,
                    bla,
                    &mono,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    4.0,
                );
                let r_hybrid = iterate_pixel_unified(
                    &orbit,
                    bla,
                    &hybrid,
                    orbit.cref,
                    dc,
                    Complex64::new(0.0, 0.0),
                    iter_max,
                    4.0,
                );
                let esc_mono = r_mono.iteration < iter_max;
                let esc_hybrid = r_hybrid.iteration < iter_max;
                if esc_mono == esc_hybrid {
                    classif_match += 1;
                }
                total += 1;
            }
        }
        assert_eq!(
            classif_match, total,
            "Hybrid avec 2 phases identiques doit classifier comme mono"
        );
    }

    /// À zoom 1e6, la classification doit matcher l'itération directe.
    #[test]
    fn unified_pixel_loop_mandelbrot_zoom_1e6() {
        let iter_max = 1500u32;
        let cx = -1.7693831791955;
        let cy = 0.004236847918736;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * 100.0 / 160.0;

        let orbit = make_ref_orbit(cx, cy, zoom, iter_max);
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        // c_norm pour la formule de merge BLA = |cref|
        let c_norm = (orbit.cref.re * orbit.cref.re + orbit.cref.im * orbit.cref.im).sqrt();
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table build");
        let bla = &tables[0];

        let mut mismatches = 0usize;
        let mut total = 0usize;
        for i in (0..160).step_by(16) {
            for j in (0..100).step_by(10) {
                let dx = ((i as f64 + 0.5) / 160.0 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / 100.0 - 0.5) * span_y;
                let dc = Complex64::new(dx, dy);
                let c_abs = Complex64::new(cx + dx, cy + dy);

                let res = iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 4.0, 0, 0);
                let escaped_unif = res.iteration < iter_max;

                let mut z = Complex64::new(0.0, 0.0);
                let mut nn = 0u32;
                while nn < iter_max && z.norm_sqr() < 16.0 {
                    z = z * z + c_abs;
                    nn += 1;
                }
                let escaped_direct = nn < iter_max;

                if escaped_unif != escaped_direct {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "  ({},{}) unif n={} esc={} bla_steps={} rebase={} | direct n={} esc={}",
                            i, j, res.iteration, escaped_unif, res.bla_steps, res.rebase_count, nn, escaped_direct
                        );
                    }
                }
                total += 1;
            }
        }
        eprintln!(
            "Unified loop zoom 1e6 : {}/{} classifications matchent",
            total - mismatches,
            total
        );
        assert_eq!(
            mismatches, 0,
            "Unified pixel loop doit classifier comme l'itération directe"
        );
    }

    /// Verrou anti-régression du guard BLA over-skip (Julia siegel disk).
    ///
    /// Scène : Julia c=-0.8+0.156i, centre (0,0), zoom 1e10 — l'orbite de
    /// référence (critique) est bornée, mais les pixels s'échappent vers
    /// l'iter ~254. Un pas BLA multi-étapes linéarisé autour de la référence
    /// bornée peut sauter PAR-DESSUS l'évasion propre du pixel et rapporter
    /// l'iter d'escape jusqu'à `l-1` trop tard (bug : tous les pixels
    /// escapaient uniformément +2 vs le GMP/f64 pur — `fractall-quality`
    /// preset `julia-siegel-disk` FAIL max_diff=2).
    ///
    /// Le guard rejette ces sauts et single-steppe → l'escape perturbation+BLA
    /// doit matcher l'itération f64 directe (même précision, path indépendant
    /// de la BLA). On vérifie que le biais moyen d'iter est ~0, pas +2.
    #[test]
    fn bla_no_overskip_past_escape_julia_siegel() {
        use crate::fractal::perturbation::{effective_pixel_size, orbit::compute_reference_orbit};

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
            compute_reference_orbit(&params, None, true).expect("compute_reference_orbit (Julia)");
        let formula = compile_formula(FractalType::Julia, 2.0).unwrap();
        let c_norm = effective_pixel_size(&params)
            * ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
        let tables =
            build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, params.bla_threshold)
                .expect("BLA table (Julia)");
        let bla = &tables[0];

        let options = UnifiedOptions {
            is_julia: true,
            ..Default::default()
        };
        let seed = Complex64::new(sx, sy);

        // Escape f64 direct (référence : z_{n+1} = z_n² + seed, z_0 = pixel).
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
        let mut worst = 0u32;
        for j in 0..height {
            for i in 0..width {
                let dx = ((i as f64 + 0.5) / width as f64 - 0.5) * span_x;
                let dy = ((j as f64 + 0.5) / height as f64 - 0.5) * span_y;
                // Julia : delta_init = pixel − centre (= pixel, centre=0), c_ref = seed, dc = 0.
                let delta_init = Complex64::new(dx, dy);
                let res = iterate_pixel_unified_full(
                    &orbit,
                    bla,
                    &formula,
                    seed,
                    Complex64::new(0.0, 0.0),
                    delta_init,
                    iter_max,
                    bailout,
                    options,
                );
                if let Some(plain) = plain_escape(dx, dy) {
                    if res.iteration < iter_max {
                        let d = res.iteration as i64 - plain as i64;
                        sum_signed += d;
                        sum_abs += d.abs();
                        worst = worst.max(d.unsigned_abs() as u32);
                        escaped += 1;
                    }
                }
            }
        }

        assert!(escaped > 500, "trop peu de pixels échappés ({escaped})");
        let mean_signed = sum_signed as f64 / escaped as f64;
        let mean_abs = sum_abs as f64 / escaped as f64;
        eprintln!(
            "[julia-siegel guard] escaped={escaped} mean_signed={mean_signed:.4} mean_abs={mean_abs:.4} worst={worst}"
        );
        // Avant le guard : tous les pixels escapaient uniformément +2 → mean≈2.
        // Après : le path BLA suit le f64 direct au pixel près (biais ~0).
        assert!(
            mean_signed.abs() < 0.5,
            "biais d'iter d'escape BLA vs f64 direct = {mean_signed:.4} (le guard anti-over-skip a régressé ?)"
        );
        assert!(
            mean_abs < 0.5,
            "erreur moyenne d'iter BLA vs f64 direct = {mean_abs:.4} (over-skip BLA)"
        );
    }
}
