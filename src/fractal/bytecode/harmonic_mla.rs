//! Prototype **Harmonic MLA** (Imagina / Zhuoran Yu, variante « Mag » de la
//! Linear Approximation harmonique) — port fidèle de
//! `Imagina-Algorithms/HarmonicMLA.{h,cpp}` (333 lignes). Design commenté :
//! `docs/imagina-algorithms-analysis.md` §HarmonicMLA. TODO G8.2 « Prototype
//! Harmonic LA », env-gated `FRACTALL_HARMONIC_LA=1`.
//!
//! Idée : remplacer la table BLA mip 2^k par des **segments LA à longueur
//! variable** coupés par magnitude (période détectée quand le min courant de
//! cheb|Z| chute sous `prevMin·2⁻⁴`), organisés en **étages**
//! période/super-période, évalués par **descente d'étage** : ~1 check de
//! validité par segment accepté au lieu d'un lookup multi-niveaux par
//! position. Mandelbrot f64 conforme uniquement (A/B complexes, PAS Mat2).
//! Le premier pas d'un segment est **quadratique EXACT** (`δ' = δ·(Z+z)` =
//! `δ·(2Z+δ)`) : le seul terme non-linéaire de la perturbation est traité
//! exactement là où δ est le plus gros relatif → rayons de validité plus
//! larges que la BLA linéarisée.
//!
//! Écarts VOLONTAIRES vs Imagina (nos garanties de qualité, cf. spec G8.2) :
//! 1. **Cap itérations** : un segment n'est appliqué que si
//!    `i + length ≤ iteration_max` (sinon descente d'étage) — l'itération
//!    rapportée ne dépasse jamais `iteration_max`.
//! 2. **Garde anti-over-skip à l'atterrissage** : si `|z|² ≥ bailout²` après
//!    application, le segment a sauté PAR-DESSUS l'escape → rejet + descente
//!    (même philosophie que `overshoots_escape` de `pixel_loop.rs`). Bailout =
//!    params (625 par défaut), PAS leur 4096.
//! 3. **Queue directe** = sémantique de `iterate_pixel_unified_mandelbrot_impl`
//!    SANS BLA : escape à `bailout_sqr`, rebase F3 strict `|Z+δ|² < |δ|²`,
//!    rebase-at-end si `atom_truncated`, sinon flag `ref_exhausted` (routé
//!    `glitched` comme aujourd'hui).
//!
//! Routage : `delta.rs::harmonic_route_active` (Mandelbrot seed=0, tier wisdom
//! F64, `cycle_period == 0`, pas de dual-numbers features, incompatible avec
//! `FRACTALL_COMPRESS_REF` — la table et la queue directe lisent `z_ref_f64`
//! PLEIN). Path par défaut (gate OFF) bit-identique : ce module n'est jamais
//! touché.

use std::sync::OnceLock;

use num_complex::Complex64;

use super::pixel_loop::UnifiedPixelResult;
use crate::fractal::perturbation::compress::chebyshev_norm;
use crate::fractal::perturbation::orbit::ReferenceOrbit;

/// `LAStep::ValidRadiusScale` (HarmonicMLA.h:17) : marge de validité 2⁻²⁴.
const VALID_RADIUS_SCALE: f64 = 1.0 / 16_777_216.0; // 2^-24, exact

/// `PeriodDetectionThreshold` (HarmonicMLA.h:83) : période détectée quand le
/// min courant de cheb|Z| chute sous `prevMin · 2⁻⁴`.
const PERIOD_DETECTION_THRESHOLD: f64 = 0.0625; // 2^-4, exact

/// Garde-fou fractall (absent d'Imagina) : plafond d'étages. La construction
/// s'arrête aussi quand un étage ne RÉDUIT plus le nombre de segments (un
/// étage identique au précédent redétecterait la même période → boucle
/// infinie théorique sur orbites pathologiques).
const MAX_STAGES: usize = 64;

/// Mirror exact de `LAStep` (HarmonicMLA.h:13-69). Conforme Mandelbrot :
/// `A = ∂δ_out/∂δ'`, `B = ∂δ_out/∂dc` complexes (pas Mat2), où `δ'` est le
/// premier pas quadratique exact du segment.
#[derive(Clone, Copy, Debug)]
pub struct LaStep {
    /// Z de DÉPART du segment (valeur de l'orbite brute à son début).
    pub z: Complex64,
    pub a: Complex64,
    pub b: Complex64,
    /// Rayon de validité sur `|δ'|` (chebyshev).
    pub valid_radius: f64,
    /// Rayon de validité sur `|dc|` (chebyshev).
    pub valid_radius_c: f64,
    /// Nombre d'itérations couvertes par le segment.
    pub length: u32,
    /// Étage 0 : index dans l'ORBITE BRUTE du début du segment.
    /// Étages > 0 : index dans le Vec global `steps` du segment de l'étage
    /// précédent où descendre.
    pub next_stage_la_index: u32,
}

impl LaStep {
    /// `LAStep(size_t i, complex z)` (HarmonicMLA.h:26-27) : segment d'un pas.
    fn new(i: u32, z: Complex64) -> Self {
        LaStep {
            z,
            a: Complex64::new(1.0, 0.0),
            b: Complex64::new(1.0, 0.0),
            valid_radius: 1.0,
            valid_radius_c: 1.0,
            length: 1,
            next_stage_la_index: i,
        }
    }

    /// `Step(complex z)` (HarmonicMLA.h:32-46) : extension du segment d'un pas
    /// (z = valeur de l'orbite à la fin actuelle du segment).
    fn step(&self, z: Complex64) -> LaStep {
        let radius = chebyshev_norm(z) * VALID_RADIUS_SCALE;
        LaStep {
            valid_radius: self.valid_radius.min(radius / chebyshev_norm(self.a)),
            valid_radius_c: self.valid_radius_c.min(radius / chebyshev_norm(self.b)),
            a: 2.0 * z * self.a,
            b: 2.0 * z * self.b + Complex64::new(1.0, 0.0),
            z: self.z,
            length: self.length + 1,
            next_stage_la_index: self.next_stage_la_index,
        }
    }

    /// `Composite(const LAStep &step)` (HarmonicMLA.h:48-68) : composition
    /// `self` puis `step`. Récurrences EXACTES du .h — le rayon intermédiaire
    /// utilise `A/B` AVANT multiplication par `step.a` (et `step.valid_radius`
    /// pour les DEUX rayons, fidèle à la source).
    fn composite(&self, step: &LaStep) -> LaStep {
        let radius = chebyshev_norm(step.z) * VALID_RADIUS_SCALE;

        let mut valid_radius = self.valid_radius.min(radius / chebyshev_norm(self.a));
        let mut valid_radius_c = self.valid_radius_c.min(radius / chebyshev_norm(self.b));
        let a = 2.0 * step.z * self.a;
        let b = 2.0 * step.z * self.b;

        valid_radius = valid_radius.min(step.valid_radius / chebyshev_norm(a));
        valid_radius_c = valid_radius_c.min(step.valid_radius / chebyshev_norm(b));
        LaStep {
            a: a * step.a,
            b: b * step.a + step.b,
            z: self.z,
            valid_radius,
            valid_radius_c,
            length: self.length + step.length,
            next_stage_la_index: self.next_stage_la_index,
        }
    }
}

/// `LAStageInfo` (HarmonicMLA.h:78-81) : bornes `[begin, end)` des SEGMENTS
/// d'un étage dans `steps` — SANS la sentinelle (poussée à l'index `end`).
#[derive(Clone, Copy, Debug)]
pub struct StageInfo {
    pub begin: u32,
    pub end: u32,
}

/// Table Harmonic MLA pour une orbite référence f64.
pub struct HarmonicMlaTable {
    pub stages: Vec<StageInfo>,
    pub steps: Vec<LaStep>,
    /// Période détectée à l'étage 0 (0 si aucune — table à 1 segment).
    pub period0: u32,
}

/// Gate env `FRACTALL_HARMONIC_LA=1`.
pub fn harmonic_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("FRACTALL_HARMONIC_LA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Construit la table depuis l'orbite brute f64. Mirror de `Prepare`
/// (HarmonicMLA.cpp:10-21) : `None` si l'orbite est trop courte
/// (`referenceLength < 8`, avec `referenceLength = len-1` — leur
/// `reference[i]` = notre `z_ref_f64[i]`, `reference[0] = 0 = z_ref_f64[0]`).
pub fn build_harmonic_mla_table(z_ref_f64: &[Complex64]) -> Option<HarmonicMlaTable> {
    if z_ref_f64.len() < 9 {
        return None;
    }
    let mut table = HarmonicMlaTable {
        stages: Vec::new(),
        steps: Vec::new(),
        period0: 0,
    };
    if create_la_from_orbit(z_ref_f64, &mut table) {
        // `while (CreateNewLAStage());` + garde-fou anti-stagnation fractall.
        loop {
            let prev_count = {
                let s = table.stages.last().unwrap();
                s.end - s.begin
            };
            if table.stages.len() >= MAX_STAGES || !create_new_la_stage(&mut table) {
                break;
            }
            let s = table.stages.last().unwrap();
            if s.end - s.begin >= prev_count {
                break;
            }
        }
    }
    Some(table)
}

/// Mirror de `CreateLAFromOrbit` (HarmonicMLA.cpp:48-102). Étage 0 : scan des
/// minima courants de cheb|Z[i]| ; période au premier min < prevMin·2⁻⁴ ;
/// seuil de coupe = moyenne géométrique `prevMin·sqrt(min/prevMin)` ; segments
/// coupés à `cheb|Z[i]| < threshold || length ≥ Period`.
/// `next_stage_la_index` = index dans l'orbite brute du début du segment.
fn create_la_from_orbit(reference: &[Complex64], table: &mut HarmonicMlaTable) -> bool {
    let reference_length = reference.len() - 1;
    let steps = &mut table.steps;

    let mut period: usize = 0;
    let mut min_magnitude = chebyshev_norm(reference[1]);
    let mut prev_min_magnitude = min_magnitude;

    let mut step = LaStep::new(0, Complex64::new(0.0, 0.0));

    let mut i = 1usize;
    while i < reference_length {
        let magnitude_z = chebyshev_norm(reference[i]);
        if magnitude_z < min_magnitude {
            prev_min_magnitude = min_magnitude;
            min_magnitude = magnitude_z;
            if min_magnitude < prev_min_magnitude * PERIOD_DETECTION_THRESHOLD {
                period = i;
                break;
            }
        }
        step = step.step(reference[i]);
        i += 1;
    }

    steps.push(step);

    if period == 0 {
        // Pas de période : table à 1 segment + sentinelle, pas d'étage suivant.
        table.stages.push(StageInfo { begin: 0, end: 1 });
        steps.push(LaStep::new(0, reference[reference_length]));
        return false;
    }
    table.period0 = period as u32;

    let threshold = prev_min_magnitude * (min_magnitude / prev_min_magnitude).sqrt();

    let mut step = LaStep::new(i as u32, reference[i]);
    i += 1;

    while i < reference_length {
        if chebyshev_norm(reference[i]) < threshold || step.length as usize >= period {
            steps.push(step);
            step = LaStep::new(i as u32, reference[i]);
        } else {
            step = step.step(reference[i]);
        }
        i += 1;
    }

    steps.push(step);
    // Bornes AVANT le push de la sentinelle (comme le .cpp :
    // `LAStages.push_back({0, LASteps.size()})`).
    table.stages.push(StageInfo {
        begin: 0,
        end: steps.len() as u32,
    });
    // Sentinelle : Z de FIN d'étage (l'évaluateur lit `steps[j].z` du segment
    // SUIVANT après application — `z = δ + steps[j].Z`).
    steps.push(LaStep::new(0, reference[reference_length]));

    true
}

/// Mirror de `CreateNewLAStage` (HarmonicMLA.cpp:104-165) : composition des
/// segments de l'étage précédent, MÊME détection de période sur les cheb|Z de
/// début de segment| (super-période plafonnée par `length ≥ Period`).
/// `next_stage_la_index` = index GLOBAL dans `steps` du segment de l'étage
/// précédent où descendre.
fn create_new_la_stage(table: &mut HarmonicMlaTable) -> bool {
    let prev_stage = *table.stages.last().unwrap();
    let steps = &mut table.steps;
    let begin = steps.len() as u32;

    let mut period: u32 = 0;
    let mut i = prev_stage.begin as usize;

    let mut min_magnitude = chebyshev_norm(steps[i + 1].z);
    let mut prev_min_magnitude = min_magnitude;

    let mut step = steps[i];
    step.next_stage_la_index = i as u32;
    i += 1;

    while i < prev_stage.end as usize {
        let magnitude_z = chebyshev_norm(steps[i].z);
        if magnitude_z < min_magnitude {
            prev_min_magnitude = min_magnitude;
            min_magnitude = magnitude_z;
            if min_magnitude < prev_min_magnitude * PERIOD_DETECTION_THRESHOLD {
                period = step.length;
                break;
            }
        }
        step = step.composite(&steps[i]);
        i += 1;
    }

    steps.push(step);

    if period == 0 {
        table.stages.push(StageInfo {
            begin,
            end: steps.len() as u32,
        });
        // Sentinelle = sentinelle de l'étage précédent (`LASteps[prevStage.End]`).
        let sentinel = steps[prev_stage.end as usize];
        steps.push(sentinel);
        return false;
    }

    let threshold = prev_min_magnitude * (min_magnitude / prev_min_magnitude).sqrt();

    let mut step = steps[i];
    step.next_stage_la_index = i as u32;
    i += 1;

    while i < prev_stage.end as usize {
        if chebyshev_norm(steps[i].z) < threshold || step.length >= period {
            steps.push(step);
            step = steps[i];
            step.next_stage_la_index = i as u32;
        } else {
            step = step.composite(&steps[i]);
        }
        i += 1;
    }

    steps.push(step);
    table.stages.push(StageInfo {
        begin,
        end: steps.len() as u32,
    });
    let sentinel = steps[prev_stage.end as usize];
    steps.push(sentinel);

    true
}

/// Évaluateur pixel — mirror de `Evaluate` (HarmonicMLA.cpp:167-228) avec les
/// trois écarts documentés en tête de module (cap `iteration_max`, garde
/// anti-over-skip au bailout params, queue directe à sémantique
/// `iterate_pixel_unified_mandelbrot_impl`).
///
/// Compteurs : `bla_steps` = segments LA appliqués, `rebase_count` = rebases
/// d'étage + rebases F3 de la queue directe.
/// `max_perturb_iterations` : cap des pas DIRECTS (0 = illimité) — clampé
/// `≥ iteration_max` par le chemin commun, comme le path BLA. Le cap
/// `max_bla_steps` n'est PAS appliqué : les segments harmoniques ne sont pas
/// des sauts mip F3, et un arrêt en pleine descente laisserait `j` en espace
/// segments (mis-index de la queue directe) ; le clamp `≥ iteration_max` le
/// rend de toute façon inopérant sur le path actuel.
pub fn iterate_pixel_harmonic_mla(
    ref_orbit: &ReferenceOrbit,
    table: &HarmonicMlaTable,
    dc: Complex64,
    iteration_max: u32,
    bailout: f64,
    max_perturb_iterations: u32,
) -> UnifiedPixelResult {
    let bailout_sqr = bailout * bailout;
    let reference = &ref_orbit.z_ref_f64;
    let ref_len = reference.len();
    if ref_len < 2 || table.stages.is_empty() {
        // Jamais atteint via le routage (build gaté `len ≥ 9`) — garde défensive.
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

    let norm_dc = chebyshev_norm(dc);

    // Invariant LA : `z` = z absolu courant = Z[position courante] + δ
    // (bit-exact : les Z de segments/sentinelles sont des copies de l'orbite).
    let mut z = Complex64::new(0.0, 0.0);
    let mut dz = Complex64::new(0.0, 0.0);
    let mut i: u32 = 0;
    let mut j: usize = table.stages[table.stages.len() - 1].begin as usize;
    let mut bla_steps = 0u32;
    let mut rebase_count = 0u32;

    // ── Phase LA : descente d'étages depuis le plus haut ───────────────────
    let mut stage = table.stages.len();
    while stage > 0 {
        stage -= 1;
        let begin = table.stages[stage].begin as usize;
        let end = table.stages[stage].end as usize;

        while i < iteration_max {
            let step = &table.steps[j];
            // Premier pas quadratique EXACT : δ' = δ·(Z+z) = δ·(2Z+δ).
            let new_dz = dz * (step.z + z);

            // 1 check de validité par segment (chebyshev, mirror).
            if chebyshev_norm(new_dz) > step.valid_radius || norm_dc > step.valid_radius_c {
                j = step.next_stage_la_index as usize;
                break;
            }
            // Écart fractall #1 : cap itérations — ne jamais dépasser
            // iteration_max (les étages fins ont des segments courts, la
            // queue directe finit).
            if i + step.length > iteration_max {
                j = step.next_stage_la_index as usize;
                break;
            }

            let cand_dz = new_dz * step.a + dc * step.b;
            // `steps[j+1].z` : Z de début du segment SUIVANT (la sentinelle
            // fournit la valeur de FIN d'étage).
            let cand_z = cand_dz + table.steps[j + 1].z;

            // Écart fractall #2 : garde anti-over-skip à l'atterrissage. Le
            // segment a sauté PAR-DESSUS l'escape → rejet (dz/i/j inchangés),
            // descente ; le check d'escape de la queue directe attrape le
            // pixel à l'itération exacte.
            if cand_z.norm_sqr() >= bailout_sqr {
                j = step.next_stage_la_index as usize;
                break;
            }

            dz = cand_dz;
            i += step.length;
            j += 1;
            z = cand_z;
            bla_steps += 1;

            // Rebase d'étage (mirror) : fin d'étage OU |z| < |δ| (chebyshev).
            if j == end || chebyshev_norm(z) < chebyshev_norm(dz) {
                j = begin;
                dz = z;
                rebase_count += 1;
            }
        }
    }

    if i >= iteration_max {
        // iteration_max atteint PENDANT la descente : `j` peut être en espace
        // segments (pas un index d'orbite) — ne pas indexer l'orbite. `z` est
        // le z absolu courant (invariant LA), et le cap #1 garantit
        // `i == iteration_max` exactement.
        return UnifiedPixelResult {
            iteration: i,
            z_final: z,
            rebase_count,
            bla_steps,
            orbit: None,
            distance: None,
            is_interior: false,
            ref_exhausted: false,
        };
    }

    // ── Queue directe sur l'orbite brute ───────────────────────────────────
    // `j` = index ORBITE (next_stage_la_index d'un segment étage 0). Réplique
    // `iterate_pixel_unified_mandelbrot_impl` sans BLA : escape en tête de
    // boucle, rebase F3 strict, rebase-at-end atom, flag ref_exhausted.
    let mut m = j;
    let mut z_m = reference[m.min(ref_len - 1)];
    let mut z_abs = z_m + dz; // == z (invariant LA)
    let mut ref_exhausted_flag = false;
    let mut iters_ptb = 0u32;

    while i < iteration_max
        && (max_perturb_iterations == 0 || iters_ptb < max_perturb_iterations)
    {
        // Bailout absolu |Z[m]+δ|² ≥ bailout² (attrape aussi l'escape des
        // segments rejetés par la garde anti-over-skip).
        if z_abs.norm_sqr() >= bailout_sqr {
            return UnifiedPixelResult {
                iteration: i,
                z_final: z_abs,
                rebase_count,
                bla_steps,
                orbit: None,
                distance: None,
                is_interior: false,
                ref_exhausted: false,
            };
        }

        // Pas direct : δ' = δ·(Z[m]+z) + dc (forme factorisée Imagina de
        // 2·Z·δ + δ² + dc, avec z = Z[m]+δ).
        dz = dz * (z_m + z_abs) + dc;
        i += 1;
        m += 1;
        iters_ptb += 1;

        // `Z[min(m, ref_len-1)]` : clamp historique (m == ref_len possible).
        let z_after = reference[m.min(ref_len - 1)];

        if !dz.re.is_finite() || !dz.im.is_finite() {
            return UnifiedPixelResult {
                iteration: i,
                z_final: z_after + dz,
                rebase_count,
                bla_steps,
                orbit: None,
                distance: None,
                is_interior: false,
                ref_exhausted: false,
            };
        }

        let end_of_ref = m + 1 >= ref_len;
        if end_of_ref {
            if ref_orbit.atom_truncated {
                // Rebase-at-end F3 (`hybrid.cc:301`) : δ := Z[end]+δ, m := 0.
                dz = z_after + dz;
                m = 0;
                rebase_count += 1;
                z_m = reference[0];
                z_abs = z_m + dz;
            } else {
                // Réf épuisée (centre escape-time) : même contrat que le path
                // BLA — flag + break, le caller route `glitched` → GMP/pixel.
                ref_exhausted_flag = true;
                z_m = z_after;
                z_abs = z_m + dz;
                break;
            }
        } else {
            let z_curr = z_after + dz;
            let z_curr_norm_sqr = z_curr.norm_sqr();
            if z_curr_norm_sqr < dz.norm_sqr() {
                // Rebase F3 strict : |Z+δ|² < |δ|² → δ := Z+δ, m := 0.
                dz = z_curr;
                m = 0;
                rebase_count += 1;
                z_m = reference[0];
                z_abs = z_m + dz;
            } else {
                z_m = z_after;
                z_abs = z_curr;
            }
        }
    }

    UnifiedPixelResult {
        iteration: i,
        z_final: z_abs,
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
    use crate::fractal::bytecode::pixel_loop::iterate_pixel_unified_mandelbrot;
    use crate::fractal::bytecode::{build_bla_table_for_formula, compile_formula};
    use crate::fractal::perturbation::orbit::compute_reference_orbit;
    use crate::fractal::{default_params_for_type, AlgorithmMode, FractalType};

    /// Orbite Mandelbrot GMP 256 b arrondie f64, avec le 0 de tête
    /// (convention `z_ref_f64[0] = 0`) — comme en production.
    fn gmp_orbit(cx: f64, cy: f64, n: usize) -> Vec<Complex64> {
        use rug::Complex;
        let prec = 256;
        let c = Complex::with_val(prec, (cx, cy));
        let mut z = Complex::with_val(prec, (0.0, 0.0));
        let mut out = vec![Complex64::new(0.0, 0.0)];
        for _ in 0..n {
            z.square_mut();
            z += &c;
            out.push(Complex64::new(z.real().to_f64(), z.imag().to_f64()));
        }
        out
    }

    /// (a) Build sur l'orbite du centre superstable période 3 (« airplane »,
    /// c = -1.7548776662466927) : le scan des minima doit détecter period0=3,
    /// produire ≥ 2 étages, et plafonner les segments étage 0 à la période.
    #[test]
    fn build_detects_period_3_on_airplane_orbit() {
        let orbit = gmp_orbit(-1.7548776662466927, 0.0, 200);
        let table = build_harmonic_mla_table(&orbit).expect("table Harmonic MLA");

        assert_eq!(table.period0, 3, "période étage 0");
        assert!(
            table.stages.len() >= 2,
            "au moins 2 étages, obtenu {}",
            table.stages.len()
        );

        let s0 = table.stages[0];
        for (k, step) in table.steps[s0.begin as usize..s0.end as usize]
            .iter()
            .enumerate()
        {
            assert!(
                step.length <= table.period0,
                "segment étage 0 #{k} : length {} > période {}",
                step.length,
                table.period0
            );
            // next_stage_la_index étage 0 = index d'orbite valide.
            assert!(
                (step.next_stage_la_index as usize) < orbit.len(),
                "segment étage 0 #{k} : index orbite hors bornes"
            );
        }
        // Structure sentinelle : chaque étage a sa sentinelle à l'index `end`.
        for (s, stage) in table.stages.iter().enumerate() {
            assert!(
                (stage.end as usize) < table.steps.len(),
                "étage {s} : sentinelle manquante"
            );
        }
    }

    /// Orbite trop courte (referenceLength < 8) → pas de table.
    #[test]
    fn build_rejects_short_orbit() {
        let orbit = gmp_orbit(-1.7548776662466927, 0.0, 7); // len 8 → refLen 7
        assert!(build_harmonic_mla_table(&orbit).is_none());
    }

    /// (b) Évaluateur vs boucle BLA actuelle sur un cas mid-zoom (needle
    /// minibrot 1e6, même vue que `render_full_image_vs_legacy_golden`) :
    /// même itération d'escape pour ≥ 99 % de 100 dc échantillonnés.
    /// L'écart résiduel = classe approximation (rayons de validité ≠).
    #[test]
    fn evaluator_matches_bla_loop_mid_zoom() {
        let width = 160u32;
        let height = 100u32;
        let iter_max = 1500u32;
        let zoom = 1e6;
        let span_x = 4.0 / zoom;
        let span_y = span_x * height as f64 / width as f64;

        let mut params = default_params_for_type(FractalType::Mandelbrot, width, height);
        params.center_x = -1.7693831791955;
        params.center_y = 0.004236847918736;
        params.span_x = span_x;
        params.span_y = span_y;
        params.iteration_max = iter_max;
        params.algorithm_mode = AlgorithmMode::Perturbation;
        let (orbit, _, _) =
            compute_reference_orbit(&params, None, true).expect("compute_reference_orbit");
        assert_eq!(orbit.cycle_period, 0, "test suppose cycle_period == 0");

        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let diag_px = ((width as f64).powi(2) + (height as f64).powi(2)).sqrt();
        let c_norm = (span_x / width as f64) * diag_px;
        let tables = build_bla_table_for_formula(&formula, &orbit.z_ref_f64, c_norm, 6e-8)
            .expect("BLA table");
        let bla = &tables[0];
        let table = build_harmonic_mla_table(&orbit.z_ref_f64).expect("table Harmonic MLA");

        let total = 100u32;
        let mut mismatches = 0u32;
        let mut max_diff = 0u32;
        for k in 0..total {
            // Grille 10×10 couvrant la vue.
            let fx = (k % 10) as f64 / 9.0 - 0.5;
            let fy = (k / 10) as f64 / 9.0 - 0.5;
            let dc = Complex64::new(fx * span_x, fy * span_y);

            let r_bla =
                iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 25.0, 0, 0);
            let r_hml =
                iterate_pixel_harmonic_mla(&orbit, &table, dc, iter_max, 25.0, 0);

            if r_bla.iteration != r_hml.iteration {
                mismatches += 1;
                max_diff = max_diff.max(r_bla.iteration.abs_diff(r_hml.iteration));
            }
        }
        eprintln!(
            "[HARMONIC TEST] mismatches={mismatches}/{total} max_iter_diff={max_diff}"
        );
        assert!(
            mismatches <= 1,
            "≥ 99 % attendu : {mismatches}/{total} divergents (max diff {max_diff})"
        );
    }
}
