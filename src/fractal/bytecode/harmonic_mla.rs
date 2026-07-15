//! Prototype **Harmonic LA** (Imagina / Zhuoran Yu) — port fidèle de
//! `Imagina-Algorithms/HarmonicMLA.{h,cpp}` (variante « Mag », premier jalon)
//! ET `HarmonicLLA.{h,cpp}` (variante aux **dips de rayon**, EN PRODUCTION
//! dans Imagina). Design commenté : `docs/imagina-algorithms-analysis.md`
//! §HarmonicLLA/§HarmonicMLA. TODO G8.2 « Prototype Harmonic LA », env-gated :
//! `FRACTALL_HARMONIC_LA=1|lla` → **LLA** (défaut du prototype),
//! `FRACTALL_HARMONIC_LA=mla` → MLA (conservée pour l'A/B).
//!
//! Idée : remplacer la table BLA mip 2^k par des **segments LA à longueur
//! variable**, organisés en **étages** période/super-période, évalués par
//! **descente d'étage** : ~1 check de validité par segment accepté au lieu
//! d'un lookup multi-niveaux par position. Mandelbrot f64 conforme uniquement
//! (A/B complexes, PAS Mat2). Le premier pas d'un segment est **quadratique
//! EXACT** (`δ' = δ·(Z+z)` = `δ·(2Z+δ)`) : le seul terme non-linéaire de la
//! perturbation est traité exactement là où δ est le plus gros relatif →
//! rayons de validité plus larges que la BLA linéarisée.
//!
//! Les deux variantes ne diffèrent QUE par la segmentation (même `LaStep`,
//! même évaluateur) :
//! - **MLA** : coupe quand le min courant de cheb|Z| chute sous `prevMin·2⁻⁴`
//!   (période) puis au seuil moyenne-géométrique. Simple, mais les segments
//!   traversent les passages près de 0 → rayons de validité étroits sur les
//!   orbites longues (e50/dragon +44/+47 % mesurés, cf. TODO G8.2).
//! - **LLA** : coupe aux **dips** = chute du rayon de validité > 2⁻¹⁰ en UN
//!   pas (`DipDetectionThreshold`, HarmonicLLA.h:18) = passage de l'orbite
//!   près de 0 (frontière d'atome). Les segments épousent la STRUCTURE de
//!   l'orbite ; le premier dip de l'étage ≈ sa période (plafond des
//!   longueurs). Un segment démarre en longueur 2 (`new2`, premier pas déjà
//!   étendu) sauf si le point suivant dipperait aussitôt (`detect_dip`) →
//!   longueur 1 (pas exact, rayon 1).
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
/// min courant de cheb|Z| chute sous `prevMin · 2⁻⁴`. (Variante MLA.)
const PERIOD_DETECTION_THRESHOLD: f64 = 0.0625; // 2^-4, exact

/// `DipDetectionThreshold` (HarmonicLLA.h:18) : dip détecté quand le rayon de
/// validité chute sous `prev · 2⁻¹⁰` en un seul pas. (Variante LLA.)
const DIP_DETECTION_THRESHOLD: f64 = 1.0 / 1024.0; // 2^-10, exact

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

    /// `LAStep(size_t i, complex z0, complex z1)` (HarmonicLLA.h:30-34) :
    /// segment de DEUX pas — le premier (quadratique exact) déjà consommé,
    /// le second linéarisé autour de `z1` (`A = 2·z1`, `B = A+1`).
    fn new2(i: u32, z0: Complex64, z1: Complex64) -> Self {
        let a = 2.0 * z1;
        let valid_radius = chebyshev_norm(z1) * VALID_RADIUS_SCALE;
        LaStep {
            z: z0,
            a,
            b: a + Complex64::new(1.0, 0.0),
            valid_radius,
            valid_radius_c: valid_radius,
            length: 2,
            next_stage_la_index: i,
        }
    }

    /// `DetectDip(complex z)` (HarmonicLLA.h:39-41) : est-ce qu'étendre le
    /// segment par `z` provoquerait un dip ? (Prédictif — sans construire
    /// l'extension.)
    fn detect_dip(&self, z: Complex64) -> bool {
        chebyshev_norm(z) * VALID_RADIUS_SCALE / chebyshev_norm(self.a)
            < self.valid_radius * DIP_DETECTION_THRESHOLD
    }

    /// `Step(complex z)` (HarmonicLLA.h:43-59) : extension du segment d'un pas
    /// (z = valeur de l'orbite à la fin actuelle du segment) + flag dip (le
    /// rayon chute sous `prev · DipDetectionThreshold` en ce pas). La math du
    /// LaStep est IDENTIQUE à HarmonicMLA.h:32-46 (MLA ignore le flag).
    fn step_dip(&self, z: Complex64) -> (LaStep, bool) {
        let radius = chebyshev_norm(z) * VALID_RADIUS_SCALE;
        let valid_radius = self.valid_radius.min(radius / chebyshev_norm(self.a));
        let dip = valid_radius < self.valid_radius * DIP_DETECTION_THRESHOLD;
        (
            LaStep {
                valid_radius,
                valid_radius_c: self.valid_radius_c.min(radius / chebyshev_norm(self.b)),
                a: 2.0 * z * self.a,
                b: 2.0 * z * self.b + Complex64::new(1.0, 0.0),
                z: self.z,
                length: self.length + 1,
                next_stage_la_index: self.next_stage_la_index,
            },
            dip,
        )
    }

    fn step(&self, z: Complex64) -> LaStep {
        self.step_dip(z).0
    }

    /// `Composite(const LAStep &step)` (HarmonicLLA.h:61-84) : composition
    /// `self` puis `step` + flag dip. Récurrences EXACTES du .h — le rayon
    /// intermédiaire utilise `A/B` AVANT multiplication par `step.a` (et
    /// `step.valid_radius` pour les DEUX rayons, fidèle à la source) ; le
    /// flag dip est évalué sur ce rayon INTERMÉDIAIRE (avant le second min),
    /// comme HarmonicLLA.h:72. Math identique à HarmonicMLA.h:48-68.
    fn composite_dip(&self, step: &LaStep) -> (LaStep, bool) {
        let radius = chebyshev_norm(step.z) * VALID_RADIUS_SCALE;

        let mut valid_radius = self.valid_radius.min(radius / chebyshev_norm(self.a));
        let mut valid_radius_c = self.valid_radius_c.min(radius / chebyshev_norm(self.b));
        let dip = valid_radius < self.valid_radius * DIP_DETECTION_THRESHOLD;
        let a = 2.0 * step.z * self.a;
        let b = 2.0 * step.z * self.b;

        valid_radius = valid_radius.min(step.valid_radius / chebyshev_norm(a));
        valid_radius_c = valid_radius_c.min(step.valid_radius / chebyshev_norm(b));
        (
            LaStep {
                a: a * step.a,
                b: b * step.a + step.b,
                z: self.z,
                valid_radius,
                valid_radius_c,
                length: self.length + step.length,
                next_stage_la_index: self.next_stage_la_index,
            },
            dip,
        )
    }

    fn composite(&self, step: &LaStep) -> LaStep {
        self.composite_dip(step).0
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

/// Variante de segmentation sélectionnée par la VALEUR du gate env.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarmonicVariant {
    /// Segmentation par minima de magnitude (`HarmonicMLA.cpp`).
    Mla,
    /// Segmentation aux dips de rayon (`HarmonicLLA.cpp`) — prod Imagina.
    Lla,
}

/// Mode de routage harmonic (G9.3) piloté par `FRACTALL_HARMONIC_LA`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarmonicMode {
    /// Jamais routé (kill switch : `0`/`off`/`false`/`bla`, ou valeur inconnue).
    Off,
    /// **Défaut (env non posé / `auto`)** : routé quand la classe « période
    /// courte » est détectée au build (`detect_period0` +
    /// `wisdom::route_harmonic_auto`), variante LLA (prod Imagina).
    Auto,
    /// Toujours routé (si éligible) : `1`/`true`/`lla` → LLA, `mla` → MLA (A/B).
    Forced(HarmonicVariant),
}

/// Parse une-fois de `FRACTALL_HARMONIC_LA`. Depuis G9.3, l'absence de la
/// variable = mode **Auto** (le prototype env-gated est devenu une technique
/// routée par le wisdom) ; les anciennes valeurs de gate forcent la variante.
pub fn harmonic_mode() -> HarmonicMode {
    static F: OnceLock<HarmonicMode> = OnceLock::new();
    *F.get_or_init(|| {
        match std::env::var("FRACTALL_HARMONIC_LA")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "auto" => HarmonicMode::Auto,
            "1" | "true" | "lla" => HarmonicMode::Forced(HarmonicVariant::Lla),
            "mla" => HarmonicMode::Forced(HarmonicVariant::Mla),
            _ => HarmonicMode::Off,
        }
    })
}

/// Variante candidate du mode courant (`None` = harmonic coupé). Auto → LLA.
pub fn harmonic_variant() -> Option<HarmonicVariant> {
    match harmonic_mode() {
        HarmonicMode::Off => None,
        HarmonicMode::Auto => Some(HarmonicVariant::Lla),
        HarmonicMode::Forced(v) => Some(v),
    }
}

/// Probe du routage AUTO (G9.3) : période de l'atome dominant = longueur du
/// premier segment de l'étage 0 (premier **dip** LLA), `0` si aucun dip (orbite
/// non périodique à l'échelle LA, ou trop courte). O(period0) dans le cas
/// routé — le scan s'arrête au premier dip ; O(orbite) sinon (une passe f64
/// triviale devant le build BLA O(M)). Réplique EXACTE du scan d'ouverture de
/// `create_la_from_orbit_lla` — égalité verrouillée par test unitaire.
pub fn detect_period0(reference: &[Complex64]) -> u32 {
    if reference.len() < 9 {
        return 0;
    }
    let reference_length = reference.len() - 1;
    let mut step = LaStep::new2(0, Complex64::new(0.0, 0.0), reference[1]);
    let mut i = 2usize;
    while i < reference_length {
        let (new_step, dip) = step.step_dip(reference[i]);
        if dip {
            return i as u32;
        }
        step = new_step;
        i += 1;
    }
    0
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

/// Construit la table LLA depuis l'orbite brute f64. Mirror de `Prepare`
/// (HarmonicLLA.cpp:10-21) — même contrat que `build_harmonic_mla_table`
/// (mêmes garde-fous plafond d'étages + anti-stagnation fractall).
pub fn build_harmonic_lla_table(z_ref_f64: &[Complex64]) -> Option<HarmonicMlaTable> {
    if z_ref_f64.len() < 9 {
        return None;
    }
    let mut table = HarmonicMlaTable {
        stages: Vec::new(),
        steps: Vec::new(),
        period0: 0,
    };
    if create_la_from_orbit_lla(z_ref_f64, &mut table) {
        loop {
            let prev_count = {
                let s = table.stages.last().unwrap();
                s.end - s.begin
            };
            if table.stages.len() >= MAX_STAGES || !create_new_la_stage_lla(&mut table) {
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

/// Mirror de `CreateLAFromOrbit` (HarmonicLLA.cpp:48-102). Étage 0 : le
/// premier segment (longueur 2 au départ, `LAStep(0, 0, ref[1])`) s'étend
/// jusqu'au premier **dip** → sa longueur = la période de l'atome dominant.
/// Ensuite : segments coupés à `dip || length ≥ Period` ; chaque nouveau
/// segment démarre en longueur 2 (`new2`) SAUF si le point suivant dipperait
/// aussitôt (`detect_dip` sur l'extension rejetée) → longueur 1.
/// `next_stage_la_index` = index dans l'orbite brute du début du segment.
fn create_la_from_orbit_lla(reference: &[Complex64], table: &mut HarmonicMlaTable) -> bool {
    let reference_length = reference.len() - 1;
    let steps = &mut table.steps;

    let mut period: usize = 0;
    let mut step = LaStep::new2(0, Complex64::new(0.0, 0.0), reference[1]);

    let mut i = 2usize;
    while i < reference_length {
        let (new_step, dip) = step.step_dip(reference[i]);
        if dip {
            period = i;
            break;
        }
        step = new_step;
        i += 1;
    }

    steps.push(step);

    if period == 0 {
        table.stages.push(StageInfo { begin: 0, end: 1 });
        steps.push(LaStep::new(0, reference[reference_length]));
        return false;
    }
    table.period0 = period as u32;

    let mut step = if i + 1 >= reference_length {
        let s = LaStep::new(i as u32, reference[i]);
        i += 1;
        s
    } else {
        let s = LaStep::new2(i as u32, reference[i], reference[i + 1]);
        i += 2;
        s
    };

    while i < reference_length {
        let (new_step, dip) = step.step_dip(reference[i]);
        if dip || step.length as usize >= period {
            // L'extension par reference[i] est REJETÉE : le point de dip
            // devient le premier pas (quadratique exact) du segment suivant.
            steps.push(step);
            if i + 1 >= reference_length || new_step.detect_dip(reference[i + 1]) {
                step = LaStep::new(i as u32, reference[i]);
            } else {
                step = LaStep::new2(i as u32, reference[i], reference[i + 1]);
                i += 1;
            }
        } else {
            step = new_step;
        }
        i += 1;
    }

    steps.push(step);
    table.stages.push(StageInfo {
        begin: 0,
        end: steps.len() as u32,
    });
    steps.push(LaStep::new(0, reference[reference_length]));

    true
}

/// Mirror de `CreateNewLAStage` (HarmonicLLA.cpp:104-175) : composition des
/// segments de l'étage précédent, dips sur le rayon composé ; la période de
/// l'étage = `step.length` au premier dip (super-période).
/// `next_stage_la_index` = index GLOBAL dans `steps` du segment de l'étage
/// précédent où descendre.
fn create_new_la_stage_lla(table: &mut HarmonicMlaTable) -> bool {
    let prev_stage = *table.stages.last().unwrap();
    let steps = &mut table.steps;
    let begin = steps.len() as u32;
    let prev_end = prev_stage.end as usize;

    let mut period: u32 = 0;
    let mut i = prev_stage.begin as usize;

    let mut step = steps[i].composite_dip(&steps[i + 1]).0;
    step.next_stage_la_index = i as u32;
    i += 2;

    while i < prev_end {
        let (new_step, dip) = step.composite_dip(&steps[i]);
        if dip {
            period = step.length;
            steps.push(step);
            if i + 1 >= prev_end || new_step.detect_dip(steps[i + 1].z) {
                step = steps[i];
                step.next_stage_la_index = i as u32;
                i += 1;
            } else {
                step = steps[i].composite_dip(&steps[i + 1]).0;
                step.next_stage_la_index = i as u32;
                i += 2;
            }
            break;
        }
        step = new_step;
        i += 1;
    }

    if period == 0 {
        steps.push(step);
        table.stages.push(StageInfo {
            begin,
            end: steps.len() as u32,
        });
        let sentinel = steps[prev_end];
        steps.push(sentinel);
        return false;
    }

    while i < prev_end {
        let (new_step, dip) = step.composite_dip(&steps[i]);
        if dip || step.length >= period {
            steps.push(step);
            if i + 1 >= prev_end || new_step.detect_dip(steps[i + 1].z) {
                step = steps[i];
                step.next_stage_la_index = i as u32;
            } else {
                step = steps[i].composite_dip(&steps[i + 1]).0;
                step.next_stage_la_index = i as u32;
                i += 1;
            }
        } else {
            step = new_step;
        }
        i += 1;
    }

    steps.push(step);
    table.stages.push(StageInfo {
        begin,
        end: steps.len() as u32,
    });
    let sentinel = steps[prev_end];
    steps.push(sentinel);

    true
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
    let mut ref_exhausted_flag = false;
    // Budget de pas directs GLOBAL au pixel (PAS remis à zéro par la
    // ré-ascension de l'écart #4).
    let mut iters_ptb = 0u32;
    let mut z_abs = Complex64::new(0.0, 0.0);

    // Écart fractall #4 : **ré-ascension après rebase de la queue directe**
    // (absent d'Imagina — leur évaluateur ne quitte jamais la queue). Un
    // rebase de la queue (F3 strict ou rebase-at-end) laisse EXACTEMENT
    // l'état d'entrée de la phase LA (`m = 0`, `z = Z[0]+δ = δ`) → on remonte
    // au sommet des étages via `continue 'render`. Récupère les pixels dont
    // les rayons redeviennent valides après rebase (e50 : −15 % mesuré).
    // Garde-fou anti-ping-pong : on ne remonte que si le DERNIER passage LA
    // a été productif (≥ 1 segment appliqué) — sinon les pixels quasi-
    // périodiques à rayons morts paieraient une descente stérile (≤
    // MAX_STAGES checks) à CHAQUE rebase (dragon : +10 % sans le garde).
    // Terminaison : chaque rebase de queue consomme ≥ 1 pas direct et `i`
    // est monotone → au plus une descente par rebase.
    let mut reascend = true;
    'render: loop {
        let segs_at_entry = bla_steps;
        // ── Phase LA : descente d'étages depuis le plus haut ───────────────
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
                if chebyshev_norm(new_dz) > step.valid_radius
                    || norm_dc > step.valid_radius_c
                {
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
                // `steps[j+1].z` : Z de début du segment SUIVANT (la
                // sentinelle fournit la valeur de FIN d'étage).
                let cand_z = cand_dz + table.steps[j + 1].z;

                // Écart fractall #2 : garde anti-over-skip à l'atterrissage.
                // Le segment a sauté PAR-DESSUS l'escape → rejet (dz/i/j
                // inchangés), descente ; le check d'escape de la queue
                // directe attrape le pixel à l'itération exacte.
                if cand_z.norm_sqr() >= bailout_sqr {
                    j = step.next_stage_la_index as usize;
                    break;
                }

                dz = cand_dz;
                i += step.length;
                j += 1;
                z = cand_z;
                bla_steps += 1;

                // Rebase d'étage (mirror) : fin d'étage OU |z| < |δ|
                // (chebyshev).
                if j == end || chebyshev_norm(z) < chebyshev_norm(dz) {
                    j = begin;
                    dz = z;
                    rebase_count += 1;
                }
            }
        }

        if i >= iteration_max {
            // iteration_max atteint PENDANT la descente : `j` peut être en
            // espace segments (pas un index d'orbite) — ne pas indexer
            // l'orbite. `z` est le z absolu courant (invariant LA), et le
            // cap #1 garantit `i == iteration_max` exactement.
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

        // Un passage LA stérile désarme la ré-ascension pour ce pixel (les
        // rebases suivants restent dans la queue directe, comme Imagina).
        // A/B mesuré (LLA, entrelacé ×3) : gardée vs jamais = e50 −24 %,
        // dragon/glitch_test_2 neutres, e113 +8 % ; gardée ≥ toujours partout.
        if bla_steps == segs_at_entry {
            reascend = false;
        }

        // ── Queue directe sur l'orbite brute ───────────────────────────────
        // `j` = index ORBITE (next_stage_la_index d'un segment étage 0).
        // Réplique `iterate_pixel_unified_mandelbrot_impl` sans BLA : escape
        // en tête de boucle, rebase F3 strict, rebase-at-end atom, flag
        // ref_exhausted.
        let mut m = j;
        let mut z_m = reference[m.min(ref_len - 1)];
        z_abs = z_m + dz; // == z (invariant LA)

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

            // `Z[min(m, ref_len-1)]` : clamp historique (m == ref_len
            // possible).
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
                    // Rebase-at-end F3 (`hybrid.cc:301`) : δ := Z[end]+δ,
                    // m := 0 — puis ré-ascension (écart #4) si armée.
                    dz = z_after + dz;
                    rebase_count += 1;
                    if reascend {
                        z = dz; // Z[0] = 0 ⇒ z absolu = δ (invariant LA)
                        j = table.stages[table.stages.len() - 1].begin as usize;
                        continue 'render;
                    }
                    m = 0;
                    z_m = reference[0];
                    z_abs = z_m + dz;
                } else {
                    // Réf épuisée (centre escape-time) : même contrat que le
                    // path BLA — flag + break, le caller route `glitched` →
                    // GMP/pixel.
                    ref_exhausted_flag = true;
                    z_abs = z_after + dz;
                    break 'render;
                }
            } else {
                let z_curr = z_after + dz;
                let z_curr_norm_sqr = z_curr.norm_sqr();
                if z_curr_norm_sqr < dz.norm_sqr() {
                    // Rebase F3 strict : |Z+δ|² < |δ|² → δ := Z+δ, m := 0 —
                    // puis ré-ascension (écart #4) si armée.
                    dz = z_curr;
                    rebase_count += 1;
                    if reascend {
                        z = dz;
                        j = table.stages[table.stages.len() - 1].begin as usize;
                        continue 'render;
                    }
                    m = 0;
                    z_m = reference[0];
                    z_abs = z_m + dz;
                } else {
                    z_m = z_after;
                    z_abs = z_curr;
                }
            }
        }

        // iteration_max ou budget max_perturb_iterations atteint.
        break 'render;
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

    /// Le probe de routage AUTO (G9.3) doit être EXACTEMENT le scan
    /// d'ouverture du build LLA : même period0 sur une vraie orbite, 0 sur
    /// une orbite trop courte (< 9, jamais routée).
    #[test]
    fn detect_period0_matches_lla_build() {
        let orbit = gmp_orbit(-1.7548776662466927, 0.0, 200);
        let table = build_harmonic_lla_table(&orbit).expect("table LLA");
        assert!(table.period0 > 0, "l'orbite airplane doit dipper");
        assert_eq!(detect_period0(&orbit), table.period0);
        assert_eq!(detect_period0(&orbit[..8]), 0, "orbite courte : probe 0");
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

    /// (a-LLA) Build LLA sur l'orbite superstable période 3 : le premier dip
    /// (|Z₃| ≈ 0) doit donner period0=3, ≥ 2 étages, segments étage 0
    /// plafonnés à la période, indices d'orbite valides, sentinelles en place.
    #[test]
    fn lla_build_detects_period_3_on_airplane_orbit() {
        let orbit = gmp_orbit(-1.7548776662466927, 0.0, 200);
        let table = build_harmonic_lla_table(&orbit).expect("table Harmonic LLA");

        assert_eq!(table.period0, 3, "période étage 0 (premier dip)");
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
            assert!(
                (step.next_stage_la_index as usize) < orbit.len(),
                "segment étage 0 #{k} : index orbite hors bornes"
            );
        }
        for (s, stage) in table.stages.iter().enumerate() {
            assert!(
                (stage.end as usize) < table.steps.len(),
                "étage {s} : sentinelle manquante"
            );
            // Cohérence descente : les next_stage_la_index des étages > 0
            // pointent dans les bornes de l'étage précédent.
            if s > 0 {
                let prev = table.stages[s - 1];
                for step in &table.steps[stage.begin as usize..stage.end as usize] {
                    assert!(
                        step.next_stage_la_index >= prev.begin
                            && step.next_stage_la_index < prev.end,
                        "étage {s} : next_stage_la_index {} hors [{}, {})",
                        step.next_stage_la_index,
                        prev.begin,
                        prev.end
                    );
                }
            }
        }
    }

    /// Orbite trop courte → pas de table LLA non plus.
    #[test]
    fn lla_build_rejects_short_orbit() {
        let orbit = gmp_orbit(-1.7548776662466927, 0.0, 7);
        assert!(build_harmonic_lla_table(&orbit).is_none());
    }

    /// (b-LLA) Évaluateur (commun) sur table LLA vs boucle BLA actuelle,
    /// même harnais que `evaluator_matches_bla_loop_mid_zoom`.
    #[test]
    fn lla_evaluator_matches_bla_loop_mid_zoom() {
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
        let table = build_harmonic_lla_table(&orbit.z_ref_f64).expect("table Harmonic LLA");

        let total = 100u32;
        let mut mismatches = 0u32;
        let mut max_diff = 0u32;
        for k in 0..total {
            let fx = (k % 10) as f64 / 9.0 - 0.5;
            let fy = (k / 10) as f64 / 9.0 - 0.5;
            let dc = Complex64::new(fx * span_x, fy * span_y);

            let r_bla =
                iterate_pixel_unified_mandelbrot(&orbit, bla, dc, iter_max, 25.0, 0, 0);
            let r_lla =
                iterate_pixel_harmonic_mla(&orbit, &table, dc, iter_max, 25.0, 0);

            if r_bla.iteration != r_lla.iteration {
                mismatches += 1;
                max_diff = max_diff.max(r_bla.iteration.abs_diff(r_lla.iteration));
            }
        }
        eprintln!(
            "[HARMONIC LLA TEST] mismatches={mismatches}/{total} max_iter_diff={max_diff}"
        );
        assert!(
            mismatches <= 1,
            "≥ 99 % attendu : {mismatches}/{total} divergents (max diff {max_diff})"
        );
    }
}
