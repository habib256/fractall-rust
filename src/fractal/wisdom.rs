//! Wisdom auto-dispatch — sélection automatique de l'algorithme et du **tier
//! numérique** (f64 / ComplexExp / double-double) à partir des paramètres de
//! frame, sur le modèle du `wisdom_lookup` de Fraktaler-3 (`wisdom.cc:240`).
//!
//! # Modèle F3
//!
//! F3 filtre les types numériques par **viabilité** (statique) puis choisit le
//! plus rapide (mesuré). Un type de mantisse `M` bits et d'exposant `E` bits est
//! viable si :
//!
//! - `required_exponent + 16 < 2^E / 2` — la plage d'exposant couvre le zoom ;
//! - `required_precision < M` — la mantisse couvre la précision pixel.
//!
//! où (`render.cc:219`) `pixel_spacing = 4 / (zoom·height)`, et
//! `required_precision = max(|offset|/pixel_spacing, hypot(w,h))`. Pour un rendu
//! **centré** (notre cas : l'orbite référence EST le centre de vue) l'offset est
//! ~0, donc `required_precision ≈ log2(hypot(w,h))` — une dizaine de bits. La
//! conséquence, capitale, est que **la mantisse f64 (53 b) suffit toujours pour
//! une frame centrée** : F3 n'escalade que sur l'**exposant** (profondeur de
//! zoom), jamais sur la mantisse. C'est exactement ce que fait fractall.
//!
//! # Ce que ce module centralise
//!
//! Auparavant la sélection de tier était dupliquée (`bytecode_path_label` ET
//! `try_bytecode_unified_path`, `delta.rs`). Ce module en est désormais la
//! **source unique** ([`number_tier`], [`dd_requested`], [`wants_exp`]) et
//! calcule le [`WisdomPlan`] inspectable (exposant/précision requis, tier,
//! précision GMP de l'orbite) pour le log `[WISDOM]` (env `FRACTALL_WISDOM=1`).
//!
//! # Ce que ce module ne fait PAS (encore)
//!
//! Le tier **dd (~106 b, « float128 »)** reste piloté à la demande
//! (`params.use_dd_tier` / `--dd-tier`). La viabilité F3 ne le réclame jamais
//! pour une frame centrée (mantisse requise ~11 b) : le besoin dd vient d'une
//! **sensibilité/conditionnement par pixel** (amplification de Lyapunov sur des
//! spirales, cf. e13/e30/e50, TODO G3) qu'aucun détecteur cheap fiable ne capte
//! aujourd'hui (le proxy `cbits` a été réfuté sur données). L'auto-firing du dd
//! est donc explicitement hors périmètre ; ce module fournit le point d'accroche
//! (`required_precision` vs `tier.mantissa_bits()`) pour le brancher plus tard.

use std::sync::OnceLock;

use crate::fractal::bytecode::harmonic_mla::{harmonic_variant, HarmonicVariant};
use crate::fractal::types::{FractalParams, FractalType, PlaneTransform};
use crate::fractal::perturbation::{
    compute_perturbation_precision_bits, effective_pixel_size, log2_zoom,
};
use crate::fractal::perturbation::compress::compress_enabled;
use crate::fractal::perturbation::delta::pixel_size_exp_threshold;
use crate::render::escape_time::{should_use_gmp_reference, should_use_perturbation};

/// Algorithme de rendu escape-time sélectionné pour la frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// f64 standard (zooms faibles, pas de perturbation).
    StandardF64,
    /// Perturbation + BLA (tier numérique dans [`WisdomPlan::tier`]).
    Perturbation,
    /// GMP complet par pixel (garde-fou / types non perturbés profonds).
    ReferenceGmp,
}

/// Device d'exécution du rendu. La sélection d'algorithme en dépend (le seuil
/// d'activation perturbation GPU f32 est ~1e5 vs ~1e12 CPU f64). Aujourd'hui le
/// device est un INPUT (le caller `--gpu`/GUI choisit) ; l'auto-sélection par
/// benchmark machine est le jalon G9.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

/// Variantes d'accélération du path perturbation CPU f64 — partie **statique**
/// (paramètres de frame + gates env) des prédicats de routage. Les conditions
/// dépendantes de l'orbite construite (réf compressée présente, `cycle_period`,
/// `phase_offset`, longueur) restent dans `delta.rs::{compressed_ref_route_
/// active, harmonic_route_active}`, qui composent CE prédicat — source unique,
/// pas de drift possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Variants {
    /// Lecture de la référence via le décompresseur à waypoints
    /// (`FRACTALL_COMPRESS_REF=1`, G8.2 phase 2).
    pub compression: bool,
    /// Table Harmonic LA au lieu de la BLA mat2 (`FRACTALL_HARMONIC_LA`,
    /// prototype G8.2 — routage wisdom = jalon G9.3).
    pub harmonic: Option<HarmonicVariant>,
}

/// Tier numérique de la boucle pixel perturbation. Analogue du type numérique
/// choisi par `wisdom_lookup` de F3 (double → floatexp → float128).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberTier {
    /// `Complex64` — 53 b mantisse, exposant f64 (~2^±1023). Le plus rapide.
    F64,
    /// `ComplexExp` — 53 b mantisse, exposant i32 (~2^±2^31). Deep zoom.
    Exp,
    /// `ComplexDDExp` double-double — ~106 b mantisse. Opt-in (sensibilité).
    Dd,
}

impl NumberTier {
    /// Label du path bytecode (aligné sur la sortie `[FRACTALL] path=…`).
    pub fn path_label(self) -> &'static str {
        match self {
            NumberTier::F64 => "bytecode_f64",
            NumberTier::Exp => "bytecode_exp",
            NumberTier::Dd => "bytecode_dd",
        }
    }

    /// Bits de mantisse effectifs du tier (viabilité F3 : `required_precision < M`).
    pub fn mantissa_bits(self) -> u32 {
        match self {
            NumberTier::F64 | NumberTier::Exp => 53,
            NumberTier::Dd => 106,
        }
    }
}

/// Plan complet inspectable pour une frame : algorithme + tier + les grandeurs
/// F3 qui les justifient. Produit par [`plan`]. Purement descriptif — n'altère
/// aucun rendu (les call-sites consomment [`number_tier`]/[`dd_requested`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WisdomPlan {
    /// Device d'exécution (INPUT du plan aujourd'hui, cf. [`Device`]).
    pub device: Device,
    pub algorithm: Algorithm,
    /// Tier numérique perturbation, `None` si le path bytecode ne s'applique pas
    /// (type non compilable, multi-phase, `--no-bytecode`) ou hors perturbation.
    pub tier: Option<NumberTier>,
    /// Variantes d'accélération actives (partie statique, cf. [`Variants`]).
    pub variants: Variants,
    /// Débit effectif benché sur CETTE machine pour la technique du plan
    /// (iters/s, `fractall-cli --wisdom-bench`, G9.2). `None` si non mesuré —
    /// purement informatif aujourd'hui ; l'arbitrage device (G9.5) le
    /// consommera pour départager les techniques VIABLES, modèle F3
    /// `wisdom_lookup` (max vitesse).
    pub bench_iters_per_sec: Option<f64>,
    /// `pixel_spacing` effectif en espace-c (HP-aware ; peut être dénormal).
    pub pixel_size: f64,
    /// `~log2(zoom)` = exposant binaire requis (magnitude). `None` si dégénéré.
    pub required_exponent: Option<f64>,
    /// Mantisse requise F3 (frame centrée ≈ `log2(hypot(w,h))`).
    pub required_precision: u32,
    /// Précision GMP de l'orbite référence (`compute_perturbation_precision_bits`).
    pub reference_precision_bits: u32,
}

/// Le tier dd est-il **demandé** ? (`use_dd_tier` + Mandelbrot escape-time).
///
/// Prédicat statique partagé par [`number_tier`] et le dispatch d'exécution
/// (`try_bytecode_unified_path`). L'exécution effective exige EN PLUS que
/// l'orbite ait la référence dd calculée (`ReferenceOrbit::has_dd`) ; sinon on
/// retombe sur le split exp/f64 par `pixel_size`.
#[inline]
pub fn dd_requested(params: &FractalParams) -> bool {
    params.use_dd_tier && matches!(params.fractal_type, FractalType::Mandelbrot)
}

/// Faut-il le tier ComplexExp plutôt que f64 pour ce `pixel_size` ?
///
/// Seuil `pixel_size < 1e-280` (`PIXEL_SIZE_EXP_THRESHOLD`, override
/// `FRACTALL_EXP_THRESHOLD`). Correspond à la borne de **viabilité d'exposant**
/// F3 pour f64 : en deçà, `δ ~ pixel_size` devient sous-normal (< 2.2e-308) et
/// la mantisse f64 se dégrade (casse observée à safari 1e307). Empiriquement
/// plus conservateur que la borne théorique F3 (~2^1008 ≈ zoom 1e301) — on garde
/// le seuil calibré sur corpus.
#[inline]
pub fn wants_exp(pixel_size: f64) -> bool {
    pixel_size < pixel_size_exp_threshold()
}

/// Tier numérique du path bytecode pour cette frame, `None` si le path bytecode
/// ne s'applique pas (mêmes conditions que l'ancien `bytecode_path_label` :
/// `use_bytecode_engine`, formule compilable, single-phase, `pixel_size > 0`).
///
/// **Source unique** de la sélection de tier. Ordre (identique à F3
/// double→floatexp→float128, hoisté sur dd) : dd demandé > exp (deep) > f64.
pub fn number_tier(params: &FractalParams) -> Option<NumberTier> {
    if !params.use_bytecode_engine {
        return None;
    }
    let formula =
        crate::fractal::bytecode::compile_formula(params.fractal_type, params.multibrot_power)?;
    if formula.phases.len() != 1 {
        // Multi-phase pas encore porté sur le pixel_loop unifié.
        return None;
    }
    let pixel_size = effective_pixel_size(params);
    if pixel_size <= 0.0 {
        return None;
    }
    // Tier dd (~106 b) opt-in : couvre tout le range (le dd loop tourne sans
    // gating de zoom), donc hoisté avant le split exp/f64.
    if dd_requested(params) {
        return Some(NumberTier::Dd);
    }
    if wants_exp(pixel_size) {
        Some(NumberTier::Exp)
    } else {
        Some(NumberTier::F64)
    }
}

/// Mantisse requise F3-style pour une frame centrée : `ceil(log2(hypot(w,h)))`.
/// L'offset centre→réf est nul (l'orbite est construite au centre de vue), donc
/// le terme dominant est la diagonale image. Sert au log `[WISDOM]` et de point
/// d'accroche futur pour l'escalade dd (`required_precision > tier.mantissa`).
fn required_precision_bits(params: &FractalParams) -> u32 {
    let diag = ((params.width as f64).powi(2) + (params.height as f64).powi(2)).sqrt();
    if diag <= 1.0 {
        return 1;
    }
    diag.log2().ceil() as u32
}

/// La famille escape-time que le dispatcher unique route en perturbation.
/// Multibrot est volontairement exclu : `should_use_perturbation` l'accepte
/// mais le dispatcher CLI/GUI ne l'a jamais routé en perturbation (bytecode
/// f64/GMP, cf. table des types CLAUDE.md).
#[inline]
pub fn perturbation_family(t: FractalType) -> bool {
    matches!(
        t,
        FractalType::Mandelbrot
            | FractalType::Julia
            | FractalType::BurningShip
            | FractalType::Tricorn
    )
}

/// Sélectionne l'algorithme escape-time pour un device donné. **Source unique
/// consommée par les dispatchers** (G9.1) : `render/escape_time.rs` (CPU),
/// `GpuRenderer::render_dispatch` (GPU, seuil perturbation f32 ~1e5) et les
/// labels/passes GUI. Ordre Auto : perturbation d'abord, puis GMP, sinon f64.
/// Les types spéciaux (densité, vectoriel, Lyapunov…) ne passent pas par ici.
pub fn select_algorithm(params: &FractalParams, device: Device) -> Algorithm {
    use crate::fractal::types::AlgorithmMode;
    // Perturbation viable : famille escape-time supportée + plan Mu (le delta
    // ne commute pas avec les transformations de plan). Miroir exact de l'ancien
    // inline du dispatcher CPU.
    let perturbation_viable = perturbation_family(params.fractal_type)
        && params.plane_transform == PlaneTransform::Mu;
    match params.algorithm_mode {
        AlgorithmMode::ReferenceGmp => Algorithm::ReferenceGmp,
        AlgorithmMode::StandardF64 => Algorithm::StandardF64,
        AlgorithmMode::Perturbation => {
            if perturbation_viable {
                Algorithm::Perturbation
            } else if params.use_gmp {
                Algorithm::ReferenceGmp
            } else {
                Algorithm::StandardF64
            }
        }
        AlgorithmMode::Auto => {
            if perturbation_viable && should_use_perturbation(params, device == Device::Gpu) {
                Algorithm::Perturbation
            } else if should_use_gmp_reference(params) {
                Algorithm::ReferenceGmp
            } else {
                Algorithm::StandardF64
            }
        }
    }
}

/// Variantes d'accélération actives pour cette frame — partie **statique** des
/// prédicats de routage (gates env + type + tier + flags de coloring). Les
/// conditions dépendantes de l'orbite (réf compressée construite,
/// `cycle_period`, `phase_offset`, longueur) sont composées par-dessus dans
/// `delta.rs`. Fast-path Mandelbrot f64 pur uniquement (cf. doc des prédicats).
/// Éligibilité commune des variantes fast-path : Mandelbrot f64 pur (pas de
/// distance/interior/orbit_traps ; tier F64 couvre `use_bytecode_engine`,
/// formule single-phase, `pixel_size > 0`). ⚠️ Coûteux (`number_tier`
/// recompile la formule, `effective_pixel_size` peut parser les spans HP) —
/// les prédicats PER-PIXEL doivent court-circuiter AVANT d'y arriver (leçon
/// G9.1 : geomean quick 0.223→0.269 sans short-circuit).
fn variant_eligible(params: &FractalParams) -> bool {
    matches!(params.fractal_type, FractalType::Mandelbrot)
        && !params.enable_orbit_traps
        && !params.enable_distance_estimation
        && !params.enable_interior_detection
        && number_tier(params) == Some(NumberTier::F64)
}

/// La variante COMPRESSION est-elle candidate ? Prédicat PER-PIXEL-safe : le
/// gate env (`FRACTALL_COMPRESS_REF`, off par défaut) court-circuite avant
/// l'éligibilité coûteuse.
pub fn compression_active(params: &FractalParams) -> bool {
    compress_enabled() && variant_eligible(params)
}

/// La variante HARMONIC est-elle candidate, et laquelle ? `None` = BLA.
/// **PER-RENDER uniquement** (build de l'entrée cache + label) : depuis le
/// mode Auto par défaut (G9.3), ce prédicat ne court-circuite plus — le path
/// per-pixel route sur la PRÉSENCE de la table dans l'entrée cache, pas sur ce
/// prédicat. Conditions : mode ≠ Off, compression INACTIVE (la table et la
/// queue directe lisent `z_ref_f64` PLEIN, que le strip compressé libère —
/// avertissement une-fois si un gate FORCÉ entre en conflit), Mandelbrot
/// seed=0 (la table suppose `Z[0]=0`, orbite critique), éligibilité commune.
/// La décision FINALE Auto (probe `period0`) est prise au build de l'entrée.
pub fn harmonic_candidate(params: &FractalParams) -> Option<HarmonicVariant> {
    use crate::fractal::bytecode::harmonic_mla::HarmonicMode;
    let mode = crate::fractal::bytecode::harmonic_mla::harmonic_mode();
    if mode == HarmonicMode::Off {
        return None;
    }
    if compress_enabled() {
        if matches!(mode, HarmonicMode::Forced(_)) {
            static WARNED: OnceLock<()> = OnceLock::new();
            WARNED.get_or_init(|| {
                eprintln!(
                    "[HARMONIC] FRACTALL_COMPRESS_REF actif — Harmonic MLA sauté \
                     (la table et la queue directe lisent z_ref_f64 plein)"
                );
            });
        }
        return None;
    }
    if params.seed.re != 0.0 || params.seed.im != 0.0 {
        return None;
    }
    if !variant_eligible(params) {
        return None;
    }
    harmonic_variant()
}

/// Politique de routage AUTO harmonic (G9.3) : router quand la **période de
/// l'atome dominant** de la référence est courte. Calibration corpus 256²
/// (A/B pixels BLA vs LLA, 3 runs, 2026-07-15) : GAGNE pour period0 ≤ 78
/// (flake p28 5.9×, test3 p28 5.7×, glitch_test_5 p7 5.8×, mitosis p24 3.7×,
/// super_dense p9 **1.74× sur orbite 695 k** — la longueur d'orbite est HORS
/// de cause, seul period0 discrimine). Seuil 100 = milieu de la zone morte
/// mesurée [79, 111]. `period0 == 0` (pas de dip) = jamais routé.
///
/// **Le seuil est une frontière de CORRECTION, pas seulement de vitesse**
/// (adjugé vs GMP pur, 2026-07-16, /improve). L'ancienne borne haute était
/// posée sur un A/B de vitesse (e50 +34 %, e113 +13 %, dragon +59 % pour LLA) ;
/// le **fix epsilon 2⁻⁵³** (2026-07-15) a ralenti la BLA f64 ~4× et INVERSÉ ce
/// classement (LLA redevient plus rapide : e50 3.48×, e113 1.81×, dragon 1.14×
/// via l'axe `wisdom-optimality`). Mais l'adjudication GMP tranche : à ces
/// profondeurs long-période **la LLA est faster-but-WRONG**. À 96² vs GMP pur :
/// e50 → BLA **pixel-exact** (max_diff=0, div_ratio=0) alors que LLA **FAIL**
/// (max_diff=418, p99=53, div_ratio=**0.036** = 3.6 % de pixels systématiquement
/// faux) ; e113 → BLA div_ratio 0.00022 vs LLA 0.00119 (~5× plus divergent).
/// Router LLA au-delà de 100 échangerait donc de la vitesse contre de la
/// **correction** — refusé par le critère G9. Le seuil reste robuste aux futurs
/// changements de coût de la BLA (il n'est plus ancré sur son débit relatif).
pub const HARMONIC_AUTO_PERIOD0_MAX: u32 = 100;

pub fn route_harmonic_auto(period0: u32) -> bool {
    (1..=HARMONIC_AUTO_PERIOD0_MAX).contains(&period0)
}

pub fn variants(params: &FractalParams) -> Variants {
    Variants {
        compression: compression_active(params),
        harmonic: harmonic_candidate(params),
    }
}

/// Calcule le [`WisdomPlan`] complet pour une frame CPU (device par défaut des
/// paths de rendu actuels). Cf. [`plan_for`].
pub fn plan(params: &FractalParams) -> WisdomPlan {
    plan_for(params, Device::Cpu)
}

/// Calcule le [`WisdomPlan`] complet pour une frame sur un device donné —
/// device, algorithme, tier, variantes et les grandeurs F3 (exposant/précision
/// requis, précision GMP orbite). Descriptif : aucun effet de bord sur le rendu
/// (les dispatchers consomment [`select_algorithm`]/[`number_tier`]/[`variants`]).
pub fn plan_for(params: &FractalParams, device: Device) -> WisdomPlan {
    let algorithm = select_algorithm(params, device);
    // Tier et variantes n'ont de sens que sur le path perturbation CPU (le
    // kernel GPU est f32, sans compression ni harmonic).
    let (tier, variants) = if algorithm == Algorithm::Perturbation && device == Device::Cpu {
        (number_tier(params), variants(params))
    } else {
        (None, Variants::default())
    };
    WisdomPlan {
        device,
        algorithm,
        tier,
        variants,
        bench_iters_per_sec: crate::fractal::wisdom_bench::lookup_iters_per_sec(
            device, algorithm, tier,
        ),
        pixel_size: effective_pixel_size(params),
        required_exponent: log2_zoom(params),
        required_precision: required_precision_bits(params),
        reference_precision_bits: compute_perturbation_precision_bits(params),
    }
}

/// Imprime le plan wisdom sur stderr si `FRACTALL_WISDOM=1` (diagnostic ; ne
/// remplace pas la ligne `[FRACTALL]` finale). Appelé sur le path perturbation.
pub fn log_plan_if_enabled(params: &FractalParams) {
    if !matches!(std::env::var("FRACTALL_WISDOM").as_deref(), Ok("1" | "true")) {
        return;
    }
    let p = plan(params);
    let tier = p.tier.map(|t| t.path_label()).unwrap_or("legacy");
    let variants = match (p.variants.compression, p.variants.harmonic) {
        (true, _) => "compress",
        (false, Some(HarmonicVariant::Lla)) => "harmonic_lla",
        (false, Some(HarmonicVariant::Mla)) => "harmonic_mla",
        (false, None) => "bla",
    };
    let bench = p
        .bench_iters_per_sec
        .map(|s| format!("{:.2e}", s))
        .unwrap_or_else(|| "-".into());
    eprintln!(
        "[WISDOM] type={:?} device={:?} algo={:?} tier={} variants={} bench={bench} req_exp={} req_prec={}b tier_mantissa={}b ref_prec={}b pixel_size={:.3e}",
        params.fractal_type,
        p.device,
        p.algorithm,
        tier,
        variants,
        p.required_exponent.map(|e| format!("{:.1}", e)).unwrap_or_else(|| "-".into()),
        p.required_precision,
        p.tier.map(|t| t.mantissa_bits()).unwrap_or(0),
        p.reference_precision_bits,
        p.pixel_size,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::default_params_for_type;
    use crate::fractal::types::AlgorithmMode;

    /// Frame Mandelbrot centrée à un zoom donné (span = 4/zoom sur les deux axes).
    fn frame(zoom: f64) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 256, 256);
        p.center_x = -0.75;
        p.center_y = 0.0;
        p.span_x = 4.0 / zoom;
        p.span_y = 4.0 / zoom;
        p.algorithm_mode = AlgorithmMode::Auto;
        p
    }

    #[test]
    fn tier_matches_legacy_thresholds_across_zoom_sweep() {
        // La sélection de tier via `number_tier` doit reproduire EXACTEMENT
        // l'ancienne logique dupliquée : dd demandé > exp (pixel<1e-280) > f64.
        for exp in [2, 6, 12, 13, 20, 50, 100, 200, 279] {
            let z = 10f64.powi(exp);
            let p = frame(z);
            let pixel_size = effective_pixel_size(&p);
            let expected = if pixel_size < pixel_size_exp_threshold() {
                NumberTier::Exp
            } else {
                NumberTier::F64
            };
            assert_eq!(
                number_tier(&p),
                Some(expected),
                "zoom 1e{exp} pixel_size={pixel_size:e}"
            );
        }
    }

    #[test]
    fn dd_requested_hoists_over_exp_and_f64() {
        // `use_dd_tier` sur Mandelbrot force le tier dd à tout zoom (hoisté).
        for exp in [8, 12, 50] {
            let mut p = frame(10f64.powi(exp));
            p.use_dd_tier = true;
            assert_eq!(number_tier(&p), Some(NumberTier::Dd), "zoom 1e{exp}");
        }
        // dd non réclamé pour un type non-Mandelbrot même si use_dd_tier.
        let mut p = frame(1e8);
        p.fractal_type = FractalType::BurningShip;
        p.use_dd_tier = true;
        assert!(!dd_requested(&p));
        assert_eq!(number_tier(&p), Some(NumberTier::F64));
    }

    #[test]
    fn no_bytecode_engine_yields_no_tier() {
        let mut p = frame(1e20);
        p.use_bytecode_engine = false;
        assert_eq!(number_tier(&p), None);
    }

    #[test]
    fn algorithm_follows_perturbation_gate() {
        // Shallow → f64 ; deep (> ~1e12) → perturbation.
        assert_eq!(plan(&frame(1e3)).algorithm, Algorithm::StandardF64);
        assert_eq!(plan(&frame(1e14)).algorithm, Algorithm::Perturbation);
        // Forcé.
        let mut p = frame(1e3);
        p.algorithm_mode = AlgorithmMode::ReferenceGmp;
        assert_eq!(plan(&p).algorithm, Algorithm::ReferenceGmp);
    }

    #[test]
    fn gpu_device_lowers_perturbation_threshold() {
        // Même frame 1e8 : CPU (seuil ~1e12) → f64 ; GPU f32 (seuil ~1e5) →
        // perturbation. C'est l'ancien `use_perturbation` inline de
        // `render_dispatch`, désormais servi par la même fonction wisdom.
        let p = frame(1e8);
        assert_eq!(select_algorithm(&p, Device::Cpu), Algorithm::StandardF64);
        assert_eq!(select_algorithm(&p, Device::Gpu), Algorithm::Perturbation);
        assert_eq!(plan_for(&p, Device::Gpu).device, Device::Gpu);
        // Le tier/variantes ne s'appliquent qu'au path perturbation CPU.
        assert_eq!(plan_for(&p, Device::Gpu).tier, None);
    }

    #[test]
    fn forced_perturbation_falls_back_outside_family() {
        // Mode Perturbation forcé sur un type hors famille escape-time
        // perturbation → fallback f64/GMP (miroir du dispatcher historique).
        let mut p = frame(1e14);
        p.fractal_type = FractalType::Newton;
        p.algorithm_mode = AlgorithmMode::Perturbation;
        assert_eq!(select_algorithm(&p, Device::Cpu), Algorithm::StandardF64);
        p.use_gmp = true;
        assert_eq!(select_algorithm(&p, Device::Cpu), Algorithm::ReferenceGmp);
    }

    #[test]
    fn variants_default_compression_off_harmonic_auto_candidate() {
        // Sans env : compression OFF (gate opt-in) ; harmonic CANDIDAT LLA
        // (mode Auto par défaut, G9.3) sur une frame Mandelbrot f64 pure —
        // la décision finale (probe period0) est prise au build de l'entrée
        // cache, candidat ≠ routé.
        let p = frame(1e14);
        let v = variants(&p);
        assert!(!v.compression);
        assert_eq!(v.harmonic, Some(HarmonicVariant::Lla));
        // Hors éligibilité (Julia = seed ≠ 0 attendu par la table) : aucun
        // candidat même en Auto.
        let mut julia = frame(1e14);
        julia.fractal_type = FractalType::Julia;
        julia.seed.re = -0.8;
        julia.seed.im = 0.156;
        assert_eq!(variants(&julia).harmonic, None);
        // Flags de coloring impurs : pas de variante.
        let mut traps = frame(1e14);
        traps.enable_distance_estimation = true;
        assert_eq!(variants(&traps), Variants::default());
    }

    #[test]
    fn route_harmonic_auto_calibrated_thresholds() {
        // Bornes de la politique (calibration corpus 2026-07-15) : period0=0
        // (pas de dip) jamais routé ; gagnants mesurés ≤ 78 routés ; perdants
        // ≥ 112 refusés — frontière de CORRECTION adjugée vs GMP (2026-07-16) :
        // au-delà, la LLA est faster-but-wrong (e50 LLA div_ratio 0.036 / FAIL
        // vs BLA pixel-exact). NE PAS relever pour capter le speedup LLA post
        // fix-epsilon : ce serait de la vitesse au prix de la correction (G9).
        assert!(!route_harmonic_auto(0));
        assert!(route_harmonic_auto(1));
        assert!(route_harmonic_auto(9)); // super_dense (orbite 695 k, 1.74×)
        assert!(route_harmonic_auto(78)); // dinosaur_fossils (1.2×)
        assert!(route_harmonic_auto(HARMONIC_AUTO_PERIOD0_MAX));
        assert!(!route_harmonic_auto(HARMONIC_AUTO_PERIOD0_MAX + 1));
        assert!(!route_harmonic_auto(112)); // e50 : LLA FAIL vs GMP (div 0.036)
        assert!(!route_harmonic_auto(3164)); // dragon : period0 ≫ 100
    }

    #[test]
    fn required_precision_is_small_for_centered_frames() {
        // La conséquence F3 clé : mantisse requise ≪ 53 b → f64 toujours viable
        // sur la mantisse ; l'escalade se fait sur l'exposant (exp), pas dd.
        let p = frame(1e50);
        let plan = plan(&p);
        assert!(
            plan.required_precision < 20,
            "required_precision={} devrait être ~log2(diag) ≪ 53",
            plan.required_precision
        );
    }
}
