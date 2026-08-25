use num_complex::Complex64;

use crate::fractal::{
    AlgorithmMode, ChannelParams, ColorParams, ColorSpace, EngineParams, FormulaParams,
    FractalParams, FractalType, OutColoringMode, PerturbationParams, PlaneTransform,
    SamplingParams,
};
use crate::fractal::lyapunov::{LyapunovConfig, LyapunovPreset};
use crate::fractal::orbit_traps::OrbitTrapType;

/// Construit des paramètres avec les valeurs par défaut du type,
/// en reprenant la logique de `fractal_definitions.c`.
/// Escape radius par défaut des fractales escape-time standard (famille
/// bytecode + perturbation : Mandelbrot, Julia, Burning Ship, Tricorn,
/// Celtic, Buffalo, Perp. Burning Ship, Multibrot + variantes Julia).
/// Aligné sur Fraktaler-3 `escape_radius = 625` (= 25², `param.h:41`) :
/// requis pour la parité N0/NF à la frontière d'évasion et rend le smooth
/// coloring (formule log-log, `palettes.rs`) nettement plus propre qu'à
/// ER=4. Les types à sémantique d'évasion particulière (Newton, Magnet,
/// Sin, Nova, Pickover, densité, vectoriel) gardent leur bailout propre.
const ESCAPE_TIME_BAILOUT: f64 = 25.0;

pub fn default_params_for_type(fractal_type: FractalType, width: u32, height: u32) -> FractalParams {
    // Valeurs communes
    let mut params = FractalParams {
        width,
        height,
        center_x: 0.0,
        center_y: 0.0,
        span_x: 0.0,
        span_y: 0.0,
        center_x_hp: None,
        center_y_hp: None,
        span_x_hp: None,
        span_y_hp: None,
        seed: Complex64::new(0.0, 0.0),
        iteration_max: 2500,
        bailout: 4.0,
        fractal_type,
        color: ColorParams {
            color_mode: 6, // SmoothPlasma (défaut dans le projet C)
            color_repeat: 40,
            color_offset: 0.0,
            color_space: ColorSpace::Rgb,
            out_coloring_mode: OutColoringMode::Smooth,
        },
        sampling: SamplingParams {
            jitter_scale: 0.0,
            aa_subpixel_offset: [0.0, 0.0],
            aa_jitter: None,
        },
        engine: EngineParams {
            algorithm_mode: AlgorithmMode::Auto,
            use_gmp: false,
            precision_bits: 256,
            // Activé par défaut depuis P3.1 Session E : path bytecode unifié
            // (BLA mat2 + delta-form + rebasing F3) remplace le path legacy
            // quand applicable.
            // Le path legacy reste actif comme fallback :
            //   - types non supportés par compile_formula (Newton, Phoenix,
            //     Magnet, Barnsley, Lyapunov, Mandelbulb, etc.)
            //   - pixel_size < 1e-13 (deep zoom GMP)
            //   - features avancées (distance estimation, interior detection,
            //     orbit traps)
            // Tu peux désactiver explicitement via --no-bytecode ou en passant
            // use_bytecode_engine = false depuis un loader TOML.
            use_bytecode_engine: true,
            use_dd_tier: false,
            find_nucleus: false,
        },
        perturbation: PerturbationParams {
            // Aligné Fraktaler-3 (`engine.cc:283`) : 1.0 / (1 << 24) ≈ 5.96e-8.
            bla_threshold: 1.0 / (1u64 << 24) as f64,
            bla_validity_scale: 1.0,
            glitch_tolerance: 1e-4,
            series_order: 2,
            series_threshold: 1e-6,
            series_error_tolerance: 1e-9,
            series_standalone: true,
            max_perturb_iterations: 1024,
            max_bla_steps: 1024,
            use_reference_precision_formula: true,
        },
        formula: FormulaParams {
            multibrot_power: 2.5,
            hybrid_phases: None,
            hybrid_opcodes: None,
            lyapunov_preset: LyapunovPreset::default(),
            lyapunov_sequence: Vec::new(),
        },
        // Les canaux annexes coûtent des dual-numbers ou l'orbite complète :
        // ils restent éteints tant qu'un mode de coloriage ne les réclame pas.
        channels: ChannelParams {
            enable_distance_estimation: false,
            enable_interior_detection: false,
            interior_threshold: 0.001,
            enable_orbit_traps: false,
            orbit_trap_type: OrbitTrapType::Point,
        },
        plane_transform: PlaneTransform::Mu,
        rotation: 0.0,
        transform_k: None,
    };

    match fractal_type {
        FractalType::VonKoch => {
            // Von Koch - flocon de neige (vectoriel)
            params.center_x = 0.5;
            params.center_y = 0.5;
            params.span_x = 1.0;
            params.span_y = 1.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 8; // Profondeur de récursion max
        }
        FractalType::Dragon => {
            // Courbe du dragon (vectoriel)
            params.center_x = 0.5;
            params.center_y = 0.5;
            params.span_x = 1.0;
            params.span_y = 1.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 20; // Profondeur de récursion max
        }
        FractalType::Mandelbrot => {
            // Mendelbrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::Julia => {
            // Julia_def: xmin=-2.0, xmax=2.0, ymin=-1.5, ymax=1.5
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::JuliaSin => {
            // JuliaSin_def: xmin=-PI, xmax=PI, ymin=-2.0, ymax=2.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 2.0 * std::f64::consts::PI;
            params.span_y = 4.0;
            params.seed = Complex64::new(1.0, 0.1);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::MandelbrotSin => {
            // MandelbrotSin: même vue que Julia Sin, z_0 = seed
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 2.0 * std::f64::consts::PI;
            params.span_y = 4.0;
            params.seed = Complex64::new(1.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Newton => {
            // Newton_def: xmin=-3.0, xmax=3.0, ymin=-2.0, ymax=2.0
            params.seed = Complex64::new(8.0, 0.0);
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 6.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 1000;
        }
        FractalType::Phoenix => {
            // Phoenix_def: xmin=-2.0, xmax=2.0, ymin=-1.5, ymax=1.5
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::BarnsleyJulia => {
            // Barnsley1j_def: xmin=-4.0, xmax=4.0, ymin=-3.0, ymax=3.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 8.0;
            params.span_y = 6.0;
            params.seed = Complex64::new(1.1, 0.6);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::BarnsleyMandelbrot => {
            // Barnsley1m_def: xmin=-3.0, xmax=3.0, ymin=-2.0, ymax=2.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 6.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::MagnetJulia => {
            // Magnet1j_def: xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0
            params.seed = Complex64::new(1.625458, -0.306159);
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::MagnetMandelbrot => {
            // Magnet1m: centré sur la structure principale
            params.center_x = 0.7;
            params.center_y = 0.0;
            params.span_x = 5.0;
            params.span_y = 4.0;
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::BurningShip => {
            // BurningShip_def: xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::Buffalo => {
            // Buffalo_def: xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::Tricorn => {
            // Tricorn_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::Mandelbulb => {
            // Mandelbulb_def: xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
        }
        FractalType::Buddhabrot => {
            // Buddhabrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 220;
            params.color.color_repeat = 1; // densité: 1 par défaut, max 8
        }
        FractalType::Lyapunov => {
            // Lyapunov_def - Zircon City par défaut
            apply_lyapunov_preset(&mut params, LyapunovPreset::ZirconCity);
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2000;
        }
        FractalType::PerpendicularBurningShip => {
            // PerpendicularBurningShip_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::Celtic => {
            // Celtic_def: xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::AlphaMandelbrot => {
            // AlphaMandelbrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2000;
        }
        FractalType::PickoverStalks => {
            // PickoverStalks_def: xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 100.0;
            params.iteration_max = 1000;
            params.color.color_repeat = 2;
        }
        FractalType::Nova => {
            // Nova_def: xmin=-3.0, xmax=3.0, ymin=-2.0, ymax=2.0
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 6.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 20.0;
            params.iteration_max = 500;
        }
        FractalType::Multibrot => {
            // Multibrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::BurningShipJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::TricornJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::CelticJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 3.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::BuffaloJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 4.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::MultibrotJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::PerpendicularBurningShipJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = ESCAPE_TIME_BAILOUT;
            params.iteration_max = 2500;
        }
        FractalType::AlphaMandelbrotJulia => {
            params.center_x = 0.0;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.36228, -0.0777);
            params.bailout = 4.0;
            params.iteration_max = 2000;
        }
        FractalType::Nebulabrot => {
            // Nebulabrot_def: xmin=-2.5, xmax=1.5, ymin=-1.5, ymax=1.5
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 2500;
            params.color.color_repeat = 1; // densité: 1 par défaut, max 8
        }
        FractalType::AntiBuddhabrot => {
            // Anti-Buddhabrot: mêmes bornes que Buddhabrot, iterations plus élevées
            params.center_x = -0.5;
            params.center_y = 0.0;
            params.span_x = 4.0;
            params.span_y = 3.0;
            params.seed = Complex64::new(0.0, 0.0);
            params.bailout = 4.0;
            params.iteration_max = 500;
            params.color.color_repeat = 1; // densité: 1 par défaut, max 8
        }
    }

    params
}

/// Applique un preset Lyapunov aux paramètres.
/// Met à jour les bornes du domaine et la séquence.
/// Paramètres par défaut de `new_type`, en CONSERVANT les préférences de
/// l'utilisateur. Frontière EXPLICITE, écrite groupe par groupe :
///
/// - le **type** définit la formule, la géométrie, le bailout et les
///   itérations — c'est tout l'intérêt de repartir de ses défauts ;
/// - l'**utilisateur** possède la couleur, l'échantillonnage, le moteur, les
///   réglages de perturbation et les canaux qu'il a demandés.
///
/// Un champ ajouté à un groupe suit donc automatiquement le bon côté de la
/// frontière. La liste blanche de six champs qu'elle remplace perdait
/// silencieusement `color_space`, `color_offset`, `jitter_scale`,
/// `use_dd_tier` et huit des dix réglages de perturbation à CHAQUE changement
/// de type.
pub fn params_for_type_keeping_preferences(
    previous: &FractalParams,
    new_type: FractalType,
    width: u32,
    height: u32,
) -> FractalParams {
    let mut params = default_params_for_type(new_type, width, height);

    // Préférences transportées, groupe entier.
    params.color = previous.color.clone();
    params.sampling = previous.sampling.clone();
    params.engine = previous.engine.clone();
    params.perturbation = previous.perturbation.clone();
    params.channels = previous.channels.clone();

    // `formula` n'est PAS transportée : c'est le type qui la définit (une
    // séquence hybride ou des opcodes hérités décriraient une autre fractale
    // que celle demandée).

    // État de rendu transitoire : il n'appartient pas à la configuration et ne
    // traverse jamais un changement de type (sa vraie place est le
    // `RenderRequest`, cf. TODO).
    params.sampling.aa_subpixel_offset = [0.0, 0.0];
    params.sampling.aa_jitter = None;

    // Le type change ⇒ l'arbitrage algorithme est refait : une perturbation
    // forcée sur un type qui ne la supporte pas, ou un GMP hérité d'un zoom
    // profond, n'a aucun sens sur la vue par défaut du nouveau type.
    params.engine.algorithm_mode = AlgorithmMode::Auto;

    // Densité : le gradient repart à 1 (une répétition héritée d'un
    // escape-time rend l'accumulation illisible).
    if matches!(
        new_type,
        FractalType::Buddhabrot | FractalType::Nebulabrot | FractalType::AntiBuddhabrot
    ) {
        params.color.color_repeat = 1;
    }

    params
}


pub fn apply_lyapunov_preset(params: &mut FractalParams, preset: LyapunovPreset) {
    let config = LyapunovConfig::from_preset(preset);
    params.formula.lyapunov_preset = preset;
    params.formula.lyapunov_sequence = config.sequence;
    params.center_x = (config.xmin + config.xmax) * 0.5;
    params.center_y = (config.ymin + config.ymax) * 0.5;
    params.span_x = config.xmax - config.xmin;
    params.span_y = config.ymax - config.ymin;
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{ColorSpace, OutColoringMode};

    /// Params avec des préférences utilisateur non triviales dans CHAQUE
    /// groupe transporté.
    fn opinionated() -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 320, 240);
        p.color.color_mode = 11;
        p.color.color_repeat = 73;
        p.color.color_space = ColorSpace::Lch;
        p.color.color_offset = 0.375;
        p.color.out_coloring_mode = OutColoringMode::Biomorphs;
        p.sampling.jitter_scale = 0.625;
        p.engine.precision_bits = 1024;
        p.engine.use_dd_tier = true;
        p.engine.use_bytecode_engine = false;
        p.engine.find_nucleus = true;
        p.perturbation.bla_threshold = 1.25e-7;
        p.perturbation.series_order = 1;
        p.perturbation.max_bla_steps = 4096;
        p.channels.enable_interior_detection = true;
        p.channels.interior_threshold = 0.007;
        p
    }

    fn dump<T: serde::Serialize>(v: &T) -> String {
        serde_json::to_string(v).expect("groupe sérialisable")
    }

    /// Verrou STRUCTUREL du changement de type : les groupes de préférences
    /// sont transportés ENTIÈREMENT (comparaison sérialisée — un champ ajouté
    /// au groupe entre dans le verrou sans qu'on y touche). La liste blanche
    /// de six champs qu'il remplace remettait silencieusement à zéro
    /// `color_space`, `color_offset`, `jitter_scale`, `use_dd_tier` et huit
    /// des dix réglages de perturbation à CHAQUE changement de type.
    #[test]
    fn changing_type_carries_whole_preference_groups() {
        let before = opinionated();
        let after =
            params_for_type_keeping_preferences(&before, FractalType::BurningShip, 320, 240);

        assert_eq!(dump(&after.color), dump(&before.color), "couleur");
        assert_eq!(dump(&after.sampling), dump(&before.sampling), "échantillonnage");
        assert_eq!(
            dump(&after.perturbation),
            dump(&before.perturbation),
            "perturbation"
        );
        assert_eq!(dump(&after.channels), dump(&before.channels), "canaux");
        // Moteur : tout sauf l'arbitrage d'algorithme, refait au changement.
        let mut engine_ref = before.engine.clone();
        engine_ref.algorithm_mode = after.engine.algorithm_mode;
        assert_eq!(dump(&after.engine), dump(&engine_ref), "moteur");
        assert_eq!(after.engine.algorithm_mode, AlgorithmMode::Auto);
    }

    /// L'autre moitié de la frontière : ce que le TYPE définit repart de ses
    /// défauts — formule, géométrie, bailout, itérations.
    #[test]
    fn changing_type_resets_what_the_type_defines() {
        let mut before = opinionated();
        before.formula.hybrid_phases =
            Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
        before.formula.hybrid_opcodes = Some("sqr rot{30} add".into());
        before.center_x = -0.743_643_887;
        before.span_x = 1e-9;
        before.span_y = 1e-9;

        let after = params_for_type_keeping_preferences(&before, FractalType::Tricorn, 320, 240);
        let fresh = default_params_for_type(FractalType::Tricorn, 320, 240);

        assert_eq!(dump(&after.formula), dump(&fresh.formula), "formule");
        assert_eq!(after.center_x, fresh.center_x);
        assert_eq!(after.span_x, fresh.span_x);
        assert_eq!(after.bailout, fresh.bailout);
        assert_eq!(after.iteration_max, fresh.iteration_max);
        assert_eq!(after.fractal_type, FractalType::Tricorn);
    }

    /// Deux exceptions explicites : l'état AA transitoire ne traverse pas un
    /// changement de type, et les types de densité repartent à un gradient de 1.
    #[test]
    fn changing_type_drops_transient_state_and_resets_density_repeat() {
        let mut before = opinionated();
        before.sampling.aa_subpixel_offset = [0.25, -0.125];
        before.sampling.aa_jitter = Some((7, 0.5));

        let after =
            params_for_type_keeping_preferences(&before, FractalType::Buddhabrot, 320, 240);
        assert_eq!(after.sampling.aa_subpixel_offset, [0.0, 0.0]);
        assert!(after.sampling.aa_jitter.is_none());
        assert_eq!(after.sampling.jitter_scale, before.sampling.jitter_scale);
        assert_eq!(after.color.color_repeat, 1, "densité : gradient à 1");
        assert_eq!(after.color.color_mode, before.color.color_mode, "palette gardée");
    }
}
