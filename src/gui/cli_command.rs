//! Commande `fractall-cli` reproduisant la vue courante de la GUI (bouton
//! « Copy CLI command »).
//!
//! Fonction PURE : les options sont émises EXACTEMENT comme `src/main.rs` les
//! lit (`--zoom` = 4 / span_x, `--plane` par identifiant numérique, noms
//! `cli_name()` des modes…), et seulement quand la valeur diffère de ce que le
//! CLI prendrait par défaut pour ce type — sauf type, centre, zoom, itérations,
//! taille et sortie, toujours présents (zoom omis pour le cadrage par défaut
//! rendu à pixels non carrés, que `--zoom` ne sait pas décrire). Ce que la ligne de commande ne sait pas
//! exprimer n'est jamais perdu en silence : listé dans `unexpressed` et ajouté
//! en commentaire shell à la fin de la commande.

use crate::fractal::bytecode::Formula;
use crate::fractal::lyapunov::LyapunovConfig;
use crate::fractal::{
    default_params_for_type, AlgorithmMode, ColorSpace, FractalParams, FractalType,
    OutColoringMode, PlaneTransform,
};
use crate::gui::coords::{self, Decimal};

/// Réglages de la GUI qui ne vivent pas dans `FractalParams` (anti-aliasing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CliExtras {
    pub aa_samples: u32,
    /// Amplitude du jitter AA (px) ; le CLI vaut 1.0 par défaut.
    pub aa_jitter_scale: f64,
    /// Centre émis tel quel (chaîne HP complète) au lieu d'être arrondi à la
    /// résolution pixel. Le bouton GUI arrondit (lisibilité : ~300 chiffres
    /// MPFR sinon) ; la commande embarquée dans un PNG ne le fait PAS — un
    /// centre qui diffère au-delà du pixel change la référence de
    /// perturbation, donc les pixels mal conditionnés (mesuré : 9 pixels sur
    /// 230 400 à 5e13, 0 avec le centre exact).
    pub exact_center: bool,
}

impl Default for CliExtras {
    fn default() -> Self {
        Self {
            aa_samples: 1,
            aa_jitter_scale: 1.0,
            exact_center: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CliCommand {
    /// Ligne de commande complète (commentaire final si `unexpressed` non vide).
    pub command: String,
    /// Réglages que le CLI ne peut pas reproduire (description en anglais).
    pub unexpressed: Vec<String>,
}

/// Nom de sortie proposé.
pub const OUTPUT_FILE: &str = "fractal.png";

/// Défauts CLI du moteur (`src/main.rs`) indépendants du type.
const CLI_DEFAULT_PALETTE: u8 = 6;
const CLI_DEFAULT_PRECISION_BITS: u32 = 256;
const CLI_DEFAULT_INTERIOR_THRESHOLD: f64 = 0.001;
const CLI_DEFAULT_JITTER_SCALE: f64 = 1.0;

/// Nom accepté par `FractalType::from_hybrid_name` (`--phases`).
fn hybrid_phase_name(t: FractalType) -> Option<&'static str> {
    Some(match t {
        FractalType::Mandelbrot => "mandelbrot",
        FractalType::Julia => "julia",
        FractalType::BurningShip => "burning_ship",
        FractalType::Tricorn => "tricorn",
        FractalType::Celtic => "celtic",
        FractalType::Buffalo => "buffalo",
        FractalType::PerpendicularBurningShip => "perpbs",
        FractalType::Multibrot => "multibrot",
        _ => return None,
    })
}

fn algorithm_cli_name(mode: AlgorithmMode) -> &'static str {
    match mode {
        AlgorithmMode::Auto => "auto",
        AlgorithmMode::StandardF64 => "f64",
        AlgorithmMode::Perturbation => "perturbation",
        AlgorithmMode::ReferenceGmp => "gmp",
    }
}

/// f64 relu à l'identique par `str::parse::<f64>` (clap), sans 300 zéros.
fn fmt_f64(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 || (1e-5..1e16).contains(&a) {
        format!("{v}")
    } else {
        format!("{v:e}")
    }
}

/// Quote shell (POSIX) si la valeur contient autre chose que des caractères sûrs.
fn shell_quote(value: &str) -> String {
    let safe = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._+-/=,:@%".contains(c));
    if safe {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

struct Args(Vec<String>);

impl Args {
    fn flag(&mut self, name: &str) {
        self.0.push(format!("--{name}"));
    }
    /// `--name value`, ou `--name=value` pour une valeur négative (toutes les
    /// options du CLI n'acceptent pas une valeur commençant par `-`).
    fn opt(&mut self, name: &str, value: &str) {
        let value = shell_quote(value);
        if value.starts_with('-') {
            self.0.push(format!("--{name}={value}"));
        } else {
            self.0.push(format!("--{name}"));
            self.0.push(value);
        }
    }
}

/// Centre émis : valeur HP exacte arrondie à la résolution pixel + 20 chiffres
/// de garde (écart ≤ 1e-20 pixel, invisible), pour éviter les ~300 chiffres
/// d'une chaîne MPFR 1024 bits à zoom faible.
fn cli_center_coordinate(value: &str, params: &FractalParams) -> String {
    let Some(decimal) = Decimal::parse(value) else {
        return value.trim().to_string();
    };
    let keep = coords::pixel_fraction_digits(params)
        .unwrap_or(coords::MIN_DISPLAY_FRACTION_DIGITS)
        + coords::CLI_GUARD_DIGITS;
    decimal.rounded(keep).to_canonical_string(20)
}

pub fn build_cli_command(params: &FractalParams, extras: CliExtras) -> CliCommand {
    let t = params.fractal_type;
    let (width, height) = (params.width.max(1), params.height.max(1));
    // Ce que le CLI aurait sans option pour ce type et cette taille.
    let base = default_params_for_type(t, width, height);
    let mut a = Args(vec!["fractall-cli".to_string()]);
    let mut unexpressed: Vec<String> = Vec::new();

    a.opt("type", &t.id().to_string());

    // Formule hybride / opcodes (prioritaires sur le type pour la formule).
    if let Some(phases) = &params.formula.hybrid_phases {
        let names: Option<Vec<&str>> = phases.iter().map(|&p| hybrid_phase_name(p)).collect();
        match names {
            Some(names) if !names.is_empty() => a.opt("phases", &names.join(",")),
            _ => unexpressed.push(format!("hybrid phases {phases:?}")),
        }
    }
    if let Some(ops) = &params.formula.hybrid_opcodes {
        a.opt("opcodes", ops);
    }

    // Vue : centre HP + zoom (span_x = 4 / zoom, span_y = span_x·h/w côté CLI).
    let (cx, cy) = coords::center_strings(params);
    if extras.exact_center {
        a.opt("center-x-hp", cx.trim());
        a.opt("center-y-hp", cy.trim());
    } else {
        a.opt("center-x-hp", &cli_center_coordinate(&cx, params));
        a.opt("center-y-hp", &cli_center_coordinate(&cy, params));
    }
    // `--zoom` impose des pixels carrés (span_y = span_x·h/w). Une vue qui ne
    // l'est pas — typiquement le cadrage par défaut du type rendu à un autre
    // rapport d'aspect, ce que fait le CLI sans `--zoom` — ne doit PAS le
    // recevoir : sans l'option, le CLI garde les spans par défaut tels quels.
    let (sx, sy) = (params.span_x, params.span_y);
    // Preset Lyapunov non défaut : `--lyapunov-preset` remplace la vue, cas
    // déjà signalé plus bas — pas de second avertissement ici.
    let preset_view = t == FractalType::Lyapunov
        && params.formula.lyapunov_preset != base.formula.lyapunov_preset;
    let square = preset_view
        || !(sx.is_normal() && sy.is_normal())
        || (sy * width as f64 - sx * height as f64).abs() <= 1e-9 * (sx * height as f64).abs();
    if square {
        a.opt("zoom", &coords::zoom_string(params));
    } else if sx != base.span_x || sy != base.span_y {
        a.opt("zoom", &coords::zoom_string(params));
        unexpressed.push(format!(
            "non-square pixels (span {} x {})",
            fmt_f64(sx),
            fmt_f64(sy)
        ));
    }
    a.opt("iterations", &params.iteration_max.to_string());

    // Couleur.
    if params.color.color_mode != CLI_DEFAULT_PALETTE {
        a.opt("palette", &params.color.color_mode.to_string());
    }
    if params.color.color_repeat != base.color.color_repeat {
        a.opt("color-repeat", &params.color.color_repeat.to_string());
    }
    if params.color.out_coloring_mode != OutColoringMode::Smooth {
        a.opt("outcoloring", params.color.out_coloring_mode.cli_name());
    }
    if params.color.color_space != ColorSpace::Rgb {
        unexpressed.push(format!("color space {:?}", params.color.color_space));
    }
    if params.color.color_offset != 0.0 {
        unexpressed.push(format!("color offset {}", fmt_f64(params.color.color_offset)));
    }

    // Formule.
    if params.formula.multibrot_power != base.formula.multibrot_power {
        if params.formula.multibrot_power > 0.0 {
            a.opt("multibrot-power", &fmt_f64(params.formula.multibrot_power));
        } else {
            unexpressed.push(format!(
                "multibrot power {}",
                fmt_f64(params.formula.multibrot_power)
            ));
        }
    }
    if params.seed != base.seed {
        if Formula::is_julia_for(t) {
            if params.seed.re != base.seed.re {
                a.opt("julia-re", &fmt_f64(params.seed.re));
            }
            if params.seed.im != base.seed.im {
                a.opt("julia-im", &fmt_f64(params.seed.im));
            }
        } else {
            unexpressed.push(format!(
                "orbit seed z0 = {} + {}i",
                fmt_f64(params.seed.re),
                fmt_f64(params.seed.im)
            ));
        }
    }
    if t == FractalType::Lyapunov {
        let preset = params.formula.lyapunov_preset;
        if preset != base.formula.lyapunov_preset {
            a.opt("lyapunov-preset", preset.cli_name());
            // main.rs applique le preset APRÈS la vue : il remplace le centre
            // et le span f64 par ceux du preset.
            unexpressed.push(format!(
                "the view: --lyapunov-preset {} resets the CLI view to the preset's own",
                preset.cli_name()
            ));
        }
        if params.formula.lyapunov_sequence != LyapunovConfig::from_preset(preset).sequence {
            unexpressed.push("custom Lyapunov sequence".to_string());
        }
    }
    if params.bailout != base.bailout {
        if params.bailout.is_finite() {
            a.opt("bailout", &fmt_f64(params.bailout));
        } else {
            unexpressed.push(format!("bailout {}", params.bailout));
        }
    }
    if params.plane_transform != PlaneTransform::Mu {
        a.opt("plane", &params.plane_transform.id().to_string());
    }
    if params.rotation != base.rotation {
        a.opt("rotation", &fmt_f64(params.rotation));
    }
    if params.transform_k.is_some() {
        unexpressed.push("transform matrix K (e.g. from the nucleus finder)".to_string());
    }

    // Moteur.
    let mode = params.engine.algorithm_mode;
    if mode != AlgorithmMode::Auto {
        a.opt("algorithm", algorithm_cli_name(mode));
    }
    match mode {
        // Le CLI force use_gmp selon le mode : rien à émettre.
        AlgorithmMode::ReferenceGmp => {}
        AlgorithmMode::StandardF64 => {
            if params.engine.use_gmp {
                unexpressed.push("GMP flag with the f64 algorithm".to_string());
            }
        }
        _ => {
            if params.engine.use_gmp {
                a.flag("gmp");
            }
        }
    }
    if params.engine.precision_bits != CLI_DEFAULT_PRECISION_BITS {
        if params.engine.precision_bits >= 64 {
            a.opt("precision-bits", &params.engine.precision_bits.to_string());
        } else {
            unexpressed.push(format!("precision {} bits", params.engine.precision_bits));
        }
    }
    if !params.engine.use_bytecode_engine {
        a.flag("no-bytecode");
    }
    if params.engine.use_dd_tier {
        a.flag("dd-tier");
    }
    if params.engine.find_nucleus {
        a.flag("find-nucleus");
    }

    // Perturbation (le CLI n'accepte que des valeurs > 0).
    let pert = &params.perturbation;
    let base_pert = &base.perturbation;
    for (name, value, default) in [
        ("bla-threshold", pert.bla_threshold, base_pert.bla_threshold),
        (
            "bla-validity-scale",
            pert.bla_validity_scale,
            base_pert.bla_validity_scale,
        ),
        ("glitch-tolerance", pert.glitch_tolerance, base_pert.glitch_tolerance),
    ] {
        if value != default {
            if value > 0.0 && value.is_finite() {
                a.opt(name, &fmt_f64(value));
            } else {
                unexpressed.push(format!("{name} {value}"));
            }
        }
    }
    // `max_perturb_iterations` / `max_bla_steps` : clampés à ≥ iteration_max
    // au rendu, donc sans effet observable — volontairement ignorés.
    if pert.series_order != base_pert.series_order
        || pert.series_threshold != base_pert.series_threshold
        || pert.series_error_tolerance != base_pert.series_error_tolerance
        || pert.series_standalone != base_pert.series_standalone
    {
        unexpressed.push("series approximation settings".to_string());
    }
    if pert.use_reference_precision_formula != base_pert.use_reference_precision_formula {
        unexpressed.push("reference precision formula switch".to_string());
    }

    // Canaux annexes. Le CLI pose lui-même les canaux requis par le mode de
    // colorisation (`ensure_required_channels`) ; on émet quand même le
    // drapeau explicite, sans effet de bord.
    if params.channels.enable_distance_estimation {
        a.flag("enable-distance-estimation");
    }
    if params.channels.enable_interior_detection {
        a.flag("enable-interior-detection");
    }
    if params.channels.interior_threshold != CLI_DEFAULT_INTERIOR_THRESHOLD {
        if params.channels.interior_threshold > 0.0 {
            a.opt("interior-threshold", &fmt_f64(params.channels.interior_threshold));
        } else {
            unexpressed.push(format!(
                "interior threshold {}",
                params.channels.interior_threshold
            ));
        }
    }
    let orbits_required = matches!(
        params.color.out_coloring_mode,
        OutColoringMode::OrbitTraps | OutColoringMode::Wings
    );
    if params.channels.enable_orbit_traps && !orbits_required {
        unexpressed.push("orbit-trap channel without an orbit-trap coloring".to_string());
    }
    if params.channels.orbit_trap_type != base.channels.orbit_trap_type {
        unexpressed.push(format!(
            "orbit trap shape {:?}",
            params.channels.orbit_trap_type
        ));
    }

    // Anti-aliasing (état GUI).
    if extras.aa_samples > 1 {
        a.opt("aa-samples", &extras.aa_samples.to_string());
        if extras.aa_jitter_scale != CLI_DEFAULT_JITTER_SCALE {
            a.opt("jitter-scale", &fmt_f64(extras.aa_jitter_scale));
        }
    }

    a.opt("width", &width.to_string());
    a.opt("height", &height.to_string());
    a.opt("output", OUTPUT_FILE);

    let mut command = a.0.join(" ");
    if !unexpressed.is_empty() {
        command.push_str("  # not expressible on the command line: ");
        command.push_str(&unexpressed.join("; "));
    }
    CliCommand {
        command,
        unexpressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{LyapunovPreset, ViewHp};

    fn gui_like(t: FractalType, w: u32, h: u32) -> FractalParams {
        // Comme la GUI : la vue HP (MPFR 1024 b) écrit les chaînes décimales.
        let mut p = default_params_for_type(t, w, h);
        let view = ViewHp::from_decimal_parts(
            &p.center_x.to_string(),
            &p.center_y.to_string(),
            &p.span_x.to_string(),
            &p.span_y.to_string(),
            w,
            h,
            1024,
        )
        .unwrap();
        view.write_to_params(&mut p);
        p
    }

    fn args(cmd: &str) -> Vec<&str> {
        cmd.split("  #").next().unwrap().split_whitespace().collect()
    }

    fn value_of<'a>(cmd: &'a str, name: &str) -> Option<&'a str> {
        let flag = format!("--{name}");
        let eq = format!("--{name}=");
        let a = args(cmd);
        for (i, tok) in a.iter().enumerate() {
            if *tok == flag {
                return a.get(i + 1).copied();
            }
            if let Some(v) = tok.strip_prefix(&eq) {
                return Some(v);
            }
        }
        None
    }

    /// Le CLI sans `--zoom` rend le cadrage par défaut du type tel quel, même
    /// à un rapport d'aspect qui donne des pixels non carrés. `--zoom` les
    /// rendrait carrés (autre image : 62 550 pixels sur 64 000 mesurés) —
    /// donc omis ; une vue non carrée autre que le défaut est signalée.
    #[test]
    fn non_square_default_view_omits_zoom() {
        let p = default_params_for_type(FractalType::Julia, 320, 200);
        assert_ne!(p.span_y * 320.0, p.span_x * 200.0, "précondition : pixels non carrés");
        let c = build_cli_command(&p, CliExtras::default());
        assert!(value_of(&c.command, "zoom").is_none(), "{}", c.command);
        assert!(c.unexpressed.is_empty(), "{:?}", c.unexpressed);

        let mut odd = p.clone();
        odd.span_x *= 0.5;
        let c = build_cli_command(&odd, CliExtras::default());
        assert!(value_of(&c.command, "zoom").is_some());
        assert!(c.unexpressed.iter().any(|u| u.contains("non-square")), "{:?}", c.unexpressed);

        // Centre exact (commande embarquée dans le PNG) : chaîne HP intacte.
        let mut hp = gui_like(FractalType::Mandelbrot, 800, 600);
        hp.center_x_hp = Some("0.0000065018695656308234292171447709786".into());
        let exact = CliExtras { exact_center: true, ..CliExtras::default() };
        assert_eq!(
            value_of(&build_cli_command(&hp, exact).command, "center-x-hp"),
            Some("0.0000065018695656308234292171447709786")
        );
    }

    #[test]
    fn default_mandelbrot_view_is_minimal() {
        let p = gui_like(FractalType::Mandelbrot, 800, 600);
        let c = build_cli_command(&p, CliExtras::default());
        assert!(c.unexpressed.is_empty(), "{:?}", c.unexpressed);
        assert_eq!(
            c.command,
            format!(
                "fractall-cli --type 3 --center-x-hp=-0.5 --center-y-hp 0 --zoom 1 \
                 --iterations {} --width 800 --height 600 --output fractal.png",
                p.iteration_max
            )
        );
    }

    #[test]
    fn deep_view_keeps_hp_centre_and_exact_zoom() {
        let mut p = gui_like(FractalType::Mandelbrot, 800, 600);
        let cx = format!("-1.8607199548737286{}", "13".repeat(40));
        let cy = format!("8.72880{}e-5", "47".repeat(40));
        let view = ViewHp::from_center_and_zoom(&cx, &cy, "3.1415e64", 800, 600, 1024).unwrap();
        view.write_to_params(&mut p);
        p.center_x_hp = Some(cx.clone());
        p.center_y_hp = Some(cy.clone());
        p.iteration_max = 70000;
        p.color.color_mode = 8;
        let c = build_cli_command(&p, CliExtras::default());
        let x = value_of(&c.command, "center-x-hp").unwrap();
        let y = value_of(&c.command, "center-y-hp").unwrap();
        // Résolution pixel ~1e-67 + 20 chiffres de garde : centre fourni (96
        // décimales) arrondi à la 87e décimale.
        assert_eq!(coords::pixel_fraction_digits(&p), Some(67));
        assert_eq!(Decimal::parse(x).unwrap(), Decimal::parse(&cx).unwrap().rounded(87));
        assert_eq!(x.len(), "-1.".len() + 87);
        let yd = Decimal::parse(y).unwrap();
        assert_eq!(yd, Decimal::parse(&cy).unwrap().rounded(87));
        assert_eq!(value_of(&c.command, "zoom"), Some("3.1415e64"));
        assert_eq!(value_of(&c.command, "iterations"), Some("70000"));
        assert_eq!(value_of(&c.command, "palette"), Some("8"));
        assert_eq!(value_of(&c.command, "color-repeat"), None);
    }

    #[test]
    fn julia_seed_palette_and_options_are_emitted() {
        let mut p = gui_like(FractalType::Julia, 640, 480);
        p.seed.re = -0.8;
        p.seed.im = 0.156;
        p.color.color_mode = 2;
        p.color.color_repeat = 12;
        p.color.out_coloring_mode = OutColoringMode::IterPlusRealImag;
        p.plane_transform = PlaneTransform::InversionShifted;
        p.rotation = -30.0;
        p.bailout = 4.0;
        p.engine.algorithm_mode = AlgorithmMode::Perturbation;
        p.engine.use_dd_tier = true;
        p.channels.enable_interior_detection = true;
        let c = build_cli_command(
            &p,
            CliExtras {
                aa_samples: 4,
                aa_jitter_scale: 0.5,
                exact_center: false,
            },
        );
        assert!(c.unexpressed.is_empty(), "{:?}", c.unexpressed);
        for (name, want) in [
            ("type", "4"),
            ("julia-re", "-0.8"),
            ("julia-im", "0.156"),
            ("palette", "2"),
            ("color-repeat", "12"),
            ("outcoloring", "iter+real/imag"),
            ("plane", "2"),
            ("rotation", "-30"),
            ("bailout", "4"),
            ("algorithm", "perturbation"),
            ("aa-samples", "4"),
            ("jitter-scale", "0.5"),
            ("width", "640"),
            ("height", "480"),
        ] {
            assert_eq!(value_of(&c.command, name), Some(want), "{name}: {}", c.command);
        }
        let a = args(&c.command);
        assert!(a.contains(&"--dd-tier"));
        assert!(a.contains(&"--enable-interior-detection"));
    }

    #[test]
    fn hybrid_and_opcodes_are_emitted_and_quoted() {
        let mut p = gui_like(FractalType::Mandelbrot, 320, 240);
        p.formula.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
        p.formula.hybrid_opcodes = Some("sqr rot{30} add".to_string());
        let c = build_cli_command(&p, CliExtras::default());
        assert_eq!(value_of(&c.command, "phases"), Some("mandelbrot,burning_ship"));
        assert!(c.command.contains("--opcodes 'sqr rot{30} add'"), "{}", c.command);
        for name in ["mandelbrot", "burning_ship"] {
            assert!(FractalType::from_hybrid_name(name).is_some());
        }
    }

    #[test]
    fn inexpressible_settings_are_reported_not_dropped() {
        let mut p = gui_like(FractalType::Mandelbrot, 320, 240);
        p.color.color_space = ColorSpace::Lch;
        p.color.color_offset = 0.25;
        p.transform_k = Some([1.0, 0.1, 0.0, 1.0]);
        p.seed.re = 0.1;
        let c = build_cli_command(&p, CliExtras::default());
        assert_eq!(c.unexpressed.len(), 4, "{:?}", c.unexpressed);
        assert!(c.command.contains("  # not expressible on the command line: color space Lch"));
    }

    #[test]
    fn lyapunov_preset_change_is_emitted_with_caveat() {
        let mut p = gui_like(FractalType::Lyapunov, 320, 240);
        crate::fractal::apply_lyapunov_preset(&mut p, LyapunovPreset::Jellyfish);
        let c = build_cli_command(&p, CliExtras::default());
        assert_eq!(value_of(&c.command, "lyapunov-preset"), Some("jellyfish"));
        assert_eq!(c.unexpressed.len(), 1);
    }

    #[test]
    fn cli_names_parse_back() {
        for m in [
            AlgorithmMode::Auto,
            AlgorithmMode::StandardF64,
            AlgorithmMode::Perturbation,
            AlgorithmMode::ReferenceGmp,
        ] {
            assert_eq!(AlgorithmMode::from_cli_name(algorithm_cli_name(m)), Some(m));
        }
        for &pt in PlaneTransform::all() {
            assert_eq!(PlaneTransform::from_cli_name(&pt.id().to_string()), Some(pt));
        }
        for id in 0..15u8 {
            let m = OutColoringMode::from_id(id).unwrap();
            assert_eq!(OutColoringMode::from_cli_name(m.cli_name()), Some(m));
            assert_eq!(shell_quote(m.cli_name()), m.cli_name());
        }
    }

    #[test]
    fn binary_expansion_of_a_typed_centre_comes_back_short() {
        let mut p = gui_like(FractalType::BurningShip, 160, 120);
        let view = ViewHp::from_center_and_zoom("-1.7619", "-0.0284", "350.5", 160, 120, 1024)
            .unwrap();
        view.write_to_params(&mut p);
        let c = build_cli_command(&p, CliExtras::default());
        assert_eq!(value_of(&c.command, "center-x-hp"), Some("-1.7619"));
        assert_eq!(value_of(&c.command, "center-y-hp"), Some("-0.0284"));
        assert_eq!(value_of(&c.command, "zoom"), Some("350.5"));
    }

    #[test]
    fn fmt_f64_round_trips() {
        for v in [0.0, -0.8, 0.156, 1e-8, 5.960464477539063e-8, 1e300, -2.5e-310, 123.456] {
            assert_eq!(fmt_f64(v).parse::<f64>().unwrap(), v);
            assert!(fmt_f64(v).len() < 30);
        }
    }
}
