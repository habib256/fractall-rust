//! Rendus de densité : Buddhabrot, Nebulabrot et Anti-Buddhabrot.
//!
//! Ces trois types partagent le même moteur d'échantillonnage
//! ([`super::density`]) — tirage des `c` sur le domaine canonique, itération de
//! l'orbite, projection des points visités dans la fenêtre — et ne diffèrent
//! que par trois choses : les orbites retenues (échappées ou prisonnières), la
//! profondeur d'itération, et la façon de coloriser la masse accumulée.
//!
//! Le moteur choisit seul son régime : tirage uniforme quand la vue couvre une
//! part suffisante du domaine, chaînes de Metropolis-Hastings dès qu'elle est
//! plus petite (sans quoi la fenêtre serait affamée par le zoom).

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;

use super::density::{self, OrbitSpec};
use crate::fractal::FractalParams;

/// Profondeurs d'itération des trois canaux du Nebulabrot.
const NEBULA_ITER_R: u32 = 50;
const NEBULA_ITER_G: u32 = 500;
const NEBULA_ITER_B: u32 = 5000;

/// Budget d'échantillons : la densité par pixel doit rester exploitable sans
/// faire exploser le temps de rendu des grandes surfaces.
fn sample_budget(pixels: usize, per_pixel: &[usize; 3], cap: usize) -> usize {
    let rate = if pixels <= 640 * 480 {
        per_pixel[0]
    } else if pixels <= 1024 * 768 {
        per_pixel[1]
    } else {
        per_pixel[2]
    };
    (pixels * rate).max(1000).min(cap)
}

fn escape_time_budget(pixels: usize) -> usize {
    sample_budget(pixels, &[20, 10, 5], 50_000_000)
}

fn nebula_budget(pixels: usize) -> usize {
    sample_budget(pixels, &[15, 8, 4], 30_000_000)
}

/// Orbites qui s'échappent, avec l'abandon anticipé historique du Buddhabrot :
/// une orbite encore blottie autour de zéro à mi-parcours ne s'échappera pas.
fn buddhabrot_spec(params: &FractalParams) -> OrbitSpec {
    let iter_max = params.iteration_max;
    OrbitSpec {
        iter_max,
        bailout_sq: params.bailout * params.bailout,
        early_exit_at: Some(if iter_max < 50 { iter_max / 2 } else { 50 }),
        nan_counts_as_escape: false,
        keep_escaped: true,
    }
}

/// Orbites prisonnières : ici un débordement numérique est une évasion, donc
/// une orbite à rejeter.
fn antibuddhabrot_spec(params: &FractalParams) -> OrbitSpec {
    OrbitSpec {
        iter_max: params.iteration_max,
        bailout_sq: params.bailout * params.bailout,
        early_exit_at: None,
        nan_counts_as_escape: true,
        keep_escaped: false,
    }
}

fn nebulabrot_spec(params: &FractalParams) -> OrbitSpec {
    OrbitSpec {
        iter_max: NEBULA_ITER_B,
        bailout_sq: params.bailout * params.bailout,
        early_exit_at: None,
        nan_counts_as_escape: false,
        keep_escaped: true,
    }
}

/// Un seul canal, nourri par toutes les orbites retenues.
fn single_channel(_escape_iter: u32) -> u32 {
    1
}

/// Trois canaux par profondeur d'évasion : le rouge ne garde que les orbites
/// courtes, le bleu les garde toutes.
fn nebula_channels(escape_iter: u32) -> u32 {
    let mut mask = 0;
    if escape_iter <= NEBULA_ITER_R {
        mask |= 1;
    }
    if escape_iter <= NEBULA_ITER_G {
        mask |= 2;
    }
    if escape_iter <= NEBULA_ITER_B {
        mask |= 4;
    }
    mask
}

/// Compression logarithmique commune, **invariante d'échelle**.
///
/// La densité couvre plusieurs ordres de grandeur : elle s'affiche en
/// logarithme. Deux précautions :
///
/// - le rapport est pris sur la densité NORMALISÉE, jamais sur sa valeur brute,
///   sinon le contraste dépend d'une échelle arbitraire — le budget
///   d'échantillons (donc la résolution demandée) et le régime
///   d'échantillonnage (une chaîne de Markov dépose environ dix fois plus de
///   masse par orbite calculée qu'un tirage uniforme). Le même lieu rendu deux
///   fois n'aurait alors pas le même rendu ;
/// - la référence est un **quantile haut**, pas le maximum. Quelques pixels
///   chauds — une chaîne qui piétine, un pic de Poisson — suffisent sinon à
///   écraser toute l'image (mesuré : maximum trois fois le 99ᵉ centile).
///
/// L'amplitude de la compression suit la dynamique de l'image (référence sur
/// moyenne des pixels nourris) : un champ creux est fortement compressé pour
/// révéler ses valeurs basses, un champ dense l'est peu.
fn log_compress(mass: &[f64]) -> Box<dyn Fn(f64) -> f64 + Sync> {
    const REFERENCE_QUANTILE: f64 = 0.999;

    let mut positives: Vec<f64> = mass.iter().copied().filter(|value| *value > 0.0).collect();
    if positives.is_empty() {
        return Box::new(move |_: f64| 0.0) as Box<dyn Fn(f64) -> f64 + Sync>;
    }
    let mean = positives.iter().sum::<f64>() / positives.len() as f64;
    let rank = ((positives.len() - 1) as f64 * REFERENCE_QUANTILE) as usize;
    let (_, anchor, _) = positives.select_nth_unstable_by(rank, f64::total_cmp);
    let anchor = *anchor;
    let amplitude = (anchor / mean).clamp(4.0, 1.0e6);
    let denominator = (1.0 + amplitude).ln();
    Box::new(move |value: f64| {
        if anchor <= 0.0 {
            0.0
        } else {
            (1.0 + (value / anchor).min(1.0) * amplitude).ln() / denominator
        }
    })
}

/// Sortie standard des types à un canal : itérations pour la colorisation,
/// `zs` pour les modes qui lisent la magnitude.
fn density_output(mass: &[f64], iter_max: u32) -> (Vec<u32>, Vec<Complex64>) {
    let compress = log_compress(mass);
    let iterations = mass
        .par_iter()
        .map(|value| (compress(*value) * iter_max as f64) as u32)
        .collect();
    let zs = mass
        .par_iter()
        .map(|value| Complex64::new(compress(*value) * 2.0, 0.0))
        .collect();
    (iterations, zs)
}

fn empty_output(size: usize) -> (Vec<u32>, Vec<Complex64>) {
    (vec![0; size], vec![Complex64::new(0.0, 0.0); size])
}

fn render_escape_time_density(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    spec: OrbitSpec,
    use_mpc: bool,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let pixels = params.width as usize * params.height as usize;
    if pixels == 0 {
        return Some(empty_output(pixels));
    }
    let mass = density::accumulate(
        params,
        cancel,
        &spec,
        escape_time_budget(pixels),
        1,
        single_channel,
        use_mpc,
    )?;
    Some(density_output(&mass[0], params.iteration_max))
}

/// Version annulable du rendu Buddhabrot en MPC.
pub fn render_buddhabrot_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_density(params, cancel, buddhabrot_spec(params), true)
}

/// Version annulable du rendu Buddhabrot (f64).
pub fn render_buddhabrot_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_density(params, cancel, buddhabrot_spec(params), false)
}

/// Version annulable du rendu Anti-Buddhabrot en MPC.
pub fn render_antibuddhabrot_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_density(params, cancel, antibuddhabrot_spec(params), true)
}

/// Version annulable du rendu Anti-Buddhabrot (f64).
pub fn render_antibuddhabrot_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_escape_time_density(params, cancel, antibuddhabrot_spec(params), false)
}

fn render_nebulabrot_density(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    use_mpc: bool,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let pixels = params.width as usize * params.height as usize;
    if pixels == 0 {
        return Some(empty_output(pixels));
    }
    let mass = density::accumulate(
        params,
        cancel,
        &nebulabrot_spec(params),
        nebula_budget(pixels),
        3,
        nebula_channels,
        use_mpc,
    )?;

    let (red, green, blue) = (
        log_compress(&mass[0]),
        log_compress(&mass[1]),
        log_compress(&mass[2]),
    );

    // Rouge et vert voyagent dans les itérations (deux octets), le bleu dans la
    // magnitude — c'est ce que la colorisation densité attend.
    let iterations = (0..pixels)
        .into_par_iter()
        .map(|i| {
            let r = (red(mass[0][i]) * 255.0) as u32;
            let g = (green(mass[1][i]) * 255.0) as u32;
            (r << 16) | (g << 8)
        })
        .collect();
    let zs = (0..pixels)
        .into_par_iter()
        .map(|i| Complex64::new(blue(mass[2][i]), 0.0))
        .collect();

    Some((iterations, zs))
}

/// Version annulable du rendu Nebulabrot en MPC.
pub fn render_nebulabrot_mpc_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_nebulabrot_density(params, cancel, true)
}

/// Version annulable du rendu Nebulabrot (f64).
pub fn render_nebulabrot_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    render_nebulabrot_density(params, cancel, false)
}

/// Masse brute d'un rendu Buddhabrot dans un régime imposé — outil de
/// comparaison des deux échantillonneurs.
#[cfg(test)]
pub(crate) fn buddhabrot_mass(
    params: &FractalParams,
    mode: density::Mode,
    budget: usize,
) -> Vec<f64> {
    let cancel = Arc::new(AtomicBool::new(false));
    density::accumulate_with_mode(
        params,
        &cancel,
        &buddhabrot_spec(params),
        budget,
        1,
        single_channel,
        false,
        mode,
    )
    .expect("rendu non annulé")
    .remove(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    fn density_params(fractal_type: FractalType) -> FractalParams {
        let mut params = default_params_for_type(fractal_type, 48, 36);
        params.iteration_max = 120;
        params
    }

    fn render(params: &FractalParams, use_mpc: bool) -> Vec<u32> {
        let cancel = Arc::new(AtomicBool::new(false));
        let out = match params.fractal_type {
            FractalType::Nebulabrot => render_nebulabrot_density(params, &cancel, use_mpc),
            FractalType::AntiBuddhabrot => {
                render_escape_time_density(params, &cancel, antibuddhabrot_spec(params), use_mpc)
            }
            _ => render_escape_time_density(params, &cancel, buddhabrot_spec(params), use_mpc),
        };
        out.expect("rendu non annulé").0
    }

    /// Le masque Nebulabrot encode la hiérarchie des profondeurs : une orbite
    /// courte nourrit les trois canaux, une longue seulement le bleu.
    #[test]
    fn nebula_channel_mask_is_nested_by_escape_depth() {
        assert_eq!(nebula_channels(10), 0b111);
        assert_eq!(nebula_channels(NEBULA_ITER_R), 0b111);
        assert_eq!(nebula_channels(NEBULA_ITER_R + 1), 0b110);
        assert_eq!(nebula_channels(NEBULA_ITER_G + 1), 0b100);
        assert_eq!(nebula_channels(NEBULA_ITER_B), 0b100);
    }

    /// Le budget d'échantillons décroît par palier de surface, et reste borné.
    #[test]
    fn sample_budget_is_bounded() {
        assert_eq!(escape_time_budget(1), 1000, "plancher");
        assert_eq!(escape_time_budget(100 * 100), 100 * 100 * 20);
        assert_eq!(escape_time_budget(800 * 800), 800 * 800 * 10);
        assert_eq!(escape_time_budget(4000 * 4000), 50_000_000, "plafond");
        assert_eq!(nebula_budget(100 * 100), 100 * 100 * 15);
        assert_eq!(nebula_budget(4000 * 4000), 30_000_000);
    }

    /// La vue par défaut EST le domaine d'échantillonnage : le rendu doit être
    /// dense et structuré, et le régime uniforme rester engagé.
    #[test]
    fn default_view_renders_dense_structure() {
        for fractal_type in [
            FractalType::Buddhabrot,
            FractalType::Nebulabrot,
            FractalType::AntiBuddhabrot,
        ] {
            let params = density_params(fractal_type);
            assert!(
                !density::uses_metropolis(&params),
                "{fractal_type:?} : la vue par défaut ne doit pas payer l'amorçage"
            );
            let iterations = render(&params, false);
            let distinct: std::collections::BTreeSet<u32> = iterations.iter().copied().collect();
            assert!(
                distinct.len() > 10,
                "{fractal_type:?} : vue par défaut dégénérée ({} valeurs)",
                distinct.len()
            );
        }
    }

    /// Les deux arithmétiques partagent projection et critères d'orbite : sur
    /// une vue que le `f64` résout encore, elles doivent voir la même image à
    /// quelques arrondis près.
    #[test]
    fn f64_and_mpc_paths_agree_on_a_shallow_view() {
        let mut params = density_params(FractalType::Buddhabrot);
        params.center_x = -0.6;
        params.span_x = 1.0;
        params.span_y = 0.75;
        let (f64_path, mpc_path) = (render(&params, false), render(&params, true));
        let pixels = f64_path.len();
        let differing = (0..pixels).filter(|i| f64_path[*i] != mpc_path[*i]).count();
        assert!(
            differing * 20 < pixels,
            "{differing}/{pixels} pixels divergent entre f64 et MPC"
        );
    }

    /// Corrélation de Pearson entre deux champs, chacun normalisé en somme :
    /// deux estimateurs du MÊME champ doivent être colinéaires, quelle que soit
    /// leur échelle absolue.
    fn correlation(left: &[f64], right: &[f64]) -> f64 {
        let normalize = |field: &[f64]| {
            let total: f64 = field.iter().sum();
            let scale = if total > 0.0 { 1.0 / total } else { 0.0 };
            field.iter().map(|value| value * scale).collect::<Vec<_>>()
        };
        let (left, right) = (normalize(left), normalize(right));
        let count = left.len() as f64;
        let mean_left = left.iter().sum::<f64>() / count;
        let mean_right = right.iter().sum::<f64>() / count;
        let covariance: f64 = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (a - mean_left) * (b - mean_right))
            .sum();
        let var_left: f64 = left.iter().map(|a| (a - mean_left).powi(2)).sum();
        let var_right: f64 = right.iter().map(|b| (b - mean_right).powi(2)).sum();
        covariance / (var_left.sqrt() * var_right.sqrt()).max(1.0e-30)
    }

    fn zoomed(center: (f64, f64), zoom: f64) -> FractalParams {
        let mut params = density_params(FractalType::Buddhabrot);
        params.iteration_max = 120;
        params.center_x = center.0;
        params.center_y = center.1;
        params.span_x = 4.0 / zoom;
        params.span_y = 3.0 / zoom;
        params
    }

    /// Verrou central de l'échantillonnage par importance : les chaînes de
    /// Markov estiment le MÊME champ que le tirage uniforme.
    ///
    /// La comparaison a lieu à un zoom où le tirage uniforme est encore
    /// solide, et sa référence est calculée à très gros budget pour que le
    /// bruit du témoin ne masque pas un biais de la chaîne. C'est ce test qui
    /// a fait tomber deux erreurs de conception : un noyau de mutation adapté
    /// PENDANT la mesure (il annule le jacobien, donc la densité — corrélation
    /// tombée à 0,06) et une cible de proximité de plein poids.
    #[test]
    fn metropolis_estimates_the_same_field_as_uniform_sampling() {
        let params = zoomed((-0.745, 0.1), 8.0);
        let reference = buddhabrot_mass(&params, density::Mode::Uniform, 20_000_000);
        let chains = buddhabrot_mass(&params, density::Mode::Metropolis, 2_000_000);
        let correlation = correlation(&reference, &chains);
        assert!(
            correlation > 0.85,
            "les deux régimes divergent (corrélation {correlation:.3})"
        );
    }

    /// Là où le tirage uniforme est affamé, les chaînes nourrissent encore la
    /// fenêtre : c'est toute la raison d'être du régime.
    #[test]
    fn metropolis_feeds_a_window_that_starves_uniform_sampling() {
        let params = zoomed((-0.745, 0.1), 300.0);
        let budget = 2_000_000;
        let uniform = buddhabrot_mass(&params, density::Mode::Uniform, budget);
        let chains = buddhabrot_mass(&params, density::Mode::Metropolis, budget);
        let fed = |field: &[f64]| field.iter().filter(|value| **value > 0.0).count();
        let (starved, nourished) = (fed(&uniform), fed(&chains));
        assert!(
            starved * 4 < field_len(&uniform),
            "le tirage uniforme n'est pas affamé ici ({starved} pixels), \
             le test ne prouve rien"
        );
        assert!(
            nourished * 2 > field_len(&chains),
            "les chaînes ne nourrissent que {nourished} pixels sur {}",
            field_len(&chains)
        );
    }

    fn field_len(field: &[f64]) -> usize {
        field.len()
    }

    /// Diagnostic : corrélation et erreur relative des deux régimes en fonction
    /// du zoom, contre une référence uniforme à très gros budget.
    ///
    /// `cargo test --release --lib regimes_diagnostic -- --ignored --nocapture`
    #[test]
    #[ignore = "diagnostic : caractérise la qualité du régime, ne verrouille rien"]
    fn regimes_diagnostic() {
        for zoom in [8.0f64, 40.0, 200.0] {
            let params = zoomed((-0.745, 0.1), zoom);
            let reference = buddhabrot_mass(&params, density::Mode::Uniform, 200_000_000);
            let chains = buddhabrot_mass(&params, density::Mode::Metropolis, 4_000_000);
            let total_reference: f64 = reference.iter().sum();
            let total_chains: f64 = chains.iter().sum();
            let error = reference
                .iter()
                .zip(&chains)
                .map(|(a, b)| (a / total_reference - b / total_chains).powi(2))
                .sum::<f64>()
                .sqrt()
                / (1.0 / reference.len() as f64);
            println!(
                "zoom=×{zoom} corr={:.4} erreur_relative={:.3} pixels_nourris_uniforme={} metropolis={}",
                correlation(&reference, &chains),
                error / (reference.len() as f64).sqrt(),
                reference.iter().filter(|v| **v > 0.0).count(),
                chains.iter().filter(|v| **v > 0.0).count(),
            );
        }
    }
}
