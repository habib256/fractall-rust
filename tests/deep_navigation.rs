//! Tests end-to-end de navigation profonde pour les quatre types « spéciaux »
//! (Buddhabrot, Nebulabrot, Anti-Buddhabrot, Lyapunov).
//!
//! Ces types ne passent pas par la perturbation : ils ont leurs propres
//! renderers f64 et MPC, et le dispatcher bascule de l'un à l'autre dès qu'un
//! pixel n'est plus résolvable en `f64` (`special_view_needs_mpc`). Jusqu'ici
//! seules des fonctions de coordonnées étaient testées unitairement — rien ne
//! vérifiait la chaîne complète *navigation GUI → paramètres → dispatcher →
//! canaux rendus*.
//!
//! Chaque test part de `ViewHp`, applique de VRAIES opérations de navigation
//! (molette ancrée, pan, sélection rectangulaire, resize), republie la vue dans
//! `FractalParams` puis rend via `render_request`, l'entrée CPU unique.
//!
//! Ce que les mesures qui ont motivé ces verrous montrent :
//! - la navigation reste définie et déterministe jusqu'à des spans ~1e-30 ;
//! - à ces profondeurs les miroirs `f64` sont GELÉS : seules les chaînes HP
//!   distinguent deux vues, ce qui donne une sonde directe « la HP est-elle
//!   réellement consommée ? » (test [`lyapunov_deep_navigation_consumes_the_high_precision_center`]) ;
//! - la projection pixel du path MPC est exacte à la translation entière ;
//! - les types de densité gardent une fenêtre nourrie à toute profondeur, ce
//!   que leur échantillonnage par importance rend seul possible.

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use fractall_cli::fractal::{default_params_for_type, FractalParams, FractalType, ViewHp};
use fractall_cli::render::{render_request, RenderOutput, RenderRequest};

const W: u32 = 64;
const H: u32 = 48;

/// Les séquences de navigation enchaînent des dizaines de rendus MPC, dont le
/// coût croît avec la précision : elles tournent sur une grille réduite.
const SEQ_W: u32 = 32;
const SEQ_H: u32 = 24;

/// Les trois types de densité : ils accumulent des trajectoires au lieu de
/// calculer une valeur par pixel.
const DENSITY_TYPES: [FractalType; 3] = [
    FractalType::Buddhabrot,
    FractalType::Nebulabrot,
    FractalType::AntiBuddhabrot,
];

/// Les quatre types routés hors du dispatcher escape-time.
const SPECIAL_TYPES: [FractalType; 4] = [
    FractalType::Buddhabrot,
    FractalType::Nebulabrot,
    FractalType::AntiBuddhabrot,
    FractalType::Lyapunov,
];

fn base_params(fractal_type: FractalType) -> FractalParams {
    sized_params(fractal_type, W, H, 200)
}

fn sized_params(
    fractal_type: FractalType,
    width: u32,
    height: u32,
    iteration_max: u32,
) -> FractalParams {
    let mut params = default_params_for_type(fractal_type, width, height);
    params.iteration_max = iteration_max;
    params.engine.use_gmp = false;
    params
}

fn render(params: &FractalParams) -> RenderOutput {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut orbit_cache = None;
    render_request(RenderRequest::new(params, &cancel), &mut orbit_cache)
        .expect("le dispatcher CPU doit rendre les types spéciaux")
}

fn apply(view: &ViewHp, params: &FractalParams) -> FractalParams {
    let mut out = params.clone();
    view.write_to_params(&mut out);
    out
}

/// Vrai quand un pixel n'est plus représentable dans les miroirs `f64` : à ce
/// stade, seule la vue HP porte encore l'information.
fn f64_mirrors_are_frozen(params: &FractalParams) -> bool {
    params.center_x + params.span_x / f64::from(params.width.max(1)) == params.center_x
}

/// Vue Lyapunov chaotique : l'exposant y est franchement positif, donc le champ
/// reste sensible aux perturbations minuscules même très loin dans le zoom.
/// Les zones périodiques, elles, contractent et deviennent numériquement plates.
const CHAOTIC_LYAPUNOV: (f64, f64) = (3.9, 3.9);

fn zoomed_view(fractal_type: FractalType, center: (f64, f64), decades: usize) -> ViewHp {
    let mut params = base_params(fractal_type);
    params.center_x = center.0;
    params.center_y = center.1;
    let mut view = ViewHp::from_params(&params);
    for _ in 0..decades {
        view.zoom_at(0.5, 0.5, 10.0);
    }
    view
}

/// Séquence de navigation « comme à la main » : molette ancrée hors centre,
/// pan, sélection rectangulaire et resize s'enchaînent sans jamais repasser par
/// les miroirs `f64`.
fn navigate(view: &mut ViewHp, step: usize) {
    match step % 4 {
        0 => view.zoom_at(0.37, 0.62, 10.0),
        1 => view.zoom_at(0.5, 0.5, 10.0),
        2 => {
            view.pan_by(0.05, -0.03);
            view.zoom_at(0.5, 0.5, 10.0);
        }
        _ => view.select_rect(0.4, 0.4, 0.5, 0.5),
    }
}

/// Verrou de base : quelle que soit la profondeur atteinte, le dispatcher rend
/// des canaux COMPLETS et finis. Une régression de mapping (span sous-fluant,
/// centre gelé, division par un pixel nul) se manifeste ici en `NaN`, en canal
/// tronqué ou en panique.
#[test]
fn navigation_sequence_keeps_channels_complete_and_finite() {
    let pixels = (SEQ_W * SEQ_H) as usize;
    for fractal_type in SPECIAL_TYPES {
        let params = sized_params(fractal_type, SEQ_W, SEQ_H, 120);
        let mut view = ViewHp::from_params(&params);
        let mut reached_frozen_mirrors = false;

        for step in 0..18 {
            let stepped = apply(&view, &params);
            let out = render(&stepped);

            assert_eq!(
                out.iterations.len(),
                pixels,
                "{fractal_type:?} étape {step} : canal itérations incomplet"
            );
            assert_eq!(
                out.zs.len(),
                pixels,
                "{fractal_type:?} étape {step} : canal zs incomplet"
            );
            assert!(
                out.zs.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
                "{fractal_type:?} étape {step} : zs non fini (span {:.3e})",
                stepped.span_x
            );
            assert!(
                stepped.span_x > 0.0 || stepped.span_x_hp.is_some(),
                "{fractal_type:?} étape {step} : span perdu"
            );
            reached_frozen_mirrors |= f64_mirrors_are_frozen(&stepped);

            navigate(&mut view, step);
        }

        assert!(
            reached_frozen_mirrors,
            "{fractal_type:?} : la séquence doit franchir la limite f64 \
             (sinon le test n'exerce pas le path MPC)"
        );
    }
}

/// La navigation profonde reste reproductible : les rendus de densité
/// accumulent en parallèle dans des compteurs atomiques, un ordre de threads
/// différent ne doit pas changer un pixel.
#[test]
fn deep_navigation_is_deterministic() {
    for fractal_type in SPECIAL_TYPES {
        let params = sized_params(fractal_type, SEQ_W, SEQ_H, 120);
        let mut view = ViewHp::from_params(&params);
        for step in 0..18 {
            navigate(&mut view, step);
        }
        let deep = apply(&view, &params);
        assert!(
            f64_mirrors_are_frozen(&deep),
            "{fractal_type:?} : la vue de test doit être au-delà de la limite f64"
        );

        let first = render(&deep);
        let second = render(&deep);
        assert_eq!(
            first.iterations, second.iterations,
            "{fractal_type:?} : itérations non déterministes en zoom profond"
        );
        assert_eq!(
            first.zs, second.zs,
            "{fractal_type:?} : zs non déterministes en zoom profond"
        );
    }
}

/// Sonde directe de la consommation HP : deux vues dont les miroirs `f64` sont
/// STRICTEMENT identiques (le décalage d'un pixel à 1e-40 disparaît en `f64`)
/// mais dont les chaînes HP diffèrent doivent produire des images différentes.
///
/// Un renderer qui retomberait sur les miroirs `f64` — ou qui parserait la HP à
/// précision fixe insuffisante — rendrait deux images identiques.
#[test]
fn lyapunov_deep_navigation_consumes_the_high_precision_center() {
    let params = base_params(FractalType::Lyapunov);
    let view = zoomed_view(FractalType::Lyapunov, CHAOTIC_LYAPUNOV, 40);
    let mut shifted = view.clone();
    shifted.pan_by(1.0 / f64::from(W), 0.0); // exactement un pixel

    let left = apply(&view, &params);
    let right = apply(&shifted, &params);

    assert!(f64_mirrors_are_frozen(&left));
    assert_eq!(
        (left.center_x, left.center_y, left.span_x, left.span_y),
        (right.center_x, right.center_y, right.span_x, right.span_y),
        "les miroirs f64 doivent être indiscernables : sinon le test ne prouve rien"
    );
    assert_ne!(
        left.center_x_hp, right.center_x_hp,
        "seule la vue HP distingue les deux rendus"
    );

    let (a, b) = (render(&left), render(&right));
    let differing = a.zs.iter().zip(&b.zs).filter(|(x, y)| x.re != y.re).count();
    assert!(
        differing > 0,
        "décalage d'un pixel à span {:.1e} sans effet : la haute précision est perdue",
        left.span_x
    );
}

/// La projection pixel du path MPC est EXACTE à la translation entière : panner
/// de `k` pixels décale l'image de `k` pixels, sans dérive de mapping. Le verrou
/// tourne dans une zone chaotique, où la moindre différence de dernier bit se
/// verrait immédiatement (elle est amplifiée exponentiellement par la carte).
#[test]
fn lyapunov_deep_pan_by_whole_pixels_is_pixel_exact() {
    const SHIFT: usize = 5;
    let params = base_params(FractalType::Lyapunov);

    for decades in [20usize, 30, 40] {
        let view = zoomed_view(FractalType::Lyapunov, CHAOTIC_LYAPUNOV, decades);
        let mut panned = view.clone();
        panned.pan_by(SHIFT as f64 / f64::from(W), 0.0);

        let before = apply(&view, &params);
        let after = apply(&panned, &params);
        assert!(
            f64_mirrors_are_frozen(&before),
            "profondeur {decades} : le path MPC doit être engagé"
        );

        let (a, b) = (render(&before), render(&after));
        let width = W as usize;
        let mut checked = 0usize;
        for j in 0..H as usize {
            for i in 0..width - SHIFT {
                assert_eq!(
                    a.iterations[j * width + i + SHIFT],
                    b.iterations[j * width + i],
                    "profondeur {decades} : pan de {SHIFT} px non exact en ({i},{j})"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, (width - SHIFT) * H as usize);

        // Sans champ non trivial, l'égalité ci-dessus serait vide de sens.
        let distinct: BTreeSet<u32> = a.iterations.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "profondeur {decades} : champ constant, le verrou serait vacide"
        );
    }
}

/// Les types de densité échantillonnent les `c` sur le domaine canonique du
/// plan des paramètres, PAS sur la fenêtre affichée : zoomer doit donc montrer
/// la densité de plus près, pas une image vide.
///
/// Le zoom est poussé bien au-delà de ce qu'un tirage uniforme peut nourrir
/// (sa fenêtre est vide dès ~×1000, la densité visible décroissant comme sa
/// surface) : c'est l'échantillonnage par importance qui doit prendre le
/// relais et garder l'image vivante à toute profondeur.
#[test]
fn zooming_density_types_keeps_structure() {
    let pixels = (W * H) as usize;
    for fractal_type in DENSITY_TYPES {
        let params = base_params(fractal_type);
        let mut view = ViewHp::from_params(&params);

        for decade in 1..=8 {
            view.zoom_at(0.4, 0.55, 10.0);
            let zoomed = apply(&view, &params);
            let out = render(&zoomed);

            let fed = out.iterations.iter().filter(|v| **v != 0).count();
            let distinct: BTreeSet<u32> = out.iterations.iter().copied().collect();
            assert!(
                fed * 4 >= pixels,
                "{fractal_type:?} : à ×1e{decade}, seuls {fed}/{pixels} pixels sont nourris"
            );
            assert!(
                distinct.len() >= 8,
                "{fractal_type:?} : à ×1e{decade}, seulement {} valeurs distinctes",
                distinct.len()
            );
        }
    }
}

/// Le domaine d'échantillonnage des `c` NE SUIT PAS la vue : naviguer déplace
/// une fenêtre sur un champ fixe, ça ne régénère pas un champ différent.
///
/// Le symptôme historique — les `c` étaient tirés DANS la vue — est une nappe
/// uniforme : n'importe quelle fenêtre, même à 10⁶ du jeu de Mandelbrot, se
/// remplissait intégralement, parce que le premier itéré `z₁ = c` retombait
/// forcément dedans. Chaque sonde ci-dessous saturait les 100 % de pixels avant
/// la correction.
#[test]
fn density_sampling_domain_does_not_follow_the_view() {
    let pixels = (W * H) as usize;
    // (type, centre, span_x) — des fenêtres qu'aucune orbite ne visite, mais
    // que l'échantillonnage historique remplissait de ses propres `c`.
    let probes = [
        (FractalType::Buddhabrot, (1.0e6, 0.0), 1.0),
        (FractalType::Buddhabrot, (40.0, 40.0), 4.0),
        (FractalType::Nebulabrot, (1.0e6, 0.0), 1.0),
        (FractalType::Nebulabrot, (40.0, 40.0), 4.0),
        // Anti-Buddhabrot : seuls les `c` PRISONNIERS comptent, la nappe
        // apparaissait donc au coeur d'un bulbe plutôt qu'à l'infini.
        (FractalType::AntiBuddhabrot, (-1.0, 0.0), 1.0e-3),
    ];

    for (fractal_type, center, span_x) in probes {
        let mut params = base_params(fractal_type);
        params.center_x = center.0;
        params.center_y = center.1;
        params.span_x = span_x;
        params.span_y = span_x * f64::from(H) / f64::from(W);
        let out = render(&params);
        let nonzero = out.iterations.iter().filter(|v| **v != 0).count();
        assert!(
            nonzero < pixels,
            "{fractal_type:?} en {center:?} (span {span_x}) : nappe uniforme sur \
             les {pixels} pixels — l'échantillonnage suit la vue"
        );
    }
}

/// Invariant propre à l'Anti-Buddhabrot : les orbites retenues sont celles qui
/// NE s'échappent pas, elles restent donc bornées. Une fenêtre placée hors du
/// disque de capture ne peut rien recevoir, à n'importe quel zoom.
#[test]
fn antibuddhabrot_never_plots_outside_the_bounded_region() {
    let mut params = base_params(FractalType::AntiBuddhabrot);
    params.center_x = 40.0;
    params.center_y = 40.0;
    params.span_x = 4.0;
    params.span_y = 3.0;
    let out = render(&params);
    assert!(
        out.iterations.iter().all(|v| *v == 0),
        "orbites prisonnières projetées hors du disque de capture"
    );
}

/// Le domaine d'échantillonnage étant fixe, la vue par défaut — qui EST ce
/// domaine — doit rester rigoureusement inchangée.
#[test]
fn default_density_view_samples_the_whole_canonical_domain() {
    for fractal_type in DENSITY_TYPES {
        let params = base_params(fractal_type);
        assert_eq!(
            (
                params.center_x,
                params.center_y,
                params.span_x,
                params.span_y
            ),
            (-0.5, 0.0, 4.0, 3.0),
            "{fractal_type:?} : la vue par défaut n'est plus le domaine canonique"
        );
        let out = render(&params);
        let distinct: BTreeSet<u32> = out.iterations.iter().copied().collect();
        assert!(
            distinct.len() > 10,
            "{fractal_type:?} : vue par défaut dégénérée ({} valeurs)",
            distinct.len()
        );
    }
}

/// Diagnostic : taux de remplissage de la fenêtre en fonction du zoom.
///
/// `cargo test --release --test deep_navigation -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic : caractérise le remplissage, ne verrouille rien"]
fn density_fill_by_zoom_diagnostic() {
    for fractal_type in DENSITY_TYPES {
        let params = base_params(fractal_type);
        let mut view = ViewHp::from_params(&params);
        for decade in 0..9 {
            let stepped = apply(&view, &params);
            let out = render(&stepped);
            let fed = out.iterations.iter().filter(|v| **v != 0).count();
            let distinct: BTreeSet<u32> = out.iterations.iter().copied().collect();
            println!(
                "{fractal_type:?} ×{:<10} span={:.2e} nourris={fed}/{} distinctes={}",
                10u64.pow(decade),
                stepped.span_x,
                out.iterations.len(),
                distinct.len()
            );
            view.zoom_at(0.4, 0.55, 10.0);
        }
    }
}
