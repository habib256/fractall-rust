use crate::fractal::FractalType;

#[derive(Debug, Clone)]
pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    pub fractal_type: FractalType,
    pub center_x_hp: &'static str,
    pub center_y_hp: &'static str,
    pub zoom: &'static str,
    pub iterations: u32,
    pub multibrot_power: Option<f64>,
    pub julia_seed: Option<(f64, f64)>,
    /// Override GMP precision in bits. If None, uses default 256.
    /// Required for zooms > 1e70 where 256 bits is insufficient.
    pub precision_bits: Option<u32>,
}

/// Fixed deep-zoom scenes covering the risk areas of the perturbation pipeline.
/// Shallow scenes (1e8-1e18) validate the fast paths; deep scenes (1e30+) validate
/// that perturbation's GMP path matches pure GMP at extreme zoom factors.
/// Deep-zoom coordinates (e30, e50, e100) are sourced from the `locations/` TOML
/// collection (rust-fractal-core format, verified against Fraktaler-3).
pub const PRESETS: &[Preset] = &[
    Preset {
        name: "seahorse-valley",
        description: "Mandelbrot Seahorse Valley at zoom 1e8 — classic shallow deep-zoom.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158704752191506114774",
        center_y_hp: "0.131825904205311970493132056385139",
        zoom: "1e8",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    // NB : PASS à ≤96² mais WARN à ≥128²/256² (post-recalibrage p99 G6 ; ~8-16 px
    // divergents dont 2 symétriques à iter_diff=210). Plancher de précision f64 du
    // δ au voisinage de l'antenne période-2 (c≈-1.75) : ~11 rebases près de
    // |Z|≈0.027 → cancellation → escape TARDIF (pert=867 > gmp=657, sur-compte).
    // `--dd-tier` corrige à 100 %. C'est le cas moteur du wisdom auto-dispatch
    // (cf. TODO.md §G3 « auto-dispatch (wisdom) »).
    // ⚠️ Trois voies vs GMP ground truth (2026-07-13, F3 Linux dispo) : fractall
    // sur-compte 16/65536 px (99.98 % exact), F3 diverge 9391/65536 px (mean 3.74,
    // max 1582) — fractall est ~600× plus fidèle à GMP que F3 ici. Le WARN N'EST
    // donc PAS un retard sur F3 : c'est 2 px dd-sensibles, F3 fait pire partout.
    Preset {
        name: "mandelbrot-e13",
        description: "Mandelbrot at zoom 1e13 — just above perturbation activation threshold.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.7499537683537087",
        center_y_hp: "0.0",
        zoom: "1e13",
        iterations: 16384,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "mandelbrot-e17",
        description: "Mandelbrot at zoom 1e17 — forces GMP perturbation path.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.7499537683537087215208540815925",
        center_y_hp: "0.0",
        zoom: "1e17",
        iterations: 32768,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "misiurewicz-m32",
        description: "Mandelbrot Misiurewicz point M32,2 — stress test for BLA escape precision.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.77568377",
        center_y_hp: "0.13646737",
        zoom: "1e12",
        iterations: 8192,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "julia-siegel-disk",
        description: "Julia c=-0.8+0.156i at zoom 1e10 — non-trivial Julia perturbation.",
        fractal_type: FractalType::Julia,
        center_x_hp: "0.0",
        center_y_hp: "0.0",
        zoom: "1e10",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: Some((-0.8, 0.156)),
        precision_bits: None,
    },
    Preset {
        name: "burning-ship-antenna",
        description: "Burning Ship antenna at zoom 1e9 — non-conformal BLA (matrix 2x2).",
        fractal_type: FractalType::BurningShip,
        center_x_hp: "-1.7492060",
        center_y_hp: "-0.0286762",
        zoom: "1e9",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "tricorn-spiral",
        description: "Tricorn spiral at zoom 1e8 — non-conformal (conjugation in iteration).",
        fractal_type: FractalType::Tricorn,
        center_x_hp: "-0.7",
        center_y_hp: "0.15",
        zoom: "1e8",
        iterations: 2048,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    // --- Celtic/Buffalo/PerpBS : familles non-conformes à pliage abs (cf. TODO G3).
    // Scènes choisies sur des frontières LISSES hors axes de pliage : la zone
    // hirsute (ex. antenne -1.75) est à sensibilité de précision extrême (GMP-256
    // lui-même non convergé, un écart pert↔GMP y reflète le plancher f64, pas un
    // bug). Sélection outillée (2026-07-13) : descente de zoom sur cellules
    // frontière à faible variance locale + itération d'échappement invariante à
    // une perturbation 2⁻⁴⁶ de c ; contrôle ground truth GMP-256 == GMP-512.
    Preset {
        name: "celtic-fold-edge",
        description: "Celtic smooth fold-edge at zoom 1e9 — non-conformal abs(Re(z²)), interior + deep escapes (~4090).",
        fractal_type: FractalType::Celtic,
        center_x_hp: "0.18034153496846556",
        center_y_hp: "-0.18571428521536293",
        zoom: "1e9",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "buffalo-fold-edge",
        description: "Buffalo smooth fold-edge at zoom 1e9 — double abs folding, interior + deep escapes (~4090).",
        fractal_type: FractalType::Buffalo,
        center_x_hp: "0.2625929825473577",
        center_y_hp: "0.144079340971075",
        zoom: "1e9",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "perpbs-escape-band",
        description: "Perpendicular Burning Ship at zoom 1e9 — all-escape band (~3000 iters), exercises escaping reference + rebase-at-end.",
        fractal_type: FractalType::PerpendicularBurningShip,
        center_x_hp: "-0.2230574102140963",
        center_y_hp: "0.16348968017846344",
        zoom: "1e9",
        iterations: 4096,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "mandelbrot-e18-minibrot",
        description: "Mandelbrot deep minibrot at zoom 1e18.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.74995376835370872152085408159254600000000",
        center_y_hp: "0.0",
        zoom: "1e18",
        iterations: 65536,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    // --- Deep scenes sourced from locations/*.toml (rust-fractal-core format) ---
    Preset {
        name: "mandelbrot-e30",
        description: "Mandelbrot deep scene at zoom 1e30 (coords from locations/e50.toml, zoomed out). 256 bits GMP.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000",
        center_y_hp: "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880",
        zoom: "1e30",
        iterations: 131072,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: Some(256),
    },
    Preset {
        name: "mandelbrot-e50",
        description: "Mandelbrot at zoom 1e50, full precision coords from locations/e50.toml. Pure GMP slow but feasible.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000",
        center_y_hp: "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880",
        zoom: "1e50",
        iterations: 263010,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: Some(384),
    },
    Preset {
        name: "mandelbrot-e100",
        description: "Mandelbrot at zoom 1e100 (coords from locations/e113.toml, zoomed out). 500 bits GMP.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.47981577613247326072298452597877854692240725774045369689878510139864920741002293820250517329282011227363313053159203914640783415609608168660705123082446357179491909705403381200",
        center_y_hp: "-0.00063911193261361727152139632255671572957303918943984736047394936471220951961813321928573067036466151147195436388486168819318341208023229522609015461543581599807510715681229605",
        zoom: "1e100",
        iterations: 35494,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: Some(500),
    },
    // Verrou « glitch de référence unique » (fuzz seed 20260714). Blob intérieur
    // (~513 px) faussement évadé à iter 304 vs 2048 réel : δ décorrèle du vrai
    // orbit sans que rebase/BLA ne le capte, et la correction GMP-delta sur la
    // MÊME référence reste glitchée (structural, pas numérique). Résolu par
    // l'escalade des pixels encore glitchés vers `iterate_point_mpc` (full GMP
    // indépendant de la référence). Sans le fix : WARN max_diff=1744 div_ratio=0.78 %.
    Preset {
        name: "single-ref-glitch-interior",
        description: "Mandelbrot 6e7 — interior blob wrongly escaped by single-reference glitch; locks the still-glitched → full-GMP escalation.",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.6152286330858866",
        center_y_hp: "0.40106525023778294",
        zoom: "6.023813e7",
        iterations: 2048,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Les presets GPU doivent rester dans le domaine du dispatch GPU actuel
    /// (types M/J/BS de `render_dispatch`) et nommés distinctement des presets
    /// CPU (préfixe `gpu-`, unicité globale) — le rapport les écrit sous
    /// `<output-dir>/gpu/` mais le summary les référence par nom.
    #[test]
    fn gpu_presets_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for p in GPU_PRESETS {
            assert!(p.name.starts_with("gpu-"), "{} doit être préfixé gpu-", p.name);
            assert!(seen.insert(p.name), "nom dupliqué : {}", p.name);
            assert!(
                matches!(
                    p.fractal_type,
                    FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip
                ),
                "{} : type {:?} hors du dispatch GPU",
                p.name,
                p.fractal_type
            );
            assert!(p.zoom.parse::<f64>().is_ok(), "{} : zoom invalide", p.name);
        }
        assert!(!GPU_PRESETS.is_empty());
        // Pas de collision avec les presets CPU.
        for p in PRESETS {
            assert!(!seen.contains(p.name), "collision CPU/GPU : {}", p.name);
        }
    }
}

pub fn find(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

pub fn names() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.name).collect()
}

/// Échelle de zoom pour la **parité GPU↔CPU** (`gpu-suite`, verrou G9.4/9.5) :
/// même centre seahorse à chaque décade du range GPU (shader std f32 jusqu'à
/// ~1e4, perturbation au-delà — seuil `GPU_PERTURBATION_THRESHOLD`). Juge =
/// GMP PUR (cf. `compare_gpu` : le CPU Auto f64-std diverge lui-même de ~6 %
/// de la vérité à 5000 iters sur bord chaotique — invalide comme juge).
/// Baseline 2026-07-15 (256², kernel perturbation F3-strict **f64 natif**) :
/// WARN sur TOUT le range perturbation 1e4→1e8 (p99=0, div 0.0007-0.0015,
/// escape_disagree=0 — le niveau exact du CPU-perturbation vs GMP, bord
/// épars) ; FAIL 1e2-1e3 (shaders std f32, mantisse 24 b, p99 215-578 —
/// gap restant, cf. TODO G9.5). Itérations 5000 : assez pour structurer le
/// bord sans saturer.
pub const GPU_PRESETS: &[Preset] = &[
    Preset {
        name: "gpu-seahorse-1e2",
        description: "Parité GPU↔CPU — seahorse zoom 1e2 (shader std f32, bord chaotique).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158",
        center_y_hp: "0.131825904205311",
        zoom: "1e2",
        iterations: 5000,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "gpu-seahorse-1e3",
        description: "Parité GPU↔CPU — seahorse zoom 1e3 (shader std f32).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158",
        center_y_hp: "0.131825904205311",
        zoom: "1e3",
        iterations: 5000,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "gpu-seahorse-1e4",
        description: "Parité GPU↔CPU — seahorse zoom 1e4 (début du path perturbation GPU, pixel<1e-5).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158",
        center_y_hp: "0.131825904205311",
        zoom: "1e4",
        iterations: 5000,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "gpu-seahorse-3e5",
        description: "Parité GPU↔CPU — seahorse zoom 3e5 (perturbation GPU).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158",
        center_y_hp: "0.131825904205311",
        zoom: "3e5",
        iterations: 5000,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "gpu-seahorse-1e6",
        description: "Parité GPU↔CPU — seahorse zoom 1e6 (perturbation GPU f64).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158",
        center_y_hp: "0.131825904205311",
        zoom: "1e6",
        iterations: 5000,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    Preset {
        name: "gpu-seahorse-1e8",
        description: "Parité GPU↔CPU — seahorse zoom 1e8 (perturbation GPU f64 profonde).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.743643887037158",
        center_y_hp: "0.131825904205311",
        zoom: "1e8",
        iterations: 5000,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    // --- Range deep (G9.4b) : au-delà de l'ancien plafond vérifié 1e8. ---
    Preset {
        name: "gpu-mandelbrot-e13",
        description: "Parité GPU — zoom 1e13 (perturbation f64, réf pleine, coords mandelbrot-e13).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-1.7499537683537087",
        center_y_hp: "0.0",
        zoom: "1e13",
        iterations: 16384,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: None,
    },
    // Verrou rebase-at-end kernel (G9.4b) : réf ATOM-TRONQUÉE (11177 iters ≪
    // iter_max) — exerce le wrap fin-de-référence F3 (`hybrid.cc:301`) + guard
    // BLA `lands_on_ref_end` portés en WGSL. ⚠️ Scène ultra-sensible (le CPU
    // f64 lui-même FAIL div 0.034 ici, cf. use_dd_tier du preset CPU homonyme) ;
    // le kernel GPU (BLA conforme) tient WARN div ~3e-4. Juge GMP ~3 min à 256².
    Preset {
        name: "gpu-mandelbrot-e30-truncated-ref",
        description: "Parité GPU — zoom 1e30, réf atom-tronquée (rebase-at-end kernel, coords mandelbrot-e30).",
        fractal_type: FractalType::Mandelbrot,
        center_x_hp: "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000",
        center_y_hp: "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880",
        zoom: "1e30",
        iterations: 131072,
        multibrot_power: None,
        julia_seed: None,
        precision_bits: Some(256),
    },
];
