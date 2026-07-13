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
];

pub fn find(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

pub fn names() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.name).collect()
}
