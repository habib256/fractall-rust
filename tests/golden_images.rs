//! Golden image tests (P2.1 du TODO).
//!
//! Pour chaque cas de test, on lance le binaire `fractall-cli`, on lit le PNG
//! produit, on compare les **pixels** (pas les bytes PNG : les métadonnées
//! diffèrent même quand l'image est identique) avec une golden image versionnée
//! dans `tests/golden/<name>.png`.
//!
//! ## Workflow
//!
//! ```bash
//! # Lancer les tests normalement :
//! cargo test --release --test golden_images
//!
//! # Régénérer toutes les goldens (à faire APRÈS avoir vérifié qu'une
//! # modification est intentionnelle, et avant de la commit) :
//! FRACTALL_UPDATE_GOLDENS=1 cargo test --release --test golden_images
//! ```
//!
//! ## Couverture
//!
//! Les cas exercent les paths critiques :
//! - `mandelbrot_default` : Mandelbrot vue par défaut, path f64 standard
//! - `julia_default` : Julia, path f64 standard
//! - `burning_ship_default` : BS, path f64 standard
//! - `tricorn_default` : Tricorn, path f64 standard
//! - `celtic_default` : Celtic, path f64 standard
//! - `multibrot_pow3` : Multibrot puissance 3 (path bytecode-éligible)
//! - `mandelbrot_perturb_1e6` : Mandelbrot perturbation zoom 1e6 (path GMP+BLA)
//! - `burning_ship_perturb_1e3` : BS perturbation zoom 1e3 (path nonconformal BLA)
//! - `newton_default` : Newton (type qui NE compile PAS en bytecode → reste sur path dédié)
//! - `mandelbrot_minibrot_1e8` : Mandelbrot zoom 1e8 sur minibrot (BLA + glitch detect)
//! - `mandelbrot_e10` / `mandelbrot_e15` / `mandelbrot_e20` : zooms intermédiaires
//!   (G6) — path perturbation par défaut `bytecode_f64` aux frontières 1e10/1e15/1e20
//! - `mandelbrot_deep_e113` : Mandelbrot deep zoom 5e113 (toml/e113.toml) — stress GMP+BLA

use std::path::{Path, PathBuf};
use std::process::Command;

/// Définition d'un cas de test golden.
struct Case {
    /// Nom du cas (utilisé pour le nom de fichier `tests/golden/<name>.png`).
    name: &'static str,
    /// Arguments CLI à passer (sans `--output`, ajouté automatiquement).
    args: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case {
        name: "mandelbrot_default",
        args: &["--type", "3", "--width", "160", "--height", "100", "--iterations", "300"],
    },
    Case {
        name: "julia_default",
        args: &["--type", "4", "--width", "160", "--height", "100", "--iterations", "300"],
    },
    Case {
        name: "burning_ship_default",
        args: &["--type", "13", "--width", "160", "--height", "100", "--iterations", "300"],
    },
    Case {
        name: "tricorn_default",
        args: &["--type", "14", "--width", "160", "--height", "100", "--iterations", "300"],
    },
    Case {
        name: "celtic_default",
        args: &["--type", "19", "--width", "160", "--height", "100", "--iterations", "300"],
    },
    Case {
        name: "multibrot_pow3",
        args: &[
            "--type", "23", "--multibrot-power", "3",
            "--width", "160", "--height", "100", "--iterations", "300",
        ],
    },
    Case {
        name: "mandelbrot_perturb_1e6",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x=-1.7693831791955", "--center-y=0.004236847918736",
            "--zoom", "1e6",
            "--width", "160", "--height", "100", "--iterations", "1500",
        ],
    },
    Case {
        name: "burning_ship_perturb_1e3",
        args: &[
            "--type", "13", "--algorithm", "perturbation",
            "--center-x=-1.762", "--center-y=-0.028",
            "--zoom", "1e3",
            "--width", "160", "--height", "100", "--iterations", "1000",
        ],
    },
    Case {
        name: "newton_default",
        args: &["--type", "6", "--width", "160", "--height", "100", "--iterations", "200"],
    },
    // Deep zoom Mandelbrot près d'un minibrot — exerce intensivement la
    // perturbation + BLA + glitch detection. Le path actuel (legacy glitch
    // detection) produit cette image. Tout refactor du pixel loop doit la
    // préserver au pixel près.
    Case {
        name: "mandelbrot_minibrot_1e8",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x=-1.7693831791955", "--center-y=0.004236847918736",
            "--zoom", "1e8",
            "--width", "160", "--height", "100", "--iterations", "3000",
        ],
    },
    // --- Zooms intermédiaires (G6) : verrouillent le path perturbation par
    // défaut aux frontières 1e10 / 1e15 / 1e20 (chemin `bytecode_f64`, le f64
    // s'étendant à 1e280). Comblent le trou golden entre 1e8 (minibrot) et
    // 1e50+ (escape-time deep). Tous Mandelbrot, structurés, revus visuellement.
    //
    // 1e10 : centre minibrot (même que perturb_1e6/minibrot_1e8), PASS
    // pixel-exact vs GMP pur (max_diff=0) — dendrites + mini-spirales.
    Case {
        name: "mandelbrot_e10",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x=-1.7693831791955", "--center-y=0.004236847918736",
            "--zoom", "1e10",
            "--width", "160", "--height", "100", "--iterations", "4000",
        ],
    },
    // 1e15 : centre e113 (180 digits HP), zoom sorti à 1e15 — spirale
    // structurée. Divergence vs GMP = bruit de bord dispersé (p99=0,
    // div_ratio≈0.0036, plancher f64), comme les goldens seahorse/e17.
    Case {
        name: "mandelbrot_e15",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x-hp=-1.47981577613247326072298452597877854692240725774045369689878510139864920741002293820250517329282011227363313053159203914640783415609608168660705123082446357179491909705403381200",
            "--center-y-hp=-0.00063911193261361727152139632255671572957303918943984736047394936471220951961813321928573067036466151147195436388486168819318341208023229522609015461543581599807510715681229605",
            "--zoom", "1e15",
            "--width", "160", "--height", "100", "--iterations", "6000",
        ],
    },
    // 1e20 : même centre e113, zoom 1e20 — étoile multi-spirale symétrique.
    // Bruit de bord dispersé vs GMP (p99=0, div_ratio≈0.005).
    Case {
        name: "mandelbrot_e20",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x-hp=-1.47981577613247326072298452597877854692240725774045369689878510139864920741002293820250517329282011227363313053159203914640783415609608168660705123082446357179491909705403381200",
            "--center-y-hp=-0.00063911193261361727152139632255671572957303918943984736047394936471220951961813321928573067036466151147195436388486168819318341208023229522609015461543581599807510715681229605",
            "--zoom", "1e20",
            "--width", "160", "--height", "100", "--iterations", "10000",
        ],
    },
    // Deep zoom Mandelbrot 5e113 (cf. toml/e113.toml) — exerce le path GMP
    // reference + BLA en très haute précision. Plusieurs dizaines de secondes
    // de rendu : c'est le but, on veut tester ce régime profond.
    Case {
        name: "mandelbrot_deep_e113",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x-hp=-1.47981577613247326072298452597877854692240725774045369689878510139864920741002293820250517329282011227363313053159203914640783415609608168660705123082446357179491909705403381200",
            "--center-y-hp=-0.00063911193261361727152139632255671572957303918943984736047394936471220951961813321928573067036466151147195436388486168819318341208023229522609015461543581599807510715681229605",
            "--zoom", "5e113",
            "--width", "160", "--height", "100", "--iterations", "35494",
        ],
    },
    // Deep escape-time 1e50 (toml/e50.toml) — verrouille le fix G2 rebase-at-end :
    // sans lui, la référence escape-time tronquée → fallback GMP / image uniforme.
    // (--toml résolu depuis la racine du package, CWD du test cargo.)
    Case {
        name: "mandelbrot_e50",
        args: &["--toml", "toml/e50.toml", "--width", "160", "--height", "100"],
    },
    // Deep escape-time 1e1000 (toml/e1000.toml) — idem, régime extrême.
    Case {
        name: "mandelbrot_e1000",
        args: &["--toml", "toml/e1000.toml", "--width", "160", "--height", "100"],
    },
    // Mid-range 1.5e115 (toml/glitch_test_2.toml) — VERROU de la troncature
    // atom-domain MID-RANGE (2026-07-12, G2) : réf intérieure 250 k iters coupée
    // à la période atom ~1143, cyclée par rebase-at-end du pixel loop f64.
    // Régression du guard BLA « lands_on_ref_end » (pixel_loop.rs) = le pas
    // direct part du graze |Z[end]|~4e-58 puis le rebase absorbe δ en f64 →
    // image intérieure uniforme (tout noir) → cette golden casse massivement.
    Case {
        name: "mandelbrot_glitch_test_2_atom",
        args: &["--toml", "toml/glitch_test_2.toml", "--width", "160", "--height", "100"],
    },
    // Deep INTÉRIEUR 1e1200 (toml/e1200.toml, 45 k iters) — VERROU de la troncature
    // atom-domain ON PAR DÉFAUT (2026-07-11). Structure en étoile (matche F3 à
    // ~3e-4 rel avec atom). Sans atom, le path deep-interior par défaut (réf pleine
    // + rebase-at-end) DIVERGE de F3 (rel Δ 0.044 % ici, jusqu'à wfs_mb rendu tout
    // noir). Régression du cyclage atom = cette golden casse. Rapide (45 k iters).
    Case {
        name: "mandelbrot_e1200_interior",
        args: &["--toml", "toml/e1200.toml", "--width", "160", "--height", "160"],
    },
    // Cusp -0.75, zoom 2.8e10 (image utilisateur) — verrouille le fix G3
    // `max_perturb_iterations` : régression = anneaux concentriques. Args
    // explicites → chemin par défaut (cap 1024 → clamp à iteration_max).
    Case {
        name: "mandelbrot_cusp_m075",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x-hp=-0.749996225516317",
            "--center-y-hp=-0.004086150721841209",
            "--zoom", "2.8172915379e10",
            "--width", "160", "--height", "100", "--iterations", "2500",
        ],
    },
    // Deep périodique 1.55e85 (toml/floral_fantasy.toml) — verrouille le fix G3
    // tolérance period-detection : avant, un faux-positif (graze) déclenchait la
    // troncation périodique → image uniforme en chemin par défaut. Doit rester
    // structuré (== NO_PERIOD == F3).
    Case {
        name: "mandelbrot_floral",
        args: &["--toml", "toml/floral_fantasy.toml", "--width", "160", "--height", "100"],
    },
    // Deep 9.7e83 (toml/glitch_test_5.toml) — VERROU du fix « Brent OFF par
    // défaut ». La détection de période Brent firait un FAUX POSITIF ici →
    // troncation erronée → 75 % des pixels faux vs GMP (vérifié : période OFF =
    // PIXEL-EXACT vs GMP, période ON = 12259/16384 faux). Rendu en config défaut
    // (Brent off) : si quelqu'un ré-active Brent par défaut, ce golden ROUGIT.
    Case {
        name: "mandelbrot_glitch5",
        args: &["--toml", "toml/glitch_test_5.toml", "--width", "128", "--height", "128"],
    },
];

fn cli_binary_path() -> PathBuf {
    // Cargo expose le chemin du binaire pour les integration tests.
    PathBuf::from(env!("CARGO_BIN_EXE_fractall-cli"))
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn temp_output() -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("fractall_golden_{}_{}.png", pid, nanos))
}

/// Décode un PNG et retourne ses pixels RGBA + dimensions.
fn read_pixels(path: &Path) -> (u32, u32, Vec<u8>) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("Impossible d'ouvrir {}: {}", path.display(), e));
    let rgba = img.to_rgba8();
    (rgba.width(), rgba.height(), rgba.into_raw())
}

fn render_case(case: &Case) -> PathBuf {
    let output = temp_output();
    let status = Command::new(cli_binary_path())
        .args(case.args)
        .arg("--output")
        .arg(&output)
        .status()
        .unwrap_or_else(|e| panic!("Échec du lancement CLI pour '{}': {}", case.name, e));
    assert!(
        status.success(),
        "fractall-cli a échoué (code {:?}) pour le cas '{}'",
        status.code(),
        case.name
    );
    assert!(
        output.exists(),
        "Le binaire n'a pas produit le PNG attendu pour '{}'",
        case.name
    );
    output
}

#[test]
fn golden_image_corpus() {
    let update = std::env::var("FRACTALL_UPDATE_GOLDENS").is_ok();
    let golden_dir = golden_dir();
    if update {
        std::fs::create_dir_all(&golden_dir).expect("création tests/golden");
    }

    let mut failures: Vec<String> = Vec::new();
    let mut updated: Vec<String> = Vec::new();
    let mut created: Vec<String> = Vec::new();

    for case in CASES {
        let actual_path = render_case(case);
        let golden_path = golden_dir.join(format!("{}.png", case.name));

        let (aw, ah, actual) = read_pixels(&actual_path);

        if update {
            // Écrase la golden avec la nouvelle image, sans tester.
            std::fs::copy(&actual_path, &golden_path)
                .unwrap_or_else(|e| panic!("écriture golden {}: {}", golden_path.display(), e));
            updated.push(case.name.to_string());
        } else if !golden_path.exists() {
            // Première exécution : on crée la golden et on note. Pas d'échec.
            std::fs::copy(&actual_path, &golden_path)
                .unwrap_or_else(|e| panic!("écriture golden {}: {}", golden_path.display(), e));
            created.push(case.name.to_string());
        } else {
            let (gw, gh, golden) = read_pixels(&golden_path);
            if (aw, ah) != (gw, gh) {
                failures.push(format!(
                    "[{}] dimensions diffèrent: golden={}×{} vs actual={}×{}",
                    case.name, gw, gh, aw, ah
                ));
            } else if actual != golden {
                let diff_px = actual
                    .chunks_exact(4)
                    .zip(golden.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count();
                let total_px = (aw as usize) * (ah as usize);
                failures.push(format!(
                    "[{}] {}/{} pixels diffèrent ({:.3}%). Si ce changement est intentionnel: \
                     `FRACTALL_UPDATE_GOLDENS=1 cargo test --release --test golden_images`",
                    case.name,
                    diff_px,
                    total_px,
                    100.0 * diff_px as f64 / total_px as f64
                ));
            }
        }

        let _ = std::fs::remove_file(&actual_path);
    }

    if !updated.is_empty() {
        eprintln!("Goldens mises à jour ({}): {}", updated.len(), updated.join(", "));
    }
    if !created.is_empty() {
        eprintln!(
            "Goldens créées pour la première fois ({}): {}",
            created.len(),
            created.join(", ")
        );
    }
    if !failures.is_empty() {
        panic!(
            "Régressions détectées ({}/{}) :\n  - {}",
            failures.len(),
            CASES.len(),
            failures.join("\n  - ")
        );
    }
}
