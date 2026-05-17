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
//! - `tricorn_perturb_1e2` : Tricorn perturbation zoom 1e2 (path GMP)
//! - `newton_default` : Newton (type qui NE compile PAS en bytecode → reste sur path dédié)

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
        name: "tricorn_perturb_1e2",
        args: &[
            "--type", "14", "--algorithm", "perturbation",
            "--center-x=-0.5", "--center-y=0.1",
            "--zoom", "1e2",
            "--width", "160", "--height", "100", "--iterations", "800",
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
    // Zoom moyennement profond Burning Ship — path nonconformal BLA.
    Case {
        name: "burning_ship_zoom_1e5",
        args: &[
            "--type", "13", "--algorithm", "perturbation",
            "--center-x=-1.7625", "--center-y=-0.0289",
            "--zoom", "1e5",
            "--width", "160", "--height", "100", "--iterations", "2000",
        ],
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
