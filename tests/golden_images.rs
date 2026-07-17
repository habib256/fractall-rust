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
    /// Variables d'environnement posées sur le process CLI (épingle un path
    /// précis quand le défaut est un routage auto — ex. `FRACTALL_HARMONIC_LA=0`
    /// pour garder un verrou sur le path BLA).
    envs: &'static [(&'static str, &'static str)],
}

/// Cas nominal : aucun env, comportement par défaut du CLI.
const NO_ENV: &[(&str, &str)] = &[];

const CASES: &[Case] = &[
    Case {
        name: "mandelbrot_default",
        args: &["--type", "3", "--width", "160", "--height", "100", "--iterations", "300"],
        envs: NO_ENV,
    },
    Case {
        name: "julia_default",
        args: &["--type", "4", "--width", "160", "--height", "100", "--iterations", "300"],
        envs: NO_ENV,
    },
    Case {
        name: "burning_ship_default",
        args: &["--type", "13", "--width", "160", "--height", "100", "--iterations", "300"],
        envs: NO_ENV,
    },
    Case {
        name: "tricorn_default",
        args: &["--type", "14", "--width", "160", "--height", "100", "--iterations", "300"],
        envs: NO_ENV,
    },
    Case {
        name: "celtic_default",
        args: &["--type", "19", "--width", "160", "--height", "100", "--iterations", "300"],
        envs: NO_ENV,
    },
    Case {
        name: "multibrot_pow3",
        args: &[
            "--type", "23", "--multibrot-power", "3",
            "--width", "160", "--height", "100", "--iterations", "300",
        ],
        envs: NO_ENV,
    },
    Case {
        name: "mandelbrot_perturb_1e6",
        args: &[
            "--type", "3", "--algorithm", "perturbation",
            "--center-x=-1.7693831791955", "--center-y=0.004236847918736",
            "--zoom", "1e6",
            "--width", "160", "--height", "100", "--iterations", "1500",
        ],
        envs: NO_ENV,
    },
    Case {
        name: "burning_ship_perturb_1e3",
        args: &[
            "--type", "13", "--algorithm", "perturbation",
            "--center-x=-1.762", "--center-y=-0.028",
            "--zoom", "1e3",
            "--width", "160", "--height", "100", "--iterations", "1000",
        ],
        envs: NO_ENV,
    },
    Case {
        name: "newton_default",
        args: &["--type", "6", "--width", "160", "--height", "100", "--iterations", "200"],
        envs: NO_ENV,
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
        envs: NO_ENV,
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
        envs: NO_ENV,
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
        envs: NO_ENV,
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
        envs: NO_ENV,
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
        envs: NO_ENV,
    },
    // Deep escape-time 1e50 (toml/e50.toml) — verrouille le fix G2 rebase-at-end :
    // sans lui, la référence escape-time tronquée → fallback GMP / image uniforme.
    // (--toml résolu depuis la racine du package, CWD du test cargo.)
    Case {
        name: "mandelbrot_e50",
        args: &["--toml", "toml/e50.toml", "--width", "160", "--height", "100"],
        envs: NO_ENV,
    },
    // Deep escape-time 1e1000 (toml/e1000.toml) — idem, régime extrême.
    Case {
        name: "mandelbrot_e1000",
        args: &["--toml", "toml/e1000.toml", "--width", "160", "--height", "100"],
        envs: NO_ENV,
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
        // ÉPINGLÉ sur BLA : depuis le routage harmonic auto (G9.3), ce cas
        // (period0=35) routerait harmonic par défaut — or ce verrou protège le
        // guard BLA `lands_on_ref_end`, qui doit rester couvert. Le path
        // harmonic auto a SON verrou : `mandelbrot_glitch_test_2_harmonic`.
        envs: &[("FRACTALL_HARMONIC_LA", "0")],
    },
    // MÊME vue, routage par défaut (G9.3) : period0=35 ≤ 100 → le wisdom route
    // Harmonic LLA (`path=bytecode_f64_harmonic_lla`). Verrou du routage auto
    // ET de l'évaluateur LA. Adjugé vs GMP pur (96², 2026-07-15) : harmonic
    // WARN max_diff=11 div_ratio=0.00033 vs BLA max_diff=38 div_ratio=0.00043
    // — le path routé est PLUS PROCHE de la vérité que BLA sur ce cas.
    // Régression du routage (cas plus routé → pixels BLA) = cette golden casse.
    Case {
        name: "mandelbrot_glitch_test_2_harmonic",
        args: &["--toml", "toml/glitch_test_2.toml", "--width", "160", "--height", "100"],
        envs: NO_ENV,
    },
    // Deep INTÉRIEUR 1e1200 (toml/e1200.toml, 45 k iters) — VERROU de la troncature
    // atom-domain ON PAR DÉFAUT (2026-07-11). Structure en étoile (matche F3 à
    // ~3e-4 rel avec atom). Sans atom, le path deep-interior par défaut (réf pleine
    // + rebase-at-end) DIVERGE de F3 (rel Δ 0.044 % ici, jusqu'à wfs_mb rendu tout
    // noir). Régression du cyclage atom = cette golden casse. Rapide (45 k iters).
    Case {
        name: "mandelbrot_e1200_interior",
        args: &["--toml", "toml/e1200.toml", "--width", "160", "--height", "160"],
        envs: NO_ENV,
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
        envs: NO_ENV,
    },
    // Deep périodique 1.55e85 (toml/floral_fantasy.toml) — verrouille le fix G3
    // tolérance period-detection : avant, un faux-positif (graze) déclenchait la
    // troncation périodique → image uniforme en chemin par défaut. Doit rester
    // structuré (== NO_PERIOD == F3).
    Case {
        name: "mandelbrot_floral",
        args: &["--toml", "toml/floral_fantasy.toml", "--width", "160", "--height", "100"],
        envs: NO_ENV,
    },
    // Deep 9.7e83 (toml/glitch_test_5.toml) — VERROU du fix « Brent OFF par
    // défaut ». La détection de période Brent firait un FAUX POSITIF ici →
    // troncation erronée → 75 % des pixels faux vs GMP (vérifié : période OFF =
    // PIXEL-EXACT vs GMP, période ON = 12259/16384 faux). Rendu en config défaut
    // (Brent off) : si quelqu'un ré-active Brent par défaut, ce golden ROUGIT.
    Case {
        name: "mandelbrot_glitch5",
        args: &["--toml", "toml/glitch_test_5.toml", "--width", "128", "--height", "128"],
        envs: NO_ENV,
    },
    // VERROU >512² (durcissement couverture 2026-07-17). SEUL golden au-dessus
    // du seuil `small_image` (max_dim > 512) → exerce le path de correction
    // perturbation qui n'est JAMAIS testé à ≤256² : à cette résolution la réf
    // intérieure frôlant zéro (centre -0.5622-0.6428i, zoom 2.66e10) flagge
    // 36 % de pixels → escalade tier dd (glitch_ratio > 0.30). Vérifié
    // pixel-exact vs GMP à la génération. Si le gate `!bytecode_path` du 2ᵉ bloc
    // secondary-refs (mod.rs ~1642) régresse, ou l'escalade dd casse, ce golden
    // ROUGIT (à ≤512² le bug était INVISIBLE — c'est précisément le trou de
    // couverture que ce cas comble). ⚠️ Lent (~14 s, escalade dd) : seul cas
    // >512², assumé pour couvrir la classe de bug réf-intérieure.
    // VERROU G4 jalon 2 — hybride multi-phase RENDU. `--phases mandelbrot,
    // burning_ship` (Mandel-Ship alternant) via le path f64 standard (seul à
    // cycler les phases). Image genuine (≠ Mandelbrot ET ≠ Burning Ship, vérifié
    // 4440/2974 px). Rougit si le câblage hybride (formula_for_params, cyclage
    // iterate_bytecode_f64) ou le routage StandardF64 régresse. L'invariant
    // [M,M]==M est verrouillé séparément par le unit test
    // `hybrid_MM_iterates_identically_to_single_M`.
    Case {
        name: "mandelbrot_hybrid_burningship",
        args: &[
            "--phases", "mandelbrot,burning_ship",
            "--width", "160", "--height", "100",
            "--center-x=-0.4", "--center-y=-0.35", "--zoom=1.2",
            "--iterations", "400",
        ],
        envs: NO_ENV,
    },
    Case {
        name: "mandelbrot_interior_ref_640",
        args: &[
            "--type", "3", "--width", "640", "--height", "438",
            "--center-x-hp=-0.5622000453758329685166664054580872759302643586355318252834150515100478622742054",
            "--center-y-hp=-0.6428513501859345446359399567738260824595364535984125281199143802544005459298801",
            "--zoom=2.6556311116408318e10", "--iterations", "2500",
        ],
        envs: NO_ENV,
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
        // Déterminisme cross-machine (G9.5) : les goldens sont pixel-exacts et
        // valident le path CPU. `--no-gpu` force le CPU quelle que soit
        // l'auto-sélection device du wisdom (qui, sur une machine à GPU f64
        // rapide, pourrait router une frame deep-perturbation vers le GPU →
        // sortie ≠ bit-à-bit du CPU). Sans ça les goldens seraient
        // machine-dépendants.
        .arg("--no-gpu")
        .envs(case.envs.iter().copied())
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

/// VERROU G4 jalon 3 — hybride multi-phase en PERTURBATION. À un zoom deep dans
/// la bande f64-perturbation (~3e10, `path=bytecode_f64` via
/// `iterate_pixel_unified_multi_phase`), l'hybride `[Mandelbrot, Mandelbrot]`
/// (2 phases identiques z²+c) DOIT rendre PIXEL-EXACT comme le Mandelbrot
/// mono-phase perturbation. Prouve que le cyclage de phases + le rebasing du
/// path perturbation multi-phase sont corrects, SANS ground truth GMP externe
/// (le GMP par-pixel ne cycle pas les phases). Rougit si le routage multi-phase
/// (delta.rs) ou `select_algorithm` (bande hybride) régresse.
#[test]
fn hybrid_mm_equals_mandelbrot_deep_perturbation() {
    let common: &[&str] = &[
        "--width", "200", "--height", "150",
        "--center-x-hp=-0.743643887037158704752191506114774",
        "--center-y-hp=0.131825904205311970493132056385139",
        "--zoom=3e10", "--iterations", "3000", "--no-gpu",
    ];
    let render = |extra: &[&str]| -> (u32, u32, Vec<u8>) {
        let out = temp_output();
        let status = Command::new(cli_binary_path())
            .args(common)
            .args(extra)
            .arg("--output")
            .arg(&out)
            .status()
            .expect("CLI launch");
        assert!(status.success(), "render échoué pour {extra:?}");
        let px = read_pixels(&out);
        let _ = std::fs::remove_file(&out);
        px
    };
    // Plain Mandelbrot (type 3, perturbation mono-phase + BLA).
    let m = render(&["--type", "3"]);
    // Hybride [M,M] (perturbation MULTI-phase, sans BLA).
    let mm = render(&["--phases", "mandelbrot,mandelbrot"]);
    assert_eq!((m.0, m.1), (mm.0, mm.1), "dims");
    let diff = m.2.iter().zip(&mm.2).filter(|(a, b)| a != b).count();
    assert_eq!(
        diff, 0,
        "[M,M] deep (perturbation multi-phase) doit être pixel-exact == Mandelbrot ; {diff} octets diffèrent"
    );
}

/// **G4 jalon 4 (deep exp)** : au-delà de 1e280 (`wants_exp`, ComplexExp),
/// l'hybride `[Mandelbrot, Mandelbrot]` DOIT rendre PIXEL-EXACT comme le
/// Mandelbrot mono-phase perturbation exp. Center e1000 (`toml/e1000.toml`,
/// `pixel_size` underflow f64 -> tier exp), 32000 iters pour de VRAIS escapes
/// (a faible iter la tuile est uniforme, invariant trivial). Prouve le cyclage
/// de phases + rebasing F3 en FloatExp de `iterate_pixel_unified_exp_multi_phase`,
/// sans ground truth GMP externe (le GMP par-pixel ne cycle pas les phases).
#[test]
fn hybrid_mm_equals_mandelbrot_deep_exp_e1000() {
    let common: &[&str] = &[
        "--width", "80", "--height", "80",
        "--center-x-hp=-1.941576032416438467025801020715296345583631070333763697155363215369995323174821426569343939280976688941799033949553513877360049760854818438847807512062527483938432108334379682374425724538963945203811384410447188902539280186443207724592660257192245282122435497500512116978510994235729842140426212659047449275392649582144536809606048256661942015415293902082484897583908489968914729856261014595046784170486406281774634659010437551209316119397222613229217917022405856521450471398240958803454546592237981445377671184500660547408817261831894272352819877027084299780680294324572817973522425697710448544266296164872469941207915318387606736780997353983062554261604503546149292936642204630313050512074878556249827082472838018479056231300425717628324576406385386489046596333813954944872237432232584712794199385182029966095629800079433529594932724703432396513169426246696336919279357704478424514838526831184178560128965986916218888035003760618018546383024162868702909340031454797870830171151871458976156454695759759455242167153480852905329612928",
        "--center-y-hp=0.007616985717772811262124668760622981437662236623940733858977202402020949904597675982567903438152747187001453792570531900379945655089068104542887520569666325496854943970482334766465568902178471662205610335655386327591088355630793583666899303983220376324437451114866304483466406624113681671822911479962978243988037880036085177763383839177390736670205923960451531786620698624310112277206323661779555548745795026511137379365757720941649670500838538041663501778850301672389321517369030732380361504406649221281161353695980774304817750907225115318738323761935375270212445755332408668293878051992878181731210653759907898838467665702947448379879550359982270646382467980986879289206553850989072075578963952995904652178779463868993619754936296821786991785780531050125323823377470573405525813057794363468880870906692986558129005828655981351971538509527655719324861000343097111273792722058002993775507514675899953901227838493650982772056135274894875568039070393925760084216277141057220628182876163008403838153533411744597352540385947641225916772",
        "--zoom=1e1000", "--iterations", "32000", "--no-gpu",
    ];
    let render = |extra: &[&str]| -> (u32, u32, Vec<u8>) {
        let out = temp_output();
        let status = Command::new(cli_binary_path())
            .args(common)
            .args(extra)
            .arg("--output")
            .arg(&out)
            .status()
            .expect("CLI launch");
        assert!(status.success(), "render echoue pour {extra:?}");
        let px = read_pixels(&out);
        let _ = std::fs::remove_file(&out);
        px
    };
    let m = render(&["--type", "3"]);
    let mm = render(&["--phases", "mandelbrot,mandelbrot"]);
    assert_eq!((m.0, m.1), (mm.0, mm.1), "dims");
    // Garde-fou : la tuile doit contenir de la structure (escapes varies), sinon
    // l'egalite est triviale. On exige > 1 valeur de canal rouge distincte.
    let reds: std::collections::HashSet<u8> = m.2.iter().step_by(3).copied().collect();
    assert!(reds.len() > 1, "tuile uniforme -- invariant trivial (reds={})", reds.len());
    let diff = m.2.iter().zip(&mm.2).filter(|(a, b)| a != b).count();
    assert_eq!(
        diff, 0,
        "[M,M] deep-exp (perturbation multi-phase ComplexExp) doit etre pixel-exact == Mandelbrot ; {diff} octets different"
    );
}

/// **G4 jalon 5d (atom-domain hybride + BLA par phase, tier f64)** : a e50
/// (263k iters, tier f64, BLA massive, refs atom-tronquees par le critere
/// mat2), l'hybride `[M,M]` DOIT rendre PIXEL-EXACT comme le Mandelbrot
/// mono-phase. Avant 5d les troncatures differaient ([M] atom complexe,
/// [M,M] refs pleines) -> 6 px de bruit de rebase ; le critere mat2 (J
/// conforme => identique au critere complexe) retablit l'exactitude.
/// Couvre d'un coup : refs par phase + tracking (5a), tables BLA cyclees +
/// prewarm (5b), troncature atom mat2 (5d).
#[test]
fn hybrid_mm_equals_mandelbrot_deep_f64_e50() {
    let common: &[&str] = &[
        "--width", "96", "--height", "96",
        "--center-x-hp=-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000",
        "--center-y-hp=-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880",
        "--zoom=1e50", "--iterations", "263010", "--no-gpu",
    ];
    let render = |extra: &[&str]| -> (u32, u32, Vec<u8>) {
        let out = temp_output();
        let status = Command::new(cli_binary_path())
            .args(common)
            .args(extra)
            .arg("--output")
            .arg(&out)
            .status()
            .expect("CLI launch");
        assert!(status.success(), "render echoue pour {extra:?}");
        let px = read_pixels(&out);
        let _ = std::fs::remove_file(&out);
        px
    };
    let m = render(&["--type", "3"]);
    let mm = render(&["--phases", "mandelbrot,mandelbrot"]);
    assert_eq!((m.0, m.1), (mm.0, mm.1), "dims");
    let reds: std::collections::HashSet<u8> = m.2.iter().step_by(3).copied().collect();
    assert!(reds.len() > 1, "tuile uniforme -- invariant trivial");
    let diff = m.2.iter().zip(&mm.2).filter(|(a, b)| a != b).count();
    assert_eq!(
        diff, 0,
        "[M,M] e50 (BLA par phase + atom mat2) doit etre pixel-exact == Mandelbrot ; {diff} octets different"
    );
}
