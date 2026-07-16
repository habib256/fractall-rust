use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rug::Float;

use crate::fractal::{FractalParams, FractalResult, FractalType, OutColoringMode, PlaneTransform};
use crate::fractal::iterations::iterate_point;
use crate::fractal::wisdom;
use crate::fractal::orbit_traps::OrbitData;
use crate::fractal::gmp::{complex_from_xy, complex_to_complex64, iterate_point_mpc, MpcParams};
use crate::fractal::{render_von_koch, render_dragon};
use crate::fractal::lyapunov::{render_lyapunov_cancellable, render_lyapunov_mpc_cancellable};
use crate::fractal::buddhabrot::{
    render_buddhabrot_cancellable,
    render_nebulabrot_cancellable,
    render_buddhabrot_mpc_cancellable,
    render_nebulabrot_mpc_cancellable,
    render_antibuddhabrot_cancellable,
    render_antibuddhabrot_mpc_cancellable,
};
use crate::fractal::perturbation::{
    render_perturbation_with_cache, ReferenceOrbitCache,
};
use crate::fractal::xaos::XaosMap;
use crate::render::tiles::{self, TileGrid, TileOpts, TileUpdate};

/// Calcule la matrice d'itérations et la matrice des valeurs finales de z
/// pour une fractale escape-time (ou algorithme spécial).
///
/// Retourne un tuple (iterations, zs) où :
/// - `iterations.len() == width * height`
/// - `zs.len() == width * height`
///
/// Le calcul est parallélisé sur plusieurs cœurs CPU avec rayon.
///
/// **Chemin de rendu UNIQUE** : cette fonction (utilisée par le CLI) délègue au
/// même dispatcher que la GUI, `render_escape_time_cancellable_with_reuse`. Il
/// n'y a donc plus qu'UNE seule implémentation de la sélection
/// type→algorithme (perturbation / GMP / f64 / spéciaux). Le CLI = rendu unique
/// plein cadre (pas de cancel, pas de reuse, pas de cache). Cf. CLAUDE.md
/// §« Chemin de rendu unique CLI ↔ GUI ».
#[allow(dead_code)] // utilisé par le bin fractall-cli (main.rs), pas par gui/quality
pub fn render_escape_time(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let mut orbit_cache: Option<Arc<ReferenceOrbitCache>> = None;
    render_escape_time_cancellable_with_reuse(
        params,
        &Arc::new(AtomicBool::new(false)),
        None,
        &mut orbit_cache,
        None,
        None,
    )
    .map(|(i, z, _orbits, _distances)| (i, z))
    .unwrap_or_else(|| (Vec::new(), Vec::new()))
}

struct ReuseData<'a> {
    iterations: &'a [u32],
    zs: &'a [Complex64],
    width: u32,
    ratio: u32,
}

fn build_reuse<'a>(
    params: &FractalParams,
    reuse: Option<(&'a [u32], &'a [Complex64], u32, u32)>,
) -> Option<ReuseData<'a>> {
    // Disable pixel reuse for coloring modes that need per-pixel orbit/distance data,
    // since reused pixels don't carry this data and would create checkerboard artifacts.
    let needs_extra_data = matches!(
        params.out_coloring_mode,
        OutColoringMode::Distance | OutColoringMode::DistanceAO | OutColoringMode::Distance3D
        | OutColoringMode::OrbitTraps | OutColoringMode::Wings
    );
    if needs_extra_data {
        return None;
    }
    let (iterations, zs, width, height) = reuse?;
    if width == 0 || height == 0 {
        return None;
    }
    let expected_len = (width * height) as usize;
    if iterations.len() != expected_len || zs.len() != expected_len {
        return None;
    }
    if params.width % width != 0 || params.height % height != 0 {
        return None;
    }
    let ratio_x = params.width / width;
    let ratio_y = params.height / height;
    if ratio_x < 2 || ratio_y < 2 || ratio_x != ratio_y {
        return None;
    }
    Some(ReuseData {
        iterations,
        zs,
        width,
        ratio: ratio_x,
    })
}

/// Version annulable du rendu escape-time.
/// Retourne None si annulé, Some(iterations, zs) sinon (orbites/distances ignorés pour compatibilité).
#[allow(dead_code)]
pub fn render_escape_time_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<(Vec<u32>, Vec<Complex64>)> {
    let mut orbit_cache: Option<Arc<ReferenceOrbitCache>> = None;
    render_escape_time_cancellable_with_reuse(params, cancel, None, &mut orbit_cache, None, None)
        .map(|(i, z, _o, _d)| (i, z))
}

/// Résultat du rendu escape-time (iterations, zs, orbites optionnelles, distances optionnelles).
pub type EscapeTimeResult = (
    Vec<u32>,
    Vec<Complex64>,
    Vec<Option<OrbitData>>,
    Vec<f64>,
);

/// Version annulable du rendu escape-time avec réutilisation d'une passe précédente.
/// Les points déjà calculés sont réutilisés quand les résolutions s'alignent.
/// Retourne (iterations, zs, orbits, distances) : orbits remplies pour f64 si enable_orbit_traps,
/// distances remplies pour perturbation si enable_distance_estimation.
///
/// **Dispatcher de rendu UNIQUE** partagé par le CLI (`render_escape_time`) et la
/// GUI (chaque passe progressive). Toute sélection type→algorithme passe par ici
/// — il n'existe plus de logique de dispatch dupliquée ailleurs.
///
/// `orbit_cache` est un in/out : en entrée l'orbite référence réutilisable (passe
/// précédente / re-pan), en sortie l'orbite mise à jour (renseignée uniquement
/// par le path perturbation). Le CLI passe `&mut None` (rendu unique).
///
/// `xaos` (G10.4) : mapping inter-frame optionnel (colonnes/lignes de la frame
/// précédente réutilisables) — les pixels mappés sont copiés au lieu d'être
/// calculés. Construit par la GUI via `fractal::xaos::build_map` (gates +
/// fingerprint validés là-bas) ; CLI/quality passent `None`.
///
/// `tiles` (G10.5) : options de la file de tuiles priorité-centre (point de
/// priorité normalisé + sink de streaming intra-passe). `None` (CLI/quality)
/// ⇒ priorité centre, pas de streaming — mêmes pixels dans tous les cas
/// (l'ordre d'exécution ne change pas les valeurs calculées).
pub fn render_escape_time_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<(&[u32], &[Complex64], u32, u32)>,
    orbit_cache: &mut Option<Arc<ReferenceOrbitCache>>,
    xaos: Option<&XaosMap>,
    tiles: Option<&TileOpts>,
) -> Option<EscapeTimeResult> {
    // Garde-fou : un mapping construit pour d'autres dimensions serait indexé
    // hors bornes par les boucles pixel → on l'ignore (rendu complet).
    let xaos = xaos.filter(|m| {
        m.src_col.len() == params.width as usize && m.src_row.len() == params.height as usize
    });
    // INVARIANT G10.4b : écho XaoS et réutilisation basse-résolution inter-passes
    // sont mutuellement EXCLUSIFS. Le `reuse` copie le pixel grossier dont le
    // centre est décalé de (ratio−1)/2 px cible ((i/r + 0.5)/(n/r) ≠ (i+0.5)/n) —
    // une approximation acceptable pour accélérer une passe, mais qui
    // contaminerait les colonnes/lignes que le map XaoS déclare FRAÎCHES
    // (col_err = 0, col_exact = true). Le raffinement union fait confiance à
    // ces axes : les pixels frais doivent être réellement calculés. L'écho
    // remplit déjà le rôle d'accélérateur quand un map est présent.
    let reuse = if xaos.is_some() { None } else { reuse };
    // Path perturbation unifié : passe par `render_perturbation_with_cache` (le
    // même coeur que le CLI et la GUI), en threadant le cache d'orbite.
    let mut run_perturbation = |reuse: Option<(&[u32], &[Complex64], u32, u32)>| {
        let (res, cache) = render_perturbation_with_cache(
            params,
            cancel,
            reuse,
            orbit_cache.as_ref(),
            xaos,
            tiles,
        )?;
        *orbit_cache = Some(cache);
        Some((res.0, res.1, Vec::new(), res.2))
    };
    // Vérifier l'annulation avant de commencer
    if cancel.load(Ordering::Relaxed) {
        return None;
    }

    // Dispatch vers les algorithmes spéciaux (pas d'orbites ni distances)
    let empty_orbits_distances = |n: usize| -> (Vec<Option<OrbitData>>, Vec<f64>) {
        (vec![None; n], vec![])
    };
    match params.fractal_type {
        FractalType::VonKoch => {
            let (i, z) = render_von_koch(params);
            let n = i.len();
            return Some((i, z, empty_orbits_distances(n).0, vec![]));
        }
        FractalType::Dragon => {
            let (i, z) = render_dragon(params);
            let n = i.len();
            return Some((i, z, empty_orbits_distances(n).0, vec![]));
        }
        FractalType::Buddhabrot => {
            return if params.use_gmp {
                render_buddhabrot_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_buddhabrot_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        FractalType::Lyapunov => {
            return if params.use_gmp {
                render_lyapunov_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_lyapunov_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        FractalType::Nebulabrot => {
            return if params.use_gmp {
                render_nebulabrot_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_nebulabrot_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        FractalType::AntiBuddhabrot => {
            return if params.use_gmp {
                render_antibuddhabrot_mpc_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            } else {
                render_antibuddhabrot_cancellable(params, cancel).map(|(i, z)| {
                    let n = i.len();
                    (i, z, empty_orbits_distances(n).0, vec![])
                })
            };
        }
        _ => {}
    }

    if wisdom::perturbation_family(params.fractal_type) {
        // Sélection wisdom (source UNIQUE, G9.1) : le plan est l'INPUT du
        // dispatcher — la même fonction est consommée par le dispatch GPU
        // (`GpuRenderer::render_dispatch`, device Gpu) et les labels/passes
        // GUI. Couvre les modes forcés, le fallback plane_transform ≠ Mu et
        // l'Auto (perturbation > ~1e12, GMP > 1e16, sinon f64).
        match wisdom::select_algorithm(params, wisdom::Device::Cpu) {
            wisdom::Algorithm::Perturbation => {
                return run_perturbation(reuse);
            }
            wisdom::Algorithm::ReferenceGmp => {
                let reuse = build_reuse(params, reuse);
                return render_escape_time_gmp_cancellable_with_reuse(
                    params, cancel, reuse, xaos, tiles,
                );
            }
            wisdom::Algorithm::StandardF64 => {
                let reuse = build_reuse(params, reuse);
                return render_escape_time_f64_cancellable_with_reuse(
                    params, cancel, reuse, xaos, tiles,
                );
            }
        }
    }

    let reuse = build_reuse(params, reuse);
    if params.use_gmp {
        return render_escape_time_gmp_cancellable_with_reuse(params, cancel, reuse, xaos, tiles);
    }
    render_escape_time_f64_cancellable_with_reuse(params, cancel, reuse, xaos, tiles)
}

/// Calcule le niveau de zoom à partir des paramètres.
/// Le zoom est calculé comme base_range / pixel_size où base_range = 4.0 (plage standard Mandelbrot).
fn compute_zoom(params: &FractalParams) -> Option<f64> {
    if params.width == 0 || params.height == 0 {
        return None;
    }
    // Use max of both axes to correctly handle non-square images
    let pixel_size = (params.span_x.abs() / params.width as f64)
        .max(params.span_y.abs() / params.height as f64);
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return None;
    }
    let base_range = 4.0;
    let zoom = base_range / pixel_size;
    if !zoom.is_finite() || zoom <= 1.0 {
        return None;
    }
    Some(zoom)
}

/// Détermine si on doit utiliser GMP reference basé sur le niveau de zoom.
/// Pour des zooms de e1 à e16 (10^1 à 10^16), on utilise CPU f64.
/// Au-delà de 10^16, on bascule sur GMP reference.
pub fn should_use_gmp_reference(params: &FractalParams) -> bool {
    let zoom = match compute_zoom(params) {
        Some(z) => z,
        None => return false,
    };
    // Seuil: zoom > 10^16 → GMP reference
    zoom > 1e16
}

pub fn should_use_perturbation(params: &FractalParams, gpu_f32: bool) -> bool {
    if params.width == 0 || params.height == 0 {
        return false;
    }
    // Disable perturbation for non-Mu plane transforms
    // Perturbation relies on delta-based calculations that don't work correctly with plane transforms
    if params.plane_transform != PlaneTransform::Mu {
        return false;
    }
    if !matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::BurningShip | FractalType::Tricorn | FractalType::Multibrot
    ) {
        return false;
    }
    // Use max of both axes to correctly handle non-square images
    let pixel_size = (params.span_x.abs() / params.width as f64)
        .max(params.span_y.abs() / params.height as f64);

    if gpu_f32 {
        // En mode GPU fp32: basculer sur perturbation pour zoom > e5 (pixel_size < 1e-5)
        const GPU_PERTURBATION_THRESHOLD: f64 = 1e-5;
        return pixel_size < GPU_PERTURBATION_THRESHOLD;
    } else {
        // En mode CPU fp64: basculer sur perturbation dès zoom > ~1e12 (pixel_size < 1e-12)
        // Pour les zooms très profonds (>1e15), la référence GMP est utilisée automatiquement
        // par should_use_gmp_reference() dans le pipeline de perturbation.
        const CPU_PERTURBATION_THRESHOLD: f64 = 1e-12;
        return pixel_size < CPU_PERTURBATION_THRESHOLD;
    }
}

#[allow(dead_code)]
fn render_escape_time_f64_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<EscapeTimeResult> {
    render_escape_time_f64_cancellable_with_reuse(params, cancel, None, None, None)
}

fn render_escape_time_f64_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<ReuseData<'_>>,
    xaos: Option<&XaosMap>,
    tiles: Option<&TileOpts>,
) -> Option<EscapeTimeResult> {
    let width = params.width as usize;
    let height = params.height as usize;
    let n = width * height;
    let mut iterations = vec![0u32; n];
    let mut zs = vec![Complex64::new(0.0, 0.0); n];
    let mut orbits: Vec<Option<OrbitData>> = vec![None; n];
    let mut distances: Vec<f64> = vec![f64::INFINITY; n]; // INFINITY = pas d'estimation

    if width == 0 || height == 0 {
        return Some((iterations, zs, orbits, distances));
    }

    let need_orbits = params.enable_orbit_traps;
    let need_distances = params.enable_distance_estimation;
    // Pré-calcul de la matrice de rotation (None si rotation == 0 → no-op).
    let rot = params.transform_matrix();
    // Offset sous-pixel AA per-frame (unités de pixel, [0,0] hors AA).
    let [aa_dx, aa_dy] = params.aa_subpixel_offset;

    // G10.5 : file de tuiles priorité-centre (curseur en GUI). Les buffers
    // sont découpés en segments disjoints par tuile ; l'ordre d'exécution ne
    // change pas les valeurs (pixels indépendants).
    let priority = tiles.map(|t| t.priority).unwrap_or((0.5, 0.5));
    let sink = tiles.and_then(|t| t.sink);
    let grid = TileGrid::new(width, height, priority);
    let it_g = grid.split(&mut iterations);
    let zs_g = grid.split(&mut zs);
    let orb_g = grid.split(&mut orbits);
    let dist_g = grid.split(&mut distances);
    let slots: Vec<_> = it_g
        .into_iter()
        .zip(zs_g)
        .zip(orb_g)
        .zip(dist_g)
        .map(|(((it, z), orb), di)| std::sync::Mutex::new(Some((it, z, orb, di))))
        .collect();

    let completed = tiles::run_prioritized(&grid.order, &slots, cancel.as_ref(), &|id, tile_bufs| {
        let (mut it_rows, mut z_rows, mut orb_rows, mut dist_rows) = tile_bufs;
        let (x0, y0, tw, th) = grid.rect(id);
        for dj in 0..th {
            let j = y0 + dj;
            let y_ratio = (j as f64 + 0.5 + aa_dy) / params.height as f64;
            let dy = (y_ratio - 0.5) * params.span_y;
            let iter_row = &mut it_rows[dj];
            let z_row = &mut z_rows[dj];
            let orbit_row = &mut orb_rows[dj];
            let dist_row = &mut dist_rows[dj];
            for di in 0..tw {
                let i = x0 + di;
                let iter = &mut iter_row[di];
                let z = &mut z_row[di];
                // G10.4 : copie inter-frame XaoS (écho produit ou refine union).
                if let Some(x) = xaos {
                    if let Some(sidx) = x.source_index(i, j) {
                        if let Some(&it) = x.iterations.get(sidx) {
                            *iter = it;
                            *z = x.zs[sidx];
                            continue;
                        }
                    }
                }
                if let Some(reuse) = reuse.as_ref() {
                    let ratio = reuse.ratio as usize;
                    if j % ratio == 0 && i % ratio == 0 {
                        let src_x = i / ratio;
                        let src_y = j / ratio;
                        let src_idx = src_y * reuse.width as usize + src_x;
                        if src_idx < reuse.iterations.len() {
                            *iter = reuse.iterations[src_idx];
                            *z = reuse.zs[src_idx];
                            continue;
                        }
                    }
                }
                let x_ratio = (i as f64 + 0.5 + aa_dx) / params.width as f64;
                let dx = (x_ratio - 0.5) * params.span_x;
                let (dx_r, dy_r) = match rot {
                    Some((a, b, c, d)) => (a * dx + b * dy, c * dx + d * dy),
                    None => (dx, dy),
                };
                let z_pixel = Complex64::new(params.center_x + dx_r, params.center_y + dy_r);
                let z_pixel = params.plane_transform.transform(z_pixel);
                let FractalResult {
                    iteration,
                    z: z_final,
                    orbit,
                    distance,
                } = iterate_point(params, z_pixel);
                *iter = iteration;
                *z = z_final;
                if need_orbits {
                    orbit_row[di] = orbit;
                }
                if need_distances {
                    dist_row[di] = distance.unwrap_or(f64::INFINITY);
                }
            }
        }
        if let Some(sink) = sink {
            sink(TileUpdate {
                x0,
                y0,
                w: tw,
                h: th,
                iterations: tiles::collect_rows(&it_rows),
                zs: tiles::collect_rows(&z_rows),
                distances: if need_distances {
                    tiles::collect_rows(&dist_rows)
                } else {
                    Vec::new()
                },
            });
        }
    });
    drop(slots);

    if !completed {
        None
    } else {
        Some((iterations, zs, orbits, distances))
    }
}

#[allow(dead_code)]
fn render_escape_time_gmp_cancellable(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Option<EscapeTimeResult> {
    render_escape_time_gmp_cancellable_with_reuse(params, cancel, None, None, None)
}

fn render_escape_time_gmp_cancellable_with_reuse(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    reuse: Option<ReuseData<'_>>,
    xaos: Option<&XaosMap>,
    tiles: Option<&TileOpts>,
) -> Option<EscapeTimeResult> {
    let width = params.width as usize;
    let height = params.height as usize;
    let n = width * height;
    let mut iterations = vec![0u32; n];
    let mut zs = vec![Complex64::new(0.0, 0.0); n];

    if width == 0 || height == 0 {
        return Some((iterations, zs, vec![], vec![]));
    }

    let gmp = MpcParams::from_params(params);
    let prec = gmp.prec;

    // IMPORTANT: Utiliser les String haute précision si disponibles pour préserver la précision GMP
    // aux zooms profonds (>e16). Sinon fallback sur f64 pour compatibilité.
    let center_x = if let Some(ref cx_hp) = params.center_x_hp {
        match Float::parse(cx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_x_hp, using f64 fallback");
                Float::with_val(prec, params.center_x)
            }
        }
    } else {
        Float::with_val(prec, params.center_x)
    };
    
    let center_y = if let Some(ref cy_hp) = params.center_y_hp {
        match Float::parse(cy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_y_hp, using f64 fallback");
                Float::with_val(prec, params.center_y)
            }
        }
    } else {
        Float::with_val(prec, params.center_y)
    };
    
    let span_x = if let Some(ref sx_hp) = params.span_x_hp {
        match Float::parse(sx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse span_x_hp, using f64 fallback");
                Float::with_val(prec, params.span_x)
            }
        }
    } else {
        Float::with_val(prec, params.span_x)
    };
    
    let span_y = if let Some(ref sy_hp) = params.span_y_hp {
        match Float::parse(sy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse span_y_hp, using f64 fallback");
                Float::with_val(prec, params.span_y)
            }
        }
    } else {
        Float::with_val(prec, params.span_y)
    };

    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let half = Float::with_val(prec, 0.5);
    // Rotation : appliquée au delta pixel→centre avant d'ajouter center_x/y.
    let rot = params.transform_matrix();
    // Offset sous-pixel AA per-frame (unités de pixel, [0,0] hors AA).
    let [aa_dx, aa_dy] = params.aa_subpixel_offset;

    // G10.5 : file de tuiles priorité-centre. Les pixels GMP sont lourds
    // (100 µs–ms) → poll d'annulation par ligne de tuile en plus du poll par
    // tuile de l'exécuteur.
    let priority = tiles.map(|t| t.priority).unwrap_or((0.5, 0.5));
    let sink = tiles.and_then(|t| t.sink);
    let grid = TileGrid::new(width, height, priority);
    let it_g = grid.split(&mut iterations);
    let zs_g = grid.split(&mut zs);
    let slots: Vec<_> = it_g
        .into_iter()
        .zip(zs_g)
        .map(|(it, z)| std::sync::Mutex::new(Some((it, z))))
        .collect();

    let completed = tiles::run_prioritized(&grid.order, &slots, cancel.as_ref(), &|id, tile_bufs| {
        let (mut it_rows, mut z_rows) = tile_bufs;
        let (x0, y0, tw, th) = grid.rect(id);
        for dj in 0..th {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let j = y0 + dj;
            let mut j_f = Float::with_val(prec, j as u32);
            j_f += &half;
            if aa_dy != 0.0 {
                j_f += aa_dy;
            }
            let mut y_ratio = j_f;
            y_ratio /= &height_f;
            y_ratio -= &half;
            let mut dy = span_y.clone();
            dy *= &y_ratio;
            let iter_row = &mut it_rows[dj];
            let z_row = &mut z_rows[dj];
            for di in 0..tw {
                let i = x0 + di;
                let iter = &mut iter_row[di];
                let z = &mut z_row[di];
                // G10.4 : copie inter-frame XaoS (écho produit ou refine union).
                if let Some(x) = xaos {
                    if let Some(sidx) = x.source_index(i, j) {
                        if let Some(&it) = x.iterations.get(sidx) {
                            *iter = it;
                            *z = x.zs[sidx];
                            continue;
                        }
                    }
                }
                if let Some(reuse) = reuse.as_ref() {
                    let ratio = reuse.ratio as usize;
                    if j % ratio == 0 && i % ratio == 0 {
                        let src_x = i / ratio;
                        let src_y = j / ratio;
                        let src_idx = src_y * reuse.width as usize + src_x;
                        if src_idx < reuse.iterations.len() {
                            *iter = reuse.iterations[src_idx];
                            *z = reuse.zs[src_idx];
                            continue;
                        }
                    }
                }
                let mut i_f = Float::with_val(prec, i as u32);
                i_f += &half;
                if aa_dx != 0.0 {
                    i_f += aa_dx;
                }
                let mut x_ratio = i_f;
                x_ratio /= &width_f;
                x_ratio -= &half;
                let mut dx = span_x.clone();
                dx *= &x_ratio;
                let (xg, yg) = match rot {
                    Some((a, b, c, d)) => {
                        let dx_r = Float::with_val(prec, &dx * a) + Float::with_val(prec, &dy * b);
                        let dy_r = Float::with_val(prec, &dx * c) + Float::with_val(prec, &dy * d);
                        (
                            Float::with_val(prec, &dx_r + &center_x),
                            Float::with_val(prec, &dy_r + &center_y),
                        )
                    }
                    None => (
                        Float::with_val(prec, &dx + &center_x),
                        Float::with_val(prec, &dy + &center_y),
                    ),
                };
                // IMPORTANT: Utiliser la version GMP de plane_transform pour éviter la perte de précision
                // aux zooms profonds (>e16). La conversion GMP → f64 → GMP perdait toute la précision.
                let z_gmp = complex_from_xy(prec, xg, yg);
                let z_transformed = params.plane_transform.transform_gmp(&z_gmp, prec);
                let z_pixel = z_transformed;
                let (iter_val, z_final) = iterate_point_mpc(&gmp, &z_pixel);
                *iter = iter_val;
                *z = complex_to_complex64(&z_final);
            }
        }
        if let Some(sink) = sink {
            sink(TileUpdate {
                x0,
                y0,
                w: tw,
                h: th,
                iterations: tiles::collect_rows(&it_rows),
                zs: tiles::collect_rows(&z_rows),
                distances: Vec::new(),
            });
        }
    });
    drop(slots);

    // Re-check : un cancel observé DANS une tuile (early-return par ligne)
    // peut ne pas être vu par l'exécuteur si la file était déjà vide.
    if !completed || cancel.load(Ordering::Relaxed) {
        None
    } else {
        Some((iterations, zs, vec![None; n], vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::default_params_for_type;

    fn small_mandelbrot(width: u32, height: u32) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, width, height);
        p.iteration_max = 200;
        // Vue au bord du set : mélange intérieur/extérieur (tuiles hétérogènes).
        p.center_x = -0.7435;
        p.center_y = 0.1314;
        p.span_x = 0.05;
        p.span_y = 0.05 * height as f64 / width as f64;
        p
    }

    fn render_with_tiles(
        params: &FractalParams,
        tiles: Option<&TileOpts>,
    ) -> EscapeTimeResult {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut cache: Option<Arc<ReferenceOrbitCache>> = None;
        render_escape_time_cancellable_with_reuse(params, &cancel, None, &mut cache, None, tiles)
            .expect("rendu non annulé")
    }

    /// G10.5 : l'ordre des tuiles ne change pas les pixels — un rendu avec
    /// priorité coin == priorité centre == sans opts, bit-exact. Dimensions
    /// non multiples de la tuile pour couvrir les bords.
    #[test]
    fn tiled_render_identical_across_priorities() {
        let params = small_mandelbrot(97, 61);
        let base = render_with_tiles(&params, None);
        for prio in [(0.0, 0.0), (1.0, 1.0), (0.13, 0.87)] {
            let opts = TileOpts {
                priority: prio,
                sink: None,
            };
            let r = render_with_tiles(&params, Some(&opts));
            assert_eq!(base.0, r.0, "iterations divergent (priorité {prio:?})");
            assert_eq!(base.1, r.1, "zs divergent (priorité {prio:?})");
        }
    }

    /// G10.5 : le sink reçoit chaque pixel exactement une fois, avec les mêmes
    /// valeurs que le buffer final.
    #[test]
    fn tile_sink_covers_every_pixel_once_with_final_values() {
        let params = small_mandelbrot(97, 61);
        let n = (params.width * params.height) as usize;
        let collected: std::sync::Mutex<Vec<TileUpdate>> = std::sync::Mutex::new(Vec::new());
        let sink = |u: TileUpdate| {
            collected.lock().unwrap().push(u);
        };
        let opts = TileOpts {
            priority: (0.5, 0.5),
            sink: Some(&sink),
        };
        let (iters, zs, _, _) = render_with_tiles(&params, Some(&opts));

        let mut seen = vec![0u8; n];
        let w = params.width as usize;
        for u in collected.lock().unwrap().iter() {
            assert_eq!(u.iterations.len(), u.w * u.h);
            assert_eq!(u.zs.len(), u.w * u.h);
            for dj in 0..u.h {
                for di in 0..u.w {
                    let idx = (u.y0 + dj) * w + u.x0 + di;
                    seen[idx] += 1;
                    assert_eq!(u.iterations[dj * u.w + di], iters[idx]);
                    assert_eq!(u.zs[dj * u.w + di], zs[idx]);
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "couverture sink non exacte");
    }
    /// G10.5 — diagnostic (non-CI) : la zone prioritaire (quart d'image autour
    /// du point) est complète bien avant la fin du rendu. Mesure le ratio
    /// t(zone prioritaire)/t(total) avec priorité coin haut-gauche.
    /// `cargo test --release --bin fractall-cli tile_priority_first_paint_diagnostic -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn tile_priority_first_paint_diagnostic() {
        let mut params = small_mandelbrot(768, 576);
        params.iteration_max = 20_000;
        let t0 = std::time::Instant::now();
        let events: std::sync::Mutex<Vec<(f64, usize, usize)>> = std::sync::Mutex::new(Vec::new());
        let sink = |u: TileUpdate| {
            events
                .lock()
                .unwrap()
                .push((t0.elapsed().as_secs_f64(), u.x0 + u.w / 2, u.y0 + u.h / 2));
        };
        let prio = (0.0f64, 0.0f64);
        let opts = TileOpts {
            priority: prio,
            sink: Some(&sink),
        };
        let _ = render_with_tiles(&params, Some(&opts));
        let total = t0.elapsed().as_secs_f64();
        let (w, h) = (params.width as f64, params.height as f64);
        let radius = 0.25 * w.hypot(h);
        let t_zone = events
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, cx, cy)| {
                let dx = *cx as f64 - prio.0 * w;
                let dy = *cy as f64 - prio.1 * h;
                (dx * dx + dy * dy).sqrt() <= radius
            })
            .map(|(t, _, _)| *t)
            .fold(0.0f64, f64::max);
        println!(
            "[G10.5] zone prioritaire (r={radius:.0}px autour du coin) complète à {t_zone:.3}s / total {total:.3}s = {:.0}%",
            100.0 * t_zone / total
        );
        assert!(t_zone <= total);
    }
}
