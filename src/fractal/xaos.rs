//! G10.4 — Réutilisation de pixels inter-frame façon XaoS (approximation
//! dynamique séparable colonnes/lignes).
//!
//! Principe : `colorize_to_rgb` ne consomme que `iterations[idx]` + `zs[idx]`,
//! valeurs ABSOLUES et indépendantes de l'orbite référence — un pixel calculé à
//! la coordonnée `c` reste réutilisable pour le même `c` à la frame suivante,
//! même en perturbation. Quand la vue bouge (pan/zoom) sans rotation, la
//! transformée frame→frame est séparable en espace pixel : `x_old = a·(x+0.5)+B`
//! par axe, avec `a`/`B` dérivés de deux ratios O(1) calculés UNE fois en HP
//! (`Δcentre/span_old`, `span_new/span_old`) puis f64 — exact à toute
//! profondeur, aucun HP dans la boucle pixel.
//!
//! Anti-dérive : chaque frame stocke par colonne/ligne l'écart signé (`col_err`/
//! `row_err`, en unités de SES pixels) entre la position nominale de la grille
//! et la position VRAIE des données copiées. Le matching de la frame suivante
//! compare la grille nominale aux positions vraies (`k + err_old[k]`) — la
//! vérité est préservée à travers les copies, l'erreur reste bornée par la
//! tolérance (≤ 0.5 px) quel que soit le nombre de frames enchaînées, au lieu
//! de s'accumuler. Le raffinement idle (GUI) recalcule ensuite exactement et
//! remet les erreurs à zéro.
//!
//! Garde-fous (cf. TODO G10) : fast-path gated sur `rotation == 0 ∧
//! transform_k == None` (la séparabilité casse en rotation), désactivé pour les
//! modes à données par-pixel (`Distance*`, `OrbitTraps`, `Wings` — même
//! exclusion que `build_reuse`), pour `find_nucleus` (le rendu re-centre), et
//! pour tout changement de paramètre non-géométrique (fingerprint JSON).

use std::sync::Arc;

use num_complex::Complex64;
use rug::Float;

use crate::fractal::{FractalParams, OutColoringMode};

/// Tolérance de matching, en unités de pixel de la frame CIBLE. 0.5 = accepte
/// tout décalage sous-pixel (pan fluide, zoom molette ≈ écho nearest-neighbor
/// immédiat) ; le raffinement idle rend l'image exacte dès la pause.
pub const XAOS_TOLERANCE_PX: f64 = 0.5;

/// Précision HP pour les ratios de la transformée frame→frame (mêmes ordres de
/// grandeur que le warp G10.1 : des ratios O(1), 256 b couvrent tout zoom
/// représentable en strings HP).
const TRANSFORM_PRECISION_BITS: u32 = 256;

/// Frame source pour la réutilisation inter-frame : buffers bruts de la
/// dernière passe complétée + vue (HP strings) + positions vraies par axe.
#[derive(Clone)]
pub struct XaosSourceFrame {
    pub iterations: Arc<Vec<u32>>,
    pub zs: Arc<Vec<Complex64>>,
    pub width: u32,
    pub height: u32,
    /// Vue de la frame (centre/span en strings HP — précision préservée au
    /// deep zoom ; fallback = formatage f64 par l'appelant).
    pub cx: String,
    pub cy: String,
    pub sx: String,
    pub sy: String,
    /// Écart signé (px de CETTE frame) entre position nominale et position
    /// vraie des données, par colonne (`len == width`) et ligne (`len == height`).
    /// 0.0 = pixel calculé exactement sur la grille.
    pub col_err: Arc<Vec<f64>>,
    pub row_err: Arc<Vec<f64>>,
    /// Fingerprint des paramètres non-géométriques (cf. `params_fingerprint`).
    pub fingerprint: String,
}

/// Mapping résolu pour UNE passe de rendu : pour chaque colonne/ligne cible,
/// l'index source (-1 = à calculer). Un pixel est copié ssi sa colonne ET sa
/// ligne matchent. Consommé par les boucles pixel (f64 / GMP / perturbation).
pub struct XaosMap {
    pub iterations: Arc<Vec<u32>>,
    pub zs: Arc<Vec<Complex64>>,
    pub src_width: usize,
    /// Index de colonne source par colonne cible (-1 = calculer).
    pub src_col: Vec<i32>,
    /// Index de ligne source par ligne cible (-1 = calculer).
    pub src_row: Vec<i32>,
    /// Positions vraies de la frame PRODUITE (px cible, 0.0 pour les colonnes/
    /// lignes calculées) — à stocker avec la frame résultat.
    pub col_err: Vec<f64>,
    pub row_err: Vec<f64>,
    /// Nombre de colonnes/lignes réutilisées (diagnostic + gate raffinement).
    pub reused_cols: usize,
    pub reused_rows: usize,
}

impl XaosMap {
    /// Fraction de pixels copiés (produit des fractions par axe).
    pub fn reused_fraction(&self, width: usize, height: usize) -> f64 {
        if width == 0 || height == 0 {
            return 0.0;
        }
        (self.reused_cols as f64 / width as f64) * (self.reused_rows as f64 / height as f64)
    }

    /// Vrai si au moins un pixel sera copié.
    pub fn any_reuse(&self) -> bool {
        self.reused_cols > 0 && self.reused_rows > 0
    }
}

/// Fingerprint JSON des paramètres NON-géométriques : deux frames sont
/// pixel-compatibles ssi leurs fingerprints sont égaux (même formule, même
/// iteration_max, même bailout, même plan, même seed Julia, …). La géométrie
/// (centre/span/dims) est neutralisée — c'est précisément ce que le mapping
/// gère. Robuste par construction à l'ajout de futurs champs : un nouveau champ
/// sérialisé entre dans la comparaison (conservateur).
pub fn params_fingerprint(params: &FractalParams) -> String {
    let mut p = params.clone();
    p.width = 0;
    p.height = 0;
    p.center_x = 0.0;
    p.center_y = 0.0;
    p.span_x = 0.0;
    p.span_y = 0.0;
    p.center_x_hp = None;
    p.center_y_hp = None;
    p.span_x_hp = None;
    p.span_y_hp = None;
    serde_json::to_string(&p).unwrap_or_default()
}

/// Gates sémantiques côté frame CIBLE : vrai si le fast-path XaoS est
/// applicable à ce rendu (indépendamment de la frame source).
pub fn params_allow_pixel_reuse(params: &FractalParams) -> bool {
    let needs_extra_data = matches!(
        params.out_coloring_mode,
        OutColoringMode::Distance
            | OutColoringMode::DistanceAO
            | OutColoringMode::Distance3D
            | OutColoringMode::OrbitTraps
            | OutColoringMode::Wings
    );
    params.rotation == 0.0
        && params.transform_k.is_none()
        && !params.find_nucleus
        && !needs_extra_data
        && params.aa_subpixel_offset == [0.0, 0.0]
}

/// Coordonnées de vue effectives (strings HP si présentes, sinon f64 formaté
/// exactement). Même convention que le snapshot de vue GUI.
pub fn view_strings(params: &FractalParams) -> (String, String, String, String) {
    let s = |hp: &Option<String>, f: f64| hp.clone().unwrap_or_else(|| format!("{f:.17e}"));
    (
        s(&params.center_x_hp, params.center_x),
        s(&params.center_y_hp, params.center_y),
        s(&params.span_x_hp, params.span_x),
        s(&params.span_y_hp, params.span_y),
    )
}

/// Coefficients (a, B) de la transformée pixel cible→source pour UN axe :
/// `p_src = a·(x+0.5) + B` (index fractionnaire source). Dérivation :
/// `c(x) = c_new + ((x+0.5)/n_new − 0.5)·s_new` et
/// `k(c) = n_old·(0.5 + (c − c_old)/s_old) − 0.5`, d'où
/// `a = (s_new/s_old)·(n_old/n_new)` et `B = n_old·(0.5 + Δc/s_old − 0.5·r) − 0.5`
/// avec `r = s_new/s_old`, `Δc = c_new − c_old`. Les deux ratios sont calculés
/// en HP (Δc de deux centres proches à deep zoom) puis convertis en f64 (O(1)).
fn axis_transform(
    c_old: &str,
    s_old: &str,
    c_new: &str,
    s_new: &str,
    n_old: u32,
    n_new: u32,
) -> Option<(f64, f64)> {
    let prec = TRANSFORM_PRECISION_BITS;
    let f = |s: &str| Float::parse(s).ok().map(|p| Float::with_val(prec, p));
    let (co, so, cn, sn) = (f(c_old)?, f(s_old)?, f(c_new)?, f(s_new)?);
    if so.is_zero() || n_old == 0 || n_new == 0 {
        return None;
    }
    let dc = (Float::with_val(prec, &cn - &co) / &so).to_f64();
    let r = Float::with_val(prec, &sn / &so).to_f64();
    if !dc.is_finite() || !r.is_finite() || r <= 0.0 {
        return None;
    }
    let a = r * n_old as f64 / n_new as f64;
    let b = n_old as f64 * (0.5 + dc - 0.5 * r) - 0.5;
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    Some((a, b))
}

/// Matching d'UN axe : pour chaque index cible x, cherche la colonne/ligne
/// source dont la position VRAIE (`k + err_old[k]`, px source) est la plus
/// proche de la position nominale mappée `p = a·(x+0.5) + B`. Match si la
/// distance, convertie en px CIBLE (`/a`), est ≤ `tol`.
///
/// Retourne (src, err_new, reused) : `src[x] = -1` si à calculer, sinon l'index
/// source ; `err_new[x]` = position vraie − nominale en px cible (0.0 si
/// calculé) ; `reused` = nombre d'indices matchés.
pub fn build_axis_map(
    n_new: usize,
    n_old: usize,
    a: f64,
    b: f64,
    err_old: &[f64],
    tol: f64,
) -> (Vec<i32>, Vec<f64>, usize) {
    let mut src = vec![-1i32; n_new];
    let mut err_new = vec![0.0f64; n_new];
    let mut reused = 0usize;
    if n_old == 0 || !(a > 0.0) || !a.is_finite() || !b.is_finite() {
        return (src, err_new, reused);
    }
    let get_err = |k: usize| err_old.get(k).copied().unwrap_or(0.0);
    for x in 0..n_new {
        let p = a * (x as f64 + 0.5) + b;
        // Les positions vraies dévient de ≤ tol de la grille source : examiner
        // round(p) et ses deux voisins suffit (positions quasi-monotones).
        let k0 = p.round() as i64;
        let mut best: Option<(f64, usize)> = None; // (distance signée en px source, k)
        for k in (k0 - 1)..=(k0 + 1) {
            if k < 0 || k >= n_old as i64 {
                continue;
            }
            let k = k as usize;
            let d = (k as f64 + get_err(k)) - p;
            if best.map_or(true, |(bd, _)| d.abs() < bd.abs()) {
                best = Some((d, k));
            }
        }
        if let Some((d, k)) = best {
            let d_target = d / a; // px cible
            if d_target.abs() <= tol {
                src[x] = k as i32;
                err_new[x] = d_target;
                reused += 1;
            }
        }
    }
    (src, err_new, reused)
}

/// Construit le mapping complet frame source → passe cible. `None` si le
/// fast-path n'est pas applicable (gates, fingerprint, transformée dégénérée)
/// ou si aucun pixel n'est réutilisable.
pub fn build_map(src: &XaosSourceFrame, params: &FractalParams) -> Option<XaosMap> {
    if !params_allow_pixel_reuse(params) {
        return None;
    }
    let expected = src.width as usize * src.height as usize;
    if expected == 0 || src.iterations.len() != expected || src.zs.len() != expected {
        return None;
    }
    if src.fingerprint != params_fingerprint(params) {
        return None;
    }
    let (cx, cy, sx, sy) = view_strings(params);
    let (ax, bx) = axis_transform(&src.cx, &src.sx, &cx, &sx, src.width, params.width)?;
    let (ay, by) = axis_transform(&src.cy, &src.sy, &cy, &sy, src.height, params.height)?;
    let (src_col, col_err, reused_cols) = build_axis_map(
        params.width as usize,
        src.width as usize,
        ax,
        bx,
        &src.col_err,
        XAOS_TOLERANCE_PX,
    );
    let (src_row, row_err, reused_rows) = build_axis_map(
        params.height as usize,
        src.height as usize,
        ay,
        by,
        &src.row_err,
        XAOS_TOLERANCE_PX,
    );
    let map = XaosMap {
        iterations: Arc::clone(&src.iterations),
        zs: Arc::clone(&src.zs),
        src_width: src.width as usize,
        src_col,
        src_row,
        col_err,
        row_err,
        reused_cols,
        reused_rows,
    };
    if !map.any_reuse() {
        return None;
    }
    Some(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    fn frame_for(params: &FractalParams, iters: Vec<u32>) -> XaosSourceFrame {
        let n = (params.width * params.height) as usize;
        assert_eq!(iters.len(), n);
        let (cx, cy, sx, sy) = view_strings(params);
        XaosSourceFrame {
            iterations: Arc::new(iters),
            zs: Arc::new(vec![Complex64::new(0.0, 0.0); n]),
            width: params.width,
            height: params.height,
            cx,
            cy,
            sx,
            sy,
            col_err: Arc::new(vec![0.0; params.width as usize]),
            row_err: Arc::new(vec![0.0; params.height as usize]),
            fingerprint: params_fingerprint(params),
        }
    }

    fn base_params(w: u32, h: u32) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, w, h);
        p.center_x = -0.5;
        p.center_y = 0.0;
        p.span_x = 4.0;
        p.span_y = 4.0 * h as f64 / w as f64;
        p
    }

    #[test]
    fn identity_view_maps_every_index_exactly() {
        let p = base_params(64, 64);
        let src = frame_for(&p, vec![7; 64 * 64]);
        let map = build_map(&src, &p).expect("map");
        assert_eq!(map.reused_cols, 64);
        assert_eq!(map.reused_rows, 64);
        for x in 0..64 {
            assert_eq!(map.src_col[x], x as i32);
            assert!(map.col_err[x].abs() < 1e-9);
        }
    }

    #[test]
    fn integer_pixel_pan_shifts_indices_with_zero_error() {
        let p = base_params(64, 64);
        let mut moved = p.clone();
        // Pan de +8 px en x : Δcx = 8 · (span_x / width).
        moved.center_x = p.center_x + 8.0 * p.span_x / 64.0;
        let src = frame_for(&p, vec![1; 64 * 64]);
        let map = build_map(&src, &moved).expect("map");
        // Colonne cible 0 → source 8 ; les 8 dernières colonnes n'existent pas.
        assert_eq!(map.src_col[0], 8);
        assert_eq!(map.src_col[55], 63);
        assert_eq!(map.src_col[56], -1);
        assert_eq!(map.reused_cols, 56);
        assert_eq!(map.reused_rows, 64);
        assert!(map.col_err[0].abs() < 1e-9, "pan entier = copie exacte");
    }

    #[test]
    fn fractional_pan_reuses_with_tracked_error() {
        let p = base_params(64, 64);
        let mut moved = p.clone();
        moved.center_x = p.center_x + 0.3 * p.span_x / 64.0; // pan +0.3 px
        let src = frame_for(&p, vec![1; 64 * 64]);
        let map = build_map(&src, &moved).expect("map");
        // Toutes les colonnes matchent (0.3 ≤ tol), y compris la dernière (0.3 px hors champ).
        assert_eq!(map.reused_cols, 64);
        // Donnée vraie à −0.3 px de la grille cible (le contenu a glissé).
        assert!((map.col_err[5] + 0.3).abs() < 1e-9, "err = {}", map.col_err[5]);
    }

    #[test]
    fn chained_fractional_pans_do_not_accumulate_error() {
        // Frame A → pan +0.3 px → frame B (err −0.3) → pan +0.3 px → frame C.
        // Sans positions vraies, C croirait ses données à −0.3 alors qu'elles
        // sont à −0.6. Avec le tracking, le matching voit la position VRAIE :
        // la colonne x (vraie à −0.6, hors tolérance) est rejetée au profit de
        // la colonne x+1 (vraie à +0.4) ou recalculée — l'erreur reste ≤ tol
        // vs la vérité, quel que soit le nombre de pans enchaînés.
        let pa = base_params(64, 64);
        let step = 0.3 * pa.span_x / 64.0;
        let mut pb = pa.clone();
        pb.center_x += step;
        let src_a = frame_for(&pa, vec![1; 64 * 64]);
        let map_ab = build_map(&src_a, &pb).expect("A→B");

        // Frame B stockée avec ses erreurs vraies.
        let mut src_b = frame_for(&pb, vec![1; 64 * 64]);
        src_b.col_err = Arc::new(map_ab.col_err.clone());
        src_b.row_err = Arc::new(map_ab.row_err.clone());

        let mut pc = pb.clone();
        pc.center_x += step;
        let map_bc = build_map(&src_b, &pc).expect("B→C");
        for x in 0..64usize {
            if map_bc.src_col[x] >= 0 {
                assert!(
                    map_bc.col_err[x].abs() <= XAOS_TOLERANCE_PX + 1e-9,
                    "dérive non bornée : err[{x}] = {}",
                    map_bc.col_err[x]
                );
                // Le match retenu est la colonne DÉCALÉE x+1 (vraie à +0.4),
                // pas la colonne x (vraie à −0.6) : preuve que la comparaison
                // se fait contre la position vraie, pas la grille nominale.
                assert_eq!(map_bc.src_col[x], x as i32 + 1);
                assert!((map_bc.col_err[x] - 0.4).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn zoom_in_2x_maps_center_half() {
        let p = base_params(64, 64);
        let mut zoomed = p.clone();
        zoomed.span_x = p.span_x / 2.0;
        zoomed.span_y = p.span_y / 2.0;
        let src = frame_for(&p, vec![1; 64 * 64]);
        let map = build_map(&src, &zoomed).expect("map");
        // a = 0.5 : la cible couvre les colonnes source 16..48. Positions
        // mappées quart-entières (15.75, 16.25, …) → distance 0.25 old px
        // = 0.5 px cible = tol → tout matche (écho nearest-neighbor).
        assert_eq!(map.reused_cols, 64);
        assert_eq!(map.src_col[0], 16);
        assert_eq!(map.src_col[63], 47);
    }

    #[test]
    fn deep_zoom_hp_pan_matches_without_f64_center_resolution() {
        // À 1e-30 de span, Δcentre d'un pan de 8 px est indiscernable en f64 :
        // seul le chemin HP du mapping le voit.
        let mut p = base_params(64, 64);
        p.center_x_hp = Some("-0.75000000000000000000000000000001".into());
        p.center_y_hp = Some("0.10000000000000000000000000000002".into());
        p.span_x_hp = Some("1e-30".into());
        p.span_y_hp = Some("1e-30".into());
        let mut moved = p.clone();
        // pan +8 px : Δcx = 8 · 1e-30 / 64 = 1.25e-31, calculé en HP (indiscernable en f64).
        let cx = Float::with_val(
            256,
            Float::parse(p.center_x_hp.as_deref().unwrap()).unwrap(),
        ) + Float::with_val(256, Float::parse("1.25e-31").unwrap());
        moved.center_x_hp = Some(cx.to_string_radix(10, Some(40)));
        let src = frame_for(&p, vec![1; 64 * 64]);
        let map = build_map(&src, &moved).expect("map");
        assert_eq!(map.src_col[0], 8, "pan HP vu par la transformée");
        assert!(map.col_err[0].abs() < 1e-6);
    }

    #[test]
    fn fingerprint_change_disables_reuse() {
        let p = base_params(64, 64);
        let src = frame_for(&p, vec![1; 64 * 64]);
        let mut other = p.clone();
        other.iteration_max += 100;
        assert!(build_map(&src, &other).is_none(), "iteration_max ≠ → pas de reuse");
        let mut geom = p.clone();
        geom.center_x += 0.01;
        assert!(build_map(&src, &geom).is_some(), "géométrie seule ≠ → reuse OK");
    }

    #[test]
    fn gates_disable_reuse() {
        let p = base_params(64, 64);
        let src = frame_for(&p, vec![1; 64 * 64]);
        let mut rot = p.clone();
        rot.rotation = 10.0;
        // fingerprint diffère aussi (rotation sérialisée) mais le gate rotation
        // doit suffire même à fingerprint égal : tester params_allow directement.
        assert!(!params_allow_pixel_reuse(&rot));
        assert!(build_map(&src, &rot).is_none());
        let mut dist = p.clone();
        dist.out_coloring_mode = OutColoringMode::Distance;
        assert!(build_map(&src, &dist).is_none());
        let mut aa = p.clone();
        aa.aa_subpixel_offset = [0.25, 0.0];
        assert!(build_map(&src, &aa).is_none());
        let mut nuc = p.clone();
        nuc.find_nucleus = true;
        assert!(build_map(&src, &nuc).is_none());
    }

    /// Bout-en-bout : rend une vue, pan de N px ENTIERS, re-rend avec le
    /// mapping XaoS → identique pixel à pixel au rendu frais de la nouvelle
    /// vue (pan entier ⇒ les pixels copiés portent exactement le même `c`).
    fn assert_integer_pan_roundtrip(mut params: FractalParams) {
        use std::sync::atomic::AtomicBool;
        use crate::render::render_escape_time_cancellable_with_reuse as render;

        params.use_bytecode_engine = true;
        let cancel = Arc::new(AtomicBool::new(false));
        let (it_a, zs_a, _, _) = render(&params, &cancel, None, &mut None, None).expect("A");

        // Pan +8 px en x, +3 px en y (via strings HP si présentes, sinon f64).
        let mut moved = params.clone();
        let pan_hp = |c_hp: &str, s_hp: &str, px_frac: f64| -> String {
            let prec = 256;
            let cf = Float::with_val(prec, Float::parse(c_hp).unwrap());
            let sf = Float::with_val(prec, Float::parse(s_hp).unwrap());
            (cf + sf * px_frac).to_string_radix(10, None)
        };
        if let (Some(cx), Some(sx), Some(cy), Some(sy)) = (
            params.center_x_hp.as_deref(),
            params.span_x_hp.as_deref(),
            params.center_y_hp.as_deref(),
            params.span_y_hp.as_deref(),
        ) {
            moved.center_x_hp = Some(pan_hp(cx, sx, 8.0 / moved.width as f64));
            moved.center_y_hp = Some(pan_hp(cy, sy, 3.0 / moved.height as f64));
        } else {
            moved.center_x += 8.0 * moved.span_x / moved.width as f64;
            moved.center_y += 3.0 * moved.span_y / moved.height as f64;
        }

        let src = XaosSourceFrame {
            iterations: Arc::new(it_a),
            zs: Arc::new(zs_a),
            width: params.width,
            height: params.height,
            cx: view_strings(&params).0,
            cy: view_strings(&params).1,
            sx: view_strings(&params).2,
            sy: view_strings(&params).3,
            col_err: Arc::new(vec![0.0; params.width as usize]),
            row_err: Arc::new(vec![0.0; params.height as usize]),
            fingerprint: params_fingerprint(&params),
        };
        let map = build_map(&src, &moved).expect("map");
        assert!(map.any_reuse(), "pan entier doit réutiliser des bandes");
        assert_eq!(map.reused_cols, params.width as usize - 8);
        assert_eq!(map.reused_rows, params.height as usize - 3);
        for &e in map.col_err.iter().chain(map.row_err.iter()) {
            assert!(e.abs() < 1e-9, "pan entier ⇒ copies exactes, err={e}");
        }

        let (it_xaos, zs_xaos, _, _) =
            render(&moved, &cancel, None, &mut None, Some(&map)).expect("B xaos");
        let (it_fresh, zs_fresh, _, _) =
            render(&moved, &cancel, None, &mut None, None).expect("B fresh");
        assert_eq!(it_xaos, it_fresh, "itérations : XaoS == frais (pan entier)");
        let zdiff = zs_xaos
            .iter()
            .zip(&zs_fresh)
            .filter(|(a, b)| (*a - *b).norm() > 1e-12)
            .count();
        assert_eq!(zdiff, 0, "zs : XaoS == frais (pan entier)");
    }

    #[test]
    fn integer_pan_roundtrip_f64_path() {
        let mut p = base_params(64, 48);
        p.center_x = -0.6;
        p.center_y = 0.3;
        p.span_x = 0.5;
        p.span_y = 0.375;
        p.iteration_max = 300;
        assert_integer_pan_roundtrip(p);
    }

    #[test]
    fn integer_pan_roundtrip_perturbation_path() {
        // Zoom 1e-13 de span → path perturbation (seuil ~1e-12 de pixel size
        // dépassé à 64 px ? pixel = 1.5e-15 < 1e-12 → perturbation). Vue
        // seahorse peu profonde, itérations modestes pour un test rapide.
        let mut p = base_params(64, 48);
        p.center_x_hp = Some("-0.74364386269".into());
        p.center_y_hp = Some("0.13182590271".into());
        p.span_x_hp = Some("1e-10".into());
        p.span_y_hp = Some("7.5e-11".into());
        p.center_x = -0.74364386269;
        p.center_y = 0.13182590271;
        p.span_x = 1e-10;
        p.span_y = 7.5e-11;
        p.iteration_max = 2000;
        p.algorithm_mode = crate::fractal::AlgorithmMode::Perturbation;
        assert_integer_pan_roundtrip(p);
    }

    /// Diagnostic perf (non-CI) : gain wall-clock d'un pan 8 px avec XaoS vs
    /// rendu complet. `cargo test --release --bin fractall-cli xaos_pan_speedup -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn xaos_pan_speedup_diagnostic() {
        use std::sync::atomic::AtomicBool;
        use std::time::Instant;
        use crate::render::render_escape_time_cancellable_with_reuse as render;

        let mut p = base_params(1024, 768);
        p.center_x = -0.743643135;
        p.center_y = 0.131825963;
        p.span_x = 2e-7;
        p.span_y = 1.5e-7;
        p.iteration_max = 20000;
        let cancel = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        let (it_a, zs_a, _, _) = render(&p, &cancel, None, &mut None, None).expect("A");
        let t_full = t0.elapsed();

        let mut moved = p.clone();
        moved.center_x += 8.0 * p.span_x / p.width as f64;
        moved.center_y += 3.0 * p.span_y / p.height as f64;
        let (cx, cy, sx, sy) = view_strings(&p);
        let src = XaosSourceFrame {
            iterations: Arc::new(it_a),
            zs: Arc::new(zs_a),
            width: p.width,
            height: p.height,
            cx, cy, sx, sy,
            col_err: Arc::new(vec![0.0; p.width as usize]),
            row_err: Arc::new(vec![0.0; p.height as usize]),
            fingerprint: params_fingerprint(&p),
        };
        let t1 = Instant::now();
        let map = build_map(&src, &moved).expect("map");
        let t_map = t1.elapsed();
        let reused_px = (map.reused_cols * map.reused_rows) as f64;
        let total_px = (p.width * p.height) as f64;
        let t2 = Instant::now();
        let _ = render(&moved, &cancel, None, &mut None, Some(&map)).expect("B xaos");
        let t_xaos = t2.elapsed();
        println!(
            "full={:?} xaos={:?} (map build {:?}) — pixels copiés {:.1}% — speedup ×{:.1}",
            t_full, t_xaos, t_map,
            100.0 * reused_px / total_px,
            t_full.as_secs_f64() / t_xaos.as_secs_f64().max(1e-9),
        );
    }

    #[test]
    fn lower_resolution_pass_maps_against_full_res_source() {
        // Passe progressive à 1/4 : mêmes vues, dims cible 16×16 vs source 64×64.
        let p = base_params(64, 64);
        let src = frame_for(&p, vec![1; 64 * 64]);
        let mut pass = p.clone();
        pass.width = 16;
        pass.height = 16;
        let map = build_map(&src, &pass).expect("map");
        // a = 4 : cible 0 → position source 4·0.5 − 0.5 + ... = 1.5 → à 0.5 old
        // px des colonnes 1 et 2 = 0.125 px cible ≤ tol → match.
        assert_eq!(map.reused_cols, 16);
        assert!(map.src_col[0] == 1 || map.src_col[0] == 2);
    }
}
