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
//! Anti-dérive : chaque frame stocke PAR PIXEL l'écart signé (`err`, en unités
//! de SES pixels) entre la position nominale de la grille et la position VRAIE
//! des données. Le candidat source est choisi par axe (hints `col_err`/
//! `row_err`), mais la COPIE est décidée par pixel contre sa position vraie —
//! la vérité est préservée à travers les copies, l'erreur reste bornée par la
//! tolérance (≤ 0.5 px) quel que soit le nombre de frames enchaînées, au lieu
//! de s'accumuler (un modèle par axe était faux sur les frames mixtes de
//! l'écho zoom-in, cf. `XaosSourceFrame::err`). Le raffinement idle (GUI)
//! recalcule ensuite exactement et remet les erreurs à zéro.
//!
//! Zoom-in (matching INJECTIF) : une colonne/ligne source ne sert qu'UN index
//! cible (le mieux aligné). Sans cela, un zoom-in de facteur ≤ 2 duplique des
//! colonnes jusqu'à couvrir 100 % de la cible sans calculer UN SEUL pixel
//! frais (écho pur qui ne fait que retarder l'image exacte, cf. clic-zoom ×2
//! centré). L'injectivité garantit ≥ (1−a)·n indices frais par axe en zoom-in
//! et est un no-op en pan (mapping bijectif) comme en zoom-out / passes
//! preview (espacement source > 1). Le raffinement idle passe une tolérance
//! EXACTE (`XAOS_EXACT_TOLERANCE_PX`) : seuls les pixels dont la position est
//! déjà vraie (calculés frais ou copiés alignés) sont conservés, les
//! approximations sont recalculées — le refine ne refait pas le travail que la
//! passe finale vient de faire.
//!
//! Garde-fous (cf. TODO G10) : fast-path gated sur `rotation == 0 ∧
//! transform_k == None` (la séparabilité casse en rotation), désactivé pour les
//! modes à données par-pixel (`Distance*`, `OrbitTraps`, `Wings` — même
//! exclusion que `build_reuse`), pour `find_nucleus` (le rendu re-centre), et
//! pour tout changement de paramètre non-géométrique (fingerprint JSON).

use std::sync::Arc;

use num_complex::Complex64;
#[cfg(test)]
use rug::Float;

use crate::fractal::{FractalParams, OutColoringMode, ViewHp};

/// Tolérance de matching, en unités de pixel de la frame CIBLE. 0.5 = accepte
/// tout décalage sous-pixel (pan fluide, zoom molette ≈ écho nearest-neighbor
/// immédiat) ; le raffinement idle rend l'image exacte dès la pause.
pub const XAOS_TOLERANCE_PX: f64 = 0.5;

/// Tolérance "exacte" (raffinement idle) : ne matche que les colonnes/lignes
/// dont la position vraie coïncide avec la grille cible au bruit f64 près
/// (copies alignées pixel-entier, pixels calculés frais). Bien au-dessus du
/// bruit des ratios HP→f64 (~1e-14 px), bien en dessous de tout décalage réel.
pub const XAOS_EXACT_TOLERANCE_PX: f64 = 1e-9;

/// Précision HP MINIMALE pour les ratios de la transformée frame→frame (mêmes ordres de
/// grandeur que le warp G10.1 : des ratios O(1), 256 b couvrent tout zoom
/// représentable en strings HP).
const TRANSFORM_PRECISION_BITS: u32 = 256;

/// Frame source pour la réutilisation inter-frame : buffers bruts de la
/// dernière passe complétée + vue (HP strings) + erreur de position PAR PIXEL.
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
    /// **Vérité par pixel** : écart signé `(dx, dy)` (px de CETTE frame) entre
    /// la position nominale `(i, j)` et la position VRAIE des données du pixel.
    /// `[0, 0]` = calculé exactement sur la grille. `len == width·height`.
    ///
    /// ⚠️ Un modèle PAR AXE (« toute la colonne k est décalée de e ») est
    /// FAUX dès qu'une frame est mixte — ce qui est le cas de TOUTE frame
    /// d'écho zoom-in (l'injectivité laisse des colonnes/lignes fraîches) : un
    /// pixel d'une colonne fraîche mais d'une ligne copiée est exact alors que
    /// `row_err` lui prête un décalage → copié avec une erreur cachée qui
    /// CROISSAIT géométriquement en zoom continu (0,48 → 1,5 px en 6 crans
    /// ×1,2, bug 2026-08-23). D'où la vérité par pixel.
    pub err: Arc<Vec<[f32; 2]>>,
    /// Hints PAR AXE pour la sélection du candidat (écart typique des pixels
    /// copiés de la colonne/ligne, 0 sinon). Ne servent qu'à choisir parmi
    /// `round(p) ± 1` ; la décision de copie est prise PAR PIXEL sur `err`.
    pub col_err: Arc<Vec<f64>>,
    pub row_err: Arc<Vec<f64>>,
    /// Fingerprint des paramètres non-géométriques (cf. `params_fingerprint`).
    pub fingerprint: String,
}

impl XaosSourceFrame {
    /// Frame entièrement exacte (rendu frais ou raffiné).
    pub fn exact(
        iterations: Arc<Vec<u32>>,
        zs: Arc<Vec<Complex64>>,
        width: u32,
        height: u32,
        view: (String, String, String, String),
        fingerprint: String,
    ) -> Self {
        let n = width as usize * height as usize;
        XaosSourceFrame {
            iterations,
            zs,
            width,
            height,
            cx: view.0,
            cy: view.1,
            sx: view.2,
            sy: view.3,
            err: Arc::new(vec![[0.0, 0.0]; n]),
            col_err: Arc::new(vec![0.0; width as usize]),
            row_err: Arc::new(vec![0.0; height as usize]),
            fingerprint,
        }
    }

    /// Frame produite par une passe avec `map` : erreurs par pixel héritées
    /// du mapping (vraie position des pixels copiés, 0 pour les calculés).
    pub fn from_map(
        iterations: Arc<Vec<u32>>,
        zs: Arc<Vec<Complex64>>,
        width: u32,
        height: u32,
        view: (String, String, String, String),
        fingerprint: String,
        map: &XaosMap,
    ) -> Self {
        let mut f = Self::exact(iterations, zs, width, height, view, fingerprint);
        f.err = Arc::new(map.produced_err(width as usize, height as usize));
        f.col_err = Arc::new(map.col_err.clone());
        f.row_err = Arc::new(map.row_err.clone());
        f
    }

    /// Erreur de position max (px) sur la frame.
    pub fn max_abs_err(&self) -> f64 {
        self.err.iter().fold(0.0f64, |m, e| {
            m.max(e[0].abs() as f64).max(e[1].abs() as f64)
        })
    }
}

/// Mapping résolu pour UNE passe de rendu : candidat source par colonne/ligne
/// cible (-1 = aucun), puis décision PAR PIXEL dans `source_index` : le pixel
/// `(i, j)` est copié depuis `(src_col[i], src_row[j])` ssi la position VRAIE
/// de ce pixel source (nominale + `src_err`) est à ≤ `tol` px cible de la
/// position nominale cible sur les DEUX axes. Consommé par les boucles pixel
/// (f64 / GMP / perturbation).
pub struct XaosMap {
    pub iterations: Arc<Vec<u32>>,
    pub zs: Arc<Vec<Complex64>>,
    pub src_width: usize,
    /// Candidat PRIMAIRE (position hintée) par colonne cible (-1 = aucun).
    pub src_col: Vec<i32>,
    /// Candidat PRIMAIRE par ligne cible (-1 = aucun).
    pub src_row: Vec<i32>,
    /// Candidat SECONDAIRE (position nominale, pour les pixels frais d'une
    /// colonne/ligne mixte) ; -1 si absent ou identique au primaire.
    pub src_col2: Vec<i32>,
    pub src_row2: Vec<i32>,
    /// Écart NOMINAL `(k − p)/a` (px cible) de chaque candidat.
    pub col_dev: Vec<f64>,
    pub row_dev: Vec<f64>,
    pub col_dev2: Vec<f64>,
    pub row_dev2: Vec<f64>,
    /// Facteur px source → px cible (`1/a`) par axe.
    pub inv_ax: f64,
    pub inv_ay: f64,
    /// Erreur par pixel de la frame SOURCE (px source).
    pub src_err: Arc<Vec<[f32; 2]>>,
    /// Tolérance de copie (px cible).
    pub tol: f64,
    /// Hints par axe pour la frame PRODUITE (écart des pixels copiés dont la
    /// source porte le hint de l'axe ; 0 pour les colonnes/lignes calculées).
    pub col_err: Vec<f64>,
    pub row_err: Vec<f64>,
    /// Nombre de colonnes/lignes ayant un candidat (diagnostic, borne sup.).
    pub reused_cols: usize,
    pub reused_rows: usize,
    /// Statistiques EXACTES calculées au build : pixels copiés, erreur max
    /// (px cible) parmi eux.
    pub copied: usize,
    pub max_err: f64,
}

impl XaosMap {
    /// Écart vrai (px cible) du pixel source candidat de `(i, j)`, `None` si
    /// un des axes n'a pas de candidat.
    #[inline(always)]
    fn check(&self, sc: i32, sr: i32, dev_x: f64, dev_y: f64) -> Option<(usize, f64, f64)> {
        if sc < 0 || sr < 0 {
            return None;
        }
        let sidx = sr as usize * self.src_width + sc as usize;
        let e = self.src_err.get(sidx).copied().unwrap_or([0.0, 0.0]);
        let dx = dev_x + e[0] as f64 * self.inv_ax;
        let dy = dev_y + e[1] as f64 * self.inv_ay;
        (dx.abs() <= self.tol && dy.abs() <= self.tol).then_some((sidx, dx, dy))
    }

    /// Premier pixel source candidat de `(i, j)` dont la position VRAIE est
    /// à ≤ tol sur les deux axes : combinaisons (primaire, secondaire) des
    /// deux axes, primaire d'abord.
    #[inline(always)]
    fn deviation(&self, i: usize, j: usize) -> Option<(usize, f64, f64)> {
        let (c1, c2) = (self.src_col[i], self.src_col2[i]);
        let (r1, r2) = (self.src_row[j], self.src_row2[j]);
        if (c1 < 0 && c2 < 0) || (r1 < 0 && r2 < 0) {
            return None;
        }
        self.check(c1, r1, self.col_dev[i], self.row_dev[j])
            .or_else(|| self.check(c1, r2, self.col_dev[i], self.row_dev2[j]))
            .or_else(|| self.check(c2, r1, self.col_dev2[i], self.row_dev[j]))
            .or_else(|| self.check(c2, r2, self.col_dev2[i], self.row_dev2[j]))
    }

    /// Index source du pixel cible `(i, j)`, `None` = à calculer. Point
    /// d'entrée UNIQUE des 4 boucles pixel (f64 / GMP / perturbation /
    /// perturbation-GMP).
    #[inline(always)]
    pub fn source_index(&self, i: usize, j: usize) -> Option<usize> {
        self.deviation(i, j).map(|(sidx, _, _)| sidx)
    }

    /// Erreur par pixel de la frame PRODUITE par cette passe (px cible) :
    /// écart vrai des pixels copiés, 0 pour les pixels calculés.
    pub fn produced_err(&self, width: usize, height: usize) -> Vec<[f32; 2]> {
        let mut out = vec![[0.0f32, 0.0]; width * height];
        if width != self.src_col.len() || height != self.src_row.len() {
            return out;
        }
        for j in 0..height {
            for i in 0..width {
                if let Some((_, dx, dy)) = self.deviation(i, j) {
                    out[j * width + i] = [dx as f32, dy as f32];
                }
            }
        }
        out
    }

    fn finalize(mut self, width: usize, height: usize) -> Self {
        let mut copied = 0usize;
        let mut max_err = 0.0f64;
        if width == self.src_col.len() && height == self.src_row.len() {
            for j in 0..height {
                if self.src_row[j] < 0 && self.src_row2[j] < 0 {
                    continue;
                }
                for i in 0..width {
                    if let Some((_, dx, dy)) = self.deviation(i, j) {
                        copied += 1;
                        max_err = max_err.max(dx.abs()).max(dy.abs());
                    }
                }
            }
        }
        self.copied = copied;
        self.max_err = max_err;
        self
    }

    /// Fraction de pixels copiés.
    pub fn reused_fraction(&self, width: usize, height: usize) -> f64 {
        if width == 0 || height == 0 {
            return 0.0;
        }
        self.copied as f64 / (width * height) as f64
    }

    /// Au moins un pixel copié.
    pub fn any_reuse(&self) -> bool {
        self.copied > 0
    }

    /// Écho PUR : 100 % des pixels copiés (aucun calcul).
    pub fn is_pure_copy(&self, width: usize, height: usize) -> bool {
        width > 0 && height > 0 && self.copied == width * height
    }

    /// Erreur positionnelle max (px cible) parmi les pixels copiés.
    /// ≤ `XAOS_EXACT_TOLERANCE_PX` ⇒ les copies sont exactes (pas de
    /// raffinement nécessaire, pas de label ≈).
    pub fn max_abs_err(&self) -> f64 {
        self.max_err
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
        && params.aa_jitter.is_none()
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

/// Transformée 1D frame source → cible : `x_old = a·(x_new + 0.5) + B` avec
/// `a = (span_new/span_old)·(n_old/n_new)`, `B = n_old·(0.5 + Δc/span_old −
/// r/2) − 0.5`. Les ratios viennent de [`ViewHp::transform_to`].
fn axis_transform_from_ratios(dc: f64, r: f64, n_old: u32, n_new: u32) -> Option<(f64, f64)> {
    if n_old == 0 || n_new == 0 {
        return None;
    }
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

/// Résultat du matching d'un axe.
pub struct AxisMap {
    /// Candidat PRIMAIRE (meilleur par position hintée) par index cible (-1 = aucun).
    pub src: Vec<i32>,
    /// Candidat SECONDAIRE (meilleur par position nominale), -1 si absent ou
    /// identique au primaire.
    pub src2: Vec<i32>,
    /// Écart NOMINAL (px cible) `(k − p)/a` de chaque candidat.
    pub dev: Vec<f64>,
    pub dev2: Vec<f64>,
    /// Hint (px cible) pour la frame produite : écart hinté du primaire.
    pub err: Vec<f64>,
    /// Nombre d'indices avec au moins un candidat.
    pub reused: usize,
}

/// Matching d'UN axe : pour chaque index cible x (position nominale mappée
/// `p = a·(x+0.5) + B`), jusqu'à DEUX candidats parmi `round(p) ± 1` :
/// le plus proche par position HINTÉE (`k + err_hint[k]`, pixels copiés de
/// la colonne) et le plus proche par position NOMINALE (`k`, pixels frais
/// d'une colonne mixte). Chacun est retenu si sa distance en px CIBLE (`/a`)
/// est ≤ `tol` — puis INJECTIVITÉ JOINTE : chaque source k ne sert que la
/// cible la mieux alignée, tous candidats confondus (cf. doc module :
/// garantit du travail frais en zoom-in, no-op en pan/zoom-out). La décision
/// finale de copie est PAR PIXEL (`XaosMap::source_index`), sur l'erreur
/// vraie du pixel source.
pub fn build_axis_map(
    n_new: usize,
    n_old: usize,
    a: f64,
    b: f64,
    err_hint: &[f64],
    tol: f64,
) -> AxisMap {
    let mut out = AxisMap {
        src: vec![-1i32; n_new],
        src2: vec![-1i32; n_new],
        dev: vec![0.0f64; n_new],
        dev2: vec![0.0f64; n_new],
        err: vec![0.0f64; n_new],
        reused: 0,
    };
    if n_old == 0 || !(a > 0.0) || !a.is_finite() || !b.is_finite() {
        return out;
    }
    let get_err = |k: usize| err_hint.get(k).copied().unwrap_or(0.0);
    // Passe 1 : candidats (score signé px cible, k) par index cible —
    // [0] = hinté, [1] = nominal.
    let mut cand: Vec<[Option<(f64, usize)>; 2]> = vec![[None, None]; n_new];
    for (x, c) in cand.iter_mut().enumerate() {
        let p = a * (x as f64 + 0.5) + b;
        // Les positions vraies dévient de ≤ tol de la grille source : examiner
        // round(p) et ses deux voisins suffit (positions quasi-monotones).
        // `as i64` sature à ±9.2e18 pour |p| énorme (frame source deep puis
        // reset à span 4) : `k0 ± 1` débordait en debug. Hors plage → aucun
        // candidat (clamp à une valeur hors [0, n_old)).
        let k0 = p.round().clamp(-2.0, n_old as f64 + 2.0) as i64;
        let mut best_h: Option<(f64, usize)> = None;
        let mut best_n: Option<(f64, usize)> = None;
        for k in (k0 - 1)..=(k0 + 1) {
            if k < 0 || k >= n_old as i64 {
                continue;
            }
            let k = k as usize;
            let d_h = (k as f64 + get_err(k)) - p;
            let d_n = k as f64 - p;
            if best_h.map_or(true, |(bd, _)| d_h.abs() < bd.abs()) {
                best_h = Some((d_h, k));
            }
            if best_n.map_or(true, |(bd, _)| d_n.abs() < bd.abs()) {
                best_n = Some((d_n, k));
            }
        }
        let accept = |bd: Option<(f64, usize)>| {
            bd.and_then(|(d, k)| {
                let d_target = d / a; // px cible
                (d_target.abs() <= tol).then_some((d_target, k))
            })
        };
        c[0] = accept(best_h);
        c[1] = accept(best_n);
        // Même k des deux côtés → un seul candidat (le hinté).
        if let (Some((_, kh)), Some((_, kn))) = (c[0], c[1]) {
            if kh == kn {
                c[1] = None;
            }
        }
    }
    // Passe 2 : injectivité PAR VARIANTE — pour chaque source k, ne garder
    // que la cible la mieux alignée de chaque variante (les deux variantes
    // visent des populations de pixels DISJOINTES de la colonne : copiés à
    // `k + hint` vs frais à `k` — une injectivité jointe laissait un nominal
    // voué à échouer par pixel voler le hinté d'une autre cible). En zoom-in
    // (a < 1) c'est ce qui force ≥ (1−a)·n indices à être recalculés (fin de
    // l'écho pur) ; en pan/zoom-out le mapping est déjà injectif.
    let mut best_target: Vec<[Option<(f64, usize)>; 2]> = vec![[None, None]; n_old]; // (|d|, x)
    for (x, c) in cand.iter().enumerate() {
        for (v, cv) in c.iter().enumerate() {
            if let Some((d, k)) = cv {
                if best_target[*k][v].map_or(true, |(bd, _)| d.abs() < bd) {
                    best_target[*k][v] = Some((d.abs(), x));
                }
            }
        }
    }
    for (x, c) in cand.iter().enumerate() {
        let p = a * (x as f64 + 0.5) + b;
        let keep = |v: usize| -> Option<usize> {
            c[v].and_then(|(_, k)| {
                matches!(best_target[k][v], Some((_, bx)) if bx == x).then_some(k)
            })
        };
        let kh = keep(0);
        let kn = keep(1);
        let mut any = false;
        if let Some(k) = kh {
            out.src[x] = k as i32;
            out.dev[x] = (k as f64 - p) / a;
            out.err[x] = ((k as f64 + get_err(k)) - p) / a;
            any = true;
        }
        if let Some(k) = kn {
            if kh.is_some() {
                out.src2[x] = k as i32;
                out.dev2[x] = (k as f64 - p) / a;
            } else {
                // Seul le nominal survit : il devient le primaire (hint = son
                // écart nominal, ses pixels copiés seront à cette position).
                out.src[x] = k as i32;
                out.dev[x] = (k as f64 - p) / a;
                out.err[x] = out.dev[x];
            }
            any = true;
        }
        if any {
            out.reused += 1;
        }
    }
    out
}

/// Construit le mapping complet frame source → passe cible. `None` si le
/// fast-path n'est pas applicable (gates, fingerprint, transformée dégénérée)
/// ou si aucun pixel n'est réutilisable.
pub fn build_map(src: &XaosSourceFrame, params: &FractalParams) -> Option<XaosMap> {
    build_map_with_tolerance(src, params, XAOS_TOLERANCE_PX)
}

/// Variante à tolérance explicite. Le rendu interactif passe
/// `XAOS_TOLERANCE_PX` (0.5 px, écho fluide) ; le raffinement idle passe
/// `XAOS_EXACT_TOLERANCE_PX` — il ne conserve que les pixels dont la position
/// est déjà exacte et recalcule uniquement les approximations.
pub fn build_map_with_tolerance(
    src: &XaosSourceFrame,
    params: &FractalParams,
    tol: f64,
) -> Option<XaosMap> {
    if !params_allow_pixel_reuse(params) {
        return None;
    }
    let expected = src.width as usize * src.height as usize;
    if expected == 0
        || src.iterations.len() != expected
        || src.zs.len() != expected
        || src.err.len() != expected
    {
        return None;
    }
    if src.fingerprint != params_fingerprint(params) {
        return None;
    }
    let (cx, cy, sx, sy) = view_strings(params);
    let old_view = ViewHp::from_decimal_parts(
        &src.cx,
        &src.cy,
        &src.sx,
        &src.sy,
        src.width,
        src.height,
        TRANSFORM_PRECISION_BITS,
    )?;
    let new_view = ViewHp::from_decimal_parts(
        &cx,
        &cy,
        &sx,
        &sy,
        params.width,
        params.height,
        TRANSFORM_PRECISION_BITS,
    )?;
    let transform = old_view.transform_to(&new_view);
    let (ax, bx) = axis_transform_from_ratios(
        transform.offset_x,
        transform.scale_x,
        src.width,
        params.width,
    )?;
    let (ay, by) = axis_transform_from_ratios(
        transform.offset_y,
        transform.scale_y,
        src.height,
        params.height,
    )?;
    let (w, h) = (params.width as usize, params.height as usize);
    let cols = build_axis_map(w, src.width as usize, ax, bx, &src.col_err, tol);
    let rows = build_axis_map(h, src.height as usize, ay, by, &src.row_err, tol);
    let map = XaosMap {
        iterations: Arc::clone(&src.iterations),
        zs: Arc::clone(&src.zs),
        src_width: src.width as usize,
        src_col: cols.src,
        src_row: rows.src,
        src_col2: cols.src2,
        src_row2: rows.src2,
        col_dev: cols.dev,
        row_dev: rows.dev,
        col_dev2: cols.dev2,
        row_dev2: rows.dev2,
        inv_ax: 1.0 / ax,
        inv_ay: 1.0 / ay,
        src_err: Arc::clone(&src.err),
        tol,
        col_err: cols.err,
        row_err: rows.err,
        reused_cols: cols.reused,
        reused_rows: rows.reused,
        copied: 0,
        max_err: 0.0,
    }
    .finalize(w, h);
    if !map.any_reuse() {
        return None;
    }
    Some(map)
}

/// Mapping de RAFFINEMENT : même vue, mêmes dims → identité par axe, tolérance
/// EXACTE : seuls les pixels dont l'erreur vraie est ≤ ε sont conservés
/// (calculés frais, ou copiés alignés), les approximations sont recalculées.
/// C'est ce qui ramène le cycle zoom écho+refine à ~100 % du coût d'un rendu
/// frais. Vue ou dims différentes → matching exact classique (toujours
/// correct).
pub fn build_refine_map(src: &XaosSourceFrame, params: &FractalParams) -> Option<XaosMap> {
    build_map_with_tolerance(src, params, XAOS_EXACT_TOLERANCE_PX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::perturbation::ReferenceOrbitCache;
    use crate::fractal::{default_params_for_type, FractalType};
    use crate::render::{render_request, ProgressiveReuse, RenderOutput, RenderRequest};
    use std::sync::atomic::AtomicBool;

    fn render<'a>(
        params: &'a FractalParams,
        cancel: &'a Arc<AtomicBool>,
        reuse: Option<(&'a [u32], &'a [Complex64], u32, u32)>,
        cache: &mut Option<Arc<ReferenceOrbitCache>>,
        xaos: Option<&'a XaosMap>,
        _tiles: Option<&crate::render::tiles::TileOpts<'a>>,
    ) -> Option<RenderOutput> {
        let mut request = RenderRequest::new(params, cancel);
        if let Some(previous) = reuse {
            request = request.with_progressive_reuse(ProgressiveReuse::from(previous));
        }
        if let Some(map) = xaos {
            request = request.with_xaos(map);
        }
        render_request(request, cache)
    }

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
            err: Arc::new(vec![[0.0, 0.0]; n]),
            col_err: Arc::new(vec![0.0; params.width as usize]),
            row_err: Arc::new(vec![0.0; params.height as usize]),
            fingerprint: params_fingerprint(params),
        }
    }

    /// Frame produite par une passe avec `map` (même règle que la GUI :
    /// erreurs par pixel héritées du mapping).
    fn frame_from_map(
        params: &FractalParams,
        iters: Vec<u32>,
        zs: Vec<Complex64>,
        map: &XaosMap,
    ) -> XaosSourceFrame {
        let (cx, cy, sx, sy) = view_strings(params);
        XaosSourceFrame::from_map(
            Arc::new(iters),
            Arc::new(zs),
            params.width,
            params.height,
            (cx, cy, sx, sy),
            params_fingerprint(params),
            map,
        )
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
        assert!(
            (map.col_err[5] + 0.3).abs() < 1e-9,
            "err = {}",
            map.col_err[5]
        );
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
        let src_b = frame_from_map(
            &pb,
            vec![1; 64 * 64],
            vec![Complex64::new(0.0, 0.0); 64 * 64],
            &map_ab,
        );

        let mut pc = pb.clone();
        pc.center_x += step;
        let map_bc = build_map(&src_b, &pc).expect("B→C");
        let err_c = map_bc.produced_err(64, 64);
        let mut copied = 0;
        for x in 0..64usize {
            // Décision PAR PIXEL : un pixel copié vient de la colonne DÉCALÉE
            // x+1 (vraie à +0.4), jamais de la colonne x (vraie à −0.6, hors
            // tolérance) — preuve que la comparaison se fait contre la
            // position vraie, pas la grille nominale.
            if let Some(sidx) = map_bc.source_index(x, 10) {
                copied += 1;
                assert_eq!(sidx % 64, x + 1, "colonne source de {x}");
                let e = err_c[10 * 64 + x];
                assert!((e[0] - 0.4).abs() < 1e-6, "err produite = {}", e[0]);
                assert!(
                    e[0].abs() as f64 <= XAOS_TOLERANCE_PX + 1e-9,
                    "dérive non bornée"
                );
            }
        }
        assert_eq!(
            copied, 63,
            "toutes les colonnes sauf la dernière (x+1 = 64 hors champ)"
        );
        assert!(map_bc.max_abs_err() <= XAOS_TOLERANCE_PX + 1e-9);
    }

    /// Verrou bug 2026-08-23 (modèle d'erreur par axe non sain) : zoom
    /// molette CONTINU ×1.2 ancré hors centre, 6 crans SANS raffinement —
    /// l'écart VRAI de chaque pixel copié (coordonnée complexe stockée dans
    /// `zs`, comparée à la coordonnée nominale) reste ≤ 0.5 px à chaque cran.
    /// Avant : 0.48 → 0.76 → 1.03 → … → 1.5 px (croissance géométrique).
    #[test]
    fn continuous_anchored_zoom_keeps_true_error_bounded() {
        let (w, h) = (101u32, 77u32);
        let coord = |p: &FractalParams, i: usize, j: usize| {
            Complex64::new(
                p.center_x + ((i as f64 + 0.5) / w as f64 - 0.5) * p.span_x,
                p.center_y + ((j as f64 + 0.5) / h as f64 - 0.5) * p.span_y,
            )
        };
        // Frame exacte initiale : zs = coordonnée vraie du pixel.
        let mut p = base_params(w, h);
        let n = (w * h) as usize;
        let zs: Vec<Complex64> = (0..n)
            .map(|k| coord(&p, k % w as usize, k / w as usize))
            .collect();
        let mut frame = frame_for(&p, vec![1; n]);
        frame.zs = Arc::new(zs);
        let (rx, ry) = (0.3, 0.7);
        for step in 0..6 {
            // Zoom ×1.2 ancré en (rx, ry) : le point sous le curseur est fixe.
            let f = 1.2;
            let mut z = p.clone();
            z.span_x = p.span_x / f;
            z.span_y = p.span_y / f;
            z.center_x = p.center_x + (rx - 0.5) * p.span_x * (1.0 - 1.0 / f);
            z.center_y = p.center_y + (ry - 0.5) * p.span_y * (1.0 - 1.0 / f);
            let map = build_map(&frame, &z).expect("écho zoom");
            // « Rendu » : copie via source_index, calcul exact sinon.
            let mut zs_new = vec![Complex64::new(0.0, 0.0); n];
            let mut worst = 0.0f64;
            let mut copied = 0usize;
            for j in 0..h as usize {
                for i in 0..w as usize {
                    let idx = j * w as usize + i;
                    zs_new[idx] = match map.source_index(i, j) {
                        Some(s) => {
                            copied += 1;
                            frame.zs[s]
                        }
                        None => coord(&z, i, j),
                    };
                    let nominal = coord(&z, i, j);
                    let dx = (zs_new[idx].re - nominal.re) / (z.span_x / w as f64);
                    let dy = (zs_new[idx].im - nominal.im) / (z.span_y / h as f64);
                    worst = worst.max(dx.abs()).max(dy.abs());
                }
            }
            assert!(copied > 0, "cran {step} : l'écho doit copier");
            assert!(
                worst <= XAOS_TOLERANCE_PX + 1e-6,
                "cran {step} : écart vrai max {worst:.3} px > tolérance (dérive)"
            );
            // L'erreur DÉCLARÉE de la frame produite doit coïncider avec l'écart vrai.
            let produced = map.produced_err(w as usize, h as usize);
            for j in 0..h as usize {
                for i in 0..w as usize {
                    let idx = j * w as usize + i;
                    let nominal = coord(&z, i, j);
                    let dx = (zs_new[idx].re - nominal.re) / (z.span_x / w as f64);
                    let dy = (zs_new[idx].im - nominal.im) / (z.span_y / h as f64);
                    assert!(
                        (produced[idx][0] as f64 - dx).abs() < 1e-4
                            && (produced[idx][1] as f64 - dy).abs() < 1e-4,
                        "cran {step} px ({i},{j}) : err déclarée {:?} ≠ vraie ({dx:.4},{dy:.4})",
                        produced[idx]
                    );
                }
            }
            frame = frame_from_map(&z, vec![1; n], zs_new, &map);
            p = z;
        }
    }

    /// G5 property test — généralisation de
    /// `continuous_anchored_zoom_keeps_true_error_bounded` : séquences
    /// ALÉATOIRES pan / zoom-in ancré / zoom-out / resize, avec un oracle en
    /// précision arbitraire (l'état de vue vit en `rug::Float`, la position
    /// vraie de chaque pixel est propagée en fractions de span). Invariant à
    /// CHAQUE pas : « écart vrai ≤ tolérance ∧ erreur déclarée == vraie ».
    /// Paramétré en profondeur jusqu'à 1e-300 : le trou de couverture
    /// 1e-30 → 1e-74 avait caché le bug de la transformée à 256 b fixes
    /// (Δcentre d'un zoom ancré arrondi à zéro, 2026-08-23).
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        /// Uniforme dans [0, 1).
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn run_random_navigation(span0: &str, prec: u32, seed: u64) {
        let deep = span0 != "4.0";
        let mut rng = XorShift(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        let (mut w, mut h) = (72u32, 56u32);
        let mut cx = Float::with_val(prec, Float::parse("-0.7436").unwrap());
        let mut cy = Float::with_val(prec, Float::parse("0.1318").unwrap());
        let mut sx = Float::with_val(prec, Float::parse(span0).unwrap());
        let mut sy = Float::with_val(prec, &sx * (h as f64 / w as f64));

        // Params depuis l'état HP. Hors deep : état QUANTIFIÉ f64 (le path
        // strings `%.17e` de `view_strings` est alors exactement l'état).
        let make_params = |cx: &Float, cy: &Float, sx: &Float, sy: &Float, w: u32, h: u32| {
            let mut p = base_params(w, h);
            p.center_x = cx.to_f64();
            p.center_y = cy.to_f64();
            p.span_x = sx.to_f64();
            p.span_y = sy.to_f64();
            if deep {
                p.center_x_hp = Some(cx.to_string_radix(10, None));
                p.center_y_hp = Some(cy.to_string_radix(10, None));
                p.span_x_hp = Some(sx.to_string_radix(10, None));
                p.span_y_hp = Some(sy.to_string_radix(10, None));
            }
            p
        };
        let quantize = |v: &mut Float| {
            if !deep {
                let f = v.to_f64();
                *v = Float::with_val(prec, f);
            }
        };

        let mut p = make_params(&cx, &cy, &sx, &sy, w, h);
        let nom = |k: usize, n: u32| (k as f64 + 0.5) / n as f64 - 0.5;
        let n0 = (w * h) as usize;
        // Oracle : position vraie de chaque pixel en FRACTION du span courant
        // (u = (coord − centre)/span) — représentable en f64 quel que soit le
        // zoom, mise à jour par la transformée HP exacte à chaque pas.
        let mut ux: Vec<f64> = (0..n0).map(|k| nom(k % w as usize, w)).collect();
        let mut uy: Vec<f64> = (0..n0).map(|k| nom(k / w as usize, h)).collect();
        let mut frame = frame_for(&p, vec![1; n0]);
        let mut copied_total = 0usize;

        for step in 0..10 {
            let (mut cx2, mut cy2, mut sx2, mut sy2) =
                (cx.clone(), cy.clone(), sx.clone(), sy.clone());
            let (mut w2, mut h2) = (w, h);
            let op = rng.unit();
            if op < 0.40 {
                // Zoom-in ancré (rx, ry) : le point sous le curseur est fixe.
                let f = 1.05 + 0.85 * rng.unit();
                let (rx, ry) = (0.1 + 0.8 * rng.unit(), 0.1 + 0.8 * rng.unit());
                cx2 += Float::with_val(prec, &sx * ((rx - 0.5) * (1.0 - 1.0 / f)));
                cy2 += Float::with_val(prec, &sy * ((ry - 0.5) * (1.0 - 1.0 / f)));
                sx2 /= f;
                sy2 /= f;
            } else if op < 0.70 {
                // Pan en pixels (fractionnaire une fois sur deux).
                let mut dx = 16.0 * rng.unit() - 8.0;
                let mut dy = 16.0 * rng.unit() - 8.0;
                if rng.unit() < 0.5 {
                    dx = dx.round();
                    dy = dy.round();
                }
                cx2 += Float::with_val(prec, &sx * (dx / w as f64));
                cy2 += Float::with_val(prec, &sy * (dy / h as f64));
            } else if op < 0.85 {
                // Zoom-out centré.
                let f = 1.05 + 0.85 * rng.unit();
                sx2 *= f;
                sy2 *= f;
            } else {
                // Resize fenêtre (mêmes centre/span x ; span y suit le ratio).
                w2 = 48 + (rng.next() % 64) as u32;
                h2 = 48 + (rng.next() % 64) as u32;
                sy2 = Float::with_val(prec, &sx2 * (h2 as f64 / w2 as f64));
            }
            quantize(&mut cx2);
            quantize(&mut cy2);
            quantize(&mut sx2);
            quantize(&mut sy2);

            let p2 = make_params(&cx2, &cy2, &sx2, &sy2, w2, h2);
            let n2 = (w2 * h2) as usize;

            // Transformée HP de l'oracle : u' = u·(s/s') + (c − c')/s'.
            let rx_o = Float::with_val(prec, &sx / &sx2).to_f64();
            let dx_o = (Float::with_val(prec, &cx - &cx2) / &sx2).to_f64();
            let ry_o = Float::with_val(prec, &sy / &sy2).to_f64();
            let dy_o = (Float::with_val(prec, &cy - &cy2) / &sy2).to_f64();

            let map = build_map(&frame, &p2);
            let mut ux2 = vec![0.0f64; n2];
            let mut uy2 = vec![0.0f64; n2];
            match &map {
                Some(m) => {
                    let produced = m.produced_err(w2 as usize, h2 as usize);
                    let mut worst = 0.0f64;
                    for j in 0..h2 as usize {
                        for i in 0..w2 as usize {
                            let idx = j * w2 as usize + i;
                            match m.source_index(i, j) {
                                Some(sidx) => {
                                    copied_total += 1;
                                    ux2[idx] = ux[sidx] * rx_o + dx_o;
                                    uy2[idx] = uy[sidx] * ry_o + dy_o;
                                }
                                None => {
                                    ux2[idx] = nom(i, w2);
                                    uy2[idx] = nom(j, h2);
                                }
                            }
                            // Écart vrai (px cible) vs tolérance ET vs erreur
                            // déclarée de la frame produite.
                            let ex = (ux2[idx] - nom(i, w2)) * w2 as f64;
                            let ey = (uy2[idx] - nom(j, h2)) * h2 as f64;
                            worst = worst.max(ex.abs()).max(ey.abs());
                            assert!(
                                ex.abs() <= XAOS_TOLERANCE_PX + 1e-6
                                    && ey.abs() <= XAOS_TOLERANCE_PX + 1e-6,
                                "span0={span0} seed={seed} pas {step} px ({i},{j}) : \
                                 écart vrai ({ex:.4},{ey:.4}) px > tolérance"
                            );
                            assert!(
                                (produced[idx][0] as f64 - ex).abs() < 2e-3
                                    && (produced[idx][1] as f64 - ey).abs() < 2e-3,
                                "span0={span0} seed={seed} pas {step} px ({i},{j}) : \
                                 err déclarée {:?} ≠ vraie ({ex:.4},{ey:.4})",
                                produced[idx]
                            );
                        }
                    }
                    let _ = worst;
                    frame = frame_from_map(&p2, vec![1; n2], vec![Complex64::new(0.0, 0.0); n2], m);
                }
                None => {
                    // Pas de réutilisation possible → rendu frais complet.
                    for j in 0..h2 as usize {
                        for i in 0..w2 as usize {
                            let idx = j * w2 as usize + i;
                            ux2[idx] = nom(i, w2);
                            uy2[idx] = nom(j, h2);
                        }
                    }
                    frame = frame_for(&p2, vec![1; n2]);
                }
            }

            ux = ux2;
            uy = uy2;
            (cx, cy, sx, sy) = (cx2, cy2, sx2, sy2);
            (w, h) = (w2, h2);
            p = p2;
            let _ = &p;
        }
        assert!(
            copied_total > 0,
            "span0={span0} seed={seed} : aucune copie sur toute la séquence (gates trop stricts ?)"
        );
    }

    /// G5 : property test des séquences de navigation, du plan f64 au deep
    /// 1e-300 (précision de l'oracle ≈ −log2(span) + large marge).
    #[test]
    fn random_navigation_sequences_keep_declared_error_true() {
        for (span0, prec) in [
            ("4.0", 256u32),
            ("1e-20", 384),
            ("1e-40", 512),
            ("1e-80", 768),
            ("1e-150", 1024),
            ("1e-300", 1536),
        ] {
            for seed in [1u64, 42, 20260824] {
                run_random_navigation(span0, prec, seed);
            }
        }
    }

    #[test]
    fn zoom_in_2x_maps_center_half_injectively() {
        let p = base_params(64, 64);
        let mut zoomed = p.clone();
        zoomed.span_x = p.span_x / 2.0;
        zoomed.span_y = p.span_y / 2.0;
        let src = frame_for(&p, vec![1; 64 * 64]);
        let map = build_map(&src, &zoomed).expect("map");
        // a = 0.5 : la cible couvre les colonnes source 16..48. Positions
        // mappées quart-entières (15.75, 16.25, …) → chaque source est à
        // 0.5 px cible de DEUX cibles ; l'injectivité n'en garde qu'une →
        // 32 colonnes copiées (écho), 32 recalculées (travail frais garanti,
        // fin du zoom "écho pur" qui ne calculait rien).
        assert_eq!(map.reused_cols, 32);
        assert_eq!(map.reused_rows, 32);
        assert_eq!(map.src_col[0], 16);
        // Aucune colonne source dupliquée.
        let mut seen = std::collections::HashSet::new();
        for &k in &map.src_col {
            if k >= 0 {
                assert!(seen.insert(k), "colonne source {k} dupliquée");
            }
        }
        // Les copies restent dans la fenêtre source du zoom ×2 (16..48).
        for &k in map.src_col.iter().filter(|&&k| k >= 0) {
            assert!((16..48).contains(&k), "source hors fenêtre : {k}");
        }
    }

    #[test]
    fn zoom_in_leaves_fresh_work_proportional_to_factor() {
        // Pour tout facteur de zoom-in, l'injectivité garantit ≥ (1−a)·n
        // colonnes fraîches par axe — un zoom ne peut plus être un écho pur.
        for factor in [2.0, 1.5, 1.25, 1.1] {
            let p = base_params(64, 64);
            let mut z = p.clone();
            z.span_x = p.span_x / factor;
            z.span_y = p.span_y / factor;
            let src = frame_for(&p, vec![1; 64 * 64]);
            let map = build_map(&src, &z).expect("map");
            let a = 1.0 / factor;
            let max_reused = (a * 64.0).ceil() as usize;
            assert!(
                map.reused_cols <= max_reused,
                "×{factor} : {} colonnes copiées > plafond injectif {max_reused}",
                map.reused_cols
            );
            assert!(!map.is_pure_copy(64, 64), "×{factor} : écho pur interdit");
            assert!(
                map.any_reuse(),
                "×{factor} : l'écho doit copier quelque chose"
            );
        }
    }

    #[test]
    fn union_refine_rejects_aligned_but_shifted_frame() {
        // Pan horizontal fractionnaire : toutes les colonnes copiées à −0.3 px,
        // toutes les lignes ALIGNÉES (err 0) — mais chaque pixel est décalé.
        // Le refine union ne doit RIEN conserver (une ligne alignée n'est pas
        // une ligne exacte : ses pixels sont décalés par l'axe colonne).
        let p = base_params(64, 64);
        let mut moved = p.clone();
        moved.center_x = p.center_x + 0.3 * p.span_x / 64.0;
        let src = frame_for(&p, vec![1; 64 * 64]);
        let echo = build_map(&src, &moved).expect("écho");
        assert_eq!(echo.reused_cols, 64, "pan 0.3 px : tout copié");
        // Frame résultat du pan : positions vraies −0.3 sur TOUS les pixels.
        let frame_b = frame_from_map(
            &moved,
            vec![1; 64 * 64],
            vec![Complex64::new(0.0, 0.0); 64 * 64],
            &echo,
        );
        assert!(frame_b
            .err
            .iter()
            .all(|e| (e[0] + 0.3).abs() < 1e-6 && e[1].abs() < 1e-6));
        assert!(
            build_refine_map(&frame_b, &moved).is_none(),
            "tout est approximé : le refine doit tout recalculer"
        );
    }

    #[test]
    fn exact_tolerance_map_keeps_only_true_positions() {
        // Frame mi-exacte (colonnes paires err 0, impaires approximées 0.4 px) :
        // le map identité à tolérance exacte (celui du raffinement idle) ne
        // copie que les colonnes vraies et recalcule les approximations.
        let p = base_params(64, 64);
        let mut src = frame_for(&p, vec![1; 64 * 64]);
        let mut err = vec![[0.0f32, 0.0]; 64 * 64];
        for (idx, e) in err.iter_mut().enumerate() {
            if (idx % 64) % 2 == 1 {
                e[0] = 0.4;
            }
        }
        src.err = Arc::new(err);
        let map = build_map_with_tolerance(&src, &p, XAOS_EXACT_TOLERANCE_PX).expect("map refine");
        assert_eq!(
            map.copied,
            32 * 64,
            "seules les colonnes exactes sont conservées"
        );
        for x in 0..64usize {
            if x % 2 == 0 {
                assert_eq!(
                    map.source_index(x, 7),
                    Some(7 * 64 + x),
                    "colonne exacte conservée"
                );
            } else {
                assert_eq!(
                    map.source_index(x, 7),
                    None,
                    "colonne approximée recalculée"
                );
            }
        }
        assert!(map.max_abs_err() <= XAOS_EXACT_TOLERANCE_PX);
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

    /// Verrou bug 2026-08-23 : transformée à précision dynamique — zoom
    /// ancré hors centre à span 1e-80 (au-delà des 256 b fixes), le map voit
    /// le décalage de centre (Δc ≈ 10⁻⁸¹, invisible à 256 b).
    #[test]
    fn deep_anchored_zoom_transform_sees_off_center_delta() {
        let (w, h) = (1000u32, 10u32);
        let prec = 512;
        let mut p = base_params(w, h);
        let cx0 = Float::with_val(prec, Float::parse("-0.75").unwrap());
        let span = Float::with_val(prec, Float::parse("1e-80").unwrap());
        let digits = Some(120);
        p.center_x_hp = Some(cx0.to_string_radix(10, digits));
        p.center_y_hp = Some("0.1".into());
        p.span_x_hp = Some(span.to_string_radix(10, digits));
        p.span_y_hp = Some(Float::with_val(prec, &span / 100).to_string_radix(10, digits));
        // Zoom ×1.2 ancré en rx = 0.9 : Δcx = (0.9−0.5)·span·(1−1/1.2).
        let f = 1.2f64;
        let mut z = p.clone();
        let dcx = Float::with_val(prec, &span * (0.4 * (1.0 - 1.0 / f)));
        z.center_x_hp = Some(Float::with_val(prec, &cx0 + &dcx).to_string_radix(10, digits));
        z.span_x_hp = Some(Float::with_val(prec, &span / f).to_string_radix(10, digits));
        z.span_y_hp = Some(Float::with_val(prec, &span / (100.0 * f)).to_string_radix(10, digits));
        let src = frame_for(&p, vec![1; (w * h) as usize]);
        let map = build_map(&src, &z).expect("map");
        // Vérité : colonne cible x ↔ position source p = a·(x+0.5)+B avec
        // a = 1/f, B = n·(0.5 + 0.4·(1−1/f) − 0.5/f) − 0.5.
        let a = 1.0 / f;
        let b = w as f64 * (0.5 + 0.4 * (1.0 - 1.0 / f) - 0.5 / f) - 0.5;
        for x in [0usize, 250, 500, 750, 999] {
            let k = map.src_col[x];
            if k >= 0 {
                let p_true = a * (x as f64 + 0.5) + b;
                assert!(
                    (k as f64 - p_true).abs() <= 0.5 + 1e-6,
                    "colonne {x} : source {k} vs position vraie {p_true:.3} (Δcentre perdu ?)"
                );
            }
        }
        assert!(map.any_reuse());
    }

    #[test]
    fn fingerprint_change_disables_reuse() {
        let p = base_params(64, 64);
        let src = frame_for(&p, vec![1; 64 * 64]);
        let mut other = p.clone();
        other.iteration_max += 100;
        assert!(
            build_map(&src, &other).is_none(),
            "iteration_max ≠ → pas de reuse"
        );
        let mut geom = p.clone();
        geom.center_x += 0.01;
        assert!(
            build_map(&src, &geom).is_some(),
            "géométrie seule ≠ → reuse OK"
        );
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
        params.use_bytecode_engine = true;
        let cancel = Arc::new(AtomicBool::new(false));
        let __out = render(&params, &cancel, None, &mut None, None, None).expect("A");
        let (it_a, zs_a) = (__out.iterations, __out.zs);

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

        let mut src = frame_for(&params, vec![0; (params.width * params.height) as usize]);
        src.iterations = Arc::new(it_a);
        src.zs = Arc::new(zs_a);
        let map = build_map(&src, &moved).expect("map");
        assert!(map.any_reuse(), "pan entier doit réutiliser des bandes");
        assert_eq!(map.reused_cols, params.width as usize - 8);
        assert_eq!(map.reused_rows, params.height as usize - 3);
        for &e in map.col_err.iter().chain(map.row_err.iter()) {
            assert!(e.abs() < 1e-9, "pan entier ⇒ copies exactes, err={e}");
        }

        let __out = render(&moved, &cancel, None, &mut None, Some(&map), None).expect("B xaos");
        let (it_xaos, zs_xaos) = (__out.iterations, __out.zs);
        let __out = render(&moved, &cancel, None, &mut None, None, None).expect("B fresh");
        let (it_fresh, zs_fresh) = (__out.iterations, __out.zs);
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

    /// Bout-en-bout ZOOM : rendu A → zoom-in ×2 avec le mapping XaoS (écho
    /// injectif : copies approximées + colonnes fraîches) → raffinement à
    /// tolérance exacte → pixel-identique au rendu frais de la vue zoomée.
    /// Verrouille le cycle interactif complet du zoom (écho → refine).
    #[test]
    fn zoom_then_exact_refine_matches_fresh_render() {
        let mut p = base_params(64, 48);
        p.center_x = -0.6;
        p.center_y = 0.3;
        p.span_x = 0.5;
        p.span_y = 0.375;
        p.iteration_max = 300;
        p.use_bytecode_engine = true;
        let cancel = Arc::new(AtomicBool::new(false));
        let __out = render(&p, &cancel, None, &mut None, None, None).expect("A");
        let (it_a, zs_a) = (__out.iterations, __out.zs);

        // Zoom ×2 centré : cas historiquement dégénéré (100 % d'écho, 0 pixel
        // frais avant l'injectivité).
        let mut z = p.clone();
        z.span_x = p.span_x / 2.0;
        z.span_y = p.span_y / 2.0;

        let mut src = frame_for(&p, vec![0; 64 * 48]);
        src.iterations = Arc::new(it_a);
        src.zs = Arc::new(zs_a);
        let map = build_map(&src, &z).expect("map écho");
        let total = 64usize * 48;
        let copied = map.copied;
        assert!(copied > 0, "l'écho zoom doit copier des pixels");
        assert!(
            !map.is_pure_copy(64, 48),
            "le zoom doit laisser du travail frais (injectivité)"
        );
        assert!(copied < total);
        let __out = render(&z, &cancel, None, &mut None, Some(&map), None).expect("B écho");
        let (it_b, zs_b) = (__out.iterations, __out.zs);

        // Frame B avec ses erreurs héritées → map de raffinement UNION :
        // conserve tout pixel dont un axe est frais (calculé à l'écho),
        // recalcule uniquement les copies approximées.
        let src_b = frame_from_map(&z, it_b, zs_b, &map);
        let refine_map = build_refine_map(&src_b, &z).expect("map refine");
        assert!(refine_map.max_abs_err() <= XAOS_EXACT_TOLERANCE_PX);
        assert!(
            refine_map.any_reuse(),
            "le refine doit garder les pixels frais de B"
        );
        // L'union garde STRICTEMENT plus que le produit : tout pixel calculé
        // frais à l'écho (colonne OU ligne fraîche) est conservé.
        let kept = refine_map.reused_fraction(64, 48);
        let echo_fresh = 1.0 - map.reused_fraction(64, 48);
        assert!(
            kept >= echo_fresh - 1e-9,
            "union ({kept:.3}) doit couvrir au moins les pixels frais de l'écho ({echo_fresh:.3})"
        );

        let __out =
            render(&z, &cancel, None, &mut None, Some(&refine_map), None).expect("C refine");
        let (it_c, zs_c) = (__out.iterations, __out.zs);
        let __out = render(&z, &cancel, None, &mut None, None, None).expect("frais");
        let (it_f, zs_f) = (__out.iterations, __out.zs);
        assert_eq!(it_c, it_f, "refine ε == rendu frais (itérations)");
        let zdiff = zs_c
            .iter()
            .zip(&zs_f)
            .filter(|(a, b)| (*a - *b).norm() > 1e-12)
            .count();
        assert_eq!(zdiff, 0, "refine ε == rendu frais (zs)");
    }

    /// INVARIANT G10.4b : écho XaoS et reuse basse-résolution inter-passes
    /// sont mutuellement exclusifs. Le reuse copie des pixels dont le centre
    /// est décalé de (ratio−1)/2 px — s'il fuyait dans les colonnes/lignes
    /// FRAÎCHES d'un rendu avec map, la frame mentirait (col_exact = true sur
    /// des pixels approximés) et le refine union garderait ces pixels faux.
    /// On rend avec un buffer reuse EMPOISONNÉ : aucune valeur ne doit fuiter.
    #[test]
    fn echo_pass_ignores_coarse_pass_reuse() {
        let mut p = base_params(64, 48);
        p.center_x = -0.6;
        p.center_y = 0.3;
        p.span_x = 0.5;
        p.span_y = 0.375;
        p.iteration_max = 300;
        p.use_bytecode_engine = true;
        let cancel = Arc::new(AtomicBool::new(false));
        let __out = render(&p, &cancel, None, &mut None, None, None).expect("A");
        let (it_a, zs_a) = (__out.iterations, __out.zs);

        let mut z = p.clone();
        z.span_x = p.span_x / 2.0;
        z.span_y = p.span_y / 2.0;
        let mut src = frame_for(&p, vec![0; 64 * 48]);
        src.iterations = Arc::new(it_a);
        src.zs = Arc::new(zs_a);
        let map = build_map(&src, &z).expect("map écho");

        // Reuse 32×24 empoisonné (valeurs impossibles) — comme une passe 1/2.
        let poison_it = vec![u32::MAX; 32 * 24];
        let poison_zs = vec![Complex64::new(1e300, -1e300); 32 * 24];
        let poisoned = Some((poison_it.as_slice(), poison_zs.as_slice(), 32u32, 24u32));

        let __out =
            render(&z, &cancel, poisoned, &mut None, Some(&map), None).expect("écho+poison");
        let (it_poison, zs_poison) = (__out.iterations, __out.zs);
        assert!(
            !it_poison.iter().any(|&it| it == u32::MAX),
            "le reuse basse-résolution a fuité dans un rendu avec map XaoS"
        );
        let __out = render(&z, &cancel, None, &mut None, Some(&map), None).expect("écho seul");
        let (it_clean, zs_clean) = (__out.iterations, __out.zs);
        assert_eq!(it_poison, it_clean, "écho+reuse == écho seul (itérations)");
        assert_eq!(
            zs_poison
                .iter()
                .zip(&zs_clean)
                .filter(|(a, b)| a != b)
                .count(),
            0,
            "écho+reuse == écho seul (zs)"
        );
        // Sans map, le reuse reste actif (comportement progressif inchangé).
        let __out = render(&z, &cancel, poisoned, &mut None, None, None).expect("reuse seul");
        let (it_no_map, _) = (__out.iterations, __out.zs);
        assert!(
            it_no_map.iter().any(|&it| it == u32::MAX),
            "sans map XaoS le reuse doit rester actif (grille alignée copiée)"
        );
    }

    /// Diagnostic perf (non-CI) : gain wall-clock d'un pan 8 px avec XaoS vs
    /// rendu complet. `cargo test --release --bin fractall-cli xaos_pan_speedup -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn xaos_pan_speedup_diagnostic() {
        use std::time::Instant;

        let mut p = base_params(1024, 768);
        p.center_x = -0.743643135;
        p.center_y = 0.131825963;
        p.span_x = 2e-7;
        p.span_y = 1.5e-7;
        p.iteration_max = 20000;
        let cancel = Arc::new(AtomicBool::new(false));
        let t0 = Instant::now();
        let __out = render(&p, &cancel, None, &mut None, None, None).expect("A");
        let (it_a, zs_a) = (__out.iterations, __out.zs);
        let t_full = t0.elapsed();

        let mut moved = p.clone();
        moved.center_x += 8.0 * p.span_x / p.width as f64;
        moved.center_y += 3.0 * p.span_y / p.height as f64;
        let mut src = frame_for(&p, vec![0; (p.width * p.height) as usize]);
        src.iterations = Arc::new(it_a);
        src.zs = Arc::new(zs_a);
        let t1 = Instant::now();
        let map = build_map(&src, &moved).expect("map");
        let t_map = t1.elapsed();
        let reused_px = (map.reused_cols * map.reused_rows) as f64;
        let total_px = (p.width * p.height) as f64;
        let t2 = Instant::now();
        let _ = render(&moved, &cancel, None, &mut None, Some(&map), None).expect("B xaos");
        let t_xaos = t2.elapsed();
        println!(
            "full={:?} xaos={:?} (map build {:?}) — pixels copiés {:.1}% — speedup ×{:.1}",
            t_full,
            t_xaos,
            t_map,
            100.0 * reused_px / total_px,
            t_full.as_secs_f64() / t_xaos.as_secs_f64().max(1e-9),
        );
    }

    /// Diagnostic perf (non-CI) : cycle zoom ×2 complet — passe écho (copies
    /// injectives + colonnes fraîches) puis raffinement ε partiel — vs rendu
    /// frais. `cargo test --release --bin fractall-cli xaos_zoom_cycle_diagnostic -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn xaos_zoom_cycle_diagnostic() {
        use std::time::Instant;

        let mut p = base_params(1024, 768);
        p.center_x = -0.743643135;
        p.center_y = 0.131825963;
        p.span_x = 2e-7;
        p.span_y = 1.5e-7;
        p.iteration_max = 20000;
        let cancel = Arc::new(AtomicBool::new(false));
        let __out = render(&p, &cancel, None, &mut None, None, None).expect("A");
        let (it_a, zs_a) = (__out.iterations, __out.zs);

        let mut z = p.clone();
        z.span_x = p.span_x / 2.0;
        z.span_y = p.span_y / 2.0;
        let t0 = Instant::now();
        let __out = render(&z, &cancel, None, &mut None, None, None).expect("frais");
        let (it_f, _) = (__out.iterations, __out.zs);
        let t_full = t0.elapsed();

        let (cx, cy, sx, sy) = view_strings(&p);
        let mut src = frame_for(&p, vec![0; 1024 * 768]);
        src.iterations = Arc::new(it_a);
        src.zs = Arc::new(zs_a);
        src.cx = cx;
        src.cy = cy;
        src.sx = sx;
        src.sy = sy;
        let map = build_map(&src, &z).expect("map écho");
        let copied = map.copied;
        let t1 = Instant::now();
        let __out = render(&z, &cancel, None, &mut None, Some(&map), None).expect("B");
        let (it_b, zs_b) = (__out.iterations, __out.zs);
        let t_echo = t1.elapsed();

        let src_b = frame_from_map(&z, it_b, zs_b, &map);
        let refine_map = build_refine_map(&src_b, &z).expect("map refine");
        let kept = (refine_map.reused_fraction(1024, 768) * 1024.0 * 768.0) as usize;
        let t2 = Instant::now();
        let __out = render(&z, &cancel, None, &mut None, Some(&refine_map), None).expect("C");
        let (it_c, _) = (__out.iterations, __out.zs);
        let t_refine = t2.elapsed();
        assert_eq!(it_c, it_f, "cycle écho+refine == rendu frais");

        let total = 1024.0 * 768.0;
        println!(
            "zoom ×2 : frais={t_full:?} | écho={t_echo:?} (copiés {:.1}%) | refine={t_refine:?} \
             (conservés {:.1}%) — écho ×{:.1}, refine ×{:.1}, cycle total {:.0}% d'un frais",
            100.0 * copied as f64 / total,
            100.0 * kept as f64 / total,
            t_full.as_secs_f64() / t_echo.as_secs_f64().max(1e-9),
            t_full.as_secs_f64() / t_refine.as_secs_f64().max(1e-9),
            100.0 * (t_echo.as_secs_f64() + t_refine.as_secs_f64()) / t_full.as_secs_f64(),
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
