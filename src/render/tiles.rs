//! G10.5 — File de tuiles priorité-centre.
//!
//! Remplace le `par_chunks_mut` monolithique des boucles pixel CPU par une
//! work-queue de tuiles rectangulaires ordonnée « point d'intérêt d'abord »
//! (curseur en GUI, centre sinon). Deux propriétés que rayon seul ne donne pas :
//!
//! 1. **Ordre d'exécution réel** : le split binaire de `par_chunks_mut`
//!    distribue les fronts de ~N segments répartis sur toute l'image ; ici une
//!    file atomique (`fetch_add`) garantit que les tuiles partent dans l'ordre
//!    de priorité (le point d'intérêt est calculé en premier).
//! 2. **Streaming intra-passe** : un `sink` optionnel reçoit chaque tuile
//!    terminée (buffers absolus iterations/zs), ce qui permet à la GUI
//!    d'afficher la progression pendant la passe au lieu d'attendre sa fin.
//!
//! Sûreté : les buffers de sortie sont découpés EN AMONT en segments de lignes
//! disjoints par tuile (`TileGrid::split`, chaîne de `split_at_mut`) — aucune
//! écriture partagée, pas d'`unsafe`. L'ordre d'exécution ne change pas les
//! valeurs calculées (chaque pixel est une fonction pure de ses coordonnées) :
//! CLI/goldens/quality sont insensibles au scheduling.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use num_complex::Complex64;

/// Options de scheduling passées par la GUI au dispatcher unique.
/// `None` (CLI/quality/tests) ⇒ priorité centre, pas de sink.
pub struct TileOpts<'a> {
    /// Point de priorité en coordonnées NORMALISÉES [0,1]² de la frame
    /// (indépendant de la résolution de passe). (0.5, 0.5) = centre.
    pub priority: (f64, f64),
    /// Callback appelé à la complétion de chaque tuile, depuis les threads
    /// workers (d'où `Sync`). Les buffers sont des copies tuile-locales.
    pub sink: Option<&'a (dyn Fn(TileUpdate) + Sync)>,
}

/// Contenu d'une tuile terminée, livré au `sink`.
#[allow(dead_code)] // champs lus par le sink GUI (bin fractall-gui) et les tests
pub struct TileUpdate {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    /// Buffers row-major `w*h` (copies locales, indépendantes de la sortie).
    pub iterations: Vec<u32>,
    pub zs: Vec<Complex64>,
    /// Vide si le path ne calcule pas de distances.
    pub distances: Vec<f64>,
}

/// Grille de tuiles d'une frame + ordre de parcours priorisé.
pub struct TileGrid {
    pub width: usize,
    pub height: usize,
    /// Côté de tuile (les tuiles de bord droit/bas sont tronquées).
    pub tile: usize,
    pub nx: usize,
    pub ny: usize,
    /// Ids de tuiles (`id = by*nx + bx`) triés par distance croissante du
    /// centre de tuile au point de priorité.
    pub order: Vec<usize>,
}

impl TileGrid {
    /// Construit la grille. La taille de tuile est choisie dans {64, 32, 16} :
    /// la plus grande qui donne ≥ 8 tuiles par thread (granularité suffisante
    /// pour le vol de travail — l'ancien chunking visait ~16 chunks/thread).
    pub fn new(width: usize, height: usize, priority_norm: (f64, f64)) -> Self {
        debug_assert!(width > 0 && height > 0);
        let threads = rayon::current_num_threads().max(1);
        let tile = [64usize, 32, 16]
            .into_iter()
            .find(|&t| width.div_ceil(t) * height.div_ceil(t) >= 8 * threads)
            .unwrap_or(16);
        let nx = width.div_ceil(tile).max(1);
        let ny = height.div_ceil(tile).max(1);
        let px = priority_norm.0.clamp(0.0, 1.0) * width as f64;
        let py = priority_norm.1.clamp(0.0, 1.0) * height as f64;
        let dist2 = |id: usize| -> f64 {
            let bx = id % nx;
            let by = id / nx;
            let cx = ((bx * tile) as f64 + (((bx * tile + tile).min(width) - bx * tile) as f64) / 2.0)
                - px;
            let cy = ((by * tile) as f64
                + (((by * tile + tile).min(height) - by * tile) as f64) / 2.0)
                - py;
            cx * cx + cy * cy
        };
        let mut order: Vec<usize> = (0..nx * ny).collect();
        order.sort_by(|&a, &b| dist2(a).total_cmp(&dist2(b)));
        Self {
            width,
            height,
            tile,
            nx,
            ny,
            order,
        }
    }

    /// Rectangle (x0, y0, w, h) de la tuile `id`.
    pub fn rect(&self, id: usize) -> (usize, usize, usize, usize) {
        let bx = id % self.nx;
        let by = id / self.nx;
        let x0 = bx * self.tile;
        let y0 = by * self.tile;
        let w = (x0 + self.tile).min(self.width) - x0;
        let h = (y0 + self.tile).min(self.height) - y0;
        (x0, y0, w, h)
    }

    /// Découpe un buffer row-major `width*height` en segments de lignes
    /// disjoints, groupés par tuile : `out[id][dj]` = ligne `dj` de la tuile
    /// `id` (longueur = largeur de la tuile). Sûr : chaîne de `split_at_mut`.
    pub fn split<'a, T>(&self, buf: &'a mut [T]) -> Vec<Vec<&'a mut [T]>> {
        assert_eq!(buf.len(), self.width * self.height);
        let mut out: Vec<Vec<&'a mut [T]>> = (0..self.nx * self.ny)
            .map(|id| Vec::with_capacity(self.rect(id).3))
            .collect();
        for (j, row) in buf.chunks_mut(self.width).enumerate() {
            let by = j / self.tile;
            let mut rest = row;
            for bx in 0..self.nx {
                let w = ((bx + 1) * self.tile).min(self.width) - bx * self.tile;
                let (seg, r) = rest.split_at_mut(w);
                rest = r;
                out[by * self.nx + bx].push(seg);
            }
        }
        out
    }
}

/// Exécute `render_tile(id, state)` pour chaque tuile dans l'ordre de
/// priorité, sur `min(threads, tuiles)` workers alimentés par une file
/// atomique. `slots[id]` porte l'état tuile-local (segments de sortie),
/// consommé une seule fois. Retourne `false` si `cancel` a été observé
/// (sortie coopérative, tuiles restantes non calculées).
pub fn run_prioritized<S: Send>(
    order: &[usize],
    slots: &[Mutex<Option<S>>],
    cancel: &AtomicBool,
    render_tile: &(dyn Fn(usize, S) + Sync),
) -> bool {
    let next = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let workers = rayon::current_num_threads().max(1).min(order.len().max(1));
    rayon::scope(|s| {
        for _ in 0..workers {
            s.spawn(|_| loop {
                if cancelled.load(Ordering::Relaxed) {
                    return;
                }
                if cancel.load(Ordering::Relaxed) {
                    cancelled.store(true, Ordering::Relaxed);
                    return;
                }
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= order.len() {
                    return;
                }
                let state = slots[order[k]].lock().ok().and_then(|mut g| g.take());
                if let Some(state) = state {
                    render_tile(order[k], state);
                }
            });
        }
    });
    !cancelled.load(Ordering::Relaxed)
}

/// Copie des segments de lignes tuile-locaux vers un buffer contigu `w*h`
/// (payload du `sink`).
pub fn collect_rows<T: Copy>(rows: &[&mut [T]]) -> Vec<T> {
    let mut out = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
    for r in rows {
        out.extend_from_slice(r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_every_pixel_exactly_once() {
        for (w, h) in [(130usize, 70usize), (64, 64), (97, 3), (16, 200), (1, 1)] {
            let grid = TileGrid::new(w, h, (0.5, 0.5));
            let mut seen = vec![0u8; w * h];
            for id in 0..grid.nx * grid.ny {
                let (x0, y0, tw, th) = grid.rect(id);
                assert!(tw > 0 && th > 0, "tuile vide {id} ({w}x{h})");
                for j in y0..y0 + th {
                    for i in x0..x0 + tw {
                        seen[j * w + i] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "couverture non exacte {w}x{h}");
            assert_eq!(grid.order.len(), grid.nx * grid.ny);
            let mut sorted = grid.order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..grid.nx * grid.ny).collect::<Vec<_>>());
        }
    }

    #[test]
    fn order_starts_at_priority_point_and_is_monotone() {
        let (w, h) = (400usize, 300usize);
        for prio in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.8)] {
            let grid = TileGrid::new(w, h, prio);
            let (px, py) = (prio.0 * w as f64, prio.1 * h as f64);
            let d2 = |id: usize| {
                let (x0, y0, tw, th) = grid.rect(id);
                let cx = x0 as f64 + tw as f64 / 2.0 - px;
                let cy = y0 as f64 + th as f64 / 2.0 - py;
                cx * cx + cy * cy
            };
            // La première tuile contient le point de priorité.
            let (x0, y0, tw, th) = grid.rect(grid.order[0]);
            assert!(
                px >= x0 as f64 && px <= (x0 + tw) as f64 && py >= y0 as f64 && py <= (y0 + th) as f64,
                "première tuile {:?} ne contient pas le point ({px}, {py})",
                grid.rect(grid.order[0])
            );
            // Distances non décroissantes le long de l'ordre.
            for pair in grid.order.windows(2) {
                assert!(d2(pair[0]) <= d2(pair[1]) + 1e-9);
            }
        }
    }

    #[test]
    fn split_segments_match_rects() {
        let (w, h) = (130usize, 70usize);
        let grid = TileGrid::new(w, h, (0.5, 0.5));
        let mut buf = vec![usize::MAX; w * h];
        let grids = grid.split(&mut buf);
        for (id, rows) in grids.into_iter().enumerate() {
            let (_, _, tw, th) = grid.rect(id);
            assert_eq!(rows.len(), th);
            for seg in rows {
                assert_eq!(seg.len(), tw);
                for v in seg.iter_mut() {
                    *v = id;
                }
            }
        }
        for j in 0..h {
            for i in 0..w {
                let expected = (j / grid.tile) * grid.nx + i / grid.tile;
                assert_eq!(buf[j * w + i], expected, "pixel ({i},{j})");
            }
        }
    }

    #[test]
    fn run_prioritized_processes_all_tiles_and_respects_cancel() {
        let grid = TileGrid::new(200, 200, (0.0, 0.0));
        let n = grid.nx * grid.ny;
        let done: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        let slots: Vec<Mutex<Option<usize>>> = (0..n).map(|id| Mutex::new(Some(id))).collect();
        let cancel = AtomicBool::new(false);
        let completed = run_prioritized(&grid.order, &slots, &cancel, &|id, state| {
            assert_eq!(id, state);
            done[id].store(true, Ordering::Relaxed);
        });
        assert!(completed);
        assert!(done.iter().all(|f| f.load(Ordering::Relaxed)));

        // Cancel pré-armé : rien ne tourne, retour false.
        let slots2: Vec<Mutex<Option<usize>>> = (0..n).map(|id| Mutex::new(Some(id))).collect();
        let cancel2 = AtomicBool::new(true);
        let completed2 = run_prioritized(&grid.order, &slots2, &cancel2, &|_, _| {
            panic!("aucune tuile ne doit tourner après cancel");
        });
        assert!(!completed2);
    }
}
