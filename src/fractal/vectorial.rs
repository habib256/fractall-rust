//! Fractales vectorielles (Von Koch et Dragon).

use num_complex::Complex64;

use crate::fractal::{jitter::sample_offset, FractalParams};

const MAX_DRAGON_DEPTH: u32 = 20;

#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn midpoint(self, other: Point) -> Self {
        Self::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }

    fn rotate(self, theta: f64, center: Point) -> Self {
        let x = self.x - center.x;
        let y = self.y - center.y;
        Point {
            x: x * theta.cos() - y * theta.sin() + center.x,
            y: x * theta.sin() + y * theta.cos() + center.y,
        }
    }
}

/// Les feuilles du tracé écrivent directement dans les deux canaux de sortie.
/// Les croisements réécrivent les mêmes valeurs.
struct Raster {
    width: usize,
    height: usize,
    ink: u32,
    radii: [f64; MAX_DRAGON_DEPTH as usize + 1],
    subpixel_depth: u32,
    iterations: Vec<u32>,
    zs: Vec<Complex64>,
}

impl Raster {
    fn new(params: &FractalParams) -> Self {
        let width = params.width as usize;
        let height = params.height as usize;
        Self {
            width,
            height,
            ink: params.iteration_max,
            radii: [0.0; MAX_DRAGON_DEPTH as usize + 1],
            subpixel_depth: 0,
            iterations: vec![0; width * height],
            zs: vec![Complex64::new(0.0, 0.0); width * height],
        }
    }

    #[inline]
    fn pixel(&mut self, x: i64, y: i64) {
        if (x as u64) < self.width as u64 && (y as u64) < self.height as u64 {
            let idx = y as usize * self.width + x as usize;
            self.iterations[idx] = self.ink;
            self.zs[idx] = Complex64::new(2.0, 0.0);
        }
    }

    fn line(&mut self, a: Point, b: Point) {
        // Bresenham, avec les mêmes règles d'arrondi aux deux extrémités.
        let (mut x, mut y) = (a.x as i64, a.y as i64);
        let (end_x, end_y) = (b.x as i64, b.y as i64);
        let dx = (end_x - x).abs();
        let dy = -(end_y - y).abs();
        let sx = if x < end_x { 1 } else { -1 };
        let sy = if y < end_y { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.pixel(x, y);
            if x == end_x && y == end_y {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Rayon du disque centré sur le milieu de chaque segment de longueur L.
    /// À profondeur 0 ou 1, tous les sommets sont à distance ≤ L/2 dans les
    /// deux constructions. Ensuite, la distance maximale des milieux enfants
    /// au milieu parent vaut L/3 pour Koch, L/(2√2) pour Dragon. Ajouter ce
    /// déplacement au rayon enfant borne donc tout le sous-arbre par induction.
    fn bound_subtrees(
        &mut self,
        a: Point,
        b: Point,
        depth: u32,
        child_scale: f64,
        child_midpoint: f64,
    ) {
        let root_length = (a.x - b.x).hypot(a.y - b.y);
        let magnitude = a.x.abs().max(a.y.abs()).max(b.x.abs()).max(b.y.abs()) + root_length;
        // Marge d'arrondi sur les coordonnées absolues pour les 20 subdivisions
        // au plus. Au voisinage d'une frontière de pixel, on descend d'un niveau.
        let margin = 1024.0 * f64::EPSILON * magnitude;
        let mut length = root_length * child_scale.powi(depth as i32);
        self.subpixel_depth = 0;
        let mut radius = 0.0;
        for level in 0..=depth {
            radius = if level <= 1 {
                length * 0.5
            } else {
                radius + child_midpoint * length
            };
            if radius + margin >= 0.5 {
                break;
            }
            self.radii[level as usize] = radius + margin;
            self.subpixel_depth = level;
            length /= child_scale;
        }
    }

    #[inline]
    fn draw_single_pixel(&mut self, center: Point, radius: f64) -> bool {
        let x = (center.x - radius) as i64;
        let y = (center.y - radius) as i64;
        if x == (center.x + radius) as i64 && y == (center.y + radius) as i64 {
            self.pixel(x, y);
            return true;
        }
        false
    }

    fn von_koch<const PRUNE: bool>(&mut self, a: Point, b: Point, depth: u32) {
        if depth == 0 {
            self.line(a, b);
            return;
        }

        if PRUNE && depth <= self.subpixel_depth {
            let center = a.midpoint(b);
            if self.draw_single_pixel(center, self.radii[depth as usize]) {
                return;
            }
        }
        let dx = (b.x - a.x) / 3.0;
        let dy = (b.y - a.y) / 3.0;
        let c = Point::new(a.x + dx, a.y + dy);
        let d = Point::new(c.x + dx, c.y + dy);
        let e = b.rotate(2.0 * std::f64::consts::PI / 3.0, d);

        self.von_koch::<PRUNE>(a, c, depth - 1);
        self.von_koch::<PRUNE>(c, e, depth - 1);
        self.von_koch::<PRUNE>(e, d, depth - 1);
        self.von_koch::<PRUNE>(d, b, depth - 1);
    }

    fn dragon<const PRUNE: bool>(&mut self, a: Point, b: Point, depth: u32) {
        if depth == 0 {
            self.line(a, b);
            return;
        }

        let center = a.midpoint(b);
        if PRUNE
            && depth <= self.subpixel_depth
            && self.draw_single_pixel(center, self.radii[depth as usize])
        {
            return;
        }
        let d = b.rotate(std::f64::consts::FRAC_PI_2, center);
        self.dragon::<PRUNE>(a, d, depth - 1);
        self.dragon::<PRUNE>(b, d, depth - 1);
    }
}

/// Une passe AA vectorielle translate toute la courbe face à la grille de
/// pixels. Les segments gardent ainsi leurs sommets communs à chaque passe.
fn sample_origin(params: &FractalParams) -> Point {
    let [dx, dy] = match params.sampling.aa_jitter {
        Some((k, scale)) => {
            let (x, y) = sample_offset(k);
            [x * scale, y * scale]
        }
        None => params.sampling.aa_subpixel_offset,
    };
    Point::new(-dx, -dy)
}

/// Rendu du flocon de Von Koch.
pub fn render_von_koch(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let mut raster = Raster::new(params);
    if !raster.iterations.is_empty() {
        let (w, h) = (params.width as f64, params.height as f64);
        let origin = sample_origin(params);
        let a = Point::new(0.5 * w + origin.x, 0.02 * h + origin.y);
        let b = Point::new(0.2 * w + origin.x, 0.71 * h + origin.y);
        let c = Point::new(0.8 * w + origin.x, 0.71 * h + origin.y);
        let depth = params.iteration_max.min(8);
        for (a, b) in [(a, b), (c, a), (b, c)] {
            raster.bound_subtrees(a, b, depth, 1.0 / 3.0, 1.0 / 3.0);
            if raster.subpixel_depth > 0 {
                raster.von_koch::<true>(a, b, depth);
            } else {
                raster.von_koch::<false>(a, b, depth);
            }
        }
    }
    (raster.iterations, raster.zs)
}

/// Rendu de la courbe du Dragon.
pub fn render_dragon(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let mut raster = Raster::new(params);
    if !raster.iterations.is_empty() {
        let (w, h) = (params.width as f64, params.height as f64);
        let origin = sample_origin(params);
        let a = Point::new(0.338 * w + origin.x, 0.208 * h + origin.y);
        let b = Point::new(0.838 * w + origin.x, (h - 0.31 * h) + origin.y);
        let depth = params.iteration_max.min(MAX_DRAGON_DEPTH);
        let child_scale = std::f64::consts::FRAC_1_SQRT_2;
        raster.bound_subtrees(a, b, depth, child_scale, 0.5 * child_scale);
        // Tester deux feuilles à la dernière bifurcation seule coûte aussi
        // cher que les tracer. Amortir ce test sur au moins deux niveaux.
        if raster.subpixel_depth >= 2 {
            raster.dragon::<true>(a, b, depth);
        } else {
            raster.dragon::<false>(a, b, depth);
        }
    }
    (raster.iterations, raster.zs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    #[test]
    fn subpixel_pruning_matches_every_leaf() {
        for offset in [0.0, 0.5, -0.125] {
            let a = Point::new(-2.0 + offset, 4.0 - offset);
            let b = Point::new(23.0 + offset, 18.0 - offset);
            for kind in [FractalType::VonKoch, FractalType::Dragon] {
                let params = default_params_for_type(kind, 32, 23);
                let mut full = Raster::new(&params);
                let mut pruned = Raster::new(&params);
                match kind {
                    FractalType::VonKoch => {
                        pruned.bound_subtrees(a, b, 7, 1.0 / 3.0, 1.0 / 3.0);
                        full.von_koch::<false>(a, b, 7);
                        pruned.von_koch::<true>(a, b, 7);
                    }
                    FractalType::Dragon => {
                        let scale = std::f64::consts::FRAC_1_SQRT_2;
                        pruned.bound_subtrees(a, b, 16, scale, 0.5 * scale);
                        full.dragon::<false>(a, b, 16);
                        pruned.dragon::<true>(a, b, 16);
                    }
                    _ => unreachable!(),
                }
                assert!(pruned.subpixel_depth >= 2);
                assert_eq!(pruned.iterations, full.iterations);
                assert_eq!(pruned.zs, full.zs);
            }
        }
    }

    #[test]
    fn vector_samples_move_curves_and_respect_zero_scale() {
        for (kind, render) in [
            (
                FractalType::VonKoch,
                render_von_koch as fn(&FractalParams) -> _,
            ),
            (
                FractalType::Dragon,
                render_dragon as fn(&FractalParams) -> _,
            ),
        ] {
            let mut p = default_params_for_type(kind, 140, 105);
            p.iteration_max = 4;
            let base = render(&p);
            p.sampling.aa_jitter = Some((5, 0.0));
            assert_eq!(render(&p), base);
            p.sampling.aa_jitter = Some((5, 1.0));
            let sample = render(&p);
            assert_ne!(sample, base);
            p.sampling.aa_jitter = None;
            let (dx, dy) = sample_offset(5);
            p.sampling.aa_subpixel_offset = [dx, dy];
            assert_eq!(render(&p), sample);
        }
    }
}
