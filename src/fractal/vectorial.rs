//! Fractales vectorielles (Von Koch et Dragon).
//!
//! Ces fractales sont rendues par dessin récursif de lignes,
//! contrairement aux fractales escape-time.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::fractal::FractalParams;

/// Point 2D pour le dessin vectoriel.
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Rotation du point autour d'un centre avec un angle theta (radians).
    fn rotate(self, theta: f64, center: Point) -> Self {
        let temp_x = self.x - center.x;
        let temp_y = self.y - center.y;

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        Point {
            x: temp_x * cos_t - temp_y * sin_t + center.x,
            y: temp_x * sin_t + temp_y * cos_t + center.y,
        }
    }
}

/// Segment de ligne à dessiner.
struct Line {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

/// Collecte récursive des lignes pour le flocon de Von Koch.
fn von_koch_collect(a: Point, b: Point, n: u32, lines: &mut Vec<Line>) {
    if n == 0 {
        lines.push(Line {
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
        });
    } else {
        // C = A + 1/3 * AB
        let c = Point::new(a.x + (b.x - a.x) / 3.0, a.y + (b.y - a.y) / 3.0);
        // D = C + 1/3 * AB
        let d = Point::new(c.x + (b.x - a.x) / 3.0, c.y + (b.y - a.y) / 3.0);
        // E = rotation de B autour de D, angle 2π/3
        let e = b.rotate(2.0 * std::f64::consts::PI / 3.0, d);

        von_koch_collect(a, c, n - 1, lines);
        von_koch_collect(c, e, n - 1, lines);
        von_koch_collect(e, d, n - 1, lines);
        von_koch_collect(d, b, n - 1, lines);
    }
}

/// Collecte récursive des lignes pour la courbe du Dragon.
fn dragon_collect(a: Point, b: Point, n: u32, lines: &mut Vec<Line>) {
    if n == 0 {
        lines.push(Line {
            x1: a.x,
            y1: a.y,
            x2: b.x,
            y2: b.y,
        });
    } else {
        // Centre du segment AB
        let c = Point::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
        // D = rotation de B autour de C, angle π/2
        let d = b.rotate(std::f64::consts::PI / 2.0, c);

        dragon_collect(a, d, n - 1, lines);
        dragon_collect(b, d, n - 1, lines);
    }
}

/// Dessine une ligne anti-aliasée dans le buffer.
fn draw_line(buffer: &mut [u8], width: u32, height: u32, line: &Line) {
    let w = width as i32;
    let h = height as i32;

    // Algorithme de Bresenham avec anti-aliasing simplifié
    let mut x0 = line.x1 as i32;
    let mut y0 = line.y1 as i32;
    let x1 = line.x2 as i32;
    let y1 = line.y2 as i32;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        // Dessiner le pixel si dans les limites
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            let idx = ((y0 as u32) * width + x0 as u32) as usize;
            if idx < buffer.len() {
                buffer[idx] = 255; // Pixel blanc
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Rendu du flocon de Von Koch.
pub fn render_von_koch(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let iterations = vec![0u32; width * height];
    let zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let w = params.width as f64;
    let h = params.height as f64;

    // Points du triangle équilatéral
    let a = Point::new(0.5 * w, 0.02 * h);
    let b = Point::new(0.2 * w, 0.71 * h);
    let c = Point::new(0.8 * w, 0.71 * h);

    // Collecter toutes les lignes
    let depth = params.iteration_max.min(8); // Max 8 pour éviter explosion
    let mut lines = Vec::new();
    von_koch_collect(a, b, depth, &mut lines);
    von_koch_collect(c, a, depth, &mut lines);
    von_koch_collect(b, c, depth, &mut lines);

    // Dessiner dans un buffer temporaire
    let mut line_buffer = vec![0u8; width * height];
    for line in &lines {
        draw_line(&mut line_buffer, params.width, params.height, line);
    }

    // Convertir en format iterations/zs pour la colorisation
    let iterations: Vec<u32> = line_buffer
        .par_iter()
        .map(|&v| if v > 0 { params.iteration_max } else { 0 })
        .collect();

    let zs: Vec<Complex64> = line_buffer
        .par_iter()
        .map(|&v| {
            if v > 0 {
                Complex64::new(2.0, 0.0) // Valeur pour colorisation
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    (iterations, zs)
}

/// Rendu de la courbe du Dragon.
pub fn render_dragon(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let iterations = vec![0u32; width * height];
    let zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let w = params.width as f64;
    let h = params.height as f64;

    // Points de départ pour le dragon
    let d = Point::new(0.338 * w, 0.208 * h);
    let e = Point::new(0.838 * w, h - 0.31 * h);

    // Collecter toutes les lignes
    let depth = params.iteration_max.min(20); // Max 20 pour éviter explosion
    let mut lines = Vec::new();
    dragon_collect(d, e, depth, &mut lines);

    // Dessiner dans un buffer temporaire
    let mut line_buffer = vec![0u8; width * height];
    for line in &lines {
        draw_line(&mut line_buffer, params.width, params.height, line);
    }

    // Convertir en format iterations/zs pour la colorisation
    let iterations: Vec<u32> = line_buffer
        .par_iter()
        .map(|&v| if v > 0 { params.iteration_max } else { 0 })
        .collect();

    let zs: Vec<Complex64> = line_buffer
        .par_iter()
        .map(|&v| {
            if v > 0 {
                Complex64::new(2.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    (iterations, zs)
}
