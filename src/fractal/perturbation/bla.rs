use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalType};

#[derive(Clone, Copy, Debug)]
pub struct BlaNode {
    pub a: Complex64,
    pub b: Complex64,
    pub c: Complex64,
    pub validity_radius: f64,
}

#[derive(Clone, Debug)]
pub struct BlaTable {
    pub levels: Vec<Vec<BlaNode>>,
}

impl BlaTable {
    pub fn empty() -> Self {
        Self { levels: Vec::new() }
    }
}

pub fn build_bla_table(ref_orbit: &[Complex64], params: &FractalParams) -> BlaTable {
    let supports_bla = matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia);
    if !supports_bla {
        return BlaTable::empty();
    }

    let base_len = ref_orbit.len().saturating_sub(1);
    if base_len == 0 {
        return BlaTable::empty();
    }

    let mut levels: Vec<Vec<BlaNode>> = Vec::new();
    let mut level0 = Vec::with_capacity(base_len);
    let base_threshold = params.bla_threshold.max(1e-16);
    for &z in ref_orbit.iter().take(base_len) {
        let a = z * 2.0;
        let a_norm = a.norm();
        let mut validity = base_threshold / (1.0 + a_norm);
        if !validity.is_finite() {
            validity = base_threshold;
        }
        validity = validity.min(base_threshold);
        level0.push(BlaNode {
            a,
            b: if params.fractal_type == FractalType::Julia {
                Complex64::new(0.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            },
            c: Complex64::new(1.0, 0.0),
            validity_radius: validity,
        });
    }
    levels.push(level0);

    let max_level = 16usize;
    for level in 1..=max_level {
        let step = 1usize << (level - 1);
        let prev = &levels[level - 1];
        if prev.len() <= step {
            break;
        }
        let mut current = Vec::with_capacity(prev.len() - step);
        for i in 0..(prev.len() - step) {
            let node1 = prev[i];
            let node2 = prev[i + step];
            let a_new = node2.a * node1.a;
            let b_new = node2.a * node1.b + node2.b;
            let c_new = node2.a * node1.c + node2.c * (node1.a * node1.a);
            let a_norm = node1.a.norm();
            let scaled = node2.validity_radius / (1.0 + a_norm);
            current.push(BlaNode {
                a: a_new,
                b: b_new,
                c: c_new,
                validity_radius: node1.validity_radius.min(scaled),
            });
        }
        levels.push(current);
    }

    BlaTable { levels }
}
