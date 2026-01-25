use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

use crate::fractal::{FractalParams, FractalResult, FractalType};
use crate::fractal::iterations::iterate_point;
use crate::fractal::gmp::{ComplexF, GmpParams, iterate_point_gmp};
use crate::fractal::{render_lyapunov, render_von_koch, render_dragon, render_buddhabrot, render_nebulabrot};

/// Calcule la matrice d'itérations et la matrice des valeurs finales de z
/// pour une fractale escape-time (ou algorithme spécial).
///
/// Retourne un tuple (iterations, zs) où :
/// - `iterations.len() == width * height`
/// - `zs.len() == width * height`
///
/// Le calcul est parallélisé sur plusieurs cœurs CPU avec rayon.
pub fn render_escape_time(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    // Dispatch vers les algorithmes spéciaux
    match params.fractal_type {
        FractalType::VonKoch => return render_von_koch(params),
        FractalType::Dragon => return render_dragon(params),
        FractalType::Buddhabrot => return render_buddhabrot(params),
        FractalType::Lyapunov => return render_lyapunov(params),
        FractalType::Nebulabrot => return render_nebulabrot(params),
        _ => {}
    }

    if params.use_gmp {
        return render_escape_time_gmp(params);
    }
    render_escape_time_f64(params)
}

fn render_escape_time_f64(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    // Même mapping que Fractal_CalculateMatrix en C :
    // xg = ((xmax - xmin) / xpixel) * i + xmin
    // yg = ((ymax - ymin) / ypixel) * j + ymin
    let x_range = params.xmax - params.xmin;
    let y_range = params.ymax - params.ymin;

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let x_step = x_range / params.width as f64;
    let y_step = y_range / params.height as f64;

    // Parallélisation par lignes avec rayon (beaucoup plus élégant que std::thread)
    iterations
        .par_chunks_mut(width)
        .zip(zs.par_chunks_mut(width))
        .enumerate()
        .for_each(|(j, (iter_row, z_row))| {
            let yg = y_step * j as f64 + params.ymin;
            for (i, (iter, z)) in iter_row.iter_mut().zip(z_row.iter_mut()).enumerate() {
                let xg = x_step * i as f64 + params.xmin;
                let z_pixel = Complex64::new(xg, yg);
                let FractalResult { iteration, z: z_final } = iterate_point(params, z_pixel);
                *iter = iteration;
                *z = z_final;
            }
        });

    (iterations, zs)
}

fn render_escape_time_gmp(params: &FractalParams) -> (Vec<u32>, Vec<Complex64>) {
    let width = params.width as usize;
    let height = params.height as usize;
    let mut iterations = vec![0u32; width * height];
    let mut zs = vec![Complex64::new(0.0, 0.0); width * height];

    if width == 0 || height == 0 {
        return (iterations, zs);
    }

    let gmp = GmpParams::from_params(params);
    let prec = gmp.prec;

    let xmin = Float::with_val(prec, params.xmin);
    let xmax = Float::with_val(prec, params.xmax);
    let ymin = Float::with_val(prec, params.ymin);
    let ymax = Float::with_val(prec, params.ymax);

    let mut x_range = xmax.clone();
    x_range -= &xmin;
    let mut y_range = ymax.clone();
    y_range -= &ymin;
    let width_f = Float::with_val(prec, params.width);
    let height_f = Float::with_val(prec, params.height);
    let mut x_step = x_range;
    x_step /= &width_f;
    let mut y_step = y_range;
    y_step /= &height_f;

    let mut yg = ymin.clone();
    for j in 0..height {
        let mut xg = xmin.clone();
        for i in 0..width {
            let idx = j * width + i;
            let z_pixel = ComplexF::new(xg.clone(), yg.clone());
            let (iter, z_final) = iterate_point_gmp(&gmp, &z_pixel);
            iterations[idx] = iter;
            zs[idx] = z_final.to_complex64();
            xg += &x_step;
        }
        yg += &y_step;
    }

    (iterations, zs)
}

