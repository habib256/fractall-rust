//! Colorisation par mode (ms par image 2560×1440) : où part le temps.
use fractall_cli::fractal::{default_params_for_type, FractalType, OutColoringMode};
use fractall_cli::io::png::colorize_to_rgb_with_extras;
use fractall_cli::render::render_escape_time;
use std::time::Instant;

fn main() {
    let mut p = default_params_for_type(FractalType::Mandelbrot, 2560, 1440);
    p.iteration_max = 1000;
    let out = render_escape_time(&p);
    let (it, zs) = (out.iterations, out.zs);
    let interior = it.iter().filter(|&&i| i >= 1000).count() as f64 / it.len() as f64;
    println!("pixels intérieurs : {:.1} %", 100.0 * interior);
    for mode in [OutColoringMode::Iter, OutColoringMode::Smooth, OutColoringMode::Potential] {
        p.color.out_coloring_mode = mode;
        let mut best = f64::INFINITY;
        for k in 0..15 {
            p.color.color_offset = k as f64 * 0.01;
            let t = Instant::now();
            std::hint::black_box(colorize_to_rgb_with_extras(&p, &it, &zs, &[], &[]));
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{mode:?} : {:.2} ms (min de 15)", best * 1e3);
    }
    // Plancher : même parcours parallèle, écriture d'une constante.
    let mut best = f64::INFINITY;
    for _ in 0..15 {
        use rayon::prelude::*;
        let t = Instant::now();
        let mut buf = vec![0u8; 2560 * 1440 * 3];
        buf.par_chunks_mut(2560 * 3).enumerate().for_each(|(y, row)| {
            for x in 0..2560 {
                let i = it[y * 2560 + x];
                row[x * 3] = i as u8;
            }
        });
        std::hint::black_box(&buf);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("plancher (alloc + parcours) : {:.2} ms", best * 1e3);
}
