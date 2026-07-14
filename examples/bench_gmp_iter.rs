//! Micro-bench : coût brut d'une itération z²+c GMP (mpc via rug) à précision
//! donnée. Usage : cargo run --release --example bench_gmp_iter -- <bits> <iters>
//! Sert à établir le plancher matériel pour les cas extrêmes (e52465 à 174k bits).

use rug::{Complex, Float};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bits: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(174_319);
    let iters: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    // c proche de -2 (cas e52465), z part de 0.
    let c = Complex::with_val(bits, (-1.9999999999, 1e-30));
    let mut z = Complex::with_val(bits, (0.0, 0.0));
    let escape = Float::with_val(64, 625.0);

    let t = Instant::now();
    let mut n = 0u32;
    while n < iters {
        z.square_mut();
        z += &c;
        // check d'évasion léger (norme via f64 approx, comme le moteur)
        let re = z.real().to_f64();
        let im = z.imag().to_f64();
        if re * re + im * im > escape.to_f64() {
            break;
        }
        n += 1;
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "bits={} iters={} total={:.3}s per_iter={:.1}µs",
        bits,
        n,
        dt,
        dt / (n.max(1) as f64) * 1e6
    );
}
