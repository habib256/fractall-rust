//! Micro-bench : itération d'orbite GMP z²+c « 2 multiplications » (celle de
//! `orbit.rs::fast_sqr_step`) en série, contre la même itération dont les deux
//! multiplications partent sur deux cœurs (`rayon::join`, ou un assistant
//! dédié synchronisé par attente active). Vérifie l'identité au bit près.
//!
//! Usage : cargo run --release --example bench_orbit_join -- <bits> <iters>

use rug::{Assign, Complex, Float};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[inline]
fn serial_step(z: &mut Complex, c: &Complex, t1: &mut Float, t2: &mut Float) {
    let (re, im) = z.as_mut_real_imag();
    t1.assign(&*re + &*im);
    t2.assign(&*re - &*im);
    *im *= &*re;
    *im <<= 1;
    *im += c.imag();
    re.assign(&*t1 * &*t2);
    *re += c.real();
}

#[inline]
fn join_step(z: &mut Complex, c: &Complex, t1: &mut Float, t2: &mut Float, p: &mut Float) {
    let (re, im) = z.as_mut_real_imag();
    t1.assign(&*re + &*im);
    t2.assign(&*re - &*im);
    {
        let re_ref: &Float = re;
        let (t1, t2): (&Float, &Float) = (t1, t2);
        rayon::join(
            || p.assign(t1 * t2),
            || {
                *im *= re_ref;
                *im <<= 1;
                *im += c.imag();
            },
        );
    }
    std::mem::swap(re, p);
    *re += c.real();
}

/// Assistant dédié : la tâche `im' = 2·im·re + ci` passe par une case
/// (propriété transférée, aucun `unsafe`), synchronisation par attente active.
struct Helper {
    job: Mutex<Option<(Float, Float)>>,
    result: Mutex<Option<(Float, Float)>>,
    state: AtomicU32, // 0 repos, 1 tâche postée, 2 faite, 3 fin
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bits: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3400);
    let iters: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    // Point intérieur borné (orbite qui ne s'évade pas).
    let c = Complex::with_val(bits, (-0.1234567891234, 0.6543219876543));

    let run_serial = || {
        let mut z = Complex::with_val(bits, (0.0, 0.0));
        let (mut t1, mut t2) = (Float::new(bits), Float::new(bits));
        let t = Instant::now();
        for _ in 0..iters {
            serial_step(&mut z, &c, &mut t1, &mut t2);
        }
        (t.elapsed().as_secs_f64(), z)
    };
    let run_join = || {
        let mut z = Complex::with_val(bits, (0.0, 0.0));
        let (mut t1, mut t2, mut p) = (Float::new(bits), Float::new(bits), Float::new(bits));
        let t = Instant::now();
        // Dans le pool : `join` y prend son chemin rapide.
        rayon::scope(|_| {
            for _ in 0..iters {
                join_step(&mut z, &c, &mut t1, &mut t2, &mut p);
            }
        });
        (t.elapsed().as_secs_f64(), z)
    };
    let run_helper = || {
        let mut z = Complex::with_val(bits, (0.0, 0.0));
        let (mut t1, mut t2, mut p) = (Float::new(bits), Float::new(bits), Float::new(bits));
        let h = Helper {
            job: Mutex::new(None),
            result: Mutex::new(None),
            state: AtomicU32::new(0),
        };
        let ci = c.imag().clone();
        let t = Instant::now();
        let mut stolen = 0u32;
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut spins = 0u32;
                loop {
                    match h.state.load(Ordering::Acquire) {
                        3 => return,
                        1 => {
                            let job = h.job.lock().unwrap().take();
                            if let Some((mut im, re)) = job {
                                im *= &re;
                                im <<= 1;
                                im += &ci;
                                *h.result.lock().unwrap() = Some((im, re));
                                h.state.store(2, Ordering::Release);
                                spins = 0;
                                continue;
                            }
                        }
                        _ => {}
                    }
                    spins += 1;
                    if spins < 50_000 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            });
            for _ in 0..iters {
                let (mut re, mut im) = std::mem::replace(&mut z, Complex::new(2)).into_real_imag();
                t1.assign(&re + &im);
                t2.assign(&re - &im);
                *h.job.lock().unwrap() = Some((im, re));
                h.state.store(1, Ordering::Release);
                p.assign(&t1 * &t2);
                // L'assistant n'a pas pris la tâche : on la reprend (jamais
                // plus lent que la série, à peu près). Sinon attente active.
                let back = h.job.lock().unwrap().take();
                if let Some((mut i2, r2)) = back {
                    i2 *= &r2;
                    i2 <<= 1;
                    i2 += &ci;
                    im = i2;
                    re = r2;
                    stolen += 1;
                } else {
                    loop {
                        if h.state.load(Ordering::Acquire) == 2 {
                            let (i2, r2) = h.result.lock().unwrap().take().unwrap();
                            im = i2;
                            re = r2;
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                h.state.store(0, Ordering::Release);
                std::mem::swap(&mut re, &mut p);
                re += c.real();
                z = Complex::from((re, im));
            }
            h.state.store(3, Ordering::Release);
        });
        if std::env::var_os("BENCH_VERBOSE").is_some() {
            eprintln!("  reprises par le fil principal : {stolen}/{iters}");
        }
        (t.elapsed().as_secs_f64(), z)
    };

    let mut best = [f64::INFINITY; 3];
    let mut zs: Vec<Complex> = Vec::new();
    for _ in 0..7 {
        let (a, za) = run_serial();
        let (b, zb) = run_join();
        let (h, zh) = run_helper();
        best[0] = best[0].min(a);
        best[1] = best[1].min(b);
        best[2] = best[2].min(h);
        zs = vec![za, zb, zh];
    }
    let same = zs[0] == zs[1] && zs[0] == zs[2];
    let per = |t: f64| t / iters as f64 * 1e9;
    println!(
        "bits={bits} iters={iters}  série {:.0} ns/it  join {:.0} ns/it (×{:.2})  assistant {:.0} ns/it (×{:.2})  identiques={same}",
        per(best[0]),
        per(best[1]),
        best[0] / best[1],
        per(best[2]),
        best[0] / best[2],
    );
}
