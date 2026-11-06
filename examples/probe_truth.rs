//! Sonde de VÉRACITÉ du rendu de production (Julia / Burning Ship / Tricorn,
//! et toute famille à juge GMP) : descente de zoom HP sur des cellules de bord
//! à fort nombre d'itérations, puis, à chaque profondeur cible, rendu de
//! production (Auto) vs GMP pur à la précision moteur P — et, s'il y a désaccord,
//! re-jugement à 2P pour écarter les pixels dont la vérité n'est pas convergée.
//!
//! Usage :
//!   cargo run --release --example probe_truth -- <type> <seed_re> <seed_im> \
//!       <rng> <iters> <size> <zooms,…>
//! Sortie : une ligne `RESULT …` par scène (centre HP inclus, rejouable par
//! `fractall-quality compare`).

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use fractall_cli::fractal::{default_params_for_type, AlgorithmMode, FractalParams, FractalType, ViewHp};
use fractall_cli::render::{render_request, RenderRequest};
use num_complex::Complex64;
use rug::Float;

const PREC: u32 = 512;

fn type_from_id(id: u8) -> FractalType {
    match id {
        3 => FractalType::Mandelbrot,
        4 => FractalType::Julia,
        13 => FractalType::BurningShip,
        14 => FractalType::Tricorn,
        _ => panic!("type {id} non géré"),
    }
}

fn params_at(ty: FractalType, seed: Complex64, cx: &Float, cy: &Float, zoom: &str, size: u32, iters: u32) -> FractalParams {
    let mut p = default_params_for_type(ty, size, size);
    p.seed = seed;
    p.iteration_max = iters;
    let sx = cx.to_string_radix(10, None);
    let sy = cy.to_string_radix(10, None);
    p.center_x_hp = Some(sx.clone());
    p.center_y_hp = Some(sy.clone());
    let v = ViewHp::from_center_and_zoom(&sx, &sy, zoom, size, size, 1024).expect("vue");
    v.write_to_params(&mut p);
    p
}

fn render(p: &FractalParams) -> (Vec<u32>, f64) {
    let cancel = Arc::new(AtomicBool::new(false));
    let t = Instant::now();
    let out = render_request(RenderRequest::new(p, &cancel), &mut None).expect("rendu");
    (out.iterations, t.elapsed().as_secs_f64())
}

fn auto_bits(zoom_log10: f64, height: u32) -> u32 {
    let b = (24.0 + (zoom_log10 * std::f64::consts::LOG2_10 + (height as f64).log2()).floor()).max(24.0) as u32;
    (b + 64).max(128)
}

/// xorshift déterministe.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn load_or<F: FnOnce() -> Vec<u32>>(key: &str, f: F) -> Vec<u32> {
    let Some(dir) = std::env::var_os("PROBE_CACHE") else { return f() };
    let path = std::path::Path::new(&dir).join(key);
    if let Ok(b) = std::fs::read(&path) {
        return b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    }
    let v = f();
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let _ = std::fs::write(&path, b);
    v
}

fn judge(ty: FractalType, seed: Complex64, sx: &str, sy: &str, z_str: &str, iters: u32, size: u32) {
    let cx = Float::with_val(PREC, Float::parse(sx).unwrap());
    let cy = Float::with_val(PREC, Float::parse(sy).unwrap());
    let target = z_str.parse::<f64>().unwrap().log10();
    let mut p = params_at(ty, seed, &cx, &cy, z_str, size, iters);
    match std::env::var("PROBE_MODE").as_deref() {
        Ok("f64") => p.engine.algorithm_mode = AlgorithmMode::StandardF64,
        Ok("pert") => p.engine.algorithm_mode = AlgorithmMode::Perturbation,
        _ => {}
    }
    if let Ok(v) = std::env::var("PROBE_SERIES") {
        p.perturbation.series_order = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("PROBE_BLA_SCALE") {
        p.perturbation.bla_validity_scale = v.parse().unwrap();
    }
    let (prod, t_prod) = render(&p);
    if std::env::var_os("PROBE_NOJUDGE").is_some() {
        println!("PROD t_prod={t_prod:.4}");
        return;
    }
    p.engine.algorithm_mode = AlgorithmMode::ReferenceGmp;
    p.engine.use_gmp = true;
    let bits = auto_bits(target, size).max(p.engine.precision_bits);
    let mut h: u64 = 1469598103934665603;
    for b in format!("{ty:?}|{}|{}|{sx}|{sy}|{z_str}|{iters}|{size}", seed.re, seed.im).bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    let t1 = Instant::now();
    let g1 = load_or(&format!("{h:016x}.g1"), || render(&p).0);
    let t_g1 = t1.elapsed().as_secs_f64();
    let diff1: Vec<usize> = (0..prod.len()).filter(|&k| prod[k] != g1[k]).collect();
    if std::env::var_os("PROBE_DIFFS").is_some() {
        // Histogramme (production, GMP) des pixels en désaccord.
        let mut h = std::collections::BTreeMap::new();
        for &k in &diff1 {
            *h.entry((prod[k], g1[k])).or_insert(0u32) += 1;
        }
        let mut v: Vec<_> = h.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for ((a, b), n) in v.iter().take(8) {
            println!("DIFF prod={a} gmp={b} n={n}");
        }
        for &k in diff1.iter().take(3) {
            println!("DIFFPX i={} j={} prod={} gmp={}", k as u32 % size, k as u32 / size, prod[k], g1[k]);
        }
        println!("SPAN {} {}", p.span_x_hp.as_deref().unwrap_or("?"), p.span_y_hp.as_deref().unwrap_or("?"));
    }
    let (mut nconv, mut wrong, mut maxd) = (0usize, 0usize, 0u32);
    let mut t_g2 = 0.0;
    if !diff1.is_empty() {
        p.engine.precision_bits = bits * 2;
        let t2 = Instant::now();
        let g2 = load_or(&format!("{h:016x}.g2"), || render(&p).0);
        t_g2 = t2.elapsed().as_secs_f64();
        nconv = (0..prod.len()).filter(|&k| g1[k] != g2[k]).count();
        for k in 0..prod.len() {
            if g1[k] == g2[k] && prod[k] != g2[k] {
                wrong += 1;
                maxd = maxd.max(prod[k].abs_diff(g2[k]));
            }
        }
    }
    let fin = prod.iter().filter(|&&v| v < iters).count();
    let maxit = prod.iter().filter(|&&v| v < iters).max().copied().unwrap_or(0);
    println!(
        "RESULT type={:?} seed={},{} zoom={} prod_ne_gmpP={} gmp_unconverged={} wrong_vs_converged={} max_diff={} escaped={} max_iter={} bits={} t_prod={:.3} t_gmp={:.1}+{:.1}",
        ty, seed.re, seed.im, z_str,
        diff1.len(), nconv, wrong, maxd, fin, maxit, bits, t_prod, t_g1, t_g2
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "judge" {
        // judge <type> <seed_re> <seed_im> <cx> <cy> <zoom> <iters> <size>
        let ty = type_from_id(a[2].parse().unwrap());
        let seed = Complex64::new(a[3].parse().unwrap(), a[4].parse().unwrap());
        judge(ty, seed, &a[5], &a[6], &a[7], a[8].parse().unwrap(), a[9].parse().unwrap());
        return;
    }
    let ty = type_from_id(a[1].parse().unwrap());
    let seed = Complex64::new(a[2].parse().unwrap(), a[3].parse().unwrap());
    let mut rng = Rng(a[4].parse::<u64>().unwrap().wrapping_mul(0x9E3779B97F4A7C15) | 1);
    let iters: u32 = a[5].parse().unwrap();
    let size: u32 = a[6].parse().unwrap();
    let zooms: Vec<f64> = a[7].split(',').map(|s| s.parse().unwrap()).collect();
    let fold = matches!(ty, FractalType::BurningShip);

    let base = default_params_for_type(ty, 64, 64);
    let smooth = std::env::var_os("PROBE_SMOOTH").is_some();
    let mut cx = Float::with_val(PREC, base.center_x);
    let mut cy = Float::with_val(PREC, base.center_y);
    let mut zoom_l = 0.0f64; // log10(zoom)
    if let Ok(st) = std::env::var("PROBE_START") {
        // "cx,cy,log10zoom" : départ de la descente.
        let v: Vec<&str> = st.split(',').collect();
        cx = Float::with_val(PREC, Float::parse(v[0]).unwrap());
        cy = Float::with_val(PREC, Float::parse(v[1]).unwrap());
        zoom_l = v[2].parse().unwrap();
    }
    let grid = 24u32;
    let mut zi = 0usize;
    while zi < zooms.len() {
        let target = zooms[zi].log10();
        // Descente ×8 jusqu'à la profondeur cible.
        while zoom_l < target - 1e-9 {
            let z_str = format!("{:e}", 10f64.powf(zoom_l));
            let p = params_at(ty, seed, &cx, &cy, &z_str, grid, iters);
            let (it, _) = render(&p);
            let g = grid as usize;
            let mut cands: Vec<(f64, usize, usize)> = Vec::new();
            for j in 1..g - 1 {
                for i in 1..g - 1 {
                    let mut fin = 0u64;
                    let mut fin2 = 0f64;
                    let mut nfin = 0u32;
                    let mut mx = 0u32;
                    let mut mn = u32::MAX;
                    for (jj, ii) in (j - 1..=j + 1).flat_map(|jj| (i - 1..=i + 1).map(move |ii| (jj, ii))) {
                        let v = it[jj * g + ii];
                        mx = mx.max(v);
                        mn = mn.min(v);
                        if v < iters {
                            fin += v as u64;
                            fin2 += (v as f64) * (v as f64);
                            nfin += 1;
                        }
                    }
                    if nfin == 0 || mx == mn {
                        continue;
                    }
                    let dx = (i as f64 + 0.5) / grid as f64 - 0.5;
                    let dy = (j as f64 + 0.5) / grid as f64 - 0.5;
                    if fold && zoom_l < 1.5 {
                        let px = cx.to_f64() + dx * p.span_x;
                        let py = cy.to_f64() + dy * p.span_y;
                        if px.abs() < 0.05 || py.abs() < 0.02 {
                            continue;
                        }
                    }
                    let mean = fin as f64 / nfin as f64;
                    if smooth {
                        // Mode lisse (à la fuzz) : bord mixte, variance minimale.
                        if mx < iters {
                            continue;
                        }
                        let var = fin2 / nfin as f64 - mean * mean;
                        cands.push((-var, i, j));
                    } else {
                        cands.push((mean, i, j));
                    }
                }
            }
            if cands.is_empty() {
                println!("DEADEND zoom=1e{zoom_l:.1}");
                return;
            }
            cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            let k = (rng.next() % cands.len().min(4) as u64) as usize;
            let (_, i, j) = cands[k];
            let dx = (i as f64 + 0.5) / grid as f64 - 0.5;
            let dy = (j as f64 + 0.5) / grid as f64 - 0.5;
            let sx = Float::with_val(PREC, Float::parse(p.span_x_hp.as_deref().expect("span hp")).expect("span"));
            let sy = Float::with_val(PREC, Float::parse(p.span_y_hp.as_deref().expect("span hp")).expect("span"));
            cx += sx * dx;
            cy += sy * dy;
            zoom_l = (zoom_l + 8f64.log10()).min(target);
        }
        // Scène.
        let z_str = format!("{:e}", zooms[zi]);
        let sx = cx.to_string_radix(10, Some((target as usize) + 25));
        let sy = cy.to_string_radix(10, Some((target as usize) + 25));
        println!("SCENE {} {} {} {} {} {} {} {}", a[1], a[2], a[3], sx, sy, z_str, iters, size);
        if std::env::var_os("PROBE_HUNT_ONLY").is_none() {
            judge(ty, seed, &sx, &sy, &z_str, iters, size);
        }
        zi += 1;
    }
}
