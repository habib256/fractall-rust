//! Calibration des itérations automatiques (`wisdom::auto_iterations`) sur le
//! corpus `toml/` et les presets de `fractall-quality`.
//!
//! Pour chaque vue : sonde (cible 1920×1080, sonde ~4 096 px) →
//! plafond auto `A` ; puis deux rendus de VÉRITÉ à la taille `--truth WxH`
//! (défaut 320×180) aux plafonds `A` et `4A` : fraction de pixels dont la
//! classification évadé/plafonné change. Rapporte aussi `A ÷ auteur`, le coût
//! de la sonde, et (avec `--full`) le coût d'un rendu 1920×1080 au plafond `A`.
//!
//! Usage :
//!   cargo run --release --example calibrate_auto_iter -- [--truth 320x180]
//!       [--full] [--timeout SECS] [--max-author N] [--presets] [fichiers.toml…]
//! Sans fichier : tout `toml/*.toml`. Une vue dont un rendu dépasse
//! `--timeout` (défaut 120 s) est marquée TIMEOUT. `--max-author N` saute les
//! fichiers dont l'auteur demande plus de N itérations.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fractall_cli::fractal::wisdom::{auto_iterations_with, AutoIterOptions};
use fractall_cli::fractal::{default_params_for_type, FractalParams, FractalType, ViewHp};
use fractall_cli::quality::presets::PRESETS;
use fractall_cli::render::{render_request, RenderRequest};

struct Case {
    name: String,
    params: FractalParams,
    author: u32,
}

fn load_toml(path: &PathBuf, w: u32, h: u32) -> Option<Case> {
    let content = std::fs::read_to_string(path).ok()?;
    let table: toml::Table = content.parse().ok()?;
    let s = |k: &str| -> Option<String> {
        let v = table.get(k).or_else(|| table.get("location").and_then(|l| l.get(k)))?;
        v.as_str()
            .map(|x| x.to_string())
            .or_else(|| v.as_float().map(|f| f.to_string()))
            .or_else(|| v.as_integer().map(|i| i.to_string()))
    };
    let real = s("real")?;
    let imag = s("imag")?;
    let zoom = s("zoom")?;
    let author: u32 = s("iterations").and_then(|x| x.parse::<u64>().ok()).map(|x| x.min(u32::MAX as u64) as u32)?;
    let mut p = default_params_for_type(FractalType::Mandelbrot, w, h);
    let view = ViewHp::from_center_and_zoom(&real, &imag, &zoom, w, h, 1024)?;
    view.write_to_params(&mut p);
    p.center_x_hp = Some(real);
    p.center_y_hp = Some(imag);
    if let Some(ph) = s("phases") {
        let v: Vec<FractalType> = ph.split(',').filter_map(|n| FractalType::from_hybrid_name(n.trim())).collect();
        p.formula.hybrid_phases = Some(v);
    }
    if let Some(ops) = s("opcodes") {
        p.formula.hybrid_opcodes = Some(ops);
    }
    if let Some(r) = s("rotate").and_then(|x| x.parse::<f64>().ok()) {
        p.rotation = r;
    }
    if let Some(r) = s("escape_radius").and_then(|x| x.parse::<f64>().ok()) {
        p.bailout = r;
    }
    p.iteration_max = author;
    Some(Case {
        name: path.file_stem()?.to_string_lossy().into_owned(),
        params: p,
        author,
    })
}

fn preset_cases(w: u32, h: u32) -> Vec<Case> {
    PRESETS
        .iter()
        .filter_map(|pr| {
            let mut p = default_params_for_type(pr.fractal_type, w, h);
            let v = ViewHp::from_center_and_zoom(pr.center_x_hp, pr.center_y_hp, pr.zoom, w, h, 1024)?;
            v.write_to_params(&mut p);
            p.center_x_hp = Some(pr.center_x_hp.to_string());
            p.center_y_hp = Some(pr.center_y_hp.to_string());
            if let Some(m) = pr.multibrot_power {
                p.formula.multibrot_power = m;
            }
            if let Some((re, im)) = pr.julia_seed {
                p.seed = num_complex::Complex64::new(re, im);
            }
            p.iteration_max = pr.iterations;
            Some(Case {
                name: format!("preset:{}", pr.name),
                params: p,
                author: pr.iterations,
            })
        })
        .collect()
}

/// Rendu avec délai maximal (annulation par un fil de garde).
fn render_timed(p: &FractalParams, timeout: Duration) -> Option<(Vec<u32>, f64)> {
    let cancel = Arc::new(AtomicBool::new(false));
    let (out, secs) = with_watchdog(&cancel, timeout, || render_request(RenderRequest::new(p, &cancel), &mut None));
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    out.map(|o| (o.iterations, secs))
}

/// Exécute `f` en levant `cancel` au-delà de `timeout`.
fn with_watchdog<T>(cancel: &Arc<AtomicBool>, timeout: Duration, f: impl FnOnce() -> T) -> (T, f64) {
    let done = Arc::new(AtomicBool::new(false));
    let (c2, d2) = (cancel.clone(), done.clone());
    let guard = std::thread::spawn(move || {
        let t = Instant::now();
        while !d2.load(Ordering::Relaxed) {
            if t.elapsed() > timeout {
                c2.store(true, Ordering::Relaxed);
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    let t = Instant::now();
    let out = f();
    let secs = t.elapsed().as_secs_f64();
    done.store(true, Ordering::Relaxed);
    let _ = guard.join();
    (out, secs)
}

fn resized(p: &FractalParams, w: u32, h: u32, cap: u32) -> FractalParams {
    let mut q = p.clone();
    q.width = w;
    q.height = h;
    q.iteration_max = cap;
    q
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut tw, mut th) = (320u32, 180u32);
    let mut full = false;
    let mut timeout = 120.0f64;
    let mut max_author = u32::MAX;
    let mut presets = false;
    let mut files: Vec<PathBuf> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--truth" => {
                let v = args.next().unwrap();
                let (a, b) = v.split_once('x').unwrap();
                tw = a.parse().unwrap();
                th = b.parse().unwrap();
            }
            "--full" => full = true,
            "--presets" => presets = true,
            "--timeout" => timeout = args.next().unwrap().parse().unwrap(),
            "--max-author" => max_author = args.next().unwrap().parse().unwrap(),
            _ => files.push(PathBuf::from(a)),
        }
    }
    if files.is_empty() && !presets {
        let mut v: Vec<PathBuf> = std::fs::read_dir("toml")
            .expect("toml/")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        v.sort();
        files = v;
    }
    let (vw, vh) = (1920u32, 1080u32);
    let mut cases: Vec<Case> = files.iter().filter_map(|f| load_toml(f, vw, vh)).collect();
    if presets {
        cases.extend(preset_cases(vw, vh));
    }
    let to = Duration::from_secs_f64(timeout);
    println!(
        "{:<28} {:>9} {:>9} {:>7} {:>9} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}  notes",
        "case", "author", "auto", "ratio", "maxesc", "rnds", "probe_s", "chg4x%", "esc%", "tA_s", "t1080_s(fresh/reuse)"
    );
    for c in cases {
        if c.author > max_author {
            println!("{:<28} {:>9} SKIP (author > {max_author})", c.name, c.author);
            continue;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let opts = AutoIterOptions::default();
        let mut cache = None;
        let (res, secs) = with_watchdog(&cancel, to * 3, || auto_iterations_with(&c.params, &cancel, opts, &mut cache));
        let r = match res {
            Ok(r) => r,
            Err(e) => {
                println!("{:<28} {:>9} PROBE {e:?} after {secs:.1} s", c.name, c.author);
                continue;
            }
        };
        let a = r.iteration_max;
        let lo = render_timed(&resized(&c.params, tw, th, a), to);
        let hi = render_timed(&resized(&c.params, tw, th, a.saturating_mul(4)), to);
        let (chg, esc, t_a) = match (&lo, &hi) {
            (Some((l, t)), Some((hh, _))) => {
                let n = l.len() as f64;
                let changed = l.iter().zip(hh).filter(|(x, y)| (**x >= a) != (**y >= a.saturating_mul(4))).count();
                let escaped = l.iter().filter(|&&x| x < a).count();
                (
                    format!("{:.3}", 100.0 * changed as f64 / n),
                    format!("{:.1}", 100.0 * escaped as f64 / n),
                    format!("{t:.2}"),
                )
            }
            _ => ("TIMEOUT".into(), "-".into(), lo.as_ref().map(|x| format!("{:.2}", x.1)).unwrap_or("-".into())),
        };
        // Coût d'un rendu 1080p au plafond choisi : frais, puis en
        // réutilisant l'orbite de la sonde (câblage GUI / CLI recommandé).
        let t_full = if full {
            let fresh = render_timed(&resized(&c.params, 1920, 1080, a), Duration::from_secs_f64(timeout * 4.0))
                .map(|x| format!("{:.2}", x.1))
                .unwrap_or_else(|| "TIMEOUT".into());
            let mut fp = c.params.clone();
            fp.iteration_max = a;
            let ccancel = Arc::new(AtomicBool::new(false));
            let t = Instant::now();
            let reused = render_request(RenderRequest::new(&resized(&fp, 1920, 1080, a), &ccancel), &mut cache)
                .map(|_| format!("{:.2}", t.elapsed().as_secs_f64()))
                .unwrap_or_else(|| "-".into());
            format!("{fresh}/{reused}")
        } else {
            "-".into()
        };
        let mut notes = String::new();
        if r.ceiling_hit {
            notes.push_str("CEILING ");
        }
        if r.no_escape {
            notes.push_str("NO_ESCAPE ");
        }
        notes.push_str(&format!("pred={:.3}% last_cap={}", 100.0 * r.predicted_change, r.last_probe_cap));
        println!(
            "{:<28} {:>9} {:>9} {:>7.3} {:>9} {:>6} {:>8.2} {:>8} {:>8} {:>8} {:>8}  {}",
            c.name,
            c.author,
            a,
            a as f64 / c.author.max(1) as f64,
            r.max_escaped,
            r.rounds,
            r.elapsed.as_secs_f64(),
            chg,
            esc,
            t_a,
            t_full,
            notes
        );
    }
}
