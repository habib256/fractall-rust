//! Banc de la sonde d'itérations automatiques (`wisdom::auto_iterations_with`) :
//! N choisi, tours, pixels CALCULÉS / plafonnés et durée de chaque tour.
//!
//! Usage :
//!   cargo run --release --example probe_bench -- <runs> <full|inc> <type> <cx|-> <cy> <zoom> [W H]
//! `full` = chaque tour re-rend tout (référence), `inc` = sonde incrémentale.
//! `-` à la place du centre = vue par défaut du type.
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use fractall_cli::fractal::wisdom::{auto_iterations_with, AutoIterOptions};
use fractall_cli::fractal::{default_params_for_type, FractalType, ViewHp};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let runs: usize = a[1].parse().expect("runs");
    let incremental = a[2] == "inc";
    let ty = FractalType::from_id(a[3].parse().expect("type")).expect("type");
    let w: u32 = a.get(7).map_or(128, |s| s.parse().expect("W"));
    let h: u32 = a.get(8).map_or(80, |s| s.parse().expect("H"));
    let mut p = default_params_for_type(ty, w, h);
    if a[4] != "-" {
        let v = ViewHp::from_center_and_zoom(&a[4], &a[5], &a[6], w, h, 1024).expect("vue");
        v.write_to_params(&mut p);
        p.center_x_hp = Some(a[4].clone());
        p.center_y_hp = Some(a[5].clone());
    }
    let opts = AutoIterOptions {
        incremental,
        ..AutoIterOptions::default()
    };
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let cancel = Arc::new(AtomicBool::new(false));
        let r = auto_iterations_with(&p, &cancel, opts, &mut None).expect("sonde");
        best = best.min(r.elapsed.as_secs_f64());
        let work: Vec<String> = r
            .round_log
            .iter()
            .map(|x| format!("{}:{}calc/{}plaf/{:.3}s", x.cap, x.computed, x.capped, x.secs))
            .collect();
        println!(
            "N={} tours={} t={:.3}s  {}",
            r.iteration_max,
            r.rounds,
            r.elapsed.as_secs_f64(),
            work.join(" ")
        );
    }
    println!("min={best:.3}s");
}
