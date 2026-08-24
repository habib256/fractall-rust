//! Rapport de progression et diagnostics de performance du rendu perturbé.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::fractal::FractalType;

fn env_flag_off(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

/// Affiche le breakdown timing perturbation par défaut sur stderr.
pub(crate) fn perf_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_flag_off("FRACTALL_PERTURB_STATS"))
}

/// Compteurs partagés pour le reporter live façon Fraktaler-3.
#[derive(Default)]
pub(crate) struct ProgressState {
    pub r#ref: AtomicU32,
    pub bla: AtomicU32,
    pub tile: AtomicU32,
    pub done: AtomicBool,
    done_wake: Mutex<bool>,
    done_cv: Condvar,
}

impl ProgressState {
    pub(crate) fn finish(&self) {
        self.done.store(true, Ordering::Relaxed);
        let mut guard = self.done_wake.lock().unwrap();
        *guard = true;
        self.done_cv.notify_all();
    }

    fn snapshot_line(&self) -> String {
        format!(
            "Frame[100%] Ref[{:>3}%] BLA[{:>3}%] Tile[{:>3}%]",
            self.r#ref.load(Ordering::Relaxed).min(100),
            self.bla.load(Ordering::Relaxed).min(100),
            self.tile.load(Ordering::Relaxed).min(100),
        )
    }
}

pub(crate) fn spawn_progress_reporter(state: Arc<ProgressState>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut last = String::new();
        let mut last_draw = Instant::now();
        loop {
            let done = state.done.load(Ordering::Relaxed);
            let line = state.snapshot_line();
            if line != last
                && (done || last.is_empty() || last_draw.elapsed() >= Duration::from_millis(250))
            {
                eprint!("\r{} ", line);
                let _ = std::io::Write::flush(&mut std::io::stderr());
                last = line;
                last_draw = Instant::now();
            }
            if done {
                eprintln!("\r{} ", state.snapshot_line());
                break;
            }
            let guard = state.done_wake.lock().unwrap();
            if !*guard {
                let _ = state
                    .done_cv
                    .wait_timeout(guard, Duration::from_millis(250));
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn print_fractall_summary(
    path: &'static str,
    fractal_type: FractalType,
    prec_bits: u32,
    iter_max: u32,
    iterations: &[u32],
    pixel_count: usize,
    t_pixels: Duration,
    t_total: Duration,
) {
    let total_iters: u64 = iterations.iter().map(|&n| n as u64).sum();
    let max_iter = iterations.iter().copied().max().unwrap_or(0);
    let avg_iter = if pixel_count > 0 {
        total_iters as f64 / pixel_count as f64
    } else {
        0.0
    };
    let ns_per_iter = if total_iters > 0 {
        t_pixels.as_secs_f64() * 1e9 / total_iters as f64
    } else {
        0.0
    };
    eprintln!(
        "[FRACTALL] type={:?} path={} prec={}b iter_max={} avg_iter/px={:.0} max_iter/px={} ns/iter={:.1} pixels={:.3}s total={:.3}s",
        fractal_type, path, prec_bits, iter_max, avg_iter, max_iter, ns_per_iter,
        t_pixels.as_secs_f64(), t_total.as_secs_f64(),
    );
}
