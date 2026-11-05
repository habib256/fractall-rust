//! Diagnostic de la boucle scaled-double (`FRACTALL_EXP_SCALED_STATS=1`) :
//! ventilation des tours de boucle (rapide / lent) et causes des replis.

use num_complex::Complex64;

use super::{p2, Scaled, NZ_MIN};
use crate::fractal::perturbation::counter::PixelCounter;

/// Compteurs de tours de boucle (diagnostic `FRACTALL_EXP_SCALED_STATS`).
#[derive(Default, Clone, Copy, Debug)]
pub struct ScaledStats {
    pub fast_direct: u64,
    pub fast_jump: u64,
    pub slow_iter: u64,
    pub fallback: u64,
    pub fast_rebase: u64,
    /// Causes des replis : [bout de réf, |Z| petit, rebase possible, bailout
    /// proche, garde arithmétique].
    pub reasons: [u64; 5],
}

/// Diagnostic : cause du repli d'un tour rapide (mêmes tests que `fast_iter`).
pub(super) fn fallback_reason(
    s: Scaled,
    c: PixelCounter,
    z_ref_f64: &[Complex64],
    bailout_fast: f64,
) -> usize {
    let m = c.m_usize();
    if m + 1 >= z_ref_f64.len() {
        return 0;
    }
    let z = z_ref_f64[m];
    let nz = z.re * z.re + z.im * z.im;
    let nd = s.d.re * s.d.re + s.d.im * s.d.im;
    if nz < NZ_MIN {
        1
    } else if nd * p2(2 * s.e + 4) >= nz {
        2
    } else if 2.0 * (nz + nd * p2(2 * s.e)) >= bailout_fast {
        3
    } else {
        4
    }
}

pub(super) fn flush_stats(st: &ScaledStats) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    #[allow(clippy::declare_interior_mutable_const)]
    const Z: AtomicU64 = AtomicU64::new(0);
    static C: [AtomicU64; 5] = [Z; 5];
    static R: [AtomicU64; 5] = [Z; 5];
    for (r, v) in R.iter().zip(st.reasons) {
        r.fetch_add(v, Relaxed);
    }
    C[0].fetch_add(st.fast_direct, Relaxed);
    C[1].fetch_add(st.fast_jump, Relaxed);
    C[2].fetch_add(st.slow_iter, Relaxed);
    C[3].fetch_add(st.fallback, Relaxed);
    let px = C[4].fetch_add(1, Relaxed) + 1;
    if px % 65536 == 0 {
        let (fd, fj, sl, fb) = (
            C[0].load(Relaxed),
            C[1].load(Relaxed),
            C[2].load(Relaxed),
            C[3].load(Relaxed),
        );
        let tot = (fd + fj + sl).max(1) as f64;
        eprintln!(
            "[EXP-SCALED] pixels={px} fast_direct={fd} ({:.1}%) fast_jump={fj} ({:.1}%) \
             slow={sl} ({:.2}%) fallback={fb} causes[fin_ref,Z_petit,rebase,bailout,garde]={:?}",
            100.0 * fd as f64 / tot,
            100.0 * fj as f64 / tot,
            100.0 * sl as f64 / tot,
            R.iter().map(|r| r.load(Relaxed)).collect::<Vec<_>>()
        );
    }
}
