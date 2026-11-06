//! Orbite de référence **quad-double** (~212 b) pour le tier qd de la
//! correction de fiabilité (TODO G14.1 h).
//!
//! Source : les points de contrôle GMP de la capture paresseuse de la partie
//! basse (`high_precision_data`, un `Z` toutes les `LO_CHECKPOINT_INTERVAL`
//! itérations) rejoués EN PARALLÈLE par tranches avec `fast_sqr_step` — les
//! MÊMES opérations que la boucle d'orbite, donc les mêmes `Z` au bit près —
//! puis arrondis en qd. Jamais d'orbite GMP complète recalculée quand ces
//! points existent.
//!
//! Repli (orbite sans points de contrôle : fast-path dd ≤ 96 b, capture
//! eager `FRACTALL_LAZY_LO=0`, frame dd) : orbite GMP séquentielle de même
//! longueur à `max(précision, QD_MIN_ORBIT_PREC)` bits. Le fast-path dd itère
//! la référence EN dd : une orbite à ~106 b n'a pas la fidélité qu'exige un
//! pixel qd (le pixel hérite de l'arrondi de la référence à chaque pas).

use rayon::prelude::*;
use rug::{Complex, Float};

use super::orbit::{fast_orbit_sqr_enabled, fast_sqr_step, ReferenceOrbit, LO_CHECKPOINT_INTERVAL};
use super::qd::ComplexQd;

/// Précision minimale d'une orbite RE-CALCULÉE pour le tier qd (repli) :
/// au-dessus des ~212 b du qd, pour que l'orbite ne soit pas le plancher.
pub const QD_MIN_ORBIT_PREC: u32 = 256;

/// Orbite qd + précision (bits) de l'orbite GMP dont elle provient — le
/// traqueur qd compte l'arrondi de la référence (2^-prec par pas) quand
/// celle-ci est moins précise que le qd.
pub struct QdOrbit {
    pub z: Vec<ComplexQd>,
    pub prec: u32,
}

/// Orbite qd de `orbit` (longueur `n` = celle de l'orbite primaire).
pub fn build_qd_orbit(orbit: &ReferenceOrbit, n: usize) -> Option<QdOrbit> {
    if n == 0 {
        return None;
    }
    let iv = LO_CHECKPOINT_INTERVAL;
    let lazy_ok = orbit.data_storage_interval == iv
        && orbit.z_ref_gmp.is_empty()
        && orbit.high_precision_data.len() == n.div_ceil(iv)
        && fast_orbit_sqr_enabled();
    if lazy_ok {
        let prec = orbit.high_precision_data[0].prec().0;
        let mut z = vec![ComplexQd::ZERO; n];
        z.par_chunks_mut(iv).enumerate().for_each(|(k, chunk)| {
            let mut zg = orbit.high_precision_data[k].clone();
            let p = zg.prec().0;
            let (mut t1, mut t2) = (Float::new(p), Float::new(p));
            let mut scratch = Float::new(p);
            for (j, out) in chunk.iter_mut().enumerate() {
                if j > 0 {
                    fast_sqr_step(&mut zg, &orbit.cref_gmp, &mut t1, &mut t2);
                }
                *out = ComplexQd::from_complex(&zg, &mut scratch);
            }
        });
        return Some(QdOrbit { z, prec });
    }
    // Repli : orbite séquentielle (Mandelbrot z0 = 0, c = cref).
    let prec = orbit.cref_gmp.prec().0.max(QD_MIN_ORBIT_PREC);
    let c = Complex::with_val(prec, &orbit.cref_gmp);
    let mut zg = Complex::new(prec);
    let (mut t1, mut t2) = (Float::new(prec), Float::new(prec));
    let mut scratch = Float::new(prec);
    let mut z = Vec::with_capacity(n);
    z.push(ComplexQd::from_complex(&zg, &mut scratch));
    for _ in 1..n {
        fast_sqr_step(&mut zg, &c, &mut t1, &mut t2);
        z.push(ComplexQd::from_complex(&zg, &mut scratch));
    }
    Some(QdOrbit { z, prec })
}
