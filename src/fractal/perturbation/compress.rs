//! Compression d'orbite de référence par waypoints — port de Zhuoran
//! `Imagina-Algorithms/PTWithCompression.h` (GPL-3, cf.
//! `docs/imagina-algorithms-analysis.md` §PTWithCompression).
//!
//! Principe : pendant le calcul haute précision de la référence, une **orbite
//! fantôme f64** itère la même formule (`z ← z² + c`) à précision pixel. Tant
//! que le fantôme colle à la vraie orbite (`cheb(z − Z) ≤ cheb(Z) · tol`,
//! tol = 2⁻³²), rien n'est stocké. Au décrochage, on émet un waypoint
//! `{Z, itération}` et on resynchronise le fantôme (`z = Z`). La décompression
//! rejoue `z² + c` en f64 et snappe aux waypoints — 1 carré complexe + 1
//! compare par pas, souvent plus rapide qu'un load DRAM raté sur les orbites
//! 10⁷+ iters (classe wfs_mb/opus2/dragon : 48 o/iter → O(waypoints)).
//!
//! Statut : **phase 1 (instrumentation)** — le compresseur tourne en parallèle
//! du build d'orbite sous `FRACTALL_COMPRESS_REF_STATS=1` et log densité de
//! waypoints + validation de tolérance, SANS changer le stockage. La phase 2
//! (swap du stockage + pont BLA `Z_atterrissage`) est gatée par ces mesures
//! (cf. TODO G8.2 : pièges connus = accès aléatoire BLA, wrap_periodic,
//! e22522 réf frôlant 0).

use num_complex::Complex64;

/// Tolérance de décrochage par défaut (Imagina : `0x1p-32`).
pub const DEFAULT_TOLERANCE: f64 = 2.328_306_436_538_696e-10; // 2^-32

/// Norme de Chebyshev (max des composantes en valeur absolue) — celle
/// d'Imagina. Plus lâche que la norme euclidienne d'un facteur ≤ √2, absorbé
/// par la tolérance.
#[inline]
pub fn chebyshev_norm(z: Complex64) -> f64 {
    z.re.abs().max(z.im.abs())
}

/// Un point de resynchronisation : la vraie valeur de la référence à
/// l'itération donnée (1-based, comme Imagina : le waypoint émis en itération
/// `i` est la valeur APRÈS `i` applications de la formule).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub z: Complex64,
    pub iteration: u32,
}

/// Orbite de référence compressée : la constante + les waypoints.
#[derive(Debug, Clone, Default)]
pub struct CompressedReference {
    pub waypoints: Vec<Waypoint>,
    pub c: Complex64,
}

impl CompressedReference {
    /// Empreinte mémoire approximative (24 o/waypoint).
    pub fn memory_bytes(&self) -> usize {
        self.waypoints.len() * std::mem::size_of::<Waypoint>()
    }
}

/// Compresseur : à appeler avec chaque valeur `Z[i]` (i = 1..) de l'orbite
/// haute précision, arrondie f64, dans l'ordre. Mirror exact de
/// `PTWithCompression::ReferenceCompressor`.
pub struct ReferenceCompressor<'a> {
    reference: &'a mut CompressedReference,
    z: Complex64,
    c: Complex64,
    iteration: u32,
    tolerance: f64,
}

impl<'a> ReferenceCompressor<'a> {
    pub fn new(reference: &'a mut CompressedReference, c: Complex64) -> Self {
        Self::with_tolerance(reference, c, DEFAULT_TOLERANCE)
    }

    pub fn with_tolerance(
        reference: &'a mut CompressedReference,
        c: Complex64,
        tolerance: f64,
    ) -> Self {
        reference.waypoints.clear();
        reference.c = c;
        ReferenceCompressor {
            reference,
            z: Complex64::new(0.0, 0.0),
            c,
            iteration: 0,
            tolerance,
        }
    }

    /// Ajoute `Z` = valeur exacte (arrondie f64) de la référence après
    /// `self.iteration + 1` itérations. Avance le fantôme et émet un waypoint
    /// si l'écart relatif dépasse la tolérance.
    ///
    /// NB : `cheb(Z)·tol` traite aussi le cas quasi-nucleus `|Z| ≈ 0`
    /// (e22522) : le seuil s'effondre avec |Z| → waypoint forcé, le fantôme
    /// se resynchronise sur la vraie valeur sous-underflow arrondie.
    pub fn add(&mut self, z_exact: Complex64) {
        self.z = self.z * self.z + self.c;
        self.iteration += 1;
        if chebyshev_norm(self.z - z_exact) > chebyshev_norm(z_exact) * self.tolerance {
            self.reference.waypoints.push(Waypoint {
                z: z_exact,
                iteration: self.iteration,
            });
            self.z = z_exact;
        }
    }

    /// Clôt la compression : la dernière valeur est TOUJOURS un waypoint
    /// (l'accès `z_ref[ref_len-1]` du rebase-at-end doit être exact).
    pub fn finalize(&mut self, z_exact: Complex64) {
        self.iteration += 1;
        self.reference.waypoints.push(Waypoint {
            z: z_exact,
            iteration: self.iteration,
        });
    }

    /// Nombre d'itérations compressées jusqu'ici.
    pub fn iterations(&self) -> u32 {
        self.iteration
    }
}

/// Décompresseur séquentiel : `next()` rend `Z[i]` pour i = 1.. dans l'ordre,
/// `reset()` rembobine à l'itération 0 (= ce que fait le rebase F3).
/// Mirror exact de `PTWithCompression::ReferenceDecompressor`.
pub struct ReferenceDecompressor<'a> {
    reference: &'a CompressedReference,
    z: Complex64,
    iteration: u32,
    next_waypoint: usize,
}

impl<'a> ReferenceDecompressor<'a> {
    pub fn new(reference: &'a CompressedReference) -> Self {
        ReferenceDecompressor {
            reference,
            z: Complex64::new(0.0, 0.0),
            iteration: 0,
            next_waypoint: 0,
        }
    }

    #[inline]
    pub fn get(&self) -> Complex64 {
        self.z
    }

    /// Avance d'une itération et rend la valeur de la référence.
    #[inline]
    pub fn next(&mut self) -> Complex64 {
        self.iteration += 1;
        let wp = &self.reference.waypoints;
        if self.next_waypoint < wp.len() && self.iteration == wp[self.next_waypoint].iteration {
            self.z = wp[self.next_waypoint].z;
            self.next_waypoint += 1;
        } else {
            self.z = self.z * self.z + self.reference.c;
        }
        self.z
    }

    /// Vrai quand la réf compressée est épuisée (dernier waypoint consommé).
    #[inline]
    pub fn end(&self) -> bool {
        self.next_waypoint >= self.reference.waypoints.len()
    }

    /// Rembobine à l'itération 0 (rebase).
    pub fn reset(&mut self) -> Complex64 {
        self.z = Complex64::new(0.0, 0.0);
        self.iteration = 0;
        self.next_waypoint = 0;
        self.z
    }
}

/// Gate d'instrumentation : `FRACTALL_COMPRESS_REF_STATS=1` fait tourner le
/// compresseur en parallèle du build d'orbite et log `[COMPRESS]` (densité,
/// ratio, validation). Zéro impact stockage/rendu.
pub fn compress_stats_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FRACTALL_COMPRESS_REF_STATS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Orbite Mandelbrot GMP 256 b arrondie f64 — comme en production
    /// (`compute_reference_orbit` stocke le f64 arrondi de l'orbite GMP).
    /// ⚠️ Un orbite calculée EN f64 serait bit-identique au fantôme du
    /// compresseur → zéro waypoint, test vide de sens.
    fn orbit(c: Complex64, n: u32) -> Vec<Complex64> {
        use rug::Complex;
        let prec = 256;
        let c_gmp = Complex::with_val(prec, (c.re, c.im));
        let mut z = Complex::with_val(prec, (0.0, 0.0));
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            z.square_mut();
            z += &c_gmp;
            out.push(Complex64::new(z.real().to_f64(), z.imag().to_f64()));
        }
        out
    }

    /// Compresse une orbite f64 exacte puis vérifie que la décompression la
    /// rejoue dans la tolérance, valeur par valeur, et que les waypoints sont
    /// snappés exacts.
    fn roundtrip(c: Complex64, n: u32) -> (usize, f64) {
        let orb = orbit(c, n);
        let mut cref = CompressedReference::default();
        {
            let mut comp = ReferenceCompressor::new(&mut cref, c);
            for z in &orb[..orb.len() - 1] {
                comp.add(*z);
            }
            comp.finalize(orb[orb.len() - 1]);
        }
        let mut dec = ReferenceDecompressor::new(&cref);
        let mut max_rel = 0.0f64;
        for (i, z_exact) in orb.iter().enumerate() {
            let z = dec.next();
            let scale = chebyshev_norm(*z_exact).max(f64::MIN_POSITIVE);
            let rel = chebyshev_norm(z - z_exact) / scale;
            assert!(
                rel <= DEFAULT_TOLERANCE * 1.0001 || z == *z_exact,
                "iter {i}: rel={rel:e} > tol"
            );
            if rel > max_rel {
                max_rel = rel;
            }
        }
        assert!(dec.end(), "dernier waypoint non consommé");
        (cref.waypoints.len(), max_rel)
    }

    #[test]
    fn roundtrip_interior_orbit_compresses_hard() {
        // Point intérieur (composante période 3) : orbite convergente,
        // fantôme ultra-stable → compression massive attendue.
        let (wps, _) = roundtrip(Complex64::new(-0.12, 0.75), 10_000);
        assert!(wps < 200, "intérieur : {wps} waypoints pour 10k iters");
    }

    #[test]
    fn roundtrip_chaotic_orbit_stays_within_tolerance() {
        // Axe réel chaotique près de −2 (borné : [−2, 0.25] ⊂ M sur l'axe
        // réel). L'écart GMP↔fantôme f64 s'amplifie au taux de Lyapunov →
        // décrochages fréquents ; les waypoints doivent maintenir la
        // tolérance partout.
        let (wps, max_rel) = roundtrip(Complex64::new(-1.9997740601362, 0.0), 5_000);
        assert!(wps > 10, "chaotique : compression étonnamment forte ({wps})");
        assert!(max_rel <= DEFAULT_TOLERANCE * 1.0001);
    }

    #[test]
    fn near_zero_reference_forces_waypoint() {
        // Piège e22522 : |Z| ≈ 0 (quasi-nucleus). Le seuil cheb(Z)·tol
        // s'effondre → tout écart force un waypoint, y compris underflow.
        let c = Complex64::new(-1.7548776662466927, 0.0); // ~racine réelle période 3
        let orb = orbit(c, 300);
        let min_norm = orb.iter().map(|z| chebyshev_norm(*z)).fold(f64::MAX, f64::min);
        assert!(min_norm < 1e-3, "l'orbite doit frôler 0 (min={min_norm:e})");
        let mut cref = CompressedReference::default();
        {
            let mut comp = ReferenceCompressor::new(&mut cref, c);
            for z in &orb[..orb.len() - 1] {
                comp.add(*z);
            }
            comp.finalize(orb[orb.len() - 1]);
        }
        // Rejoue : chaque valeur quasi-nulle doit être restituée exacte
        // (waypoint) — sinon le pixel loop rebase sur une valeur fausse.
        let mut dec = ReferenceDecompressor::new(&cref);
        for z_exact in &orb {
            let z = dec.next();
            if chebyshev_norm(*z_exact) < 1e-6 {
                assert_eq!(z, *z_exact, "valeur quasi-nucleus non snappée");
            }
        }
    }

    #[test]
    fn reset_replays_identically() {
        // Point borné (axe réel chaotique) : pas d'escape → pas de NaN
        // (NaN != NaN casserait l'assert d'égalité).
        let c = Complex64::new(-1.7893654301, 0.0);
        let orb = orbit(c, 1_000);
        let mut cref = CompressedReference::default();
        {
            let mut comp = ReferenceCompressor::new(&mut cref, c);
            for z in &orb[..999] {
                comp.add(*z);
            }
            comp.finalize(orb[999]);
        }
        let mut dec = ReferenceDecompressor::new(&cref);
        let first: Vec<Complex64> = (0..1000).map(|_| dec.next()).collect();
        dec.reset();
        let second: Vec<Complex64> = (0..1000).map(|_| dec.next()).collect();
        assert_eq!(first, second, "reset() doit rejouer bit-identique");
    }
}
