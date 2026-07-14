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
//! Statut : **phase 2 (swap de stockage env-gated)** — sous
//! `FRACTALL_COMPRESS_REF=1`, le path bytecode **f64 Mandelbrot** lit la
//! référence via [`ReferenceDecompressor`] (waypoints + replay `z²+c`) au lieu
//! du tableau `z_ref_f64`, et les tableaux `z_ref_f64`/`z_ref` sont libérés
//! après le build de la table BLA (cf. `pixel_loop::RefF64Source`,
//! `mod.rs::strip_orbit_arrays_for_compress`). Le pont BLA (accès aléatoire au
//! point d'atterrissage d'un saut) passe par `BlaMultiStep::z_land` +
//! [`ReferenceDecompressor::seek`]. Sans le gate : zéro changement (le path
//! par défaut reste bit-identique). L'instrumentation phase 1
//! (`FRACTALL_COMPRESS_REF_STATS=1`, log `[COMPRESS]`) reste disponible.

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
    /// Longueur LOGIQUE de l'orbite compressée = nombre de valeurs stockées de
    /// l'ancien `z_ref_f64` (index 0 = Z₀ = 0 inclus). Posée par
    /// [`ReferenceCompressor::seal`] / `finalize` ; sert de `ref_len` au pixel
    /// loop quand les tableaux pleins sont libérés.
    pub len: u32,
}

impl CompressedReference {
    /// Empreinte mémoire approximative (24 o/waypoint).
    pub fn memory_bytes(&self) -> usize {
        self.waypoints.len() * std::mem::size_of::<Waypoint>()
    }

    /// `Z[len-1]` exact (rebase-at-end) : par contrat `seal`/`finalize`, la
    /// dernière valeur de l'orbite est TOUJOURS un waypoint.
    pub fn end_value(&self) -> Complex64 {
        self.waypoints
            .last()
            .map(|w| w.z)
            .unwrap_or(Complex64::new(0.0, 0.0))
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
    /// Usage « add pour tout SAUF la dernière valeur, puis finalize(last) »
    /// (cf. tests roundtrip). Le build d'orbite production, qui appelle `add`
    /// pour TOUTES les valeurs au fil de l'eau, doit clore par [`seal`].
    #[allow(dead_code)] // API mirror Imagina, consommée par les tests roundtrip.
    pub fn finalize(&mut self, z_exact: Complex64) {
        self.iteration += 1;
        self.reference.waypoints.push(Waypoint {
            z: z_exact,
            iteration: self.iteration,
        });
        self.reference.len = self.iteration + 1;
    }

    /// Clôture PRODUCTION (phase 2) : `add` a déjà été appelé pour chaque
    /// valeur stockée, y compris la dernière — on garantit seulement que la
    /// dernière itération est snappée en waypoint EXACT (contrat rebase-at-end
    /// / [`CompressedReference::end_value`]), SANS avancer le compteur.
    /// Idempotent si `add` a déjà émis le waypoint terminal.
    pub fn seal(&mut self, z_exact: Complex64) {
        if self.iteration > 0 {
            let already = self
                .reference
                .waypoints
                .last()
                .is_some_and(|w| w.iteration == self.iteration);
            if !already {
                self.reference.waypoints.push(Waypoint {
                    z: z_exact,
                    iteration: self.iteration,
                });
                self.z = z_exact;
            }
        }
        // Longueur logique = index 0 (Z₀) + `iteration` valeurs ajoutées.
        self.reference.len = self.iteration + 1;
    }

    /// Nombre d'itérations compressées jusqu'ici.
    #[allow(dead_code)] // API mirror Imagina (instrumentation).
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
    #[allow(dead_code)] // API mirror Imagina, consommée par les tests.
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
    #[allow(dead_code)] // API mirror Imagina, consommée par les tests.
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

    /// Téléporte l'état à l'itération `m` avec la valeur EXACTE fournie
    /// (atterrissage d'un saut BLA : `z_exact` = `BlaMultiStep::z_land`,
    /// bit-copie de l'ancienne `z_ref_f64[m]`).
    ///
    /// Sûreté : repartir d'une valeur exacte (arrondie f64) donne une erreur
    /// de replay ≤ celle du fantôme canonique au même point — même dynamique
    /// (`z²+c`), départ au moins aussi proche de la vraie orbite — donc la
    /// tolérance 2⁻³² tient jusqu'au prochain snap waypoint.
    pub fn seek(&mut self, m: u32, z_exact: Complex64) {
        self.z = z_exact;
        self.iteration = m;
        // Premier waypoint STRICTEMENT après m (ceux ≤ m sont consommés).
        self.next_waypoint = self
            .reference
            .waypoints
            .partition_point(|w| w.iteration <= m);
    }

    /// `Z[len-1]` exact (cf. [`CompressedReference::end_value`]).
    #[inline]
    pub fn end_value(&self) -> Complex64 {
        self.reference.end_value()
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

/// Gate phase 2 : `FRACTALL_COMPRESS_REF=1` active le stockage compressé de la
/// référence (build dans `orbit.rs`) + le routage du pixel loop f64 Mandelbrot
/// vers le décompresseur (`delta.rs`) + la libération des tableaux pleins
/// (`mod.rs`). Sans lui : zéro changement de comportement.
pub fn compress_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FRACTALL_COMPRESS_REF")
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

    /// Compression production : `add` pour TOUTES les valeurs + `seal(last)`.
    /// Mapping : itération k du décompresseur ↔ orb[k-1] (orb sans le Z₀=0).
    fn compress_prod(c: Complex64, orb: &[Complex64]) -> CompressedReference {
        let mut cref = CompressedReference::default();
        {
            let mut comp = ReferenceCompressor::new(&mut cref, c);
            for z in orb {
                comp.add(*z);
            }
            comp.seal(orb[orb.len() - 1]);
        }
        cref
    }

    /// `seek(m, Z[m] exact)` mi-segment puis replay séquentiel : chaque valeur
    /// suivante doit rester dans la tolérance vs l'orbite GMP exacte (⚠️
    /// orbite sourcée en GMP 256 b, JAMAIS f64 — sinon bit-identique au
    /// fantôme et le test est vide), et snapper exact aux waypoints. Sûreté
    /// `seek` : départ exact ⇒ erreur ≤ fantôme canonique.
    #[test]
    fn seek_mid_segment_then_replay_stays_within_tolerance() {
        let c = Complex64::new(-1.9997740601362, 0.0); // borné, chaotique
        let orb = orbit(c, 5_000);
        let cref = compress_prod(c, &orb);
        assert!(cref.waypoints.len() > 10, "orbite chaotique attendue dense");
        assert_eq!(cref.len as usize, orb.len() + 1);

        // Milieu d'un segment inter-waypoints d'au moins 8 pas.
        let (mut lo, mut hi) = (0u32, 0u32);
        for pair in cref.waypoints.windows(2) {
            if pair[1].iteration - pair[0].iteration >= 8 {
                lo = pair[0].iteration;
                hi = pair[1].iteration;
                break;
            }
        }
        assert!(hi > lo, "aucun segment ≥ 8 pas trouvé");
        let m = lo + (hi - lo) / 2;

        let mut dec = ReferenceDecompressor::new(&cref);
        dec.seek(m, orb[(m - 1) as usize]); // valeur EXACTE à l'itération m
        let waypoint_iters: std::collections::HashSet<u32> =
            cref.waypoints.iter().map(|w| w.iteration).collect();
        for k in (m + 1)..=(orb.len() as u32) {
            let z = dec.next();
            let z_exact = orb[(k - 1) as usize];
            let scale = chebyshev_norm(z_exact).max(f64::MIN_POSITIVE);
            let rel = chebyshev_norm(z - z_exact) / scale;
            assert!(
                rel <= DEFAULT_TOLERANCE * 1.0001 || z == z_exact,
                "iter {k} après seek({m}): rel={rel:e} > tol"
            );
            if waypoint_iters.contains(&k) {
                assert_eq!(z, z_exact, "waypoint {k} non snappé après seek");
            }
        }
        assert!(dec.end(), "waypoint terminal non consommé après seek+replay");
    }

    /// `seek(0, 0)` ≡ `reset()` : même état, même replay bit-identique.
    #[test]
    fn seek_zero_equals_reset() {
        let c = Complex64::new(-1.7893654301, 0.0); // borné (pas de NaN)
        let orb = orbit(c, 1_000);
        let cref = compress_prod(c, &orb);

        let mut dec = ReferenceDecompressor::new(&cref);
        dec.reset();
        let via_reset: Vec<Complex64> = (0..orb.len()).map(|_| dec.next()).collect();
        dec.seek(0, Complex64::new(0.0, 0.0));
        assert_eq!(dec.get(), Complex64::new(0.0, 0.0));
        let via_seek: Vec<Complex64> = (0..orb.len()).map(|_| dec.next()).collect();
        assert_eq!(via_reset, via_seek, "seek(0,0) doit rejouer comme reset()");
    }

    /// `seal` : la dernière valeur est toujours un waypoint exact à l'itération
    /// finale (pas de +1 fantôme), `len` = valeurs stockées + Z₀, `end_value`
    /// exact — même quand le fantôme n'a pas décroché à la dernière itération.
    #[test]
    fn seal_snaps_terminal_waypoint_without_advancing() {
        let c = Complex64::new(-0.12, 0.75); // intérieur : peu de waypoints
        let orb = orbit(c, 2_000);
        let cref = compress_prod(c, &orb);
        let last_wp = cref.waypoints.last().expect("waypoint terminal");
        assert_eq!(last_wp.iteration, orb.len() as u32);
        assert_eq!(last_wp.z, orb[orb.len() - 1]);
        assert_eq!(cref.len as usize, orb.len() + 1);
        assert_eq!(cref.end_value(), orb[orb.len() - 1]);
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
