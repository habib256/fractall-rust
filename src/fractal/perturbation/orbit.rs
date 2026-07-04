use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use rug::{Assign, Complex, Float};

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::bytecode::{compile_formula, GmpInterpState, Formula};
use crate::fractal::gmp::{complex_to_complex64, pow_f64_mpc};
use crate::fractal::perturbation::bla::{BlaTable, build_bla_table};
use crate::fractal::perturbation::types::ComplexExp;
use crate::fractal::perturbation::series::{SeriesTable, build_series_table_ho, validate_series_with_probes_tiled, compute_adaptive_series_order};

/// Convertit `z` (Complex GMP borné de l'orbite référence) en `ComplexExp`, en
/// évitant les 2 clones GMP 676 b de `ComplexExp::from_gmp` dans le cas dominant.
///
/// `z_f64` est la valeur f64 de `z` déjà calculée par l'appelant (pour
/// `z_ref_f64`). Une composante f64 **normale** (≥ `MIN_POSITIVE`) round-trippe
/// exactement depuis GMP : `from_complex64(z_f64).c == from_gmp(z).c`
/// bit-à-bit (même arrondi 676 b → 53 b). Une composante f64 nulle est sûre SSI
/// le GMP est vraiment nul. Le seul cas dangereux — GMP petit non-nul mais f64
/// underflow (dénormal/0) — retombe sur `from_gmp` qui préserve l'exposant
/// étendu (nécessaire aux orbites qui frôlent 0, cf. `extended_iterations`).
#[inline]
fn z_ref_complexexp(z: &Complex, z_f64: Complex64) -> ComplexExp {
    #[inline(always)]
    fn component_safe(f: f64, g: &Float) -> bool {
        if f == 0.0 {
            g.is_zero() // f64 nul : sûr seulement si le GMP est vraiment nul
        } else {
            f.abs() >= f64::MIN_POSITIVE // non-nul : sûr si normal (pas dénormal)
        }
    }
    if component_safe(z_f64.re, z.real()) && component_safe(z_f64.im, z.imag()) {
        ComplexExp::from_complex64(z_f64)
    } else {
        ComplexExp::from_gmp(z)
    }
}

/// Convertit un `Float` GMP en `DoubleDoubleExp` (~106 bits) : `hi` = arrondi
/// f64, `lo` = reste (value − hi) arrondi f64. Capte min(106, prec(value)) bits.
fn gmp_float_to_ddexp(value: &Float) -> super::dd::DoubleDoubleExp {
    use super::dd::{DoubleDouble, DoubleDoubleExp};
    let hi = value.to_f64();
    if hi == 0.0 || !hi.is_finite() {
        return DoubleDoubleExp::from_f64(hi);
    }
    let mut rem = value.clone();
    rem -= hi;
    let lo = rem.to_f64();
    DoubleDoubleExp::normalized(DoubleDouble::new(hi, lo), 0)
}

/// Orbite référence Mandelbrot itérée en **double-double** (~106 bits) au lieu de
/// GMP — chemin rapide pour la tranche zoom où `precision ≤ ~100 bits`
/// (~1e13–1e28). L'orbite est ITÉRÉE en dd (précision de la trajectoire) mais
/// STOCKÉE en 53 bits (`z_ref` ComplexExp + `z_ref_f64`), comme le path GMP : le
/// delta de perturbation par pixel est déjà 53 bits, seule la précision de
/// l'itération de la référence compte. NON bit-identique au path GMP (arrondi
/// dd 106 b ≠ MPFR 676 b sur les derniers ULP) — validé vs GMP par test.
///
/// Renvoie `(z_ref, z_ref_f64)`. Mirror de la structure de boucle de
/// `compute_reference_orbit` (bailout `REFERENCE_BAILOUT_SQR` en tête, break sur
/// évasion). Mandelbrot uniquement (`z = z² + c`).
fn dd_reference_orbit_mandelbrot(
    cref_gmp: &Complex,
    iteration_max: u32,
    cancel: Option<&AtomicBool>,
) -> Option<(Vec<ComplexExp>, Vec<Complex64>)> {
    use super::dd::ComplexDDExp;
    let cref = ComplexDDExp {
        re: gmp_float_to_ddexp(cref_gmp.real()),
        im: gmp_float_to_ddexp(cref_gmp.imag()),
    };
    let mut z = ComplexDDExp::ZERO;
    let mut z_ref = Vec::with_capacity(iteration_max as usize + 1);
    let mut z_ref_f64 = Vec::with_capacity(iteration_max as usize + 1);
    let mut zf = z.to_complex64_approx();
    z_ref.push(ComplexExp::from_complex64(zf));
    z_ref_f64.push(zf);
    for i in 0..iteration_max {
        if let Some(cancel) = cancel {
            if i % 4096 == 0 && cancel.load(Ordering::Relaxed) {
                return None;
            }
        }
        // Bailout f64 (z borné O(1) < ~1e5 avant évasion, exact en f64).
        if zf.re * zf.re + zf.im * zf.im > REFERENCE_BAILOUT_SQR {
            break;
        }
        z = z.sqr().add(cref);
        if (i + 1) % 250 == 0 {
            z.reduce();
        }
        zf = z.to_complex64_approx();
        z_ref.push(ComplexExp::from_complex64(zf));
        z_ref_f64.push(zf);
    }
    Some((z_ref, z_ref_f64))
}

/// Squared escape radius used while iterating the reference orbit.
///
/// Matches Fraktaler-3 `hybrid.cc:87` (`norm(Zp[i]) < 1e10`). Using
/// `params.bailout²` here (typically 16 when ER=4) cuts the reference orbit
/// short whenever the center sits outside the M-set: each surviving pixel
/// then inherits `z_ref[effective_len-1] + delta`, collapsing the entire
/// frame to a single iteration count (see line.toml @ zoom 1e28 +
/// triangle/dragon @ zoom > 1e190).
///
/// The pixel-side bailout is independent — pixel escape is tested against
/// `params.bailout` in `iterate_pixel` / `pixel_loop` / `pixel_loop_exp`.
pub const REFERENCE_BAILOUT_SQR: f64 = 1e10;

/// Hybrid BLA: Multiple references for different phases of a periodic loop.
/// For a hybrid loop with multiple phases, you need multiple references, one starting at
/// each phase in the loop. Rebasing switches to the reference for the current phase.
/// You need one BLA table per reference.
#[derive(Clone, Debug)]
pub struct HybridBlaReferences {
    /// Primary reference (phase 0)
    pub primary: ReferenceOrbit,
    /// Primary BLA table (phase 0)
    pub primary_bla: BlaTable,
    /// Secondary references for other phases (if cycle detected)
    pub phases: Vec<ReferenceOrbit>,
    /// BLA tables for each phase (one per reference)
    pub phase_bla_tables: Vec<BlaTable>,
    /// Cycle period (0 if no cycle detected)
    pub cycle_period: u32,
    /// Cycle start index in the orbit
    pub cycle_start: u32,
}

impl HybridBlaReferences {
    /// Get the BLA table for a specific phase
    pub fn get_bla_table(&self, phase: u32) -> &BlaTable {
        if phase == 0 {
            &self.primary_bla
        } else if (phase as usize) <= self.phase_bla_tables.len() {
            &self.phase_bla_tables[(phase - 1) as usize]
        } else {
            // Fallback to primary if phase doesn't exist
            &self.primary_bla
        }
    }

    /// Get the reference for a specific phase
    pub fn get_reference(&self, phase: u32) -> &ReferenceOrbit {
        if phase == 0 {
            &self.primary
        } else if (phase as usize) <= self.phases.len() {
            &self.phases[(phase - 1) as usize]
        } else {
            // Fallback to primary if phase doesn't exist
            &self.primary
        }
    }

    /// Get the current phase based on iteration count
    pub fn get_current_phase(&self, iteration: u32) -> u32 {
        if self.cycle_period == 0 {
            0
        } else {
            (iteration.saturating_sub(self.cycle_start)) % self.cycle_period
        }
    }
}

/// Detect periodic cycles in the reference orbit.
/// Returns (cycle_start, cycle_period) if a cycle is detected, None otherwise.
/// A cycle is detected when z_ref[i] ≈ z_ref[j] for i < j within tolerance.
#[allow(dead_code)]
fn detect_cycle(z_ref: &[Complex64], tolerance: f64) -> Option<(u32, u32)> {
    if z_ref.len() < 10 {
        return None;
    }

    let tolerance_sqr = tolerance * tolerance;
    let min_cycle_length = 2u32;
    let max_cycle_length = (z_ref.len() / 2).min(1000) as u32;

    // Look for cycles: z[i] ≈ z[i+period] for multiple consecutive iterations
    for period in min_cycle_length..=max_cycle_length {
        let period_usize = period as usize;
        if period_usize * 2 >= z_ref.len() {
            continue;
        }

        // Check if we have a cycle starting at some point
        for start in 0..(z_ref.len() - period_usize * 2) {
            let mut cycle_valid = true;
            let check_length = period_usize.min(z_ref.len() - start - period_usize);

            // Verify that z[start+k] ≈ z[start+period+k] for k in [0, check_length)
            for k in 0..check_length {
                let z1 = z_ref[start + k];
                let z2 = z_ref[start + period_usize + k];
                let diff_sqr = (z1 - z2).norm_sqr();
                if diff_sqr > tolerance_sqr {
                    cycle_valid = false;
                    break;
                }
            }

            if cycle_valid && check_length >= period_usize.min(3) {
                return Some((start as u32, period));
            }
        }
    }

    None
}

#[derive(Clone, Debug)]
pub struct ReferenceOrbit {
    pub cref: Complex64,
    /// High precision reference orbit (extended exponent range via FloatExp)
    pub z_ref: Vec<ComplexExp>,
    /// Fast path: f64 version of reference orbit for shallow zooms
    pub z_ref_f64: Vec<Complex64>,
    /// Full GMP precision reference orbit for very deep zooms (>10^15)
    /// This preserves full GMP precision without conversion to f64
    pub z_ref_gmp: Vec<Complex>,
    /// Full GMP precision center point for very deep zooms
    pub cref_gmp: Complex,
    /// Phase offset for Hybrid BLA (0 for single reference, >0 for multi-phase)
    pub phase_offset: u32,
    /// Iterations where z_ref is very small (|re| < 1e-300 and |im| < 1e-300),
    /// requiring extended precision for the perturbation formula.
    /// Inspired by rust-fractal-core's extended_iterations tracking.
    /// The fast f64 batch path should break at these iterations to avoid precision loss.
    pub extended_iterations: Vec<u32>,
    /// Sparse high-precision orbit data stored at intervals for memory-efficient glitch resolution.
    ///
    /// Inspired by rust-fractal-core's `high_precision_data` + `data_storage_interval`:
    /// Instead of storing full GMP data at every iteration (which uses O(N * precision_bits) memory),
    /// we store it only every `data_storage_interval` iterations. For glitch resolution, we create
    /// new references starting from the nearest stored checkpoint.
    ///
    /// Access: `high_precision_data[iteration / data_storage_interval]`
    pub high_precision_data: Vec<Complex>,
    /// Interval between stored high-precision orbit points.
    /// Inspired by rust-fractal-core's `data_storage_interval`.
    /// Default: 10 (stores every 10th iteration). 1 = store every iteration (original behavior).
    pub data_storage_interval: usize,
    /// Période détectée du cycle (0 si pas de périodicité).
    /// Quand non-zéro, `z_ref[cycle_start + k] == z_ref[cycle_start + (k % cycle_period)]`.
    /// Utilisé par `pixel_loop` pour cycler l'orbite via modulo au lieu de rebaser à m=0
    /// quand on atteint la fin de l'orbite — fix le rendu uniforme sur centres
    /// périodiques intérieurs (ex. glitch_test_1 zoom 3e46 = magenta uniforme sans cyclage).
    pub cycle_period: u32,
    /// Index du début du cycle dans l'orbite (= `period_check_iter` lors de la détection
    /// Brent's). z_ref[cycle_start] et z_ref[cycle_start + cycle_period] sont équivalents
    /// modulo tolérance numérique.
    pub cycle_start: u32,
}

impl ReferenceOrbit {
    /// Get the reference point at iteration m, accounting for phase offset
    pub fn get_z_ref_f64(&self, m: u32) -> Option<Complex64> {
        let idx = (m + self.phase_offset) as usize;
        self.z_ref_f64.get(idx).copied()
    }

    /// Get the reference point at iteration m (high precision), accounting for phase offset
    pub fn get_z_ref(&self, m: u32) -> Option<ComplexExp> {
        let idx = (m + self.phase_offset) as usize;
        self.z_ref.get(idx).copied()
    }

    /// Get the reference point at iteration m (full GMP precision), accounting for phase offset
    pub fn get_z_ref_gmp(&self, m: u32) -> Option<&Complex> {
        let idx = (m + self.phase_offset) as usize;
        self.z_ref_gmp.get(idx)
    }

    /// Get the effective length of the reference orbit for this phase
    pub fn effective_len(&self) -> usize {
        self.z_ref_f64.len().saturating_sub(self.phase_offset as usize)
    }

    /// Renvoie l'index `m` cyclé dans `[cycle_start, cycle_start + cycle_period)` quand
    /// l'orbite est périodique et `m` dépasse `cycle_start`. Sinon `None` (le caller doit
    /// rebaser ou stopper). Permet à `pixel_loop` d'étendre l'orbite indéfiniment via
    /// modulo sur les centres périodiques intérieurs, évitant le rebase qui produit
    /// une image uniforme.
    #[inline]
    pub fn wrap_periodic(&self, m: u32) -> Option<u32> {
        if self.cycle_period == 0 || m < self.cycle_start {
            return None;
        }
        Some(self.cycle_start + (m - self.cycle_start) % self.cycle_period)
    }

    /// Create a glitch-resolving reference starting at a given iteration.
    ///
    /// Inspired by rust-fractal-core's `get_glitch_resolving_reference()`:
    /// Creates a new reference orbit starting from z_ref[iteration] + current_delta,
    /// with c = cref + reference_delta. This produces a reference centered on the
    /// glitch cluster, starting from the iteration where the glitch was detected.
    ///
    /// This is more efficient than computing a full orbit from scratch because
    /// we only need to iterate from the glitch point onward.
    pub fn create_glitch_reference(
        &self,
        iteration: u32,
        reference_delta_re: f64,
        reference_delta_im: f64,
        current_delta_re: f64,
        current_delta_im: f64,
        params: &FractalParams,
        cancel: Option<&AtomicBool>,
    ) -> Option<ReferenceOrbit> {
        use crate::fractal::perturbation::compute_perturbation_precision_bits;
        let prec = compute_perturbation_precision_bits(params);

        let iter_idx = iteration as usize;
        if iter_idx >= self.z_ref_gmp.len() {
            return None;
        }

        // New c = cref + reference_delta (center of glitch cluster)
        let mut new_c = Complex::with_val(prec, (self.cref_gmp.real(), self.cref_gmp.imag()));
        let ref_delta_re = Float::with_val(prec, reference_delta_re);
        let ref_delta_im = Float::with_val(prec, reference_delta_im);
        *new_c.mut_real() += &ref_delta_re;
        *new_c.mut_imag() += &ref_delta_im;

        // New z = z_ref[iteration] + current_delta
        let mut z = Complex::with_val(prec, (self.z_ref_gmp[iter_idx].real(), self.z_ref_gmp[iter_idx].imag()));
        let cur_delta_re = Float::with_val(prec, current_delta_re);
        let cur_delta_im = Float::with_val(prec, current_delta_im);
        *z.mut_real() += &cur_delta_re;
        *z.mut_imag() += &cur_delta_im;

        let cref_f64 = Complex64::new(
            new_c.real().to_f64(),
            new_c.imag().to_f64(),
        );

        // Reference orbit uses the F3-style fixed bailout (1e10 squared norm),
        // not the per-pixel bailout. See REFERENCE_BAILOUT_SQR doc.
        let bailout_sqr = Float::with_val(prec, REFERENCE_BAILOUT_SQR);

        let seed = Complex::with_val(prec, (params.seed.re, params.seed.im));
        let _is_julia = params.fractal_type == FractalType::Julia;
        let remaining_iters = params.iteration_max.saturating_sub(iteration);

        let mut z_ref = Vec::with_capacity(remaining_iters as usize + 1);
        let mut z_ref_f64 = Vec::with_capacity(remaining_iters as usize + 1);
        let mut z_ref_gmp = Vec::with_capacity(remaining_iters as usize + 1);

        z_ref.push(ComplexExp::from_gmp(&z));
        z_ref_f64.push(complex_to_complex64(&z));
        z_ref_gmp.push(z.clone());

        for i in 0..remaining_iters {
            if let Some(cancel) = cancel {
                if i % 256 == 0 && cancel.load(Ordering::Relaxed) {
                    return None;
                }
            }

            let z_norm_sqr = {
                let re = Float::with_val(prec, z.real());
                let im = Float::with_val(prec, z.imag());
                let mut re2 = re.clone(); re2 *= &re;
                let mut im2 = im.clone(); im2 *= &im;
                re2 += &im2;
                re2
            };
            if z_norm_sqr > bailout_sqr {
                break;
            }

            z = match params.fractal_type {
                FractalType::Mandelbrot => {
                    let mut z_sq = z.clone();
                    z_sq *= &z;
                    z_sq += &new_c;
                    z_sq
                }
                FractalType::Julia => {
                    let mut z_sq = z.clone();
                    z_sq *= &z;
                    z_sq += &seed;
                    z_sq
                }
                FractalType::BurningShip => {
                    let re_abs = z.real().clone().abs();
                    let im_abs = z.imag().clone().abs();
                    let z_abs_val = Complex::with_val(prec, (re_abs, im_abs));
                    let mut z_sq = z_abs_val.clone();
                    z_sq *= &z_abs_val;
                    z_sq += &new_c;
                    z_sq
                }
                FractalType::Tricorn => {
                    let z_conj = z.clone().conj();
                    let mut z_temp = z_conj.clone();
                    z_temp *= &z_conj;
                    z_temp += &new_c;
                    z_temp
                }
                _ => break,
            };

            z_ref.push(ComplexExp::from_gmp(&z));
            z_ref_f64.push(complex_to_complex64(&z));
            z_ref_gmp.push(z.clone());
        }

        let extended_iterations: Vec<u32> = z_ref_f64
            .iter()
            .enumerate()
            .filter(|(_, z)| z.re.abs() < 1e-300 && z.im.abs() < 1e-300)
            .map(|(i, _)| i as u32)
            .collect();

        Some(ReferenceOrbit {
            cref: cref_f64,
            z_ref,
            z_ref_f64,
            z_ref_gmp: z_ref_gmp.clone(),
            cref_gmp: new_c,
            phase_offset: 0,
            extended_iterations,
            // Glitch-resolving references store every iteration (interval=1)
            // since they are short-lived and need full precision access.
            high_precision_data: z_ref_gmp,
            data_storage_interval: 1,
            cycle_period: 0,
            cycle_start: 0,
        })
    }

}

/// Cache for reference orbit and BLA table to avoid recomputation between frames.
/// Supports Hybrid BLA with multiple references (one per phase).
#[derive(Clone, Debug)]
pub struct ReferenceOrbitCache {
    pub orbit: ReferenceOrbit,
    pub bla_table: BlaTable,
    /// Hybrid BLA references (multiple phases if cycle detected)
    pub hybrid_refs: Option<HybridBlaReferences>,
    /// Standalone series table for iteration skipping (optional)
    pub series_table: Option<SeriesTable>,
    /// Center X in GMP precision (stored as string for Clone/Debug)
    pub center_x_gmp: String,
    /// Center Y in GMP precision (stored as string for Clone/Debug)
    pub center_y_gmp: String,
    pub fractal_type: FractalType,
    pub precision_bits: u32,
    pub iteration_max: u32,
    /// Julia seed (for Julia-type fractals)
    pub seed_re: f64,
    pub seed_im: f64,
    /// BLA threshold used when building the table
    pub bla_threshold: f64,
    /// BLA validity scale used when building the table
    pub bla_validity_scale: f64,
}

impl ReferenceOrbitCache {
    /// Check if the cache is valid for the given parameters.
    /// The cache is valid if: same center (GMP precision), same type, precision >= required, iteration_max >= required.
    /// 
    /// IMPORTANT: Compare la précision calculée (via compute_perturbation_precision_bits)
    /// avec la précision stockée dans le cache (qui est aussi la précision calculée, pas le preset).
    pub fn is_valid_for(&self, params: &FractalParams) -> bool {
        // Compute center in GMP precision for exact comparison
        // Utiliser la précision calculée pour la comparaison
        use crate::fractal::perturbation::compute_perturbation_precision_bits;
        let required_prec = compute_perturbation_precision_bits(params);
        // Utiliser la précision maximale entre celle requise et celle du cache pour la comparaison
        let prec = required_prec.max(self.precision_bits);
        
        // Utiliser les String haute précision si disponibles pour la comparaison
        let center_x = if let Some(ref cx_hp) = params.center_x_hp {
            match Float::parse(cx_hp) {
                Ok(parse_result) => Float::with_val(prec, parse_result),
                Err(_) => Float::with_val(prec, params.center_x),
            }
        } else {
            Float::with_val(prec, params.center_x)
        };
        let center_y = if let Some(ref cy_hp) = params.center_y_hp {
            match Float::parse(cy_hp) {
                Ok(parse_result) => Float::with_val(prec, parse_result),
                Err(_) => Float::with_val(prec, params.center_y),
            }
        } else {
            Float::with_val(prec, params.center_y)
        };

        // Compare as GMP strings with full precision
        let cx_str = center_x.to_string_radix(10, None);
        let cy_str = center_y.to_string_radix(10, None);

        // Vérifier que la précision calculée nécessaire est <= précision du cache
        // self.precision_bits contient maintenant la précision calculée (pas le preset)
        // donc on compare directement avec la précision calculée requise
        // Garde-fou : une orbite bâtie sur le path bytecode ne stocke pas le GMP
        // dense (`z_ref_gmp` vide, cf. `compute_reference_orbit` force_dense_gmp).
        // Un rendu legacy en aurait besoin (glitch/GMP per-pixel) → ne pas
        // réutiliser dans ce cas. `use_bytecode_engine` étant constant en session,
        // ce cas ne survient pas en pratique ; garde défensive.
        let gmp_ok = !self.orbit.z_ref_gmp.is_empty()
            || (super::uses_bytecode_path(params)
                && !super::should_use_full_gmp_perturbation(params));

        self.fractal_type == params.fractal_type
            && gmp_ok
            && self.center_x_gmp == cx_str
            && self.center_y_gmp == cy_str
            && self.precision_bits >= required_prec  // Comparer précision calculée du cache avec précision calculée requise
            && self.iteration_max >= params.iteration_max
            && (self.seed_re - params.seed.re).abs() < 1e-15
            && (self.seed_im - params.seed.im).abs() < 1e-15
            && (self.bla_threshold - params.bla_threshold).abs() < 1e-20
            && (self.bla_validity_scale - params.bla_validity_scale).abs() < 1e-10
    }

    /// Create a new cache from computed orbit and BLA table.
    /// IMPORTANT: Stocke la précision calculée (via compute_perturbation_precision_bits)
    /// au lieu du preset (params.precision_bits) pour garantir la cohérence.
    pub fn new(
        orbit: ReferenceOrbit,
        bla_table: BlaTable,
        series_table: Option<SeriesTable>,
        params: &FractalParams,
        center_x_gmp: String,
        center_y_gmp: String,
        hybrid_refs: Option<HybridBlaReferences>,
    ) -> Self {
        // Utiliser la précision calculée au lieu du preset
        use crate::fractal::perturbation::compute_perturbation_precision_bits;
        let computed_precision = compute_perturbation_precision_bits(params);
        
        Self {
            orbit,
            bla_table,
            series_table,
            center_x_gmp,
            center_y_gmp,
            fractal_type: params.fractal_type,
            precision_bits: computed_precision,  // Stocker la précision calculée au lieu du preset
            iteration_max: params.iteration_max,
            seed_re: params.seed.re,
            seed_im: params.seed.im,
            bla_threshold: params.bla_threshold,
            bla_validity_scale: params.bla_validity_scale,
            hybrid_refs,
        }
    }
}

/// Build Hybrid BLA references from a primary orbit.
/// Detects cycles and creates references for each phase if a cycle is found.
/// For Hybrid BLA: you need one BLA table per reference.
fn build_hybrid_bla_references(
    primary_orbit: &ReferenceOrbit,
    primary_bla: &BlaTable,
    params: &FractalParams,
    cancel: Option<&AtomicBool>,
) -> Option<HybridBlaReferences> {
    // Brent's period detection en GMP (compute_reference_orbit) est canonique :
    // il utilise une tolerance ~2^(-prec*0.4) sur les valeurs GMP, ce qui est
    // strict et fiable. La re-détection f64 sur z_ref_f64 ici utilise une
    // tolerance heuristique (1e-10 typiquement) qui produit des faux positifs
    // à zoom >1e85 (span_x f64 underflows → tolerance fallback) et clone
    // l'orbite GMP (prec 3k+ bits × orbit_len 30k+ × phases jusqu'à 512) →
    // explosion mémoire 10-15 GB sur escape-time deep zoom (cf. e1000).
    //
    // Si Brent n'a pas trouvé de cycle (cycle_period=0), on respecte cette
    // décision : le centre est escape-time, hybrid_bla n'aiderait pas.
    let detected = if primary_orbit.cycle_period > 0 {
        Some((primary_orbit.cycle_start, primary_orbit.cycle_period))
    } else {
        None
    };
    if let Some((cycle_start, cycle_period)) = detected {
        // Cap mémoire : chaque phase clone z_ref_f64 + z_ref_gmp + z_ref. À
        // prec >500 bits × orbit_len >16k × period >5k, on monte à >30 GB
        // (cf. flake.toml period 7884). Au-dessus du seuil, on dégrade
        // gracieusement vers la référence primaire seule — l'image perd
        // l'optimisation hybrid_bla mais reste correcte via le wrap_periodic
        // intégré au pixel_loop{,_exp}.
        const HYBRID_BLA_PERIOD_CAP: u32 = 512;
        if cycle_period > HYBRID_BLA_PERIOD_CAP {
            if crate::fractal::perturbation::perf_enabled() {
                eprintln!(
                    "[HYBRID_BLA] cycle_period={} > {} (mémoire prohibitive) → fallback primary only",
                    cycle_period, HYBRID_BLA_PERIOD_CAP,
                );
            }
            return Some(HybridBlaReferences {
                primary: primary_orbit.clone(),
                primary_bla: primary_bla.clone(),
                phases: Vec::new(),
                phase_bla_tables: Vec::new(),
                cycle_period: 0, // Force fallback to single primary in iterate_pixel_hybrid_bla
                cycle_start: 0,
            });
        }
        // Cycle detected: create references for each phase
        // For Hybrid BLA: you need multiple references, one starting at each phase in the loop
        let mut phases = Vec::new();
        let mut phase_bla_tables = Vec::new();

        for phase in 1..cycle_period {
            if let Some(cancel) = cancel {
                if cancel.load(Ordering::Relaxed) {
                    return None;
                }
            }
            
            // Create a reference starting at this phase offset
            let phase_offset = cycle_start + phase;
            if phase_offset as usize >= primary_orbit.z_ref_f64.len() {
                break;
            }
            
            // Create reference orbit with phase offset
            let phase_orbit = ReferenceOrbit {
                cref: primary_orbit.cref,
                z_ref: primary_orbit.z_ref.clone(),
                z_ref_f64: primary_orbit.z_ref_f64.clone(),
                z_ref_gmp: primary_orbit.z_ref_gmp.clone(),
                cref_gmp: primary_orbit.cref_gmp.clone(),
                phase_offset,
                extended_iterations: primary_orbit.extended_iterations.clone(),
                high_precision_data: primary_orbit.high_precision_data.clone(),
                data_storage_interval: primary_orbit.data_storage_interval,
                cycle_period: primary_orbit.cycle_period,
                cycle_start: primary_orbit.cycle_start,
            };
            
            // Build BLA table for this phase reference (one BLA table per reference)
            let phase_bla = build_bla_table(&phase_orbit.z_ref_f64, params, phase_orbit.cref);
            
            phases.push(phase_orbit);
            phase_bla_tables.push(phase_bla);
        }
        
        Some(HybridBlaReferences {
            primary: primary_orbit.clone(),
            primary_bla: primary_bla.clone(),
            phases,
            phase_bla_tables,
            cycle_period,
            cycle_start,
        })
    } else {
        // No cycle detected : PAS de hybrid_refs. `single(clone)` dupliquait
        // l'orbite primaire entière (z_ref + z_ref_f64 + z_ref_gmp, ce dernier
        // ~170 o/iter × iter_max → 850 Mo à 5 M iters/dragon) alors que le
        // dispatch pixel (mod.rs:1137) prend déjà `cache.orbit`/`cache.bla_table`
        // — les MÊMES données — via la branche `else`. `iterate_pixel_hybrid_bla`
        // avec `cycle_period==0` ne faisait que reléguer à `iterate_pixel(primary)`,
        // donc renvoyer `None` est bit-identique et supprime le clone (0.6 s sur
        // dragon, + demi-mémoire). Le hybrid_refs n'a de sens qu'avec des phases
        // réelles (cycle détecté, branche ci-dessus).
        None
    }
}

/// Compute the reference orbit and BLA table, using cache if available.
/// Supports Hybrid BLA: detects cycles and creates multiple references (one per phase).
pub fn compute_reference_orbit_cached(
    params: &FractalParams,
    cancel: Option<&AtomicBool>,
    cache: Option<&Arc<ReferenceOrbitCache>>,
) -> Option<Arc<ReferenceOrbitCache>> {
    let perf = crate::fractal::perturbation::perf_enabled();
    let t_all = Instant::now();

    // Check if cache is valid
    if let Some(cached) = cache {
        if cached.is_valid_for(params) {
            if perf {
                eprintln!(
                    "[PERTURB PERF] reference_cache=hit prec={} iters={} type={:?} total={:.3}s",
                    cached.precision_bits,
                    cached.iteration_max,
                    cached.fractal_type,
                    t_all.elapsed().as_secs_f64()
                );
            }
            return Some(Arc::clone(cached));
        }
    }

    // Auto-adjustment of iteration_max based on series skip ratio.
    // Inspired by rust-fractal-core: if the series skips >25% of iterations,
    // double iteration_max and recompute (up to 2 rounds, capped at 10M).
    // This reveals detail hidden behind an insufficient iteration count.
    const MAX_AUTO_ADJUST_ROUNDS: u32 = 2;
    const SKIP_RATIO_INCREASE: f64 = 0.25;
    const SKIP_RATIO_DECREASE: f64 = 0.125;
    const MAX_ITERATION_CAP: u32 = 10_000_000;

    let mut adjusted_params = params.clone();

    // Nucleus finder (Mandelbrot only, opt-in via `find_nucleus`). À deep
    // zoom escape-time, l'utilisateur cible un point près d'un minibrot dont
    // l'orbite escape avant `iteration_max`. Le finder localise la période
    // via le critère atom-domain F3 (`|z|² < s²·|dz|²`) puis raffine le centre
    // via Newton vers le centre exact du minibrot. Inspiré de `hybrid_period`
    // + `hybrid_center` de Fraktaler-3.1.
    if params.find_nucleus && matches!(params.fractal_type, FractalType::Mandelbrot) {
        // Précision dérivée de la formule perturbation (auto si span HP fournie).
        // À zoom 1e227, params.precision_bits=256 ne suffit pas pour itérer le
        // candidat près du minibrot exact ; il faut ~780 bits.
        let prec = crate::fractal::perturbation::compute_perturbation_precision_bits(
            &adjusted_params,
        );
        let cx = if let Some(ref s) = adjusted_params.center_x_hp {
            match rug::Float::parse(s) {
                Ok(p) => rug::Float::with_val(prec, p),
                Err(_) => rug::Float::with_val(prec, adjusted_params.center_x),
            }
        } else {
            rug::Float::with_val(prec, adjusted_params.center_x)
        };
        let cy = if let Some(ref s) = adjusted_params.center_y_hp {
            match rug::Float::parse(s) {
                Ok(p) => rug::Float::with_val(prec, p),
                Err(_) => rug::Float::with_val(prec, adjusted_params.center_y),
            }
        } else {
            rug::Float::with_val(prec, adjusted_params.center_y)
        };

        // Échelle de vue `s = max(span_x, span_y) / 2`. F3 utilise `r = 1/zoom`
        // (i.e. ~ span/2 puisque `zoom = 2/span`). On reconstruit via FloatExp
        // pour survivre aux underflows f64 (span < 1e-300 ⇒ params.span_x = 0).
        let view_scale_fexp = {
            let (sx, sy) = crate::fractal::perturbation::effective_spans_fexp(&adjusted_params);
            // max(sx, sy) via PartialOrd FloatExp
            if sx > sy { sx } else { sy }
        };
        // view_scale_fexp → Float : reconstruire via shifts pour préserver
        // l'exposant arbitraire (Float::with_val ne supporte que f64).
        let s_gmp = {
            let mut f = Float::with_val(prec, view_scale_fexp.mantissa);
            // Mantissa est dans [0.5, 1) ; on applique 2^(exponent-1) pour
            // recomposer la magnitude exacte.
            let e = view_scale_fexp.exponent.saturating_sub(1);
            if e >= 0 {
                f <<= e as u32;
            } else {
                f >>= (-e) as u32;
            }
            f
        };
        let t_nucleus = Instant::now();
        if let Some(result) = crate::fractal::perturbation::nucleus::find_nucleus(
            &cx, &cy, adjusted_params.iteration_max, &s_gmp, prec,
        ) {
            if perf {
                eprintln!(
                    "[NUCLEUS] found period={} after {} Newton steps in {:.3}s — re-centering",
                    result.period, result.newton_steps, t_nucleus.elapsed().as_secs_f64(),
                );
            }
            // Réécrit center_x_hp / center_y_hp et f64. La résolution exacte
            // du nucleus rend l'orbite référence non-évadée pour ces périodes.
            adjusted_params.center_x_hp = Some(result.center_x.to_string());
            adjusted_params.center_y_hp = Some(result.center_y.to_string());
            adjusted_params.center_x = result.center_x.to_f64();
            adjusted_params.center_y = result.center_y.to_f64();

            // P1.6.b — hybrid_size : extrait l'orientation du minibrot via la
            // matrice K (port de `fraktaler-3-3.1/src/hybrid.cc:544`). Pour les
            // minibrots non-axis-aligned (flake, olbaid*), la rotation extraite
            // de K aligne le frame de rendu sur celui du minibrot, sinon les
            // pixels échantillonnent à travers les branches voisines.
            let t_size = Instant::now();
            match crate::fractal::perturbation::nucleus::hybrid_size_mat2(
                &result.center_x,
                &result.center_y,
                result.period,
                prec,
            ) {
                Some(hs) => {
                    // F3 remplace `transform.rotate` par la matrice K (cf.
                    // engine.cc:208-212). On suit cette sémantique : la
                    // rotation user pré-existante est écrasée par celle dérivée
                    // du minibrot trouvé.
                    //
                    // On stocke K normalisée à det=1 dans `transform_k` :
                    // pour Mandelbrot conformal `K = R(θ)/β²`, donc K/sqrt|det K|
                    // = R(θ) (pure rotation, on conserve la zoom utilisateur
                    // au lieu de la multiplier implicitement par 1/β²). Pour
                    // les hybrides non-conformes, la normalisation préserve
                    // le skew/scale relatif sans modifier l'échelle absolue.
                    let new_rotation_deg = hs.rotation_degrees();
                    let det_k = hs.k[0] * hs.k[3] - hs.k[1] * hs.k[2];
                    let sqrt_abs_det = det_k.abs().sqrt();
                    let k_normalized = if sqrt_abs_det.is_finite() && sqrt_abs_det > 0.0 {
                        [
                            hs.k[0] / sqrt_abs_det,
                            hs.k[1] / sqrt_abs_det,
                            hs.k[2] / sqrt_abs_det,
                            hs.k[3] / sqrt_abs_det,
                        ]
                    } else {
                        // det = 0 : K dégénérée, on ne stocke rien — l'extraction
                        // de rotation seule sera utilisée comme fallback.
                        [f64::NAN, 0.0, 0.0, f64::NAN]
                    };
                    if new_rotation_deg.is_finite() {
                        let old = adjusted_params.rotation;
                        adjusted_params.rotation = new_rotation_deg;
                        if k_normalized.iter().all(|x| x.is_finite()) {
                            adjusted_params.transform_k = Some(k_normalized);
                        }
                        if perf {
                            eprintln!(
                                "[NUCLEUS] hybrid_size: rotation {:.3}° (was {:.3}°), K_norm=[{:.3}, {:.3}; {:.3}, {:.3}], canonical size mantissa={:.4} exp={} ({:.3}s)",
                                new_rotation_deg,
                                old,
                                k_normalized[0],
                                k_normalized[1],
                                k_normalized[2],
                                k_normalized[3],
                                hs.size.mantissa,
                                hs.size.exponent,
                                t_size.elapsed().as_secs_f64(),
                            );
                        }
                    } else if perf {
                        eprintln!(
                            "[NUCLEUS] hybrid_size returned non-finite rotation — keep current",
                        );
                    }
                }
                None => {
                    if perf {
                        eprintln!(
                            "[NUCLEUS] hybrid_size dégénéré (atome bord ou orbite escape) — rotation inchangée ({:.3}s)",
                            t_size.elapsed().as_secs_f64(),
                        );
                    }
                }
            }
        } else if perf {
            eprintln!(
                "[NUCLEUS] period not found or Newton did not converge (after {:.3}s) — keep original center",
                t_nucleus.elapsed().as_secs_f64(),
            );
        }
    }

    let pixel_count = (params.width as u64).saturating_mul(params.height as u64);
    let small_image = params.width.max(params.height) <= 512;
    let disable_series = match std::env::var("FRACTALL_PERTURB_DISABLE_SERIES") {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    };
    let force_series = match std::env::var("FRACTALL_PERTURB_FORCE_SERIES") {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    };
    let is_julia = params.fractal_type == FractalType::Julia;

    // La série (Taylor HO) n'est CONSOMMÉE que par : (a) l'auto-adjust
    // d'`iteration_max` (via `validated_skip`), et (b) le path perturbation
    // LEGACY (`compute_series_skip` par pixel). Le path bytecode (BLA mat2 +
    // rebase F3), qui rend Mandelbrot/Julia/BS/… par défaut, l'ignore
    // totalement. Donc si le rendu passe par bytecode ET que l'auto-adjust est
    // désactivé (parité F3 / `FRACTALL_NO_AUTO_ADJUST=1`), construire la série
    // est du travail MORT — 0.77 s sur dragon (orbite 5 M, interval 10 →
    // 500 k entrées order-4). On l'évite alors (zéro changement de sortie).
    let auto_adjust_enabled = std::env::var("FRACTALL_NO_AUTO_ADJUST")
        .ok()
        .map(|v| v != "1" && v.to_lowercase() != "true")
        .unwrap_or(true);
    let series_will_be_used =
        auto_adjust_enabled || !super::uses_bytecode_path(params);

    // GMP dense (`z_ref_gmp`) requis par : (1) le rendu principal full-GMP
    // (`render_perturbation_gmp_path` → `iterate_pixel_gmp` sur TOUS les pixels)
    // et (2) la correction glitch. On le stocke eager pour (1) et pour le path
    // legacy (non-bytecode) ; sur le path bytecode nominal (pas full-GMP) on le
    // SAUTE — la pass glitch le recompute à la demande si des pixels glitchent
    // (cf. `compute_reference_orbit` force_dense_gmp + mod.rs glitch pass).
    let store_dense_gmp = !super::uses_bytecode_path(params)
        || super::should_use_full_gmp_perturbation(params);

    // Compute orbit + BLA + series, potentially re-running with doubled iteration_max
    let t_orbit_first = Instant::now();
    let (mut orbit, mut center_x_gmp, mut center_y_gmp) =
        compute_reference_orbit(&adjusted_params, cancel, store_dense_gmp)?;
    let mut dt_orbit = t_orbit_first.elapsed();

    let t_bla_first = Instant::now();
    let mut bla_table = build_bla_table(&orbit.z_ref_f64, &adjusted_params, orbit.cref);
    let mut dt_bla = t_bla_first.elapsed();

    let mut series_table: Option<SeriesTable> = None;
    let mut dt_series = std::time::Duration::ZERO;

    for round in 0..=MAX_AUTO_ADJUST_ROUNDS {
        if round > 0 {
            // Recompute orbit + BLA with adjusted iteration_max
            let t_orbit_start = Instant::now();
            let result = compute_reference_orbit(&adjusted_params, cancel, store_dense_gmp)?;
            orbit = result.0;
            center_x_gmp = result.1;
            center_y_gmp = result.2;
            dt_orbit = t_orbit_start.elapsed();

            let t_bla_start = Instant::now();
            bla_table = build_bla_table(&orbit.z_ref_f64, &adjusted_params, orbit.cref);
            dt_bla = t_bla_start.elapsed();
        }

        // Build series table for standalone series approximation (if enabled).
        // `series_will_be_used` court-circuite le build mort sur le path bytecode
        // sans auto-adjust (cf. définition plus haut). `force_series` (debug)
        // reste prioritaire.
        let should_build_series = !disable_series
            && (force_series
                || (series_will_be_used
                    && adjusted_params.series_standalone
                    && matches!(adjusted_params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
                    && (!small_image || adjusted_params.iteration_max >= 5000)
                    && (pixel_count >= 16_384 || adjusted_params.iteration_max >= 10_000)));

        let t_series_start = Instant::now();
        series_table = if should_build_series {
            let pixel_size = (adjusted_params.span_x.abs() / adjusted_params.width.max(1) as f64)
                .max(adjusted_params.span_y.abs() / adjusted_params.height.max(1) as f64);
            let adaptive_order = compute_adaptive_series_order(
                pixel_size,
                adjusted_params.iteration_max,
                adjusted_params.series_order,
            );
            let series_order = adaptive_order.max(4);
            let interval = if orbit.z_ref_f64.len() > 100_000 { 10 } else { 1 };
            let mut table = build_series_table_ho(&orbit.z_ref_f64, is_julia, series_order, interval);

            if pixel_size > 0.0 && pixel_size.is_finite() {
                let tiled = validate_series_with_probes_tiled(
                    &table,
                    &orbit.z_ref_f64,
                    is_julia,
                    pixel_size,
                    adjusted_params.width as usize,
                    adjusted_params.height as usize,
                    4,
                );
                table.validated_skip = tiled.global_min;
                table.tiled_validation = Some(tiled);
            }

            Some(table)
        } else {
            None
        };
        dt_series = t_series_start.elapsed();

        // Auto-adjust iteration_max based on series skip ratio
        // (only on rounds before the last, and only if series is active).
        //
        // `auto_adjust_enabled` (hoisté plus haut, désactivable via
        // `FRACTALL_NO_AUTO_ADJUST=1`) : utile pour les tests de parité
        // Fraktaler-3 (F3 ne fait pas cet ajustement, donc l'avoir actif fait
        // diverger iter_max → mismatch sur les coins évadés tardivement). À
        // l'usage normal, l'auto-adjust révèle les détails masqués derrière un
        // iter_max trop bas.
        if round < MAX_AUTO_ADJUST_ROUNDS && auto_adjust_enabled {
            if let Some(ref table) = series_table {
                let skip = table.validated_skip as f64;
                let max_iter = adjusted_params.iteration_max as f64;
                if max_iter > 0.0 {
                    let skip_ratio = skip / max_iter;
                    if skip_ratio > SKIP_RATIO_INCREASE
                        && adjusted_params.iteration_max < MAX_ITERATION_CAP
                    {
                        // Series skips >25% of iterations: double and recompute
                        let new_max = (adjusted_params.iteration_max * 2).min(MAX_ITERATION_CAP);
                        if perf {
                            eprintln!(
                                "[PERTURB AUTO-ADJUST] round={} skip_ratio={:.1}% ({}/{}) → doubling iteration_max {} → {}",
                                round, skip_ratio * 100.0, table.validated_skip, adjusted_params.iteration_max,
                                adjusted_params.iteration_max, new_max,
                            );
                        }
                        adjusted_params.iteration_max = new_max;
                        continue; // Recompute with higher iteration_max
                    } else if skip_ratio < SKIP_RATIO_DECREASE
                        && adjusted_params.iteration_max > params.iteration_max
                    {
                        // Series skips <12.5%: iteration_max may be too high (only reduce
                        // back towards user's original value, never below it)
                        let new_max = adjusted_params.iteration_max
                            .min(adjusted_params.iteration_max / 2)
                            .max(params.iteration_max);
                        if new_max < adjusted_params.iteration_max {
                            if perf {
                                eprintln!(
                                    "[PERTURB AUTO-ADJUST] round={} skip_ratio={:.1}% → reducing iteration_max {} → {}",
                                    round, skip_ratio * 100.0, adjusted_params.iteration_max, new_max,
                                );
                            }
                            adjusted_params.iteration_max = new_max;
                            continue;
                        }
                    }
                }
            }
        }

        break; // No adjustment needed, use current results
    }

    // Build Hybrid BLA references (detect cycles and create references for each phase)
    // For Hybrid BLA: you need one BLA table per reference
    let t_hybrid = Instant::now();
    let hybrid_refs = build_hybrid_bla_references(&orbit, &bla_table, &adjusted_params, cancel);
    let dt_hybrid = t_hybrid.elapsed();

    if perf {
        let adjusted = if adjusted_params.iteration_max != params.iteration_max {
            format!(" (auto-adjusted from {})", params.iteration_max)
        } else {
            String::new()
        };
        eprintln!(
            "[PERTURB PERF] reference_cache=miss size={}x{} pixels={} small_image={} type={:?} prec={}b iters={}{} orbit={:.3}s bla={:.3}s series={:.3}s hybrid={:.3}s total={:.3}s",
            params.width,
            params.height,
            pixel_count,
            small_image,
            params.fractal_type,
            adjusted_params.precision_bits,
            adjusted_params.iteration_max,
            adjusted,
            dt_orbit.as_secs_f64(),
            dt_bla.as_secs_f64(),
            dt_series.as_secs_f64(),
            dt_hybrid.as_secs_f64(),
            t_all.elapsed().as_secs_f64(),
        );
    }

    Some(Arc::new(ReferenceOrbitCache::new(
        orbit,
        bla_table,
        series_table,
        &adjusted_params,
        center_x_gmp,
        center_y_gmp,
        hybrid_refs,
    )))
}

/// Calcule l'orbite de référence haute précision au centre de l'image.
///
/// L'orbite de référence `Z_m` est calculée en haute précision (GMP) au point central
/// de l'image. Cette orbite est ensuite utilisée comme référence pour calculer les
/// pixels environnants via la méthode de perturbation.
///
/// ## Initialisation selon le type de fractale
///
/// - **Mandelbrot/BurningShip/Multibrot/Tricorn**: `Z_0 = seed` (généralement 0)
/// - **Julia**: `Z_0 = C` (le point central de l'image)
///
/// ## Itération
///
/// L'orbite est calculée avec la formule standard de chaque fractale:
/// - Mandelbrot: `Z_{m+1} = Z_m² + C`
/// - Julia: `Z_{m+1} = Z_m² + seed`
/// - BurningShip: `Z_{m+1} = (|Re(Z_m)| + i|Im(Z_m)|)² + C`
/// - etc.
///
/// L'orbite est stockée à la fois en haute précision (`z_ref: Vec<ComplexExp>`) et
/// en f64 (`z_ref_f64: Vec<Complex64>`) pour optimiser les performances.
///
/// # Arguments
///
/// * `params` - Paramètres de la fractale
/// * `cancel` - Flag d'annulation (optionnel)
///
/// # Retour
///
/// Retourne `Some((orbit, center_x_gmp_string, center_y_gmp_string))` si le calcul réussit.
/// Les chaînes GMP sont utilisées pour la validation du cache.
pub fn compute_reference_orbit(
    params: &FractalParams,
    cancel: Option<&AtomicBool>,
    // Stocke l'orbite GMP dense (`z_ref_gmp` + `high_precision_data`), lue
    // UNIQUEMENT par la correction glitch perturbation-GMP (`iterate_pixel_gmp`)
    // et `new_reference_from`. Sur le path bytecode (BLA + rebase-at-end), aucun
    // glitch en régime nominal → stockage inutile (~0.7 s + 850 Mo sur dragon,
    // 5 M iters). Le caller met `false` sur ce path ; la pass glitch recompute
    // l'orbite dense localement (avec `true`) SEULEMENT si des pixels glitchent
    // — même chemin bytecode donc `z_ref_gmp` bit-identique à l'ancien eager.
    force_dense_gmp: bool,
) -> Option<(ReferenceOrbit, String, String)> {
    // Utiliser la précision calculée au lieu du preset
    use crate::fractal::perturbation::compute_perturbation_precision_bits;
    let prec = compute_perturbation_precision_bits(params);

    // Utiliser les String haute précision si disponibles, sinon fallback sur f64
    let center_x_gmp = if let Some(ref cx_hp) = params.center_x_hp {
        match Float::parse(cx_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_x_hp, using f64 fallback");
                Float::with_val(prec, params.center_x)
            }
        }
    } else {
        Float::with_val(prec, params.center_x)
    };
    
    let center_y_gmp = if let Some(ref cy_hp) = params.center_y_hp {
        match Float::parse(cy_hp) {
            Ok(parse_result) => Float::with_val(prec, parse_result),
            Err(_) => {
                eprintln!("[PRECISION WARNING] Failed to parse center_y_hp, using f64 fallback");
                Float::with_val(prec, params.center_y)
            }
        }
    } else {
        Float::with_val(prec, params.center_y)
    };
    
    // Référence = centre de vue (standard F3). L'ancien **auto-snap nucleus**
    // (`optimize_reference_center`, inspiré rust-fractal-core) snappait la réf vers
    // le nucleus voisin pour rendre l'orbite périodique (anti-glitch). DÉSACTIVÉ
    // (2026-05-22) : il produisait des rendus FAUX — la réf devenue périodique
    // déclenchait soit la troncation lossy (test2 @1920, mandelbrot_perturb_1e6 :
    // ~94 % de pixels ≠ GMP), soit un HANG sans troncation. Le rebase-at-end (G2)
    // gère déjà les références qui s'évadent, donc le bénéfice anti-glitch est
    // subsumé. `--find-nucleus` reste pour le raffinement EXPLICITE vers minibrot.
    // Réactivable via `FRACTALL_NUCLEUS_SNAP=1` (debug/expérimentation).
    let nucleus_snap_enabled = std::env::var("FRACTALL_NUCLEUS_SNAP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let (ref_center_x, ref_center_y) = if nucleus_snap_enabled
        && matches!(params.fractal_type, FractalType::Mandelbrot)
    {
        if let Some(nucleus) = optimize_reference_center(
            &center_x_gmp, &center_y_gmp,
            params.span_x, params.span_y,
            prec,
        ) {
            let nx = Float::with_val(prec, nucleus.real());
            let ny = Float::with_val(prec, nucleus.imag());
            eprintln!("[NUCLEUS] Reference snapped to nucleus at ({}, {})",
                nx.to_string_radix(10, Some(15)),
                ny.to_string_radix(10, Some(15)));
            (nx, ny)
        } else {
            (center_x_gmp.clone(), center_y_gmp.clone())
        }
    } else {
        (center_x_gmp.clone(), center_y_gmp.clone())
    };

    // Store GMP strings for cache validation (use original center, not nucleus)
    let cx_str = center_x_gmp.to_string_radix(10, None);
    let cy_str = center_y_gmp.to_string_radix(10, None);

    let cref = Complex::with_val(prec, (&ref_center_x, &ref_center_y));
    let cref_f64 = Complex64::new(ref_center_x.to_f64(), ref_center_y.to_f64());

    // ── Chemin double-double (Phase 1b) ────────────────────────────────────
    // Itère l'orbite référence en dd (~106 b, `dd_reference_orbit_mandelbrot`)
    // au lieu de GMP quand : Mandelbrot standard (seed=0), path bytecode
    // (`!force_dense_gmp` → z_ref_gmp paresseux), et requirement F3 **non clampé**
    // ≤ 96 b (marge sous les ~106 b de dd). `compute_perturbation_precision_bits`
    // clampe à ≥128 b donc ne peut pas servir de gate — on recalcule les bits
    // formule bruts. ~14× plus rapide que GMP sur cette tranche (~1e13–1e19).
    // Pas de period-detection ici (cycle_period=0) : le pixel loop gère les
    // références non-évadantes par rebase-at-end F3 (cf. G2). NON bit-identique
    // au path GMP (arrondi dd ≠ MPFR) → goldens mid-range régénérés.
    let dd_eligible = !force_dense_gmp
        && matches!(params.fractal_type, FractalType::Mandelbrot)
        && params.seed.re == 0.0
        && params.seed.im == 0.0
        && {
            let px = super::effective_pixel_size(params);
            px > 0.0 && px.is_finite() && {
                let log2_zoom = (4.0 / px).log2();
                let log2_h = (params.height as f64).max(1.0).log2();
                let formula_bits = 24.0 + (log2_zoom + log2_h).floor();
                (0.0..=96.0).contains(&formula_bits)
            }
        };
    if dd_eligible {
        let (z_ref, z_ref_f64) =
            dd_reference_orbit_mandelbrot(&cref, params.iteration_max, cancel)?;
        let extended_iterations: Vec<u32> = z_ref_f64
            .iter()
            .enumerate()
            .filter(|(_, z)| z.re.abs() < 1e-300 && z.im.abs() < 1e-300)
            .map(|(i, _)| i as u32)
            .collect();
        return Some((
            ReferenceOrbit {
                cref: cref_f64,
                z_ref,
                z_ref_f64,
                z_ref_gmp: Vec::new(), // paresseux (glitch → recompute GMP)
                cref_gmp: cref,
                phase_offset: 0,
                extended_iterations,
                high_precision_data: Vec::new(),
                data_storage_interval: 1,
                cycle_period: 0,
                cycle_start: 0,
            },
            cx_str,
            cy_str,
        ));
    }
    // ───────────────────────────────────────────────────────────────────────

    let mut z = match params.fractal_type {
        FractalType::Mandelbrot | FractalType::BurningShip | FractalType::Multibrot | FractalType::Tricorn => {
            Complex::with_val(prec, (params.seed.re, params.seed.im))
        }
        FractalType::Julia => cref.clone(),
        _ => return None,
    };
    let seed = Complex::with_val(prec, (params.seed.re, params.seed.im));

    // Path bytecode (P3.1) : si activé et type supporté, on remplace le match
    // per-type dans la boucle d'itération par un interpréteur unifié. La
    // constante `c` ajoutée par Op::Add est `seed` pour les variantes Julia,
    // `cref` sinon (cf. F3 `hybrid_reference`).
    let bytecode_formula: Option<Formula> = if params.use_bytecode_engine {
        compile_formula(params.fractal_type, params.multibrot_power)
    } else {
        None
    };
    let bytecode_c: Option<&Complex> = bytecode_formula.as_ref().map(|_| {
        if Formula::is_julia_for(params.fractal_type) {
            &seed
        } else {
            &cref
        }
    });
    let mut bytecode_state: Option<GmpInterpState> = bytecode_formula
        .as_ref()
        .map(|_| GmpInterpState::new(prec, z.clone()));

    // Reference orbit uses the F3-style fixed bailout (1e10 squared norm),
    // not the per-pixel bailout. See REFERENCE_BAILOUT_SQR doc.
    let bailout_sqr = Float::with_val(prec, REFERENCE_BAILOUT_SQR);

    let mut z_ref = Vec::with_capacity(params.iteration_max as usize + 1);
    let mut z_ref_f64 = Vec::with_capacity(params.iteration_max as usize + 1);
    let store_dense_gmp = force_dense_gmp;
    let mut z_ref_gmp =
        Vec::with_capacity(if store_dense_gmp { params.iteration_max as usize + 1 } else { 0 });

    // Inspired by rust-fractal-core's data_storage_interval:
    // Store high-precision orbit data at intervals to reduce memory usage.
    // For orbits with >100k iterations, store every 10th; for >1M, every 100th.
    let data_storage_interval = if params.iteration_max > 1_000_000 {
        100
    } else if params.iteration_max > 100_000 {
        10
    } else {
        1
    };
    let mut high_precision_data = Vec::with_capacity(
        if store_dense_gmp { (params.iteration_max as usize / data_storage_interval) + 2 } else { 0 }
    );

    // Store high-precision, f64, and full GMP versions (cf. `z_ref_complexexp`).
    let z_f64 = complex_to_complex64(&z);
    z_ref.push(z_ref_complexexp(&z, z_f64));
    z_ref_f64.push(z_f64);
    if store_dense_gmp {
        z_ref_gmp.push(z.clone());
        high_precision_data.push(z.clone());
    }

    // Period detection state for early termination (inspired by rust-fractal-core).
    //
    // rust-fractal-core detects when the reference orbit becomes periodic and stops
    // the orbit computation early. This saves enormous memory and time for orbits
    // that converge to an attracting cycle (interior points of the Mandelbrot set).
    //
    // Algorithm: Track z at previous "checkpoint" iterations. When |z_n - z_checkpoint|²
    // falls below a tolerance proportional to precision, a cycle is detected.
    // The tolerance is computed as 2^(-prec/2) to match the GMP precision level.
    //
    // Only for Mandelbrot (for Julia, every pixel has a different c, so the reference
    // orbit periodicity doesn't help).
    // Period-detection truncation est une optimisation : à la première
    // approche de l'orbite vers un checkpoint (Brent) sous tolérance, on
    // tronque la référence et on cycle via `wrap_periodic`. F3 ne tronque
    // jamais (calcule la référence complète). Sur les faux positifs
    // (orbite qui frôle un checkpoint sans être vraiment périodique), le
    // wrap diverge → image uniforme/escape erroné (cf. glitch_test_5 :
    // F3 intérieur, fractall escape uniforme à iter 844). Désactivable via
    // `FRACTALL_NO_PERIOD=1` pour la parité F3.
    let period_disabled_env = std::env::var("FRACTALL_NO_PERIOD")
        .ok()
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let enable_period_detection =
        matches!(params.fractal_type, FractalType::Mandelbrot) && !period_disabled_env;
    let period_tolerance = if enable_period_detection {
        // Tolérance ∝ précision GMP. **2^(-prec·0.85)** (et non 0.4 historique) :
        // une période GENUINE matche à ~2^(-prec) (à l'accumulation près,
        // ~2^(-prec+log2(cycle_start))), alors qu'un FAUX-POSITIF (l'orbite frôle
        // un checkpoint sans être périodique) ne matche qu'à ~2^(-prec·0.4). Le
        // 0.4 attrapait ces grazes → troncation + wrap → **image uniforme**
        // (cf. floral_fantasy 1.55e85, period 284 graze). 0.85 rejette les grazes
        // tout en gardant les vraies périodes. Les cas non-détectés retombent sur
        // le path escape-time + rebase-at-end (correct, cf. fix G2).
        let tol_exp = -(prec as f64) * 0.85;
        10.0f64.powf(tol_exp * 0.301) // 2^tol_exp ≈ 10^(tol_exp * log10(2))
    } else {
        0.0
    };
    let mut period_check_z = Complex::with_val(prec, (0, 0));
    let mut period_diff = Complex::with_val(prec, (0, 0));
    let period_tol_gmp = Float::with_val(prec, period_tolerance * period_tolerance);
    let mut period_check_iter = 0u32;
    let mut period_step = 1u32; // Floyd's cycle detection: double step size periodically
    let mut detected_period = 0u32;
    let mut detected_cycle_start = 0u32;

    // Reusable GMP scratch buffers to eliminate per-iteration allocations.
    // The hot orbit loop previously created 4-6 fresh Float/Complex values per
    // iteration (norm sqr components, z_prec clone, z_sq clone). For deep zooms
    // (170k+ iterations) that adds up to millions of redundant GMP allocations.
    let mut scratch = Complex::with_val(prec, (0, 0));
    let mut norm_re = Float::with_val(prec, 0);
    let mut norm_im = Float::with_val(prec, 0);
    let mut norm_sqr = Float::with_val(prec, 0);
    let ftype = params.fractal_type;
    let multibrot_power = params.multibrot_power;

    for i in 0..params.iteration_max {
        if let Some(cancel) = cancel {
            if i % 256 == 0 && cancel.load(Ordering::Relaxed) {
                return None;
            }
        }
        // Bailout check. Fast pré-check f64 avant la norme GMP (2 carrés 676 b +
        // add — ~la moitié du temps orbite sur les orbites BORNÉES, ex. dragon
        // qui tourne 5 M iters sans jamais s'évader → la norme GMP ne déclenche
        // JAMAIS mais coûte à chaque tour). L'orbite référence est bornée par
        // `REFERENCE_BAILOUT_SQR`=1e10 avant évasion ⇒ |z| < 1e5, exactement en
        // range f64. `z_ref_f64.last()` = f64 du z courant (stocké au tour
        // précédent). L'écart |round(z)|² vs |z|² est ~2^-51 relatif (~4e-6 à
        // 1e10) ≪ la marge de 1 % ⇒ si `f64_norm < 0.99·bailout`, la vraie norme
        // GMP est < bailout (aucune évasion ratée). Sinon (près de la frontière),
        // on calcule la norme GMP exacte → décision bit-identique.
        let z_curr_f64 = z_ref_f64[z_ref_f64.len() - 1];
        let f64_norm = z_curr_f64.re * z_curr_f64.re + z_curr_f64.im * z_curr_f64.im;
        if !(f64_norm < REFERENCE_BAILOUT_SQR * 0.99) {
            // Inline bailout check using scratch buffers (avoids 4 allocs per iter).
            norm_re.assign(z.real());
            norm_re.square_mut();
            norm_im.assign(z.imag());
            norm_im.square_mut();
            norm_sqr.assign(&norm_re);
            norm_sqr += &norm_im;
            if norm_sqr > bailout_sqr {
                break;
            }
        }

        // Period detection using Brent's algorithm variant (inspired by rust-fractal-core).
        // Compares z_n to a checkpoint value, doubling the step size when needed.
        // When |z_n - z_checkpoint| < tolerance, the orbit is periodic.
        //
        // NOTE (parité F3) : la troncation + `wrap_periodic` n'est exacte que si
        // l'orbite est VRAIMENT périodique. La tolérance resserrée (2^(-prec·0.85),
        // cf. plus haut) rejette les faux-positifs (grazes) qui produisaient des
        // images uniformes (ex. floral_fantasy, CORRIGÉ — golden `mandelbrot_floral`).
        // Les cas non-détectés retombent sur escape-time + rebase-at-end (correct,
        // fix G2). `FRACTALL_NO_PERIOD=1` désactive (parité harness). À terme :
        // ne tronquer qu'aux nucleus exacts (--find-nucleus), cf. TODO.
        if enable_period_detection && i > 0 {
            period_diff.assign(&z - &period_check_z);
            let diff_norm = complex_norm_sqr(&period_diff, prec);
            if diff_norm < period_tol_gmp && i > period_check_iter {
                detected_period = i - period_check_iter;
                detected_cycle_start = period_check_iter;
                // Stop the orbit: we've detected periodicity, meaning the center
                // is an interior point. The orbit will just repeat from here.
                break;
            }
            // Brent's algorithm: after period_step iterations, update checkpoint
            if i - period_check_iter >= period_step {
                period_check_z.assign(&z);
                period_check_iter = i;
                period_step *= 2;
            }
        }

        if let (Some(state), Some(formula), Some(c_phase)) =
            (bytecode_state.as_mut(), bytecode_formula.as_ref(), bytecode_c)
        {
            state.step(formula, c_phase);
            z.assign(&state.z);
        } else {
            // In-place iteration update using a scratch Complex. Previously each
            // match arm allocated 2-3 fresh Complex values per iteration.
            match ftype {
                FractalType::Mandelbrot => {
                    scratch.assign(&z);
                    z *= &scratch; // z = z * z
                    z += &cref; // z = z² + cref
                }
                FractalType::Julia => {
                    scratch.assign(&z);
                    z *= &scratch;
                    z += &seed;
                }
                FractalType::BurningShip => {
                    // z = (|Re(z)| + i|Im(z)|)² + cref
                    z.mut_real().abs_mut();
                    z.mut_imag().abs_mut();
                    scratch.assign(&z);
                    z *= &scratch;
                    z += &cref;
                }
                FractalType::Multibrot => {
                    let mut z_pow = pow_f64_mpc(&z, multibrot_power, prec);
                    z_pow += &cref;
                    z = z_pow;
                }
                FractalType::Tricorn => {
                    // z = conj(z)² + cref
                    z.conj_mut();
                    scratch.assign(&z);
                    z *= &scratch;
                    z += &cref;
                }
                _ => return None,
            }
        }
        // Store high-precision, f64, and full GMP versions.
        // `z_ref` (ComplexExp) : fast-path depuis le f64 déjà calculé quand z
        // tient dans le range f64 normal (cas dominant, z borné par l'escape
        // radius) — évite les 2 clones GMP 676 b de `from_gmp`. Fallback
        // `from_gmp` seulement si une composante underflow (z proche de 0) où
        // l'exposant étendu doit être préservé. Bit-identique (cf.
        // `z_ref_complexexp`).
        let z_f64 = complex_to_complex64(&z);
        z_ref.push(z_ref_complexexp(&z, z_f64));
        z_ref_f64.push(z_f64);
        // GMP dense sauté hors correction glitch (cf. `store_dense_gmp`).
        if store_dense_gmp {
            z_ref_gmp.push(z.clone());
            // Store sparse high-precision data at intervals (inspired by rust-fractal-core)
            let iter_num = (i + 1) as usize;
            if data_storage_interval == 1 || iter_num % data_storage_interval == 0 {
                high_precision_data.push(z.clone());
            }
        }
    }

    if detected_period > 0 && crate::fractal::perturbation::perf_enabled() {
        eprintln!("[PERIOD] Detected period {} at iteration {} (orbit len={}). Center is interior.",
            detected_period, z_ref_f64.len(), z_ref_f64.len());
    }

    // Track iterations where z_ref is very small (near f64 underflow).
    // At these iterations, the f64 z_ref values lose precision, so the fast f64
    // batch path should fall back to extended precision arithmetic.
    let extended_iterations: Vec<u32> = z_ref_f64
        .iter()
        .enumerate()
        .filter(|(_, z)| z.re.abs() < 1e-300 && z.im.abs() < 1e-300)
        .map(|(i, _)| i as u32)
        .collect();

    if std::env::var("FRACTALL_DEBUG_ORBIT").as_deref() == Ok("1") {
        let len = z_ref_f64.len();
        let max_norm = z_ref_f64.iter().map(|z| z.norm_sqr()).fold(0.0f64, f64::max);
        let first_over_pixel_er = z_ref_f64
            .iter()
            .position(|z| z.norm_sqr() > 16.0)
            .map(|i| i as i64)
            .unwrap_or(-1);
        eprintln!(
            "[DEBUG_ORBIT] len={} max_|z|²={:.3e} first_|z|²>16 at iter={}",
            len, max_norm, first_over_pixel_er
        );
    }
    Some((
        ReferenceOrbit {
            cref: cref_f64,
            z_ref,
            z_ref_f64,
            z_ref_gmp,
            cref_gmp: cref,
            phase_offset: 0,
            extended_iterations,
            high_precision_data,
            data_storage_interval,
            cycle_period: detected_period,
            cycle_start: detected_cycle_start,
        },
        cx_str,
        cy_str,
    ))
}

fn complex_norm_sqr(value: &Complex, prec: u32) -> Float {
    // IMPORTANT: Créer des copies avec la précision explicite pour éviter la perte de précision
    // lors de la conversion Float -> f64 -> Float
    let re_prec = Float::with_val(prec, value.real());
    let im_prec = Float::with_val(prec, value.imag());

    let mut re2 = re_prec.clone();
    re2 *= &re_prec;
    let mut im2 = im_prec.clone();
    im2 *= &im_prec;

    let mut sum = Float::with_val(prec, &re2);
    sum += &im2;
    sum
}

/// Newton-Raphson nucleus finding for Mandelbrot set reference point optimization.
///
/// Inspired by rust-fractal-core's `root_finding.rs` module, which implements
/// Newton-Raphson iteration to find the nearest nucleus (periodic point) of the
/// Mandelbrot set. Placing the reference orbit at a nucleus reduces glitches
/// because the orbit is exactly periodic, making the perturbation formula more
/// stable for surrounding pixels.
///
/// The algorithm finds c such that f^p(0, c) = 0 where f(z,c) = z² + c
/// and p is the period. The Newton step is:
///   c_new = c - f^p(0, c) / (df^p/dc)(0, c)
///
/// Returns the refined center if convergence is achieved, None otherwise.
pub fn find_nucleus(
    center: &Complex,
    period: u32,
    prec: u32,
    max_newton_iters: u32,
) -> Option<Complex> {
    if period == 0 {
        return None;
    }

    let mut c = Complex::with_val(prec, (center.real(), center.imag()));
    let epsilon_sqr = {
        // Convergence threshold: 2^(-prec * 0.8) squared
        let exp = -(prec as f64) * 0.8;
        let eps = 2.0f64.powf(exp);
        Float::with_val(prec, eps * eps)
    };

    for _ in 0..max_newton_iters {
        // Iterate z → z² + c for `period` steps, accumulating df/dc
        let mut z = Complex::with_val(prec, (0, 0));
        let mut dz_dc = Complex::with_val(prec, (0, 0));

        for _ in 0..period {
            // dz_dc = 2*z*dz_dc + 1
            let mut two_z = Complex::with_val(prec, (z.real(), z.imag()));
            *two_z.mut_real() *= 2;
            *two_z.mut_imag() *= 2;
            dz_dc *= &two_z;
            *dz_dc.mut_real() += 1;

            // z = z² + c
            let z_clone = z.clone();
            z *= &z_clone;
            z += &c;
        }

        // Newton step: c_new = c - z / dz_dc
        let dz_dc_norm = complex_norm_sqr(&dz_dc, prec);
        let dz_dc_norm_f64 = dz_dc_norm.to_f64();
        if dz_dc_norm_f64 < 1e-300 {
            // Derivative too small, can't converge
            return None;
        }

        // Compute z / dz_dc using conjugate division
        let dz_dc_conj = dz_dc.clone().conj();
        let mut quotient = Complex::with_val(prec, &z * &dz_dc_conj);
        *quotient.mut_real() /= &dz_dc_norm;
        *quotient.mut_imag() /= &dz_dc_norm;

        // c = c - quotient
        c -= &quotient;

        // Check convergence: |quotient|² < epsilon²
        let step_norm = complex_norm_sqr(&quotient, prec);
        if step_norm < epsilon_sqr {
            return Some(c);
        }
    }

    None
}

/// Detect the period of the nearest nucleus to the given center point.
///
/// Inspired by rust-fractal-core's approach: iterate z → z² + c and look for
/// near-zero |z| values. The first iteration where |z| is very small likely
/// indicates the period of the nearest nucleus.
///
/// Returns the detected period, or None if no period is found within max_period.
pub fn detect_nucleus_period(
    center: &Complex,
    prec: u32,
    max_period: u32,
) -> Option<u32> {
    let mut z = Complex::with_val(prec, (0, 0));
    let c = Complex::with_val(prec, (center.real(), center.imag()));

    // Track the minimum |z|² and its iteration
    let mut min_norm = Float::with_val(prec, f64::MAX);
    let mut min_period = 0u32;

    for i in 1..=max_period {
        let z_clone = z.clone();
        z *= &z_clone;
        z += &c;

        let norm = complex_norm_sqr(&z, prec);
        if norm < min_norm {
            min_norm = norm.clone();
            min_period = i;
        }

        // If |z|² is extremely small, we've found the period
        let norm_f64 = norm.to_f64();
        if norm_f64 < 1e-6 {
            return Some(i);
        }
    }

    // Return the period with minimum |z| if it's reasonably small
    let min_f64 = min_norm.to_f64();
    if min_f64 < 1.0 {
        Some(min_period)
    } else {
        None
    }
}

/// Try to optimize the reference point by finding the nearest Mandelbrot nucleus.
///
/// Inspired by rust-fractal-core's root_finding module. This function:
/// 1. Detects the likely period of the nearest nucleus
/// 2. Uses Newton-Raphson to refine the center to the exact nucleus
/// 3. Validates that the refined point is within the pixel's visible area
///
/// Returns the optimized center point, or None if optimization fails or
/// the nucleus is too far from the original center.
pub fn optimize_reference_center(
    center_x: &Float,
    center_y: &Float,
    span_x: f64,
    span_y: f64,
    prec: u32,
) -> Option<Complex> {
    let center = Complex::with_val(prec, (center_x, center_y));

    // Detect period (limit search to reasonable range)
    let max_period = 1000u32;
    let period = detect_nucleus_period(&center, prec, max_period)?;

    // Find the nucleus using Newton-Raphson
    let nucleus = find_nucleus(&center, period, prec, 64)?;

    // Validate: the nucleus should be within the visible area
    // (within half the span from the original center)
    let dx = {
        let mut d = Float::with_val(prec, nucleus.real());
        d -= center_x;
        d.to_f64().abs()
    };
    let dy = {
        let mut d = Float::with_val(prec, nucleus.imag());
        d -= center_y;
        d.to_f64().abs()
    };

    // Only use the nucleus if it's within the visible area
    let max_offset_x = span_x.abs() * 0.4;
    let max_offset_y = span_y.abs() * 0.4;
    if dx <= max_offset_x && dy <= max_offset_y {
        Some(nucleus)
    } else {
        None
    }
}

#[cfg(test)]
mod dd_orbit_tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    /// Verrou Phase 1a : l'orbite référence itérée en double-double (~106 b) doit
    /// matcher celle itérée en GMP (676 b) à la précision de stockage f64, sur un
    /// zoom moyen où dd est suffisant. Valide l'arithmétique dd avant le câblage.
    #[test]
    fn dd_orbit_matches_gmp_midrange() {
        let mut params = default_params_for_type(FractalType::Mandelbrot, 64, 64);
        // Seahorse valley : orbite longue, non triviale.
        params.center_x = -0.743643887037158;
        params.center_y = 0.131825904205330;
        params.span_x = 1e-12;
        params.span_y = 1e-12;
        params.iteration_max = 3000;

        let (orbit_gmp, _, _) =
            compute_reference_orbit(&params, None, true).expect("orbite GMP");
        let (_, dd_f64) =
            dd_reference_orbit_mandelbrot(&orbit_gmp.cref_gmp, params.iteration_max, None)
                .expect("orbite dd");

        // Préfixe commun (period-detection GMP peut tronquer plus tôt que dd).
        let n = orbit_gmp.z_ref_f64.len().min(dd_f64.len());
        assert!(n > 100, "préfixe commun trop court: {n}");
        let mut max_err = 0.0f64;
        for i in 0..n {
            let g = orbit_gmp.z_ref_f64[i];
            let d = dd_f64[i];
            max_err = max_err.max((g.re - d.re).abs()).max((g.im - d.im).abs());
        }
        assert!(
            max_err < 1e-9,
            "orbite dd diverge de GMP: max_err={max_err:.3e} sur {n} points"
        );
    }
}
