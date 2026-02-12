use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use rug::{Complex, Float};

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::gmp::{complex_to_complex64, pow_f64_mpc};
use crate::fractal::perturbation::bla::{BlaTable, build_bla_table};
use crate::fractal::perturbation::types::ComplexExp;
use crate::fractal::perturbation::series::{SeriesTable, build_series_table};

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
    /// Create a single-reference HybridBlaReferences (no cycle detected)
    pub fn single(orbit: ReferenceOrbit, bla_table: BlaTable) -> Self {
        Self {
            primary: orbit,
            primary_bla: bla_table,
            phases: Vec::new(),
            phase_bla_tables: Vec::new(),
            cycle_period: 0,
            cycle_start: 0,
        }
    }

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
        self.fractal_type == params.fractal_type
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
    // Detect cycle in the reference orbit
    // Use a tolerance based on the precision and zoom level
    let pixel_size = params.span_x / params.width as f64;
    let tolerance = (pixel_size * 1e-3).max(1e-10).min(1e-6);
    
    if let Some((cycle_start, cycle_period)) = detect_cycle(&primary_orbit.z_ref_f64, tolerance) {
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
        // No cycle detected: single reference
        Some(HybridBlaReferences::single(primary_orbit.clone(), primary_bla.clone()))
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

    // Compute fresh orbit and BLA table
    let t_orbit = Instant::now();
    let (orbit, center_x_gmp, center_y_gmp) = compute_reference_orbit(params, cancel)?;
    let dt_orbit = t_orbit.elapsed();

    // Use z_ref_f64 for BLA table building (BLA works with f64 coefficients)
    let t_bla = Instant::now();
    let bla_table = build_bla_table(&orbit.z_ref_f64, params, orbit.cref);
    let dt_bla = t_bla.elapsed();

    // Build series table for standalone series approximation (if enabled)
    // Only for Mandelbrot and Julia; Burning Ship has abs() which breaks series
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
    let should_build_series = !disable_series
        && (force_series
            || (params.series_standalone
                && matches!(params.fractal_type, FractalType::Mandelbrot | FractalType::Julia)
                // Heuristique: sur petites images, éviter de payer un coût fixe si iters faibles
                && (!small_image || params.iteration_max >= 5000)
                && (pixel_count >= 16_384 || params.iteration_max >= 10_000)));

    let t_series = Instant::now();
    let is_julia = params.fractal_type == FractalType::Julia;
    let series_table = if should_build_series {
        Some(build_series_table(&orbit.z_ref_f64, is_julia))
    } else {
        None
    };
    let dt_series = t_series.elapsed();

    // Build Hybrid BLA references (detect cycles and create references for each phase)
    // For Hybrid BLA: you need one BLA table per reference
    let t_hybrid = Instant::now();
    let hybrid_refs = build_hybrid_bla_references(&orbit, &bla_table, params, cancel);
    let dt_hybrid = t_hybrid.elapsed();

    if perf {
        eprintln!(
            "[PERTURB PERF] reference_cache=miss size={}x{} pixels={} small_image={} type={:?} iters={} orbit={:.3}s bla={:.3}s series={:.3}s hybrid={:.3}s total={:.3}s",
            params.width,
            params.height,
            pixel_count,
            small_image,
            params.fractal_type,
            params.iteration_max,
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
        params,
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
    
    // Log de diagnostic pour zoom profond (une seule fois)
    let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
    if pixel_size < 1e-15 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static LAST_LOGGED_ORBIT: AtomicU64 = AtomicU64::new(0);
        let log_key = (prec as u64) << 32 | (params.center_x_hp.is_some() as u64);
        let last_logged = LAST_LOGGED_ORBIT.load(Ordering::Relaxed);
        if log_key != last_logged {
            LAST_LOGGED_ORBIT.store(log_key, Ordering::Relaxed);
            eprintln!("[PRECISION DEBUG] compute_reference_orbit: prec={}, using_hp={}, center_x f64={:.20e}, center_y f64={:.20e}",
                prec, params.center_x_hp.is_some(), params.center_x, params.center_y);
            eprintln!("[PRECISION DEBUG] center_x_gmp={}, center_y_gmp={}",
                center_x_gmp.to_string_radix(10, Some(30)), center_y_gmp.to_string_radix(10, Some(30)));
        }
    }

    // Store GMP strings for cache validation
    let cx_str = center_x_gmp.to_string_radix(10, None);
    let cy_str = center_y_gmp.to_string_radix(10, None);

    let cref = Complex::with_val(prec, (&center_x_gmp, &center_y_gmp));
    let cref_f64 = Complex64::new(center_x_gmp.to_f64(), center_y_gmp.to_f64());

    let mut z = match params.fractal_type {
        FractalType::Mandelbrot | FractalType::BurningShip | FractalType::Multibrot | FractalType::Tricorn => {
            Complex::with_val(prec, (params.seed.re, params.seed.im))
        }
        FractalType::Julia => cref.clone(),
        _ => return None,
    };
    let seed = Complex::with_val(prec, (params.seed.re, params.seed.im));

    let bailout = Float::with_val(prec, params.bailout);
    let mut bailout_sqr = bailout.clone();
    bailout_sqr *= &bailout;

    let mut z_ref = Vec::with_capacity(params.iteration_max as usize + 1);
    let mut z_ref_f64 = Vec::with_capacity(params.iteration_max as usize + 1);
    let mut z_ref_gmp = Vec::with_capacity(params.iteration_max as usize + 1);
    // Store high-precision, f64, and full GMP versions
    z_ref.push(ComplexExp::from_gmp(&z));
    z_ref_f64.push(complex_to_complex64(&z));
    z_ref_gmp.push(z.clone());

    for i in 0..params.iteration_max {
        if let Some(cancel) = cancel {
            if i % 256 == 0 && cancel.load(Ordering::Relaxed) {
                return None;
            }
        }
        if complex_norm_sqr(&z, prec) > bailout_sqr {
            break;
        }
        z = match params.fractal_type {
            FractalType::Mandelbrot => {
                // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
                // Créer une copie avec la précision explicite pour garantir la cohérence
                let z_prec = Complex::with_val(prec, (z.real(), z.imag()));
                let mut z_sq = z_prec.clone();
                z_sq *= &z_prec;
                z_sq += &cref;
                z_sq
            }
            FractalType::Julia => {
                // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
                // Créer une copie avec la précision explicite pour garantir la cohérence
                let z_prec = Complex::with_val(prec, (z.real(), z.imag()));
                let mut z_sq = z_prec.clone();
                z_sq *= &z_prec;
                z_sq += &seed;
                z_sq
            }
            FractalType::BurningShip => {
                // IMPORTANT: Créer des copies avec la précision explicite pour éviter la perte de précision
                // lors de la conversion Float -> f64 -> Float
                let re_prec = Float::with_val(prec, z.real());
                let im_prec = Float::with_val(prec, z.imag());
                let re_abs = re_prec.abs();
                let im_abs = im_prec.abs();
                let mut z_abs = Complex::with_val(prec, (re_abs, im_abs));
                z_abs *= z_abs.clone();
                z_abs += &cref;
                z_abs
            }
            FractalType::Multibrot => {
                // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
                // pow_f64_mpc crée déjà une nouvelle valeur avec la précision prec, donc c'est OK
                let mut z_pow = pow_f64_mpc(&z, params.multibrot_power, prec);
                z_pow += &cref;
                z_pow
            }
            FractalType::Tricorn => {
                // Tricorn: z' = conj(z)² + c
                // IMPORTANT: S'assurer que toutes les opérations utilisent la même précision prec
                // Créer une copie avec la précision explicite pour garantir la cohérence
                let z_prec = Complex::with_val(prec, (z.real(), z.imag()));
                let z_conj = z_prec.conj();
                let mut z_temp = Complex::with_val(prec, (z_conj.real(), z_conj.imag()));
                z_temp *= &z_conj;
                z_temp += &cref;
                z_temp
            }
            _ => return None,
        };
        // Store high-precision, f64, and full GMP versions
        z_ref.push(ComplexExp::from_gmp(&z));
        z_ref_f64.push(complex_to_complex64(&z));
        z_ref_gmp.push(z.clone());
    }

    Some((
        ReferenceOrbit {
            cref: cref_f64,
            z_ref,
            z_ref_f64,
            z_ref_gmp,
            cref_gmp: cref,
            phase_offset: 0, // Primary reference starts at phase 0
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
