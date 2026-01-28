use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalType};
use crate::fractal::perturbation::nonconformal::{
    compute_tricorn_bla_coefficients,
    compute_nonconformal_validity_radius,
    merge_nonconformal_bla,
    Matrix2x2,
};

#[derive(Clone, Copy, Debug)]
pub struct BlaNode {
    /// Coefficient linéaire: δ' = A·δ + ...
    pub a: Complex64,
    /// Coefficient dc: δ' = ... + B·dc + ...
    pub b: Complex64,
    /// Coefficient quadratique (ordre 2): δ' = ... + C·δ² + ...
    pub c: Complex64,
    /// Coefficient cubique (ordre 3): δ' = ... + D·δ³ + ...
    /// Pour Mandelbrot z²+c, D = 0. Utile pour Multibrot z^d+c.
    pub d: Complex64,
    /// Coefficient quartique (ordre 4): δ' = ... + E·δ⁴ + ...
    /// Pour Mandelbrot z²+c, E = 0. Utile pour Multibrot z^d+c.
    pub e: Complex64,
    pub validity_radius: f64,
    /// Pour Burning Ship: indique si le BLA est valide (z_ref reste dans le même quadrant)
    pub burning_ship_valid: bool,
    /// Pour Burning Ship optimisé: signe du quadrant Re (1.0 si Re >= 0, -1.0 sinon)
    pub sign_re: f64,
    /// Pour Burning Ship optimisé: signe du quadrant Im (1.0 si Im >= 0, -1.0 sinon)
    pub sign_im: f64,
}

/// Nœud BLA non-conforme pour formules non-analytiques (comme Tricorn).
/// Utilise des matrices 2×2 au lieu de nombres complexes pour les coefficients.
#[derive(Clone, Copy, Debug)]
pub struct BlaNodeNonConformal {
    /// Coefficient linéaire A (matrice 2×2)
    pub a: Matrix2x2,
    /// Coefficient dc B (matrice 2×2)
    pub b: Matrix2x2,
    /// Rayon de validité pour ce BLA
    pub validity_radius: f64,
}

#[derive(Clone, Debug)]
pub struct BlaTable {
    pub levels: Vec<Vec<BlaNode>>,
    /// Table BLA non-conforme pour formules non-analytiques (optionnelle)
    pub nonconformal_levels: Option<Vec<Vec<BlaNodeNonConformal>>>,
}

impl BlaTable {
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            nonconformal_levels: None,
        }
    }
}

/// Vérifie si le BLA est valide pour Burning Ship sur une plage d'itérations.
/// Le BLA est valide si z_ref ne change pas de quadrant (signe de Re et Im) pendant les steps.
/// 
/// # Arguments
/// * `z_ref` - L'orbite de référence
/// * `n` - L'indice de départ
/// * `steps` - Le nombre d'itérations à vérifier
/// 
/// # Returns
/// `true` si z_ref reste dans le même quadrant pendant les steps
fn burning_ship_bla_validity(z_ref: &[Complex64], n: usize, steps: usize) -> bool {
    if n >= z_ref.len() {
        return false;
    }
    
    let z0 = z_ref[n];
    let sign_re = z0.re >= 0.0;
    let sign_im = z0.im >= 0.0;
    
    // Vérifier que z_ref reste dans le même quadrant
    for i in 1..=steps.min(z_ref.len().saturating_sub(n).saturating_sub(1)) {
        let z = z_ref[n + i];
        if (z.re >= 0.0) != sign_re || (z.im >= 0.0) != sign_im {
            return false;
        }
    }
    
    true
}

/// Calcule les coefficients BLA pour Burning Ship.
/// Pour Burning Ship: z' = (|Re(z)|, |Im(z)|)² + c
/// Si z reste dans le même quadrant, la dérivée est similaire à Mandelbrot
/// avec un facteur de signe pour Re et Im.
fn compute_burning_ship_bla_coefficients(
    z: Complex64,
    quadrant_stable: bool,
) -> BlaCoefficients {
    let zero = Complex64::new(0.0, 0.0);
    
    if !quadrant_stable {
        // Si le quadrant n'est pas stable, les coefficients ne sont pas utilisés
        return BlaCoefficients {
            a: zero,
            b: Complex64::new(1.0, 0.0),
            c: zero,
            d: zero,
            e: zero,
        };
    }
    
    // Pour Burning Ship avec quadrant stable:
    // z' = (|Re|, |Im|)² + c
    // En termes de perturbation, si z_ref + δ reste dans le même quadrant:
    // (|Re(z_ref + δ)|, |Im(z_ref + δ)|) ≈ (|Re(z_ref)| + sign(Re)*δ_re, |Im(z_ref)| + sign(Im)*δ_im)
    // La dérivée devient: 2 * (sign_re * |Re|, sign_im * |Im|) * δ
    
    let sign_re = if z.re >= 0.0 { 1.0 } else { -1.0 };
    let sign_im = if z.im >= 0.0 { 1.0 } else { -1.0 };
    
    // z_abs = (|Re|, |Im|)
    let re_abs = z.re.abs();
    let im_abs = z.im.abs();
    
    // Pour z_abs² + c, la dérivée par rapport à δ est:
    // d(z_abs²)/dδ = 2 * z_abs * d(z_abs)/dδ
    // où d(z_abs)/dδ = (sign_re, sign_im) composante par composante
    // Donc A = 2 * (sign_re * |Re| + i * sign_im * |Im|)
    let a = Complex64::new(2.0 * sign_re * re_abs, 2.0 * sign_im * im_abs);
    let b = Complex64::new(1.0, 0.0);
    let c = Complex64::new(1.0, 0.0);  // Terme quadratique
    
    // Pour Burning Ship, les termes d'ordre supérieur sont nuls (comme Mandelbrot z²)
    BlaCoefficients { a, b, c, d: zero, e: zero }
}

/// Coefficients BLA pour une itération
struct BlaCoefficients {
    a: Complex64,  // Linéaire
    b: Complex64,  // dc
    c: Complex64,  // δ²
    d: Complex64,  // δ³
    e: Complex64,  // δ⁴
}

/// Compute BLA coefficients (A_{n,1}, B_{n,1}, C, D, E) for a single iteration at z_ref.
/// For z^d + c iteration: δ' = A_{n,1}·δ + B_{n,1}·dc + C·δ² + D·δ³ + E·δ⁴
/// where:
/// - A_{n,1} = d·z^(d-1) (for single step, A_{n,1} = 2·Z_n for Mandelbrot)
/// - B_{n,1} = 1 (Mandelbrot) or 0 (Julia)
/// - C = d(d-1)/2·z^(d-2)
/// - D = d(d-1)(d-2)/6·z^(d-3)  (pour Multibrot, 0 pour Mandelbrot)
/// - E = d(d-1)(d-2)(d-3)/24·z^(d-4)  (pour Multibrot, 0 pour Mandelbrot)
fn compute_bla_coefficients(
    z: Complex64,
    fractal_type: FractalType,
    power: f64,
) -> BlaCoefficients {
    let is_julia = fractal_type == FractalType::Julia;
    let zero = Complex64::new(0.0, 0.0);

    match fractal_type {
        FractalType::Mandelbrot | FractalType::Julia => {
            // Standard z² + c: A_{n,1} = 2·Z_n, C = 1, D = E = 0
            let a = z * 2.0;  // A_{n,1} = 2·Z_n
            let b = if is_julia { zero } else { Complex64::new(1.0, 0.0) };
            let c = Complex64::new(1.0, 0.0);
            // Pour z², les termes d'ordre supérieur sont nuls
            BlaCoefficients { a, b, c, d: zero, e: zero }
        }
        FractalType::Multibrot => {
            // z^d + c: calcul des coefficients jusqu'à l'ordre 4
            let d_pow = power;
            let z_norm = z.norm();

            // For very small z, use simplified coefficients
            if z_norm < 1e-15 {
                return BlaCoefficients {
                    a: zero,
                    b: Complex64::new(1.0, 0.0),
                    c: zero,
                    d: zero,
                    e: zero,
                };
            }

            // A = d · z^(d-1)
            let a = z.powf(d_pow - 1.0) * d_pow;
            let b = Complex64::new(1.0, 0.0);

            // C = d(d-1)/2 · z^(d-2)
            let c_coeff = d_pow * (d_pow - 1.0) / 2.0;
            let c = if (d_pow - 2.0).abs() < 1e-10 {
                Complex64::new(c_coeff, 0.0)
            } else {
                z.powf(d_pow - 2.0) * c_coeff
            };

            // D = d(d-1)(d-2)/6 · z^(d-3)
            let d_coeff = d_pow * (d_pow - 1.0) * (d_pow - 2.0) / 6.0;
            let d_term = if d_pow > 2.5 && z_norm > 1e-10 {
                z.powf(d_pow - 3.0) * d_coeff
            } else {
                zero
            };

            // E = d(d-1)(d-2)(d-3)/24 · z^(d-4)
            let e_coeff = d_pow * (d_pow - 1.0) * (d_pow - 2.0) * (d_pow - 3.0) / 24.0;
            let e_term = if d_pow > 3.5 && z_norm > 1e-10 {
                z.powf(d_pow - 4.0) * e_coeff
            } else {
                zero
            };

            // Validate coefficients
            let a = if a.re.is_finite() && a.im.is_finite() { a } else { z * 2.0 };
            let c = if c.re.is_finite() && c.im.is_finite() { c } else { Complex64::new(1.0, 0.0) };
            let d_term = if d_term.re.is_finite() && d_term.im.is_finite() { d_term } else { zero };
            let e_term = if e_term.re.is_finite() && e_term.im.is_finite() { e_term } else { zero };

            BlaCoefficients { a, b, c, d: d_term, e: e_term }
        }
        _ => {
            // Fallback: use Mandelbrot coefficients
            let a = z * 2.0;
            let b = Complex64::new(1.0, 0.0);
            let c = Complex64::new(1.0, 0.0);
            BlaCoefficients { a, b, c, d: zero, e: zero }
        }
    }
}

/// Build BLA table from the f64 reference orbit.
///
/// # Bivariate Linear Approximation
///
/// Sometimes, `l` iterations starting at `n` can be approximated by bivariate linear function:
/// `z_{n+l} = A_{n,l}·z_n + B_{n,l}·c`
///
/// # Note on Deep Zooms
///
/// For very deep zooms (>10^15), the f64 precision of the reference orbit and BLA coefficients
/// becomes insufficient. In such cases, BLA should be disabled and full GMP perturbation used instead.
/// This function will return an empty BLA table if the zoom is too deep for f64 precision.
///
/// This is valid when the non-linear part of the full perturbation iterations is so small that
/// omitting it would cause fewer problems than the rounding error of the low precision data type.
///
/// # Single Step BLA
///
/// Approximation of a single step by bilinear form is valid when:
/// ```
/// |z_n²| << |2·Z_n·z_n + c|
/// ⇑ assume negligibility of c << |2·Z_n·z_n|
/// ⇑ factor out z_n
/// |z_n| << |2·Z_n|
/// ⇑ definition of A_{n,1}, B_{n,1} for single step
/// |z_n| << |A_{n,1}| =: R_{n,1}
/// ```
///
/// # BLA Table Construction
///
/// This style of table construction is suboptimal according to Zhuoran.
///
/// Suppose the reference has `M` iterations. Create `M` BLAs each skipping 1 iteration
/// (this can be done in parallel). Then merge neighbours without overlap to create `⌈M/2⌉`
/// each skipping 2 iterations (except for perhaps the last which skips less). Repeat until
/// there is only 1 BLA skipping `M-1` iterations: it's best to start the merge from iteration 1
/// because reference iteration 0 always corresponds to a non-linear perturbation step as `Z=0`.
///
/// The resulting table has `O(M)` elements.
///
/// Note: Use z_ref_f64 from ReferenceOrbit, not z_ref (which is Vec<ComplexExp>).
pub fn build_bla_table(ref_orbit: &[Complex64], params: &FractalParams) -> BlaTable {
    // Note: Tricorn support in BLA would require non-conformal matrices (see nonconformal.rs)
    // For now, Tricorn uses perturbation without BLA acceleration
    let supports_bla = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::Multibrot | FractalType::BurningShip
    );
    if !supports_bla {
        return BlaTable::empty();
    }

    // M = number of iterations in reference orbit (minus 1, as we don't need BLA for last iteration)
    let base_len = ref_orbit.len().saturating_sub(1);  // M
    if base_len == 0 {
        return BlaTable::empty();
    }

    // Check if zoom is too deep for f64-based BLA
    // For pixel_size < 1e-15 (zoom > 10^15), f64 precision is insufficient
    // Disable BLA and use full GMP perturbation instead
    if params.width > 0 && params.height > 0 {
        let pixel_size = params.span_x.abs().max(params.span_y.abs()) / params.width as f64;
        if pixel_size.is_finite() && pixel_size > 0.0 && pixel_size < 1e-15 {
            // Zoom too deep for f64-based BLA: return empty table
            // The perturbation code will detect this and use full GMP path
            return BlaTable::empty();
        }
    }

    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let mut levels: Vec<Vec<BlaNode>> = Vec::new();
    // Step 1: Create M BLAs each skipping 1 iteration (this can be done in parallel)
    let mut level0 = Vec::with_capacity(base_len);
    let base_threshold = params.bla_threshold.max(1e-16);
    let validity_scale = params.bla_validity_scale.clamp(0.1, 100.0);
    let power = params.multibrot_power;

    // Cap maximum validity at a reasonable multiple of base_threshold
    // This allows larger radii when z_ref is small, while preventing unbounded growth
    let max_validity = base_threshold * validity_scale * 10.0;

    for (i, &z) in ref_orbit.iter().enumerate().take(base_len) {
        let (coeffs, bs_valid) = if is_burning_ship {
            // Pour Burning Ship, vérifier la stabilité du quadrant
            let quadrant_stable = burning_ship_bla_validity(ref_orbit, i, 1);
            let coeffs = compute_burning_ship_bla_coefficients(z, quadrant_stable);
            (coeffs, quadrant_stable)
        } else {
            let coeffs = compute_bla_coefficients(z, params.fractal_type, power);
            (coeffs, true)  // Toujours valide pour les autres types
        };
        
        let a_norm = coeffs.a.norm();

        // Single Step BLA validity formula:
        // Approximation of a single step by bilinear form is valid when:
        // |z_n²| << |2·Z_n·z_n + c|
        // ⇑ assume negligibility of c << |2·Z_n·z_n|
        // ⇑ factor out z_n
        // |z_n| << |2·Z_n|
        // ⇑ definition of A_{n,1}, B_{n,1} for single step
        // |z_n| << |A_{n,1}| =: R_{n,1}
        //
        // Formula: R_{n,1} = ε·|A_{n,1}| where ε is the threshold (base_threshold * validity_scale)
        let epsilon = base_threshold * validity_scale;
        let validity = if a_norm > 1e-20 {
            epsilon * a_norm  // R_{n,1} = ε·|A_{n,1}|
        } else {
            0.0
        };
        
        // Cap at max_validity to prevent numerical issues
        let mut validity = validity.min(max_validity);
        
        // Pour Burning Ship, réduire la validité si le quadrant n'est pas stable
        if is_burning_ship && !bs_valid {
            validity = 0.0;
        } else if is_burning_ship && bs_valid {
            // ABS Variation BLA for Burning Ship:
            // The only problem with the Mandelbrot set is the non-linearity, but some other formulas
            // have other problems, for example the Burning Ship, defined by:
            // X + iY → (|X| + i|Y|)² + C
            //
            // The absolute value folds the plane when X or Y are near 0, so the single step BLA radius
            // becomes the minimum of the non-linearity radius and the folding radii:
            // R = max{0, min{ε·inf|A| - sup|B|·|c| / inf|A|, |X|, |Y|}}
            //
            // Currently Fraktaler 3 uses a fudge factor for paranoia, dividing |X| and |Y| by 2.
            // The merged BLA step radius is unchanged.
            //
            // Note: For Burning Ship, since it's conformal within each stable quadrant,
            // we have inf|A| = |A| and sup|B| = |B| = 1, so:
            // ε·inf|A| - sup|B|·|c| / inf|A| ≈ ε·|A| - |c| / |A|
            // For deep zooms where |c| << |A|, this simplifies to ε·|A|.
            // We use ε·|A| as the non-linearity radius (valid in stable quadrant).
            let nonlinearity_radius = validity;  // ε·|A| ≈ ε·inf|A| - sup|B|·|c| / inf|A| (for conformal case)
            let folding_radius_re = z.re.abs() / 2.0;  // |X|/2 (fudge factor for paranoia)
            let folding_radius_im = z.im.abs() / 2.0;  // |Y|/2 (fudge factor for paranoia)
            // R = max{0, min{nonlinearity_radius, |X|/2, |Y|/2}}
            validity = nonlinearity_radius.min(folding_radius_re).min(folding_radius_im).max(0.0);
        }

        // Calculate quadrant signs for Burning Ship optimization
        let sign_re = if z.re >= 0.0 { 1.0 } else { -1.0 };
        let sign_im = if z.im >= 0.0 { 1.0 } else { -1.0 };

        level0.push(BlaNode {
            a: coeffs.a,
            b: coeffs.b,
            c: coeffs.c,
            d: coeffs.d,
            e: coeffs.e,
            validity_radius: validity,
            burning_ship_valid: bs_valid,
            sign_re,
            sign_im,
        });
    }
    levels.push(level0);

    let zero = Complex64::new(0.0, 0.0);
    let max_level = 16usize;
    for level in 1..=max_level {
        let step = 1usize << (level - 1);
        let prev = &levels[level - 1];
        if prev.len() <= step {
            break;
        }
        let mut current = Vec::with_capacity(prev.len() - step);
        // Merging BLA Steps:
        // If T_x skips l_x iterations from iteration m_x when |z| < R_x
        // and T_y skips l_y iterations from iteration m_x + l_x when |z| < R_y,
        // then T_z = T_y ∘ T_x skips l_x + l_y iterations from iteration m_x when |z| < R_z
        //
        // Start merge from iteration 1 for the first level (level 1) because
        // iteration 0 always corresponds to a non-linear perturbation step as Z=0.
        // For subsequent levels, we can start from 0 as we're merging already-merged BLAs.
        let start_idx = if level == 1 { 1 } else { 0 };
        for i in start_idx..(prev.len() - step) {
            let node1 = prev[i];      // T_x: skips l_x = step iterations from m_x = i
            let node2 = prev[i + step];  // T_y: skips l_y = step iterations from m_x + l_x = i + step
            
            // Merging BLA Steps:
            // If T_x skips l_x iterations from iteration m_x when |z| < R_x
            // and T_y skips l_y iterations from iteration m_x + l_x when |z| < R_y,
            // then T_z = T_y ∘ T_x skips l_x + l_y iterations from iteration m_x when |z| < R_z:
            //
            // z_{m_x + l_x + l_y} = A_{m_x, l_x + l_y}·z_{m_x} + B_{m_x, l_x + l_y}·c
            //
            // where:
            // - A_{m_x, l_x + l_y} = A_z = A_y·A_x
            // - B_{m_x, l_x + l_y} = B_z = A_y·B_x + B_y
            //
            // Composition des coefficients:
            let a_new = node2.a * node1.a;  // A_z = A_y·A_x
            let b_new = node2.a * node1.b + node2.b;  // B_z = A_y·B_x + B_y
            let a1_sq = node1.a * node1.a;
            let c_new = node2.a * node1.c + node2.c * a1_sq;
            
            // Coefficients d'ordre supérieur pour Multibrot
            // D_new ≈ A2·D1 + 2·C2·A1·C1 + D2·A1³
            let d_new = if params.fractal_type == FractalType::Multibrot {
                node2.a * node1.d + node2.c * node1.a * node1.c * 2.0 + node2.d * a1_sq * node1.a
            } else {
                zero
            };
            
            // E_new ≈ A2·E1 + 2·C2·(A1·D1 + C1²) + 3·D2·A1²·C1 + E2·A1⁴
            let e_new = if params.fractal_type == FractalType::Multibrot {
                let c1_sq = node1.c * node1.c;
                node2.a * node1.e 
                    + node2.c * (node1.a * node1.d + c1_sq) * 2.0
                    + node2.d * a1_sq * node1.c * 3.0
                    + node2.e * a1_sq * a1_sq
            } else {
                zero
            };
            
            // Merging BLA validity formula:
            // R_{m_x, l_x + l_y} = R_z = max{0, min{R_x, R_y - |B_x|·|c| / |A_x|}}
            //
            // where:
            // - R_x = node1.validity_radius (validity radius of T_x)
            // - R_y = node2.validity_radius (validity radius of T_y)
            // - |A_x| = |node1.a| (norm of A_x coefficient)
            // - |B_x| = |node1.b| (norm of B_x coefficient)
            // - |c| = |cref| (norm of reference point C)
            let a1_norm = node1.a.norm();  // |A_x|
            let is_julia = params.fractal_type == FractalType::Julia;
            let validity = if a1_norm > 1e-20 && !is_julia {
                // For Mandelbrot: R_z = max{0, min{R_x, R_y - |B_x|·|c| / |A_x|}}
                let b1_norm = node1.b.norm();  // |B_x|
                let cref_norm = params.center_x.hypot(params.center_y);  // |c|
                let adjustment = b1_norm * cref_norm / a1_norm;  // |B_x|·|c| / |A_x|
                node1.validity_radius.min((node2.validity_radius - adjustment).max(0.0)).max(0.0)
            } else {
                // For Julia (B_x = 0) or when |A_x| is too small, use simpler formula:
                // R_z = max{0, min{R_x, R_y}}
                node1.validity_radius.min(node2.validity_radius)
            };
            
            // Pour les niveaux supérieurs de Burning Ship, le BLA est valide 
            // seulement si les deux nœuds sont valides.
            // The merged BLA step radius is unchanged (same formula as conformal case).
            let bs_valid = node1.burning_ship_valid && node2.burning_ship_valid;
            let validity = if is_burning_ship && !bs_valid {
                0.0
            } else {
                validity  // Merged BLA step radius unchanged for Burning Ship
            };
            
            // Valider les coefficients d'ordre supérieur
            let d_new = if d_new.re.is_finite() && d_new.im.is_finite() { d_new } else { zero };
            let e_new = if e_new.re.is_finite() && e_new.im.is_finite() { e_new } else { zero };

            // Propagate signs from starting node (node1)
            // For multi-step BLA, the starting quadrant determines the sign
            current.push(BlaNode {
                a: a_new,
                b: b_new,
                c: c_new,
                d: d_new,
                e: e_new,
                validity_radius: validity,
                burning_ship_valid: bs_valid,
                sign_re: node1.sign_re,
                sign_im: node1.sign_im,
            });
        }
        levels.push(current);  // Level k: ⌈M/2^k⌉ BLAs each skipping 2^k iterations
    }
    // The resulting table has O(M) elements: M + M/2 + M/4 + ... = 2M - 1 = O(M)

    // Build non-conformal table for Tricorn if applicable
    let nonconformal_levels = if params.fractal_type == FractalType::Tricorn {
        build_bla_table_nonconformal(ref_orbit, params)
    } else {
        None
    };

    BlaTable {
        levels,
        nonconformal_levels,
    }
}

/// Build non-conformal BLA table from the f64 reference orbit for Tricorn.
/// Uses 2×2 real matrices instead of complex numbers for coefficients.
pub fn build_bla_table_nonconformal(ref_orbit: &[Complex64], params: &FractalParams) -> Option<Vec<Vec<BlaNodeNonConformal>>> {
    if params.fractal_type != FractalType::Tricorn {
        return None;
    }

    let base_len = ref_orbit.len().saturating_sub(1);
    if base_len == 0 {
        return None;
    }

    let base_threshold = params.bla_threshold.max(1e-16);
    let validity_scale = params.bla_validity_scale.clamp(0.1, 100.0);
    let max_validity = base_threshold * validity_scale * 10.0;
    let cref_norm = params.center_x.hypot(params.center_y);

    let mut levels: Vec<Vec<BlaNodeNonConformal>> = Vec::new();
    let mut level0 = Vec::with_capacity(base_len);

    // Build level 0: single-step BLAs
    for (_i, &z) in ref_orbit.iter().enumerate().take(base_len) {
        let coeffs = compute_tricorn_bla_coefficients(z);
        
        // Calculate validity radius using non-conformal formula
        let validity = compute_nonconformal_validity_radius(
            coeffs.a,
            coeffs.b,
            base_threshold * validity_scale,
            cref_norm,
        ).min(max_validity).max(0.0);

        level0.push(BlaNodeNonConformal {
            a: coeffs.a,
            b: coeffs.b,
            validity_radius: validity,
        });
    }
    levels.push(level0);

    // Merge levels: start from iteration 1 (iteration 0 is always non-linear)
    let max_level = 16usize;
    for level in 1..=max_level {
        let step = 1usize << (level - 1);
        let prev = &levels[level - 1];
        if prev.len() <= step {
            break;
        }
        let mut current = Vec::with_capacity(prev.len() - step);
        // Start merge from iteration 1 for first level
        let start_idx = if level == 1 { 1 } else { 0 };
        for i in start_idx..(prev.len() - step) {
            let node1 = prev[i];
            let node2 = prev[i + step];
            
            // Merge using non-conformal formula
            let (az, bz, rz) = merge_nonconformal_bla(
                node1.a,
                node1.b,
                node1.validity_radius,
                node2.a,
                node2.b,
                node2.validity_radius,
                cref_norm,
            );
            
            current.push(BlaNodeNonConformal {
                a: az,
                b: bz,
                validity_radius: rz,
            });
        }
        levels.push(current);
    }

    Some(levels)
}

#[cfg(test)]
mod nonconformal_tests {
    use super::*;
    use crate::fractal::{AlgorithmMode, FractalType};
    use num_complex::Complex64;

    fn test_tricorn_params() -> FractalParams {
        FractalParams {
            width: 100,
            height: 100,
            center_x: 0.0,
            center_y: 0.0,
            span_x: 4.0,
            span_y: 4.0,
            seed: Complex64::new(0.0, 0.0),
            iteration_max: 100,
            bailout: 4.0,
            fractal_type: FractalType::Tricorn,
            color_mode: 0,
            color_repeat: 2,
            use_gmp: false,
            precision_bits: 192,
            algorithm_mode: AlgorithmMode::Perturbation,
            bla_threshold: 1e-6,
            bla_validity_scale: 1.0,
            glitch_tolerance: 1e-4,
            series_order: 2,
            series_threshold: 1e-6,
            series_error_tolerance: 1e-9,
            glitch_neighbor_pass: false,
            series_standalone: false,
            max_secondary_refs: 3,
            min_glitch_cluster_size: 100,
            multibrot_power: 2.5,
            lyapunov_preset: Default::default(),
            lyapunov_sequence: Vec::new(),
            enable_distance_estimation: false,
            enable_interior_detection: false,
            interior_threshold: 0.001,
        }
    }

    #[test]
    fn build_nonconformal_table() {
        let params = test_tricorn_params();
        let ref_orbit = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        let table = build_bla_table_nonconformal(&ref_orbit, &params);
        assert!(table.is_some());
        let levels = table.unwrap();
        assert!(!levels.is_empty());
        assert!(!levels[0].is_empty());
    }

    #[test]
    fn nonconformal_table_merging() {
        let params = test_tricorn_params();
        let ref_orbit: Vec<Complex64> = (0..10)
            .map(|i| Complex64::new(i as f64 * 0.1, 0.0))
            .collect();
        let table = build_bla_table_nonconformal(&ref_orbit, &params);
        assert!(table.is_some());
        let levels = table.unwrap();
        // Should have multiple levels after merging
        assert!(levels.len() > 1);
        // Level 1 should start from iteration 1
        if levels.len() > 1 {
            assert!(levels[1].len() < levels[0].len());
        }
    }
}
