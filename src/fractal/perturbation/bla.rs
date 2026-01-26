use num_complex::Complex64;

use crate::fractal::{FractalParams, FractalType};

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
}

#[derive(Clone, Debug)]
pub struct BlaTable {
    pub levels: Vec<Vec<BlaNode>>,
}

impl BlaTable {
    pub fn empty() -> Self {
        Self { levels: Vec::new() }
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

/// Compute BLA coefficients (A, B, C, D, E) for a single iteration at z_ref.
/// For z^d + c iteration: δ' = A·δ + B·dc + C·δ² + D·δ³ + E·δ⁴
/// where:
/// - A = d·z^(d-1)
/// - B = 1 (Mandelbrot) or 0 (Julia)
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
            // Standard z² + c: A = 2z, C = 1, D = E = 0
            let a = z * 2.0;
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

pub fn build_bla_table(ref_orbit: &[Complex64], params: &FractalParams) -> BlaTable {
    let supports_bla = matches!(
        params.fractal_type,
        FractalType::Mandelbrot | FractalType::Julia | FractalType::Multibrot | FractalType::BurningShip
    );
    if !supports_bla {
        return BlaTable::empty();
    }

    let base_len = ref_orbit.len().saturating_sub(1);
    if base_len == 0 {
        return BlaTable::empty();
    }

    let is_burning_ship = params.fractal_type == FractalType::BurningShip;
    let mut levels: Vec<Vec<BlaNode>> = Vec::new();
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

        // When a_norm is small, validity can exceed base_threshold
        // The formula: threshold / (1 + |a|) scales inversely with derivative magnitude
        let mut validity = (base_threshold * validity_scale) / (1.0 + a_norm);
        if !validity.is_finite() {
            validity = base_threshold * validity_scale;
        }
        // Allow validity to exceed base_threshold when z_ref is small (a_norm small)
        // but cap at max_validity to prevent numerical issues
        validity = validity.min(max_validity);
        
        // Pour Burning Ship, réduire la validité si le quadrant n'est pas stable
        if is_burning_ship && !bs_valid {
            validity = 0.0;
        }

        level0.push(BlaNode {
            a: coeffs.a,
            b: coeffs.b,
            c: coeffs.c,
            d: coeffs.d,
            e: coeffs.e,
            validity_radius: validity,
            burning_ship_valid: bs_valid,
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
        for i in 0..(prev.len() - step) {
            let node1 = prev[i];
            let node2 = prev[i + step];
            
            // Composition des coefficients pour les niveaux supérieurs
            // δ' = A1·δ + B1·dc + C1·δ² + D1·δ³ + E1·δ⁴
            // δ'' = A2·δ' + B2·dc + C2·δ'² + D2·δ'³ + E2·δ'⁴
            // En substituant et en gardant les termes jusqu'à l'ordre 4:
            let a_new = node2.a * node1.a;
            let b_new = node2.a * node1.b + node2.b;
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
            
            let a_norm = node1.a.norm();
            let scaled = node2.validity_radius / (1.0 + a_norm);
            
            // Pour les niveaux supérieurs de Burning Ship, le BLA est valide 
            // seulement si les deux nœuds sont valides
            let bs_valid = node1.burning_ship_valid && node2.burning_ship_valid;
            let validity = if is_burning_ship && !bs_valid {
                0.0
            } else {
                node1.validity_radius.min(scaled)
            };
            
            // Valider les coefficients d'ordre supérieur
            let d_new = if d_new.re.is_finite() && d_new.im.is_finite() { d_new } else { zero };
            let e_new = if e_new.re.is_finite() && e_new.im.is_finite() { e_new } else { zero };
            
            current.push(BlaNode {
                a: a_new,
                b: b_new,
                c: c_new,
                d: d_new,
                e: e_new,
                validity_radius: validity,
                burning_ship_valid: bs_valid,
            });
        }
        levels.push(current);
    }

    BlaTable { levels }
}
