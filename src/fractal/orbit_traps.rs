use num_complex::Complex64;
use std::sync::OnceLock;

/// Type d'Orbit Trap (forme géométrique pour détecter la proximité de l'orbite)
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub enum OrbitTrapType {
    /// Distance minimale à un point (généralement l'origine)
    Point,
    /// Distance minimale à une ligne (horizontale, verticale, ou diagonale)
    Line { angle: f64 }, // angle en radians
    /// Distance minimale à une croix (ligne horizontale + verticale)
    Cross,
    /// Distance minimale à un cercle
    Circle { center: Complex64, radius: f64 },
}

impl OrbitTrapType {
    #[allow(dead_code)]
    pub fn all() -> &'static [OrbitTrapType] {
        static TRAPS: OnceLock<Vec<OrbitTrapType>> = OnceLock::new();
        TRAPS.get_or_init(|| vec![
            OrbitTrapType::Point,
            OrbitTrapType::Line { angle: 0.0 }, // Horizontal
            OrbitTrapType::Line { angle: std::f64::consts::PI / 2.0 }, // Vertical
            OrbitTrapType::Cross,
            OrbitTrapType::Circle { center: Complex64::new(0.0, 0.0), radius: 1.0 },
        ])
    }
    
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            OrbitTrapType::Point => "Point",
            OrbitTrapType::Line { angle: 0.0 } => "Line (H)",
            OrbitTrapType::Line { angle: _ } => "Line",
            OrbitTrapType::Cross => "Cross",
            OrbitTrapType::Circle { .. } => "Circle",
        }
    }
}

/// Données d'orbite pour orbit traps
#[derive(Clone, Debug)]
pub struct OrbitData {
    /// Points de l'orbite (stockés jusqu'à escape ou max_iterations)
    pub points: Vec<Complex64>,
    /// Distance minimale à l'orbit trap détectée
    pub min_distance: f64,
    /// Itération à laquelle la distance minimale a été atteinte
    pub min_distance_iter: u32,
    /// Type d'orbit trap utilisé
    pub trap_type: OrbitTrapType,
}

impl OrbitData {
    pub fn new(trap_type: OrbitTrapType) -> Self {
        Self {
            points: Vec::new(),
            min_distance: f64::INFINITY,
            min_distance_iter: 0,
            trap_type,
        }
    }
    
    /// Ajoute un point à l'orbite et calcule la distance à l'orbit trap
    pub fn add_point(&mut self, z: Complex64, iteration: u32) {
        self.points.push(z);
        let distance = self.compute_distance(z);
        if distance < self.min_distance {
            self.min_distance = distance;
            self.min_distance_iter = iteration;
        }
    }
    
    /// Calcule la distance de z à l'orbit trap
    fn compute_distance(&self, z: Complex64) -> f64 {
        match self.trap_type {
            OrbitTrapType::Point => {
                // Distance à l'origine
                z.norm()
            }
            OrbitTrapType::Line { angle } => {
                // Distance à une ligne passant par l'origine avec angle donné
                // Ligne: ax + by = 0 où (a, b) = (sin(angle), -cos(angle))
                let a = angle.sin();
                let b = -angle.cos();
                (a * z.re + b * z.im).abs()
            }
            OrbitTrapType::Cross => {
                // Distance minimale à la croix (ligne horizontale ou verticale)
                z.re.abs().min(z.im.abs())
            }
            OrbitTrapType::Circle { center, radius } => {
                // Distance au cercle = |distance au centre - rayon|
                let dist_to_center = (z - center).norm();
                (dist_to_center - radius).abs()
            }
        }
    }
}

/// Calcule la distance minimale à un orbit trap pour une orbite complète
#[allow(dead_code)]
pub fn compute_orbit_trap_distance(
    orbit: &[Complex64],
    trap_type: OrbitTrapType,
) -> (f64, u32) {
    let mut min_distance = f64::INFINITY;
    let mut min_iter = 0;
    
    for (i, &z) in orbit.iter().enumerate() {
        let distance = match trap_type {
            OrbitTrapType::Point => z.norm(),
            OrbitTrapType::Line { angle } => {
                let a = angle.sin();
                let b = -angle.cos();
                (a * z.re + b * z.im).abs()
            }
            OrbitTrapType::Cross => z.re.abs().min(z.im.abs()),
            OrbitTrapType::Circle { center, radius } => {
                let dist_to_center = (z - center).norm();
                (dist_to_center - radius).abs()
            }
        };
        
        if distance < min_distance {
            min_distance = distance;
            min_iter = i as u32;
        }
    }
    
    (min_distance, min_iter)
}
