//! Échantillonnage des rendus de densité (Buddhabrot, Nebulabrot,
//! Anti-Buddhabrot).
//!
//! Ces trois types ne calculent pas une valeur par pixel : ils tirent des
//! paramètres `c` sur le **domaine canonique** du plan des paramètres, itèrent
//! l'orbite de chacun, et projettent les points visités dans la fenêtre
//! affichée. La fenêtre ne sert qu'à PROJETER — elle ne définit pas où l'on
//! échantillonne, sans quoi naviguer régénérerait un champ différent à chaque
//! déplacement.
//!
//! ## Deux régimes d'échantillonnage
//!
//! - **Uniforme** : les `c` sont tirés uniformément sur le domaine. Estimateur
//!   non biaisé et trivialement parallèle, mais la fraction des trajectoires
//!   qui atteignent la fenêtre décroît comme sa SURFACE : au-delà d'un zoom
//!   ~×100 l'image est affamée, puis vide. C'est le régime historique, conservé
//!   tant que la vue couvre une part suffisante du domaine.
//! - **Metropolis-Hastings** (Boswell, *The Metropolis-Hastings Algorithm and
//!   the Buddhabrot*) : une chaîne de Markov échantillonne les `c` selon leur
//!   CONTRIBUTION à la fenêtre. Une fois qu'une chaîne a trouvé un `c` qui
//!   contribue, ses mutations locales en trouvent d'autres — la fenêtre reste
//!   nourrie quel que soit le zoom.
//!
//! ## Non-biais
//!
//! La chaîne converge vers `π(c) ∝ f(c)`, où `f(c)` compte les points de
//! l'orbite tombant dans la fenêtre. Or la densité recherchée est
//! `D(p) = ∫ h_p(c) dc` (`h_p` = points tombant dans le pixel `p`), et
//! `∫ h_p = Z · E_π[h_p / f]`. Chaque état de la chaîne est donc projeté avec
//! le **poids `1/f`** : chaque pas dépose exactement une unité de masse,
//! répartie sur les points de son orbite. Sans ce poids, l'image serait
//! proportionnelle à `f · D` — les zones denses seraient doublement comptées.
//!
//! ## Amorçage recuit
//!
//! Une chaîne ne peut démarrer que depuis un `c` qui contribue déjà, ce qui est
//! précisément introuvable par tirage uniforme en zoom profond. L'amorçage
//! interpole donc la fenêtre depuis le domaine canonique jusqu'à la vue cible
//! (`K ≈ log2(zoom)` étapes, centre ET échelle) : à l'étape 0 la « fenêtre »
//! est le domaine entier, donc toute orbite retenue contribue ; chaque étape
//! resserre d'un facteur ~2 et la chaîne migre. Une chaîne qui perd toute
//! contribution est déclarée morte et abandonne son budget.
//!
//! ## Déterminisme
//!
//! Les chaînes sont indépendantes, graine dérivée de leur indice, et la masse
//! est accumulée en **virgule fixe** dans des `AtomicU64` : l'addition entière
//! commute, donc l'image ne dépend ni du nombre de threads ni de leur ordre.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use rayon::prelude::*;
use rug::{Complex, Float};

use super::FractalParams;

/// Domaine canonique des `c` : la vue par défaut des types de densité.
pub(crate) const SAMPLE_CENTER_X: f64 = -0.5;
pub(crate) const SAMPLE_CENTER_Y: f64 = 0.0;
pub(crate) const SAMPLE_SPAN_X: f64 = 4.0;
pub(crate) const SAMPLE_SPAN_Y: f64 = 3.0;

/// Résolution de l'accumulateur en virgule fixe. Un pas de chaîne dépose
/// `WEIGHT_SCALE` unités de masse au total, réparties sur ses points.
const WEIGHT_SCALE: u64 = 1 << 20;

/// Longueur visée de la phase de mesure d'une chaîne (cf. [`MetropolisPlan`]).
const CHAIN_LENGTH: usize = 1024;

/// Le régime Metropolis n'est engagé que si la vue est nettement plus petite
/// que le domaine : en deçà, le tirage uniforme ne souffre d'aucune famine et
/// reste préférable (aucune corrélation de chaîne, aucun amorçage à payer).
const METROPOLIS_MIN_ZOOM: f64 = 4.0;

/// Générateur historique (LCG) du régime uniforme. Conservé tel quel : la vue
/// par défaut doit rester reproductible au tirage près.
pub(crate) struct Rng {
    seed: u32,
}

impl Rng {
    pub(crate) fn new(seed: u32) -> Self {
        Self { seed }
    }

    fn next(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        self.seed
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next() & 0x7FFF_FFFF) as f64 / 2147483647.0
    }
}

/// Générateur des chaînes de Markov. Le LCG historique a des bits de poids
/// faible trop corrélés pour les millions de tirages d'une chaîne.
struct ChainRng {
    state: u64,
}

impl ChainRng {
    fn new(seed: u64) -> Self {
        // SplitMix64 pour décorréler des graines consécutives.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self {
            state: (z ^ (z >> 31)) | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Ce qui distingue les trois types : durée d'orbite, sortie retenue et
/// quelques particularités historiques d'arrêt.
#[derive(Clone, Copy)]
pub(crate) struct OrbitSpec {
    pub iter_max: u32,
    pub bailout_sq: f64,
    /// Buddhabrot seulement : abandon anticipé des orbites encore proches de 0.
    pub early_exit_at: Option<u32>,
    /// Anti-Buddhabrot : un débordement numérique compte comme une évasion.
    pub nan_counts_as_escape: bool,
    /// `true` = on garde les orbites qui s'échappent (Buddhabrot, Nebulabrot),
    /// `false` = celles qui restent prisonnières (Anti-Buddhabrot).
    pub keep_escaped: bool,
}

/// Issue d'une orbite.
#[derive(Clone, Copy, Default)]
struct Trace {
    /// L'orbite est-elle du côté retenu par [`OrbitSpec::keep_escaped`] ?
    kept: bool,
    escape_iter: u32,
}

/// Trajectoire en coordonnées **fraction de pixel** de la vue cible : `(0, 0)`
/// est le coin haut-gauche, `(width, height)` le coin bas-droit. Ce système
/// garde toute la résolution près de la fenêtre, même quand celle-ci est à
/// 1e-40 du reste du plan.
type Trajectory = Vec<(f64, f64)>;

/// Destination des projections pendant la phase de mesure — `None` pendant
/// l'amorçage, qui ne dépose rien.
type DensitySink<'a> = Option<(&'a Density, fn(u32) -> u32)>;

/// Fenêtre de l'étape d'amorçage, dans le même système.
#[derive(Clone, Copy)]
struct Window {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

/// Ce qu'une orbite vaut pour une fenêtre.
#[derive(Clone, Copy, Default)]
struct Score {
    /// Cible de la chaîne : somme d'un noyau qui vaut 1 dans la fenêtre et
    /// décroît continûment au dehors.
    soft: f64,
    /// Points réellement DANS la fenêtre — ceux qui seront projetés.
    hits: u32,
}

impl Window {
    fn contains(&self, point: &(f64, f64)) -> bool {
        point.0 >= self.min_x
            && point.0 <= self.max_x
            && point.1 >= self.min_y
            && point.1 <= self.max_y
    }

    /// Distance d'un point à la fenêtre, en demi-largeurs de celle-ci : nulle à
    /// l'intérieur, 1 à une fenêtre de distance.
    fn distance(&self, point: &(f64, f64)) -> f64 {
        let half_x = (self.max_x - self.min_x) * 0.5;
        let half_y = (self.max_y - self.min_y) * 0.5;
        let center_x = (self.min_x + self.max_x) * 0.5;
        let center_y = (self.min_y + self.max_y) * 0.5;
        let dx = ((point.0 - center_x).abs() - half_x).max(0.0) / half_x.max(f64::MIN_POSITIVE);
        let dy = ((point.1 - center_y).abs() - half_y).max(0.0) / half_y.max(f64::MIN_POSITIVE);
        dx.hypot(dy)
    }

    /// Une cible strictement binaire (« l'orbite touche-t-elle la fenêtre ? »)
    /// ne guide pas la chaîne : au zoom où une orbite ne dépose qu'un ou deux
    /// points, resserrer la fenêtre éteint la population avant d'arriver.
    ///
    /// La cible reste donc le NOMBRE de points touchés — un point dedans pèse
    /// toujours plus que toutes les quasi-touches réunies — augmenté d'un terme
    /// de proximité minuscule. Ce terme ne départage que des orbites à égalité
    /// de touches, et leur donne la pente à remonter vers la fenêtre. Un
    /// noyau de proximité de plein poids échoue : la masse des points
    /// extérieurs domine, et la population converge vers des orbites qui
    /// frôlent la fenêtre sans jamais y entrer (mesuré : plus rien dès ×1000).
    fn score(&self, trajectory: &Trajectory) -> Score {
        /// Poids du terme de proximité, assez petit pour que la somme sur une
        /// orbite entière reste très en dessous d'une seule touche.
        const PROXIMITY: f64 = 1.0e-6;

        let mut score = Score::default();
        for point in trajectory {
            if !point.0.is_finite() || !point.1.is_finite() {
                continue;
            }
            if self.contains(point) {
                score.soft += 1.0;
                score.hits += 1;
            } else {
                let spread = 1.0 + self.distance(point);
                score.soft += PROXIMITY / (spread * spread);
            }
        }
        score
    }
}

/// Tirage des `c` et itération des orbites, pour une arithmétique donnée.
trait Sampler: Sync {
    type Point: Clone + Send;

    /// Tirage uniforme du régime historique (deux appels au LCG).
    fn uniform_legacy(&self, rng: &mut Rng) -> Self::Point;

    /// Tirage uniforme sur le domaine canonique pour les chaînes.
    fn uniform_chain(&self, rng: &mut ChainRng) -> Self::Point;

    /// Déplace `c` d'un rayon log-uniforme, exprimé en **spans de la vue
    /// cible** (donc minuscule en zoom profond : la haute précision de `c` doit
    /// survivre au déplacement).
    fn mutate(&self, c: &Self::Point, radius_spans: f64, rng: &mut ChainRng) -> Self::Point;

    /// Itère l'orbite de `c` et remplit `out` des points visités, en fraction
    /// de pixel de la vue cible.
    fn trace(&self, c: &Self::Point, spec: &OrbitSpec, out: &mut Trajectory) -> Trace;
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmétique f64
// ─────────────────────────────────────────────────────────────────────────────

struct F64Sampler {
    center_x: f64,
    center_y: f64,
    span_x: f64,
    span_y: f64,
    scale_x: f64,
    scale_y: f64,
}

impl F64Sampler {
    fn new(params: &FractalParams) -> Self {
        let span_x = params.span_x;
        let span_y = params.span_y;
        Self {
            center_x: params.center_x,
            center_y: params.center_y,
            span_x,
            span_y,
            scale_x: f64::from(params.width) / span_x,
            scale_y: f64::from(params.height) / span_y,
        }
    }

    #[inline]
    fn project(&self, z: Complex64) -> (f64, f64) {
        (
            (z.re - self.center_x + self.span_x * 0.5) * self.scale_x,
            (z.im - self.center_y + self.span_y * 0.5) * self.scale_y,
        )
    }
}

impl Sampler for F64Sampler {
    type Point = Complex64;

    fn uniform_legacy(&self, rng: &mut Rng) -> Complex64 {
        let x = SAMPLE_CENTER_X + (rng.next_f64() - 0.5) * SAMPLE_SPAN_X;
        let y = SAMPLE_CENTER_Y + (rng.next_f64() - 0.5) * SAMPLE_SPAN_Y;
        Complex64::new(x, y)
    }

    fn uniform_chain(&self, rng: &mut ChainRng) -> Complex64 {
        let x = SAMPLE_CENTER_X + (rng.next_f64() - 0.5) * SAMPLE_SPAN_X;
        let y = SAMPLE_CENTER_Y + (rng.next_f64() - 0.5) * SAMPLE_SPAN_Y;
        Complex64::new(x, y)
    }

    fn mutate(&self, c: &Complex64, radius_spans: f64, rng: &mut ChainRng) -> Complex64 {
        let (dx, dy) = mutation_offset(radius_spans, rng);
        Complex64::new(c.re + dx * self.span_x, c.im + dy * self.span_x)
    }

    fn trace(&self, c: &Complex64, spec: &OrbitSpec, out: &mut Trajectory) -> Trace {
        out.clear();
        let mut z = Complex64::new(0.0, 0.0);
        let mut escaped = false;
        let mut escape_iter = 0u32;

        for iter in 0..spec.iter_max {
            z = z * z + c;

            if !z.re.is_finite() || !z.im.is_finite() {
                escaped = spec.nan_counts_as_escape;
                break;
            }
            if spec.early_exit_at == Some(iter) && z.norm_sqr() < 0.25 {
                break;
            }

            out.push(self.project(z));

            if z.norm_sqr() > spec.bailout_sq {
                escaped = true;
                escape_iter = iter;
                break;
            }
        }

        Trace {
            kept: escaped == spec.keep_escaped && !out.is_empty(),
            escape_iter,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arithmétique MPC — vues dont un pixel n'est plus résolvable en f64
// ─────────────────────────────────────────────────────────────────────────────

struct MpcSampler {
    center_x: Float,
    center_y: Float,
    span_x: Float,
    span_y: Float,
    width: f64,
    height: f64,
    bailout_sq: Float,
    prec: u32,
}

impl MpcSampler {
    fn new(params: &FractalParams, spec: &OrbitSpec) -> Self {
        let prec = crate::fractal::perturbation::compute_perturbation_precision_bits(params)
            .max(params.precision_bits)
            .max(64);
        let parse = |hp: Option<&String>, fallback: f64| {
            hp.and_then(|value| Float::parse(value).ok())
                .map(|value| Float::with_val(prec, value))
                .unwrap_or_else(|| Float::with_val(prec, fallback))
        };
        Self {
            center_x: parse(params.center_x_hp.as_ref(), params.center_x),
            center_y: parse(params.center_y_hp.as_ref(), params.center_y),
            span_x: parse(params.span_x_hp.as_ref(), params.span_x),
            span_y: parse(params.span_y_hp.as_ref(), params.span_y),
            width: f64::from(params.width),
            height: f64::from(params.height),
            bailout_sq: Float::with_val(prec, spec.bailout_sq),
            prec,
        }
    }

    #[inline]
    fn project(&self, z: &Complex) -> (f64, f64) {
        let mut x = Float::with_val(self.prec, z.real());
        x -= &self.center_x;
        x /= &self.span_x;
        x += 0.5;
        let mut y = Float::with_val(self.prec, z.imag());
        y -= &self.center_y;
        y /= &self.span_y;
        y += 0.5;
        (x.to_f64() * self.width, y.to_f64() * self.height)
    }

    #[inline]
    fn norm_sqr(&self, z: &Complex) -> Float {
        let mut re2 = z.real().clone();
        re2 *= z.real();
        let mut im2 = z.imag().clone();
        im2 *= z.imag();
        let mut sum = Float::with_val(self.prec, re2);
        sum += im2;
        sum
    }
}

impl Sampler for MpcSampler {
    type Point = Complex;

    fn uniform_legacy(&self, rng: &mut Rng) -> Complex {
        let mut x = Float::with_val(self.prec, SAMPLE_SPAN_X);
        x *= rng.next_f64() - 0.5;
        x += SAMPLE_CENTER_X;
        let mut y = Float::with_val(self.prec, SAMPLE_SPAN_Y);
        y *= rng.next_f64() - 0.5;
        y += SAMPLE_CENTER_Y;
        Complex::with_val(self.prec, (x, y))
    }

    fn uniform_chain(&self, rng: &mut ChainRng) -> Complex {
        let mut x = Float::with_val(self.prec, SAMPLE_SPAN_X);
        x *= rng.next_f64() - 0.5;
        x += SAMPLE_CENTER_X;
        let mut y = Float::with_val(self.prec, SAMPLE_SPAN_Y);
        y *= rng.next_f64() - 0.5;
        y += SAMPLE_CENTER_Y;
        Complex::with_val(self.prec, (x, y))
    }

    fn mutate(&self, c: &Complex, radius_spans: f64, rng: &mut ChainRng) -> Complex {
        let (dx, dy) = mutation_offset(radius_spans, rng);
        // Le déplacement est calculé DANS le span : à 1e-40, un offset f64
        // s'évanouirait à l'addition.
        let mut x = Float::with_val(self.prec, &self.span_x * dx);
        x += c.real();
        let mut y = Float::with_val(self.prec, &self.span_x * dy);
        y += c.imag();
        Complex::with_val(self.prec, (x, y))
    }

    fn trace(&self, c: &Complex, spec: &OrbitSpec, out: &mut Trajectory) -> Trace {
        out.clear();
        let mut z = Complex::with_val(self.prec, (0.0, 0.0));
        let mut escaped = false;
        let mut escape_iter = 0u32;

        for iter in 0..spec.iter_max {
            let mut next = z.clone();
            next *= &z;
            next += c;
            z = next;

            if !z.real().is_finite() || !z.imag().is_finite() {
                escaped = spec.nan_counts_as_escape;
                break;
            }

            let mag2 = self.norm_sqr(&z);
            if spec.early_exit_at == Some(iter) && mag2 < 0.25 {
                break;
            }

            out.push(self.project(&z));

            if mag2 > self.bailout_sq {
                escaped = true;
                escape_iter = iter;
                break;
            }
        }

        Trace {
            kept: escaped == spec.keep_escaped && !out.is_empty(),
            escape_iter,
        }
    }
}

/// Déplacement log-uniforme : couvre trois décades d'échelles autour de la
/// fenêtre courante, pour explorer aussi bien la structure fine que la
/// structure large sans réglage par scène.
#[inline]
fn mutation_offset(radius_spans: f64, rng: &mut ChainRng) -> (f64, f64) {
    const RATIO: f64 = 1.0e-3;
    let hi = radius_spans * 0.5;
    let lo = hi * RATIO;
    let radius = lo * (-RATIO.ln() * rng.next_f64()).exp();
    let angle = std::f64::consts::TAU * rng.next_f64();
    (radius * angle.cos(), radius * angle.sin())
}

// ─────────────────────────────────────────────────────────────────────────────
// Accumulation
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulateur en virgule fixe : `channels` plans de `width × height`.
struct Density {
    channels: Vec<Vec<AtomicU64>>,
    width: usize,
    height: usize,
}

impl Density {
    fn new(channels: usize, width: usize, height: usize) -> Self {
        Self {
            channels: (0..channels)
                .map(|_| (0..width * height).map(|_| AtomicU64::new(0)).collect())
                .collect(),
            width,
            height,
        }
    }

    /// Dépose `weight` sur chaque point de la trajectoire tombant dans la
    /// fenêtre, pour chaque canal sélectionné par `mask`.
    fn splat(&self, trajectory: &Trajectory, weight: u64, mask: u32) {
        if weight == 0 || mask == 0 {
            return;
        }
        for point in trajectory {
            // `floor`, pas une troncature : un point à gauche ou au-dessus de
            // la fenêtre (coordonnée dans `]-1, 0[`) doit être REJETÉ, pas
            // replié sur la première colonne ou la première ligne. Le path MPC
            // arrondissait déjà correctement — c'est le path f64 qui salissait
            // ses deux bords.
            let px = point.0.floor();
            let py = point.1.floor();
            if !(px >= 0.0 && px < self.width as f64 && py >= 0.0 && py < self.height as f64) {
                continue;
            }
            let idx = py as usize * self.width + px as usize;
            for (channel, plane) in self.channels.iter().enumerate() {
                if mask & (1 << channel) != 0 {
                    plane[idx].fetch_add(weight, Ordering::Relaxed);
                }
            }
        }
    }

    /// Masse par pixel, en unités « points d'orbite » : le régime uniforme y
    /// retrouve exactement ses comptes entiers.
    fn into_mass(self) -> Vec<Vec<f64>> {
        self.channels
            .into_iter()
            .map(|plane| {
                plane
                    .into_iter()
                    .map(|cell| cell.into_inner() as f64 / WEIGHT_SCALE as f64)
                    .collect()
            })
            .collect()
    }
}

/// Fenêtre cible en fraction de pixel.
fn target_window(width: usize, height: usize) -> Window {
    Window {
        min_x: 0.0,
        max_x: width as f64,
        min_y: 0.0,
        max_y: height as f64,
    }
}

/// Géométrie de l'amorçage : de combien la fenêtre cible doit être dilatée
/// pour englober le domaine canonique. Calculée une fois, en haute précision
/// quand la vue l'exige — à 1e-40 le rapport ne tient plus dans un `f64` naïf.
///
/// Les fenêtres intermédiaires sont **concentriques** sur la vue cible et
/// décroissent géométriquement : la suite est donc emboîtée, chaque étape
/// contenant la suivante. Une interpolation qui déplacerait aussi le centre
/// paraît plus directe mais échoue : sur une vue décentrée, les dernières
/// étapes translatent la fenêtre de plusieurs dizaines de largeurs alors
/// qu'elle ne mesure plus qu'une largeur — aucune chaîne ne peut suivre un
/// saut pareil, et la population s'éteint (mesuré : plus rien dès ×10 000).
struct AnnealGeometry {
    cover_x: f64,
    cover_y: f64,
}

impl AnnealGeometry {
    fn new(params: &FractalParams, mpc: Option<&MpcSampler>) -> Self {
        let (ratio_x, ratio_y, offset_x, offset_y) = match mpc {
            Some(sampler) => {
                let prec = sampler.prec;
                let ratio_x = Float::with_val(prec, SAMPLE_SPAN_X) / &sampler.span_x;
                let ratio_y = Float::with_val(prec, SAMPLE_SPAN_Y) / &sampler.span_y;
                let offset_x =
                    Float::with_val(prec, SAMPLE_CENTER_X - &sampler.center_x) / &sampler.span_x;
                let offset_y =
                    Float::with_val(prec, SAMPLE_CENTER_Y - &sampler.center_y) / &sampler.span_y;
                (
                    ratio_x.to_f64(),
                    ratio_y.to_f64(),
                    offset_x.to_f64(),
                    offset_y.to_f64(),
                )
            }
            None => (
                SAMPLE_SPAN_X / params.span_x,
                SAMPLE_SPAN_Y / params.span_y,
                (SAMPLE_CENTER_X - params.center_x) / params.span_x,
                (SAMPLE_CENTER_Y - params.center_y) / params.span_y,
            ),
        };
        // Dilatation qui, centrée sur la vue, contient tout le domaine.
        Self {
            cover_x: (ratio_x + 2.0 * offset_x.abs()).max(1.0),
            cover_y: (ratio_y + 2.0 * offset_y.abs()).max(1.0),
        }
    }

    /// Fenêtre de l'étape `t ∈ [0, 1]` : `t = 0` englobe le domaine canonique,
    /// `t = 1` est exactement la vue.
    fn window(&self, t: f64, width: usize, height: usize) -> Window {
        let rest = 1.0 - t;
        let half_x = 0.5 * self.cover_x.powf(rest) * width as f64;
        let half_y = 0.5 * self.cover_y.powf(rest) * height as f64;
        let (center_x, center_y) = (0.5 * width as f64, 0.5 * height as f64);
        Window {
            min_x: center_x - half_x,
            max_x: center_x + half_x,
            min_y: center_y - half_y,
            max_y: center_y + half_y,
        }
    }

    /// Largeur de la fenêtre de l'étape, en spans de la vue cible : c'est
    /// l'échelle de départ des mutations à cette étape.
    fn span_at(&self, t: f64) -> f64 {
        self.cover_x.powf(1.0 - t)
    }

    fn is_usable(&self) -> bool {
        self.cover_x.is_finite() && self.cover_y.is_finite()
    }
}

/// Budget d'une exécution Metropolis. Tout est dérivé du nombre d'échantillons
/// du régime uniforme : le coût total reste comparable.
struct MetropolisPlan {
    chains: usize,
    stages: usize,
    steps_per_stage: usize,
    burn_in: usize,
    chain_steps: usize,
}

impl MetropolisPlan {
    /// Une étape d'amorçage par octave de zoom : la fenêtre se resserre d'un
    /// facteur ~2 à chaque fois, ce qu'une chaîne peut suivre par mutations
    /// locales.
    ///
    /// À budget CONSTANT, la longueur des chaînes arbitre entre mélange et
    /// indépendance : trop courtes, elles restent où l'amorçage les a laissées
    /// et n'ont pas amorti son coût ; trop longues, elles explorent bien mais
    /// sont trop peu nombreuses pour couvrir l'image. Mesuré contre un tirage
    /// uniforme à très gros budget (corrélation à ×8, erreur relative à ×40),
    /// l'optimum est large et plat autour du millier de pas :
    ///
    /// | pas    |   64 |  256 |  512 | 1024 | 4096 |
    /// |--------|------|------|------|------|------|
    /// | corr   | 0,81 | 0,93 | 0,95 | 0,96 | 0,96 |
    /// | erreur | 0,18 | 0,15 | 0,15 | 0,15 | 0,18 |
    fn new(num_samples: usize, cover: f64) -> Self {
        let stages = (cover.max(2.0).log2().ceil() as usize).clamp(2, 192);
        let steps_per_stage = 32;
        let warmup = stages * steps_per_stage;
        let per_chain = warmup + CHAIN_LENGTH + CHAIN_LENGTH / 8;
        let chains = (num_samples / per_chain.max(1)).clamp(32, 65_536);
        let budget = (num_samples / chains).saturating_sub(warmup).max(64);
        let burn_in = budget / 8;
        Self {
            chains,
            stages,
            steps_per_stage,
            burn_in,
            chain_steps: budget - burn_in,
        }
    }
}

/// Régime d'échantillonnage. `Auto` est le seul utilisé en production ; les
/// deux régimes forcés servent à les comparer entre eux sur une vue où ils sont
/// tous deux viables.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Auto,
    Uniform,
    Metropolis,
}

/// Forçage manuel du régime (`FRACTALL_DENSITY_SAMPLING=uniform|metropolis`) :
/// sert à comparer les deux échantillonneurs sur une même scène, et de porte de
/// sortie si l'un d'eux se comporte mal sur une vue particulière.
fn forced_mode() -> Option<Mode> {
    match std::env::var("FRACTALL_DENSITY_SAMPLING").ok()?.as_str() {
        "uniform" | "uniforme" => Some(Mode::Uniform),
        "metropolis" | "mh" => Some(Mode::Metropolis),
        _ => None,
    }
}

/// Sélection du régime. Publique pour que les tests puissent l'interroger.
pub(crate) fn uses_metropolis(params: &FractalParams) -> bool {
    let span_x = params
        .span_x_hp
        .as_ref()
        .and_then(|hp| Float::parse(hp).ok())
        .map(|parsed| Float::with_val(128, parsed))
        .unwrap_or_else(|| Float::with_val(128, params.span_x));
    if !span_x.is_finite() || span_x <= 0 {
        return false;
    }
    Float::with_val(128, SAMPLE_SPAN_X) / span_x > METROPOLIS_MIN_ZOOM
}

/// Accumule la densité des orbites dans `channels` plans.
///
/// `channel_mask` sélectionne les plans nourris par une orbite selon son
/// itération d'évasion (Nebulabrot expose trois profondeurs ; les autres types
/// renvoient toujours `1`).
pub(crate) fn accumulate(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    spec: &OrbitSpec,
    num_samples: usize,
    channels: usize,
    channel_mask: fn(u32) -> u32,
    use_mpc: bool,
) -> Option<Vec<Vec<f64>>> {
    accumulate_with_mode(
        params,
        cancel,
        spec,
        num_samples,
        channels,
        channel_mask,
        use_mpc,
        Mode::Auto,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_with_mode(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    spec: &OrbitSpec,
    num_samples: usize,
    channels: usize,
    channel_mask: fn(u32) -> u32,
    use_mpc: bool,
    mode: Mode,
) -> Option<Vec<Vec<f64>>> {
    let width = params.width as usize;
    let height = params.height as usize;
    let density = Density::new(channels, width, height);

    let completed = if use_mpc {
        let sampler = MpcSampler::new(params, spec);
        let geometry = AnnealGeometry::new(params, Some(&sampler));
        drive(
            &sampler,
            &geometry,
            params,
            cancel,
            spec,
            num_samples,
            channel_mask,
            &density,
            mode,
        )
    } else {
        let sampler = F64Sampler::new(params);
        let geometry = AnnealGeometry::new(params, None);
        drive(
            &sampler,
            &geometry,
            params,
            cancel,
            spec,
            num_samples,
            channel_mask,
            &density,
            mode,
        )
    };

    completed.then(|| density.into_mass())
}

#[allow(clippy::too_many_arguments)]
fn drive<S: Sampler>(
    sampler: &S,
    geometry: &AnnealGeometry,
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    spec: &OrbitSpec,
    num_samples: usize,
    channel_mask: fn(u32) -> u32,
    density: &Density,
    mode: Mode,
) -> bool {
    let metropolis = match forced_mode().unwrap_or(mode) {
        Mode::Auto => uses_metropolis(params),
        Mode::Uniform => false,
        Mode::Metropolis => true,
    };
    if metropolis && geometry.is_usable() {
        run_metropolis(
            sampler,
            geometry,
            cancel,
            spec,
            num_samples,
            channel_mask,
            density,
        )
    } else {
        run_uniform(sampler, cancel, spec, num_samples, channel_mask, density)
    }
}

/// Régime historique : un tirage uniforme par échantillon, poids unitaire.
fn run_uniform<S: Sampler>(
    sampler: &S,
    cancel: &Arc<AtomicBool>,
    spec: &OrbitSpec,
    num_samples: usize,
    channel_mask: fn(u32) -> u32,
    density: &Density,
) -> bool {
    let cancelled = AtomicBool::new(false);

    (0..num_samples).into_par_iter().for_each(|sample_idx| {
        if sample_idx % 10_000 == 0 && cancel.load(Ordering::Relaxed) {
            cancelled.store(true, Ordering::Relaxed);
            return;
        }
        if cancelled.load(Ordering::Relaxed) {
            return;
        }

        let mut rng = Rng::new((sample_idx as u32).wrapping_mul(12345).wrapping_add(42));
        let c = sampler.uniform_legacy(&mut rng);

        TRAJECTORY.with(|buf| {
            let mut trajectory = buf.borrow_mut();
            let trace = sampler.trace(&c, spec, &mut trajectory);
            if trace.kept {
                density.splat(&trajectory, WEIGHT_SCALE, channel_mask(trace.escape_iter));
            }
        });
    });

    !cancelled.load(Ordering::Relaxed)
}

/// État courant d'une chaîne.
#[derive(Clone)]
struct ChainState<P> {
    point: P,
    trajectory: Trajectory,
    escape_iter: u32,
    /// Contribution à la fenêtre de l'étape courante.
    contribution: Score,
    /// Échelle de mutation, en fenêtres de l'étape courante. Adaptée au fil des
    /// pas : la sensibilité `dz/dc` varie de plusieurs ordres de grandeur d'une
    /// scène à l'autre, aucun rayon fixe ne convient partout.
    radius_scale: f64,
}

/// Régime Metropolis-Hastings : amorçage recuit par population de chaînes,
/// puis échantillonnage pondéré `1/contribution`.
///
/// L'amorçage progresse par étapes SYNCHRONISÉES. Une chaîne qui perd toute
/// contribution en resserrant la fenêtre est perdue pour de bon — mesuré ~25 %
/// d'attrition par étape, soit zéro survivant au bout d'une dizaine d'octaves.
/// À chaque barrière, les chaînes mortes sont donc **ré-échantillonnées** sur
/// une survivante (Monte-Carlo séquentiel) : la population reste entière
/// jusqu'à la vue cible, à n'importe quelle profondeur.
fn run_metropolis<S: Sampler>(
    sampler: &S,
    geometry: &AnnealGeometry,
    cancel: &Arc<AtomicBool>,
    spec: &OrbitSpec,
    num_samples: usize,
    channel_mask: fn(u32) -> u32,
    density: &Density,
) -> bool {
    let plan = MetropolisPlan::new(num_samples, geometry.cover_x.max(geometry.cover_y));
    let (width, height) = (density.width, density.height);
    let final_window = target_window(width, height);
    let cancelled = AtomicBool::new(false);

    let mut states: Vec<Option<ChainState<S::Point>>> = vec![None; plan.chains];
    let mut rngs: Vec<ChainRng> = (0..plan.chains)
        .map(|chain| ChainRng::new(chain as u64))
        .collect();

    for stage in 0..plan.stages {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        let t = stage as f64 / (plan.stages - 1).max(1) as f64;
        let window = geometry.window(t, width, height);
        let stage_span = geometry.span_at(t);

        states
            .par_iter_mut()
            .zip(rngs.par_iter_mut())
            .for_each(|(state, rng)| {
                if cancelled.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(state) = state.as_mut() {
                    state.contribution = window.score(&state.trajectory);
                }
                TRAJECTORY.with(|buf| {
                    step_chain(
                        sampler,
                        spec,
                        &window,
                        stage_span,
                        plan.steps_per_stage,
                        rng,
                        state,
                        &mut buf.borrow_mut(),
                        None,
                        cancel,
                    );
                });
            });

        if std::env::var("FRACTALL_MH_DEBUG").is_ok() {
            let alive = states
                .iter()
                .filter(|state| {
                    state
                        .as_ref()
                        .is_some_and(|state| state.contribution.hits > 0)
                })
                .count();
            eprintln!(
                "[MH] étape {stage}/{} chaînes_vivantes={alive}/{} fenêtre={stage_span:.3e} spans",
                plan.stages, plan.chains
            );
        }
        if !resample(&mut states) {
            // Aucune chaîne ne contribue à cette échelle : la fenêtre visée est
            // vide de densité, il n'y a rien à tracer.
            return !cancelled.load(Ordering::Relaxed);
        }
    }

    states
        .par_iter_mut()
        .zip(rngs.par_iter_mut())
        .for_each(|(state, rng)| {
            if cancel.load(Ordering::Relaxed) {
                cancelled.store(true, Ordering::Relaxed);
                return;
            }
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            if let Some(state) = state.as_mut() {
                state.contribution = final_window.score(&state.trajectory);
            }
            TRAJECTORY.with(|buf| {
                let mut trajectory = buf.borrow_mut();
                // Rodage : le noyau finit de s'ajuster à la fenêtre CIBLE avant
                // qu'on ne dépose quoi que ce soit. Sans lui, une chaîne au
                // rayon mal réglé reste bloquée des milliers de pas sur le même
                // état — dont elle redépose la masse à chaque pas, ce qui crible
                // l'image de points chauds.
                step_chain(
                    sampler,
                    spec,
                    &final_window,
                    1.0,
                    plan.burn_in,
                    rng,
                    state,
                    &mut trajectory,
                    None,
                    cancel,
                );
                step_chain(
                    sampler,
                    spec,
                    &final_window,
                    1.0,
                    plan.chain_steps,
                    rng,
                    state,
                    &mut trajectory,
                    Some((density, channel_mask)),
                    cancel,
                );
            });
        });

    !cancelled.load(Ordering::Relaxed)
}

/// Remplace les chaînes éteintes par une copie d'une survivante. La source est
/// choisie par une fonction de l'indice, pas par un tirage : la population
/// reconstituée ne dépend donc pas de l'ordre d'exécution.
///
/// Renvoie `false` si plus aucune chaîne ne contribue.
fn resample<P: Clone>(states: &mut [Option<ChainState<P>>]) -> bool {
    let alive: Vec<usize> = states
        .iter()
        .enumerate()
        .filter(|(_, state)| {
            state
                .as_ref()
                .is_some_and(|state| state.contribution.hits > 0)
        })
        .map(|(index, _)| index)
        .collect();
    if alive.is_empty() {
        return false;
    }
    for index in 0..states.len() {
        let dead = states[index]
            .as_ref()
            .is_none_or(|state| state.contribution.hits == 0);
        if dead {
            let source = alive[index.wrapping_mul(2_654_435_761) % alive.len()];
            states[index] = states[source].clone();
        }
    }
    true
}

/// Avance la chaîne de `steps` pas sur `window`. Quand `sink` est fourni,
/// l'état courant est projeté à CHAQUE pas — accepté ou non : c'est l'état de
/// la chaîne, pas la proposition, qui échantillonne la distribution cible.
#[allow(clippy::too_many_arguments)]
fn step_chain<S: Sampler>(
    sampler: &S,
    spec: &OrbitSpec,
    window: &Window,
    radius_spans: f64,
    steps: usize,
    rng: &mut ChainRng,
    current: &mut Option<ChainState<S::Point>>,
    proposal: &mut Trajectory,
    sink: DensitySink<'_>,
    cancel: &Arc<AtomicBool>,
) {
    /// Fraction des propositions tirées à neuf sur tout le domaine : évite
    /// qu'une chaîne reste piégée dans un bassin isolé.
    const RESTART_RATE: f64 = 0.1;
    /// Adaptation de l'échelle de mutation (Robbins-Monro) : on grandit après
    /// une acceptation, on rétrécit après un rejet. Le point fixe est un taux
    /// d'acceptation d'environ 18 %, dans la plage utile d'une marche
    /// aléatoire. L'adaptation doit être RAPIDE : chaque étape d'amorçage
    /// resserre la fenêtre, et une chaîne n'y dispose que de quelques dizaines
    /// de pas pour retrouver la bonne échelle.
    const GROW: f64 = 1.4;
    const SHRINK: f64 = 0.93;
    const SCALE_BOUNDS: (f64, f64) = (1.0e-12, 8.0);

    for step in 0..steps {
        // Une chaîne profonde peut occuper plusieurs secondes en MPC : elle
        // doit rester interruptible sans attendre sa fin.
        if step % 64 == 0 && cancel.load(Ordering::Relaxed) {
            return;
        }
        let scale = current.as_ref().map_or(1.0, |state| state.radius_scale);
        let restart = current.is_none() || rng.next_f64() < RESTART_RATE;
        let candidate = match (restart, current.as_ref()) {
            (false, Some(state)) => sampler.mutate(&state.point, radius_spans * scale, rng),
            _ => sampler.uniform_chain(rng),
        };

        let trace = sampler.trace(&candidate, spec, proposal);
        let contribution = if trace.kept {
            window.score(proposal)
        } else {
            Score::default()
        };

        // Proposition symétrique (déplacement isotrope ou tirage uniforme) :
        // le rapport des probabilités de transition vaut 1, l'acceptation se
        // réduit au rapport des contributions. Une chaîne sans contribution
        // accepte le premier candidat viable — c'est son amorçage.
        let accept = match current.as_ref() {
            None => contribution.soft > 0.0,
            Some(state) if state.contribution.soft <= 0.0 => contribution.soft > 0.0,
            Some(state) => {
                contribution.soft >= state.contribution.soft
                    || contribution.soft / state.contribution.soft > rng.next_f64()
            }
        };

        if accept {
            if let Some(state) = current.as_mut() {
                std::mem::swap(&mut state.trajectory, proposal);
                state.point = candidate;
                state.escape_iter = trace.escape_iter;
                state.contribution = contribution;
            } else {
                *current = Some(ChainState {
                    point: candidate,
                    trajectory: std::mem::take(proposal),
                    escape_iter: trace.escape_iter,
                    contribution,
                    radius_scale: 1.0,
                });
            }
        }

        // ⚠️ L'adaptation est réservée à l'amorçage. Un noyau qui s'ajuste au
        // taux d'acceptation local égalise le pas de la marche DANS LA FENÊTRE,
        // c'est-à-dire qu'il annule exactement le jacobien `dz/dc` — or c'est
        // lui qui fait la densité. Mesuré : image quasi uniforme et corrélation
        // 0,06 avec le tirage uniforme dès ×40. Pendant la mesure, le noyau est
        // donc figé (il reste propre à chaque chaîne, ce qui préserve la
        // réversibilité de chacune).
        if sink.is_none() {
            if let Some(state) = current.as_mut() {
                let adjusted = state.radius_scale * if accept { GROW } else { SHRINK };
                state.radius_scale = adjusted.clamp(SCALE_BOUNDS.0, SCALE_BOUNDS.1);
            }
        }

        if let (Some((density, mask)), Some(state)) = (sink, current.as_ref()) {
            if state.contribution.hits > 0 {
                // Poids 1/f : chaque pas dépose au plus une unité de masse au
                // total, ce qui corrige le biais d'échantillonnage de la
                // chaîne. Un état qui touche la fenêtre a `soft ≥ 1`, donc un
                // poids borné.
                let weight = (WEIGHT_SCALE as f64 / state.contribution.soft).round() as u64;
                density.splat(
                    &state.trajectory,
                    weight.min(WEIGHT_SCALE),
                    mask(state.escape_iter),
                );
            }
        }
    }
}

thread_local! {
    static TRAJECTORY: std::cell::RefCell<Trajectory> =
        const { std::cell::RefCell::new(Vec::new()) };
}
