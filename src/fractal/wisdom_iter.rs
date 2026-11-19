//! Le sage choisit le **nombre d'itérations** (`iteration_max`) à la place de
//! l'utilisateur novice.
//!
//! # Pourquoi une mesure et pas une formule
//!
//! Une formule « itérations = f(profondeur) » se trompe trop souvent d'un
//! ordre de grandeur : à profondeur égale, une vue sur un minibrot réclame
//! 30 000 itérations, une autre 2 millions. On MESURE donc la vue : on la rend
//! en petit et on regarde la distribution des itérations d'évasion.
//!
//! # Ce qu'est un plafond « suffisant »
//!
//! Le plafond `C` est suffisant si le rendre ×4 plus haut ne change presque
//! rien à l'image : la fraction de pixels dont la classification
//! (évadé / plafonné) change entre `C` et `4C` doit rester sous
//! [`TARGET_CHANGE`]. Cette fraction est EXTRAPOLÉE de la queue de la
//! distribution des évasions, sans rendu supplémentaire (cf.
//! [`predicted_change`]) : les pixels de bord s'évadent de plus en plus tard
//! selon une loi de puissance (masse par octave de « surplus » `it − m`, `m`
//! = quantile bas des évasions, qui absorbe le décalage des zooms profonds où
//! rien ne s'évade avant 30 000 itérations). Les pixels intérieurs, eux, ne
//! s'évadent jamais : ils ne comptent pas, seule la queue observée compte.
//!
//! # Deux couches
//!
//! 1. **Pure, par image** — [`IterationStats::from_iterations`] puis
//!    [`assess_iterations`] : la GUI l'appelle sur CHAQUE image terminée (le
//!    buffer `iterations` qu'elle vient de recevoir, rendu au plafond
//!    `params.iteration_max`) et applique le verdict pour le rendu suivant.
//!    Hystérésis : on ne grandit qu'au-delà de [`GROW_THRESHOLD`] (et alors
//!    jusqu'à viser [`TARGET_CHANGE`]), on ne rétrécit que si un plafond ≤
//!    `C/`[`SHRINK_FACTOR`] satisfait déjà la cible — pas d'oscillation en
//!    zoom continu.
//! 2. **Sonde ponctuelle** — [`auto_iterations`] : pour le CLI et le
//!    démarrage à froid de la GUI. Rend la vue en ~4 000 pixels par le
//!    dispatcher UNIQUE (`render::render_request`, CPU), applique la couche
//!    1, recommence avec un plafond plus haut si besoin.
//!
//! # Types concernés
//!
//! [`auto_iterations_applicable`] : les types « escape-time » dont un pixel
//! non évadé au plafond est noir — familles Mandelbrot/Julia bytecode et
//! perturbation (Mandelbrot, Julia, Burning Ship, Tricorn, Celtic, Buffalo,
//! Perpendicular BS, Multibrot et leurs variantes Julia), hybrides/opcodes,
//! Alpha Mandelbrot (+Julia), Mandelbrot Sin / Julia Sin, Phoenix,
//! Barnsley (Mandelbrot + Julia). NON applicables (le type garde son défaut) :
//! Von Koch / Dragon (profondeur de récursion), Buddhabrot / Nebulabrot /
//! Anti-Buddhabrot (densité : le plafond est une longueur d'orbite), Lyapunov
//! (exposant), Newton / Nova / Magnet (convergence vers une racine : le
//! compte mesure la convergence, pas l'évasion), Mandelbulb (3D), Pickover
//! Stalks (colorisation par piège).
//!
//! Aucun champ `FractalParams` n'est ajouté : « auto » est un état CLI/GUI,
//! seule la valeur résolue est stockée dans `iteration_max`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::fractal::perturbation::{compute_perturbation_precision_bits, log2_zoom, ReferenceOrbitCache};
use crate::fractal::types::{FractalParams, FractalType, OutColoringMode};
use crate::fractal::wisdom::{select_algorithm, Algorithm, Device};
use crate::render::{render_request, RenderRequest};

/// Fraction de pixels dont la classification peut changer entre `C` et `4C`
/// que l'on vise. Calibrée sur le corpus `toml/` (cf.
/// `examples/calibrate_auto_iter.rs`) : la prédiction vise 0,2 %, la mesure
/// réelle reste sous ~0,5 % sur la quasi-totalité du corpus.
pub const TARGET_CHANGE: f64 = 0.002;

/// Seuil de CROISSANCE de la couche par image (GUI) : on ne relève le
/// plafond qu'au-delà de 0,5 % prédits — 2,5 × la cible, de sorte qu'après
/// une croissance (qui vise [`TARGET_CHANGE`]) le zoom doive approfondir
/// nettement la vue avant la suivante.
pub const GROW_THRESHOLD: f64 = 0.005;

/// On ne RÉTRÉCIT que si un plafond ≤ `C / SHRINK_FACTOR` satisfait déjà la
/// cible : un facteur ×4 de marge évite le va-et-vient croissance/réduction.
pub const SHRINK_FACTOR: u32 = 4;

/// Plancher absolu du plafond automatique.
pub const MIN_ITERATIONS: u32 = 100;

/// Facteur de vérification : « suffisant » = l'image ne change pas jusqu'à
/// `CHECK_FACTOR × C`.
pub const CHECK_FACTOR: f64 = 4.0;

/// Le plafond recommandé ne descend jamais sous `TAIL_MARGIN ×` le quantile
/// [`TAIL_QUANTILE`] des évasions observées (cf. [`assess_iterations`]).
pub const TAIL_MARGIN: f64 = 1.25;

/// Relèvement maximal du plafond suffisant par la marge de queue observée.
pub const TAIL_MARGIN_MAX_LIFT: f64 = 2.0;

/// Quantile de la queue observée protégé par [`TAIL_MARGIN`].
pub const TAIL_QUANTILE: f64 = 0.999;

/// Quantile bas des évasions servant d'origine au « surplus » `it − m`.
const BASE_QUANTILE: f64 = 0.01;

/// Taille max de l'échantillon trié d'évasions (les buffers GUI de plusieurs
/// mégapixels sont sous-échantillonnés à pas fixe — déterministe).
const MAX_SAMPLE: usize = 1 << 16;

/// Le type a-t-il un plafond d'itérations au sens « escape-time » ? Cf.
/// l'en-tête du module pour la liste et le pourquoi. `false` ⇒ l'appelant
/// garde le défaut du type.
pub fn auto_iterations_applicable(params: &FractalParams) -> bool {
    use FractalType::*;
    if params.is_hybrid_formula() {
        return true;
    }
    matches!(
        params.fractal_type,
        Mandelbrot
            | Julia
            | BurningShip
            | Tricorn
            | Celtic
            | Buffalo
            | PerpendicularBurningShip
            | Multibrot
            | BurningShipJulia
            | TricornJulia
            | CelticJulia
            | BuffaloJulia
            | MultibrotJulia
            | PerpendicularBurningShipJulia
            | AlphaMandelbrot
            | AlphaMandelbrotJulia
            | MandelbrotSin
            | JuliaSin
            | Phoenix
            | BarnsleyMandelbrot
            | BarnsleyJulia
    )
}

/// Statistiques d'un buffer d'itérations rendu au plafond `cap`.
#[derive(Debug, Clone, PartialEq)]
pub struct IterationStats {
    /// Plafond auquel le buffer a été rendu.
    pub cap: u32,
    /// Nombre total de pixels.
    pub total: usize,
    /// Pixels non évadés au plafond (`it >= cap`).
    pub capped: usize,
    /// Pixels évadés (`it < cap`).
    pub escaped: usize,
    /// Itération d'évasion maximale observée (0 si aucune).
    pub max_escaped: u32,
    /// Échantillon TRIÉ des itérations d'évasion (≤ 65 536 valeurs, pas fixe).
    sample: Vec<u32>,
}

impl IterationStats {
    /// Construit les statistiques depuis le canal `iterations` d'un rendu au
    /// plafond `cap` (= `params.iteration_max` de ce rendu). Coût : un
    /// parcours + tri d'au plus 65 536 valeurs (~1 ms).
    pub fn from_iterations(iterations: &[u32], cap: u32) -> Self {
        let cap = cap.max(1);
        let escaped = iterations.iter().filter(|&&it| it < cap).count();
        let stride = escaped.div_ceil(MAX_SAMPLE).max(1);
        let mut sample: Vec<u32> = iterations
            .iter()
            .copied()
            .filter(|&it| it < cap)
            .step_by(stride)
            .collect();
        sample.sort_unstable();
        let max_escaped = iterations
            .iter()
            .copied()
            .filter(|&it| it < cap)
            .max()
            .unwrap_or(0);
        Self {
            cap,
            total: iterations.len(),
            capped: iterations.len() - escaped,
            escaped,
            max_escaped,
            sample,
        }
    }

    /// Fraction de pixels évadés.
    pub fn escaped_fraction(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.escaped as f64 / self.total as f64
        }
    }

    /// Quantile `q ∈ [0,1]` des itérations d'évasion (`None` sans évasion).
    pub fn escaped_quantile(&self, q: f64) -> Option<u32> {
        if self.sample.is_empty() {
            return None;
        }
        let idx = ((self.sample.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
        Some(self.sample[idx])
    }

    /// Poids d'un élément de l'échantillon en pixels.
    fn weight(&self) -> f64 {
        if self.sample.is_empty() {
            0.0
        } else {
            self.escaped as f64 / self.sample.len() as f64
        }
    }

    /// Nombre d'éléments de l'échantillon `< x`.
    fn below(&self, x: f64) -> usize {
        self.sample.partition_point(|&v| (v as f64) < x)
    }
}

/// Fraction de pixels (de 0 à 1) dont la classification évadé/plafonné
/// changerait entre le plafond `cap` et `CHECK_FACTOR × cap`, estimée avec
/// l'information qu'un rendu au plafond `cap` aurait, plus les évasions
/// CONNUES du buffer entre `cap` et `4·cap` (plancher de la prédiction quand
/// `cap < stats.cap`). Exige `cap ≤ stats.cap`.
///
/// Modèle : surplus `e = it − m` (`m` = quantile 1 % des évasions `< cap`),
/// `E = cap − m` ; masses `n1` sur `[E/4, E/2)` et `n2` sur `[E/2, E)` ; ratio
/// par octave `r = (n2+½)/(n1+½)` plafonné à 1 (une queue encore montante
/// est prolongée à plat, faute de mieux) ; masse prédite au-delà = `n2 ·
/// Σ_{j=1..k} r^j` sur les `k = log2((4·cap − m)/E)` octaves jusqu'à `4·cap`,
/// bornée par le nombre de pixels plafonnés.
pub fn predicted_change(stats: &IterationStats, cap: u32) -> f64 {
    debug_assert!(cap <= stats.cap);
    if stats.total == 0 {
        return 0.0;
    }
    let cap_f = cap as f64;
    let n_below = stats.below(cap_f);
    let w = stats.weight();
    let capped_at = stats.total as f64 - n_below as f64 * w;
    if capped_at <= 0.0 {
        return 0.0;
    }
    if n_below == 0 {
        // Rien d'évadé sous `cap` : aucune queue à extrapoler. Si le buffer
        // montre des évasions au-delà, `cap` est trop bas ; sinon (vue
        // intérieure) on ne peut rien dire — cas `no_escape` de l'appelant.
        return if stats.escaped > 0 {
            1.0f64.min(capped_at / stats.total as f64)
        } else {
            0.0
        };
    }
    // Évasions CONNUES entre `cap` et `4·cap` (quand `cap < stats.cap`) :
    // la prédiction ne peut pas valoir moins.
    let known = (stats.below((CHECK_FACTOR * cap_f).min(stats.cap as f64)) - n_below) as f64 * w;
    let base_idx = ((n_below - 1) as f64 * BASE_QUANTILE).floor() as usize;
    let m = stats.sample[base_idx] as f64;
    let e = (cap_f - m).max(1.0);
    let count = |lo: f64, hi: f64| -> f64 {
        let a = stats.below(m + lo);
        let b = stats.below((m + hi).min(cap_f));
        b.saturating_sub(a) as f64
    };
    let n2 = count(e / 2.0, e);
    let n1 = count(e / 4.0, e / 2.0);
    let r = ((n2 + 0.5) / (n1 + 0.5)).min(1.0);
    let k = ((CHECK_FACTOR * cap_f - m) / e).max(1.0).log2();
    // Σ_{j=1..k} r^j, k réel.
    let sum = if r >= 1.0 - 1e-12 {
        k
    } else {
        r * (1.0 - r.powf(k)) / (1.0 - r)
    };
    let pred = (n2 * w * sum).min(capped_at).max(known);
    pred / stats.total as f64
}

/// Verdict de la couche par image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationVerdict {
    /// Garder le plafond.
    Sufficient,
    /// Relever le plafond à cette valeur (et re-rendre).
    Grow(u32),
    /// Le plafond peut descendre à cette valeur sans changer l'image.
    Shrink(u32),
}

/// Diagnostic complet de [`assess_iterations`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterationAdvice {
    pub verdict: IterationVerdict,
    /// [`predicted_change`] au plafond courant (fraction 0..1).
    pub predicted_change: f64,
    /// Plus petit plafond satisfaisant [`TARGET_CHANGE`] d'après les données
    /// (≤ plafond courant si suffisant ; extrapolé au-delà sinon).
    pub recommended: u32,
    /// Aucun pixel ne s'est évadé : vue entièrement intérieure OU plafond
    /// bien trop bas (zoom profond) — indiscernables sur ce seul buffer.
    pub no_escape: bool,
    /// La queue des évasions est lourde ([`HEAVY_TAIL_RATIO`]) : chaque ×4
    /// de plafond n'achète presque rien (type « fuyant »).
    pub heavy_tail: bool,
}

/// Ratio de masse par octave de la queue observée sous `stats.cap`
/// (`n2 / n1`, cf. [`predicted_change`]) ; `None` sans évasion.
fn tail_ratio(stats: &IterationStats) -> Option<f64> {
    let cap_f = stats.cap as f64;
    let n_below = stats.below(cap_f);
    if n_below == 0 {
        return None;
    }
    let base_idx = ((n_below - 1) as f64 * BASE_QUANTILE).floor() as usize;
    let m = stats.sample[base_idx] as f64;
    let e = (cap_f - m).max(1.0);
    let count = |lo: f64, hi: f64| -> f64 {
        stats
            .below((m + hi).min(cap_f))
            .saturating_sub(stats.below(m + lo)) as f64
    };
    Some((count(e / 2.0, e) + 0.5) / (count(e / 4.0, e / 2.0) + 0.5))
}

/// Le plafond est-il DANS la queue ? Vrai si le surplus du plafond vaut au
/// moins [`IN_TAIL_FACTOR`] fois le surplus médian des évasions. Tant qu'il
/// est encore dans le gros de la distribution, un ratio par octave proche de
/// 1 ne dit rien de la queue (seahorse 1e7 au plafond de départ 2 500 :
/// médiane ~1 000, `r` ≥ 0,7, puis 0,3-0,5 dès 10 000).
fn cap_is_in_tail(stats: &IterationStats) -> bool {
    let cap_f = stats.cap as f64;
    let n_below = stats.below(cap_f);
    if n_below < 2 {
        return false;
    }
    let m = stats.sample[((n_below - 1) as f64 * BASE_QUANTILE).floor() as usize] as f64;
    let median = stats.sample[(n_below - 1) / 2] as f64;
    cap_f - m >= IN_TAIL_FACTOR * (median - m).max(1.0)
}

/// Queue lourde : plafond dans la queue ET masse par octave qui décroît de
/// moins de 30 % ([`HEAVY_TAIL_RATIO`]).
fn tail_is_heavy(stats: &IterationStats) -> bool {
    cap_is_in_tail(stats) && tail_ratio(stats).is_some_and(|r| r >= HEAVY_TAIL_RATIO)
}

/// Arrondit vers le haut à 2 chiffres significatifs (34 565 → 35 000).
pub fn round_up_nice(n: u64) -> u32 {
    if n <= 100 {
        return n.max(1) as u32;
    }
    let digits = (n as f64).log10().floor() as u32;
    let unit = 10u64.pow(digits - 1);
    (n.div_ceil(unit) * unit).min(u32::MAX as u64) as u32
}

/// Plus petit plafond (grille géométrique ×2^(1/4) sous `stats.cap`) dont
/// la prédiction tronquée tient la cible — et dont toutes les valeurs de la
/// grille au-dessus la tiennent aussi.
fn smallest_sufficient(stats: &IterationStats) -> u32 {
    let mut best = stats.cap;
    let step = 2f64.powf(-0.25);
    let mut c = stats.cap as f64 * step;
    loop {
        let ci = (c.ceil() as u32).max(MIN_ITERATIONS);
        if ci >= best || predicted_change(stats, ci) > TARGET_CHANGE {
            break;
        }
        best = ci;
        if ci == MIN_ITERATIONS {
            break;
        }
        c *= step;
    }
    best.max(MIN_ITERATIONS.min(stats.cap))
}

/// Ratio par octave au-delà duquel la queue est « lourde » : l'extrapolation
/// géométrique `Σ r^j ≈ 1/(1−r)` y devient si sensible au bruit sur `r`
/// qu'elle ne vaut plus rien. Mesuré le 2026-11-19 (axe `wisdom` du
/// harness) : Barnsley Mandelbrot ×5 (queue 0,52 → 0,43 → 0,24 → 0,23 →
/// 0,17 → 0,09 % par octave, `r` ≈ 0,8) — la sonde sautait à 2,9 M
/// itérations (37 s à 128×80) quand son propre critère est satisfait vers
/// 150 000 ; la marge [`TAIL_MARGIN`] sur la queue OBSERVÉE rend ensuite le
/// dépassement irréversible. Seahorse / éléphants décroissent d'un facteur
/// ≥ 2 par octave (`r` ≤ 0,5) : non concernés.
pub const HEAVY_TAIL_RATIO: f64 = 0.7;

/// Rapport surplus du plafond / surplus médian à partir duquel le plafond
/// est « dans la queue » (cf. `cap_is_in_tail`).
pub const IN_TAIL_FACTOR: f64 = 8.0;

/// Croissance maximale par tour de la SONDE sur une queue lourde : on avance
/// par ×4 et on REMESURE, au lieu d'extrapoler.
pub const HEAVY_TAIL_GROWTH: u32 = 4;

/// Plafond > `stats.cap` à viser quand la prédiction dépasse la cible :
/// prolonge le modèle de queue sur une grille ×2^(1/4) (jusqu'à ×64), au
/// moins ×1,5 ; ×4 si la queue ne décroît pas encore (impossible à
/// extrapoler).
fn grow_target(stats: &IterationStats) -> u32 {
    let cap_f = stats.cap as f64;
    let n_below = stats.below(cap_f);
    if n_below == 0 {
        return (cap_f * 4.0).min(u32::MAX as f64) as u32;
    }
    let w = stats.weight();
    let base_idx = ((n_below - 1) as f64 * BASE_QUANTILE).floor() as usize;
    let m = stats.sample[base_idx] as f64;
    let e = (cap_f - m).max(1.0);
    let count = |lo: f64, hi: f64| -> f64 {
        stats
            .below((m + hi).min(cap_f))
            .saturating_sub(stats.below(m + lo)) as f64
    };
    let n2 = count(e / 2.0, e);
    let n1 = count(e / 4.0, e / 2.0);
    let r = (n2 + 0.5) / (n1 + 0.5);
    if r >= 0.95 {
        return (cap_f * 4.0).min(u32::MAX as f64) as u32;
    }
    let total = stats.total as f64;
    let capped = stats.capped as f64;
    let step = 2f64.powf(0.25);
    let mut c = cap_f * 1.5;
    while c <= cap_f * 64.0 {
        // Surplus à ce plafond hypothétique, masse de l'octave courante.
        let e2 = (c - m).max(1.0);
        let octaves = (e2 / e).log2();
        let n2c = n2 * r.powf(octaves);
        // La masse déjà « consommée » entre cap et c sort des plafonnés.
        let consumed = n2 * r * (1.0 - r.powf(octaves)) / (1.0 - r);
        let k = ((CHECK_FACTOR * c - m) / e2).max(1.0).log2();
        let sum = r * (1.0 - r.powf(k)) / (1.0 - r);
        let pred = (n2c * w * sum).min((capped - consumed * w).max(0.0));
        if pred / total <= TARGET_CHANGE {
            return c.ceil() as u32;
        }
        c *= step;
    }
    (cap_f * 64.0).min(u32::MAX as f64) as u32
}

/// Couche 1 : verdict sur un buffer rendu au plafond `stats.cap`.
///
/// La GUI l'appelle après chaque image terminée (pleine passe CPU ou GPU,
/// pas une passe intermédiaire basse résolution — elle marcherait, mais avec
/// plus de bruit) : `Grow(n)` ⇒ `params.iteration_max = n` et re-rendu (la
/// vue est alors visiblement incomplète) ; `Shrink(n)` ⇒ appliquer au rendu
/// SUIVANT (l'image courante est juste, seulement plus chère que
/// nécessaire) ; `Sufficient` ⇒ rien.
pub fn assess_iterations(stats: &IterationStats) -> IterationAdvice {
    if stats.escaped == 0 {
        return IterationAdvice {
            verdict: IterationVerdict::Sufficient,
            predicted_change: 0.0,
            recommended: stats.cap,
            no_escape: stats.total > 0,
            heavy_tail: false,
        };
    }
    let now = predicted_change(stats, stats.cap);
    if now > TARGET_CHANGE {
        let to = round_up_nice(grow_target(stats) as u64).max(stats.cap.saturating_add(1));
        let verdict = if now > GROW_THRESHOLD {
            IterationVerdict::Grow(to)
        } else {
            IterationVerdict::Sufficient
        };
        return IterationAdvice {
            verdict,
            predicted_change: now,
            recommended: to,
            no_escape: false,
            heavy_tail: tail_is_heavy(stats),
        };
    }
    // Marge sur la queue OBSERVÉE : jamais sous `TAIL_MARGIN × q99,9 %` des
    // évasions — une image pleine résolution échantillonne la queue plus loin
    // qu'une sonde de 4 000 pixels, et ces derniers pixels de bord restent
    // ainsi colorés au lieu de virer au noir.
    //
    // La marge ne relève le plafond SUFFISANT que d'un facteur
    // [`TAIL_MARGIN_MAX_LIFT`] au plus : sur une queue lourde le quantile
    // observé suit le plafond sondé, et un tour qui a dépassé le besoin
    // rendait ce dépassement irréversible (seahorse 1e7 : sonde à 640 k ⇒
    // 490 k choisis quand 90 k tiennent la cible ; Barnsley : 2,9 M).
    let suff = smallest_sufficient(stats) as f64;
    let tail = (stats.escaped_quantile(TAIL_QUANTILE).unwrap_or(0) as f64 * TAIL_MARGIN)
        .min(suff * TAIL_MARGIN_MAX_LIFT);
    let rec = round_up_nice(suff.max(tail).ceil() as u64).min(stats.cap);
    let verdict = if (rec as u64) * (SHRINK_FACTOR as u64) <= stats.cap as u64 {
        IterationVerdict::Shrink(rec)
    } else {
        IterationVerdict::Sufficient
    };
    IterationAdvice {
        verdict,
        predicted_change: now,
        recommended: rec,
        no_escape: false,
        heavy_tail: false,
    }
}

/// Plafond de départ maximal : au-delà de 1e1000 la formule de profondeur
/// sur-estime (e8000 : départ 1 M pour un besoin de 140 k, l'orbite de
/// 27 000 bits itérée jusqu'au plafond = sonde de 3,3 s au lieu de ~1 s ;
/// e22522 : 5,6 M pour 0,8 M). Une vue dont rien ne s'évade avant le départ
/// (e1298, infinity : ~0,7 M) est rattrapée par les croissances
/// [`MIN_NO_ESCAPE_GROWTHS`].
pub const MAX_START: u32 = 250_000;

/// Croissances ×4 toujours tentées quand rien ne s'évade, avant de consulter
/// le budget de temps : sur e1298 / infinity / e227 rien ne s'évade avant
/// 0,7-1 M itérations et le premier tour coûte déjà 5 s.
pub const MIN_NO_ESCAPE_GROWTHS: u32 = 2;

/// Plafond de DÉPART de la sonde : défaut du type, relevé avec la profondeur
/// (`250 · log10(zoom)` au-delà de 1e10, zoom = 4/span, borné à [`MAX_START`]). Ce n'est qu'un point
/// de départ pour limiter le nombre de tours ; la sonde mesure.
pub fn starting_iterations(params: &FractalParams) -> u32 {
    let default = crate::fractal::default_params_for_type(params.fractal_type, 64, 64)
        .iteration_max
        .max(MIN_ITERATIONS);
    let depth10 = log2_zoom(params)
        .map(|l2| (l2 - (params.height.max(1) as f64).log2()) * std::f64::consts::LOG10_2)
        .unwrap_or(0.0)
        .max(0.0);
    let depth_cap = if depth10 > 10.0 {
        (250.0 * depth10).min(MAX_START as f64)
    } else {
        0.0
    };
    round_up_nice((default as f64).max(depth_cap) as u64)
}

/// Réglages de la sonde.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoIterOptions {
    /// Pixels de l'image sonde (même rapport d'aspect que la vue).
    pub probe_pixels: u32,
    /// Plafond maximal que la sonde peut choisir.
    pub ceiling: u32,
    /// Nombre maximal de rendus sonde.
    pub max_rounds: u32,
    /// Plafond de départ ; `None` = [`starting_iterations`].
    pub start: Option<u32>,
    /// Tant qu'aucun pixel ne s'évade, la sonde multiplie le plafond par 4
    /// (zoom profond dont les évasions commencent loin) tant que son temps
    /// cumulé reste sous ce budget ; au-delà elle conclut « vue intérieure ».
    pub no_escape_budget: Duration,
}

impl Default for AutoIterOptions {
    fn default() -> Self {
        Self {
            probe_pixels: 4096,
            ceiling: 10_000_000,
            max_rounds: 7,
            start: None,
            no_escape_budget: Duration::from_secs(8),
        }
    }
}

/// Ce que la sonde a choisi et observé.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoIterResult {
    /// Plafond choisi (à écrire dans `params.iteration_max`).
    pub iteration_max: u32,
    /// Rendus sonde effectués.
    pub rounds: u32,
    /// Plafond du dernier rendu sonde.
    pub last_probe_cap: u32,
    /// Temps mural total de la sonde.
    pub elapsed: Duration,
    /// Dimensions de l'image sonde.
    pub probe_size: (u32, u32),
    /// Itération d'évasion max observée au dernier tour (0 si aucune).
    pub max_escaped: u32,
    /// Quantile 99 % des évasions au dernier tour.
    pub q99_escaped: Option<u32>,
    /// Fraction de pixels évadés au dernier tour.
    pub escaped_fraction: f64,
    /// Changement prédit au plafond choisi (fraction 0..1).
    pub predicted_change: f64,
    /// Le plafond `ceiling` a été atteint sans satisfaire la cible.
    pub ceiling_hit: bool,
    /// Aucun pixel ne s'est évadé, même après croissance (vue intérieure).
    pub no_escape: bool,
}

impl AutoIterResult {
    /// Ligne d'information pour l'utilisateur (anglais), p. ex.
    /// `Auto iterations: 40000 (max escape 34565 on a 85x48 probe, 3 rounds, 0.41 s)`.
    pub fn summary_line(&self) -> String {
        let mut s = format!(
            "Auto iterations: {} (max escape {} on a {}x{} probe, {} round{}, {:.2} s",
            self.iteration_max,
            self.max_escaped,
            self.probe_size.0,
            self.probe_size.1,
            self.rounds,
            if self.rounds == 1 { "" } else { "s" },
            self.elapsed.as_secs_f64(),
        );
        if self.ceiling_hit {
            s.push_str(", ceiling reached: the view may need more");
        }
        if self.no_escape {
            s.push_str(", no pixel escaped: interior view");
        }
        s.push(')');
        s
    }
}

/// Pourquoi la sonde n'a pas de résultat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoIterError {
    /// Type sans plafond « escape-time » : garder le défaut du type.
    NotApplicable,
    /// `cancel` levé pendant la sonde.
    Cancelled,
}

/// Dimensions de la sonde : ~`pixels` pixels au rapport d'aspect de la vue.
pub fn probe_dimensions(width: u32, height: u32, pixels: u32) -> (u32, u32) {
    let aspect = width.max(1) as f64 / height.max(1) as f64;
    let p = pixels.max(16) as f64;
    let w = (p * aspect).sqrt().round().max(4.0);
    let h = (p / w).round().max(4.0);
    (w as u32, h as u32)
}

/// Paramètres du rendu sonde : même centre / spans / formule, petite taille,
/// pas de canaux annexes, pas d'AA. Sur le path perturbation, la précision
/// de l'orbite est alignée sur celle de la vue PLEINE taille : l'orbite du
/// dernier tour reste alors valide (`ReferenceOrbitCache::is_valid_for`)
/// pour le rendu final si son plafond ≤ celui de la sonde.
fn probe_params(params: &FractalParams, w: u32, h: u32, cap: u32) -> FractalParams {
    let mut p = params.clone();
    p.width = w;
    p.height = h;
    p.iteration_max = cap;
    p.channels.enable_distance_estimation = false;
    p.channels.enable_orbit_traps = false;
    p.color.out_coloring_mode = OutColoringMode::Smooth;
    p.sampling.aa_jitter = None;
    p.sampling.aa_subpixel_offset = [0.0, 0.0];
    if select_algorithm(params, Device::Cpu) == Algorithm::Perturbation {
        p.engine.precision_bits = p
            .engine
            .precision_bits
            .max(compute_perturbation_precision_bits(params));
    }
    p
}

/// Sonde ponctuelle avec les réglages par défaut et sans cache d'orbite.
/// CLI : `match auto_iterations(&params, &cancel) { Ok(r) => { params.
/// iteration_max = r.iteration_max; eprintln!("{}", r.summary_line()) }
/// Err(NotApplicable) => {/* défaut du type */}, Err(Cancelled) => … }`.
pub fn auto_iterations(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
) -> Result<AutoIterResult, AutoIterError> {
    auto_iterations_with(params, cancel, AutoIterOptions::default(), &mut None)
}

/// Sonde complète. `orbit_cache` est passé au dispatcher à chaque tour ; la
/// GUI peut y passer le sien : à la sortie il contient l'orbite du dernier
/// tour, réutilisable par le rendu final de la même vue (même centre/span,
/// précision alignée, plafond final ≤ plafond sondé). Rendu CPU, jamais GPU.
pub fn auto_iterations_with(
    params: &FractalParams,
    cancel: &Arc<AtomicBool>,
    opts: AutoIterOptions,
    orbit_cache: &mut Option<Arc<ReferenceOrbitCache>>,
) -> Result<AutoIterResult, AutoIterError> {
    if !auto_iterations_applicable(params) {
        return Err(AutoIterError::NotApplicable);
    }
    let started = Instant::now();
    let (w, h) = probe_dimensions(params.width, params.height, opts.probe_pixels);
    let ceiling = opts.ceiling.max(MIN_ITERATIONS);
    let mut cap = opts
        .start
        .unwrap_or_else(|| starting_iterations(params))
        .clamp(MIN_ITERATIONS, ceiling);
    let first_cap = cap;
    let mut no_escape_growths = 0u32;
    let mut rounds = 0u32;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(AutoIterError::Cancelled);
        }
        let round_started = Instant::now();
        let p = probe_params(params, w, h, cap);
        let out = render_request(RenderRequest::new(&p, cancel), orbit_cache)
            .ok_or(AutoIterError::Cancelled)?;
        if cancel.load(Ordering::Relaxed) {
            return Err(AutoIterError::Cancelled);
        }
        rounds += 1;
        let stats = IterationStats::from_iterations(&out.iterations, cap);
        let advice = assess_iterations(&stats);
        let last_round = rounds >= opts.max_rounds || cap >= ceiling;

        let finish = |chosen: u32, ceiling_hit: bool, no_escape: bool, pred: f64| AutoIterResult {
            iteration_max: chosen,
            rounds,
            last_probe_cap: cap,
            elapsed: started.elapsed(),
            probe_size: (w, h),
            max_escaped: stats.max_escaped,
            q99_escaped: stats.escaped_quantile(0.99),
            escaped_fraction: stats.escaped_fraction(),
            predicted_change: pred,
            ceiling_hit,
            no_escape,
        };

        if advice.no_escape {
            // Vue intérieure, ou zoom profond dont rien ne s'évade encore
            // (e227 : aucune évasion avant ~1 M). On multiplie par 4 au moins
            // MIN_NO_ESCAPE_GROWTHS fois, puis tant que le tour suivant
            // (≈ 4 × le dernier) tient dans le budget.
            let next_cost = round_started.elapsed().as_secs_f64() * 4.0;
            if !last_round
                && (no_escape_growths < MIN_NO_ESCAPE_GROWTHS
                    || started.elapsed().as_secs_f64() + next_cost
                        <= opts.no_escape_budget.as_secs_f64())
            {
                no_escape_growths += 1;
                cap = cap.saturating_mul(4).min(ceiling);
                continue;
            }
            // Rien ne s'évade jusqu'au dernier plafond sondé : tout plafond
            // donne la même image (noire) — on rend le moins cher, le départ.
            return Ok(finish(first_cap, false, true, 0.0));
        }
        if advice.predicted_change <= TARGET_CHANGE {
            let chosen = advice.recommended.clamp(MIN_ITERATIONS.min(cap), cap);
            let pred = predicted_change(&stats, chosen);
            return Ok(finish(chosen, false, false, pred));
        }
        // Queue lourde : un ×4 de plus coûte ×4 et ne change presque rien.
        // On s'arrête au seuil de croissance de la couche par image
        // ([`GROW_THRESHOLD`], celui que la GUI applique en régime établi),
        // pas à la cible — sinon la sonde court jusqu'au plafond maximal.
        if advice.heavy_tail && rounds >= 2 && advice.predicted_change <= GROW_THRESHOLD {
            return Ok(finish(cap, false, false, advice.predicted_change));
        }
        if last_round {
            return Ok(finish(cap, cap >= ceiling, false, advice.predicted_change));
        }
        let mut next = advice.recommended.max(cap.saturating_add(1));
        if advice.heavy_tail {
            next = next.min(cap.saturating_mul(HEAVY_TAIL_GROWTH));
        }
        cap = next.min(ceiling);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::default_params_for_type;

    /// Type « fuyant » (Barnsley Mandelbrot, queue lourde) : la sonde ne
    /// court plus jusqu'au plafond maximal. Avant le 2026-11-19 cette vue
    /// donnait 2,9 M itérations (37 s à 128×80) ; son propre critère est
    /// satisfait vers 150 000.
    #[test]
    fn leaky_type_probe_stops_on_a_confirmed_heavy_tail() {
        let mut p = default_params_for_type(FractalType::BarnsleyMandelbrot, 96, 60);
        p.center_x = -0.4921875;
        p.center_y = -1.225;
        p.span_x = 0.75;
        p.span_y = 0.75 * 60.0 / 96.0;
        let cancel = Arc::new(AtomicBool::new(false));
        let r = auto_iterations(&p, &cancel).expect("sonde");
        assert!(!r.no_escape && !r.ceiling_hit);
        assert!(
            r.iteration_max <= 700_000,
            "queue lourde : la sonde s'emballe encore ({})",
            r.iteration_max
        );
        assert!(r.iteration_max >= 5_000, "plafond trop bas ({})", r.iteration_max);
        assert!(r.predicted_change <= GROW_THRESHOLD, "{}", r.predicted_change);
    }

    /// `heavy_tail` suit le ratio de masse par octave de la queue.
    #[test]
    fn heavy_tail_flag_follows_the_octave_ratio() {
        let cap = 2_500u32;
        // alpha = 0,3 : la masse par octave ne perd que 19 % (r ≈ 0,81).
        let heavy = IterationStats::from_iterations(&synthetic(40_000, 0.4, 0.0, 20.0, 0.3, cap), cap);
        let a = assess_iterations(&heavy);
        assert!(a.predicted_change > TARGET_CHANGE && a.heavy_tail);
        assert!(tail_ratio(&heavy).unwrap() >= HEAVY_TAIL_RATIO && cap_is_in_tail(&heavy));
        // alpha = 2 : la masse par octave est divisée par 4.
        let light = IterationStats::from_iterations(&synthetic(40_000, 0.4, 0.0, 300.0, 2.0, cap), cap);
        assert!(tail_ratio(&light).unwrap() < HEAVY_TAIL_RATIO);
        assert!(!assess_iterations(&light).heavy_tail);
        // Plafond encore dans le gros de la distribution : jamais « lourd »,
        // même si la masse par octave monte encore.
        let bulk: Vec<u32> = (0..10_000u32).map(|i| if i % 3 == 0 { cap } else { 600 + i % 1800 }).collect();
        let bs = IterationStats::from_iterations(&bulk, cap);
        assert!(!cap_is_in_tail(&bs) && !tail_is_heavy(&bs));
    }

    /// Buffer synthétique : `interior` pixels plafonnés + évasions en loi de
    /// puissance `P(T > t) ∝ (t/t0)^-alpha` au-delà de `offset`.
    fn synthetic(n: usize, interior: f64, offset: f64, t0: f64, alpha: f64, cap: u32) -> Vec<u32> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            if u < interior {
                v.push(cap);
                continue;
            }
            // Quantile inverse (déterministe) sur la partie évadée.
            let q = (u - interior) / (1.0 - interior);
            let t = offset + t0 * (1.0 - q).powf(-1.0 / alpha);
            v.push(if t >= cap as f64 { cap } else { t as u32 });
        }
        v
    }

    #[test]
    fn nice_rounding() {
        assert_eq!(round_up_nice(34_565), 35_000);
        assert_eq!(round_up_nice(40_000), 40_000);
        assert_eq!(round_up_nice(101), 110);
        assert_eq!(round_up_nice(57), 57);
    }

    #[test]
    fn all_exterior_is_sufficient_and_shrinks() {
        let iters: Vec<u32> = (0..4096).map(|i| 3 + (i % 50)).collect();
        let s = IterationStats::from_iterations(&iters, 10_000);
        let a = assess_iterations(&s);
        assert_eq!(a.predicted_change, 0.0);
        assert_eq!(a.verdict, IterationVerdict::Shrink(MIN_ITERATIONS));
    }

    #[test]
    fn all_interior_reports_no_escape() {
        let iters = vec![5000u32; 4096];
        let a = assess_iterations(&IterationStats::from_iterations(&iters, 5000));
        assert!(a.no_escape);
        assert_eq!(a.verdict, IterationVerdict::Sufficient);
    }

    #[test]
    fn heavy_tail_grows_light_tail_does_not() {
        // Queue lourde (alpha 0,5), 30 % intérieur, cap 1000 : beaucoup de
        // pixels s'évaderaient entre 1000 et 4000.
        let cap = 1000;
        let heavy = synthetic(8192, 0.3, 0.0, 5.0, 0.5, cap);
        let a = assess_iterations(&IterationStats::from_iterations(&heavy, cap));
        match a.verdict {
            IterationVerdict::Grow(to) => assert!(to > cap, "{a:?}"),
            v => panic!("attendu Grow, obtenu {v:?} ({a:?})"),
        }
        // Queue légère (alpha 3) : suffisant.
        let light = synthetic(8192, 0.3, 0.0, 5.0, 3.0, cap);
        let a = assess_iterations(&IterationStats::from_iterations(&light, cap));
        assert!(!matches!(a.verdict, IterationVerdict::Grow(_)), "{a:?}");
    }

    #[test]
    fn deep_offset_is_absorbed_by_the_baseline() {
        // Zoom profond : rien ne s'évade avant 30 000, puis queue légère.
        let cap = 60_000;
        let iters = synthetic(8192, 0.2, 30_000.0, 50.0, 2.0, cap);
        let a = assess_iterations(&IterationStats::from_iterations(&iters, cap));
        assert!(a.predicted_change <= TARGET_CHANGE, "{a:?}");
        // Plafond trop bas : la masse est encore en train d'arriver.
        let cap = 30_200;
        let iters = synthetic(8192, 0.2, 30_000.0, 50.0, 2.0, cap);
        let a = assess_iterations(&IterationStats::from_iterations(&iters, cap));
        assert!(matches!(a.verdict, IterationVerdict::Grow(_)), "{a:?}");
    }

    /// Hystérésis : appliquer le verdict puis ré-évaluer la même vue ne doit
    /// jamais repartir dans l'autre sens.
    #[test]
    fn verdicts_do_not_oscillate() {
        for &(alpha, interior) in &[(0.7, 0.3), (1.0, 0.1), (1.5, 0.5), (2.5, 0.0)] {
            let mut cap = 500u32;
            let mut history = Vec::new();
            for _ in 0..12 {
                let iters = synthetic(16_384, interior, 0.0, 3.0, alpha, cap);
                let a = assess_iterations(&IterationStats::from_iterations(&iters, cap));
                history.push((cap, a.verdict));
                match a.verdict {
                    IterationVerdict::Grow(to) | IterationVerdict::Shrink(to) => cap = to,
                    IterationVerdict::Sufficient => {}
                }
            }
            let tail: Vec<_> = history[6..].iter().map(|h| h.1).collect();
            assert!(
                tail.iter().all(|v| *v == IterationVerdict::Sufficient),
                "alpha {alpha} : {history:?}"
            );
        }
    }

    #[test]
    fn applicability() {
        let p = default_params_for_type(FractalType::Mandelbrot, 64, 64);
        assert!(auto_iterations_applicable(&p));
        for t in [
            FractalType::Buddhabrot,
            FractalType::Lyapunov,
            FractalType::VonKoch,
            FractalType::Newton,
            FractalType::Nova,
            FractalType::MagnetMandelbrot,
        ] {
            let p = default_params_for_type(t, 64, 64);
            assert!(!auto_iterations_applicable(&p), "{t:?}");
            let cancel = Arc::new(AtomicBool::new(false));
            assert_eq!(auto_iterations(&p, &cancel), Err(AutoIterError::NotApplicable));
        }
    }

    #[test]
    fn probe_dims_keep_aspect() {
        let (w, h) = probe_dimensions(1920, 1080, 4096);
        assert!((w as f64 / h as f64 - 16.0 / 9.0).abs() < 0.1);
        assert!((w * h).abs_diff(4096) < 200);
    }

    fn view(cx: &str, cy: &str, zoom: &str) -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 640, 360);
        let v = crate::fractal::ViewHp::from_center_and_zoom(cx, cy, zoom, 640, 360, 1024).unwrap();
        v.write_to_params(&mut p);
        p.center_x_hp = Some(cx.to_string());
        p.center_y_hp = Some(cy.to_string());
        p
    }

    #[test]
    fn probe_default_view() {
        let p = default_params_for_type(FractalType::Mandelbrot, 640, 360);
        let cancel = Arc::new(AtomicBool::new(false));
        let r = auto_iterations(&p, &cancel).unwrap();
        assert!(!r.no_escape && !r.ceiling_hit, "{r:?}");
        assert!((MIN_ITERATIONS..=5000).contains(&r.iteration_max), "{r:?}");
    }

    #[test]
    fn probe_seahorse_e13() {
        let p = view("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", "1e13");
        let cancel = Arc::new(AtomicBool::new(false));
        let r = auto_iterations(&p, &cancel).unwrap();
        assert!(!r.no_escape && !r.ceiling_hit, "{r:?}");
        assert!(r.iteration_max > r.max_escaped / 2, "{r:?}");
        assert!(r.iteration_max < 200_000, "{r:?}");
    }

    /// Path perturbation : l'orbite du dernier tour reste valide pour le
    /// rendu final pleine taille au plafond choisi (pas de recalcul d'orbite).
    #[test]
    fn probe_orbit_cache_is_reusable_by_final_render() {
        let mut p = view("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", "1e15");
        let cancel = Arc::new(AtomicBool::new(false));
        let mut cache = None;
        let r = auto_iterations_with(&p, &cancel, AutoIterOptions::default(), &mut cache).unwrap();
        assert_eq!(select_algorithm(&p, Device::Cpu), Algorithm::Perturbation);
        p.iteration_max = r.iteration_max;
        let cache = cache.expect("orbite de la sonde");
        assert!(cache.is_valid_for(&p), "{r:?}");
    }

    #[test]
    fn probe_all_interior_view() {
        // Cœur de la cardioïde principale.
        let p = view("-0.1", "0.0", "1e3");
        let cancel = Arc::new(AtomicBool::new(false));
        let r = auto_iterations(&p, &cancel).unwrap();
        assert!(r.no_escape, "{r:?}");
        assert_eq!(r.escaped_fraction, 0.0);
    }

    #[test]
    fn probe_all_exterior_view() {
        let p = view("1.5", "1.5", "100");
        let cancel = Arc::new(AtomicBool::new(false));
        let r = auto_iterations(&p, &cancel).unwrap();
        assert_eq!(r.rounds, 1);
        assert_eq!(r.escaped_fraction, 1.0);
        assert_eq!(r.iteration_max, MIN_ITERATIONS);
    }

    #[test]
    fn probe_cancellation() {
        let p = default_params_for_type(FractalType::Mandelbrot, 640, 360);
        let cancel = Arc::new(AtomicBool::new(true));
        assert_eq!(auto_iterations(&p, &cancel), Err(AutoIterError::Cancelled));
    }
}
