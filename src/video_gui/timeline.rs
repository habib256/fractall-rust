//! Timeline du studio vidéo (G13) — logique PURE, zéro egui : miniatures des
//! keyframes (canaux bruts sous-échantillonnés, recolorisables au changement
//! de palette SANS relire le disque), courbe de vitesse par zone compilée
//! vers la spline temporelle `video.velocity` du manifest, et ETA par
//! régression linéaire sur les durées de rendu mesurées.
//!
//! Les workers threadés (scan de maps, scrubbing, première miniature) vivent
//! dans `job.rs` ; `mod.rs` ne fait que dessiner et router les interactions.
//!
//! Point d'architecture clé : la vitesse n'est consommée qu'à l'ASSEMBLAGE
//! (`assemble::timeline`), jamais au rendu des keyframes — éditer la courbe
//! de vitesse ne coûte donc qu'un « Ré-assembler », aucun recalcul fractal.

use num_complex::Complex64;
use rug::Float;

use crate::io::fmap::FractalMap;

/// Hauteur maximale d'une miniature (px) ; la largeur suit l'aspect de la map.
pub const THUMB_MAX_H: u32 = 54;

/// Contenu d'une cellule de la timeline.
#[derive(Clone, Debug)]
pub enum ThumbSlot {
    /// Aucune donnée encore (cadre placeholder numéroté).
    Empty,
    /// RGB provisoire : copie réduite de la preview pour la keyframe FINALE
    /// (la cible est déjà à l'écran) — remplacé dès que la vraie map existe.
    Rgb { rgb: Vec<u8>, w: u32, h: u32 },
    /// Canaux bruts sous-échantillonnés de la map. `provisional` = rendu
    /// d'attente (itérations plafonnées, keyframe 0 « vue pleine ») remplacé
    /// par la map réelle dès qu'elle est rendue.
    Channels {
        w: u32,
        h: u32,
        iter_max: u32,
        iterations: Vec<u32>,
        zs: Vec<Complex64>,
        provisional: bool,
    },
}

impl ThumbSlot {
    /// Miniature définitive (issue d'une vraie map) : ne doit jamais être
    /// écrasée par un provisoire.
    pub fn is_final(&self) -> bool {
        matches!(self, ThumbSlot::Channels { provisional: false, .. })
    }
}

/// Sous-échantillonne les canaux bruts d'une map vers une miniature de
/// hauteur ≤ `max_h` (nearest au CENTRE du texel — pas de coin haut-gauche,
/// qui biaiserait la miniature vers le haut-gauche de l'image).
///
/// On garde les canaux (pas le RGB) : recoloriser une miniature au changement
/// de palette coûte ~5 k pixels au lieu d'une relecture/décompression du
/// `.fmap` complet (~18 Mo par keyframe).
pub fn thumb_channels(map: &FractalMap, max_h: u32) -> (u32, u32, Vec<u32>, Vec<Complex64>) {
    let (w, h) = (map.params.width as usize, map.params.height as usize);
    let th = (max_h as usize).min(h).max(1);
    let tw = (((w as f64 / h as f64) * th as f64).round() as usize).max(1);
    let mut iterations = Vec::with_capacity(tw * th);
    let mut zs = Vec::with_capacity(tw * th);
    for j in 0..th {
        let sj = (((j as f64 + 0.5) * h as f64 / th as f64) as usize).min(h - 1);
        for i in 0..tw {
            let si = (((i as f64 + 0.5) * w as f64 / tw as f64) as usize).min(w - 1);
            let idx = sj * w + si;
            iterations.push(map.iterations[idx]);
            zs.push(map.zs[idx]);
        }
    }
    (tw as u32, th as u32, iterations, zs)
}

/// Réduit un buffer RGB (3 octets/px) vers une miniature de hauteur ≤ `max_h`
/// (nearest centre). Sert au provisoire « keyframe finale » depuis la preview.
pub fn downscale_rgb_nearest(rgb: &[u8], w: u32, h: u32, max_h: u32) -> (Vec<u8>, u32, u32) {
    let (w, h) = (w as usize, h as usize);
    debug_assert_eq!(rgb.len(), w * h * 3);
    let th = (max_h as usize).min(h).max(1);
    let tw = (((w as f64 / h as f64) * th as f64).round() as usize).max(1);
    let mut out = Vec::with_capacity(tw * th * 3);
    for j in 0..th {
        let sj = (((j as f64 + 0.5) * h as f64 / th as f64) as usize).min(h - 1);
        for i in 0..tw {
            let si = (((i as f64 + 0.5) * w as f64 / tw as f64) as usize).min(w - 1);
            let o = (sj * w + si) * 3;
            out.extend_from_slice(&rgb[o..o + 3]);
        }
    }
    (out, tw as u32, th as u32)
}

/// Span exact de la keyframe `k` (string décimale COMPLÈTE) — même formule
/// que `video::keyframe_params` (`4/2^k`, expansion décimale finie). Utilisé
/// par le clic-sur-miniature pour amener la preview à cette profondeur.
pub fn span_at_keyframe(k: u32) -> String {
    let prec = (k + 96).max(256);
    let span = Float::with_val(prec, 4.0) >> k;
    let digits = k as usize * 7 / 10 + 60;
    span.to_string_radix(10, Some(digits))
}

// ---------------------------------------------------------------------------
// Courbe de vitesse
// ---------------------------------------------------------------------------

/// Courbe de vitesse par zone : points (position keyframe ∈ [0, n],
/// multiplicateur > 0) reliés LINÉAIREMENT, appliqués à la vitesse de base.
/// Compilée vers la spline temporelle du manifest (`video.velocity`,
/// format `"t/v,t/v,…"` de `video/spline.rs`) — le backend existant fait le
/// reste, y compris le lissage monotone des transitions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeedCurve {
    /// Points triés par position croissante. Vide = courbe plate ×1.
    pub points: Vec<(f64, f64)>,
}

impl SpeedCurve {
    /// Multiplicateur à la position `p` : interpolation linéaire, clamp aux
    /// extrêmes, 1.0 si la courbe est vide.
    pub fn multiplier_at(&self, p: f64) -> f64 {
        let pts = &self.points;
        match pts.len() {
            0 => 1.0,
            1 => pts[0].1,
            _ => {
                if p <= pts[0].0 {
                    return pts[0].1;
                }
                if p >= pts[pts.len() - 1].0 {
                    return pts[pts.len() - 1].1;
                }
                let i = pts.partition_point(|&(x, _)| x <= p) - 1;
                let (x0, m0) = pts[i];
                let (x1, m1) = pts[i + 1];
                if x1 <= x0 {
                    return m0;
                }
                let u = (p - x0) / (x1 - x0);
                m0 + (m1 - m0) * u
            }
        }
    }

    /// Courbe équivalente à une constante (vide, un seul point, ou tous les
    /// multiplicateurs identiques) → les consommateurs prennent le chemin
    /// constant exact (verrou spline plate == constante du backend).
    pub fn flat_multiplier(&self) -> Option<f64> {
        match self.points.first() {
            None => Some(1.0),
            Some(&(_, m0)) => self.points.iter().all(|&(_, m)| m == m0).then_some(m0),
        }
    }

    /// Compile la courbe vers la valeur `video.velocity` du manifest.
    ///
    /// * Courbe plate → constante `base·m` en Display f64 EXACT (bit-identique
    ///   au chemin historique, verrou) ;
    /// * sinon → spline temporelle : `t(p) = ∫ dp / v(p)` intégrée au trapèze
    ///   (8 sous-pas par segment), un nœud `t_k/v(k)` par frontière de
    ///   keyframe, plus un nœud de GARDE 0.5 s après la fin à vitesse finale —
    ///   l'intégrateur point-milieu de `assemble::timeline` atteint ainsi
    ///   p = n à coup sûr (clamp), au prix d'une courte tenue sur l'image
    ///   finale.
    pub fn compile(&self, base: f64, n: u32) -> String {
        let base = base.max(1e-3);
        if let Some(m) = self.flat_multiplier() {
            return format!("{}", base * m);
        }
        const SUB: u32 = 8;
        let mut t = 0.0f64;
        let mut knots = Vec::with_capacity(n as usize + 2);
        knots.push((0.0, base * self.multiplier_at(0.0).max(1e-3)));
        for k in 0..n.max(1) {
            for s in 0..SUB {
                let p0 = k as f64 + s as f64 / SUB as f64;
                let p1 = k as f64 + (s + 1) as f64 / SUB as f64;
                let v0 = base * self.multiplier_at(p0).max(1e-3);
                let v1 = base * self.multiplier_at(p1).max(1e-3);
                t += (1.0 / v0 + 1.0 / v1) * 0.5 * (p1 - p0);
            }
            knots.push((t, base * self.multiplier_at((k + 1) as f64).max(1e-3)));
        }
        let last_v = knots.last().map(|&(_, v)| v).unwrap_or(base);
        knots.push((t + 0.5, last_v));
        knots
            .iter()
            .map(|(t, v)| format!("{t}/{v}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

// ---------------------------------------------------------------------------
// ETA
// ---------------------------------------------------------------------------

/// Temps restant estimé (s) : régression linéaire `secondes = a + b·k` sur
/// les durées mesurées (les keyframes profondes coûtent plus cher —
/// itérations croissantes, précision GMP), extrapolée aux keyframes
/// restantes. `None` tant qu'aucune mesure ; `Some(0)` si rien ne reste.
pub fn eta_seconds(measured: &[(u32, f64)], remaining: &[u32]) -> Option<f64> {
    if remaining.is_empty() {
        return Some(0.0);
    }
    if measured.is_empty() {
        return None;
    }
    let nf = measured.len() as f64;
    let mx = measured.iter().map(|&(k, _)| k as f64).sum::<f64>() / nf;
    let my = measured.iter().map(|&(_, s)| s).sum::<f64>() / nf;
    let (mut sxx, mut sxy) = (0.0f64, 0.0f64);
    for &(k, s) in measured {
        let dx = k as f64 - mx;
        sxx += dx * dx;
        sxy += dx * (s - my);
    }
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let a = my - b * mx;
    Some(remaining.iter().map(|&k| (a + b * k as f64).max(0.0)).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};
    use crate::video::assemble::timeline as assemble_timeline;
    use crate::video::spline::Dynamic;

    fn tiny_map(w: u32, h: u32) -> FractalMap {
        let n = (w * h) as usize;
        FractalMap {
            params: default_params_for_type(FractalType::Mandelbrot, w, h),
            iterations: (0..n as u32).collect(),
            zs: (0..n).map(|i| Complex64::new(i as f64, -(i as f64))).collect(),
            distances: None,
        }
    }

    /// Nearest CENTRE : la miniature 4×3 d'une map 8×6 échantillonne les
    /// lignes 1/3/5 et colonnes 1/3/5/7 (pas les coins haut-gauche).
    #[test]
    fn thumb_channels_samples_texel_centers() {
        let map = tiny_map(8, 6);
        let (tw, th, its, zs) = thumb_channels(&map, 3);
        assert_eq!((tw, th), (4, 3));
        let expected: Vec<u32> = [1usize, 3, 5]
            .iter()
            .flat_map(|&sj| [1usize, 3, 5, 7].iter().map(move |&si| (sj * 8 + si) as u32))
            .collect();
        assert_eq!(its, expected);
        assert_eq!(zs[0], Complex64::new(9.0, -9.0)); // idx 1*8+1
        // Une map déjà plus petite que max_h n'est pas agrandie.
        let small = tiny_map(4, 2);
        let (tw2, th2, its2, _) = thumb_channels(&small, 54);
        assert_eq!((tw2, th2), (4, 2));
        assert_eq!(its2.len(), 8);
    }

    #[test]
    fn downscale_rgb_matches_channel_mapping() {
        let (w, h) = (8u32, 6u32);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| [i as u8, 0, 0]).collect();
        let (out, tw, th) = downscale_rgb_nearest(&rgb, w, h, 3);
        assert_eq!((tw, th), (4, 3));
        assert_eq!(out.len(), 4 * 3 * 3);
        assert_eq!(out[0], 9, "premier texel = centre (1,1) = idx 9");
    }

    /// `span_at_keyframe` == `4/2^k` exact en GMP (même formule que
    /// `keyframe_params`), y compris en régime deep hors f64.
    #[test]
    fn span_at_keyframe_is_exact() {
        for k in [0u32, 7, 53, 200, 1200] {
            let prec = (k + 128).max(256);
            let reloaded = Float::with_val(prec, Float::parse(span_at_keyframe(k)).unwrap());
            let expected = Float::with_val(prec, 4.0) >> k;
            assert_eq!(reloaded, expected, "span_at_keyframe({k})");
        }
    }

    /// Verrou : courbe plate (vide / un point / tous égaux) → constante en
    /// Display f64 exact, IDENTIQUE au chemin historique `format!("{v}")`.
    #[test]
    fn flat_curve_compiles_to_exact_constant() {
        let base = 0.7500000000000003f64;
        assert_eq!(SpeedCurve::default().compile(base, 40), format!("{base}"));
        let one_pt = SpeedCurve { points: vec![(3.0, 2.0)] };
        assert_eq!(one_pt.compile(base, 40), format!("{}", base * 2.0));
        let all_same = SpeedCurve { points: vec![(0.0, 0.5), (10.0, 0.5), (40.0, 0.5)] };
        assert_eq!(all_same.compile(base, 40), format!("{}", base * 0.5));
        // Et le backend la parse en Constant (chemin exact).
        let d = Dynamic::parse(&SpeedCurve::default().compile(base, 40)).unwrap();
        assert_eq!(d.eval(1.0).to_bits(), base.to_bits());
    }

    /// Un ralentissement local ALLONGE la durée, la spline compilée est
    /// parsable, et l'intégrateur de l'assembleur atteint exactement p = n
    /// (le nœud de garde couvre l'erreur d'intégration).
    #[test]
    fn slowdown_zone_lengthens_and_reaches_end() {
        let n = 4u32;
        let curve = SpeedCurve {
            points: vec![(0.0, 1.0), (1.5, 1.0), (2.0, 0.25), (3.0, 0.25), (3.5, 1.0)],
        };
        let compiled = curve.compile(1.0, n);
        let d = Dynamic::parse(&compiled).expect("spline compilée parsable");
        // Durée (dernier nœud) : plate = 4 s ; ici la zone ×0.25 rallonge.
        let end = d.end_time().expect("spline ⇒ end_time");
        assert!(end > 6.0 && end < 12.0, "durée inattendue: {end}");
        let positions = assemble_timeline(n, 30, &d).unwrap();
        assert_eq!(*positions.last().unwrap(), n as f64, "doit atteindre la fin");
        assert!(positions.windows(2).all(|w| w[1] >= w[0]), "positions monotones");
        // La zone lente occupe plus de frames : compter les frames par segment.
        let frames_in = |a: f64, b: f64| positions.iter().filter(|&&p| p >= a && p < b).count();
        assert!(
            frames_in(2.0, 3.0) > 2 * frames_in(0.0, 1.0),
            "la zone ×0.25 doit contenir bien plus de frames ({} vs {})",
            frames_in(2.0, 3.0),
            frames_in(0.0, 1.0)
        );
    }

    #[test]
    fn multiplier_interpolates_linearly_and_clamps() {
        let c = SpeedCurve { points: vec![(1.0, 1.0), (3.0, 3.0)] };
        assert_eq!(c.multiplier_at(-5.0), 1.0);
        assert_eq!(c.multiplier_at(1.0), 1.0);
        assert_eq!(c.multiplier_at(2.0), 2.0);
        assert_eq!(c.multiplier_at(3.0), 3.0);
        assert_eq!(c.multiplier_at(99.0), 3.0);
        assert_eq!(SpeedCurve::default().multiplier_at(7.0), 1.0);
    }

    /// ETA : ajustement exact sur données linéaires, None sans mesure,
    /// 0 sans reste.
    #[test]
    fn eta_fits_linear_growth() {
        let measured = [(0u32, 1.0f64), (10, 2.0)];
        let eta = eta_seconds(&measured, &[20, 30]).unwrap();
        assert!((eta - 7.0).abs() < 1e-9, "eta = {eta}"); // 3.0 + 4.0
        assert_eq!(eta_seconds(&[], &[1]), None);
        assert_eq!(eta_seconds(&measured, &[]), Some(0.0));
        // Une seule mesure → extrapolation par la moyenne.
        let eta1 = eta_seconds(&[(5, 2.0)], &[8, 9]).unwrap();
        assert!((eta1 - 4.0).abs() < 1e-9);
    }
}
