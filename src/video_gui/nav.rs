//! Navigation HP pure du studio vidéo (G12 jalon 6) — zoom ancré + pan.
//!
//! La preview du studio manipule une vue `centre + span_x` en strings GMP
//! (le span_y est TOUJOURS dérivé `span_x · aspect` au moment du rendu — pas
//! de désynchronisation possible quand le panneau change de taille).
//!
//! Ces helpers réimplémentent le zoom/pan HP de `gui/app.rs` en fonctions
//! PURES testables : contrainte « ne pas toucher au générateur » (les
//! méthodes du générateur sont liées à `FractallApp`). Même règle de
//! précision dynamique que la GUI et le pipeline vidéo :
//! `-log2(span) + 96` bits, plancher 256.
//!
//! Convention écran : coordonnées curseur/drag NORMALISÉES dans l'image
//! ((0,0) = coin haut-gauche, (1,1) = bas-droit). Le mapping pixel→c du
//! moteur fait `y = cy + (v−0.5)·span_y` (la ligne 0 est à cy − span_y/2) —
//! aucun retournement de signe ici.

use crate::fractal::ViewHp;
#[cfg(test)]
use rug::Float;

/// Vue de la preview : centre + span_x en strings HP (vérité absolue).
#[derive(Clone, Debug, PartialEq)]
pub struct HpView {
    pub cx: String,
    pub cy: String,
    pub sx: String,
}

impl HpView {
    pub fn new(cx: f64, cy: f64, sx: f64) -> Self {
        Self {
            cx: cx.to_string(),
            cy: cy.to_string(),
            sx: sx.to_string(),
        }
    }
}

/// Précision GMP pour l'arithmétique d'une vue de span `sx` :
/// `-log2(span) + 96` bits, plancher 256 (règle GUI/`span_precision`).
pub fn view_precision(sx: &str) -> u32 {
    ViewHp::from_horizontal_span("0", "0", sx, 1.0, 1, 1, 256)
        .map(|view| view.precision())
        .unwrap_or(256)
}

/// Parse les trois composantes de la vue à la précision donnée.
#[cfg(test)]
fn parse_view(v: &HpView, prec: u32) -> Option<(Float, Float, Float)> {
    let cx = Float::parse(&v.cx).ok().map(|p| Float::with_val(prec, p))?;
    let cy = Float::parse(&v.cy).ok().map(|p| Float::with_val(prec, p))?;
    let sx = Float::parse(&v.sx).ok().map(|p| Float::with_val(prec, p))?;
    if !sx.is_finite() || sx <= 0.0 {
        return None;
    }
    Some((cx, cy, sx))
}

fn core_view(v: &HpView, aspect: f64) -> Option<ViewHp> {
    ViewHp::from_horizontal_span(&v.cx, &v.cy, &v.sx, aspect, 1, 1, 256)
}

fn from_core(view: &ViewHp) -> HpView {
    let (cx, cy, sx, _) = view.decimal_parts();
    HpView { cx, cy, sx }
}

/// Zoom ancré au curseur : le point du plan sous `cursor` (normalisé image)
/// reste au même endroit à l'écran. `factor > 1` = zoom in (span divisé).
/// `aspect = height/width` du panneau (span_y = span_x·aspect).
///
/// Dérivation : `p = c + (u−0.5)·s` doit être invariant ⇒
/// `c' = c + (u−0.5)·(s − s/factor)`.
pub fn zoom_anchored(v: &HpView, cursor: (f64, f64), aspect: f64, factor: f64) -> Option<HpView> {
    if !(factor.is_finite() && factor > 0.0 && aspect.is_finite() && aspect > 0.0) {
        return None;
    }
    let mut view = core_view(v, aspect)?;
    view.zoom_at(cursor.0, cursor.1, factor);
    Some(from_core(&view))
}

/// Pan par glisser : `drag` = déplacement de la souris en fraction d'image.
/// L'image suit la souris ⇒ le centre recule : `c' = c − drag·span`.
pub fn pan(v: &HpView, drag: (f64, f64), aspect: f64) -> Option<HpView> {
    if !(aspect.is_finite() && aspect > 0.0) {
        return None;
    }
    let mut view = core_view(v, aspect)?;
    view.pan_by(-drag.0, -drag.1);
    // Le span est INCHANGÉ : on garde la string d'origine telle quelle (un
    // round-trip parse→serialize dériverait la représentation décimale, ex.
    // "0.01" → "1.000…001e-2", et invaliderait les comparaisons d'égalité).
    let mut out = from_core(&view);
    out.sx = v.sx.clone();
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Point du plan sous le curseur, en GMP.
    fn point_under_cursor(
        v: &HpView,
        cursor: (f64, f64),
        aspect: f64,
        prec: u32,
    ) -> (Float, Float) {
        let (cx, cy, sx) = parse_view(v, prec).unwrap();
        let sy = Float::with_val(prec, &sx * aspect);
        (
            Float::with_val(prec, &cx + (cursor.0 - 0.5) * sx),
            Float::with_val(prec, &cy + (cursor.1 - 0.5) * sy),
        )
    }

    /// Verrou : le zoom ancré garde le point sous le curseur INVARIANT
    /// (erreur relative au span < 1e-20), y compris en régime deep (1e-30,
    /// centre à 40 décimales — au-delà du f64).
    #[test]
    fn anchored_zoom_keeps_cursor_point_invariant() {
        let cases = [
            HpView::new(-0.75, 0.1, 0.004),
            HpView {
                cx: "-0.743643887037158704752191506114774".into(),
                cy: "0.131825904205311970493132056385139".into(),
                sx: "1e-30".into(),
            },
            HpView {
                cx: "-0.74364388703715100000000000000000000000000000000000000000000000000000000000000000001".into(),
                cy: "0.13182590420533000000000000000000000000000000000000000000000000000000000000000000001".into(),
                sx: "1e-80".into(),
            },
        ];
        let aspect = 0.75;
        for v in &cases {
            let prec = view_precision(&v.sx) + 64;
            for factor in [1.2, 2.0, 1.0 / 1.2] {
                let cursor = (0.31, 0.77);
                let before = point_under_cursor(v, cursor, aspect, prec);
                let after_view = zoom_anchored(v, cursor, aspect, factor).unwrap();
                let after = point_under_cursor(&after_view, cursor, aspect, prec);

                let (_, _, sx) = parse_view(v, prec).unwrap();
                let tol = Float::with_val(prec, &sx * 1e-20f64);
                let dx = Float::with_val(prec, &before.0 - &after.0);
                let dy = Float::with_val(prec, &before.1 - &after.1);
                assert!(
                    dx.abs() < tol,
                    "dérive X au zoom ancré (factor {factor}, vue {v:?})"
                );
                assert!(
                    dy.abs() < Float::with_val(prec, &sx * (aspect * 1e-20f64)),
                    "dérive Y au zoom ancré (factor {factor}, vue {v:?})"
                );
            }
        }
    }

    /// Pan aller-retour : revenir exactement (au bruit d'ulp GMP près) à la
    /// vue de départ, span inchangé À L'IDENTIQUE (le pan ne touche pas sx).
    #[test]
    fn pan_round_trip_returns_to_origin() {
        let v = HpView::new(-0.5, 0.25, 0.01);
        let aspect = 0.5625;
        let once = pan(&v, (0.3, -0.2), aspect).unwrap();
        let back = pan(&once, (-0.3, 0.2), aspect).unwrap();

        let prec = view_precision(&v.sx) + 32;
        let (cx0, cy0, sx0) = parse_view(&v, prec).unwrap();
        let (cx1, cy1, sx1) = parse_view(&back, prec).unwrap();
        let tol = Float::with_val(prec, &sx0 * 1e-20f64);
        assert!(Float::with_val(prec, &cx0 - &cx1).abs() < tol);
        assert!(Float::with_val(prec, &cy0 - &cy1).abs() < tol);
        assert_eq!(sx0, sx1, "le pan ne modifie jamais le span");
        assert_eq!(once.sx, v.sx, "string span inchangée");
    }

    /// Zoom in ×f puis out ×1/f : span de retour ≈ span initial. La borne
    /// d'erreur vient du FACTEUR f64 lui-même (1.2·fl(1/1.2) = 1 ± ~4e-17),
    /// pas de l'arithmétique GMP — un couple molette avant/arrière dérive
    /// d'un ulp f64, invisible et non cumulatif en pratique.
    #[test]
    fn zoom_in_out_round_trip_is_stable() {
        let v = HpView::new(0.0, 0.0, 4.0);
        let inn = zoom_anchored(&v, (0.5, 0.5), 0.75, 1.2).unwrap();
        let out = zoom_anchored(&inn, (0.5, 0.5), 0.75, 1.0 / 1.2).unwrap();
        let prec = 320;
        let (_, _, sx0) = parse_view(&v, prec).unwrap();
        let (_, _, sx1) = parse_view(&out, prec).unwrap();
        let rel = Float::with_val(prec, Float::with_val(prec, &sx0 - &sx1) / &sx0);
        assert!(
            rel.clone().abs() < 1e-15,
            "span non restauré : {sx1} (rel {rel})"
        );
        // Zoom centré : le centre ne bouge pas (comparaison en VALEUR).
        let (cx1, cy1, _) = parse_view(&inn, prec).unwrap();
        assert_eq!(cx1, 0);
        assert_eq!(cy1, 0);
    }

    /// Entrées invalides → None (pas de panic).
    #[test]
    fn invalid_inputs_return_none() {
        let v = HpView::new(0.0, 0.0, 4.0);
        assert!(zoom_anchored(&v, (0.5, 0.5), 0.75, 0.0).is_none());
        assert!(zoom_anchored(&v, (0.5, 0.5), -1.0, 1.2).is_none());
        let bad = HpView {
            cx: "abc".into(),
            cy: "0".into(),
            sx: "4".into(),
        };
        assert!(pan(&bad, (0.1, 0.1), 0.75).is_none());
        let neg = HpView {
            cx: "0".into(),
            cy: "0".into(),
            sx: "-4".into(),
        };
        assert!(zoom_anchored(&neg, (0.5, 0.5), 0.75, 1.2).is_none());
    }
}
