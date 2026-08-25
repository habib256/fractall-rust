//! Vue du plan complexe dont la haute précision est la représentation canonique.
//!
//! Ce type est volontairement indépendant du rendu. Il concentre les opérations
//! géométriques qui ne doivent jamais repasser par les miroirs `f64` de
//! [`FractalParams`](super::FractalParams).

use rug::Float;

use super::FractalParams;

const MIN_VIEW_PRECISION: u32 = 256;

#[derive(Clone, Debug)]
pub struct ViewHp {
    center_x: Float,
    center_y: Float,
    span_x: Float,
    span_y: Float,
    width: u32,
    height: u32,
    precision: u32,
}

/// Transformation affine d'une grille cible vers une grille source :
/// `source = scale * target + offset`, en coordonnées normalisées centrées.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTransform {
    pub scale_x: f64,
    pub scale_y: f64,
    pub offset_x: f64,
    pub offset_y: f64,
}

impl ViewHp {
    /// Construit la vue CLI canonique depuis un centre décimal et une
    /// magnification (`span_x = 4 / zoom`).
    pub fn from_center_and_zoom(
        center_x: &str,
        center_y: &str,
        zoom: &str,
        width: u32,
        height: u32,
        minimum_precision: u32,
    ) -> Option<Self> {
        let zoom_probe = Float::with_val(64, Float::parse(zoom).ok()?);
        let depth_bits = zoom_probe
            .get_exp()
            .map(|exp| (128i64 + i64::from(exp).max(0)).clamp(128, u32::MAX as i64) as u32)
            .unwrap_or(128);
        let probe_precision = minimum_precision.max(MIN_VIEW_PRECISION).max(depth_bits);
        let zoom = Float::with_val(probe_precision, Float::parse(zoom).ok()?);
        if !zoom.is_finite() || zoom <= 0 {
            return None;
        }
        let span_x = Float::with_val(probe_precision, 4) / zoom;
        let aspect = Float::with_val(probe_precision, height.max(1))
            / Float::with_val(probe_precision, width.max(1));
        let span_y = Float::with_val(probe_precision, &span_x * aspect);
        Self::from_decimal_parts(
            center_x,
            center_y,
            &span_x.to_string_radix(10, None),
            &span_y.to_string_radix(10, None),
            width,
            height,
            probe_precision,
        )
    }

    /// Variante pour les interfaces qui persistent seulement le span
    /// horizontal et dérivent le vertical depuis `height/width`.
    pub fn from_horizontal_span(
        center_x: &str,
        center_y: &str,
        span_x: &str,
        aspect_y_over_x: f64,
        width: u32,
        height: u32,
        minimum_precision: u32,
    ) -> Option<Self> {
        if !aspect_y_over_x.is_finite() || aspect_y_over_x <= 0.0 {
            return None;
        }
        let precision = minimum_precision
            .max(MIN_VIEW_PRECISION)
            .max(required_precision(Some(span_x), 1.0));
        let parsed_span = Float::with_val(precision, Float::parse(span_x).ok()?);
        if !parsed_span.is_finite() || parsed_span <= 0 {
            return None;
        }
        let span_y =
            Float::with_val(precision, &parsed_span * aspect_y_over_x).to_string_radix(10, None);
        Self::from_decimal_parts(
            center_x, center_y, span_x, &span_y, width, height, precision,
        )
    }

    /// Construit une vue depuis sa représentation décimale persistée. Cette
    /// entrée stricte sert aux snapshots : une chaîne invalide ou un span nul
    /// invalide la transformée au lieu d'introduire un fallback silencieux.
    pub fn from_decimal_parts(
        center_x: &str,
        center_y: &str,
        span_x: &str,
        span_y: &str,
        width: u32,
        height: u32,
        minimum_precision: u32,
    ) -> Option<Self> {
        let precision = minimum_precision
            .max(MIN_VIEW_PRECISION)
            .max(required_precision(Some(span_x), 1.0))
            .max(required_precision(Some(span_y), 1.0));
        let parse = |value: &str| {
            Float::parse(value)
                .ok()
                .map(|parsed| Float::with_val(precision, parsed))
        };
        let center_x = parse(center_x)?;
        let center_y = parse(center_y)?;
        let span_x = parse(span_x)?;
        let span_y = parse(span_y)?;
        if !center_x.is_finite()
            || !center_y.is_finite()
            || !span_x.is_finite()
            || !span_y.is_finite()
            || span_x <= 0
            || span_y <= 0
        {
            return None;
        }
        Some(Self {
            center_x,
            center_y,
            span_x,
            span_y,
            width: width.max(1),
            height: height.max(1),
            precision,
        })
    }

    pub fn from_params(params: &FractalParams) -> Self {
        let precision = params.engine.precision_bits.max(MIN_VIEW_PRECISION).max(
            required_precision(params.span_x_hp.as_deref(), params.span_x).max(required_precision(
                params.span_y_hp.as_deref(),
                params.span_y,
            )),
        );
        Self {
            center_x: parse_or_f64(params.center_x_hp.as_deref(), params.center_x, precision),
            center_y: parse_or_f64(params.center_y_hp.as_deref(), params.center_y, precision),
            span_x: positive_or_f64(params.span_x_hp.as_deref(), params.span_x, precision),
            span_y: positive_or_f64(params.span_y_hp.as_deref(), params.span_y, precision),
            width: params.width.max(1),
            height: params.height.max(1),
            precision,
        }
    }

    pub fn precision(&self) -> u32 {
        self.precision
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn center_x(&self) -> &Float {
        &self.center_x
    }
    pub fn center_y(&self) -> &Float {
        &self.center_y
    }
    pub fn span_x(&self) -> &Float {
        &self.span_x
    }
    pub fn span_y(&self) -> &Float {
        &self.span_y
    }
    pub fn decimal_parts(&self) -> (String, String, String, String) {
        (
            self.center_x.to_string_radix(10, None),
            self.center_y.to_string_radix(10, None),
            self.span_x.to_string_radix(10, None),
            self.span_y.to_string_radix(10, None),
        )
    }
    /// Magnification horizontale selon la convention commune `4 / span_x`.
    pub fn zoom_string(&self) -> String {
        (Float::with_val(self.precision, 4) / &self.span_x).to_string_radix(10, None)
    }

    /// Zoom ancré sur `(rx, ry)` normalisé dans la vue. `factor > 1` zoome.
    pub fn zoom_at(&mut self, rx: f64, ry: f64, factor: f64) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let fx = Float::with_val(self.precision, rx - 0.5);
        let fy = Float::with_val(self.precision, ry - 0.5);
        let inv = Float::with_val(self.precision, 1.0 / factor);
        let keep = Float::with_val(self.precision, 1.0 - &inv);
        self.center_x += Float::with_val(self.precision, &self.span_x * fx * &keep);
        self.center_y += Float::with_val(self.precision, &self.span_y * fy * &keep);
        self.span_x *= &inv;
        self.span_y *= inv;
    }

    /// Place le point `(rx, ry)` au centre puis applique le zoom. Contrairement
    /// à [`zoom_at`](Self::zoom_at), le point choisi ne reste pas sous le
    /// curseur : c'est la sémantique du double-clic historique de la GUI.
    pub fn focus_at(&mut self, rx: f64, ry: f64, factor: f64) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        self.center_x += Float::with_val(
            self.precision,
            &self.span_x * Float::with_val(self.precision, rx - 0.5),
        );
        self.center_y += Float::with_val(
            self.precision,
            &self.span_y * Float::with_val(self.precision, ry - 0.5),
        );
        let inv = Float::with_val(self.precision, 1.0 / factor);
        self.span_x *= &inv;
        self.span_y *= inv;
    }

    /// Déplace le centre d'une fraction des spans courants.
    pub fn pan_by(&mut self, dx: f64, dy: f64) {
        if !dx.is_finite() || !dy.is_finite() {
            return;
        }
        self.center_x += Float::with_val(self.precision, &self.span_x * dx);
        self.center_y += Float::with_val(self.precision, &self.span_y * dy);
    }

    /// Cadre une sélection normalisée, en l'élargissant sur un seul axe pour
    /// conserver le ratio de la surface et des pixels carrés.
    pub fn select_rect(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        let (x1, x2) = (x1.min(x2), x1.max(x2));
        let (y1, y2) = (y1.min(y2), y1.max(y2));
        if x2 <= x1 || y2 <= y1 {
            return;
        }
        self.center_x += Float::with_val(
            self.precision,
            &self.span_x * Float::with_val(self.precision, (x1 + x2) * 0.5 - 0.5),
        );
        self.center_y += Float::with_val(
            self.precision,
            &self.span_y * Float::with_val(self.precision, (y1 + y2) * 0.5 - 0.5),
        );
        let selected_x = Float::with_val(self.precision, &self.span_x * (x2 - x1));
        let selected_y = Float::with_val(self.precision, &self.span_y * (y2 - y1));
        let target = Float::with_val(self.precision, self.width)
            / Float::with_val(self.precision, self.height);
        if Float::with_val(self.precision, &selected_x / &selected_y) > target {
            self.span_x = selected_x;
            self.span_y = Float::with_val(self.precision, &self.span_x / target);
        } else {
            self.span_y = selected_y;
            self.span_x = Float::with_val(self.precision, &self.span_y * target);
        }
    }

    /// Change la surface en conservant des pixels carrés et toute la vue
    /// précédente (la dimension non contraignante est élargie).
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        let target =
            Float::with_val(self.precision, width) / Float::with_val(self.precision, height);
        let current = Float::with_val(self.precision, &self.span_x / &self.span_y);
        if current > target {
            self.span_y = Float::with_val(self.precision, &self.span_x / &target);
        } else {
            self.span_x = Float::with_val(self.precision, &self.span_y * &target);
        }
        self.width = width;
        self.height = height;
    }

    pub fn transform_to(&self, target: &Self) -> ViewTransform {
        let precision = self.precision.max(target.precision);
        let scale_x = Float::with_val(precision, &target.span_x / &self.span_x).to_f64();
        let scale_y = Float::with_val(precision, &target.span_y / &self.span_y).to_f64();
        let offset_x = Float::with_val(
            precision,
            Float::with_val(precision, &target.center_x - &self.center_x) / &self.span_x,
        )
        .to_f64();
        let offset_y = Float::with_val(
            precision,
            Float::with_val(precision, &target.center_y - &self.center_y) / &self.span_y,
        )
        .to_f64();
        ViewTransform {
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        }
    }

    /// Met à jour ensemble la représentation HP et ses miroirs f64.
    pub fn write_to_params(&self, params: &mut FractalParams) {
        params.width = self.width;
        params.height = self.height;
        params.center_x = self.center_x.to_f64();
        params.center_y = self.center_y.to_f64();
        params.span_x = self.span_x.to_f64();
        params.span_y = self.span_y.to_f64();
        params.center_x_hp = Some(self.center_x.to_string_radix(10, None));
        params.center_y_hp = Some(self.center_y.to_string_radix(10, None));
        params.span_x_hp = Some(self.span_x.to_string_radix(10, None));
        params.span_y_hp = Some(self.span_y.to_string_radix(10, None));
    }
}

fn parse_or_f64(value: Option<&str>, fallback: f64, precision: u32) -> Float {
    value
        .and_then(|s| Float::parse(s).ok())
        .map(|v| Float::with_val(precision, v))
        .unwrap_or_else(|| Float::with_val(precision, fallback))
}

fn positive_or_f64(value: Option<&str>, fallback: f64, precision: u32) -> Float {
    let parsed = parse_or_f64(value, fallback, precision);
    if parsed.is_finite() && parsed > 0 {
        parsed
    } else {
        Float::with_val(precision, fallback.abs().max(f64::MIN_POSITIVE))
    }
}

/// Assez de bits pour additionner un déplacement de l'ordre d'un pixel au
/// centre O(1), avec 128 bits de marge pour les opérations intermédiaires et
/// les invariants sous-pixel des consommateurs.
fn required_precision(value: Option<&str>, fallback: f64) -> u32 {
    let probe = parse_or_f64(value, fallback, 64);
    match probe.get_exp() {
        Some(exp) if exp < 0 => (128i64 - i64::from(exp)).clamp(128, u32::MAX as i64) as u32,
        _ => 128,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    #[test]
    fn deep_zoom_anchor_does_not_pass_through_f64() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        p.center_x_hp = Some("-0.7436438870371510000000000000000000001".into());
        p.span_x_hp = Some("1e-80".into());
        p.span_y_hp = Some("7.5e-81".into());
        let mut view = ViewHp::from_params(&p);
        let before = view.center_x.clone();
        view.zoom_at(0.75, 0.5, 2.0);
        assert_ne!(view.center_x, before);
        let expected = Float::with_val(view.precision(), Float::parse("5e-81").unwrap());
        assert_eq!(view.span_x, expected);
    }

    #[test]
    fn resize_preserves_center_and_expands_only_one_axis() {
        let p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        let mut view = ViewHp::from_params(&p);
        let center = view.center_x.clone();
        let old_x = view.span_x.clone();
        view.resize(1600, 900);
        assert_eq!(view.center_x, center);
        assert!(view.span_x >= old_x);
        assert_eq!(view.dimensions(), (1600, 900));
    }

    #[test]
    fn write_keeps_hp_and_f64_mirrors_coherent() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 10, 10);
        let mut view = ViewHp::from_params(&p);
        view.zoom_at(0.25, 0.75, 3.0);
        view.write_to_params(&mut p);
        let cx = Float::with_val(256, Float::parse(p.center_x_hp.as_ref().unwrap()).unwrap());
        let sy = Float::with_val(256, Float::parse(p.span_y_hp.as_ref().unwrap()).unwrap());
        assert_eq!(p.center_x, cx.to_f64());
        assert_eq!(p.span_y, sy.to_f64());
    }

    #[test]
    fn selected_rectangle_keeps_surface_aspect_in_hp() {
        let p = default_params_for_type(FractalType::Mandelbrot, 1600, 900);
        let mut view = ViewHp::from_params(&p);
        view.select_rect(0.2, 0.1, 0.6, 0.9);
        let aspect = Float::with_val(view.precision(), view.span_x() / view.span_y());
        let expected = Float::with_val(view.precision(), 16) / Float::with_val(view.precision(), 9);
        assert!((aspect - expected).abs() < 1e-60);
    }

    #[test]
    fn center_and_zoom_materializes_consistent_ultra_deep_view() {
        let view = ViewHp::from_center_and_zoom(
            "-0.74364388703715100000000000000000000000000000000000000000000000000000000000000000001",
            "0.13182590420533",
            "4e80",
            1600,
            900,
            256,
        )
        .unwrap();
        let expected_span = Float::with_val(view.precision, Float::parse("1e-80").unwrap());
        assert!(Float::with_val(view.precision, &view.span_x - expected_span).abs() < 1e-110);
        let aspect = Float::with_val(view.precision, &view.span_x / &view.span_y);
        let expected = Float::with_val(view.precision, 16) / Float::with_val(view.precision, 9);
        assert!((aspect - expected).abs() < 1e-90);
        assert!(ViewHp::from_center_and_zoom("0", "0", "0", 1, 1, 256).is_none());
    }
}
