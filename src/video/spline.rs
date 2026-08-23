//! Valeurs dynamiques (G12 jalon 4) — splines temporelles façon DeepDrill.
//!
//! Une valeur de manifest peut être soit une constante (`"0.5"`), soit une
//! spline définie par des nœuds `temps/valeur` séparés par des virgules :
//! `"0:0/0,0:2/1,0:4/-1"`. Le temps accepte `M:S` (minutes:secondes,
//! secondes décimales OK) ou des secondes simples (`"12.5/3"`).
//!
//! Interpolation : cubique de Hermite **monotone** (Fritsch-Carlson) — pas
//! d'overshoot entre deux nœuds, exactement les valeurs aux nœuds. Hors
//! plage : clamp aux valeurs extrêmes.
//!
//! **Verrou** : une spline dont tous les nœuds portent la même valeur est
//! détectée constante (`as_constant`) et évaluée par le MÊME chemin qu'une
//! constante littérale — bit-identique, aucune arithmétique d'interpolation.

/// Valeur dynamique : constante ou spline temporelle.
#[derive(Clone, Debug)]
pub enum Dynamic {
    Constant(f64),
    Spline(MonotoneCubic),
}

impl Dynamic {
    /// Parse `"0.5"` (constante) ou `"t0/v0,t1/v1,…"` (spline, ≥ 1 nœud,
    /// temps strictement croissants).
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if let Ok(v) = s.parse::<f64>() {
            if !v.is_finite() {
                return Err(format!("valeur non finie: '{s}'"));
            }
            return Ok(Dynamic::Constant(v));
        }
        let mut knots: Vec<(f64, f64)> = Vec::new();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (t_str, v_str) = part
                .rsplit_once('/')
                .ok_or_else(|| format!("nœud invalide '{part}' (attendu temps/valeur)"))?;
            let t = parse_time(t_str)?;
            let v = v_str
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("valeur invalide '{v_str}': {e}"))?;
            if !v.is_finite() {
                return Err(format!("valeur non finie: '{v_str}'"));
            }
            if let Some(&(prev_t, _)) = knots.last() {
                if t <= prev_t {
                    return Err(format!(
                        "temps non croissants: {prev_t}s puis {t}s (les nœuds doivent être triés)"
                    ));
                }
            }
            knots.push((t, v));
        }
        match knots.len() {
            0 => Err(format!("dynamique illisible: '{s}'")),
            1 => Ok(Dynamic::Constant(knots[0].1)),
            _ => Ok(Dynamic::Spline(MonotoneCubic::new(knots))),
        }
    }

    /// Valeur constante équivalente, si la dynamique en est une (constante
    /// littérale OU spline plate). Permet aux consommateurs de prendre le
    /// chemin exact (verrou bit-identique).
    pub fn as_constant(&self) -> Option<f64> {
        match self {
            Dynamic::Constant(v) => Some(*v),
            Dynamic::Spline(sp) => {
                let v0 = sp.knots[0].1;
                sp.knots.iter().all(|&(_, v)| v == v0).then_some(v0)
            }
        }
    }

    pub fn eval(&self, t: f64) -> f64 {
        // Chemin constant exact d'abord (spline plate incluse).
        if let Some(v) = self.as_constant() {
            return v;
        }
        match self {
            Dynamic::Constant(v) => *v,
            Dynamic::Spline(sp) => sp.eval(t),
        }
    }

    /// Fin de la plage temporelle définie (dernier nœud) — None pour une
    /// constante. L'assembleur s'en sert comme durée de la vidéo quand la
    /// vélocité est une spline.
    pub fn end_time(&self) -> Option<f64> {
        match self {
            Dynamic::Constant(_) => None,
            Dynamic::Spline(sp) => Some(sp.knots.last().unwrap().0),
        }
    }
}

/// `"M:S"` (ex. `1:30` = 90 s) ou secondes décimales (`"12.5"`).
fn parse_time(s: &str) -> Result<f64, String> {
    let s = s.trim();
    let value = if let Some((m, sec)) = s.split_once(':') {
        let m: f64 = m.trim().parse().map_err(|e| format!("minutes invalides '{s}': {e}"))?;
        let sec: f64 = sec.trim().parse().map_err(|e| format!("secondes invalides '{s}': {e}"))?;
        m * 60.0 + sec
    } else {
        s.parse::<f64>().map_err(|e| format!("temps invalide '{s}': {e}"))?
    };
    if !value.is_finite() || value < 0.0 {
        return Err(format!("temps non fini ou négatif: '{s}'"));
    }
    Ok(value)
}

/// Spline cubique de Hermite à tangentes Fritsch-Carlson (monotone par
/// segment : jamais d'overshoot hors de [y_i, y_{i+1}]).
#[derive(Clone, Debug)]
pub struct MonotoneCubic {
    /// Nœuds (t, y), t strictement croissants, ≥ 2 nœuds.
    knots: Vec<(f64, f64)>,
    /// Tangentes aux nœuds.
    tangents: Vec<f64>,
}

impl MonotoneCubic {
    fn new(knots: Vec<(f64, f64)>) -> Self {
        let n = knots.len();
        debug_assert!(n >= 2);
        // Pentes des sécantes.
        let d: Vec<f64> = (0..n - 1)
            .map(|i| (knots[i + 1].1 - knots[i].1) / (knots[i + 1].0 - knots[i].0))
            .collect();
        // Tangentes initiales : moyenne des sécantes adjacentes (0 si signes
        // opposés → extremum local respecté), sécante aux bords.
        let mut m: Vec<f64> = (0..n)
            .map(|i| {
                if i == 0 {
                    d[0]
                } else if i == n - 1 {
                    d[n - 2]
                } else if d[i - 1] * d[i] <= 0.0 {
                    0.0
                } else {
                    (d[i - 1] + d[i]) / 2.0
                }
            })
            .collect();
        // Limiteur Fritsch-Carlson : α² + β² ≤ 9 par segment.
        for i in 0..n - 1 {
            if d[i] == 0.0 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
                continue;
            }
            let alpha = m[i] / d[i];
            let beta = m[i + 1] / d[i];
            let s = alpha * alpha + beta * beta;
            if s > 9.0 {
                let tau = 3.0 / s.sqrt();
                m[i] = tau * alpha * d[i];
                m[i + 1] = tau * beta * d[i];
            }
        }
        // Valeurs extrêmes (~1e300) : une sécante infinie donne tau = 0 et
        // m = 0·inf = NaN → p = NaN à l'assemblage → panic interpolate_frame
        // (sans abort ffmpeg). Tangente non finie → 0 (spline localement plate,
        // toujours monotone).
        for t in m.iter_mut() {
            if !t.is_finite() {
                *t = 0.0;
            }
        }
        Self { knots, tangents: m }
    }

    fn eval(&self, t: f64) -> f64 {
        let knots = &self.knots;
        if t <= knots[0].0 {
            return knots[0].1;
        }
        if t >= knots[knots.len() - 1].0 {
            return knots[knots.len() - 1].1;
        }
        // Segment contenant t (partition_point : premier nœud > t).
        let i = knots.partition_point(|&(x, _)| x <= t) - 1;
        let (x0, y0) = knots[i];
        let (x1, y1) = knots[i + 1];
        if t == x0 {
            return y0; // valeur de nœud EXACTE
        }
        let h = x1 - x0;
        let u = (t - x0) / h;
        let (m0, m1) = (self.tangents[i], self.tangents[i + 1]);
        // Base de Hermite.
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verrou jalon 4 : spline constante == valeur fixe **bit-identique**
    /// (les trois écritures passent par le même chemin exact).
    #[test]
    fn constant_spline_is_bit_identical_to_fixed_value() {
        let fixed = Dynamic::parse("0.7500000000000003").unwrap();
        let flat = Dynamic::parse("0/0.7500000000000003,10/0.7500000000000003").unwrap();
        let single = Dynamic::parse("3/0.7500000000000003").unwrap();
        for t in [-5.0, 0.0, 0.3333333333333333, 5.0, 9.999999, 10.0, 42.0] {
            let expect = 0.7500000000000003f64.to_bits();
            assert_eq!(fixed.eval(t).to_bits(), expect);
            assert_eq!(flat.eval(t).to_bits(), expect, "spline plate à t={t}");
            assert_eq!(single.eval(t).to_bits(), expect, "nœud unique à t={t}");
        }
        assert_eq!(flat.as_constant(), Some(0.7500000000000003));
    }

    /// Les nœuds sont interpolés EXACTEMENT, le clamp hors-plage tient, et le
    /// format temps M:S est compris.
    #[test]
    fn spline_hits_knots_exactly_and_clamps() {
        let d = Dynamic::parse("0:0/0,0:2/1,0:4/-1,1:30/2").unwrap();
        assert_eq!(d.eval(0.0), 0.0);
        assert_eq!(d.eval(2.0), 1.0);
        assert_eq!(d.eval(4.0), -1.0);
        assert_eq!(d.eval(90.0), 2.0);
        assert_eq!(d.eval(-10.0), 0.0, "clamp avant le premier nœud");
        assert_eq!(d.eval(1000.0), 2.0, "clamp après le dernier nœud");
        assert_eq!(d.end_time(), Some(90.0));
    }

    #[test]
    fn spline_rejects_non_finite_and_negative_times() {
        for value in ["inf/1,inf/2", "NaN/1,2/2", "-1/1,2/2", "-1:30/1,2/2"] {
            assert!(Dynamic::parse(value).is_err(), "devait refuser {value}");
        }
    }

    /// Monotonie Fritsch-Carlson : entre deux nœuds, la valeur reste dans
    /// [y_i, y_{i+1}] (pas d'overshoot) sur des données monotones.
    #[test]
    fn monotone_segments_do_not_overshoot() {
        let d = Dynamic::parse("0/0,1/0.1,2/5,3/5.05,4/9").unwrap();
        let mut prev = d.eval(0.0);
        for i in 1..=400 {
            let t = i as f64 * 0.01;
            let v = d.eval(t);
            assert!(v >= prev - 1e-12, "non-monotone à t={t}: {v} < {prev}");
            assert!((0.0..=9.0).contains(&v), "overshoot à t={t}: {v}");
            prev = v;
        }
    }

    /// Verrou 2026-08-23 : des valeurs ~1e300 ne produisent pas de tangente
    /// NaN (eval reste fini).
    #[test]
    fn extreme_values_keep_eval_finite() {
        let d = Dynamic::parse("0/0,1e-10/1e300,1/2e300,2/3e300").unwrap();
        for i in 0..=200 {
            let v = d.eval(i as f64 * 0.01);
            assert!(v.is_finite(), "t={} → {v}", i as f64 * 0.01);
        }
    }

    #[test]
    fn parse_errors_are_explicit() {
        assert!(Dynamic::parse("abc").is_err());
        assert!(Dynamic::parse("0/1,0/2").is_err(), "temps non croissants");
        assert!(Dynamic::parse("5/1,3/2").is_err(), "temps décroissants");
        assert!(Dynamic::parse("1/x").is_err(), "valeur illisible");
        assert!(Dynamic::parse("").is_err());
    }
}
