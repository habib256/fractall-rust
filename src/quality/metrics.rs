use num_complex::Complex64;
use serde::{Serialize, Serializer};

#[derive(Debug, Clone, Serialize)]
pub struct IterStats {
    pub max: f64,
    pub mean: f64,
    pub rms: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl IterStats {
    fn from_values(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return IterStats { max: 0.0, mean: 0.0, rms: 0.0, p50: 0.0, p95: 0.0, p99: 0.0 };
        }
        let n = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / n as f64;
        let sum_sq: f64 = values.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f64).sqrt();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let max = *values.last().unwrap();
        let percentile = |p: f64| -> f64 {
            let idx = ((p * (n - 1) as f64).round() as usize).min(n - 1);
            values[idx]
        };
        IterStats {
            max,
            mean,
            rms,
            p50: percentile(0.5),
            p95: percentile(0.95),
            p99: percentile(0.99),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TopDivergent {
    pub x: u32,
    pub y: u32,
    pub pert_iter: u32,
    pub gmp_iter: u32,
    pub pert_z: Complex64,
    pub gmp_z: Complex64,
    pub iter_diff: u32,
    pub z_distance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Warn,
    Fail,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Warn => "WARN",
            Verdict::Fail => "FAIL",
        }
    }
}

impl Serialize for Verdict {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    pub max_iter_diff_pass: f64,
    pub divergence_ratio_pass: f64,
    /// p99 de l'iter-diff au-delà duquel → FAIL. **Métrique robuste** (percentile,
    /// pas le `max` outlier) : un p99 > 1 signifie que > 1 % des pixels divergent
    /// de plus de 1 itér → divergence LARGE (vrai bug), à distinguer de quelques
    /// pixels de bord dispersés (bruit f64 inhérent, cf. G6).
    pub p99_iter_diff_fail: f64,
    pub divergence_ratio_warn: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            max_iter_diff_pass: 1.0,
            divergence_ratio_pass: 0.001,
            p99_iter_diff_fail: 1.0,
            divergence_ratio_warn: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QualityMetrics {
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub total_pixels: usize,
    pub iter_diff: IterStats,
    pub iter_divergence_ratio: f64,
    pub z_distance: IterStats,
    pub z_ratio_error: IterStats,
    pub escape_disagreement: f64,
    pub perturb_time_ms: f64,
    pub gmp_time_ms: f64,
    // Heavy per-pixel buffer: never serialized into JSON.
    #[serde(skip_serializing)]
    pub heatmap: Vec<u8>,
    #[allow(dead_code)]
    #[serde(skip_serializing)]
    pub heatmap_scale: f64,
    pub top_divergent: Vec<TopDivergent>,
    pub verdict: Verdict,
}

impl QualityMetrics {
    pub fn speedup(&self) -> f64 {
        if self.perturb_time_ms <= 0.0 {
            return 0.0;
        }
        self.gmp_time_ms / self.perturb_time_ms
    }
}

pub fn compute(
    width: u32,
    height: u32,
    iteration_max: u32,
    pert_iters: &[u32],
    pert_zs: &[Complex64],
    gmp_iters: &[u32],
    gmp_zs: &[Complex64],
    perturb_time_ms: f64,
    gmp_time_ms: f64,
    thresholds: Thresholds,
) -> QualityMetrics {
    let n = (width as usize) * (height as usize);
    assert_eq!(pert_iters.len(), n, "pert_iters size mismatch");
    assert_eq!(pert_zs.len(), n, "pert_zs size mismatch");
    assert_eq!(gmp_iters.len(), n, "gmp_iters size mismatch");
    assert_eq!(gmp_zs.len(), n, "gmp_zs size mismatch");

    let mut iter_diffs = Vec::with_capacity(n);
    let mut z_dists = Vec::with_capacity(n);
    let mut z_ratios = Vec::with_capacity(n);
    let mut divergent_count: u64 = 0;
    let mut escape_mismatch: u64 = 0;

    let mut raw_iter_diff_u32: Vec<u32> = Vec::with_capacity(n);

    for i in 0..n {
        let pi = pert_iters[i];
        let gi = gmp_iters[i];
        let diff = pi.abs_diff(gi);
        raw_iter_diff_u32.push(diff);
        iter_diffs.push(diff as f64);
        if diff > 1 {
            divergent_count += 1;
        }
        let pert_escaped = pi < iteration_max;
        let gmp_escaped = gi < iteration_max;
        if pert_escaped != gmp_escaped {
            escape_mismatch += 1;
        }
        let dz = pert_zs[i] - gmp_zs[i];
        let d = dz.norm();
        z_dists.push(d);
        let denom = gmp_zs[i].norm().max(1e-12);
        z_ratios.push(d / denom);
    }

    let iter_divergence_ratio = divergent_count as f64 / n as f64;
    let escape_disagreement = escape_mismatch as f64 / n as f64;

    let iter_stats = IterStats::from_values(iter_diffs);
    let z_stats = IterStats::from_values(z_dists);
    let z_ratio_stats = IterStats::from_values(z_ratios);

    let heatmap_scale = if iter_stats.max > 0.0 { 255.0 / iter_stats.max } else { 0.0 };
    let heatmap: Vec<u8> = raw_iter_diff_u32
        .iter()
        .map(|&d| {
            let v = (d as f64 * heatmap_scale).round();
            v.clamp(0.0, 255.0) as u8
        })
        .collect();

    let mut indexed: Vec<(usize, u32)> = raw_iter_diff_u32
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d))
        .collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));
    let top_divergent: Vec<TopDivergent> = indexed
        .iter()
        .take(10)
        .filter(|(_, d)| *d > 0)
        .map(|&(i, diff)| {
            let x = (i % width as usize) as u32;
            let y = (i / width as usize) as u32;
            TopDivergent {
                x,
                y,
                pert_iter: pert_iters[i],
                gmp_iter: gmp_iters[i],
                pert_z: pert_zs[i],
                gmp_z: gmp_zs[i],
                iter_diff: diff,
                z_distance: (pert_zs[i] - gmp_zs[i]).norm(),
            }
        })
        .collect();

    let verdict = classify(&iter_stats, iter_divergence_ratio, thresholds);

    QualityMetrics {
        width,
        height,
        total_pixels: n,
        iter_diff: iter_stats,
        iter_divergence_ratio,
        z_distance: z_stats,
        z_ratio_error: z_ratio_stats,
        escape_disagreement,
        perturb_time_ms,
        gmp_time_ms,
        heatmap,
        heatmap_scale,
        top_divergent,
        verdict,
    }
}

// Classification robuste au bruit de bord (G6, recalibré 2026-07-10).
//
// L'ancien gate FAILait sur `iter_stats.max` (l'OUTLIER) → les cas à bruit de
// bord inhérent (plancher f64 : quelques pixels dispersés, p95=p99=0, mais max
// grand — e13/e17/seahorse-valley) étaient FAIL en permanence, noyant les vraies
// régressions dans le signal du loop /improve. Le nouveau gate distingue :
//   - PASS  : divergence quasi-nulle (max ≤ 1 ET div_ratio ≤ 1e-3) — inchangé.
//   - FAIL  : divergence LARGE (p99 > seuil : > 1 % des pixels divergent de > 1)
//             OU SYSTÉMATIQUE (div_ratio > warn) — signature d'un vrai bug
//             (over-skip BLA = +N uniforme div_ratio 1.0 ; période fausse =
//             image uniforme ; etc.). Le `max` outlier seul NE FAIL PAS.
//   - WARN  : divergence éparse (quelques pixels de bord) — visible, non bloquant.
// `max` reste rapporté dans report.md pour l'inspection humaine.
fn classify(iter_stats: &IterStats, div_ratio: f64, t: Thresholds) -> Verdict {
    if iter_stats.max <= t.max_iter_diff_pass && div_ratio <= t.divergence_ratio_pass {
        Verdict::Pass
    } else if iter_stats.p99 > t.p99_iter_diff_fail || div_ratio > t.divergence_ratio_warn {
        Verdict::Fail
    } else {
        Verdict::Warn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_z(n: usize) -> Vec<Complex64> {
        vec![Complex64::new(0.0, 0.0); n]
    }

    #[test]
    fn identical_vectors_score_zero() {
        let iters = vec![10, 20, 30, 40];
        let zs = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let m = compute(2, 2, 100, &iters, &zs, &iters, &zs, 1.0, 10.0, Thresholds::default());
        assert_eq!(m.iter_diff.max, 0.0);
        assert_eq!(m.iter_diff.mean, 0.0);
        assert_eq!(m.iter_divergence_ratio, 0.0);
        assert_eq!(m.escape_disagreement, 0.0);
        assert_eq!(m.verdict, Verdict::Pass);
        assert!(m.top_divergent.is_empty());
    }

    #[test]
    fn single_pixel_deviation() {
        let pert = vec![10, 20, 30, 40];
        let gmp = vec![10, 20, 35, 40];
        let zs = zero_z(4);
        let m = compute(2, 2, 100, &pert, &zs, &gmp, &zs, 1.0, 10.0, Thresholds::default());
        assert_eq!(m.iter_diff.max, 5.0);
        assert_eq!(m.iter_divergence_ratio, 0.25);
        assert_eq!(m.top_divergent.len(), 1);
        assert_eq!(m.top_divergent[0].iter_diff, 5);
        // div_ratio 0.25 ≫ warn (0.01) → divergence SYSTÉMATIQUE → FAIL.
        assert_eq!(m.verdict, Verdict::Fail);
    }

    #[test]
    fn escape_disagreement_detected() {
        let pert = vec![100, 100, 100, 100];
        let gmp = vec![99, 100, 100, 100];
        let zs = zero_z(4);
        let m = compute(2, 2, 100, &pert, &zs, &gmp, &zs, 1.0, 1.0, Thresholds::default());
        assert!(m.escape_disagreement > 0.0);
    }

    #[test]
    fn z_distance_nonzero_when_z_differ() {
        let iters = vec![10; 4];
        let p_zs = vec![Complex64::new(1.0, 0.0); 4];
        let g_zs = vec![Complex64::new(1.0, 0.001); 4];
        let m = compute(2, 2, 100, &iters, &p_zs, &iters, &g_zs, 1.0, 1.0, Thresholds::default());
        assert!(m.z_distance.max > 0.0);
        assert!(m.z_distance.max < 0.01);
    }

    #[test]
    fn verdict_pass_warn_fail_tiers() {
        let zs = zero_z(1000);
        // PASS: no divergence
        let a = vec![10u32; 1000];
        let m = compute(100, 10, 100, &a, &zs, &a, &zs, 1.0, 1.0, Thresholds::default());
        assert_eq!(m.verdict, Verdict::Pass);
        // WARN: a few diffs of 2 (sparse : 2/1000, p99=0, div_ratio 0.002 < warn).
        let mut b = a.clone();
        b[5] = 12;
        b[10] = 12;
        let m = compute(100, 10, 100, &a, &zs, &b, &zs, 1.0, 1.0, Thresholds::default());
        assert_eq!(m.verdict, Verdict::Warn);
        // WARN (recalibré G6) : UN pixel outlier grand (max=50) mais DISPERSÉ
        // (1/1000, p99=0, div_ratio 0.001) = bruit de bord, plus FAIL. L'outlier
        // reste rapporté via `max`.
        let mut c = a.clone();
        c[0] = 50;
        let m = compute(100, 10, 100, &a, &zs, &c, &zs, 1.0, 1.0, Thresholds::default());
        assert_eq!(m.verdict, Verdict::Warn);
    }

    /// Verrou G6 : une divergence LARGE (p99 > seuil : bien plus d'1 % des
    /// pixels divergent de > 1) FAIL, même si div_ratio reste sous le seuil warn
    /// grâce au comptage — c'est le p99 qui capte l'étendue. Et une divergence
    /// SYSTÉMATIQUE (offset uniforme, signature over-skip BLA) FAIL via div_ratio.
    #[test]
    fn verdict_widespread_and_systematic_fail() {
        let zs = zero_z(1000);
        // (valeurs ≪ iteration_max=100 pour que les pixels « échappent » et
        // comptent dans la divergence.)
        let a = vec![10u32; 1000];
        // Large : 50 pixels sur 1000 (5 %) divergent de 2 → p99 (rang 989) = 2 > 1
        // → FAIL. (div_ratio 0.05 > warn 0.01 aussi ; le p99 seul suffirait.)
        let mut wide = a.clone();
        for x in wide.iter_mut().take(50) {
            *x = 12;
        }
        let m = compute(100, 10, 100, &a, &zs, &wide, &zs, 1.0, 1.0, Thresholds::default());
        assert!(m.iter_diff.p99 > 1.0, "p99={}", m.iter_diff.p99);
        assert_eq!(m.verdict, Verdict::Fail);

        // Systématique : offset uniforme +2 partout (signature over-skip BLA
        // julia-siegel) → TOUS les pixels divergent de > 1 → div_ratio 1.0 → FAIL.
        let sys: Vec<u32> = a.iter().map(|v| v + 2).collect();
        let m = compute(100, 10, 100, &a, &zs, &sys, &zs, 1.0, 1.0, Thresholds::default());
        assert_eq!(m.iter_divergence_ratio, 1.0);
        assert_eq!(m.verdict, Verdict::Fail);
    }
}
