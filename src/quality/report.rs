use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use num_complex::Complex64;
use png::Encoder;
use serde::Serialize;

use crate::fractal::FractalParams;
use crate::io::png::save_png_with_metadata;
use crate::quality::metrics::{IterStats, QualityMetrics, Thresholds, TopDivergent, Verdict};

pub struct ReportInputs<'a> {
    pub preset_name: &'a str,
    pub params: &'a FractalParams,
    pub pert_iters: &'a [u32],
    pub pert_zs: &'a [Complex64],
    pub gmp_iters: &'a [u32],
    pub gmp_zs: &'a [Complex64],
    pub metrics: &'a QualityMetrics,
}

pub fn write_report(output_dir: &Path, input: &ReportInputs) -> Result<(), Box<dyn std::error::Error>> {
    let scene_dir = output_dir.join(input.preset_name);
    fs::create_dir_all(&scene_dir)?;

    let (cx_hp, cy_hp, sx_hp, sy_hp) = hp_strings(input.params);

    save_png_with_metadata(
        input.params,
        input.pert_iters,
        input.pert_zs,
        &scene_dir.join("pert.png"),
        &cx_hp, &cy_hp, &sx_hp, &sy_hp,
    )?;
    save_png_with_metadata(
        input.params,
        input.gmp_iters,
        input.gmp_zs,
        &scene_dir.join("gmp.png"),
        &cx_hp, &cy_hp, &sx_hp, &sy_hp,
    )?;

    write_heatmap_png(&scene_dir.join("diff.png"), input.metrics)?;
    write_markdown(&scene_dir.join("report.md"), input)?;
    write_report_json(&scene_dir.join("report.json"), input.preset_name, input.metrics)?;

    Ok(())
}

fn hp_strings(params: &FractalParams) -> (String, String, String, String) {
    let cx = params.center_x_hp.clone().unwrap_or_else(|| params.center_x.to_string());
    let cy = params.center_y_hp.clone().unwrap_or_else(|| params.center_y.to_string());
    let sx = params.span_x_hp.clone().unwrap_or_else(|| params.span_x.to_string());
    let sy = params.span_y_hp.clone().unwrap_or_else(|| params.span_y.to_string());
    (cx, cy, sx, sy)
}

fn write_heatmap_png(path: &Path, m: &QualityMetrics) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = Encoder::new(writer, m.width, m.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header()?;

    let mut rgb = Vec::with_capacity(m.heatmap.len() * 3);
    for &v in &m.heatmap {
        // Black -> deep red -> orange -> yellow ramp for visual distinctness.
        let t = v as f32 / 255.0;
        let r = (t.min(1.0) * 255.0) as u8;
        let g = ((t - 0.4).max(0.0) * 1.67 * 255.0).min(255.0) as u8;
        let b = ((t - 0.8).max(0.0) * 5.0 * 255.0).min(255.0) as u8;
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    png_writer.write_image_data(&rgb)?;
    Ok(())
}

fn write_markdown(path: &Path, input: &ReportInputs) -> Result<(), Box<dyn std::error::Error>> {
    let m = input.metrics;
    let p = input.params;
    let mut f = BufWriter::new(File::create(path)?);

    writeln!(f, "# Quality report — {}", input.preset_name)?;
    writeln!(f)?;
    writeln!(f, "**Verdict: {}**", m.verdict.as_str())?;
    writeln!(f)?;
    writeln!(f, "## Scene")?;
    writeln!(f)?;
    writeln!(f, "| Field | Value |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| fractal_type | {:?} |", p.fractal_type)?;
    writeln!(f, "| width x height | {} x {} |", p.width, p.height)?;
    writeln!(f, "| center_x_hp | `{}` |", p.center_x_hp.clone().unwrap_or_else(|| p.center_x.to_string()))?;
    writeln!(f, "| center_y_hp | `{}` |", p.center_y_hp.clone().unwrap_or_else(|| p.center_y.to_string()))?;
    writeln!(f, "| span_x_hp | `{}` |", p.span_x_hp.clone().unwrap_or_else(|| p.span_x.to_string()))?;
    writeln!(f, "| span_y_hp | `{}` |", p.span_y_hp.clone().unwrap_or_else(|| p.span_y.to_string()))?;
    writeln!(f, "| iteration_max | {} |", p.iteration_max)?;
    writeln!(f, "| precision_bits | {} |", p.precision_bits)?;
    writeln!(f, "| seed | {} + {}i |", p.seed.re, p.seed.im)?;
    writeln!(f)?;

    writeln!(f, "## Timing")?;
    writeln!(f)?;
    writeln!(f, "| Mode | Time (ms) |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| Perturbation | {:.1} |", m.perturb_time_ms)?;
    writeln!(f, "| GMP reference | {:.1} |", m.gmp_time_ms)?;
    writeln!(f, "| Speedup (gmp/pert) | {:.2}x |", m.speedup())?;
    writeln!(f)?;

    writeln!(f, "## Iteration diff")?;
    writeln!(f)?;
    writeln!(f, "| Statistic | Value |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| max | {:.0} |", m.iter_diff.max)?;
    writeln!(f, "| mean | {:.3} |", m.iter_diff.mean)?;
    writeln!(f, "| rms | {:.3} |", m.iter_diff.rms)?;
    writeln!(f, "| p50 | {:.0} |", m.iter_diff.p50)?;
    writeln!(f, "| p95 | {:.0} |", m.iter_diff.p95)?;
    writeln!(f, "| p99 | {:.0} |", m.iter_diff.p99)?;
    writeln!(f, "| divergence_ratio (>1) | {:.5} |", m.iter_divergence_ratio)?;
    writeln!(f, "| escape_disagreement | {:.5} |", m.escape_disagreement)?;
    writeln!(f)?;

    writeln!(f, "## Z distance (|z_pert - z_gmp|)")?;
    writeln!(f)?;
    writeln!(f, "| Statistic | Value |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| max | {:.3e} |", m.z_distance.max)?;
    writeln!(f, "| mean | {:.3e} |", m.z_distance.mean)?;
    writeln!(f, "| rms | {:.3e} |", m.z_distance.rms)?;
    writeln!(f, "| p95 | {:.3e} |", m.z_distance.p95)?;
    writeln!(f, "| p99 | {:.3e} |", m.z_distance.p99)?;
    writeln!(f)?;

    writeln!(f, "## Z relative error (|dz| / |z_gmp|)")?;
    writeln!(f)?;
    writeln!(f, "| Statistic | Value |")?;
    writeln!(f, "|---|---|")?;
    writeln!(f, "| max | {:.3e} |", m.z_ratio_error.max)?;
    writeln!(f, "| mean | {:.3e} |", m.z_ratio_error.mean)?;
    writeln!(f, "| p95 | {:.3e} |", m.z_ratio_error.p95)?;
    writeln!(f, "| p99 | {:.3e} |", m.z_ratio_error.p99)?;
    writeln!(f)?;

    writeln!(f, "## Top 10 divergent pixels")?;
    writeln!(f)?;
    if m.top_divergent.is_empty() {
        writeln!(f, "_(no divergence)_")?;
    } else {
        writeln!(f, "| x | y | pert_iter | gmp_iter | diff | |dz| | z_pert | z_gmp |")?;
        writeln!(f, "|---|---|---|---|---|---|---|---|")?;
        for d in &m.top_divergent {
            writeln!(
                f,
                "| {} | {} | {} | {} | {} | {:.3e} | {} | {} |",
                d.x, d.y, d.pert_iter, d.gmp_iter, d.iter_diff, d.z_distance,
                format_z(d.pert_z), format_z(d.gmp_z),
            )?;
        }
    }
    writeln!(f)?;

    writeln!(f, "## Files")?;
    writeln!(f)?;
    writeln!(f, "- `pert.png` — perturbation render (loadable via drag-and-drop in fractall-gui)")?;
    writeln!(f, "- `gmp.png` — pure GMP reference render")?;
    writeln!(f, "- `diff.png` — per-pixel iteration-diff heatmap (black = match, yellow = max diff = {:.0})", m.iter_diff.max)?;
    writeln!(f, "- `report.md` — this file")?;

    Ok(())
}

fn format_z(z: Complex64) -> String {
    format!("{:.3e}+{:.3e}i", z.re, z.im)
}

// ---------------------------------------------------------------------------
// Machine-readable JSON output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct IterDiffJson {
    max: f64,
    mean: f64,
    rms: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

impl IterDiffJson {
    fn from(s: &IterStats) -> Self {
        IterDiffJson {
            max: s.max,
            mean: s.mean,
            rms: s.rms,
            p50: s.p50,
            p95: s.p95,
            p99: s.p99,
        }
    }
}

#[derive(Serialize)]
struct ZStatsJson {
    max: f64,
    mean: f64,
    rms: f64,
    p95: f64,
    p99: f64,
}

impl ZStatsJson {
    fn from(s: &IterStats) -> Self {
        ZStatsJson { max: s.max, mean: s.mean, rms: s.rms, p95: s.p95, p99: s.p99 }
    }
}

#[derive(Serialize)]
struct TopDivergentJson {
    x: u32,
    y: u32,
    pert_iter: u32,
    gmp_iter: u32,
    iter_diff: u32,
    z_distance: f64,
    pert_z: [f64; 2],
    gmp_z: [f64; 2],
}

impl TopDivergentJson {
    fn from(d: &TopDivergent) -> Self {
        TopDivergentJson {
            x: d.x,
            y: d.y,
            pert_iter: d.pert_iter,
            gmp_iter: d.gmp_iter,
            iter_diff: d.iter_diff,
            z_distance: d.z_distance,
            pert_z: [d.pert_z.re, d.pert_z.im],
            gmp_z: [d.gmp_z.re, d.gmp_z.im],
        }
    }
}

/// Compact per-preset object used in the suite summary JSON.
#[derive(Serialize)]
struct PresetSummaryJson {
    name: String,
    verdict: Verdict,
    iter_diff: IterDiffJson,
    divergence_ratio: f64,
    escape_disagreement: f64,
    z_distance_max: f64,
    time_pert_ms: f64,
    time_gmp_ms: f64,
    speedup: f64,
}

impl PresetSummaryJson {
    fn from(name: &str, m: &QualityMetrics) -> Self {
        PresetSummaryJson {
            name: name.to_string(),
            verdict: m.verdict,
            iter_diff: IterDiffJson::from(&m.iter_diff),
            divergence_ratio: m.iter_divergence_ratio,
            escape_disagreement: m.escape_disagreement,
            z_distance_max: m.z_distance.max,
            time_pert_ms: m.perturb_time_ms,
            time_gmp_ms: m.gmp_time_ms,
            speedup: m.speedup(),
        }
    }
}

/// Full per-preset object written to `<name>/report.json` (heatmap excluded).
#[derive(Serialize)]
struct ReportJson {
    name: String,
    verdict: Verdict,
    width: u32,
    height: u32,
    iter_diff: IterDiffJson,
    divergence_ratio: f64,
    escape_disagreement: f64,
    z_distance: ZStatsJson,
    z_ratio_error: ZStatsJson,
    time_pert_ms: f64,
    time_gmp_ms: f64,
    speedup: f64,
    top_divergent: Vec<TopDivergentJson>,
}

impl ReportJson {
    fn from(name: &str, m: &QualityMetrics) -> Self {
        ReportJson {
            name: name.to_string(),
            verdict: m.verdict,
            width: m.width,
            height: m.height,
            iter_diff: IterDiffJson::from(&m.iter_diff),
            divergence_ratio: m.iter_divergence_ratio,
            escape_disagreement: m.escape_disagreement,
            z_distance: ZStatsJson::from(&m.z_distance),
            z_ratio_error: ZStatsJson::from(&m.z_ratio_error),
            time_pert_ms: m.perturb_time_ms,
            time_gmp_ms: m.gmp_time_ms,
            speedup: m.speedup(),
            top_divergent: m.top_divergent.iter().map(TopDivergentJson::from).collect(),
        }
    }
}

#[derive(Serialize)]
struct ThresholdsJson {
    max_iter_diff_pass: f64,
    divergence_ratio_pass: f64,
    max_iter_diff_warn: f64,
    divergence_ratio_warn: f64,
}

impl ThresholdsJson {
    fn from(t: &Thresholds) -> Self {
        ThresholdsJson {
            max_iter_diff_pass: t.max_iter_diff_pass,
            divergence_ratio_pass: t.divergence_ratio_pass,
            max_iter_diff_warn: t.max_iter_diff_warn,
            divergence_ratio_warn: t.divergence_ratio_warn,
        }
    }
}

#[derive(Serialize)]
struct TotalsJson {
    pass: usize,
    warn: usize,
    fail: usize,
}

#[derive(Serialize)]
struct SuiteJson {
    generated_utc: String,
    thresholds: ThresholdsJson,
    presets: Vec<PresetSummaryJson>,
    totals: TotalsJson,
}

fn write_report_json(
    path: &Path,
    name: &str,
    m: &QualityMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    let report = ReportJson::from(name, m);
    let json = serde_json::to_string_pretty(&report)?;
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(json.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}

/// Write `<output-dir>/suite-summary.json`. Always called by the suite command.
pub fn write_suite_summary_json(
    output_dir: &Path,
    rows: &[(String, QualityMetrics)],
    thresholds: &Thresholds,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let mut counts = TotalsJson { pass: 0, warn: 0, fail: 0 };
    let presets: Vec<PresetSummaryJson> = rows
        .iter()
        .map(|(name, m)| {
            match m.verdict {
                Verdict::Pass => counts.pass += 1,
                Verdict::Warn => counts.warn += 1,
                Verdict::Fail => counts.fail += 1,
            }
            PresetSummaryJson::from(name, m)
        })
        .collect();

    let suite = SuiteJson {
        generated_utc: now_utc_rfc3339(),
        thresholds: ThresholdsJson::from(thresholds),
        presets,
        totals: counts,
    };
    let json = serde_json::to_string_pretty(&suite)?;
    let mut f = BufWriter::new(File::create(output_dir.join("suite-summary.json"))?);
    f.write_all(json.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}

/// Current UTC time as an RFC 3339 string (`YYYY-MM-DDTHH:MM:SSZ`), no deps.
fn now_utc_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Civil-from-days (Howard Hinnant's algorithm), epoch 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

pub fn write_suite_summary(
    output_dir: &Path,
    rows: &[(String, QualityMetrics)],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output_dir)?;
    let mut f = BufWriter::new(File::create(output_dir.join("suite-summary.md"))?);
    writeln!(f, "# Quality suite summary")?;
    writeln!(f)?;
    writeln!(f, "| Preset | Verdict | max_iter_diff | p99_iter_diff | divergence_ratio | escape_disagree | max_|dz| | time_pert_ms | time_gmp_ms | speedup |")?;
    writeln!(f, "|---|---|---|---|---|---|---|---|---|---|")?;
    let mut counts = [0usize; 3];
    for (name, m) in rows {
        let verdict_cell = format!("**{}**", m.verdict.as_str());
        match m.verdict {
            Verdict::Pass => counts[0] += 1,
            Verdict::Warn => counts[1] += 1,
            Verdict::Fail => counts[2] += 1,
        }
        writeln!(
            f,
            "| [{name}]({name}/report.md) | {verdict} | {max:.0} | {p99:.0} | {dr:.5} | {ed:.5} | {zmax:.3e} | {tp:.0} | {tg:.0} | {sp:.2}x |",
            name = name,
            verdict = verdict_cell,
            max = m.iter_diff.max,
            p99 = m.iter_diff.p99,
            dr = m.iter_divergence_ratio,
            ed = m.escape_disagreement,
            zmax = m.z_distance.max,
            tp = m.perturb_time_ms,
            tg = m.gmp_time_ms,
            sp = m.speedup(),
        )?;
    }
    writeln!(f)?;
    writeln!(f, "Totals: {} PASS · {} WARN · {} FAIL", counts[0], counts[1], counts[2])?;
    Ok(())
}

#[allow(dead_code)]
pub fn print_summary_line(name: &str, m: &QualityMetrics) {
    println!(
        "[{:>5}] {:<30} max_diff={:>4} p99={:>3} div_ratio={:.5} time pert={:.0}ms gmp={:.0}ms speedup={:.2}x",
        m.verdict.as_str(),
        name,
        m.iter_diff.max as u32,
        m.iter_diff.p99 as u32,
        m.iter_divergence_ratio,
        m.perturb_time_ms,
        m.gmp_time_ms,
        m.speedup(),
    );
}

