use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use num_complex::Complex64;
use png::Encoder;

use crate::fractal::FractalParams;
use crate::io::png::save_png_with_metadata;
use crate::quality::metrics::{QualityMetrics, Verdict};

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

