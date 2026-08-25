//! Export des paramètres résolus au format natif Fraktaler-3 3.1.

use std::path::Path;

use crate::fractal::{FractalParams, FractalType, ViewHp};

fn quoted(value: &str) -> String {
    toml::Value::String(value.to_owned()).to_string()
}

fn phase_opcodes(kind: FractalType) -> &'static str {
    match kind {
        FractalType::BurningShip => "absx absy sqr add",
        FractalType::Tricorn => "sqr negy add",
        FractalType::Celtic => "sqr absx add",
        FractalType::Buffalo => "sqr absx absy add",
        FractalType::PerpendicularBurningShip => "absx sqr add",
        _ => "sqr add",
    }
}

pub fn to_f3_toml(params: &FractalParams) -> String {
    let view = ViewHp::from_params(params);
    let (view_real, view_imag, _, _) = view.decimal_parts();
    // Une saisie décimale HP est le contrat exact de l'utilisateur ; ne pas la
    // remplacer par l'expansion de sa matérialisation MPFR finie.
    let real = params.center_x_hp.as_deref().unwrap_or(&view_real);
    let imag = params.center_y_hp.as_deref().unwrap_or(&view_imag);
    let mut text = format!(
        "program = \"fraktaler-3\"\nversion = \"3.1\"\n\n[location]\nreal = {}\nimag = {}\nzoom = {}\n\n[bailout]\niterations = {}\nmaximum_reference_iterations = {}\nmaximum_perturb_iterations = {}\nmaximum_bla_steps = {}\nescape_radius = {}\n\n[image]\nwidth = {}\nheight = {}\nsubframes = 1\n\n[transform]\nrotate = {}\n",
        quoted(real), quoted(imag), quoted(&view.zoom_string()),
        params.iteration_max, params.iteration_max,
        params.max_perturb_iterations.max(params.iteration_max),
        params.max_bla_steps.max(params.iteration_max), params.bailout,
        params.width, params.height, params.rotation,
    );

    if let Some(opcodes) = params.hybrid_opcodes.as_deref() {
        let mut phase = Vec::new();
        for word in opcodes.split_whitespace() {
            phase.push(word);
            if word == "add" {
                text.push_str(&format!("\n[[formula]]\nopcodes = {}\n", quoted(&phase.join(" "))));
                phase.clear();
            }
        }
    } else if let Some(phases) = params.hybrid_phases.as_deref() {
        for &kind in phases {
            text.push_str(&format!("\n[[formula]]\nopcodes = {}\n", quoted(phase_opcodes(kind))));
        }
    } else {
        text.push_str(&format!("\n[[formula]]\nopcodes = {}\n", quoted(phase_opcodes(params.fractal_type))));
    }
    text
}

pub fn save_f3_toml(path: &Path, params: &FractalParams) -> std::io::Result<()> {
    std::fs::write(path, to_f3_toml(params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::default_params_for_type;

    #[test]
    fn native_export_round_trips_structural_fields() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 320, 180);
        p.center_x_hp = Some("-0.743643887037158704752191506114774".into());
        p.center_y_hp = Some("0.131825904205311970493132056385139".into());
        p.iteration_max = 1200;
        p.rotation = 7.5;
        p.hybrid_opcodes = Some("sqr add absx absy sqr add".into());
        let text = to_f3_toml(&p);
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(table["location"]["real"].as_str(), p.center_x_hp.as_deref());
        assert_eq!(table["bailout"]["iterations"].as_integer(), Some(1200));
        assert_eq!(table["image"]["width"].as_integer(), Some(320));
        assert_eq!(table["formula"].as_array().unwrap().len(), 2);
    }
}
