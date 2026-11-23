//! Export des paramètres résolus au format natif Fraktaler-3 3.1.

use std::io::{Error, ErrorKind};
use std::path::Path;

use crate::fractal::bytecode::{formula_for_params, Formula, Op};
use crate::fractal::{FractalParams, ViewHp};

fn quoted(value: &str) -> String {
    toml::Value::String(value.to_owned()).to_string()
}

pub fn to_f3_toml(params: &FractalParams) -> std::io::Result<String> {
    let unsupported = || Error::new(
        ErrorKind::InvalidInput,
        format!("the formula for {:?} cannot be represented in this Fraktaler-3 export", params.fractal_type),
    );
    if Formula::is_julia_for(params.fractal_type) {
        return Err(unsupported());
    }
    let formula = formula_for_params(params).ok_or_else(unsupported)?;
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
        params.perturbation.max_perturb_iterations.max(params.iteration_max),
        params.perturbation.max_bla_steps.max(params.iteration_max), params.bailout,
        params.width, params.height, params.rotation,
    );

    if let Some(opcodes) = params.formula.hybrid_opcodes.as_deref().filter(|ops| !ops.trim().is_empty()) {
        let mut phase = Vec::new();
        for word in opcodes.split_whitespace() {
            phase.push(word);
            if word == "add" {
                text.push_str(&format!("\n[[formula]]\nopcodes = {}\n", quoted(&phase.join(" "))));
                phase.clear();
            }
        }
    } else {
        for phase in &formula.phases {
            let mut words = Vec::with_capacity(phase.ops.len());
            for op in &phase.ops {
                words.push(match op {
                    Op::Add => "add",
                    Op::Sqr => "sqr",
                    Op::Mul => "mul",
                    Op::Store => "store",
                    Op::AbsX => "absx",
                    Op::AbsY => "absy",
                    Op::NegX => "negx",
                    Op::NegY => "negy",
                    // Rotations only occur in explicit opcodes, handled above
                    // to retain the original angle before f32 rounding.
                    Op::Rot { .. } => return Err(unsupported()),
                });
            }
            text.push_str(&format!("\n[[formula]]\nopcodes = {}\n", quoted(&words.join(" "))));
        }
    }
    Ok(text)
}

pub fn save_f3_toml(path: &Path, params: &FractalParams) -> std::io::Result<()> {
    std::fs::write(path, to_f3_toml(params)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    #[test]
    fn exported_builtin_phases_use_the_renderers_formula() {
        let mut p = default_params_for_type(FractalType::Multibrot, 48, 32);
        p.formula.multibrot_power = 3.0;
        p.formula.hybrid_phases = Some(vec![FractalType::Multibrot, FractalType::PerpendicularBurningShip]);
        let table: toml::Table = to_f3_toml(&p).unwrap().parse().unwrap();
        assert_eq!(table["formula"][0]["opcodes"].as_str(), Some("store sqr mul add"));
        assert_eq!(table["formula"][1]["opcodes"].as_str(), Some("absy negy sqr add"));
        p.formula.multibrot_power = 2.5;
        assert!(to_f3_toml(&p).is_err());
    }

    #[test]
    fn native_export_round_trips_structural_fields() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 320, 180);
        p.center_x_hp = Some("-0.743643887037158704752191506114774".into());
        p.center_y_hp = Some("0.131825904205311970493132056385139".into());
        p.iteration_max = 1200;
        p.rotation = 7.5;
        p.formula.hybrid_opcodes = Some("sqr add absx absy sqr add".into());
        let text = to_f3_toml(&p).unwrap();
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(table["location"]["real"].as_str(), p.center_x_hp.as_deref());
        assert_eq!(table["bailout"]["iterations"].as_integer(), Some(1200));
        assert_eq!(table["image"]["width"].as_integer(), Some(320));
        assert_eq!(table["formula"].as_array().unwrap().len(), 2);
    }
}
