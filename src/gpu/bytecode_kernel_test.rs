//! Validation syntaxique du shader bytecode_kernel.wgsl via naga.
//!
//! Pas de test runtime GPU (nécessite hardware + Adapter), juste parsing.
//! Catch les erreurs WGSL au `cargo test` sans avoir besoin de tourner le binaire.

#[cfg(test)]
mod tests {
    /// Le shader doit parser via naga (l'analyseur WGSL utilisé par wgpu).
    #[test]
    fn bytecode_kernel_wgsl_parses() {
        let source = include_str!("bytecode_kernel.wgsl");
        let module = naga::front::wgsl::parse_str(source);
        assert!(
            module.is_ok(),
            "bytecode_kernel.wgsl ne parse pas : {:?}",
            module.err()
        );
    }

    /// Le shader doit aussi valider (typage, contraintes WGSL).
    #[test]
    fn bytecode_kernel_wgsl_validates() {
        let source = include_str!("bytecode_kernel.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .expect("parse failed");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        let result = validator.validate(&module);
        assert!(
            result.is_ok(),
            "bytecode_kernel.wgsl ne valide pas : {:?}",
            result.err()
        );
    }

    /// Idem pour le kernel perturbation F3-strict (f64 natif → capability
    /// FLOAT64, mirror du device SHADER_F64).
    #[test]
    fn perturbation_wgsl_validates() {
        let source = include_str!("perturbation.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .expect("perturbation.wgsl ne parse pas");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default() | naga::valid::Capabilities::FLOAT64,
        );
        let result = validator.validate(&module);
        assert!(
            result.is_ok(),
            "perturbation.wgsl ne valide pas : {:?}",
            result.err()
        );
    }
}
