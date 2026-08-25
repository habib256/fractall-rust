//! `RenderOutput` : résultat TYPÉ du dispatcher escape-time (chantier G5).
//!
//! Remplace le tuple `(iterations, zs, orbits, distances)` : jeter un canal
//! ne compile plus silencieusement (accès par champ nommé), et la
//! colorisation VÉRIFIE que les canaux requis par le mode sont présents
//! (`required_channels` + `ChannelRequirements::validate`) — une image
//! « plausible mais fausse » (retombée silencieuse sur Smooth) devient une
//! erreur explicite. Classe fermée : copie CLI, boucle AA, pipeline vidéo,
//! auto-GPU (4 bugs « Smooth silencieux », chasse 2026-08-23).
//!
//! Sémantique des canaux : un canal ABSENT est un `Vec` VIDE (le path ne
//! l'a pas produit) ; un canal produit a exactement `width × height`
//! entrées (les pixels sans valeur portent `f64::INFINITY` / `None`).

use num_complex::Complex64;

use crate::fractal::orbit_traps::OrbitData;
use crate::fractal::types::{FractalParams, FractalType};

/// Résultat complet d'un rendu escape-time (dispatcher unique CLI ↔ GUI).
#[derive(Debug, Clone, Default)]
pub struct RenderOutput {
    /// Compte d'itération par pixel (row-major).
    pub iterations: Vec<u32>,
    /// z final par pixel (smooth iteration, biomorphs, …).
    pub zs: Vec<Complex64>,
    /// Canal orbit-traps (modes OrbitTraps/Wings). VIDE si non produit —
    /// rempli par le path f64 standard quand `enable_orbit_traps`.
    pub orbits: Vec<Option<OrbitData>>,
    /// Canal distance (modes Distance/DistanceAO/Distance3D). VIDE si non
    /// produit — rempli par les paths f64/perturbation quand
    /// `enable_distance_estimation` (`INFINITY` = pas d'estimation au pixel).
    pub distances: Vec<f64>,
}

impl RenderOutput {
    /// Rendu sans canaux annexes (types spéciaux, GPU, paths qui n'en
    /// produisent pas).
    pub fn without_extras(iterations: Vec<u32>, zs: Vec<Complex64>) -> Self {
        Self {
            iterations,
            zs,
            orbits: Vec::new(),
            distances: Vec::new(),
        }
    }

    /// Vérifie que les canaux requis par `params.out_coloring_mode` sont
    /// présents. À appeler au point de colorisation (cf.
    /// `io::png::colorize_output`) : `Err` au lieu d'une image
    /// plausible-mais-fausse.
    pub fn validate_channels(&self, params: &FractalParams) -> Result<(), String> {
        required_channels(params).validate(self.iterations.len(), &self.distances, &self.orbits)
    }
}

/// Canaux annexes que la colorisation de `params` consomme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelRequirements {
    /// Le mode consomme le canal `distances`.
    pub distances: bool,
    /// Le mode consomme le canal `orbits`.
    pub orbits: bool,
}

impl ChannelRequirements {
    /// Aucun canal annexe requis.
    pub fn none(self) -> bool {
        !self.distances && !self.orbits
    }

    /// Vérifie la présence des canaux requis pour `n` pixels. Les slices
    /// vides signifient « canal non produit par le path de rendu ».
    pub fn validate(
        self,
        n: usize,
        distances: &[f64],
        orbits: &[Option<OrbitData>],
    ) -> Result<(), String> {
        if self.distances && distances.len() < n {
            return Err(
                "le mode de colorisation Distance/DistanceAO/Distance3D requiert le canal \
                 `distances`, absent de ce rendu (l'image retomberait silencieusement sur \
                 Smooth) — activer l'estimation de distance (--enable-distance-estimation / \
                 [fractal] distance_estimation) et vérifier que le path de rendu la supporte"
                    .to_string(),
            );
        }
        if self.orbits && orbits.len() < n {
            return Err(
                "le mode de colorisation OrbitTraps/Wings requiert le canal `orbits`, absent \
                 de ce rendu (l'image retomberait silencieusement sur Smooth) — ces modes ne \
                 sont produits que par le path f64 standard (enable_orbit_traps)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Canaux requis par la colorisation de `params`. Les types à colorisation
/// dédiée (densité Buddhabrot/Nebulabrot/Anti-Buddhabrot, vectoriels
/// Von Koch/Dragon) ignorent `out_coloring_mode` → aucun canal requis.
pub fn required_channels(params: &FractalParams) -> ChannelRequirements {
    let mode_driven = !matches!(
        params.fractal_type,
        FractalType::Buddhabrot
            | FractalType::Nebulabrot
            | FractalType::AntiBuddhabrot
            | FractalType::VonKoch
            | FractalType::Dragon
    );
    ChannelRequirements {
        distances: mode_driven && params.color.out_coloring_mode.requires_distance_channel(),
        orbits: mode_driven && params.color.out_coloring_mode.requires_orbit_channel(),
    }
}

/// Rétablit l'invariant INTER-GROUPES `channels ⊇ required_channels(params)` :
/// un mode de coloriage qui CONSOMME un canal doit le faire PRODUIRE.
///
/// Règle UNIQUE, à appliquer à chaque frontière d'entrée d'un rendu (CLI, GUI
/// fenêtre / HQ / preview). Elle était réimplémentée à trois endroits avec
/// trois tables différentes ; la GUI ne la posait que sur le clone du rendu
/// fenêtre, si bien qu'un export haute résolution en mode Distance partait
/// sans le canal et **échouait** à la colorisation vérifiée.
///
/// Monotone : n'active jamais que ce que le mode exige, ne désactive rien
/// (l'utilisateur peut demander un canal sans mode correspondant). Les types
/// à colorisation dédiée (densité, vectoriels) n'exigent aucun canal —
/// `required_channels` le sait déjà, ils ne sont donc pas touchés.
///
/// Retourne `true` si un canal a dû être activé.
pub fn ensure_required_channels(params: &mut FractalParams) -> bool {
    let req = required_channels(params);
    let mut changed = false;
    if req.distances && !params.channels.enable_distance_estimation {
        params.channels.enable_distance_estimation = true;
        changed = true;
    }
    if req.orbits && !params.channels.enable_orbit_traps {
        params.channels.enable_orbit_traps = true;
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, types::OutColoringMode};

    fn params(t: FractalType, mode: OutColoringMode) -> FractalParams {
        let mut p = default_params_for_type(t, 8, 8);
        p.color.out_coloring_mode = mode;
        p
    }

    #[test]
    fn required_channels_follows_outcoloring_mode() {
        let p = params(FractalType::Mandelbrot, OutColoringMode::Smooth);
        assert!(required_channels(&p).none());
        for m in [
            OutColoringMode::Distance,
            OutColoringMode::DistanceAO,
            OutColoringMode::Distance3D,
        ] {
            let p = params(FractalType::Mandelbrot, m);
            let req = required_channels(&p);
            assert!(req.distances && !req.orbits, "{m:?}");
        }
        for m in [OutColoringMode::OrbitTraps, OutColoringMode::Wings] {
            let p = params(FractalType::Mandelbrot, m);
            let req = required_channels(&p);
            assert!(req.orbits && !req.distances, "{m:?}");
        }
    }

    #[test]
    fn density_and_vectorial_types_require_no_channels() {
        for t in [
            FractalType::Buddhabrot,
            FractalType::Nebulabrot,
            FractalType::AntiBuddhabrot,
            FractalType::VonKoch,
            FractalType::Dragon,
        ] {
            let p = params(t, OutColoringMode::Distance);
            assert!(required_channels(&p).none(), "{t:?}");
        }
    }

    #[test]
    fn validate_channels_errs_on_missing_channel() {
        let n = 64usize;
        let out_bare = RenderOutput::without_extras(vec![0; n], vec![Complex64::new(0.0, 0.0); n]);

        // Smooth : aucun canal requis → OK sans extras.
        let p = params(FractalType::Mandelbrot, OutColoringMode::Smooth);
        assert!(out_bare.validate_channels(&p).is_ok());

        // Distance sans canal → Err ; avec canal → OK.
        let p = params(FractalType::Mandelbrot, OutColoringMode::Distance);
        assert!(out_bare.validate_channels(&p).is_err());
        let mut out = out_bare.clone();
        out.distances = vec![f64::INFINITY; n];
        assert!(out.validate_channels(&p).is_ok());

        // OrbitTraps sans canal → Err ; avec canal → OK.
        let p = params(FractalType::Mandelbrot, OutColoringMode::Wings);
        assert!(out_bare.validate_channels(&p).is_err());
        let mut out = out_bare.clone();
        out.orbits = vec![None; n];
        assert!(out.validate_channels(&p).is_ok());
    }

    /// L'invariant tient pour TOUS les modes, sur un type piloté par le mode.
    #[test]
    fn ensure_required_channels_satisfies_every_mode() {
        for mode in OutColoringMode::all() {
            let mut p = params(FractalType::Mandelbrot, *mode);
            p.channels.enable_distance_estimation = false;
            p.channels.enable_orbit_traps = false;

            let changed = ensure_required_channels(&mut p);
            let req = required_channels(&p);
            assert_eq!(
                changed,
                !req.none(),
                "{mode:?} : `changed` doit refléter l'activation effective"
            );
            assert!(!req.distances || p.channels.enable_distance_estimation);
            assert!(!req.orbits || p.channels.enable_orbit_traps);
            // Idempotent : une seconde passe ne change plus rien.
            assert!(!ensure_required_channels(&mut p), "{mode:?} : non idempotent");
        }
    }

    /// Monotone : un canal demandé explicitement survit à la normalisation,
    /// même si le mode ne le consomme pas (l'utilisateur peut vouloir la
    /// donnée sans la coloriser).
    #[test]
    fn ensure_required_channels_never_disables_a_requested_channel() {
        let mut p = params(FractalType::Mandelbrot, OutColoringMode::Smooth);
        p.channels.enable_distance_estimation = true;
        p.channels.enable_orbit_traps = true;
        assert!(!ensure_required_channels(&mut p));
        assert!(p.channels.enable_distance_estimation);
        assert!(p.channels.enable_orbit_traps);
    }

    /// Types à colorisation dédiée : `out_coloring_mode` est ignoré, donc
    /// aucun canal n'est requis — et surtout aucun canal coûteux activé.
    #[test]
    fn ensure_required_channels_leaves_dedicated_coloring_types_alone() {
        for t in [FractalType::Buddhabrot, FractalType::VonKoch] {
            let mut p = params(t, OutColoringMode::Distance);
            assert!(!ensure_required_channels(&mut p), "{t:?}");
            assert!(!p.channels.enable_distance_estimation, "{t:?}");
        }
    }
}
