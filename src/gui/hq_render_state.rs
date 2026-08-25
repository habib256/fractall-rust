use crate::fractal::FractalParams;

/// Params du rendu haute résolution : dimensions cibles, span ajusté au ratio
/// d'aspect demandé, et **invariant mode → canaux** posé ici comme sur toute
/// autre frontière de rendu (`render::ensure_required_channels`).
///
/// Fonction PURE, donc verrouillable : avant extraction, l'export HQ clonait
/// les params de la fenêtre sans la règle des canaux, si bien qu'un export 4K
/// en mode Distance/OrbitTraps partait sans son canal et mourait sur la
/// colorisation vérifiée — la règle n'existait que sur le clone du rendu
/// fenêtre.
pub fn hq_render_params(
    params: &FractalParams,
    render_width: u32,
    render_height: u32,
) -> FractalParams {
    let mut out = params.clone();
    crate::render::ensure_required_channels(&mut out);
    out.width = render_width;
    out.height = render_height;

    // Conserver le ratio d'aspect : on ÉLARGIT l'axe le plus contraint (jamais
    // de recadrage par rapport à ce que l'utilisateur voit).
    let current_aspect = params.span_x / params.span_y;
    let target_aspect = render_width as f64 / render_height as f64;
    if current_aspect > target_aspect {
        out.span_y = out.span_x / target_aspect;
    } else {
        out.span_x = out.span_y * target_aspect;
    }
    out
}

/// Événement produit par le worker de rendu haute qualité.
pub enum HqRenderEvent {
    Progress(f32),
    Done(String),
    Error(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum HqRenderResult {
    Saved(String),
    Error(String),
}

/// Cycle de vie du rendu haute qualité, indépendant d'egui et du canal worker.
#[derive(Debug, Default)]
pub struct HqRenderState {
    running: bool,
    progress: f32,
    result: Option<HqRenderResult>,
}

impl HqRenderState {
    pub fn begin(&mut self) {
        self.running = true;
        self.progress = 0.0;
        self.result = None;
    }

    pub fn apply(&mut self, event: HqRenderEvent) {
        if !self.running {
            return;
        }
        match event {
            HqRenderEvent::Progress(progress) => {
                self.progress = self.progress.max(progress.clamp(0.0, 1.0));
            }
            HqRenderEvent::Done(filename) => {
                self.running = false;
                self.progress = 1.0;
                self.result = Some(HqRenderResult::Saved(filename));
            }
            HqRenderEvent::Error(error) => {
                self.running = false;
                self.result = Some(HqRenderResult::Error(error));
            }
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn progress(&self) -> f32 {
        self.progress
    }

    pub fn result(&self) -> Option<&HqRenderResult> {
        self.result.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_resets_previous_terminal_state() {
        let mut state = HqRenderState::default();
        state.begin();
        state.apply(HqRenderEvent::Error("disk full".into()));
        state.begin();

        assert!(state.is_running());
        assert_eq!(state.progress(), 0.0);
        assert_eq!(state.result(), None);
    }

    #[test]
    fn progress_is_clamped_and_monotone() {
        let mut state = HqRenderState::default();
        state.begin();
        state.apply(HqRenderEvent::Progress(0.8));
        state.apply(HqRenderEvent::Progress(0.2));
        assert_eq!(state.progress(), 0.8);

        state.apply(HqRenderEvent::Progress(2.0));
        assert_eq!(state.progress(), 1.0);
    }

    #[test]
    fn terminal_result_rejects_late_worker_events() {
        let mut state = HqRenderState::default();
        state.begin();
        state.apply(HqRenderEvent::Done("image.png".into()));
        state.apply(HqRenderEvent::Error("late error".into()));
        state.apply(HqRenderEvent::Progress(0.1));

        assert!(!state.is_running());
        assert_eq!(state.progress(), 1.0);
        assert_eq!(
            state.result(),
            Some(&HqRenderResult::Saved("image.png".into()))
        );
    }
}


#[cfg(test)]
mod hq_params_tests {
    use super::hq_render_params;
    use crate::fractal::{default_params_for_type, FractalType, OutColoringMode};
    use crate::render::required_channels;

    /// Tout mode de coloriage qui consomme un canal repart de l'export HQ avec
    /// ce canal PRODUIT : sinon la colorisation vérifiée refuse l'image et le
    /// rendu 4K échoue.
    #[test]
    fn hq_params_satisfy_required_channels_for_every_mode() {
        for mode in OutColoringMode::all() {
            let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
            p.color.out_coloring_mode = *mode;
            p.channels.enable_distance_estimation = false;
            p.channels.enable_orbit_traps = false;

            let hq = hq_render_params(&p, 3840, 2160);
            let req = required_channels(&hq);
            assert!(
                !req.distances || hq.channels.enable_distance_estimation,
                "{mode:?} : canal distances manquant"
            );
            assert!(
                !req.orbits || hq.channels.enable_orbit_traps,
                "{mode:?} : canal orbits manquant"
            );
        }
    }

    /// Le ratio d'aspect cible est atteint et l'axe conservé n'est jamais
    /// rétréci (pas de recadrage vs la fenêtre).
    #[test]
    fn hq_params_match_target_aspect_without_cropping() {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        p.span_x = 4.0;
        p.span_y = 3.0;

        let wide = hq_render_params(&p, 3840, 2160);
        assert_eq!((wide.width, wide.height), (3840, 2160));
        let aspect = wide.span_x / wide.span_y;
        assert!((aspect - 3840.0 / 2160.0).abs() < 1e-12, "aspect = {aspect}");
        assert!(wide.span_x >= p.span_x - 1e-12 && wide.span_y >= p.span_y - 1e-12);

        let tall = hq_render_params(&p, 1080, 1920);
        let aspect = tall.span_x / tall.span_y;
        assert!((aspect - 1080.0 / 1920.0).abs() < 1e-12, "aspect = {aspect}");
        assert!(tall.span_x >= p.span_x - 1e-12 && tall.span_y >= p.span_y - 1e-12);
    }
}
