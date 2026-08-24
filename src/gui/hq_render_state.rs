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
