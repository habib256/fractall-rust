//! Machine d'état pure de progression rendu progressif + anti-aliasing.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderProgress {
    completed_passes: u8,
    total_passes: u8,
    aa: Option<(u32, u32)>,
}

impl RenderProgress {
    pub fn begin(&mut self, total_passes: usize, aa_samples: u32) {
        self.completed_passes = 0;
        self.total_passes = total_passes.min(u8::MAX as usize) as u8;
        self.aa = (aa_samples > 1).then_some((0, aa_samples));
    }

    pub fn pass_ready(&mut self, pass_index: u8) {
        self.completed_passes = self
            .completed_passes
            .max(pass_index.saturating_add(1).min(self.total_passes));
    }

    pub fn aa_ready(&mut self, sample: u32, total: u32) {
        if total > 1 {
            self.aa = Some((sample.min(total), total));
        }
    }

    pub fn passes(self) -> (u8, u8) {
        (self.completed_passes, self.total_passes)
    }

    pub fn aa(self) -> Option<(u32, u32)> {
        self.aa
    }

    pub fn fraction(self) -> f32 {
        self.completed_passes as f32 / self.total_passes.max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_is_monotone_and_bounded_by_plan() {
        let mut p = RenderProgress::default();
        p.begin(3, 1);
        p.pass_ready(0);
        assert_eq!(p.passes(), (1, 3));
        p.pass_ready(u8::MAX);
        assert_eq!(p.passes(), (3, 3));
        p.pass_ready(0); // texture asynchrone ancienne arrivée en retard
        assert_eq!(p.passes(), (3, 3));
        assert_eq!(p.fraction(), 1.0);
        assert_eq!(p.aa(), None);
    }

    #[test]
    fn aa_progress_is_clamped_and_reset_on_new_render() {
        let mut p = RenderProgress::default();
        p.begin(2, 4);
        assert_eq!(p.aa(), Some((0, 4)));
        p.aa_ready(9, 4);
        assert_eq!(p.aa(), Some((4, 4)));
        p.begin(1, 1);
        assert_eq!(p.aa(), None);
        assert_eq!(p.passes(), (0, 1));
    }
}
