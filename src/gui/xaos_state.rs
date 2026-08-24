//! Cycle de vie pur de la frame source XaoS et du raffinement différé.

use crate::fractal::xaos::XaosSourceFrame;

#[derive(Clone, Default)]
pub struct XaosLifecycle {
    source: Option<XaosSourceFrame>,
    refine_pending: bool,
}

impl XaosLifecycle {
    /// Un nouveau rendu supplante le raffinement programmé, sans jeter la
    /// source qui peut encore accélérer ce rendu.
    pub fn begin_render(&mut self) {
        self.refine_pending = false;
    }

    /// L'AA exige des échantillons exacts et ne doit jamais réutiliser XaoS.
    pub fn source_for_render(&self, aa_samples: u32) -> Option<XaosSourceFrame> {
        (aa_samples <= 1).then(|| self.source.clone()).flatten()
    }

    /// Accepte une nouvelle source seulement si elle n'est pas moins résolue
    /// que la meilleure source courante. `None` (GPU/écho pur) la conserve.
    pub fn accept_pass(&mut self, frame: Option<XaosSourceFrame>, approximate: bool) {
        if let Some(frame) = frame {
            let new_pixels = frame.width as u64 * frame.height as u64;
            let keep_existing = self
                .source
                .as_ref()
                .is_some_and(|old| old.width as u64 * old.height as u64 > new_pixels);
            if !keep_existing {
                self.source = Some(frame);
            }
        }
        self.refine_pending = approximate;
    }

    pub fn refine_pending(&self) -> bool {
        self.refine_pending
    }

    pub fn take_refine(&mut self) -> bool {
        std::mem::take(&mut self.refine_pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use std::sync::Arc;

    fn frame(w: u32, h: u32) -> XaosSourceFrame {
        XaosSourceFrame::exact(
            Arc::new(vec![0; (w * h) as usize]),
            Arc::new(vec![Complex64::new(0.0, 0.0); (w * h) as usize]),
            w,
            h,
            ("0".into(), "0".into(), "4".into(), "4".into()),
            "test".into(),
        )
    }

    #[test]
    fn coarse_late_pass_never_replaces_full_source() {
        let mut s = XaosLifecycle::default();
        s.accept_pass(Some(frame(800, 600)), false);
        s.accept_pass(Some(frame(200, 150)), true);
        let source = s.source_for_render(1).unwrap();
        assert_eq!((source.width, source.height), (800, 600));
        assert!(s.refine_pending());
    }

    #[test]
    fn aa_disables_reuse_and_new_render_cancels_pending_refine() {
        let mut s = XaosLifecycle::default();
        s.accept_pass(Some(frame(80, 60)), true);
        assert!(s.source_for_render(1).is_some());
        assert!(s.source_for_render(2).is_none());
        s.begin_render();
        assert!(!s.refine_pending());
        assert!(!s.take_refine());
    }
}
