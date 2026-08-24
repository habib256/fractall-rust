//! État provisoire pur du scrub de timeline.

use crate::gui::async_version::AsyncVersion;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScrubState {
    version: AsyncVersion,
    active: bool,
    position: f64,
}

impl ScrubState {
    pub fn start(&mut self, position: f64) -> u64 {
        self.active = true;
        self.position = position;
        self.version.issue()
    }

    pub fn close(&mut self) {
        self.active = false;
    }

    pub fn accepts(self, version: u64) -> bool {
        self.active && self.version.accepts(version)
    }

    pub fn active(self) -> bool {
        self.active
    }

    pub fn position(self) -> f64 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragging_invalidates_previous_reply_and_tracks_position() {
        let mut s = ScrubState::default();
        let old = s.start(1.25);
        let current = s.start(2.5);
        assert!(!s.accepts(old));
        assert!(s.accepts(current));
        assert_eq!(s.position(), 2.5);
    }

    #[test]
    fn closing_rejects_even_current_inflight_reply() {
        let mut s = ScrubState::default();
        let version = s.start(3.0);
        s.close();
        assert!(!s.active());
        assert!(!s.accepts(version));
    }
}
