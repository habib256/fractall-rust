//! Versioning pur des travaux GUI asynchrones.
//!
//! Chaque nouvelle requête invalide toutes les réponses précédentes. Les
//! workers ne transportent qu'un `u64`; la GUI accepte uniquement la version
//! courante. Le compteur saute zéro au wrap pour conserver `0` comme état
//! « aucune requête émise » dans les diagnostics.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AsyncVersion {
    current: u64,
}

impl AsyncVersion {
    pub fn issue(&mut self) -> u64 {
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            self.current = 1;
        }
        self.current
    }

    pub fn accepts(self, version: u64) -> bool {
        version != 0 && version == self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_request_invalidates_every_older_reply() {
        let mut gate = AsyncVersion::default();
        let first = gate.issue();
        let second = gate.issue();
        assert!(!gate.accepts(first));
        assert!(gate.accepts(second));
        assert!(!gate.accepts(0));
    }

    #[test]
    fn wrap_never_issues_reserved_zero() {
        let mut gate = AsyncVersion { current: u64::MAX };
        assert_eq!(gate.issue(), 1);
        assert!(gate.accepts(1));
    }
}
