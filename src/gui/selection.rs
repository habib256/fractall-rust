use eframe::egui::{Pos2, Rect};

/// État transitoire d'une sélection rectangulaire dans l'image.
///
/// La géométrie du zoom reste dans le contrôleur GUI ; ce type ne gère que le
/// cycle de vie du geste afin qu'une annulation ou une fin de sélection ne
/// laisse jamais de coordonnées périmées.
#[derive(Debug, Default)]
pub struct SelectionState {
    start: Option<Pos2>,
    current: Option<Pos2>,
}

impl SelectionState {
    pub fn is_active(&self) -> bool {
        self.start.is_some()
    }

    pub fn begin(&mut self, position: Pos2) {
        self.start = Some(position);
        self.current = Some(position);
    }

    pub fn update(&mut self, position: Pos2) {
        if self.is_active() {
            self.current = Some(position);
        }
    }

    pub fn rect(&self) -> Option<Rect> {
        Some(Rect::from_two_pos(self.start?, self.current?))
    }

    /// Termine le geste et restitue son rectangle, quelle que soit sa taille.
    pub fn finish(&mut self) -> Option<Rect> {
        let rect = self.rect();
        self.cancel();
        rect
    }

    pub fn cancel(&mut self) {
        self.start = None;
        self.current = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_lifecycle_normalizes_rectangle_and_clears_on_finish() {
        let mut selection = SelectionState::default();
        selection.begin(Pos2::new(8.0, 9.0));
        selection.update(Pos2::new(2.0, 3.0));

        assert!(selection.is_active());
        let rect = selection.finish().unwrap();
        assert_eq!(rect.min, Pos2::new(2.0, 3.0));
        assert_eq!(rect.max, Pos2::new(8.0, 9.0));
        assert!(!selection.is_active());
        assert!(selection.rect().is_none());
    }

    #[test]
    fn update_without_active_selection_is_ignored() {
        let mut selection = SelectionState::default();
        selection.update(Pos2::new(4.0, 5.0));

        assert!(!selection.is_active());
        assert!(selection.finish().is_none());
    }

    #[test]
    fn cancel_discards_both_endpoints() {
        let mut selection = SelectionState::default();
        selection.begin(Pos2::new(1.0, 2.0));
        selection.update(Pos2::new(3.0, 4.0));
        selection.cancel();

        assert!(!selection.is_active());
        assert!(selection.rect().is_none());
    }
}
