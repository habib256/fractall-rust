//! Le sage décide du **mode de navigation** de la GUI : zoom CONTINU à la
//! XaoS quand le moteur suit, navigation PAS À PAS (clic = ×2, sélection
//! rectangle) quand il ne suit plus — et retour au continu dès que la vue
//! redevient rapide. Logique PURE (aucune horloge, aucune dépendance egui) :
//! la GUI possède un [`NavModeState`] et le nourrit après chaque rendu.
//!
//! # Pourquoi
//!
//! Le zoom continu (`gui::nav`) n'a de sens que si une image de navigation
//! tient dans un budget de l'ordre du quart de seconde : au-delà, la
//! résolution dynamique plafonne à 1/[`DEFAULT_MAX_DIVISOR`] de côté et la
//! vitesse de croisière s'effondre (`gui::nav::cruise_rate` borne l'avance à
//! ×1,35 par image) — un débutant voit un diaporama flou qui « ne zoome
//! pas ». Le mode pas à pas, lui, ne coûte qu'UN rendu par clic.
//!
//! # Quelle mesure la GUI doit fournir
//!
//! [`FrameCost`] décrit UN rendu terminé (dernière passe), tel que la GUI le
//! mesure déjà pour `gui::nav::nav_divisor` :
//! - `elapsed_secs` : durée murale du rendu (`last_render_time`) ;
//! - `divisor` : diviseur de résolution de ce rendu (`last_render_divisor`,
//!   1 = pleine résolution) — le coût varie en `d²` ;
//! - `fresh_fraction` : fraction des pixels de la passe réellement CALCULÉS,
//!   soit `1 − XaosMap::reused_fraction(w, h)` (1.0 sans map XaoS). Pendant
//!   un zoom continu la plupart des pixels sont COPIÉS de l'image précédente :
//!   la durée d'une image écho n'est pas celle d'une image fraîche. Le coût
//!   plein cadre équivalent est extrapolé par `elapsed·d² / fresh`. Une image
//!   dont moins de [`MIN_FRESH_FRACTION`] des pixels ont été calculés est
//!   IGNORÉE (les frais fixes — orbite, BLA, colorisation — y domineraient et
//!   l'extrapolation exploserait), de même qu'une image annulée.
//!
//! Il faut donc appeler [`NavModeState::observe_frame`] pour TOUT rendu CPU
//! terminé (navigation, stabilisation, raffinement idle, rendu classique) —
//! les images trop réutilisées sont filtrées ici. Les rendus GPU peuvent être
//! fournis aussi (leur coût est celui que l'utilisateur subit).
//!
//! # Décision
//!
//! Grandeur jugée : le coût projeté d'une image de navigation à la résolution
//! dynamique la plus basse, `plein_cadre / max_divisor²` (hypothèse
//! pessimiste : aucune réutilisation XaoS).
//! - passage en pas à pas : projeté > [`STEPPED_ENTER_SECS`] sur
//!   [`STREAK`] mesures consécutives (une seule si > [`IMMEDIATE_FACTOR`] ×
//!   le seuil : inutile d'infliger deux images de 10 s) ;
//! - retour au continu : projeté < [`STEPPED_LEAVE_SECS`] sur [`STREAK`]
//!   mesures consécutives ;
//! - entre les deux seuils, rien ne change et les compteurs repartent à zéro
//!   (bande morte : un coût limite ne fait jamais clignoter le mode).
//!
//! Préférence forcée ([`NavPreference::Continuous`] / `Stepped`) : toujours
//! respectée. `Continuous` forcé sur une vue lente renvoie néanmoins UN
//! message consultatif (une fois par épisode lent).

/// Préférence de l'utilisateur (état GUI, jamais sérialisé dans les params).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavPreference {
    /// Le sage choisit selon le coût mesuré de la vue (défaut).
    #[default]
    Auto,
    /// Zoom continu XaoS forcé.
    Continuous,
    /// Navigation pas à pas forcée.
    Stepped,
}

/// Mode effectif appliqué par la GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    /// Bouton maintenu = zoom continu ancré (gauche avant, droit arrière).
    Continuous,
    /// Clic = zoom ×2, glisser = sélection rectangle.
    Stepped,
}

/// Mesure d'un rendu terminé, cf. l'en-tête du module pour la sémantique.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameCost {
    /// Durée murale du rendu (s).
    pub elapsed_secs: f64,
    /// Diviseur de résolution de ce rendu (1 = pleine résolution).
    pub divisor: u8,
    /// Fraction de pixels calculés (1.0 sans réutilisation XaoS).
    pub fresh_fraction: f64,
}

impl FrameCost {
    /// Rendu plein cadre sans réutilisation.
    pub fn full(elapsed_secs: f64) -> Self {
        Self {
            elapsed_secs,
            divisor: 1,
            fresh_fraction: 1.0,
        }
    }

    /// Coût plein cadre frais équivalent (s), `None` si la mesure n'est pas
    /// exploitable (trop de pixels réutilisés, durée non finie).
    pub fn full_frame_secs(&self) -> Option<f64> {
        if !self.elapsed_secs.is_finite()
            || self.elapsed_secs < 0.0
            || !self.fresh_fraction.is_finite()
            || self.fresh_fraction < MIN_FRESH_FRACTION
        {
            return None;
        }
        let d = self.divisor.max(1) as f64;
        Some(self.elapsed_secs * d * d / self.fresh_fraction.min(1.0))
    }
}

/// Décision renvoyée à chaque appel.
#[derive(Debug, Clone, PartialEq)]
pub struct NavModeDecision {
    /// Mode à appliquer.
    pub mode: NavMode,
    /// Le mode vient de changer (la GUI bascule son comportement souris).
    pub changed: bool,
    /// Message court (anglais) pour la barre d'état, `Some` lors d'une
    /// bascule ou d'un avertissement consultatif ; `None` sinon.
    pub message: Option<String>,
}

/// Budget d'une image de navigation (s) — miroir de
/// `gui::nav::FRAME_TARGET_SECS` (verrouillé par test). La résolution
/// dynamique vise ce temps par image.
pub const NAV_FRAME_TARGET_SECS: f64 = 0.25;

/// Diviseur de résolution maximal en navigation — miroir de
/// `gui::nav::MAX_DIVISOR` (1/4 de côté = 1/16 des pixels).
pub const DEFAULT_MAX_DIVISOR: u8 = 4;

/// Seuil d'ENTRÉE en pas à pas : image de navigation projetée (au diviseur
/// maximal) plus lente que 0,75 s, soit 3 × le budget de 0,25 s.
///
/// Why 3× : jusqu'à ~0,74 s/image, `gui::nav::cruise_rate` garde une vitesse
/// ≥ 1,35^(1/0,75) ≈ 1,49×/s (92 % de la croisière nominale 1,624×/s en
/// échelle log) — le zoom reste fluide grâce au warp. Au-delà la vitesse
/// tombe vite (1,35×/s à 1 s, 1,16×/s à 2 s) et l'image ne se précise plus.
pub const STEPPED_ENTER_SECS: f64 = 3.0 * NAV_FRAME_TARGET_SECS;

/// Seuil de SORTIE du pas à pas : image projetée < 0,3 s, à peine au-dessus
/// du budget — pleine vitesse de croisière garantie. L'écart 0,3 ↔ 0,75 s
/// (×2,5) absorbe le bruit de mesure (±30 % d'une image à l'autre).
pub const STEPPED_LEAVE_SECS: f64 = 1.2 * NAV_FRAME_TARGET_SECS;

/// Nombre de mesures consécutives du même côté d'un seuil pour basculer.
pub const STREAK: u32 = 2;

/// Une mesure au-delà de `IMMEDIATE_FACTOR × STEPPED_ENTER_SECS` bascule
/// immédiatement en pas à pas (pas de seconde image interminable).
pub const IMMEDIATE_FACTOR: f64 = 4.0;

/// Une image dont moins de cette fraction de pixels a été calculée n'est pas
/// une mesure (frais fixes amplifiés au-delà de ×4 par l'extrapolation).
pub const MIN_FRESH_FRACTION: f64 = 0.25;

/// Coût PRÉDIT (avant toute mesure) au-delà duquel on démarre en pas à pas :
/// une prédiction depuis le débit benché est grossière, on exige une marge
/// ×2 sur le seuil d'entrée.
pub const PREDICTION_MARGIN: f64 = 2.0;

/// État possédé par la GUI.
#[derive(Debug, Clone)]
pub struct NavModeState {
    preference: NavPreference,
    /// Mode choisi par le sage en `Auto` (conservé même sous préférence forcée
    /// pour revenir sans à-coup à `Auto`).
    auto_mode: NavMode,
    max_divisor: u8,
    slow_streak: u32,
    fast_streak: u32,
    /// Dernier coût plein cadre exploitable (s).
    last_full_secs: Option<f64>,
    /// Une mesure réelle a été reçue depuis le dernier `reset_view`.
    measured: bool,
    /// Avertissement « continu forcé sur vue lente » déjà émis pour l'épisode.
    advised: bool,
}

impl Default for NavModeState {
    fn default() -> Self {
        Self::new(NavPreference::Auto)
    }
}

impl NavModeState {
    /// État initial : continu (aucune mesure), diviseur max par défaut.
    pub fn new(preference: NavPreference) -> Self {
        Self {
            preference,
            auto_mode: NavMode::Continuous,
            max_divisor: DEFAULT_MAX_DIVISOR,
            slow_streak: 0,
            fast_streak: 0,
            last_full_secs: None,
            measured: false,
            advised: false,
        }
    }

    /// Diviseur de résolution maximal que la navigation peut employer
    /// (`gui::nav::MAX_DIVISOR` aujourd'hui).
    pub fn with_max_divisor(mut self, max_divisor: u8) -> Self {
        self.max_divisor = max_divisor.max(1);
        self
    }

    pub fn preference(&self) -> NavPreference {
        self.preference
    }

    /// Mode effectif courant.
    pub fn mode(&self) -> NavMode {
        match self.preference {
            NavPreference::Auto => self.auto_mode,
            NavPreference::Continuous => NavMode::Continuous,
            NavPreference::Stepped => NavMode::Stepped,
        }
    }

    /// Dernier coût plein cadre équivalent retenu (s), mesuré ou prédit.
    pub fn last_full_frame_secs(&self) -> Option<f64> {
        self.last_full_secs
    }

    /// Coût projeté d'une image de navigation au diviseur maximal (s).
    fn projected(&self, full_secs: f64) -> f64 {
        let d = self.max_divisor.max(1) as f64;
        full_secs / (d * d)
    }

    /// Change la préférence. Renvoie `changed` si le mode EFFECTIF change.
    pub fn set_preference(&mut self, preference: NavPreference) -> NavModeDecision {
        let before = self.mode();
        self.preference = preference;
        self.advised = false;
        let after = self.mode();
        let mut message = (before != after).then(|| self.switch_message(after));
        if after == NavMode::Continuous && self.is_slow_now() {
            message = Some(self.advisory_message());
            self.advised = true;
        }
        NavModeDecision {
            mode: after,
            changed: before != after,
            message,
        }
    }

    /// Oublie les mesures (nouveau type de fractale, nouvelle formule, saut
    /// de vue par chargement d'un PNG…) : le coût mesuré ne vaut plus. Le mode
    /// courant est CONSERVÉ jusqu'à la prochaine mesure (pas de bascule
    /// arbitraire), la prochaine mesure ou prédiction décide.
    pub fn reset_view(&mut self) {
        self.slow_streak = 0;
        self.fast_streak = 0;
        self.last_full_secs = None;
        self.measured = false;
    }

    /// Coût PRÉDIT d'une image plein cadre (s), à fournir avant le premier
    /// rendu d'une vue (démarrage, `reset_view`) — par exemple
    /// `pixels × itérations_moyennes / WisdomPlan::bench_iters_per_sec`.
    /// Sans effet dès qu'une mesure réelle existe. Ne bascule en pas à pas
    /// que si la prédiction dépasse le seuil d'entrée avec une marge
    /// [`PREDICTION_MARGIN`], et ne ramène jamais au continu (seule une
    /// mesure le fait).
    pub fn predict(&mut self, predicted_full_frame_secs: f64) -> NavModeDecision {
        if self.measured
            || !predicted_full_frame_secs.is_finite()
            || predicted_full_frame_secs < 0.0
        {
            return self.unchanged();
        }
        self.last_full_secs = Some(predicted_full_frame_secs);
        let p = self.projected(predicted_full_frame_secs);
        if p > STEPPED_ENTER_SECS * PREDICTION_MARGIN && self.auto_mode == NavMode::Continuous {
            return self.switch_auto(NavMode::Stepped);
        }
        self.unchanged()
    }

    /// Nourrit le sage avec un rendu TERMINÉ (cf. en-tête du module).
    pub fn observe_frame(&mut self, cost: FrameCost) -> NavModeDecision {
        let Some(full) = cost.full_frame_secs() else {
            return self.unchanged();
        };
        self.measured = true;
        self.last_full_secs = Some(full);
        let p = self.projected(full);

        if p > STEPPED_ENTER_SECS {
            self.fast_streak = 0;
            self.slow_streak = self.slow_streak.saturating_add(1);
        } else if p < STEPPED_LEAVE_SECS {
            self.slow_streak = 0;
            self.fast_streak = self.fast_streak.saturating_add(1);
            self.advised = false;
        } else {
            // Bande morte : ni lent ni rapide, rien ne bascule.
            self.slow_streak = 0;
            self.fast_streak = 0;
        }

        match self.auto_mode {
            NavMode::Continuous
                if self.slow_streak >= STREAK
                    || (self.slow_streak >= 1 && p > STEPPED_ENTER_SECS * IMMEDIATE_FACTOR) =>
            {
                self.switch_auto(NavMode::Stepped)
            }
            NavMode::Stepped if self.fast_streak >= STREAK => self.switch_auto(NavMode::Continuous),
            _ => {
                // Continu forcé sur vue lente : avertir une fois.
                if self.preference == NavPreference::Continuous
                    && self.slow_streak >= 1
                    && !self.advised
                {
                    self.advised = true;
                    return NavModeDecision {
                        mode: NavMode::Continuous,
                        changed: false,
                        message: Some(self.advisory_message()),
                    };
                }
                self.unchanged()
            }
        }
    }

    fn is_slow_now(&self) -> bool {
        self.last_full_secs
            .is_some_and(|f| self.projected(f) > STEPPED_ENTER_SECS)
    }

    fn unchanged(&self) -> NavModeDecision {
        NavModeDecision {
            mode: self.mode(),
            changed: false,
            message: None,
        }
    }

    /// Bascule le mode Auto ; n'est visible (changed + message) que si la
    /// préférence est Auto.
    fn switch_auto(&mut self, to: NavMode) -> NavModeDecision {
        self.auto_mode = to;
        self.slow_streak = 0;
        self.fast_streak = 0;
        if self.preference != NavPreference::Auto {
            if self.preference == NavPreference::Continuous && to == NavMode::Stepped && !self.advised {
                self.advised = true;
                return NavModeDecision {
                    mode: NavMode::Continuous,
                    changed: false,
                    message: Some(self.advisory_message()),
                };
            }
            return self.unchanged();
        }
        NavModeDecision {
            mode: to,
            changed: true,
            message: Some(self.switch_message(to)),
        }
    }

    fn switch_message(&self, to: NavMode) -> String {
        match to {
            NavMode::Stepped => match self.last_full_secs {
                Some(f) if self.preference == NavPreference::Auto => format!(
                    "Step navigation: frames take ~{} here (continuous zoom needs < {}). \
                     Click = zoom x2, drag = select area.",
                    fmt_secs(f),
                    fmt_secs(STEPPED_ENTER_SECS * f64::from(self.max_divisor).powi(2)),
                ),
                _ => "Step navigation: click = zoom x2, drag = select area.".to_string(),
            },
            NavMode::Continuous => "Continuous navigation is back: hold the left button to \
                                    zoom in, right to zoom out."
                .to_string(),
        }
    }

    fn advisory_message(&self) -> String {
        format!(
            "Continuous zoom forced on a slow view (frames ~{}): motion will be jerky. \
             Step navigation is recommended here.",
            fmt_secs(self.last_full_secs.unwrap_or(0.0)),
        )
    }
}

/// Durée lisible : « 0.25 s », « 2.3 s », « 14 s », « 2.5 min ».
pub fn fmt_secs(s: f64) -> String {
    if s >= 120.0 {
        format!("{:.1} min", s / 60.0)
    } else if s >= 10.0 {
        format!("{:.0} s", s)
    } else if s >= 1.0 {
        format!("{:.1} s", s)
    } else {
        format!("{:.2} s", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Coût plein cadre qui projette `p` secondes au diviseur 4.
    fn full_for_projected(p: f64) -> f64 {
        p * 16.0
    }

    #[test]
    fn constants_mirror_gui_nav_budget() {
        assert_eq!(NAV_FRAME_TARGET_SECS, crate::gui::nav::FRAME_TARGET_SECS);
        assert_eq!(DEFAULT_MAX_DIVISOR, crate::gui::nav::MAX_DIVISOR);
        assert!(STEPPED_LEAVE_SECS < STEPPED_ENTER_SECS);
    }

    #[test]
    fn fast_view_stays_continuous() {
        let mut s = NavModeState::default();
        for _ in 0..20 {
            let d = s.observe_frame(FrameCost::full(0.08));
            assert_eq!(d.mode, NavMode::Continuous);
            assert!(!d.changed && d.message.is_none());
        }
    }

    #[test]
    fn no_measurement_yet_is_continuous() {
        let s = NavModeState::default();
        assert_eq!(s.mode(), NavMode::Continuous);
        assert_eq!(s.last_full_frame_secs(), None);
    }

    #[test]
    fn slow_view_switches_once_with_message() {
        let mut s = NavModeState::default();
        let slow = FrameCost::full(full_for_projected(1.5)); // 24 s plein cadre
        let d1 = s.observe_frame(slow);
        assert!(!d1.changed, "une seule mesure lente ne suffit pas");
        let d2 = s.observe_frame(slow);
        assert!(d2.changed);
        assert_eq!(d2.mode, NavMode::Stepped);
        let msg = d2.message.unwrap();
        assert!(msg.starts_with("Step navigation: frames take ~24 s"), "{msg}");
        assert!(msg.contains("Click = zoom x2"));
        for _ in 0..5 {
            let d = s.observe_frame(slow);
            assert!(!d.changed && d.message.is_none());
            assert_eq!(d.mode, NavMode::Stepped);
        }
    }

    #[test]
    fn very_slow_frame_switches_immediately() {
        let mut s = NavModeState::default();
        let d = s.observe_frame(FrameCost::full(full_for_projected(STEPPED_ENTER_SECS * 5.0)));
        assert!(d.changed);
        assert_eq!(d.mode, NavMode::Stepped);
    }

    #[test]
    fn borderline_cost_does_not_flap() {
        let mut s = NavModeState::default();
        // Alternance autour du seuil d'entrée, et coûts dans la bande morte.
        let costs = [0.7, 0.8, 0.7, 0.8, 0.5, 0.8, 0.4, 0.9, 0.31, 0.74];
        for c in costs.iter().cycle().take(60) {
            let d = s.observe_frame(FrameCost::full(full_for_projected(*c)));
            assert!(!d.changed, "bascule sur coût limite {c}");
        }
        // Côté pas à pas : même chose autour du seuil de sortie.
        let mut s = NavModeState::default();
        s.observe_frame(FrameCost::full(full_for_projected(10.0)));
        assert_eq!(s.mode(), NavMode::Stepped);
        let costs = [0.29, 0.35, 0.28, 0.5, 0.2, 0.31];
        for c in costs.iter().cycle().take(60) {
            let d = s.observe_frame(FrameCost::full(full_for_projected(*c)));
            assert!(!d.changed, "bascule sur coût limite {c}");
        }
    }

    #[test]
    fn recovery_to_continuous() {
        let mut s = NavModeState::default();
        s.observe_frame(FrameCost::full(full_for_projected(10.0)));
        assert_eq!(s.mode(), NavMode::Stepped);
        let fast = FrameCost::full(0.5);
        assert!(!s.observe_frame(fast).changed);
        let d = s.observe_frame(fast);
        assert!(d.changed);
        assert_eq!(d.mode, NavMode::Continuous);
        assert!(d.message.unwrap().starts_with("Continuous navigation is back"));
    }

    #[test]
    fn echo_frames_are_extrapolated_or_ignored() {
        // 50 % de pixels frais, diviseur 4 : 0,5 s mesuré → 16 s plein cadre.
        let c = FrameCost {
            elapsed_secs: 0.5,
            divisor: 4,
            fresh_fraction: 0.5,
        };
        assert_eq!(c.full_frame_secs(), Some(16.0));
        // Image presque entièrement copiée : pas une mesure.
        let echo = FrameCost {
            elapsed_secs: 0.01,
            divisor: 1,
            fresh_fraction: 0.05,
        };
        assert_eq!(echo.full_frame_secs(), None);
        let mut s = NavModeState::default();
        s.observe_frame(FrameCost::full(full_for_projected(10.0)));
        assert_eq!(s.mode(), NavMode::Stepped);
        for _ in 0..10 {
            assert!(!s.observe_frame(echo).changed, "un écho ne ramène pas au continu");
        }
    }

    #[test]
    fn forced_modes_are_respected() {
        let mut s = NavModeState::new(NavPreference::Stepped);
        assert_eq!(s.mode(), NavMode::Stepped);
        for _ in 0..5 {
            let d = s.observe_frame(FrameCost::full(0.01));
            assert_eq!(d.mode, NavMode::Stepped);
            assert!(!d.changed && d.message.is_none());
        }

        let mut s = NavModeState::new(NavPreference::Continuous);
        let slow = FrameCost::full(full_for_projected(2.0));
        let d1 = s.observe_frame(slow);
        assert_eq!(d1.mode, NavMode::Continuous);
        assert!(!d1.changed);
        let advisory = d1.message.expect("avertissement consultatif");
        assert!(advisory.contains("forced"), "{advisory}");
        // Une seule fois par épisode lent.
        for _ in 0..5 {
            let d = s.observe_frame(slow);
            assert_eq!(d.mode, NavMode::Continuous);
            assert!(d.message.is_none());
        }
        // Retour en Auto sur une vue lente : bascule immédiate en pas à pas.
        let d = s.set_preference(NavPreference::Auto);
        assert_eq!(d.mode, NavMode::Stepped);
        assert!(d.changed);
        assert!(d.message.unwrap().starts_with("Step navigation"));
    }

    #[test]
    fn prediction_only_before_measurement() {
        let mut s = NavModeState::default();
        // Prédiction modérément lente : marge exigée, reste continu.
        assert!(!s.predict(full_for_projected(1.0)).changed);
        // Prédiction franchement lente : démarre en pas à pas.
        let d = s.predict(full_for_projected(5.0));
        assert!(d.changed);
        assert_eq!(d.mode, NavMode::Stepped);
        // Une prédiction rapide ne ramène jamais au continu…
        assert!(!s.predict(0.01).changed);
        // … une mesure si.
        s.observe_frame(FrameCost::full(0.1));
        assert!(s.observe_frame(FrameCost::full(0.1)).changed);
        // Après une mesure, les prédictions sont ignorées.
        assert!(!s.predict(full_for_projected(100.0)).changed);
        assert_eq!(s.mode(), NavMode::Continuous);
        // reset_view rouvre la porte aux prédictions sans changer le mode.
        s.reset_view();
        assert_eq!(s.mode(), NavMode::Continuous);
        assert!(s.predict(full_for_projected(100.0)).changed);
    }
}
