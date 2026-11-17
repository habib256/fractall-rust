//! Branchement GUI des décisions du sage (`fractal::wisdom`) : mode de
//! navigation (`wisdom_nav`) et itérations automatiques (`wisdom_iter`).
//!
//! Logique PURE (aucune dépendance egui) : quelles mesures nourrir, quand une
//! passe est éligible à l'évaluation des itérations, garde-fou anti-boucle des
//! re-rendus automatiques, bascule différée du mode souris, avis de la barre
//! d'état. `gui/app.rs` ne fait qu'exécuter ces décisions — les clics ne sont
//! pas automatisables, d'où l'extraction testée ici.

use std::time::{Duration, Instant};

use crate::fractal::wisdom::{
    assess_iterations, FrameCost, IterationStats, IterationVerdict, NavMode,
};

/// Durée d'affichage d'un avis du sage dans la barre d'état.
pub const NOTICE_DURATION: Duration = Duration::from_secs(6);

/// Nombre maximal de re-rendus automatiques CONSÉCUTIFS (croissance des
/// itérations) pour une même vue : au-delà, la vue doit changer avant qu'une
/// nouvelle croissance soit permise (anti-boucle).
pub const MAX_CONSECUTIVE_AUTO_GROWS: u32 = 3;

/// Description de la passe FINALE d'un rendu, construite par le thread de
/// rendu (là où le map XaoS est connu) et transmise avec `PassComplete`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinalPassInfo {
    /// Diviseur de résolution de la passe (1 = pleine résolution).
    pub divisor: u8,
    /// Passe rendue sur GPU.
    pub from_gpu: bool,
    /// Fraction des pixels de la passe réellement calculés
    /// (`1 − XaosMap::reused_fraction`, 1.0 sans map ou sur GPU).
    pub fresh_fraction: f64,
    /// Des pixels copiés dévient de la grille (> ε) : le buffer n'est pas
    /// exact (écho XaoS approximé).
    pub approx_copies: bool,
    /// `iteration_max` auquel la passe a été rendue.
    pub iteration_cap: u32,
}

/// Mesure à fournir au sage de navigation pour un rendu terminé, `None` pour
/// un écho pur (rien calculé). `elapsed_secs` = durée murale du rendu entier ;
/// avec l'AA multi-sample, le rendu enchaîne `aa_samples` images plein cadre
/// après les passes : on ramène la durée à UNE image.
pub fn frame_cost(info: &FinalPassInfo, elapsed_secs: f64, aa_samples: u32) -> Option<FrameCost> {
    if info.fresh_fraction <= 0.0 {
        return None;
    }
    let frames = if aa_samples > 1 { aa_samples + 1 } else { 1 };
    Some(FrameCost {
        elapsed_secs: elapsed_secs / frames as f64,
        divisor: info.divisor,
        fresh_fraction: info.fresh_fraction,
    })
}

/// La passe peut-elle juger le plafond d'itérations ? Seulement une passe
/// CPU pleine résolution dont le buffer est EXACT partout (aucun pixel copié
/// approximé) et qui a calculé quelque chose. Les passes intermédiaires ne
/// sont pas finales (jamais transmises) ; le raffinement idle (copies ≤ ε)
/// est éligible ; un écho XaoS approximé ne l'est pas.
pub fn iterations_assessable(info: &FinalPassInfo) -> bool {
    info.divisor == 1 && !info.from_gpu && !info.approx_copies && info.fresh_fraction > 0.0
}

/// Garde-fou anti-boucle des re-rendus automatiques : compte les croissances
/// consécutives pour une même vue (clé opaque fournie par l'appelant).
#[derive(Debug, Clone, Default)]
pub struct AutoGrowGuard {
    view_key: Option<String>,
    grows: u32,
}

impl AutoGrowGuard {
    /// Autorise (et compte) une croissance pour `view_key`. Une vue
    /// différente remet le compteur à zéro.
    pub fn try_grow(&mut self, view_key: &str) -> bool {
        if self.view_key.as_deref() != Some(view_key) {
            self.view_key = Some(view_key.to_string());
            self.grows = 0;
        }
        if self.grows >= MAX_CONSECUTIVE_AUTO_GROWS {
            return false;
        }
        self.grows += 1;
        true
    }

    /// Oublie l'historique (changement de type, réactivation d'Auto…).
    pub fn reset(&mut self) {
        self.view_key = None;
        self.grows = 0;
    }
}

/// Action à exécuter par la GUI après l'évaluation d'une passe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterAction {
    /// Rien à faire.
    Keep,
    /// Relever le plafond et RE-RENDRE la vue.
    GrowAndRender(u32),
    /// Abaisser le plafond pour le rendu SUIVANT (pas de re-rendu).
    ShrinkNext(u32),
}

/// Contexte de l'évaluation, côté GUI.
#[derive(Debug, Clone, Copy)]
pub struct IterContext<'a> {
    /// Case « Auto » cochée.
    pub auto_on: bool,
    /// `wisdom::auto_iterations_applicable(params)`.
    pub applicable: bool,
    /// `params.iteration_max` ACTUEL (un plafond changé depuis le lancement
    /// du rendu rend le buffer obsolète).
    pub current_cap: u32,
    /// Un geste de navigation est en cours (bouton maintenu, inertie) :
    /// jamais de re-rendu automatique pendant le geste.
    pub gesture_active: bool,
    /// Clé de la vue courante (garde-fou anti-boucle).
    pub view_key: &'a str,
}

/// Couche 1 du sage des itérations appliquée à une passe terminée.
pub fn decide_iterations(
    info: &FinalPassInfo,
    iterations: &[u32],
    ctx: IterContext<'_>,
    guard: &mut AutoGrowGuard,
) -> IterAction {
    if !ctx.auto_on
        || !ctx.applicable
        || ctx.gesture_active
        || !iterations_assessable(info)
        || info.iteration_cap != ctx.current_cap
        || iterations.is_empty()
    {
        return IterAction::Keep;
    }
    let stats = IterationStats::from_iterations(iterations, info.iteration_cap);
    match assess_iterations(&stats).verdict {
        IterationVerdict::Sufficient => IterAction::Keep,
        IterationVerdict::Grow(n) if n > ctx.current_cap => {
            if guard.try_grow(ctx.view_key) {
                IterAction::GrowAndRender(n)
            } else {
                IterAction::Keep
            }
        }
        IterationVerdict::Grow(_) => IterAction::Keep,
        IterationVerdict::Shrink(n) if n < ctx.current_cap => IterAction::ShrinkNext(n),
        IterationVerdict::Shrink(_) => IterAction::Keep,
    }
}

/// Plafond abaissé en attente (`IterAction::ShrinkNext`), consommé au
/// lancement d'un rendu : `Some(n)` = à appliquer à ce rendu. L'attente est
/// TOUJOURS consommée — si Auto n'est plus actif ou si `n` ne réduit plus le
/// plafond courant, le verdict est périmé (le rendu suivant réévaluera).
pub fn take_pending_shrink(pending: &mut Option<u32>, auto_active: bool, current_cap: u32) -> Option<u32> {
    pending.take().filter(|&n| auto_active && n < current_cap)
}

/// Bascule du mode souris : le mode décidé (`wanted`) ne remplace le mode
/// appliqué qu'en dehors d'un geste (bouton maintenu, sélection, inertie) —
/// un changement en plein geste attend le relâchement. `Some(mode)` = le mode
/// à appliquer maintenant.
pub fn nav_switch_now(applied: NavMode, wanted: NavMode, gesture_in_progress: bool) -> Option<NavMode> {
    (applied != wanted && !gesture_in_progress).then_some(wanted)
}

/// Avis temporaire du sage dans la barre d'état (un seul à la fois : le
/// suivant remplace le précédent).
#[derive(Debug, Clone, Default)]
pub struct SageNotice {
    current: Option<(String, Instant)>,
}

impl SageNotice {
    pub fn post(&mut self, text: impl Into<String>, now: Instant) {
        self.current = Some((text.into(), now));
    }

    /// Texte visible à `now` et temps restant avant expiration ; expire et
    /// oublie l'avis après [`NOTICE_DURATION`].
    pub fn visible(&mut self, now: Instant) -> Option<(&str, Duration)> {
        let expired = self
            .current
            .as_ref()
            .is_some_and(|(_, at)| now.saturating_duration_since(*at) >= NOTICE_DURATION);
        if expired {
            self.current = None;
        }
        self.current
            .as_ref()
            .map(|(t, at)| (t.as_str(), NOTICE_DURATION - now.saturating_duration_since(*at)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_cpu(cap: u32) -> FinalPassInfo {
        FinalPassInfo {
            divisor: 1,
            from_gpu: false,
            fresh_fraction: 1.0,
            approx_copies: false,
            iteration_cap: cap,
        }
    }

    fn ctx(cap: u32) -> IterContext<'static> {
        IterContext {
            auto_on: true,
            applicable: true,
            current_cap: cap,
            gesture_active: false,
            view_key: "view",
        }
    }

    /// Buffer dont la queue d'évasion bute sur le plafond : 30 % des pixels
    /// plafonnés, évasions étalées uniformément jusqu'au plafond (queue qui
    /// ne décroît pas) ⇒ verdict Grow.
    fn truncated_tail(cap: u32) -> Vec<u32> {
        let mut v: Vec<u32> = (0..7000).map(|i| 1 + (i as u64 * (cap as u64 - 2) / 7000) as u32).collect();
        v.extend(std::iter::repeat_n(cap, 3000));
        v
    }

    /// Buffer dont toutes les évasions sont précoces (< 50) avec un plafond
    /// de 100 000 ⇒ verdict Shrink.
    fn early_escapes() -> Vec<u32> {
        let mut v: Vec<u32> = (0..9000).map(|i| 5 + (i % 40) as u32).collect();
        v.extend(std::iter::repeat_n(100_000, 1000));
        v
    }

    #[test]
    fn echo_and_coarse_passes_are_not_assessable() {
        assert!(iterations_assessable(&full_cpu(1000)));
        let mut p = full_cpu(1000);
        p.approx_copies = true;
        assert!(!iterations_assessable(&p), "approximate XaoS echo");
        let mut p = full_cpu(1000);
        p.divisor = 2;
        assert!(!iterations_assessable(&p), "dynamic-resolution nav frame");
        let mut p = full_cpu(1000);
        p.from_gpu = true;
        assert!(!iterations_assessable(&p), "GPU pass");
        let mut p = full_cpu(1000);
        p.fresh_fraction = 0.0;
        assert!(!iterations_assessable(&p), "pure echo");
        // Raffinement exact : peu de pixels recalculés, buffer exact.
        let mut p = full_cpu(1000);
        p.fresh_fraction = 0.05;
        assert!(iterations_assessable(&p));
    }

    #[test]
    fn frame_cost_skips_pure_echo_and_normalises_aa() {
        let mut p = full_cpu(1000);
        p.fresh_fraction = 0.0;
        assert!(frame_cost(&p, 1.0, 1).is_none());
        let p = FinalPassInfo { divisor: 2, fresh_fraction: 0.5, ..full_cpu(1000) };
        let c = frame_cost(&p, 0.3, 1).unwrap();
        assert_eq!((c.elapsed_secs, c.divisor, c.fresh_fraction), (0.3, 2, 0.5));
        let c = frame_cost(&full_cpu(1000), 5.0, 4).unwrap();
        assert!((c.elapsed_secs - 1.0).abs() < 1e-12);
    }

    #[test]
    fn grow_is_capped_per_view_and_resets_on_view_change() {
        let mut g = AutoGrowGuard::default();
        for _ in 0..MAX_CONSECUTIVE_AUTO_GROWS {
            assert!(g.try_grow("a"));
        }
        assert!(!g.try_grow("a"), "loop guard");
        assert!(g.try_grow("b"), "new view resets the counter");
        g.reset();
        assert!(g.try_grow("b"));
    }

    #[test]
    fn decide_grows_then_stops_after_guard() {
        let cap = 1000;
        let buf = truncated_tail(cap);
        let mut g = AutoGrowGuard::default();
        let first = decide_iterations(&full_cpu(cap), &buf, ctx(cap), &mut g);
        let IterAction::GrowAndRender(n) = first else {
            panic!("expected a grow, got {first:?}");
        };
        assert!(n > cap);
        // Même buffer (simulé) re-présenté : au plus MAX croissances.
        let mut grows = 1;
        for _ in 0..10 {
            if let IterAction::GrowAndRender(_) =
                decide_iterations(&full_cpu(cap), &buf, ctx(cap), &mut g)
            {
                grows += 1;
            }
        }
        assert_eq!(grows, MAX_CONSECUTIVE_AUTO_GROWS);
    }

    #[test]
    fn decide_respects_manual_gesture_and_stale_buffers() {
        let cap = 1000;
        let buf = truncated_tail(cap);
        let mut g = AutoGrowGuard::default();
        let mut c = ctx(cap);
        c.auto_on = false;
        assert_eq!(decide_iterations(&full_cpu(cap), &buf, c, &mut g), IterAction::Keep);
        let mut c = ctx(cap);
        c.applicable = false;
        assert_eq!(decide_iterations(&full_cpu(cap), &buf, c, &mut g), IterAction::Keep);
        let mut c = ctx(cap);
        c.gesture_active = true;
        assert_eq!(decide_iterations(&full_cpu(cap), &buf, c, &mut g), IterAction::Keep);
        // Plafond modifié depuis le lancement du rendu : buffer obsolète.
        assert_eq!(decide_iterations(&full_cpu(cap), &buf, ctx(2 * cap), &mut g), IterAction::Keep);
        let mut echo = full_cpu(cap);
        echo.approx_copies = true;
        assert_eq!(decide_iterations(&echo, &buf, ctx(cap), &mut g), IterAction::Keep);
    }

    #[test]
    fn decide_shrinks_for_next_render_without_rerender() {
        let cap = 100_000;
        let mut g = AutoGrowGuard::default();
        match decide_iterations(&full_cpu(cap), &early_escapes(), ctx(cap), &mut g) {
            IterAction::ShrinkNext(n) => assert!(n < cap && n >= 100),
            other => panic!("expected a shrink, got {other:?}"),
        }
    }

    #[test]
    fn pending_shrink_is_consumed_once_and_dropped_when_stale() {
        let mut p = Some(500);
        assert_eq!(take_pending_shrink(&mut p, true, 2500), Some(500));
        assert_eq!(p, None);
        assert_eq!(take_pending_shrink(&mut p, true, 2500), None, "consumed");
        let mut p = Some(500);
        assert_eq!(take_pending_shrink(&mut p, false, 2500), None, "auto off");
        assert_eq!(p, None);
        let mut p = Some(500);
        assert_eq!(take_pending_shrink(&mut p, true, 400), None, "cap already lower");
    }

    #[test]
    fn nav_switch_waits_for_gesture_end() {
        use NavMode::*;
        assert_eq!(nav_switch_now(Continuous, Stepped, true), None);
        assert_eq!(nav_switch_now(Continuous, Stepped, false), Some(Stepped));
        assert_eq!(nav_switch_now(Stepped, Stepped, false), None);
    }

    #[test]
    fn notice_expires_after_duration_and_is_replaced() {
        let t0 = Instant::now();
        let mut n = SageNotice::default();
        assert!(n.visible(t0).is_none());
        n.post("hello", t0);
        let (text, left) = n.visible(t0 + Duration::from_secs(1)).unwrap();
        assert_eq!(text, "hello");
        assert_eq!(left, NOTICE_DURATION - Duration::from_secs(1));
        n.post("next", t0 + Duration::from_secs(2));
        assert_eq!(n.visible(t0 + Duration::from_secs(7)).unwrap().0, "next");
        assert!(n.visible(t0 + Duration::from_secs(9)).is_none());
    }
}
