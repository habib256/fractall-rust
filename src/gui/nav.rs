//! Machine à états de la navigation continue (mode XaoS) et décisions de
//! cadence / résolution — **logique pure, testable sans fenêtre**.
//!
//! Why : `gui/app.rs` mêle orchestration du rendu, entrées, menus, arithmétique
//! HP et threading. Toute la logique d'interaction y était noyée et n'était
//! vérifiable qu'à la souris — une rampe de vitesse introduite le 2026-08-09 y
//! a créé un blocage circulaire (tick jamais enregistré ⇒ `dt` toujours nul ⇒
//! vitesse jamais non-nulle) qu'aucun test ne pouvait attraper. Ce module isole
//! ce qui se décide (vitesse, ancrage, quand relancer un rendu, à quelle
//! résolution) de ce qui s'exécute (zoom HP, spawn de thread, peinture).
//!
//! Aucune dépendance à egui ni à `Instant` : le temps entre par `dt`, l'état
//! est explicite, chaque `tick` est une fonction de (état, entrée) → sortie.
//!
//! Références XaoS (sources 3.6) : `ui_helper.cpp` pour la rampe de vitesse
//! (`uih_zoom` / `uih_slowdown`) et `engine/algorithms.md` §Dynamic Resolution
//! pour le compromis résolution/temps.

/// Vitesse de croisière du zoom continu : facteur d'échelle par seconde.
/// Valeur XaoS : `MAXSTEP = 0.024` de span par frame nominale de 1/20 s
/// (`config.h`, `uih_zoomupdate` : `mmul = (1-step)^mul`), soit
/// `(1-0.024)^-20 ≈ 1.62` par seconde.
pub const ZOOM_RATE: f64 = 1.624;

/// Durée de montée en vitesse (s). XaoS accélère par paliers `speedup*2` par
/// frame (`uih_zoom`) : `MAXSTEP / (2·STEP) ≈ 6.7` frames de 1/20 s.
pub const ACCEL_SECS: f64 = 0.33;

/// Durée de décélération après relâchement (s). XaoS décroît deux fois plus
/// lentement qu'il n'accélère (`uih_slowdown` retire `speedup` par frame) :
/// le zoom continue en s'amortissant au lieu de s'arrêter net.
pub const DECEL_SECS: f64 = 0.67;

/// Sous cette vitesse (en `ln(facteur)/s`) la navigation est considérée à l'arrêt.
pub const VEL_EPSILON: f64 = 1e-3;

/// Borne du pas de temps utilisé pour la PHYSIQUE : une frame lente ne doit pas
/// produire un saut de zoom incontrôlable. (Le temps réel, non borné, continue
/// d'alimenter le compteur de cadence de rendu.)
pub const MAX_PHYSICS_DT: f64 = 0.1;

/// Intervalle minimal entre deux rendus pendant la navigation (~30 Hz).
/// Sans ce plancher, une scène peu profonde (rendu ~5 ms) déclencherait des
/// centaines de spawns de thread + uploads de texture par seconde, ce qui
/// affame le thread UI → frames irrégulières = saccades. Le warp assure la
/// continuité visuelle entre deux rendus.
pub const MIN_RENDER_INTERVAL_SECS: f64 = 0.033;

/// Avance de vue MAXIMALE tolérée entre deux images fraîches, en facteur de
/// zoom. La vitesse s'adapte au débit réel du moteur pour tenir cette borne.
///
/// Why : à grande profondeur une image coûte ~1 s ; à pleine vitesse la vue
/// avancerait de ×1.6 entre deux images → la texture warpée est étirée d'autant
/// avant d'être remplacée d'un coup = « pop » périodique. Borner l'écart réduit
/// l'amplitude du pop ET le coût du rendu : la réutilisation pixels inter-frame
/// ne récupère que la partie commune des deux vues.
pub const MAX_STEP_PER_RENDER: f64 = 1.35;

/// Plancher de vitesse : même moteur très lent, la navigation garde une
/// progression perceptible (facteur par seconde).
pub const MIN_ZOOM_RATE: f64 = 1.05;

/// Durée VISÉE d'une image de navigation (s). Au-delà, la résolution dynamique
/// entre en jeu (port de la « dynamic resolution » XaoS : « calculate only the
/// details that can be determined within a time interval »).
pub const FRAME_TARGET_SECS: f64 = 0.25;

/// Diviseur de résolution maximal en navigation (1/4 de côté = 1/16 des pixels).
pub const MAX_DIVISOR: u8 = 4;

/// Vitesse de croisière à viser compte tenu du débit mesuré du moteur.
/// `last_render_secs` = durée du dernier rendu terminé (`None` = inconnu).
///
/// Pleine vitesse tant que le rendu suit ; sinon la vitesse est réduite pour
/// que la vue n'avance pas de plus de [`MAX_STEP_PER_RENDER`] entre deux images.
pub fn cruise_rate(last_render_secs: Option<f64>) -> f64 {
    match last_render_secs {
        Some(t) if t > 0.0 => ZOOM_RATE
            .min(MAX_STEP_PER_RENDER.powf(1.0 / t))
            .max(MIN_ZOOM_RATE),
        _ => ZOOM_RATE,
    }
}

/// Diviseur de résolution pour un rendu de navigation.
///
/// Le coût varie en `d²` ; on extrapole le coût plein cadre depuis la dernière
/// durée mesurée ET son propre diviseur — sans ce second terme, une mesure déjà
/// réduite serait lue comme un coût plein cadre et le diviseur ne
/// redescendrait jamais.
pub fn nav_divisor(last_render_secs: Option<f64>, last_divisor: u8) -> u8 {
    let full_cost = last_render_secs.map(|t| t * (last_divisor.max(1) as f64).powi(2));
    match full_cost {
        Some(c) if c > FRAME_TARGET_SECS => {
            let d = (c / FRAME_TARGET_SECS).sqrt().ceil() as u32;
            (d.next_power_of_two() as u8).min(MAX_DIVISOR)
        }
        _ => 1,
    }
}

/// Rendu à lancer à l'issue d'un tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavRender {
    /// Rien à lancer.
    None,
    /// Image de navigation : passe unique, résolution dynamique, streaming partiel.
    Moving,
    /// Stabilisation (mouvement terminé) : passe unique PLEINE résolution.
    Settle,
}

/// Entrée d'un tick de navigation.
#[derive(Debug, Clone, Copy)]
pub struct NavInput {
    /// +1 = zoom avant, -1 = arrière, 0 = aucun bouton (⇒ décélération).
    /// L'appelant a déjà validé que le geste a commencé sur l'image.
    pub dir: f64,
    /// Temps réel écoulé depuis le tick précédent (s).
    pub dt: f64,
    /// Ancrage normalisé [0,1]² si le curseur est sur l'image ; `None` conserve
    /// le dernier ancrage connu (le zoom reste ancré pendant la décélération,
    /// même curseur sorti).
    pub anchor: Option<(f32, f32)>,
    /// Vitesse de croisière visée (facteur/s), cf. [`cruise_rate`].
    pub cruise: f64,
    /// Un rendu est déjà en cours (on n'en empile pas un second).
    pub rendering: bool,
}

/// Sortie d'un tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavOutcome {
    /// Zoom à appliquer : `(ancrage_x, ancrage_y, facteur)`.
    pub zoom: Option<(f32, f32, f64)>,
    pub render: NavRender,
    /// Demander une nouvelle frame (mouvement en cours).
    pub repaint: bool,
}

impl NavOutcome {
    const IDLE: Self = Self {
        zoom: None,
        render: NavRender::None,
        repaint: false,
    };
}

/// État de la navigation continue.
#[derive(Debug, Clone, Default)]
pub struct NavState {
    /// Vitesse courante en `ln(facteur)` par seconde (signée).
    vel: f64,
    anchor: Option<(f32, f32)>,
    /// Temps écoulé depuis le dernier rendu LANCÉ (s).
    since_render: f64,
    /// Navigation en cours (bouton tenu OU décélération résiduelle).
    active: bool,
}

impl NavState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Vitesse courante en `ln(facteur)/s` (diagnostic et tests).
    pub fn velocity(&self) -> f64 {
        self.vel
    }

    /// Navigation en cours (bouton tenu ou décélération résiduelle).
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Abandonne tout mouvement (sortie du mode, changement de fractale…).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Avance la machine d'un pas de temps.
    pub fn tick(&mut self, input: NavInput) -> NavOutcome {
        if input.anchor.is_some() {
            self.anchor = input.anchor;
        }

        let dt_phys = input.dt.clamp(0.0, MAX_PHYSICS_DT);
        self.since_render += input.dt.max(0.0);

        // Rampe accel/décel (port XaoS `uih_zoom` / `uih_slowdown`) : le bouton
        // pilote une CIBLE de vitesse, pas la vitesse elle-même.
        let cruise = input.cruise.max(1.0).ln();
        let target = input.dir * cruise;
        let ramp_secs = if input.dir != 0.0 {
            ACCEL_SECS
        } else {
            DECEL_SECS
        };
        let max_delta = (cruise / ramp_secs) * dt_phys;
        let dv = (target - self.vel).clamp(-max_delta, max_delta);
        self.vel += dv;
        if input.dir == 0.0 && self.vel.abs() < VEL_EPSILON {
            self.vel = 0.0;
        }

        // ⚠️ « En mouvement » inclut le bouton enfoncé même à vitesse encore
        // nulle. Sans ce `|| dir != 0`, la 1re frame (dt = 0, donc vel reste 0)
        // ne marquerait pas l'état actif, `since_render`/`dt` ne progresseraient
        // pas et la rampe ne démarrerait JAMAIS (régression du 2026-08-09,
        // verrouillée par `ramp_starts_from_rest`).
        let moving = self.vel != 0.0 || input.dir != 0.0;

        if moving {
            if !self.active {
                // Début de geste : le premier rendu ne doit pas attendre le
                // plancher de cadence.
                self.since_render = f64::INFINITY;
                self.active = true;
            }

            let zoom = if dt_phys > 0.0 && self.vel != 0.0 {
                self.anchor
                    .map(|(rx, ry)| (rx, ry, (self.vel * dt_phys).exp()))
            } else {
                None
            };

            // Un rendu ne démarre que si la vue a bougé, qu'aucun rendu n'est en
            // cours et que le plancher de cadence est franchi. Entre deux, le
            // warp transporte la texture vers la vue live.
            let render = if zoom.is_some() && !input.rendering && self.since_render >= MIN_RENDER_INTERVAL_SECS
            {
                self.since_render = 0.0;
                NavRender::Moving
            } else {
                NavRender::None
            };

            NavOutcome {
                zoom,
                render,
                repaint: true,
            }
        } else if self.active {
            // Arrêt complet : UNE stabilisation, puis plus rien.
            self.active = false;
            self.since_render = f64::INFINITY;
            NavOutcome {
                zoom: None,
                render: NavRender::Settle,
                repaint: false,
            }
        } else {
            NavOutcome::IDLE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(dir: f64, dt: f64) -> NavInput {
        NavInput {
            dir,
            dt,
            anchor: Some((0.5, 0.5)),
            cruise: ZOOM_RATE,
            rendering: false,
        }
    }

    /// RÉGRESSION (2026-08-09) : la rampe doit démarrer depuis l'arrêt. Le tick
    /// initial a `dt = 0` (aucun tick précédent) ; si l'état actif n'était pas
    /// marqué à ce moment, le tick suivant aurait encore `dt = 0` et la vitesse
    /// resterait nulle indéfiniment — le zoom ne démarrait jamais (la coche
    /// XaoS paraissait sans effet).
    #[test]
    fn ramp_starts_from_rest() {
        let mut s = NavState::new();
        let first = s.tick(input(1.0, 0.0));
        assert!(first.repaint, "le 1er tick doit marquer la navigation active");
        assert!(s.is_active());

        let second = s.tick(input(1.0, 1.0 / 60.0));
        assert!(s.velocity() > 0.0, "la vitesse doit monter dès dt > 0");
        assert!(second.zoom.is_some(), "un zoom doit être produit");
        let (_, _, f) = second.zoom.unwrap();
        assert!(f > 1.0, "zoom avant ⇒ facteur > 1, obtenu {f}");
    }

    /// La montée atteint la croisière en ACCEL_SECS (± une frame).
    #[test]
    fn ramp_reaches_cruise_in_accel_secs() {
        let mut s = NavState::new();
        let dt = 1.0 / 60.0;
        let steps = (ACCEL_SECS / dt).ceil() as usize + 1;
        for _ in 0..steps {
            s.tick(input(1.0, dt));
        }
        let cruise_ln = ZOOM_RATE.ln();
        assert!(
            (s.velocity() - cruise_ln).abs() < 1e-6,
            "vitesse {} attendue ≈ {cruise_ln}",
            s.velocity()
        );
        // Et elle ne dépasse pas la croisière.
        s.tick(input(1.0, dt));
        assert!(s.velocity() <= cruise_ln + 1e-12);
    }

    /// Après relâchement, le zoom continue en s'amortissant puis s'arrête,
    /// et produit exactement UNE stabilisation.
    #[test]
    fn release_decelerates_then_settles_once() {
        let mut s = NavState::new();
        let dt = 1.0 / 60.0;
        for _ in 0..40 {
            s.tick(input(1.0, dt));
        }
        assert!(s.velocity() > 0.0);

        let mut zooms_after_release = 0;
        let mut settles = 0;
        for _ in 0..200 {
            let o = s.tick(input(0.0, dt));
            if o.zoom.is_some() {
                zooms_after_release += 1;
            }
            if o.render == NavRender::Settle {
                settles += 1;
            }
        }
        assert!(
            zooms_after_release > 10,
            "le zoom doit s'amortir, pas s'arrêter net ({zooms_after_release} pas)"
        );
        assert_eq!(settles, 1, "exactement une stabilisation");
        assert_eq!(s.velocity(), 0.0);
        assert!(!s.is_active());

        // Au repos : plus aucun événement.
        let o = s.tick(input(0.0, dt));
        assert_eq!(o, NavOutcome::IDLE);
    }

    /// La décélération est plus lente que l'accélération (rapport XaoS).
    #[test]
    fn deceleration_is_slower_than_acceleration() {
        let dt = 1.0 / 240.0;
        let mut accel = NavState::new();
        let mut n_accel = 0;
        while accel.velocity() < ZOOM_RATE.ln() - 1e-9 && n_accel < 10_000 {
            accel.tick(input(1.0, dt));
            n_accel += 1;
        }
        let mut decel = accel.clone();
        let mut n_decel = 0;
        while decel.velocity() != 0.0 && n_decel < 10_000 {
            decel.tick(input(0.0, dt));
            n_decel += 1;
        }
        let ratio = n_decel as f64 / n_accel as f64;
        assert!(
            (ratio - DECEL_SECS / ACCEL_SECS).abs() < 0.1,
            "rapport décel/accel {ratio} attendu ≈ {}",
            DECEL_SECS / ACCEL_SECS
        );
    }

    /// Cadence : pas de rendu tant qu'un rendu tourne, ni avant le plancher.
    #[test]
    fn render_cadence_is_floored_and_never_stacked() {
        let mut s = NavState::new();
        let dt = 1.0 / 240.0;
        s.tick(input(1.0, 0.0));

        // Premier rendu : immédiat (pas d'attente en début de geste).
        let mut o = s.tick(input(1.0, dt));
        while o.render == NavRender::None {
            o = s.tick(input(1.0, dt));
        }
        assert_eq!(o.render, NavRender::Moving);

        // Juste après, sous le plancher : rien.
        let o = s.tick(input(1.0, dt));
        assert_eq!(o.render, NavRender::None);

        // Même bien au-delà du plancher, un rendu en cours bloque.
        let mut busy = input(1.0, MIN_RENDER_INTERVAL_SECS);
        busy.rendering = true;
        for _ in 0..10 {
            assert_eq!(s.tick(busy).render, NavRender::None);
        }

        // Rendu terminé + plancher franchi ⇒ relance.
        let o = s.tick(input(1.0, MIN_RENDER_INTERVAL_SECS));
        assert_eq!(o.render, NavRender::Moving);
    }

    /// L'ancrage est conservé quand le curseur quitte l'image.
    #[test]
    fn anchor_is_retained_when_pointer_leaves() {
        let mut s = NavState::new();
        let dt = 1.0 / 60.0;
        let mut i = input(1.0, dt);
        i.anchor = Some((0.25, 0.75));
        s.tick(i);
        s.tick(i);

        let mut without = input(0.0, dt);
        without.anchor = None;
        let o = s.tick(without);
        let (rx, ry, _) = o.zoom.expect("le zoom doit continuer en décélération");
        assert_eq!((rx, ry), (0.25, 0.75));
    }

    /// Un pas de temps aberrant (frame très lente) ne produit pas un saut de
    /// zoom incontrôlable : la physique est bornée par MAX_PHYSICS_DT.
    #[test]
    fn huge_dt_is_clamped() {
        let mut s = NavState::new();
        s.tick(input(1.0, 1.0 / 60.0));
        let o = s.tick(input(1.0, 30.0));
        let (_, _, f) = o.zoom.unwrap();
        let max_factor = (ZOOM_RATE.ln() * MAX_PHYSICS_DT).exp();
        assert!(f <= max_factor + 1e-9, "facteur {f} > borne {max_factor}");
    }

    /// Zoom arrière : facteur < 1, symétrique du zoom avant.
    #[test]
    fn zoom_out_produces_factor_below_one() {
        let mut s = NavState::new();
        let dt = 1.0 / 60.0;
        s.tick(input(-1.0, 0.0));
        for _ in 0..30 {
            s.tick(input(-1.0, dt));
        }
        let o = s.tick(input(-1.0, dt));
        let (_, _, f) = o.zoom.unwrap();
        assert!(f < 1.0, "zoom arrière ⇒ facteur < 1, obtenu {f}");
        assert!(s.velocity() < 0.0);
    }

    #[test]
    fn cruise_rate_tracks_engine_throughput() {
        // Moteur rapide : pleine vitesse.
        assert_eq!(cruise_rate(Some(0.02)), ZOOM_RATE);
        assert_eq!(cruise_rate(None), ZOOM_RATE);
        // 1 s par image : au plus MAX_STEP_PER_RENDER entre deux images.
        assert!((cruise_rate(Some(1.0)) - MAX_STEP_PER_RENDER).abs() < 1e-12);
        // Moteur très lent : plancher, la navigation reste perceptible.
        assert_eq!(cruise_rate(Some(100.0)), MIN_ZOOM_RATE);
        // Monotone : plus c'est lent, plus c'est doux.
        assert!(cruise_rate(Some(0.5)) >= cruise_rate(Some(2.0)));
    }

    #[test]
    fn nav_divisor_extrapolates_full_cost() {
        // Rapide : pleine résolution.
        assert_eq!(nav_divisor(Some(0.05), 1), 1);
        assert_eq!(nav_divisor(None, 1), 1);
        // 1 s en pleine résolution → besoin de d² ≥ 4 → d = 2.
        assert_eq!(nav_divisor(Some(1.0), 1), 2);
        // ⚠️ 0.5 s MESURÉ À d=2 ⇒ coût plein cadre 2 s → d ≥ 2.83 → 4.
        // Sans la prise en compte du diviseur de la mesure, on lirait 0.5 s
        // comme un coût plein cadre et on resterait à d = 2 (jamais de
        // redescente correcte).
        assert_eq!(nav_divisor(Some(0.5), 2), 4);
        // Plafonné.
        assert_eq!(nav_divisor(Some(60.0), 4), MAX_DIVISOR);
        // Redescente : une fois la scène redevenue peu coûteuse à d=4.
        assert_eq!(nav_divisor(Some(0.005), 4), 1);
    }
}
