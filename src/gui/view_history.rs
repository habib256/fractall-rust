//! Historique de navigation de la GUI (retour / avance) et reprise de session.
//!
//! Logique PURE (aucune dépendance egui, horloge passée en argument) :
//! `gui/app.rs` ne fait que nourrir l'historique à chaque frame
//! ([`ViewHistory::observe`]), signaler les changements de scène
//! ([`ViewHistory::commit`]) et appliquer les vues rendues par
//! [`ViewHistory::back`] / [`ViewHistory::forward`].
//!
//! # Unité stockée
//!
//! Une entrée est un snapshot `FractalParams` COMPLET (léger : quelques
//! dizaines de champs, chaînes HP comprises). Deux entrées sont « la même
//! vue » si leur [`view_key`] est égale : la clé raisonne par GROUPE — elle
//! retire des params sérialisés les clés des groupes de PRÉFÉRENCES
//! (`color`, `sampling`, `engine`, `perturbation`, `channels`, la même
//! frontière que `params_for_type_keeping_preferences`) et trois grandeurs
//! qui ne sont pas de la navigation : les dimensions et `span_x` (dérivés de
//! la fenêtre), et `iteration_max` (le sage des itérations le relève sans
//! que l'utilisateur ait bougé). Un changement de préférence ou d'itérations
//! met donc à jour l'entrée COURANTE en place au lieu d'en empiler une.
//!
//! # Regroupement par geste
//!
//! Une vue n'est empilée qu'une fois STABLE : aucun geste en cours (bouton
//! maintenu, sélection, inertie) et clé inchangée depuis [`SETTLE`]. Une
//! rafale de molette ou un pan produit ainsi UNE entrée. Un changement de
//! scène (type, fichier déposé, reset) passe par [`ViewHistory::commit`],
//! immédiat, qui fige d'abord la vue en attente.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::fractal::view::ViewHp;
use crate::fractal::FractalParams;

/// Nombre maximal d'entrées (les plus anciennes sont oubliées).
pub const HISTORY_CAPACITY: usize = 200;

/// Durée de stabilité (sans geste ni changement) avant d'empiler une vue.
pub const SETTLE: Duration = Duration::from_millis(600);

/// Version du format de session.
pub const SESSION_VERSION: u32 = 1;

/// Clé « même vue » : params sérialisés, groupes de préférences retirés
/// (cf. en-tête du module).
pub fn view_key(params: &FractalParams) -> String {
    let Ok(serde_json::Value::Object(mut map)) = serde_json::to_value(params) else {
        return String::new();
    };
    // Groupes de préférences : on retire EXACTEMENT les clés que chaque groupe
    // sérialise (flatten), sans jamais les énumérer à la main.
    let groups = [
        serde_json::to_value(&params.color),
        serde_json::to_value(&params.sampling),
        serde_json::to_value(&params.engine),
        serde_json::to_value(&params.perturbation),
        serde_json::to_value(&params.channels),
    ];
    for group in groups.into_iter().flatten() {
        if let serde_json::Value::Object(keys) = group {
            for key in keys.keys() {
                map.remove(key);
            }
        }
    }
    // Ni dimensions (fenêtre), ni span horizontal (dérivé du ratio de la
    // fenêtre), ni plafond d'itérations (relevé par le sage).
    for derived in ["width", "height", "span_x", "span_x_hp", "iteration_max"] {
        map.remove(derived);
    }
    serde_json::Value::Object(map).to_string()
}

/// Même scène : même type et même formule. Sert à décider quels groupes une
/// restauration reprend de l'entrée.
fn same_scene(a: &FractalParams, b: &FractalParams) -> bool {
    a.fractal_type == b.fractal_type
        && serde_json::to_value(&a.formula).ok() == serde_json::to_value(&b.formula).ok()
}

/// Params à appliquer pour revenir à `entry` depuis l'état `current`.
///
/// - Même scène (type + formule) : seule la navigation bouge — géométrie,
///   itérations et champs de premier niveau viennent de l'entrée, les
///   groupes de préférences (couleur, échantillonnage, moteur, perturbation,
///   canaux) restent ceux de l'utilisateur.
/// - Autre scène : l'entrée entière (ses couleurs étaient celles de son type,
///   ex. `color_repeat` ≤ 8 des densités).
///
/// Dimensions : celles de `current` (la fenêtre). Si elles diffèrent de
/// l'entrée, la vue est redimensionnée en HP comme un redimensionnement de
/// fenêtre (`ViewHp::resize`) ; les chaînes HP du CENTRE restent celles de
/// l'entrée, octet pour octet.
pub fn restore_params(entry: &FractalParams, current: &FractalParams) -> FractalParams {
    let mut restored = entry.clone();
    if same_scene(entry, current) {
        restored.color = current.color.clone();
        restored.sampling = current.sampling.clone();
        restored.engine = current.engine.clone();
        restored.perturbation = current.perturbation.clone();
        restored.channels = current.channels.clone();
    }
    // État transitoire d'AA : jamais restauré.
    restored.sampling.aa_subpixel_offset = [0.0, 0.0];
    restored.sampling.aa_jitter = None;

    let (width, height) = (current.width.max(1), current.height.max(1));
    if (restored.width, restored.height) != (width, height) {
        let center = (restored.center_x_hp.clone(), restored.center_y_hp.clone());
        let mut view = ViewHp::from_params(&restored);
        view.resize(width, height);
        view.write_to_params(&mut restored);
        if center.0.is_some() && center.1.is_some() {
            restored.center_x_hp = center.0;
            restored.center_y_hp = center.1;
        }
    }
    restored
}

#[derive(Debug, Clone)]
struct Entry {
    params: FractalParams,
    key: String,
}

impl Entry {
    fn new(params: &FractalParams) -> Self {
        Self {
            params: params.clone(),
            key: view_key(params),
        }
    }
}

#[derive(Debug, Clone)]
struct Pending {
    entry: Entry,
    since: Instant,
}

/// Pile de vues avec curseur (comportement navigateur).
#[derive(Debug, Clone)]
pub struct ViewHistory {
    entries: Vec<Entry>,
    cursor: usize,
    capacity: usize,
    pending: Option<Pending>,
}

impl ViewHistory {
    /// Historique d'une seule entrée : la vue initiale.
    pub fn new(initial: &FractalParams) -> Self {
        Self::with_capacity(initial, HISTORY_CAPACITY)
    }

    pub fn with_capacity(initial: &FractalParams, capacity: usize) -> Self {
        Self {
            entries: vec![Entry::new(initial)],
            cursor: 0,
            capacity: capacity.max(1),
            pending: None,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn can_back(&self) -> bool {
        self.cursor > 0 || self.pending.is_some()
    }

    pub fn can_forward(&self) -> bool {
        self.cursor + 1 < self.entries.len() && self.pending.is_none()
    }

    /// Une vue attend d'être empilée (geste en cours ou pas encore stable).
    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Entrée courante (params tels que vus en dernier à cette position).
    pub fn current(&self) -> &FractalParams {
        &self.entries[self.cursor].params
    }

    /// Temps restant avant que la vue en attente soit empilée (pour
    /// programmer un repaint). `None` sans attente.
    pub fn time_to_settle(&self, now: Instant) -> Option<Duration> {
        self.pending
            .as_ref()
            .map(|p| SETTLE.saturating_sub(now.saturating_duration_since(p.since)))
    }

    /// À appeler à chaque frame avec l'état courant. Renvoie `true` si une
    /// entrée a été empilée.
    pub fn observe(&mut self, params: &FractalParams, now: Instant, gesture_active: bool) -> bool {
        let entry = Entry::new(params);
        if entry.key == self.entries[self.cursor].key {
            // Même vue : préférences / itérations à jour, rien d'empilé.
            self.entries[self.cursor] = entry;
            self.pending = None;
            return false;
        }
        let since = match &self.pending {
            Some(p) if p.entry.key == entry.key && !gesture_active => p.since,
            _ => now,
        };
        self.pending = Some(Pending { entry, since });
        if !gesture_active && now.saturating_duration_since(since) >= SETTLE {
            let pending = self.pending.take().expect("pending posé ci-dessus");
            self.push(pending.entry);
            return true;
        }
        false
    }

    /// Enregistre immédiatement `params` (changement de scène, avant un
    /// retour…) : la vue en attente est d'abord figée, puis `params` empilé
    /// s'il diffère de l'entrée courante (sinon mise à jour en place).
    pub fn commit(&mut self, params: &FractalParams) {
        if let Some(pending) = self.pending.take() {
            if pending.entry.key != self.entries[self.cursor].key {
                self.push(pending.entry);
            }
        }
        let entry = Entry::new(params);
        if entry.key == self.entries[self.cursor].key {
            self.entries[self.cursor] = entry;
        } else {
            self.push(entry);
        }
    }

    /// Remplace l'entrée courante par `params` sans rien empiler (la vue
    /// appliquée après une restauration, un redimensionnement de fenêtre).
    pub fn refresh_current(&mut self, params: &FractalParams) {
        self.pending = None;
        self.entries[self.cursor] = Entry::new(params);
    }

    /// Retour : enregistre l'état courant puis recule d'une entrée. Renvoie
    /// les params à appliquer (cf. [`restore_params`]), `None` en butée.
    pub fn back(&mut self, current: &FractalParams) -> Option<FractalParams> {
        self.commit(current);
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        Some(restore_params(&self.entries[self.cursor].params, current))
    }

    /// Avance (après un ou plusieurs retours). `None` en butée ou si une
    /// nouvelle vue vient d'être empilée (l'avant est alors tronqué).
    pub fn forward(&mut self, current: &FractalParams) -> Option<FractalParams> {
        self.commit(current);
        if self.cursor + 1 >= self.entries.len() {
            return None;
        }
        self.cursor += 1;
        Some(restore_params(&self.entries[self.cursor].params, current))
    }

    fn push(&mut self, entry: Entry) {
        // Naviguer après un retour tronque l'avant.
        self.entries.truncate(self.cursor + 1);
        self.entries.push(entry);
        if self.entries.len() > self.capacity {
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(..excess);
        }
        self.cursor = self.entries.len() - 1;
    }

    /// Instantané sérialisable (session). La vue en attente est incluse
    /// comme entrée la plus récente.
    pub fn to_session(&self, iter_auto: bool, window_size: Option<[f32; 2]>) -> Session {
        let mut history = self.clone();
        if let Some(pending) = history.pending.take() {
            history.push(pending.entry);
        }
        Session {
            version: SESSION_VERSION,
            cursor: history.cursor,
            entries: history.entries.into_iter().map(|e| e.params).collect(),
            iter_auto,
            // NaN/∞ s'écriraient `null` et rendraient la session illisible.
            window_size: window_size.filter(|[w, h]| w.is_finite() && h.is_finite()),
        }
    }

    /// Historique reconstruit depuis une session déjà validée.
    pub fn from_session(session: &Session) -> Self {
        let mut history = Self::new(&session.entries[0]);
        history.entries = session.entries.iter().map(Entry::new).collect();
        history.cursor = session.cursor.min(history.entries.len() - 1);
        history.truncate_to_capacity();
        history
    }

    fn truncate_to_capacity(&mut self) {
        if self.entries.len() > self.capacity {
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(..excess);
            self.cursor = self.cursor.saturating_sub(excess);
        }
    }
}

/// Session GUI persistée à la fermeture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    /// Historique, du plus ancien au plus récent.
    pub entries: Vec<FractalParams>,
    /// Index de la vue affichée à la fermeture.
    pub cursor: usize,
    /// Case « Iter: Auto ».
    #[serde(default = "default_true")]
    pub iter_auto: bool,
    /// Taille intérieure de la fenêtre (points logiques).
    #[serde(default)]
    pub window_size: Option<[f32; 2]>,
}

fn default_true() -> bool {
    true
}

impl Session {
    /// Vue à reprendre.
    pub fn current(&self) -> &FractalParams {
        &self.entries[self.cursor]
    }
}

fn entry_is_sane(p: &FractalParams) -> bool {
    let valid_span = |mirror: f64, hp: Option<&str>| {
        if mirror == 0.0 {
            return hp.and_then(|s| rug::Float::parse(s).ok())
                .map(|v| rug::Float::with_val(64, v))
                .is_some_and(|v| v > 0 && v.to_f64() == 0.0);
        }
        mirror.is_finite() && mirror > 0.0
    };
    p.width > 0
        && p.height > 0
        && p.center_x.is_finite()
        && p.center_y.is_finite()
        && valid_span(p.span_x, p.span_x_hp.as_deref())
        && valid_span(p.span_y, p.span_y_hp.as_deref())
}

/// Lit une session depuis son texte JSON. `None` (jamais de panique) si le
/// texte est illisible, d'une autre version, vide, ou si l'entrée courante
/// est incohérente ; les entrées incohérentes du reste de la pile sont
/// écartées.
pub fn parse_session(text: &str) -> Option<Session> {
    let mut session: Session = serde_json::from_str(text).ok()?;
    if session.version != SESSION_VERSION || session.cursor >= session.entries.len() {
        return None;
    }
    if !entry_is_sane(&session.entries[session.cursor]) {
        return None;
    }
    let cursor_entry = session.cursor;
    let mut kept = Vec::with_capacity(session.entries.len());
    let mut new_cursor = 0;
    for (i, entry) in session.entries.into_iter().enumerate() {
        if entry_is_sane(&entry) {
            if i == cursor_entry {
                new_cursor = kept.len();
            }
            kept.push(entry);
        }
    }
    session.entries = kept;
    session.cursor = new_cursor;
    if let Some([w, h]) = session.window_size {
        if !(w.is_finite() && h.is_finite() && w >= 100.0 && h >= 100.0 && w <= 16384.0 && h <= 16384.0)
        {
            session.window_size = None;
        }
    }
    Some(session)
}

/// Chemin du fichier de session : `FRACTALL_GUI_SESSION_FILE`, sinon
/// `~/.config/fractall/gui-session.json` (à côté de `wisdom.toml`).
pub fn session_file_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FRACTALL_GUI_SESSION_FILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    let home = std::env::var("HOME")
        .ok()
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var("USERPROFILE").ok().filter(|h| !h.is_empty()))?;
    Some(PathBuf::from(home).join(".config/fractall/gui-session.json"))
}

/// Charge une session ; absente ou corrompue ⇒ `None` (message sur stderr
/// pour une session corrompue).
pub fn load_session(path: &Path) -> Option<Session> {
    let text = std::fs::read_to_string(path).ok()?;
    let session = parse_session(&text);
    if session.is_none() {
        eprintln!(
            "Ignoring unreadable GUI session {} (starting from the default view)",
            path.display()
        );
    }
    session
}

/// Écrit la session (fichier temporaire puis renommage : une fermeture
/// interrompue ne laisse jamais un fichier tronqué).
pub fn save_session(path: &Path, session: &Session) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let text = serde_json::to_string(session).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{default_params_for_type, FractalType};

    fn base() -> FractalParams {
        let mut p = default_params_for_type(FractalType::Mandelbrot, 800, 600);
        ViewHp::from_params(&p).write_to_params(&mut p);
        p
    }

    fn zoomed(p: &FractalParams, factor: f64) -> FractalParams {
        let mut q = p.clone();
        let mut view = ViewHp::from_params(&q);
        view.zoom_at(0.3, 0.6, factor);
        view.write_to_params(&mut q);
        q
    }

    fn deep() -> FractalParams {
        let mut p = base();
        p.center_x_hp = Some(
            "-1.7490015207983421550227838357219282645939862148013214623780930457".into(),
        );
        p.center_y_hp = Some("1.2345678901234567890123456789012345678901234567e-40".into());
        p.span_x_hp = Some("4.0e-30".into());
        p.span_y_hp = Some("3.0e-30".into());
        p.center_x = -1.749;
        p.span_x = 4e-30;
        p.span_y = 3e-30;
        p
    }

    #[test]
    fn settled_view_is_pushed_once() {
        let t0 = Instant::now();
        let a = base();
        let mut h = ViewHistory::new(&a);
        let b = zoomed(&a, 2.0);
        assert!(!h.observe(&b, t0, false));
        assert!(!h.observe(&b, t0 + SETTLE / 2, false));
        assert!(h.observe(&b, t0 + SETTLE, false));
        assert!(!h.observe(&b, t0 + SETTLE * 3, false));
        assert_eq!(h.len(), 2);
        assert_eq!(h.cursor(), 1);
    }

    #[test]
    fn gesture_is_grouped_into_one_entry() {
        let t0 = Instant::now();
        let a = base();
        let mut h = ViewHistory::new(&a);
        // Rafale de molette : 10 crans espacés de 50 ms puis repos.
        let mut v = a.clone();
        for k in 0..10 {
            v = zoomed(&v, 1.2);
            h.observe(&v, t0 + Duration::from_millis(50 * k), false);
        }
        // Pan bouton maintenu : longtemps, mais geste actif.
        for k in 0..20 {
            v = zoomed(&v, 1.01);
            h.observe(&v, t0 + Duration::from_millis(500 + 100 * k), true);
        }
        assert_eq!(h.len(), 1, "rien d'empilé pendant le geste");
        let end = t0 + Duration::from_millis(2500);
        h.observe(&v, end, false);
        assert!(h.observe(&v, end + SETTLE, false));
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn preference_and_iteration_changes_update_in_place() {
        let t0 = Instant::now();
        let a = base();
        let mut h = ViewHistory::new(&a);
        let mut b = a.clone();
        b.iteration_max = 12345;
        b.color.color_mode = 3;
        b.engine.use_dd_tier = true;
        b.width = 1024; // redimensionnement : pas de la navigation
        b.span_x *= 1.5;
        b.span_x_hp = None;
        assert!(!h.observe(&b, t0, false));
        assert!(!h.observe(&b, t0 + SETTLE * 2, false));
        assert_eq!(h.len(), 1);
        assert_eq!(h.current().iteration_max, 12345);
    }

    #[test]
    fn back_forward_and_truncation() {
        let a = base();
        let b = zoomed(&a, 2.0);
        let c = zoomed(&b, 2.0);
        let mut h = ViewHistory::new(&a);
        h.commit(&b);
        h.commit(&c);
        assert_eq!(h.len(), 3);
        let r = h.back(&c).unwrap();
        assert_eq!(view_key(&r), view_key(&b));
        h.refresh_current(&r);
        let r = h.back(&r).unwrap();
        assert_eq!(view_key(&r), view_key(&a));
        assert!(h.back(&r).is_none(), "butée");
        let r = h.forward(&r).unwrap();
        assert_eq!(view_key(&r), view_key(&b));
        assert!(h.can_forward());
        // Nouvelle navigation depuis b : l'avant (c) est tronqué.
        let d = zoomed(&r, 3.0);
        h.commit(&d);
        assert_eq!(h.len(), 3);
        assert!(!h.can_forward());
        assert!(h.forward(&d).is_none());
        let r = h.back(&d).unwrap();
        assert_eq!(view_key(&r), view_key(&b));
    }

    #[test]
    fn back_flushes_pending_view_first() {
        let t0 = Instant::now();
        let a = base();
        let b = zoomed(&a, 2.0);
        let mut h = ViewHistory::new(&a);
        h.observe(&b, t0, false); // pas encore stable
        assert!(h.can_back());
        let r = h.back(&b).unwrap();
        assert_eq!(view_key(&r), view_key(&a));
        // b a été enregistré : on peut y revenir.
        let r = h.forward(&r).unwrap();
        assert_eq!(view_key(&r), view_key(&b));
    }

    #[test]
    fn capacity_is_bounded() {
        let a = base();
        let mut h = ViewHistory::with_capacity(&a, 5);
        let mut v = a.clone();
        for _ in 0..12 {
            v = zoomed(&v, 1.5);
            h.commit(&v);
        }
        assert_eq!(h.len(), 5);
        assert_eq!(h.cursor(), 4);
        assert_eq!(view_key(h.current()), view_key(&v));
    }

    #[test]
    fn restore_keeps_hp_strings_exact() {
        let d = deep();
        let mut h = ViewHistory::new(&d);
        let e = zoomed(&d, 10.0);
        h.commit(&e);
        let r = h.back(&e).unwrap();
        assert_eq!(r.center_x_hp, d.center_x_hp);
        assert_eq!(r.center_y_hp, d.center_y_hp);
        assert_eq!(r.span_x_hp, d.span_x_hp);
        assert_eq!(r.span_y_hp, d.span_y_hp);
    }

    #[test]
    fn restore_with_other_window_size_keeps_center_and_aspect() {
        let d = deep();
        let mut cur = zoomed(&d, 10.0);
        let mut view = ViewHp::from_params(&cur);
        view.resize(1200, 600);
        view.write_to_params(&mut cur);
        let r = restore_params(&d, &cur);
        assert_eq!((r.width, r.height), (1200, 600));
        assert_eq!(r.center_x_hp, d.center_x_hp);
        assert_eq!(r.center_y_hp, d.center_y_hp);
        assert!((r.span_x / r.span_y - 2.0).abs() < 1e-12);
    }

    #[test]
    fn restore_same_scene_keeps_preferences_other_scene_takes_entry() {
        let a = base();
        let mut cur = zoomed(&a, 2.0);
        cur.color.color_mode = 11;
        cur.color.color_repeat = 77;
        let r = restore_params(&a, &cur);
        assert_eq!(r.color.color_mode, 11);
        assert_eq!(r.color.color_repeat, 77);

        let mut buddha = default_params_for_type(FractalType::Buddhabrot, 800, 600);
        buddha.color.color_repeat = 2;
        let r = restore_params(&buddha, &cur);
        assert_eq!(r.fractal_type, FractalType::Buddhabrot);
        assert_eq!(r.color.color_repeat, 2);
    }

    #[test]
    fn type_change_is_a_new_entry() {
        let a = base();
        let mut h = ViewHistory::new(&a);
        let j = default_params_for_type(FractalType::Julia, 800, 600);
        h.commit(&j);
        assert_eq!(h.len(), 2);
        let mut hyb = a.clone();
        hyb.formula.hybrid_phases = Some(vec![FractalType::Mandelbrot, FractalType::BurningShip]);
        h.commit(&hyb);
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn session_round_trip_is_exact() {
        let d = deep();
        let mut h = ViewHistory::new(&base());
        h.commit(&d);
        let e = zoomed(&d, 4.0);
        h.observe(&e, Instant::now(), true); // vue en attente : sauvée aussi
        let session = h.to_session(false, Some([1280.0, 720.0]));
        let text = serde_json::to_string(&session).unwrap();
        let back = parse_session(&text).expect("session relisible");
        assert_eq!(back.entries.len(), 3);
        assert_eq!(back.cursor, 2);
        assert!(!back.iter_auto);
        assert_eq!(back.window_size, Some([1280.0, 720.0]));
        assert_eq!(back.current().center_x_hp, e.center_x_hp);
        assert_eq!(back.entries[1].center_x_hp, d.center_x_hp);
        assert_eq!(back.entries[1].span_y_hp, d.span_y_hp);
        let h2 = ViewHistory::from_session(&back);
        assert_eq!(h2.cursor(), 2);
        assert_eq!(view_key(h2.current()), view_key(&e));
    }

    #[test]
    fn session_restores_spans_that_underflow_f64() {
        let mut p = base();
        let view = ViewHp::from_decimal_parts("-0.5", "0", "1e-400", "1e-400", 800, 600, 256).unwrap();
        view.write_to_params(&mut p);
        assert_eq!(p.span_x, 0.0);
        let text = serde_json::to_string(&ViewHistory::new(&p).to_session(true, None)).unwrap();
        assert_eq!(parse_session(&text).unwrap().current().span_x_hp, p.span_x_hp);
        p.span_x_hp = None;
        let text = serde_json::to_string(&ViewHistory::new(&p).to_session(true, None)).unwrap();
        assert!(parse_session(&text).is_none());
    }

    #[test]
    fn corrupt_session_is_rejected_without_panic() {
        assert!(parse_session("").is_none());
        assert!(parse_session("{not json").is_none());
        assert!(parse_session("{\"version\":1,\"entries\":[],\"cursor\":0}").is_none());
        let session = ViewHistory::new(&base()).to_session(true, None);
        let mut v = serde_json::to_value(&session).unwrap();
        v["cursor"] = serde_json::json!(5);
        assert!(parse_session(&v.to_string()).is_none());
        let mut v = serde_json::to_value(&session).unwrap();
        v["version"] = serde_json::json!(99);
        assert!(parse_session(&v.to_string()).is_none());
        let mut v = serde_json::to_value(&session).unwrap();
        v["entries"][0]["fractal_type"] = serde_json::json!("NoSuchFractal");
        assert!(parse_session(&v.to_string()).is_none());
        let mut v = serde_json::to_value(&session).unwrap();
        v["entries"][0]["span_x"] = serde_json::json!(-1.0);
        assert!(parse_session(&v.to_string()).is_none());
    }

    #[test]
    fn insane_entries_are_dropped_cursor_follows() {
        let a = base();
        let mut h = ViewHistory::new(&a);
        h.commit(&zoomed(&a, 2.0));
        h.commit(&zoomed(&a, 4.0));
        let session = h.to_session(true, Some([10.0, 50.0]));
        let mut v = serde_json::to_value(&session).unwrap();
        v["entries"][0]["width"] = serde_json::json!(0);
        let s = parse_session(&v.to_string()).unwrap();
        assert_eq!(s.entries.len(), 2);
        assert_eq!(s.cursor, 1);
        assert_eq!(s.window_size, None);
    }

    #[test]
    fn save_and_load_file() {
        let dir = std::env::temp_dir().join(format!("fractall-session-test-{}", std::process::id()));
        let path = dir.join("sub/gui-session.json");
        let session = ViewHistory::new(&deep()).to_session(true, None);
        save_session(&path, &session).unwrap();
        let loaded = load_session(&path).unwrap();
        assert_eq!(loaded.current().center_x_hp, deep().center_x_hp);
        std::fs::write(&path, "garbage").unwrap();
        assert!(load_session(&path).is_none());
        assert!(load_session(&dir.join("missing.json")).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
