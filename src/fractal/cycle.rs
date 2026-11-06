//! Raccourci des **cycles exacts** pour les boucles d'itération scalaires.
//!
//! Un point qui ne s'évade pas coûte `iteration_max` itérations. Mais une
//! orbite attirée par un cycle (intérieur d'un ensemble, racine d'un Newton,
//! point fixe d'un Magnet) devient, en arithmétique flottante, **exactement
//! périodique** : son état reprend bit pour bit une valeur déjà visitée.
//! Quand le pas est une fonction déterministe de cet état, la suite se répète
//! alors pour toujours — aucune sortie de boucle ne se déclenchera plus, et
//! l'état à `iteration_max` est celui atteint `(reste) mod (longueur du
//! cycle)` itérations plus loin. [`CycleCut::observe`] avance donc le
//! compteur d'un nombre ENTIER de cycles et laisse la boucle finir les
//! quelques itérations restantes : **même compte et même état final que
//! l'itération complète**, par construction.
//!
//! Ce n'est PAS un test de convergence : aucune tolérance, comparaison des
//! motifs binaires (`NaN` ne s'égale jamais, `+0` ≠ `−0`). Une quasi-période
//! n'est pas un cycle et n'est pas raccourcie.
//!
//! Conditions d'emploi, à vérifier pour chaque boucle :
//! - `state` contient TOUT ce dont dépend le pas suivant (ex. Phoenix :
//!   `z` et `y`) ; les constantes de la boucle (`c`, graine) n'en font pas
//!   partie ;
//! - la boucle ne cumule rien qui dépende du NOMBRE de passages (somme,
//!   moyenne). Un minimum / maximum sur l'orbite est permis : au moment de la
//!   détection un cycle entier a déjà été parcouru, le répéter ne change
//!   rien.
//!
//! Même principe que le noyau par lots (`bytecode/interp_batch.rs`), même
//! interrupteur : `FRACTALL_STD_CYCLE=0`.

/// Itérations entre deux tests. Puissance de deux.
const CHECK_PERIOD: u32 = 8;

/// Motif binaire qu'aucun état atteignable ne porte (NaN à charge pleine).
const NO_CHECKPOINT: u64 = u64::MAX;

/// Raccourci actif ? (`FRACTALL_STD_CYCLE=0` le coupe.)
pub(crate) fn shortcut_enabled() -> bool {
    #[cfg(test)]
    if FORCE_OFF.with(|f| f.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRACTALL_STD_CYCLE").ok().as_deref().map(str::trim),
            Some("0") | Some("off")
        )
    })
}

#[cfg(test)]
thread_local! {
    /// Coupe le raccourci pour le fil courant (tests de bit-identité).
    pub(crate) static FORCE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Nombre de sauts effectués par le fil courant (preuve que le raccourci
    /// a bien raccourci, sans dépendre d'un chronomètre).
    pub(crate) static CUTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Détecteur de cycle exact (Brent) sur un état de `N` flottants.
pub struct CycleCut<const N: usize> {
    ck: [u64; N],
    ck_iter: u32,
    ck_next: u32,
}

impl<const N: usize> CycleCut<N> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            ck: [NO_CHECKPOINT; N],
            ck_iter: 0,
            ck_next: if shortcut_enabled() { CHECK_PERIOD } else { u32::MAX },
        }
    }

    /// À appeler juste après l'incrément du compteur `i`, avec l'état que la
    /// prochaine itération va lire. Peut AVANCER `i` (jamais au-delà de
    /// `iteration_max - 1`).
    #[inline(always)]
    pub fn observe(&mut self, i: &mut u32, iteration_max: u32, state: [f64; N]) {
        if *i % CHECK_PERIOD == 0 {
            self.check(i, iteration_max, state);
        }
    }

    /// Comme [`Self::observe`], mais teste à CHAQUE itération : pour les
    /// boucles dont un pas coûte cher (puissance complexe, division, sinus)
    /// et dont les cycles ne sont pas de longueur 1 — un test tous les 8 pas
    /// ne voit un cycle de longueur `L` qu'au bout de `ppcm(L, 8)` pas.
    #[inline(always)]
    pub fn observe_each(&mut self, i: &mut u32, iteration_max: u32, state: [f64; N]) {
        self.check(i, iteration_max, state);
    }

    #[inline]
    fn check(&mut self, i: &mut u32, iteration_max: u32, state: [f64; N]) {
        let bits = state.map(f64::to_bits);
        if bits == self.ck {
            if *i < iteration_max {
                let len = *i - self.ck_iter;
                *i += (iteration_max - 1 - *i) / len * len;
                #[cfg(test)]
                CUTS.with(|c| c.set(c.get() + 1));
            }
            self.ck_next = u32::MAX;
        } else if *i >= self.ck_next {
            self.ck = bits;
            self.ck_iter = *i;
            self.ck_next = i.saturating_mul(2);
        }
    }
}

impl<const N: usize> Default for CycleCut<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Boucle modèle `z ← z² + c` avec et sans raccourci : mêmes sorties.
    fn run(c: (f64, f64), max: u32, cut: bool) -> (u32, f64, f64) {
        FORCE_OFF.with(|f| f.set(!cut));
        let (mut x, mut y) = (0.0f64, 0.0f64);
        let mut i = 0u32;
        let mut cyc = CycleCut::<2>::new();
        while i < max && x * x + y * y < 4.0 {
            let nx = x * x - y * y + c.0;
            y = 2.0 * x * y + c.1;
            x = nx;
            i += 1;
            cyc.observe(&mut i, max, [x, y]);
        }
        FORCE_OFF.with(|f| f.set(false));
        (i, x, y)
    }

    #[test]
    fn cut_loop_matches_full_loop_bit_for_bit() {
        let points = [
            (0.0, 0.0),
            (-1.0, 0.0),
            (-0.1, 0.6),
            (-0.12, 0.74),
            (0.25, 0.0),
            (-0.75, 0.01),
            (-1.75, 0.0),
            (0.3, 0.3),
            (-1.7548776662466927, 1e-4),
            (0.26, 0.0),
        ];
        for &c in &points {
            for max in [0u32, 1, 7, 8, 9, 16, 17, 255, 1000, 1001, 30_011] {
                let a = run(c, max, true);
                let b = run(c, max, false);
                assert!(
                    a.0 == b.0 && a.1.to_bits() == b.1.to_bits() && a.2.to_bits() == b.2.to_bits(),
                    "c={c:?} max={max} : raccourci {a:?} ≠ complet {b:?}"
                );
            }
        }
    }

    #[test]
    fn interior_point_is_cut_short() {
        // Un point du cardioïde, plafond de 50 millions : le raccourci saute
        // (et le test ne dure pas le temps de 50 millions d'itérations).
        let before = CUTS.with(|c| c.get());
        let r = run((-0.1, 0.2), 50_000_000, true);
        assert_eq!(r.0, 50_000_000);
        assert!(CUTS.with(|c| c.get()) > before);
    }
}
