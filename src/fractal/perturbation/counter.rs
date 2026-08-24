//! Compteur d'itération COMMUN des boucles pixel perturbation (chantier G5).
//!
//! Chaque boucle (f64 / exp / dd / GMP / multi-phase) maintenait à la main la
//! paire `n` (itération ABSOLUE) / `m` (index dans la référence). La classe de
//! bug : `iterate_pixel_gmp` n'avait QUE `m` et sa garde de boucle le testait —
//! au rebase d'un pixel intérieur (`m := 0`) elle bouclait à l'infini (v0.8.2).
//!
//! Invariants portés par ce type :
//! - `n` est MONOTONE : aucune méthode ne le diminue ni ne le remet à zéro ;
//! - `m ≤ n` toujours (vérifié en debug) ;
//! - la garde de boucle passe par [`PixelCounter::keep_iterating`], qui teste
//!   `n` — une boucle écrite avec ce type ne peut pas boucler sur un rebase.
//!
//! Toutes les méthodes sont `#[inline(always)]` : en release le codegen est
//! identique aux deux `u32` locaux historiques (vérifié goldens pixel-exact).

/// Paire (itération absolue `n`, index de référence `m`) d'une boucle pixel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelCounter {
    n: u32,
    m: u32,
}

impl PixelCounter {
    /// Départ de boucle : `n = m = 0`.
    #[inline(always)]
    pub fn new() -> Self {
        Self { n: 0, m: 0 }
    }

    /// Itération ABSOLUE — c'est elle que les gardes de boucle et le compte
    /// d'itération rapporté consomment.
    #[inline(always)]
    pub fn n(self) -> u32 {
        self.n
    }

    /// Index courant dans l'orbite de référence (repart de 0 au rebase).
    #[inline(always)]
    pub fn m(self) -> u32 {
        self.m
    }

    /// `m` prêt pour l'indexation de slice.
    #[inline(always)]
    pub fn m_usize(self) -> usize {
        self.m as usize
    }

    /// Garde de boucle : `n < iteration_max`. À utiliser comme condition du
    /// `while` — le compteur qui borne est l'ABSOLU, jamais `m`.
    #[inline(always)]
    pub fn keep_iterating(self, iteration_max: u32) -> bool {
        debug_assert!(
            self.m <= self.n,
            "invariant m ≤ n violé ({} > {})",
            self.m,
            self.n
        );
        debug_assert!(
            self.n <= iteration_max,
            "itération absolue hors borne ({} > {})",
            self.n,
            iteration_max
        );
        self.n < iteration_max
    }

    /// Un pas direct : `n += 1 ; m += 1`.
    #[inline(always)]
    pub fn step(&mut self) {
        self.n += 1;
        self.m += 1;
    }

    /// Candidat après un saut BLA de `l` pas (saturating, comme les boucles
    /// historiques). À COMMITTER par simple affectation (`c = cand`) une fois
    /// le saut validé — `n` du candidat est ≥ `n` courant par construction.
    #[inline(always)]
    #[must_use]
    pub fn after_jump(self, l: u32) -> Self {
        Self {
            n: self.n.saturating_add(l),
            m: self.m.saturating_add(l),
        }
    }

    /// Rebase F3 : la référence repart de `Z[0]`, `n` INCHANGÉ.
    #[inline(always)]
    pub fn rebase(&mut self) {
        self.m = 0;
    }

    /// Positionnement arbitraire dans la référence (wrap périodique des réfs
    /// tronquées). Ne touche jamais `n`.
    #[inline(always)]
    pub fn set_ref_index(&mut self, m: u32) {
        debug_assert!(
            m <= self.n,
            "wrap m={m} > n={} (index hors histoire)",
            self.n
        );
        self.m = m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_never_resets_absolute_counter() {
        let mut c = PixelCounter::new();
        for _ in 0..10 {
            c.step();
        }
        assert_eq!((c.n(), c.m()), (10, 10));
        c.rebase();
        assert_eq!((c.n(), c.m()), (10, 0), "rebase ⇒ m=0, n intact");
        // La garde teste n : une boucle qui rebase à chaque pas TERMINE.
        let mut c = PixelCounter::new();
        let iter_max = 1000u32;
        let mut guard = 0u32;
        while c.keep_iterating(iter_max) {
            c.step();
            c.rebase(); // pire cas : rebase à chaque itération (pixel intérieur)
            guard += 1;
            assert!(guard <= iter_max, "boucle infinie malgré le rebase");
        }
        assert_eq!(c.n(), iter_max);
    }

    #[test]
    fn jump_advances_both_and_saturates() {
        let mut c = PixelCounter::new();
        c.step();
        let cand = c.after_jump(5);
        assert_eq!((cand.n(), cand.m()), (6, 6));
        assert_eq!(
            (c.n(), c.m()),
            (1, 1),
            "after_jump est sans effet avant commit"
        );
        c = cand;
        assert_eq!((c.n(), c.m()), (6, 6));
        let sat = c.after_jump(u32::MAX);
        assert_eq!(sat.n(), u32::MAX);
        assert!(!sat.keep_iterating(u32::MAX));
    }

    #[test]
    fn set_ref_index_wraps_without_touching_n() {
        let mut c = PixelCounter::new();
        for _ in 0..7 {
            c.step();
        }
        c.set_ref_index(3);
        assert_eq!((c.n(), c.m()), (7, 3));
    }
}
