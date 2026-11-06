//! Interpréteur bytecode f64 **par lots**, à voies entrelacées.
//!
//! [`super::interp::iterate_bytecode_f64`] itère UN point : sa boucle est une
//! seule chaîne de dépendance `z → z² + c → …` (~10 cycles de latence par
//! itération) pendant que le cœur pourrait en mener plusieurs de front, et
//! l'interprétation (`match op` par opcode) est payée à chaque itération de
//! chaque point.
//!
//! Ici [`LANES`] points avancent en **lockstep** : chaque opcode est appliqué
//! à toutes les voies dans une petite boucle à borne fixe (vectorisable), le
//! `match` est amorti sur les voies, et les chaînes de dépendance des voies
//! se recouvrent. Dès qu'une voie termine (évasion, non-fini ou plafond
//! d'itérations) son résultat est écrit et la voie est **rechargée** avec le
//! point suivant du lot : les voies restent pleines tant qu'il reste du travail.
//!
//! **Bit-identique à l'interpréteur scalaire, par construction** : chaque voie
//! exécute exactement la même suite d'opérations flottantes, dans le même
//! ordre (pas de FMA, pas de réassociation), et la sortie est détectée à
//! l'itération même où le scalaire sort. Verrou :
//! `batch_matches_scalar_interpreter_bit_for_bit`.
//!
//! Limite : formule **mono-phase** (un hybride fait avancer chaque point dans
//! une phase qui dépend de SON compteur d'itérations — les voies ne
//! partageraient plus le même opcode).
//!
//! ## Points intérieurs : cycle EXACT (2026-11-06)
//!
//! Un point qui ne s'évade pas coûtait `iteration_max` passes — à lui seul le
//! temps de rendu des vues qui montrent l'intérieur de l'ensemble avec un
//! plafond élevé. Or une orbite attirée par un cycle devient, en arithmétique
//! flottante, **exactement périodique** : l'état `(z, registre)` reprend à
//! l'identique, bit pour bit, une valeur déjà visitée (médiane mesurée :
//! 250 à 1000 itérations dans le cardioïde, un bulbe ou un minibrot). La
//! passe étant une fonction déterministe de cet état (et de `c`, constant),
//! la suite se répète alors pour toujours : le point n'atteindra jamais
//! l'évasion, et son état à `iteration_max` est celui qu'il a `(reste) mod
//! (longueur du cycle)` passes plus loin. La voie SAUTE donc son compteur
//! d'un nombre entier de cycles et termine les quelques passes restantes
//! normalement : **même compte d'itérations et même z final que sans le
//! saut**, par construction (verrou
//! `cycle_shortcut_matches_full_iteration_bit_for_bit`).
//!
//! Détection façon Brent, tous les [`CYCLE_CHECK_PERIOD`] passes seulement
//! (le test est hors du corps chaud) : comparaison BINAIRE de l'état à un
//! point de contrôle par voie, rafraîchi quand le compteur a doublé. Aucune
//! tolérance — une quasi-période n'est pas un cycle, et `NaN` ne s'égale
//! jamais. `FRACTALL_STD_CYCLE=0` coupe le raccourci (A/B).

use num_complex::Complex64;

use super::{Op, Phase};

/// Nombre de points menés de front. 8 = quatre vecteurs NEON/SSE2 de deux
/// f64 (deux AVX2), assez pour recouvrir la latence de la chaîne `z² + c` sur
/// un cœur à quatre unités flottantes.
pub const LANES: usize = 8;

/// Voie sans point (lot épuisé) : elle continue d'itérer des zéros inoffensifs.
const NO_SLOT: usize = usize::MAX;

/// Nombre de passes entre deux tests de cycle. Puissance de deux. Un cycle
/// de longueur `L` est vu au plus `ppcm(L, période)` passes après son
/// installation : 16 garde le test sous ~5 % des instructions de la boucle.
pub const CYCLE_CHECK_PERIOD: u32 = 16;

/// Motif binaire qu'aucun état atteignable ne porte (NaN à charge pleine) :
/// point de contrôle « vide » d'une voie fraîchement chargée.
const NO_CHECKPOINT: u64 = u64::MAX;

/// Raccourci des cycles exacts actif ? (`FRACTALL_STD_CYCLE=0` le coupe.)
fn cycle_shortcut_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRACTALL_STD_CYCLE").ok().as_deref().map(str::trim),
            Some("0") | Some("off")
        )
    })
}

/// Itère `points` avec la phase `phase`, écrit le compte d'itérations et le z
/// final de chaque point dans `out_iter` / `out_z` (mêmes indices).
///
/// Convention (identique au dispatch scalaire `iterate_via_bytecode`) :
/// - `is_julia == false` (Mandelbrot-like) : `z₀ = seed`, `c = point` ;
/// - `is_julia == true` : `z₀ = point`, `c = seed`.
///
/// # Panics
/// Si `out_iter` ou `out_z` n'ont pas la longueur de `points`.
pub fn iterate_bytecode_f64_batch(
    phase: &Phase,
    points: &[Complex64],
    seed: Complex64,
    is_julia: bool,
    iteration_max: u32,
    bailout: f64,
    out_iter: &mut [u32],
    out_z: &mut [Complex64],
) {
    iterate_batch_impl(
        phase,
        points,
        seed,
        is_julia,
        iteration_max,
        bailout,
        out_iter,
        out_z,
        cycle_shortcut_enabled(),
    );
}

/// Corps de [`iterate_bytecode_f64_batch`] ; `cycle_shortcut` explicite pour
/// les tests (les deux réglages rendent les mêmes sorties).
#[allow(clippy::too_many_arguments)]
fn iterate_batch_impl(
    phase: &Phase,
    points: &[Complex64],
    seed: Complex64,
    is_julia: bool,
    iteration_max: u32,
    bailout: f64,
    out_iter: &mut [u32],
    out_z: &mut [Complex64],
    cycle_shortcut: bool,
) {
    let n = points.len();
    assert_eq!(out_iter.len(), n);
    assert_eq!(out_z.len(), n);
    let bailout_sqr = bailout * bailout;
    // Nombre de `Add` de la phase (= incrément du compteur par passe ; 1 en
    // pratique, `Add` terminant la phase).
    let adds = phase.ops.iter().filter(|op| matches!(op, Op::Add)).count() as u32;

    let mut re = [0.0f64; LANES];
    let mut im = [0.0f64; LANES];
    let mut stored_re = [0.0f64; LANES];
    let mut stored_im = [0.0f64; LANES];
    let mut c_re = [0.0f64; LANES];
    let mut c_im = [0.0f64; LANES];
    let mut iter = [0u32; LANES];
    let mut slot = [NO_SLOT; LANES];
    let mut done = [false; LANES];
    // Point de contrôle par voie (état binaire + compteur) et compteur à
    // partir duquel le rafraîchir — cf. doc du module.
    let mut ck = [[NO_CHECKPOINT; 4]; LANES];
    let mut ck_iter = [0u32; LANES];
    let mut ck_next = [CYCLE_CHECK_PERIOD; LANES];
    let mut passes = 0u32;

    let mut next = 0usize;
    let mut live = 0usize;

    // Charge le prochain point non trivial dans la voie `l`. Les points
    // résolus sans itérer (déjà échappés à l'entrée, plafond nul) sont écrits
    // directement — mêmes sorties anticipées que l'interpréteur scalaire.
    macro_rules! load {
        ($l:expr) => {{
            let l = $l;
            slot[l] = NO_SLOT;
            re[l] = 0.0;
            im[l] = 0.0;
            stored_re[l] = 0.0;
            stored_im[l] = 0.0;
            c_re[l] = 0.0;
            c_im[l] = 0.0;
            iter[l] = 0;
            ck[l] = [NO_CHECKPOINT; 4];
            ck_iter[l] = 0;
            ck_next[l] = CYCLE_CHECK_PERIOD;
            while next < n {
                let i = next;
                next += 1;
                let p = points[i];
                let (z0, c) = if is_julia { (p, seed) } else { (seed, p) };
                // `>=` exact du scalaire (un z₀ NaN n'est PAS une sortie ici :
                // il itère une fois puis sort « non fini » à l'itération 1).
                if z0.norm_sqr() >= bailout_sqr || iteration_max == 0 {
                    out_iter[i] = 0;
                    out_z[i] = z0;
                    continue;
                }
                re[l] = z0.re;
                im[l] = z0.im;
                stored_re[l] = z0.re;
                stored_im[l] = z0.im;
                c_re[l] = c.re;
                c_im[l] = c.im;
                slot[l] = i;
                live += 1;
                break;
            }
        }};
    }

    for l in 0..LANES {
        load!(l);
    }

    while live > 0 {
        // Passes de phase jusqu'à ce qu'au moins une voie termine.
        loop {
            for op in &phase.ops {
                match *op {
                    Op::Sqr => {
                        for l in 0..LANES {
                            let (a, b) = (re[l], im[l]);
                            re[l] = a * a - b * b;
                            im[l] = a * b + b * a;
                        }
                    }
                    Op::Mul => {
                        for l in 0..LANES {
                            let (a, b) = (re[l], im[l]);
                            let (sa, sb) = (stored_re[l], stored_im[l]);
                            re[l] = a * sa - b * sb;
                            im[l] = a * sb + b * sa;
                        }
                    }
                    Op::Store => {
                        stored_re = re;
                        stored_im = im;
                    }
                    Op::AbsX => {
                        for v in &mut re {
                            *v = v.abs();
                        }
                    }
                    Op::AbsY => {
                        for v in &mut im {
                            *v = v.abs();
                        }
                    }
                    Op::NegX => {
                        for v in &mut re {
                            *v = -*v;
                        }
                    }
                    Op::NegY => {
                        for v in &mut im {
                            *v = -*v;
                        }
                    }
                    Op::Add => {
                        for l in 0..LANES {
                            re[l] += c_re[l];
                            im[l] += c_im[l];
                        }
                    }
                    Op::Rot {
                        cos_theta,
                        sin_theta,
                    } => {
                        for l in 0..LANES {
                            let (a, b) = (re[l], im[l]);
                            re[l] = a * cos_theta - b * sin_theta;
                            im[l] = a * sin_theta + b * cos_theta;
                        }
                    }
                }
            }
            // Fin de passe. Le scalaire sort sur « non fini » PUIS sur
            // `|z|² ≥ ER²` PUIS sur le plafond : `!(|z|² < ER²)` couvre les
            // deux premiers (une composante non finie donne une norme
            // infinie ou NaN), avec le même z et le même compteur.
            let mut any = false;
            for l in 0..LANES {
                iter[l] += adds;
                let norm_sqr = re[l] * re[l] + im[l] * im[l];
                let d = !(norm_sqr < bailout_sqr) | (iter[l] >= iteration_max);
                done[l] = d;
                any |= d;
            }
            if any {
                break;
            }
            passes = passes.wrapping_add(1);
            if cycle_shortcut && passes % CYCLE_CHECK_PERIOD == 0 {
                for l in 0..LANES {
                    let state = [
                        re[l].to_bits(),
                        im[l].to_bits(),
                        stored_re[l].to_bits(),
                        stored_im[l].to_bits(),
                    ];
                    if state == ck[l] {
                        // État revu à l'identique `len` itérations plus
                        // tard : cycle exact. Saut d'un nombre ENTIER de
                        // cycles en gardant ≥ 1 itération avant le plafond ;
                        // les passes restantes (< un cycle) s'exécutent.
                        if slot[l] != NO_SLOT {
                            let len = iter[l] - ck_iter[l];
                            let k = (iteration_max - 1 - iter[l]) / len;
                            iter[l] += k * len;
                            ck_next[l] = u32::MAX;
                        }
                    } else if iter[l] >= ck_next[l] {
                        ck[l] = state;
                        ck_iter[l] = iter[l];
                        ck_next[l] = iter[l].saturating_mul(2);
                    }
                }
            }
        }
        for l in 0..LANES {
            if done[l] {
                if slot[l] != NO_SLOT {
                    out_iter[slot[l]] = iter[l];
                    out_z[slot[l]] = Complex64::new(re[l], im[l]);
                    live -= 1;
                }
                load!(l);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::bytecode::{
        compile_formula, iterate_bytecode_f64, parse_opcodes_formula, Formula,
    };
    use crate::fractal::FractalType;

    /// Générateur déterministe (xorshift) — pas de dépendance de test.
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Points couvrant l'ensemble, son bord, l'extérieur et les valeurs
    /// pathologiques (NaN, infinis, zéro signé, déjà échappé à l'entrée).
    fn probe_points(count: usize, seed: u64) -> Vec<Complex64> {
        let mut r = rng(seed);
        let mut pts: Vec<Complex64> = (0..count)
            .map(|_| Complex64::new(r() * 4.0 - 2.5, r() * 3.0 - 1.5))
            .collect();
        // Bord fin : beaucoup d'itérations, sorties à des instants variés.
        pts.extend((0..count / 4).map(|k| {
            let t = k as f64 / (count / 4) as f64;
            Complex64::new(-0.75 + 1e-3 * (t - 0.5), 0.1 + 1e-4 * t)
        }));
        pts.extend([
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.0, 0.0),
            Complex64::new(f64::NAN, 0.3),
            Complex64::new(0.2, f64::NAN),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
            Complex64::new(1e200, 1e200),
            Complex64::new(30.0, 0.0),
            Complex64::new(25.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(f64::MIN_POSITIVE, -f64::MIN_POSITIVE),
        ]);
        pts
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    fn assert_batch_matches_scalar(
        label: &str,
        formula: &Formula,
        points: &[Complex64],
        seed: Complex64,
        is_julia: bool,
        iteration_max: u32,
        bailout: f64,
    ) {
        assert_eq!(formula.phases.len(), 1, "{label} : mono-phase attendu");
        let mut it = vec![u32::MAX; points.len()];
        let mut zs = vec![Complex64::new(f64::NAN, f64::NAN); points.len()];
        iterate_bytecode_f64_batch(
            &formula.phases[0],
            points,
            seed,
            is_julia,
            iteration_max,
            bailout,
            &mut it,
            &mut zs,
        );
        for (i, &p) in points.iter().enumerate() {
            let (z0, c) = if is_julia { (p, seed) } else { (seed, p) };
            let r = iterate_bytecode_f64(formula, z0, c, iteration_max, bailout);
            assert!(
                it[i] == r.iteration && same(zs[i].re, r.z.re) && same(zs[i].im, r.z.im),
                "{label} point #{i} {p:?} (julia={is_julia}, max={iteration_max}) : \
                 lot ({}, {:?}) ≠ scalaire ({}, {:?})",
                it[i],
                zs[i],
                r.iteration,
                r.z
            );
        }
    }

    /// Verrou central : le lot rend, pour chaque point, EXACTEMENT ce que rend
    /// l'interpréteur scalaire — compte d'itérations et z final au bit près —
    /// sur toutes les familles compilables, en convention Mandelbrot et Julia,
    /// y compris les entrées pathologiques et les tailles de lot qui ne
    /// remplissent pas les voies.
    #[test]
    fn batch_matches_scalar_interpreter_bit_for_bit() {
        let types = [
            FractalType::Mandelbrot,
            FractalType::BurningShip,
            FractalType::Tricorn,
            FractalType::Celtic,
            FractalType::Buffalo,
            FractalType::PerpendicularBurningShip,
        ];
        let points = probe_points(600, 0x9E37_79B9_7F4A_7C15);
        let julia_seed = Complex64::new(-0.8, 0.156);
        for ty in types {
            let formula = compile_formula(ty, 2.0).expect("formule compilable");
            for max in [0u32, 1, 2, 7, 300] {
                assert_batch_matches_scalar(
                    &format!("{ty:?}"),
                    &formula,
                    &points,
                    Complex64::new(0.0, 0.0),
                    false,
                    max,
                    25.0,
                );
                assert_batch_matches_scalar(
                    &format!("{ty:?}"),
                    &formula,
                    &points,
                    julia_seed,
                    true,
                    max,
                    25.0,
                );
            }
        }
        // Puissances entières (chaînes Store/Sqr/Mul) et seed Mandelbrot non nul.
        for power in [3.0, 4.0, 5.0, 7.0] {
            let formula = compile_formula(FractalType::Multibrot, power).expect("multibrot entier");
            assert_batch_matches_scalar(
                &format!("Multibrot^{power}"),
                &formula,
                &points,
                Complex64::new(0.05, -0.02),
                false,
                120,
                25.0,
            );
        }
        // Opcodes F3 avec rotation et négations, bailout historique 2.
        let formula = parse_opcodes_formula("negx sqr rot{30} absy add").expect("opcodes");
        assert_batch_matches_scalar(
            "opcodes rot30",
            &formula,
            &points,
            Complex64::new(0.0, 0.0),
            false,
            200,
            2.0,
        );
    }

    /// Raccourci des cycles exacts : sur des points INTÉRIEURS (cardioïde,
    /// bulbes, minibrot, bord lent) et un plafond élevé, le lot avec
    /// raccourci rend exactement ce que rend l'itération complète — scalaire
    /// ET lot sans raccourci — et il a bien raccourci quelque chose.
    #[test]
    fn cycle_shortcut_matches_full_iteration_bit_for_bit() {
        let mut r = rng(0xC0FF_EE11_2233_4455);
        let mut pts: Vec<Complex64> = Vec::new();
        // Cardioïde et disque de période 2 (cycles courts, convergence rapide
        // au centre, lente au bord).
        pts.extend((0..160).map(|_| Complex64::new(r() * 1.6 - 1.3, r() * 1.2 - 0.6)));
        // Bulbe de période 3 et son voisinage.
        pts.extend((0..80).map(|_| Complex64::new(-0.1226 + 0.2 * (r() - 0.5), 0.7449 + 0.2 * (r() - 0.5))));
        // Minibrot de période 3 de l'axe réel (cycles longs).
        pts.extend((0..80).map(|_| {
            Complex64::new(-1.7548776662466927 + 0.03 * (r() - 0.5), 0.03 * (r() - 0.5))
        }));
        // Centres exacts (super-attractifs) et zéros signés.
        pts.extend([
            Complex64::new(0.0, 0.0),
            Complex64::new(-0.0, -0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(-0.1225611668766536, 0.7448617666197442),
            Complex64::new(0.25, 0.0),
            Complex64::new(-0.75, 0.0),
            Complex64::new(-2.0, 0.0),
        ]);
        let cases: Vec<(String, Formula, Complex64, bool, f64)> = vec![
            (
                "Mandelbrot".into(),
                compile_formula(FractalType::Mandelbrot, 2.0).unwrap(),
                Complex64::new(0.0, 0.0),
                false,
                25.0,
            ),
            (
                "Julia".into(),
                compile_formula(FractalType::Mandelbrot, 2.0).unwrap(),
                Complex64::new(-0.12, 0.74),
                true,
                25.0,
            ),
            (
                "BurningShip".into(),
                compile_formula(FractalType::BurningShip, 2.0).unwrap(),
                Complex64::new(0.0, 0.0),
                false,
                25.0,
            ),
            (
                "Multibrot^3".into(),
                compile_formula(FractalType::Multibrot, 3.0).unwrap(),
                Complex64::new(0.0, 0.0),
                false,
                25.0,
            ),
            (
                "opcodes store/mul".into(),
                parse_opcodes_formula("store sqr mul add").unwrap(),
                Complex64::new(0.0, 0.0),
                false,
                2.0,
            ),
        ];
        for (label, formula, seed, is_julia, bailout) in &cases {
            // Plafonds qui tombent à des positions variées dans le cycle.
            for max in [1u32, 15, 16, 17, 33, 1000, 1001, 4096, 20_003] {
                let run = |shortcut: bool| {
                    let mut it = vec![u32::MAX; pts.len()];
                    let mut zs = vec![Complex64::new(f64::NAN, f64::NAN); pts.len()];
                    iterate_batch_impl(
                        &formula.phases[0],
                        &pts,
                        *seed,
                        *is_julia,
                        max,
                        *bailout,
                        &mut it,
                        &mut zs,
                        shortcut,
                    );
                    (it, zs)
                };
                let (it_s, zs_s) = run(true);
                let (it_f, zs_f) = run(false);
                for i in 0..pts.len() {
                    assert!(
                        it_s[i] == it_f[i]
                            && same(zs_s[i].re, zs_f[i].re)
                            && same(zs_s[i].im, zs_f[i].im),
                        "{label} point #{i} {:?} max={max} : raccourci ({}, {:?}) ≠ complet ({}, {:?})",
                        pts[i],
                        it_s[i],
                        zs_s[i],
                        it_f[i],
                        zs_f[i]
                    );
                }
            }
            // … et contre l'interpréteur scalaire (qui n'a pas de raccourci).
            assert_batch_matches_scalar(label, formula, &pts, *seed, *is_julia, 5000, *bailout);
        }
    }

    /// Le raccourci RACCOURCIT : sur des points intérieurs et un plafond de
    /// dix millions, le lot termine en un temps sans rapport avec le plafond
    /// (l'itération complète prendrait des minutes), avec le bon compte.
    #[test]
    fn cycle_shortcut_ends_interior_points_early() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).unwrap();
        let pts: Vec<Complex64> = (0..64)
            .map(|k| Complex64::new(-0.2 + 0.005 * k as f64, 0.1))
            .collect();
        let max = 10_000_000u32;
        let mut it = vec![0u32; pts.len()];
        let mut zs = vec![Complex64::new(0.0, 0.0); pts.len()];
        let t = std::time::Instant::now();
        iterate_batch_impl(
            &formula.phases[0],
            &pts,
            Complex64::new(0.0, 0.0),
            false,
            max,
            25.0,
            &mut it,
            &mut zs,
            true,
        );
        assert!(it.iter().all(|&i| i == max), "points intérieurs : plafond attendu");
        assert!(
            t.elapsed().as_secs_f64() < 2.0,
            "raccourci inopérant : {:.1} s pour 64 points intérieurs",
            t.elapsed().as_secs_f64()
        );
    }

    /// Lots plus petits que le nombre de voies, vides, et multiples exacts.
    #[test]
    fn batch_handles_every_batch_size() {
        let formula = compile_formula(FractalType::Mandelbrot, 2.0).expect("mandelbrot");
        let points = probe_points(40, 42);
        for len in [0usize, 1, 2, LANES - 1, LANES, LANES + 1, 2 * LANES, 37] {
            assert_batch_matches_scalar(
                &format!("len={len}"),
                &formula,
                &points[..len],
                Complex64::new(0.0, 0.0),
                false,
                500,
                25.0,
            );
        }
    }
}
