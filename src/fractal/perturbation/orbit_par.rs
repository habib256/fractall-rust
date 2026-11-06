//! Itération GMP de l'orbite de référence sur DEUX cœurs (G14.3).
//!
//! L'orbite de référence est le seul travail strictement séquentiel du
//! rendu : pendant qu'elle tourne, tous les autres cœurs attendent. Sur les
//! zooms très profonds c'est elle qui fait le temps (corpus à 64², Apple M4 :
//! triangle 14 s sur 17, wfs_extended 16 s sur 19, orion 6,5 s sur 10). Or
//! chaque itération `fast_sqr_step` fait deux multiplications pleine
//! précision INDÉPENDANTES :
//!
//! ```text
//! re' = (re+im)·(re−im) + cr        im' = 2·re·im + ci
//! ```
//!
//! [`ParSqr`] confie `im'` à un fil assistant pendant que le fil appelant
//! calcule `(re+im)·(re−im)`. Ce sont les mêmes opérations MPFR, aux mêmes
//! précisions, dans le même ordre par composante : l'orbite est
//! **identique au bit près** à celle de `fast_sqr_step` (verrou
//! `par_sqr_matches_serial_step_bit_for_bit`), donc aussi à sa relecture
//! `lazy_low_parts`.
//!
//! Synchronisation : une itération dure ~1 µs, un réveil de fil 10 à 50 µs —
//! l'assistant ATTEND ACTIVEMENT entre deux tâches. Aucun `unsafe` : les
//! `Float` changent de propriétaire par une case (`Mutex<Option<…>>`, jamais
//! disputée — les accès alternent).
//!
//! ⚠️ **Une tâche se RÉCLAME par un échange atomique** (`POSTED → TAKEN` pour
//! l'assistant, `POSTED → IDLE` pour la reprise par l'appelant) : celui qui
//! gagne l'échange, et lui seul, vide la case. La première version laissait
//! l'assistant vider la case sur la foi d'un `POSTED` LU plus tôt : retardé
//! entre la lecture et la case, il pouvait prendre la tâche SUIVANTE avant
//! que l'appelant n'ait écrit son `POSTED`, la finir, écrire `DONE` — que le
//! `POSTED` tardif écrasait : appelant en attente de `DONE` pour toujours,
//! deux cœurs à 100 %, rendu bloqué (trouvé à la relecture, 2026-11-06,
//! reproduit sur modèle avec deux retards injectés ; protocole exercé sous
//! préemption par `claimed_handoff_survives_preemption_at_every_step`).
//!
//! Garde-fous :
//! - **reprise** : si l'assistant n'a pas encore pris la tâche quand le fil
//!   appelant a fini sa multiplication, celui-ci la reprend et la fait
//!   lui-même — machine saturée, on retombe au coût série plus deux verrous ;
//! - **mise à l'écart** : une attente anormalement longue (assistant
//!   désordonnancé en pleine multiplication) ou des reprises en rafale
//!   renvoient au pas série, assistant ENDORMI (`park`), pour une fenêtre
//!   qui double à chaque récidive (`BACKOFF_ITERS` → `BACKOFF_MAX`) : sur
//!   une machine saturée le coût converge vers celui de la série ;
//! - **seuil de précision** (`FRACTALL_ORBIT_PAR_MIN_BITS`) : en dessous, la
//!   synchronisation coûte plus que la multiplication épargnée ;
//! - `FRACTALL_ORBIT_PAR=0` coupe tout (A/B).

use rug::{Assign, Complex, Float};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

/// Précision minimale (bits) par défaut à partir de laquelle l'assistant est
/// lancé. Calibré sur Apple M4 (`examples/bench_orbit_join.rs`).
pub const PAR_MIN_BITS_DEFAULT: u32 = 2048;

/// Première fenêtre série après une attente anormale ou une rafale de
/// reprises ; doublée à chaque récidive jusqu'à `BACKOFF_MAX`, ramenée ici
/// après une fenêtre parallèle sans incident.
const BACKOFF_ITERS: u64 = 4_096;
const BACKOFF_MAX: u64 = 1 << 20;
/// Tours d'attente active avant de regarder l'horloge (quelques µs).
const SPINS_BEFORE_CLOCK: u32 = 2_000;
/// Reprises consécutives au-delà desquelles l'assistant est jugé affamé.
const STARVED_STEALS: u32 = 4_096;

const IDLE: u32 = 0;
const POSTED: u32 = 1;
const DONE: u32 = 2;
const QUIT: u32 = 3;
/// Tâche réclamée par l'assistant, calcul en cours.
const TAKEN: u32 = 5;
/// Assistant endormi pendant une fenêtre série.
const PARKED: u32 = 4;

fn par_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FRACTALL_ORBIT_PAR").ok().as_deref().map(str::trim),
            Some("0") | Some("off")
        )
    })
}

fn par_min_bits() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FRACTALL_ORBIT_PAR_MIN_BITS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(PAR_MIN_BITS_DEFAULT)
    })
}

struct Shared {
    /// Tâche postée : `(im, re)` à l'entrée du pas.
    job: Mutex<Option<(Float, Float)>>,
    /// Tâche faite : `(im', re)`.
    result: Mutex<Option<(Float, Float)>>,
    state: AtomicU32,
    /// Partie imaginaire de `c` (constante de l'orbite).
    ci: Float,
}

/// `im' = 2·im·re + ci` — la moitié « imaginaire » de `fast_sqr_step`.
#[inline]
fn imag_half(im: &mut Float, re: &Float, ci: &Float) {
    *im *= re;
    *im <<= 1;
    *im += ci;
}

fn helper_loop(sh: &Shared) {
    let mut spins = 0u32;
    loop {
        match sh.state.load(Ordering::Acquire) {
            QUIT => return,
            PARKED => {
                // Fenêtre série : dormir jusqu'au `unpark` (réveils parasites
                // tolérés : l'état est relu).
                std::thread::park();
                spins = 0;
                continue;
            }
            POSTED => {
                // Réclamation atomique : la lecture ci-dessus peut être
                // périmée (tâche reprise entre-temps, tâche suivante en cours
                // de dépôt). Seul le gagnant de l'échange touche la case.
                if sh
                    .state
                    .compare_exchange(POSTED, TAKEN, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    test_delay(DelayPoint::HelperClaimed);
                    let (mut im, re) = sh
                        .job
                        .lock()
                        .unwrap()
                        .take()
                        .expect("tâche réclamée mais case vide");
                    imag_half(&mut im, &re, &sh.ci);
                    *sh.result.lock().unwrap() = Some((im, re));
                    sh.state.store(DONE, Ordering::Release);
                    spins = 0;
                    continue;
                }
            }
            _ => {}
        }
        test_delay(DelayPoint::HelperIdle);
        spins = spins.saturating_add(1);
        if spins < 50_000 {
            std::hint::spin_loop();
        } else {
            // Fil appelant occupé ailleurs : rendre la main.
            std::thread::yield_now();
        }
    }
}

/// Points où les tests injectent un retard (simulation d'une préemption).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
enum DelayPoint {
    HelperIdle,
    HelperClaimed,
    CallerBetweenJobAndPost,
    CallerAfterPost,
    CallerBeforeReclaim,
}

#[cfg(not(test))]
#[inline(always)]
fn test_delay(_point: DelayPoint) {}

#[cfg(test)]
fn test_delay(point: DelayPoint) {
    let (target, every, micros) = *TEST_DELAY.lock().unwrap();
    if target == Some(point) {
        let n = TEST_DELAY_COUNT.fetch_add(1, Ordering::Relaxed);
        if n % every == 0 {
            std::thread::sleep(std::time::Duration::from_micros(micros));
        }
    }
}

#[cfg(test)]
static TEST_DELAY: Mutex<(Option<DelayPoint>, u32, u64)> = Mutex::new((None, 1, 0));
#[cfg(test)]
static TEST_DELAY_COUNT: AtomicU32 = AtomicU32::new(0);

/// Pas d'orbite `z ← z² + c` sur deux cœurs, bit-identique à
/// `orbit::fast_sqr_step`.
pub(super) struct ParSqr {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
    /// Tampon du produit `(re+im)(re−im)`, à la précision de l'orbite.
    p: Float,
    /// Deux `Float` de substitution logés dans `z` pendant que ses
    /// composantes sont chez l'assistant (aucune allocation par itération).
    spare_re: Option<Float>,
    spare_im: Option<Float>,
    iter: u64,
    /// Fin de la fenêtre série en cours (assistant endormi) ; 0 = aucune.
    serial_until: u64,
    /// Longueur de la prochaine fenêtre série.
    backoff: u64,
    /// Itération à partir de laquelle une fenêtre parallèle sans incident
    /// ramène `backoff` à son plancher.
    clean_since: u64,
    steals_in_a_row: u32,
    /// Attente au-delà de laquelle l'assistant est jugé désordonnancé.
    slow_wait: std::time::Duration,
}

impl ParSqr {
    /// Assistant pour une orbite à `prec` bits autour de `cref`, ou `None`
    /// si le dispositif ne s'applique pas (coupé, précision sous le seuil,
    /// machine à un seul cœur).
    pub(super) fn new(cref: &Complex, prec: u32) -> Option<Self> {
        if !par_enabled() || prec < par_min_bits() {
            return None;
        }
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        if cores < 2 || rayon::current_num_threads() < 2 {
            return None;
        }
        Self::spawn(cref, prec)
    }

    /// Sans les conditions d'entrée (tests, banc).
    pub(super) fn spawn(cref: &Complex, prec: u32) -> Option<Self> {
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            result: Mutex::new(None),
            state: AtomicU32::new(IDLE),
            ci: cref.imag().clone(),
        });
        let sh = Arc::clone(&shared);
        let handle = std::thread::Builder::new()
            .name("fractall-orbit-par".into())
            .spawn(move || helper_loop(&sh))
            .ok()?;
        Some(Self {
            shared,
            handle: Some(handle),
            p: Float::new(prec),
            spare_re: Some(Float::new(2)),
            spare_im: Some(Float::new(2)),
            iter: 0,
            serial_until: 0,
            backoff: BACKOFF_ITERS,
            clean_since: 0,
            steals_in_a_row: 0,
            // ~30 µs + 2 ns par bit : large devant le décalage normal entre
            // deux multiplications de même taille, court devant un quantum.
            slow_wait: std::time::Duration::from_nanos(30_000 + 2 * prec as u64),
        })
    }

    /// Ouvre une fenêtre série : l'assistant s'endort, la fenêtre suivante
    /// sera deux fois plus longue.
    #[cold]
    fn back_off(&mut self) {
        self.serial_until = self.iter + self.backoff;
        self.backoff = (self.backoff * 2).min(BACKOFF_MAX);
        self.steals_in_a_row = 0;
        self.shared.state.store(PARKED, Ordering::Release);
    }

    /// Un pas `z ← z² + cref`. `t1`/`t2` : tampons à la précision de `z`
    /// (ceux de `fast_sqr_step`).
    #[inline]
    pub(super) fn step(&mut self, z: &mut Complex, cref: &Complex, t1: &mut Float, t2: &mut Float) {
        self.iter += 1;
        if self.serial_until != 0 {
            if self.iter < self.serial_until {
                super::orbit::fast_sqr_step(z, cref, t1, t2);
                return;
            }
            // Fin de fenêtre série : réveil de l'assistant.
            self.serial_until = 0;
            self.clean_since = self.iter;
            self.shared.state.store(IDLE, Ordering::Release);
            if let Some(h) = self.handle.as_ref() {
                h.thread().unpark();
            }
        } else if self.backoff > BACKOFF_ITERS && self.iter - self.clean_since > 4 * self.backoff {
            self.backoff = BACKOFF_ITERS;
        }
        let sh = &*self.shared;
        let (re, im) = z.as_mut_real_imag();
        t1.assign(&*re + &*im);
        t2.assign(&*re - &*im);
        // Les composantes quittent `z` (remplacées par les substituts) et
        // partent dans la case de tâche.
        let mut job_re = self.spare_re.take().expect("substitut re");
        let mut job_im = self.spare_im.take().expect("substitut im");
        std::mem::swap(re, &mut job_re);
        std::mem::swap(im, &mut job_im);
        // État IDLE ici (seul l'appelant l'écrit, et il vient de le faire) :
        // l'assistant ne peut rien réclamer avant le `POSTED` ci-dessous, et
        // la case est pleine quand il le voit.
        *sh.job.lock().unwrap() = Some((job_im, job_re));
        test_delay(DelayPoint::CallerBetweenJobAndPost);
        sh.state.store(POSTED, Ordering::Release);
        test_delay(DelayPoint::CallerAfterPost);

        self.p.assign(&*t1 * &*t2);

        // Reprise si l'assistant n'a pas RÉCLAMÉ la tâche (échange atomique :
        // un seul des deux fils vide la case) ; sinon attente active.
        test_delay(DelayPoint::CallerBeforeReclaim);
        let mut incident = false;
        let reclaimed = sh
            .state
            .compare_exchange(POSTED, IDLE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        let (mut new_im, mut old_re) = if reclaimed {
            let (mut i2, r2) = sh
                .job
                .lock()
                .unwrap()
                .take()
                .expect("tâche reprise mais case vide");
            imag_half(&mut i2, &r2, &sh.ci);
            self.steals_in_a_row += 1;
            incident = self.steals_in_a_row >= STARVED_STEALS;
            (i2, r2)
        } else {
            self.steals_in_a_row = 0;
            let mut spins = 0u32;
            let mut since: Option<std::time::Instant> = None;
            while sh.state.load(Ordering::Acquire) != DONE {
                spins = spins.wrapping_add(1);
                if spins < SPINS_BEFORE_CLOCK {
                    std::hint::spin_loop();
                } else if spins % 256 == 0 {
                    let t0 = *since.get_or_insert_with(std::time::Instant::now);
                    if t0.elapsed() > self.slow_wait {
                        incident = true;
                        // Assistant mort (panique) : mieux vaut une erreur
                        // franche qu'une attente sans fin.
                        if self.handle.as_ref().is_some_and(|h| h.is_finished()) {
                            panic!("orbite sur deux cœurs : le fil assistant s'est arrêté en pleine tâche");
                        }
                        std::thread::yield_now();
                    }
                } else {
                    std::hint::spin_loop();
                }
            }
            let r = sh.result.lock().unwrap().take().expect("résultat de l'assistant");
            sh.state.store(IDLE, Ordering::Release);
            r
        };
        if incident {
            self.back_off();
        }

        // Retour dans `z` : im ← im', re ← produit ; l'ancien tampon de `re`
        // (à la précision de l'orbite) devient le tampon produit suivant.
        std::mem::swap(im, &mut new_im);
        std::mem::swap(re, &mut self.p);
        std::mem::swap(&mut self.p, &mut old_re);
        self.spare_im = Some(new_im);
        self.spare_re = Some(old_re);
        *re += cref.real();
    }
}

impl Drop for ParSqr {
    fn drop(&mut self) {
        self.shared.state.store(QUIT, Ordering::Release);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L'orbite sur deux cœurs est celle de `fast_sqr_step` au bit près, à
    /// chaque itération — y compris quand des pas série s'intercalent
    /// (mise à l'écart) et pour une orbite qui s'évade (valeurs énormes).
    #[test]
    fn par_sqr_matches_serial_step_bit_for_bit() {
        for &(prec, cr, ci, iters) in &[
            (192u32, -0.1234567891234, 0.6543219876543, 3000u32),
            (1024, -1.7499, 1e-9, 2000),
            (4096, -0.1234567891234, 0.6543219876543, 600),
            // Extérieur : évasion puis croissance jusqu'à l'infini MPFR.
            (512, 0.4, 0.4, 40),
        ] {
            let c = Complex::with_val(prec, (cr, ci));
            let mut zs = Complex::with_val(prec, (0.0, 0.0));
            let mut zp = Complex::with_val(prec, (0.0, 0.0));
            let (mut s1, mut s2) = (Float::new(prec), Float::new(prec));
            let (mut p1, mut p2) = (Float::new(prec), Float::new(prec));
            let mut par = ParSqr::spawn(&c, prec).expect("assistant");
            for i in 0..iters {
                // Force des fenêtres série (assistant endormi puis réveillé)
                // au milieu de l'orbite.
                if i == iters / 3 || i == iters / 2 {
                    par.back_off();
                    par.serial_until = par.iter + 50;
                }
                crate::fractal::perturbation::orbit::fast_sqr_step(&mut zs, &c, &mut s1, &mut s2);
                par.step(&mut zp, &c, &mut p1, &mut p2);
                let same = |a: &Float, b: &Float| {
                    (a.is_nan() && b.is_nan()) || (a == b && a.prec() == b.prec())
                };
                assert!(
                    same(zs.real(), zp.real()) && same(zs.imag(), zp.imag()),
                    "orbite ≠ à l'itération {i} (prec {prec})"
                );
            }
        }
    }

    /// Le passage de tâche tient quand l'un ou l'autre fil est retardé à
    /// N'IMPORTE quel point du protocole (préemption simulée) : pas de
    /// blocage, orbite toujours identique au bit près. (Le blocage de la
    /// première version demandait DEUX retards conjoints ; il est exclu ici
    /// par construction — la case n'est vidée que par le gagnant d'un échange
    /// atomique — et ce test exerce le protocole sous préemption, point par
    /// point, y compris la reprise de tâche et l'attente de l'assistant.)
    #[test]
    fn claimed_handoff_survives_preemption_at_every_step() {
        // Un seul test à la fois règle les retards (état global de test).
        static SERIAL: Mutex<()> = Mutex::new(());
        let _guard = SERIAL.lock().unwrap();
        let prec = 256u32;
        let c = Complex::with_val(prec, (-0.1234567891234, 0.6543219876543));
        let points = [
            DelayPoint::HelperIdle,
            DelayPoint::HelperClaimed,
            DelayPoint::CallerBetweenJobAndPost,
            DelayPoint::CallerAfterPost,
            DelayPoint::CallerBeforeReclaim,
        ];
        for &point in &points {
            for &(every, micros) in &[(1u32, 30u64), (3, 200), (7, 5)] {
                *TEST_DELAY.lock().unwrap() = (Some(point), every, micros);
                let (done_tx, done_rx) = std::sync::mpsc::channel();
                let c2 = c.clone();
                let worker = std::thread::spawn(move || {
                    let mut zs = Complex::with_val(prec, (0.0, 0.0));
                    let mut zp = Complex::with_val(prec, (0.0, 0.0));
                    let (mut s1, mut s2) = (Float::new(prec), Float::new(prec));
                    let (mut p1, mut p2) = (Float::new(prec), Float::new(prec));
                    let mut par = ParSqr::spawn(&c2, prec).expect("assistant");
                    let mut ok = true;
                    for _ in 0..400 {
                        crate::fractal::perturbation::orbit::fast_sqr_step(&mut zs, &c2, &mut s1, &mut s2);
                        par.step(&mut zp, &c2, &mut p1, &mut p2);
                        ok &= zs == zp;
                    }
                    let _ = done_tx.send(ok);
                });
                let ok = done_rx
                    .recv_timeout(std::time::Duration::from_secs(20))
                    .unwrap_or_else(|_| panic!("BLOCAGE avec retard à {point:?} (1/{every}, {micros} µs)"));
                *TEST_DELAY.lock().unwrap() = (None, 1, 0);
                worker.join().unwrap();
                assert!(ok, "orbite ≠ avec retard à {point:?}");
            }
        }
    }

    /// Le fil assistant se termine quand `ParSqr` est lâché, même sans avoir
    /// jamais reçu de tâche.
    #[test]
    fn helper_thread_stops_on_drop() {
        let c = Complex::with_val(256, (-0.5, 0.1));
        let par = ParSqr::spawn(&c, 256).expect("assistant");
        drop(par);
        // … et même endormi dans une fenêtre série.
        let mut par = ParSqr::spawn(&c, 256).expect("assistant");
        par.back_off();
        std::thread::sleep(std::time::Duration::from_millis(5));
        drop(par);
    }
}
