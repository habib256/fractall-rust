//! Benchmarks machine persistés (G9.2) — modèle Fraktaler-3 `wisdom.cc`
//! (`wisdom_benchmark_device`, `wisdom.cc:393`) adapté :
//!
//! - F3 mesure `pixels/seconde` par (device × type numérique) sur une vue
//!   standard, en grossissant la frame jusqu'à une durée cible, et persiste le
//!   résultat en TOML par machine ; `wisdom_lookup` départage ensuite les
//!   candidats VIABLES par vitesse mesurée.
//! - fractall mesure des **iters/seconde effectifs** (Σ iterations / wall) sur
//!   des rendus RÉELS passant par le dispatcher UNIQUE (`render_escape_time`) —
//!   les skips BLA/harmonic, le build d'orbite et le parallélisme rayon sont
//!   inclus : c'est le débit que verra une frame réelle, l'unité qui permet de
//!   départager deux devices sur la même classe de frame.
//!
//! Une entrée par **technique** (`BenchKey`) : les tiers f64/exp ne se
//! concurrencent jamais (viabilité d'exposant), mais chaque entrée donne au
//! plan une estimation de débit, et les paires comparables (CpuStdF64 vs
//! GpuStdF32 aujourd'hui ; kernel GPU deep vs CPU en G9.4/9.5) partagent la
//! même frame de bench. Consommateur actuel : `WisdomPlan.bench_iters_per_sec`
//! (log `[WISDOM]`) ; l'arbitrage device = jalon G9.5.
//!
//! Persistance : `~/.config/fractall/wisdom.toml` (override
//! `FRACTALL_WISDOM_FILE`), régénéré par `fractall-cli --wisdom-bench`.
//! Jamais mesuré implicitement au premier rendu (un bench = plusieurs
//! secondes ; décision explicite de l'utilisateur, comme le `-b` de F3).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::fractal::types::{AlgorithmMode, FractalType};
use crate::fractal::{default_params_for_type, FractalParams};
use crate::fractal::wisdom::{Algorithm, Device, NumberTier};

/// Technique benchée. Chaîne stable = clé du TOML (compat ascendante).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchKey {
    /// CPU f64 standard (vue par défaut, sans perturbation).
    CpuStdF64,
    /// CPU perturbation tier f64 (deep e50).
    CpuPerturbF64,
    /// CPU perturbation tier ComplexExp (deep e318).
    CpuPerturbExp,
    /// CPU perturbation tier double-double (deep e50, `--dd-tier`).
    CpuPerturbDd,
    /// GPU shader f32 standard (même vue que [`BenchKey::CpuStdF64`] —
    /// directement comparable pour l'arbitrage device G9.5). ⚠️ **f32 = 24 b
    /// de mantisse → FAUX sur les fractales escape-time** (leçon F3 : 9391 px
    /// faux) : l'arbitrage device N'utilise JAMAIS ce débit (le GPU n'est
    /// correct que via le kernel perturbation f64, cf. [`GpuPerturbF64`]).
    GpuStdF32,
    /// GPU kernel perturbation f64 natif (`perturbation.wgsl`, SHADER_F64,
    /// G9.4). Le SEUL path GPU correct pour la plage perturbation (≥ ~1e5) —
    /// c'est le débit consommé par l'arbitrage device G9.5
    /// (`wisdom::select_device`). Absent du wisdom.toml tant que
    /// `--wisdom-bench` ne le mesure pas → l'arbitrage retombe conservativement
    /// sur le CPU (jalon suivant : mesure GPU-perturb dans `run_bench`).
    GpuPerturbF64,
}

impl BenchKey {
    pub const ALL: [BenchKey; 6] = [
        BenchKey::CpuStdF64,
        BenchKey::CpuPerturbF64,
        BenchKey::CpuPerturbExp,
        BenchKey::CpuPerturbDd,
        BenchKey::GpuStdF32,
        BenchKey::GpuPerturbF64,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            BenchKey::CpuStdF64 => "cpu_std_f64",
            BenchKey::CpuPerturbF64 => "cpu_perturb_f64",
            BenchKey::CpuPerturbExp => "cpu_perturb_exp",
            BenchKey::CpuPerturbDd => "cpu_perturb_dd",
            BenchKey::GpuStdF32 => "gpu_std_f32",
            BenchKey::GpuPerturbF64 => "gpu_perturb_f64",
        }
    }

    fn from_str(s: &str) -> Option<BenchKey> {
        BenchKey::ALL.iter().copied().find(|k| k.as_str() == s)
    }

    /// Clé correspondant à un plan (device, algorithme, tier). `None` si la
    /// technique n'est pas benchée (GMP par-pixel, types spéciaux).
    pub fn for_plan(
        device: Device,
        algorithm: Algorithm,
        tier: Option<NumberTier>,
    ) -> Option<BenchKey> {
        match (device, algorithm) {
            (Device::Cpu, Algorithm::StandardF64) => Some(BenchKey::CpuStdF64),
            (Device::Cpu, Algorithm::Perturbation) => match tier {
                Some(NumberTier::F64) => Some(BenchKey::CpuPerturbF64),
                Some(NumberTier::Exp) => Some(BenchKey::CpuPerturbExp),
                Some(NumberTier::Dd) => Some(BenchKey::CpuPerturbDd),
                None => None,
            },
            (Device::Gpu, Algorithm::StandardF64) => Some(BenchKey::GpuStdF32),
            // GPU perturbation = kernel f64 natif (pas de tier CPU-style : la
            // hiérarchie exp/dd est propre au CPU). Clé unique.
            (Device::Gpu, Algorithm::Perturbation) => Some(BenchKey::GpuPerturbF64),
            _ => None,
        }
    }
}

/// Une mesure persistée.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchEntry {
    /// `BenchKey::as_str` — clé stable.
    pub key: String,
    /// Débit effectif : Σ iterations / wall-clock du rendu complet.
    pub iters_per_sec: f64,
    /// Contexte de la mesure (diagnostic ; non consommé).
    pub width: u32,
    pub height: u32,
    pub seconds: f64,
    /// Epoch Unix (s) de la mesure.
    pub measured_unix: u64,
}

/// Fichier wisdom par machine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct WisdomBenchFile {
    pub version: u32,
    /// Descriptif machine (informatif — le fichier est déjà par machine).
    pub machine: String,
    #[serde(default)]
    pub bench: Vec<BenchEntry>,
}

impl WisdomBenchFile {
    pub fn lookup(&self, key: BenchKey) -> Option<&BenchEntry> {
        self.bench.iter().find(|e| e.key == key.as_str())
    }
}

/// Chemin du fichier wisdom : `FRACTALL_WISDOM_FILE` sinon
/// `~/.config/fractall/wisdom.toml`. `None` si indéterminable (pas de HOME).
pub fn bench_file_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FRACTALL_WISDOM_FILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".config/fractall/wisdom.toml"))
}

/// Charge le fichier wisdom (une fois par process). `None` si absent/illisible
/// — le plan reste viabilité-seule, aucun rendu n'en dépend.
pub fn cached() -> Option<&'static WisdomBenchFile> {
    static CACHE: OnceLock<Option<WisdomBenchFile>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let path = bench_file_path()?;
            let text = std::fs::read_to_string(path).ok()?;
            toml::from_str(&text).ok()
        })
        .as_ref()
}

/// Débit benché pour un plan, si mesuré sur cette machine.
pub fn lookup_iters_per_sec(
    device: Device,
    algorithm: Algorithm,
    tier: Option<NumberTier>,
) -> Option<f64> {
    let key = BenchKey::for_plan(device, algorithm, tier)?;
    Some(cached()?.lookup(key)?.iters_per_sec)
}

// ── Frames de bench ─────────────────────────────────────────────────────────
// Coordonnées de vues STRUCTURÉES connues (corpus, validées vs GMP) : le débit
// mesuré doit refléter un vrai travail par pixel, pas un extérieur uniforme.

/// Deep e50 (toml/e50.toml) — spirale, tier f64 (pixel ~1e-52), orbite 86 k.
const E50_RE: &str = "-0.0494700290631040937516922267273536301187457124882248793181049402326421947726869034279915499747594190000000000000000000";
const E50_IM: &str = "-0.6747875758446753640113920531305976563347707068224034806979997947909941983454845111514208499540310299999999999999999880";

/// Deep e318 (toml/e318.toml) — tier ComplexExp (pixel ~1e-321), orbite 567 k.
const E318_RE: &str = "-0.160671574544376235234299806562325714463713365698322466075294476769843023909577654790769037756572078750948504637062800069784283006963539070396938476002191397367130415091653379715775734728331195648418196184720437315573495294645574770752080340338379190322084091695814872716225514834195509029187461868448533384966001341735332278497598401959";
const E318_IM: &str = "1.0369425716116291171688106641088455988538567298124323080597903890054138172127708543024909014955338074002178250818737097944716605548909996311034876508441714228244078222420638636408073445850546946976971476583353238902945581912155462907695065826318848338789529151847288915828593504988849176913597129353591778847015303039106385284417320876575";

fn deep_frame(re: &str, im: &str, span_hp: &str, iterations: u32, size: u32) -> FractalParams {
    let mut p = default_params_for_type(FractalType::Mandelbrot, size, size);
    p.center_x_hp = Some(re.to_string());
    p.center_y_hp = Some(im.to_string());
    p.center_x = re.parse().unwrap_or(0.0);
    p.center_y = im.parse().unwrap_or(0.0);
    // Underflow f64 possible (e318) : les spans HP font foi (HP-aware helpers).
    let span: f64 = span_hp.parse().unwrap_or(0.0);
    p.span_x = span;
    p.span_y = span;
    p.span_x_hp = Some(span_hp.to_string());
    p.span_y_hp = Some(span_hp.to_string());
    p.iteration_max = iterations;
    p.algorithm_mode = AlgorithmMode::Auto;
    p
}

/// Frame d'un `BenchKey` à une taille donnée. La vue par défaut (std f64/f32)
/// est PARTAGÉE entre CPU et GPU — comparabilité directe.
fn bench_frame(key: BenchKey, size: u32) -> FractalParams {
    match key {
        BenchKey::CpuStdF64 | BenchKey::GpuStdF32 => {
            let mut p = default_params_for_type(FractalType::Mandelbrot, size, size);
            p.iteration_max = 2000;
            p.algorithm_mode = AlgorithmMode::Auto;
            p
        }
        // GPU-perturb : même vue deep e50 que le CPU-perturb f64 → débits
        // directement comparables pour l'arbitrage device (G9.5). Frame
        // seulement — la MESURE GPU-perturb (rendu via `render_dispatch`) est le
        // jalon suivant ; aujourd'hui `measure_cpu` ne benche pas cette clé.
        BenchKey::CpuPerturbF64 | BenchKey::GpuPerturbF64 => {
            deep_frame(E50_RE, E50_IM, "4e-50", 263_010, size)
        }
        BenchKey::CpuPerturbExp => deep_frame(E318_RE, E318_IM, "1e-318", 212_138, size),
        BenchKey::CpuPerturbDd => {
            let mut p = deep_frame(E50_RE, E50_IM, "4e-50", 263_010, size);
            p.use_dd_tier = true;
            p
        }
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Mesure une technique CPU : rendus via le dispatcher UNIQUE, taille doublée
/// jusqu'à `target_seconds` (F3 fait pareil jusqu'à 10 s ; on vise ~1.5 s par
/// technique pour garder `--wisdom-bench` sous la minute). Retourne `None` si
/// le rendu ne produit rien (technique indisponible).
fn measure_cpu(key: BenchKey, target_seconds: f64) -> Option<BenchEntry> {
    let mut size = 128u32;
    let mut best: Option<BenchEntry> = None;
    loop {
        let params = bench_frame(key, size);
        let t = Instant::now();
        let (iters, _zs) = crate::render::escape_time::render_escape_time(&params);
        let seconds = t.elapsed().as_secs_f64();
        if iters.is_empty() {
            return best;
        }
        let total: u64 = iters.iter().map(|&i| i as u64).sum();
        best = Some(BenchEntry {
            key: key.as_str().to_string(),
            iters_per_sec: total as f64 / seconds.max(1e-9),
            width: size,
            height: size,
            seconds,
            measured_unix: unix_now(),
        });
        if seconds >= target_seconds || size >= 1024 {
            return best;
        }
        size *= 2;
    }
}

/// Lance le bench CPU complet (toutes les techniques CPU) + l'éventuel bench
/// GPU fourni par le caller (le CLI construit le `GpuRenderer` et passe une
/// closure — ce module ne dépend pas de `gpu/`). Retourne le fichier persisté.
pub fn run_bench(
    gpu_std: Option<&dyn Fn(&FractalParams) -> Option<Vec<u32>>>,
    target_seconds: f64,
) -> WisdomBenchFile {
    let mut file = WisdomBenchFile {
        version: 1,
        machine: machine_descr(),
        bench: Vec::new(),
    };
    for key in [
        BenchKey::CpuStdF64,
        BenchKey::CpuPerturbF64,
        BenchKey::CpuPerturbExp,
        BenchKey::CpuPerturbDd,
    ] {
        eprintln!("[WISDOM-BENCH] {} …", key.as_str());
        if let Some(entry) = measure_cpu(key, target_seconds) {
            eprintln!(
                "[WISDOM-BENCH] {} : {:.3e} iters/s ({}x{}, {:.2}s)",
                key.as_str(), entry.iters_per_sec, entry.width, entry.height, entry.seconds
            );
            file.bench.push(entry);
        }
    }
    if let Some(render) = gpu_std {
        let key = BenchKey::GpuStdF32;
        eprintln!("[WISDOM-BENCH] {} …", key.as_str());
        let mut size = 256u32;
        let mut best: Option<BenchEntry> = None;
        loop {
            let params = bench_frame(key, size);
            let t = Instant::now();
            let Some(iters) = render(&params) else { break };
            let seconds = t.elapsed().as_secs_f64();
            let total: u64 = iters.iter().map(|&i| i as u64).sum();
            best = Some(BenchEntry {
                key: key.as_str().to_string(),
                iters_per_sec: total as f64 / seconds.max(1e-9),
                width: size,
                height: size,
                seconds,
                measured_unix: unix_now(),
            });
            if seconds >= target_seconds || size >= 2048 {
                break;
            }
            size *= 2;
        }
        if let Some(entry) = best {
            eprintln!(
                "[WISDOM-BENCH] {} : {:.3e} iters/s ({}x{}, {:.2}s)",
                key.as_str(), entry.iters_per_sec, entry.width, entry.height, entry.seconds
            );
            file.bench.push(entry);
        }
    }
    file
}

/// Écrit le fichier wisdom (répertoires créés). Erreur = message, pas de panic.
pub fn save(file: &WisdomBenchFile) -> Result<PathBuf, String> {
    let path = bench_file_path().ok_or("chemin wisdom indéterminable (pas de HOME)")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    }
    let text = toml::to_string_pretty(file).map_err(|e| format!("sérialisation TOML: {e}"))?;
    std::fs::write(&path, text).map_err(|e| format!("écriture {}: {e}", path.display()))?;
    Ok(path)
}

fn machine_descr() -> String {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    format!("{model} · {threads} threads")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_round_trip_and_lookup() {
        let file = WisdomBenchFile {
            version: 1,
            machine: "test".into(),
            bench: vec![BenchEntry {
                key: BenchKey::CpuPerturbF64.as_str().into(),
                iters_per_sec: 1.25e9,
                width: 256,
                height: 256,
                seconds: 1.5,
                measured_unix: 0,
            }],
        };
        let text = toml::to_string_pretty(&file).expect("toml");
        let back: WisdomBenchFile = toml::from_str(&text).expect("parse");
        assert_eq!(back, file);
        assert_eq!(
            back.lookup(BenchKey::CpuPerturbF64).map(|e| e.iters_per_sec),
            Some(1.25e9)
        );
        assert_eq!(back.lookup(BenchKey::GpuStdF32), None);
    }

    #[test]
    fn bench_key_plan_mapping() {
        // Le mapping plan → clé couvre les techniques benchées et refuse GMP.
        assert_eq!(
            BenchKey::for_plan(Device::Cpu, Algorithm::Perturbation, Some(NumberTier::Exp)),
            Some(BenchKey::CpuPerturbExp)
        );
        assert_eq!(
            BenchKey::for_plan(Device::Gpu, Algorithm::StandardF64, None),
            Some(BenchKey::GpuStdF32)
        );
        assert_eq!(BenchKey::for_plan(Device::Cpu, Algorithm::ReferenceGmp, None), None);
        // Chaînes stables (clé TOML) : round-trip.
        for k in BenchKey::ALL {
            assert_eq!(BenchKey::from_str(k.as_str()), Some(k));
        }
    }

    #[test]
    fn bench_frames_route_to_expected_techniques() {
        // Chaque frame de bench doit exercer LA technique de sa clé (sinon le
        // débit mesuré étiquette la mauvaise entrée).
        use crate::fractal::wisdom;
        let f = bench_frame(BenchKey::CpuStdF64, 128);
        assert_eq!(wisdom::select_algorithm(&f, Device::Cpu), Algorithm::StandardF64);
        let f = bench_frame(BenchKey::CpuPerturbF64, 128);
        assert_eq!(wisdom::select_algorithm(&f, Device::Cpu), Algorithm::Perturbation);
        assert_eq!(wisdom::number_tier(&f), Some(NumberTier::F64));
        let f = bench_frame(BenchKey::CpuPerturbExp, 128);
        assert_eq!(wisdom::select_algorithm(&f, Device::Cpu), Algorithm::Perturbation);
        assert_eq!(wisdom::number_tier(&f), Some(NumberTier::Exp));
        let f = bench_frame(BenchKey::CpuPerturbDd, 128);
        assert_eq!(wisdom::number_tier(&f), Some(NumberTier::Dd));
    }
}
