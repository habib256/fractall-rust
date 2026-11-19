#!/usr/bin/env python3
"""Verrou du garde-fou crash : réconciliation journal↔quarantaine.

Régression couverte : un cas `died_uncleanly` dans le journal append-only qui
n'est PAS en quarantaine (quarantine.json dérivé/reverté) DOIT être re-exclu des
sweeps (invariant « crash journalisé ⇒ quarantainé »), sans réintégrer les cas
résolus manuellement ni les interruptions gracieuses.

Lancer : python3 scripts/test_harness_guard.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harnesslib import guard  # noqa: E402
from harnesslib import gaps as gaps_mod  # noqa: E402
from harnesslib import axes_perf  # noqa: E402
from harnesslib import config  # noqa: E402
from harnesslib import cli as harnesslib_cli  # noqa: E402
from harnesslib import axes_precision  # noqa: E402
from harnesslib import hotspots  # noqa: E402
from harnesslib import axes_ddtruth  # noqa: E402
from harnesslib import axes_stdtruth  # noqa: E402
from harnesslib import axes_wisdom  # noqa: E402
from harnesslib import axes_types  # noqa: E402
from harnesslib import levers as levers_mod  # noqa: E402


def _use_tmp(tmp: Path):
    # Les fonctions de guard lisent ces globals à l'appel → patch du module.
    guard.HARNESS_DIR = tmp
    guard.CRASH_JOURNAL = tmp / "crash-journal.jsonl"
    guard.QUARANTINE = tmp / "quarantine.json"
    guard.RESOLVED = tmp / "resolved.json"


def _journal(recs):
    with guard.CRASH_JOURNAL.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def run():
    with tempfile.TemporaryDirectory() as d:
        _use_tmp(Path(d))

        # 1. died_uncleanly loose → re-quarantainé
        # NB : la phase `speed_f3` (crash CÔTÉ F3) est volontairement ignorée par
        # la réconciliation — la quarantaine protège la machine des sweeps
        # FRACTALL. La fixture utilise donc une phase fractall (`speed`) ; le cas
        # `f3only` ci-dessous verrouille l'exclusion F3.
        _journal([
            {"case": "e22522", "phase": "speed",
             "outcome": "died_uncleanly", "detected_utc": "2026-07-07T09:16:10+00:00"},
            {"case": "f3only", "phase": "speed_f3",
             "outcome": "aborted", "ended_utc": "2026-07-07T09:20:00+00:00"},
            {"case": "triangle", "phase": "preflight",
             "outcome": "interrupted", "ended_utc": "2026-07-07T09:28:27+00:00"},
            {"case": "somef3", "phase": "speed_f3",
             "outcome": "fail", "ended_utc": "2026-07-07T09:30:00+00:00"},
            {"case": "hog", "phase": "preflight",
             "outcome": "aborted", "detected_utc": "2026-07-07T08:10:20+00:00"},
        ])
        guard.save_quarantine({})  # dérive : quarantaine vidée
        readded = guard.reconcile_quarantine_from_journal()
        q = guard.load_quarantine()
        _check("e22522" in q, "died_uncleanly loose → re-quarantainé")
        _check("hog" in q, "aborted loose → re-quarantainé")
        _check("triangle" not in q, "interrupted gracieux → PAS quarantainé")
        _check("somef3" not in q, "fail bénin (rc≠0) → PAS quarantainé")
        _check("f3only" not in q, "crash côté F3 (phase speed_f3) → PAS quarantainé")
        _check(set(readded) == {"e22522", "hog"}, "readded == cas hard-crash")

        # 2. idempotent : re-lancer ne double rien
        readded2 = guard.reconcile_quarantine_from_journal()
        _check(readded2 == [], "réconciliation idempotente (rien à re-ajouter)")

        # 3. remove pose un tombstone → PAS re-quarantainé
        guard.remove_quarantine("e22522")
        _check("e22522" in guard.load_resolved(), "remove → tombstone résolu")
        readded3 = guard.reconcile_quarantine_from_journal()
        _check("e22522" not in readded3,
               "résolu après incident → non re-quarantainé")
        _check("e22522" not in guard.load_quarantine(),
               "résolu reste hors quarantaine")

        # 4. incident PLUS RÉCENT que le tombstone → re-quarantainé
        with guard.CRASH_JOURNAL.open("a") as f:
            f.write(json.dumps({"case": "e22522", "phase": "speed",
                                "outcome": "killed_oom",
                                "detected_utc": "2027-01-01T00:00:00+00:00"}) + "\n")
        readded4 = guard.reconcile_quarantine_from_journal()
        _check("e22522" in readded4,
               "nouvel incident postérieur au tombstone → re-quarantainé")

        # 5. add périme le tombstone résolu
        guard.save_quarantine({})
        guard.save_resolved({"hog": "2027-06-01T00:00:00+00:00"})
        guard.add_quarantine("hog", "manuel")
        _check("hog" not in guard.load_resolved(),
               "add quarantine périme le tombstone résolu")

    run_regression_gates()
    run_profile_parsing()
    run_self_speed_gates()
    run_baseline_resolution()
    run_precision_axis()
    run_hotspots_parsing()
    run_truth_axis()
    run_wisdom_axis()
    run_levers()
    run_types_axis()

    print("\nTOUS LES TESTS PASSENT ✅")


def run_regression_gates():
    """Verrou (2026-07-11) : la régression geomean VITESSE ne se déclenche que
    sur la MÊME machine que la baseline (HARNESS.md « baseline par machine »).
    Une baseline d'une autre machine (≠ cpu/nproc) NE DOIT PAS flagger une fausse
    régression de vitesse ; les régressions de CORRECTION (quality FAIL) restent
    machine-indépendantes."""
    def _card(cpu, nproc, geomean, n_fail=0):
        return {
            "meta": {"tier": "quick", "machine": {"cpu": cpu, "nproc": nproc}},
            "speed": {"geomean_ratio": geomean},
            "quality": {"n_fail": n_fail, "fail_presets": [], "warn_presets": []},
            "parity": {"cases": {}},
            "goldens": {"passed": True},
        }

    i7 = ("Intel i7-10700F", 16)
    xeon = ("Intel Xeon", 4)

    # (a) Même machine, geomean dégradé >10 % → régression flaggée.
    base = _card(*i7, 0.32)
    card = _card(*i7, 0.40)  # +25 %
    gaps = gaps_mod.compute_gaps(card, base)
    _check(any(g["axis"] == "speed" and g["case"] == "<geomean>" for g in gaps),
           "geomean régresse sur MÊME machine → flaggé")

    # (b) Machine différente, geomean « pire » → PAS de fausse alarme vitesse.
    card_x = _card(*xeon, 0.63)  # 0.63 > 0.32*1.1 mais autre machine
    gaps_x = gaps_mod.compute_gaps(card_x, base)
    _check(not any(g["axis"] == "speed" and g["case"] == "<geomean>" for g in gaps_x),
           "geomean cross-machine → PAS de fausse régression vitesse")

    # (c) Correction (quality FAIL) reste machine-indépendante.
    card_xf = _card(*xeon, 0.30, n_fail=2)
    gaps_xf = gaps_mod.compute_gaps(card_xf, base)
    _check(any(g["axis"] == "quality" and "n_fail" in str(g.get("metric"))
               for g in gaps_xf),
           "régression quality (n_fail) flaggée même cross-machine")



SAMPLE_CLI_OUTPUT = """\
[WISDOM] type=Mandelbrot device=Cpu algo=Perturbation tier=f64 variants=- bench=- req_exp=166.1 req_prec=9b tier_mantissa=53b ref_prec=196b pixel_size=3.906e-53 iter=200000 iter_src=source
[PERTURB PERF] 256x256 pixels=65536 zoom=1.00e50 small_image=true orbit=0.120s pixels=0.800s post=0.010s total=0.950s avg_iter/px=12345 max_iter/px=99999 ns/iter=2.1 glitched_initial=0 corrections=0 fallback_ratio=0.000
[RELIABILITY] mode=Escalate runaway=1e30 flagged=12/65536 (0.0183%) type=Mandelbrot
[FRACTALL] type=Mandelbrot path=bytecode_f64 prec=196b iter_max=200000 avg_iter/px=12345 max_iter/px=99999 ns/iter=2.1 pixels=0.800s total=0.950s
Done in 1.10s (render: 0.95s, save: 0.05s)
"""


def run_profile_parsing():
    """Verrou de l'axe profile : le parseur lit les diagnostics RÉELS du moteur
    (formats de progress.rs / mod.rs / wisdom.rs / main.rs) et la ventilation
    par phase somme au wall-clock."""
    pr = axes_perf.parse_profile_output(SAMPLE_CLI_OUTPUT)
    _check(pr["path"] == "bytecode_f64", "profile : path lu depuis [FRACTALL]")
    _check(pr["ns_per_iter"] == 2.1 and pr["avg_iter_px"] == 12345.0,
           "profile : ns/iter + iter/px lus")
    _check(pr["orbit_s"] == 0.12 and pr["pixels_s"] == 0.8 and pr["post_s"] == 0.01,
           "profile : phases [PERTURB PERF] lues")
    _check(pr["tier"] == "f64" and pr["algo"] == "Perturbation",
           "profile : plan [WISDOM] lu")
    _check(pr["reliability_flagged_pct"] == 0.0183, "profile : % fiabilité lu")
    _check(pr["render_s"] == 0.95 and pr["save_s"] == 0.05,
           "profile : rendu/sauvegarde lus (stdout CLI)")
    _check(pr["dd_escalation"] is False, "profile : pas d'escalade dd")
    bd = axes_perf.phase_breakdown(pr, wall_s=1.20)
    s = bd["secs"]
    _check(abs(sum(s.values()) - 1.20) < 1e-9, "profile : phases sommées = wall")
    _check(bd["dominant"] == "pixels", "profile : phase dominante = pixels")
    _check(abs(s["other"] - 0.22) < 1e-9, "profile : résidu `other` = wall − phases")
    # deux rendus (escalade dd) → phases sommées, dernier [FRACTALL] retenu
    two = SAMPLE_CLI_OUTPUT + "[DD-ESCALATION] unreliable_ratio=0.0600 > 0.05 → re-render tier dd\n" \
        + SAMPLE_CLI_OUTPUT.replace("path=bytecode_f64", "path=bytecode_dd")
    pr2 = axes_perf.parse_profile_output(two)
    _check(pr2["dd_escalation"] and pr2["path"] == "bytecode_dd"
           and abs(pr2["pixels_s"] - 1.6) < 1e-9,
           "profile : escalade dd → phases sommées, path final")
    # G14.4a : `bla=` chronométré à part (binaires récents) — phase propre,
    # retirée du résidu ; les sorties antérieures (sans `bla=`) restent lues.
    with_bla = SAMPLE_CLI_OUTPUT.replace("orbit=0.120s pixels=",
                                         "orbit=0.120s bla=0.030s pixels=")
    prb = axes_perf.parse_profile_output(with_bla)
    _check(prb["bla_s"] == 0.03 and prb["orbit_s"] == 0.12 and prb["pixels_s"] == 0.8,
           "profile : phase `bla=` lue")
    bdb = axes_perf.phase_breakdown(prb, wall_s=1.20)
    _check(abs(bdb["secs"]["other"] - 0.19) < 1e-9
           and abs(sum(bdb["secs"].values()) - 1.20) < 1e-9,
           "profile : `bla` sort du résidu, phases sommées = wall")
    _check(pr["bla_s"] is None and bd["secs"]["bla"] == 0.0,
           "profile : sortie sans `bla=` → phase bla nulle (rétro-compatible)")
    # ns/iter précis + normalisé par thread (le moteur imprime à 0,1 ns près)
    pn = dict(prb)
    axes_perf.precise_ns_per_iter(pn, 256 * 256, 4)
    _check(abs(pn["ns_per_iter"] - 0.8e9 / (12345 * 65536)) < 1e-4
           and abs(pn["ns_per_iter_thread"] - 4 * 0.8e9 / (12345 * 65536)) < 4e-4
           and pn["threads"] == 4,
           "profile : ns/iter recalculé et normalisé par thread")
    _check(axes_perf.effective_threads({"RAYON_NUM_THREADS": "3"}) == 3,
           "profile : threads effectifs = RAYON_NUM_THREADS")
    # path f64 standard (pas de [PERTURB PERF]) : tout le rendu = pixels
    std = "Done in 0.30s (render: 0.20s, save: 0.05s)\n"
    bd3 = axes_perf.phase_breakdown(axes_perf.parse_profile_output(std), 0.30)
    _check(bd3["secs"]["pixels"] == 0.2 and bd3["dominant"] == "pixels",
           "profile : path std → rendu = pixels")
    # ligne française des binaires antérieurs (baselines) : toujours lue
    legacy = "Génération terminée en 0.30s (rendu: 0.20s, sauvegarde: 0.05s)\n"
    prl = axes_perf.parse_profile_output(legacy)
    _check(prl["render_s"] == 0.2 and prl["save_s"] == 0.05
           and prl["gen_total_s"] == 0.3,
           "profile : ligne CLI française (binaires antérieurs) lue")
    # bench TOML
    ent = axes_perf.parse_bench_toml(
        'version = 1\nmachine = "x"\n[[bench]]\nkey = "cpu_perturb_f64"\n'
        'iters_per_sec = 5.0e8\nwidth = 512\nheight = 512\nseconds = 1.6\n'
        'measured_unix = 1\n')
    _check(ent["cpu_perturb_f64"]["iters_per_sec"] == 5.0e8, "bench : TOML wisdom lu")


def run_self_speed_gates():
    """Verrou vitesse INTRINSÈQUE (sans F3) : régression par cas détectée sur la
    même machine, jamais cross-machine, jamais sous le plancher de bruit."""
    def _card(cpu, nproc, cases, profile=None, bench=None):
        return {
            "meta": {"tier": "quick", "machine": {"cpu": cpu, "nproc": nproc},
                     "git_sha": "abc"},
            "speed": {"geomean_ratio": None, "cases": cases},
            "quality": {"n_fail": 0, "fail_presets": [], "warn_presets": []},
            "parity": {"cases": {}}, "goldens": {"passed": True},
            "profile": {"cases": profile or {}}, "bench": {"entries": bench or {}},
        }
    i7 = ("i7", 16)
    base = _card(*i7, {"e50": {"fractall_s": 1.0}, "test5": {"fractall_s": 0.010},
                       "dragon": {"fractall_s": 2.0, "fractall_s_spread": 0.30}},
                 profile={"e50": {"status": "ok", "ns_per_iter": 2.0, "path": "bytecode_f64",
                                  "ns_per_iter_thread": 32.0, "secs": {"pixels": 0.8}},
                          "test5": {"status": "ok", "ns_per_iter": 20.0, "path": "bytecode_f64",
                                    "ns_per_iter_thread": 320.0, "secs": {"pixels": 0.005}},
                          "e113": {"status": "ok", "ns_per_iter": 0.3, "path": "bytecode_f64"}},
                 bench={"cpu_perturb_f64": {"iters_per_sec": 1e9}})
    card = _card(*i7, {"e50": {"fractall_s": 1.30},        # +30 % → gap
                       "test5": {"fractall_s": 0.014},     # +40 % mais < 50 ms → bruit
                       "dragon": {"fractall_s": 2.5, "fractall_s_spread": 0.30}},  # +25 % < bruit 30 %
                 profile={"e50": {"status": "ok", "ns_per_iter": 2.6, "path": "bytecode_f64",
                                  "ns_per_iter_thread": 41.6, "secs": {"pixels": 1.04}},
                          # ×2 mais phase pixel de 10 ms → bruit de chrono
                          "test5": {"status": "ok", "ns_per_iter": 40.0, "path": "bytecode_f64",
                                    "ns_per_iter_thread": 640.0, "secs": {"pixels": 0.010}},
                          # baseline lue à 0,1 ns près (ancien schéma) → non comparable
                          "e113": {"status": "ok", "ns_per_iter": 0.378, "path": "bytecode_f64",
                                   "ns_per_iter_thread": 6.0, "secs": {"pixels": 0.4}}},
                 bench={"cpu_perturb_f64": {"iters_per_sec": 7e8}})
    ss = gaps_mod.self_speed_ratio(card, base)
    _check(ss["n"] == 3 and ss["ratios"]["e50"] == 1.3, "self-speed : ratios par cas")
    g = gaps_mod.compute_gaps(card, base)
    names = {(x["axis"], x["case"], x["metric"]) for x in g}
    _check(("speed", "e50", "fractall_s") in names, "self-speed : +30 % même machine → gap")
    _check(("speed", "test5", "fractall_s") not in names,
           "self-speed : +4 ms sous le plancher → pas de gap")
    _check(("speed", "dragon", "fractall_s") not in names,
           "self-speed : delta < bruit mesuré (spread) → pas de gap")
    _check(("speed", "<self-geomean>", "self_geomean") in names,
           "self-speed : geomean > 1.10 → gap agrégé")
    _check(("profile", "e50", "ns_per_iter") in names, "profile : ns/iter +30 % → gap")
    _check(("profile", "test5", "ns_per_iter") not in names,
           "profile : phase pixel sous le plancher 50 ms → pas de gap")
    _check(("profile", "e113", "ns_per_iter") not in names,
           "profile : baseline à ns/iter arrondi (ancien schéma) → pas de gap")
    _check(("bench", "cpu_perturb_f64", "iters_per_sec") in names,
           "bench : débit −30 % → gap")
    # cross-machine : rien de tout ça
    xeon = _card("Xeon", 4, {"e50": {"fractall_s": 3.0}},
                 profile={"e50": {"status": "ok", "ns_per_iter": 5.0, "path": "bytecode_f64"}},
                 bench={"cpu_perturb_f64": {"iters_per_sec": 1e8}})
    gx = gaps_mod.compute_gaps(xeon, base)
    _check(not any(x["axis"] in ("speed", "profile", "bench") for x in gx),
           "cross-machine → aucune fausse régression intrinsèque")
    _check(gaps_mod.self_speed_ratio(xeon, base) == {}, "cross-machine → pas d'auto-ratio")
    # signal profil indépendant de la baseline : sauvegarde > rendu
    slow_save = _card(*i7, {}, profile={"gt5": {"status": "ok", "wall_s": 0.5,
                                                "render_s": 0.13, "save_s": 0.28}})
    gs = gaps_mod.compute_gaps(slow_save, None)
    _check(any(x["axis"] == "profile" and x["metric"] == "save_s" for x in gs),
           "profile : sauvegarde > rendu → gap vitesse")


def run_baseline_resolution():
    """Verrou baselines PAR MACHINE : la machine courante lit SA baseline si
    elle existe, sinon le repli `baseline.json` ; le slug est stable et sûr
    pour un nom de fichier."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        config.BASELINE = tmp / "baseline.json"
        config.BASELINES_DIR = tmp / "baselines"
        i7 = {"cpu": "Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz", "nproc": 16}
        xeon = {"cpu": "Intel(R) Xeon(R) Processor", "nproc": 4}
        _check(config.machine_slug(i7) == "intel-r-core-tm-i7-10700f-cpu-2-90ghz-16t",
               "slug machine normalisé")
        _check(config.resolve_baseline(xeon) is None, "aucune baseline → None")
        config.BASELINE.write_text(json.dumps({"meta": {"machine": i7, "git_sha": "legacy"}}))
        _check(config.load_baseline(xeon)["meta"]["git_sha"] == "legacy",
               "sans baseline propre → repli baseline.json")
        config.BASELINES_DIR.mkdir()
        config.baseline_path_for(xeon).write_text(
            json.dumps({"meta": {"machine": xeon, "git_sha": "mine"}}))
        _check(config.load_baseline(xeon)["meta"]["git_sha"] == "mine",
               "baseline propre prioritaire sur le repli")
        _check(config.load_baseline(i7)["meta"]["git_sha"] == "legacy",
               "l'autre machine garde le repli")


def run_precision_axis():
    """Verrou (2026-11-05) de l'axe `precision` (juge orbite ×4) : lecteur
    `.fmap` stdlib conforme au layout `io/fmap.rs`, seuils de verdict, et
    gap de sévérité « correction » sur FAIL."""
    import struct
    import zlib
    print("\n== axe precision ==")
    iters = [0, 7, 4096, 2**32 - 1]
    raw = struct.pack(f"<{len(iters)}I", *iters)
    js = b'{"width":2}'
    blob = bytearray(b"FMAP") + struct.pack("<I", 1) + struct.pack("<I", len(js)) + js
    blob += struct.pack("<I", 2)
    zs = zlib.compress(b"\x00" * 64)  # canal zs AVANT itérations : ordre libre
    blob += bytes([1]) + struct.pack("<QQ", 64, len(zs)) + zs
    comp = zlib.compress(raw)
    blob += bytes([0]) + struct.pack("<QQ", len(raw), len(comp)) + comp
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.fmap"
        p.write_bytes(bytes(blob))
        _check(axes_precision.read_fmap_iterations(p) == iters,
               "lecture du canal itérations d'un .fmap (canaux dans le désordre)")
    _check(axes_precision.verdict_for(0.0) == "PASS", "0 px → PASS")
    _check(axes_precision.verdict_for(1 / 9216) == "PASS",
           "1 px à 96² (résidu e227) → PASS")
    _check(axes_precision.verdict_for(0.04) == "FAIL",
           "4 % (précision F3 nue, sentinelles) → FAIL")
    card = {"meta": {"tier": "quick", "machine": {}},
            "precision": {"judge_factor": 4, "cases": {
                "golden_spider": {"status": "ok", "verdict": "FAIL",
                                  "diff_px": 466, "frac": 0.05, "prec": 541},
                "e113": {"status": "ok", "verdict": "PASS", "diff_px": 0,
                         "frac": 0.0, "prec": 478}}}}
    gp = [g for g in gaps_mod.compute_gaps(card, None) if g["axis"] == "precision"]
    _check(len(gp) == 1 and gp[0]["case"] == "golden_spider"
           and gp[0]["severity"] == 1,
           "FAIL précision → gap correction (sévérité 1), PASS → aucun gap")


SAMPLE_REPORT = """\
Call graph:
    2984 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
      2984 start  (in dyld) + 6992  [0x198e3eb50]

Total number in stack (recursive counted multiple, when >=5):

Sort by top of stack, same collapsed (when >= 5):
        _RNvNtNtNtCs2eIBBeHrl05_12fractall_cli7fractal8bytecode14pixel_loop_exp38iterate_pixel_unified_exp_single_phase  (in fractall-cli)        3020
        __psynch_cvwait  (in libsystem_kernel.dylib)        2989
        _RNvNtNtCsbT3fELgpM5P_3rug5float3big5parse  (in fractall-cli)        2239
        __gmpn_submul_1  (in fractall-cli)        1657
        ???  (in fractall-cli)  load address 0x102094000 + 0x3597c4  [0x1023ed7c4]        209
        swtch_pri  (in libsystem_kernel.dylib)        100

Binary Images:
       0x100960000 -        0x100da88af +fractall-cli (0) <0614B602> /Users/x/fractall-cli
"""


def run_hotspots_parsing():
    """Verrou de `harness.py hotspots` : lecture d'un rapport `sample` réel
    (format macOS), démanglage Rust v0, exclusion de l'attente des workers."""
    print("\n== hotspots ==")
    rows = hotspots.parse_sample_top(SAMPLE_REPORT)
    _check(len(rows) == 6, "hotspots : 6 sommets de pile lus")
    _check(rows[0][0] == "fractal::bytecode::pixel_loop_exp::"
           "iterate_pixel_unified_exp_single_phase" and rows[0][2] == 3020,
           "hotspots : symbole Rust v0 démanglé en chemin de module")
    _check(rows[2][0] == "rug::float::big::parse",
           "hotspots : symbole d'une autre crate démanglé")
    _check(rows[4][0].startswith("???") and rows[4][2] == 209,
           "hotspots : adresse non symbolisée conservée avec son compte")
    summ = hotspots.summarize(rows)
    _check(summ["idle"] == 2989 + 100 and summ["busy"] == 3020 + 2239 + 1657 + 209,
           "hotspots : attente des workers exclue du temps occupé")
    top = summ["rows"][0]
    _check(top[1] == 3020 and abs(top[2] - 100.0 * 3020 / 7125) < 1e-9,
           "hotspots : part calculée sur le temps occupé")
    _check(hotspots.demangle("__gmpn_submul_1") == "__gmpn_submul_1",
           "hotspots : symbole C rendu tel quel")

def run_truth_axis():
    """Verrou de l'axe truth : décompte des pixels ≠ juge dd, verdicts, et gap
    de correction (sévérité 1) sur un FAIL."""
    print("\n== axe truth ==")
    _check(axes_ddtruth.count_diff([1, 2, 3, 4], [1, 2, 9, 4]) == 1,
           "truth : pixels différents comptés")
    _check(axes_ddtruth.count_diff([1, 2, 3], [1, 2]) == 1,
           "truth : longueurs différentes comptées")
    card = {"meta": {"machine": {"cpu": "x", "nproc": 1}}, "truth": {"status": "ok", "cases": {
        "e227": {"status": "ok", "verdict": "FAIL", "diff_px": 384, "frac": 0.0059,
                 "path": "bytecode_f64"},
        "e50": {"status": "ok", "verdict": "PASS", "diff_px": 1, "frac": 0.0001,
                "path": "bytecode_f64"},
        "rug": {"status": "ok", "verdict": "WARN", "diff_px": 20, "frac": 0.002,
                "path": "bytecode_f64"},
        "spiral": {"status": "n/a", "path": "std_f64"}}}}
    g = {(x["axis"], x["case"]): x["severity"] for x in gaps_mod.compute_gaps(card, None)}
    _check(g.get(("truth", "e227")) == 1, "truth : FAIL → gap correction (sévérité 1)")
    _check(g.get(("truth", "rug")) == 4, "truth : WARN → gap qualité (sévérité 4)")
    _check(("truth", "e50") not in g and ("truth", "spiral") not in g,
           "truth : PASS et n/a → aucun gap")
    _check("truth" in harnesslib_cli.ALL_AXES.split(","), "truth : axe déclaré")
    two = ("[FRACTALL] type=Mandelbrot path=bytecode_f64 prec=256b x\n"
           "[FRACTALL] type=Mandelbrot path=bytecode_dd prec=256b x\n")
    _check(axes_ddtruth.last_path(two) == "bytecode_dd",
           "truth : path du DERNIER [FRACTALL] (escalade dd = deux lignes)")
    _check(axes_ddtruth.is_dd_path("bytecode_dd")
           and not axes_ddtruth.is_dd_path("bytecode_f64")
           and not axes_ddtruth.is_dd_path("bytecode_f64_harmonic_lla")
           and not axes_ddtruth.is_dd_path(None),
           "truth : un juge qui n'a pas rendu en dd n'est pas un juge (n/a)")
    # Volet path f64 standard (chantier de correction n° 1) : scènes jugées
    # contre le GMP pur, fusionnées dans les cas de l'axe.
    quick = axes_stdtruth.scenes(False)
    full = axes_stdtruth.scenes(True)
    auto_cap = axes_stdtruth.AUTO_CAP_SCENES
    _check(len(full) == 18 + len(auto_cap) and len(quick) == 9 + len(auto_cap)
           and set(quick) <= set(full),
           "truth std : 18 scènes + plafonds auto, quick ⊂ full")
    _check(all(it == 2500 for sc in quick if sc not in auto_cap
               for _, _, it in [sc]),
           "truth std : tier quick = plafond par défaut (2500)")
    _check(("elephants", "1e7", 60000) in quick,
           "truth std : la scène au plafond AUTO est jugée dès le tier quick")
    _check(axes_stdtruth.scene_name("seahorse", "1e10", 2500) == "std:seahorse_1e10_2500",
           "truth std : nom de scène stable")
    cmd = axes_stdtruth._cmd("seahorse", "1e10", 2500, Path("x.fmap"), ["--algorithm", "gmp"])
    _check("--no-gpu" in cmd and cmd[-2:] == ["--algorithm", "gmp"]
           and "--iterations" in cmd,
           "truth std : juge GMP pur, CPU, plafond explicite")
    card = {"meta": {"machine": {"cpu": "x", "nproc": 1}}, "truth": {"status": "ok", "cases": {
        "std:seahorse_1e10_2500": {"status": "ok", "verdict": "FAIL", "diff_px": 3179,
                                   "frac": 0.0776, "path": "std_f64"}}}}
    g = {(x["axis"], x["case"]): x["severity"] for x in gaps_mod.compute_gaps(card, None)}
    _check(g.get(("truth", "std:seahorse_1e10_2500")) == 1,
           "truth std : FAIL du path standard → gap correction")


def run_wisdom_axis():
    """Verrous de l'axe wisdom : le regret ne compte qu'à sortie équivalente,
    le bruit ne fait pas un verdict, l'adjudication GMP tranche dans le bon
    ordre, et le volet itérations juge la classe évadé/plafonné."""
    print("\n== axe wisdom ==")
    W = axes_wisdom
    _check("wisdom" in harnesslib_cli.ALL_AXES.split(","), "wisdom : axe déclaré")
    _check(W.diff_frac([1, 2, 3, 4], [1, 2, 9, 4]) == 0.25
           and W.diff_frac([1, 2], [1, 2, 3, 4]) == 0.5,
           "wisdom : fraction de pixels ≠ (longueurs comprises)")
    # alternative équivalente nettement plus rapide → FAIL de vitesse
    r = W.regret_verdict(2.0, {"a": {"secs": 1.0, "frac": 0.0},
                               "b": {"secs": 3.0, "frac": 0.0}})
    _check(r["verdict"] == "FAIL" and r["best_equiv"] == "a" and r["regret"] == 2.0,
           "wisdom : auto ×2 plus lent à sortie équivalente → FAIL")
    r = W.regret_verdict(1.15, {"a": {"secs": 1.0, "frac": 0.0}})
    _check(r["verdict"] == "WARN", "wisdom : regret 1,15 → WARN")
    # sous le plancher de bruit absolu : jamais un verdict
    r = W.regret_verdict(0.10, {"a": {"secs": 0.05, "frac": 0.0}})
    _check(r["verdict"] == "PASS", "wisdom : écart < 60 ms → PASS (bruit)")
    # plus rapide mais sortie ≠ → ADJUDICATE, PAS un gap de vitesse
    r = W.regret_verdict(2.0, {"a": {"secs": 1.0, "frac": 0.03},
                               "b": {"secs": 1.95, "frac": 0.0}})
    _check(r["verdict"] == "ADJUDICATE" and r["best_other"] == "a"
           and r["best_equiv"] == "b",
           "wisdom : alternative rapide à sortie ≠ → ADJUDICATE")
    base = {"verdict": "ADJUDICATE", "chosen_s": 2.0, "best_other": "a",
            "best_other_s": 1.0, "path": "std_f64"}
    j = W.adjudged_verdict(base, 0.0104, 0.0416)
    _check(j["verdict"] == "FAIL" and j["adjudged"] == "auto_wrong",
           "wisdom : production fausse vs GMP → FAIL de correction")
    j2 = W.adjudged_verdict(base, 0.0, 0.0416)
    _check(j2["verdict"] == "PASS" and j2["adjudged"] == "alt_wrong",
           "wisdom : alternative fausse, production juste → PASS (wisdom a raison)")
    j3 = W.adjudged_verdict(base, 0.0, 0.0)
    _check(j3["verdict"] == "FAIL" and j3["adjudged"] == "alt_right"
           and j3["regret"] == 2.0,
           "wisdom : alternative juste ET plus rapide → vrai regret")
    # gaps : sévérités
    def card(c):
        return {"meta": {"machine": {"cpu": "x", "nproc": 1}},
                "wisdom": {"status": "ok", "cases": {"k": c}}}
    def sev(c):
        g = [x for x in gaps_mod.compute_gaps(card(c), None) if x["axis"] == "wisdom"]
        return g[0]["severity"] if g else None
    dec = {"status": "ok", "kind": "decision", "path": "std_f64",
           "out_frac": {"a": 0.03}, "best_other": "a", "best_other_s": 1.0,
           "best_equiv": "b", "best_equiv_s": 1.95, "regret": 1.4,
           "regret_s": 0.5}
    _check(sev({**dec, **j, "kind": "decision", "status": "ok"}) == 1,
           "wisdom : production fausse → gap correction (sévérité 1)")
    _check(sev({**dec, "verdict": "FAIL"}) == 3,
           "wisdom : regret FAIL → gap vitesse (sévérité 3)")
    _check(sev({**dec, "verdict": "ADJUDICATE", "regret_other": 1.5}) == 4,
           "wisdom : ADJUDICATE non tranché → gap qualité (sévérité 4)")
    _check(sev({**dec, "verdict": "ADJUDICATE", "adj_status": "known"}) is None,
           "wisdom : ADJUDICATE déjà adjugé → aucun gap")
    _check(sev({**dec, "verdict": "PASS"}) is None, "wisdom : PASS → aucun gap")
    # volet itérations
    n = 100
    at_n = [100, 100, 50, 100]          # 3 plafonnés à N
    at_4n = [400, 150, 50, 400]         # un seul s'évade avant 4N
    _check(W.under_iteration_frac(at_n, at_4n, n) == 0.25,
           "wisdom iter : pixels plafonnés à N qui s'évadent avant 4N")
    flat = [5] * 999 + [1600]           # rien n'évade entre 25 et 1600
    _check(W.smallest_sufficient(flat, 1600) == 25,
           "wisdom iter : plus petit N/4^k qui satisfait encore le critère")
    busy = [5] * 900 + [300] * 100      # 10 % évadent entre 100 et 400
    _check(W.smallest_sufficient(busy, 1600) == 400,
           "wisdom iter : descente stoppée dès que le critère casse")
    _check(W.iter_verdict(0.03, 1, None) == "FAIL"
           and W.iter_verdict(0.01, 1, None) == "WARN"
           and W.iter_verdict(0.001, 4, 3.0) == "PASS"
           and W.iter_verdict(0.001, 16, 3.0, 0.5) == "WASTE"
           and W.iter_verdict(0.001, 16, 3.0, 0.01) == "PASS",
           "wisdom iter : verdicts (4× de marge est voulu ; bruit ignoré)")
    _check(sev({"status": "ok", "kind": "iter", "verdict": "FAIL",
                "under_frac": 0.03, "auto_iterations": 750}) == 1,
           "wisdom iter : sous-itération FAIL → gap correction")
    _check(sev({"status": "timeout", "kind": "iter", "budget_s": 20.0}) == 2,
           "wisdom iter : sonde + rendu hors budget → gap robustesse")
    _check({"barnsley_m_zoom", "multibrot_zoom"} <= set(W.ITER_QUICK),
           "wisdom iter : les types fuyants sont sondés dès le tier quick")
    txt = ("Iterations: auto → 90000 (max escape 91702 on a 81x51 probe, "
           "2 rounds, 0.56 s) — --iterations N overrides\n"
           "[RELIABILITY-STD] flagged=441 (10.675 %) corrected=441 via=Perturbation time=0.466s\n"
           "[RELIABILITY-STD] flagged=4315 (10.535 %) corrected=4315 via=Perturbation time=0.508s\n"
           "Done in 1.63s (render: 1.61s, save: 0.02s)\n")
    _check(W.parse_auto_line(txt) == (90000, 0.56),
           "wisdom iter : N et durée de sonde lus dans la ligne auto")
    _check(W.parse_render_s(txt) == 1.61, "wisdom iter : durée de rendu lue")
    _check(W.parse_reliability(txt) == (10.535, 0.508),
           "wisdom iter : fiabilité du DERNIER rendu (pas celle de la sonde)")
    for dname, dim in W.DIMENSIONS.items():
        _check("auto" in dim["variants"] and set(dim["quick"]) <= set(dim["full"]),
               f"wisdom : dimension {dname} — variante auto, quick ⊂ full")
        for sc, a in (dim.get("scenes") or {}).items():
            _check("--iterations" in a,
                   f"wisdom : scène {sc} à plafond EXPLICITE (sonde hors chrono)")
    _check(all("--iterations" not in W.ITER_SCENES[s] for s in W.ITER_FULL)
           and set(W.ITER_QUICK) <= set(W.ITER_FULL),
           "wisdom iter : scènes SANS plafond (c'est la sonde qu'on juge)")
    md = "\n".join(W.wisdom_md({"counts": {"PASS": 1}, "runs": 3, "size": [256, 160],
                                "cases": {"decision:algorithm:x": {**dec, **j,
                                          "status": "ok", "kind": "decision",
                                          "chosen_s": 2.0}}}))
    _check("adjugé GMP" in md and "1.04 % faux" in md,
           "wisdom : le scorecard montre l'adjudication GMP")


def run_levers():
    """Verrous des leviers : classés par secondes, planchers respectés, et
    présents même quand il n'y a aucun gap."""
    print("\n== leviers ==")
    card = {"meta": {"machine": {"cpu": "x", "nproc": 1}},
            "profile": {"status": "ok", "cases": {
                "dragon": {"status": "ok", "wall_s": 2.8, "path": "bytecode_f64",
                           "reliability_flagged_pct": 0.72, "ns_per_iter_thread": 0.04,
                           "secs": {"orbit": 1.04, "bla": 0.04, "pixels": 0.71,
                                    "post": 0.94, "save": 0.0, "other": 0.04}},
                "test5": {"status": "ok", "wall_s": 0.02, "path": "bytecode_f64",
                          "secs": {"orbit": 0.001, "pixels": 0.001, "other": 0.02}}}},
            "wisdom": {"status": "ok", "cases": {
                "iter:elephants_1e7": {"status": "ok", "kind": "iter", "verdict": "PASS",
                                       "auto_iterations": 60000, "render_s": 9.0,
                                       "probe_s": 1.6, "fiab_s": 7.3, "fiab_pct": 43.3,
                                       "path": "std_f64"}}}}
    lv = levers_mod.compute_levers(card)
    _check(not gaps_mod.compute_gaps(card, None) and len(lv) >= 4,
           "leviers : une liste de travail même sans aucun gap")
    _check([x["secs"] for x in lv] == sorted((x["secs"] for x in lv), reverse=True),
           "leviers : triés par secondes décroissantes")
    _check(lv[0]["case"] == "iter:elephants_1e7" and "fiabilité" in lv[0]["what"],
           "leviers : la correction de fiabilité à N auto arrive en tête")
    _check(all(x["case"] != "test5" for x in lv),
           "leviers : rien sous le plancher de mesure (50 ms)")
    _check(not any(x["case"] == "dragon" and "BLA" in x["what"] for x in lv),
           "leviers : une phase < 10 % de son cas n'est pas un levier")
    md = "\n".join(levers_mod.levers_md(lv))
    _check("## Leviers" in md and "elephants_1e7" in md and "| 1 |" in md,
           "leviers : section du scorecard")
    _check("aucun levier" in "\n".join(levers_mod.levers_md([])),
           "leviers : section présente même vide")


def run_types_axis():
    """Verrous de l'axe types : tous les types ont une scène figée à plafond
    explicite, un FAIL est toujours un juge CONVERGÉ, et le tier micro reste
    léger."""
    print("\n== axe types ==")
    T = axes_types
    _check("types" in config.ALL_AXES.split(","), "types : axe déclaré")
    sc = T.load_scenes()
    _check(set(sc) == {str(t) for t in T.ALL_TYPES},
           "types : une scène figée pour CHACUN des 33 types")
    judged = [t for t in T.ALL_TYPES if t not in T.NO_JUDGE]
    _check(all(sc[str(t)].get("edge") for t in judged),
           "types : chaque type jugé a une vue de bord")
    _check(all("--iterations" in T.scene_args(t, sc[str(t)], w)
               for t in judged for w in ("home", "edge")),
           "types : plafond EXPLICITE (la sonde auto est jugée par wisdom)")
    _check(all(0.02 < sc[str(t)]["edge"]["capped"] < 0.98
               or sc[str(t)]["edge"]["distinct"] >= 15 for t in judged),
           "types : vues de bord non uniformes")
    # éclaireur : voisinage mixte plafonné/évadé préféré à une équipotentielle
    w, h, cap = 16, 16, 100
    it = [cap if x < 8 else 5 + (x % 3) for y in range(h) for x in range(w)]
    x, _ = T.edge_pixel(it, w, h, cap)
    _check(5 <= x <= 9, "types : l'éclaireur vise la frontière intérieur/extérieur")
    _check(T.scene_verdict(0.03, None, False, 0.0) == "FAIL"
           and T.scene_verdict(0.03, None, False, 0.8) == "CHAOS"
           and T.scene_verdict(0.0, None, False) == "PASS"
           and T.scene_verdict(None, True, False) == "PASS"
           and T.scene_verdict(None, False, False) == "FAIL"
           and T.scene_verdict(None, True, True) == "WARN",
           "types : juge non convergé → CHAOS ; non déterministe → FAIL")
    def sev(c):
        card = {"meta": {"machine": {"cpu": "x", "nproc": 1}},
                "types": {"status": "ok", "cases": {"14:Tricorn:edge": c}}}
        g = [x for x in gaps_mod.compute_gaps(card, None) if x["axis"] == "types"]
        return g[0]["severity"] if g else None
    base = {"status": "ok", "path": "std_f64", "diff_px": 65, "frac": 0.025}
    _check(sev({**base, "verdict": "FAIL", "judge_frac": 0.0}) == 1,
           "types : FAIL à juge convergé → gap correction (sévérité 1)")
    _check(sev({**base, "verdict": "FAIL"}) == 4,
           "types : FAIL à juge NON vérifié → sévérité 4, pas une alerte rouge")
    _check(sev({**base, "verdict": "CHAOS", "judge_frac": 0.8}) is None,
           "types : scène chaotique → aucun gap")
    _check(sev({"status": "timeout"}) == 2, "types : rendu hors délai → robustesse")
    _check(T.ns_per_px_iter(1.0, 1000.0, 1000, 4) == 4000.0,
           "types : ns par pixel·itération·thread")
    micro = config.tier_config("micro")
    _check(micro["width"] <= 128 and micro["scene_size"][0] <= 128
           and "quality" not in micro["axes"] and "types" in micro["axes"]
           and "wisdom" in micro["axes"],
           "tier micro : petites images, axes essentiels (cycle d'A/B léger)")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
