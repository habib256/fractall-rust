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
from harnesslib import axes_precision  # noqa: E402


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
[WISDOM] type=Mandelbrot device=Cpu algo=Perturbation tier=f64 variants=- bench=- req_exp=166.1 req_prec=9b tier_mantissa=53b ref_prec=196b pixel_size=3.906e-53
[PERTURB PERF] 256x256 pixels=65536 zoom=1.00e50 small_image=true orbit=0.120s pixels=0.800s post=0.010s total=0.950s avg_iter/px=12345 max_iter/px=99999 ns/iter=2.1 glitched_initial=0 corrections=0 fallback_ratio=0.000
[RELIABILITY] mode=Escalate runaway=1e30 flagged=12/65536 (0.0183%) type=Mandelbrot
[FRACTALL] type=Mandelbrot path=bytecode_f64 prec=196b iter_max=200000 avg_iter/px=12345 max_iter/px=99999 ns/iter=2.1 pixels=0.800s total=0.950s
Génération terminée en 1.10s (rendu: 0.95s, sauvegarde: 0.05s)
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
    # path f64 standard (pas de [PERTURB PERF]) : tout le rendu = pixels
    std = "Génération terminée en 0.30s (rendu: 0.20s, sauvegarde: 0.05s)\n"
    bd3 = axes_perf.phase_breakdown(axes_perf.parse_profile_output(std), 0.30)
    _check(bd3["secs"]["pixels"] == 0.2 and bd3["dominant"] == "pixels",
           "profile : path std → rendu = pixels")
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
                 profile={"e50": {"status": "ok", "ns_per_iter": 2.0, "path": "bytecode_f64"}},
                 bench={"cpu_perturb_f64": {"iters_per_sec": 1e9}})
    card = _card(*i7, {"e50": {"fractall_s": 1.30},        # +30 % → gap
                       "test5": {"fractall_s": 0.014},     # +40 % mais < 50 ms → bruit
                       "dragon": {"fractall_s": 2.5, "fractall_s_spread": 0.30}},  # +25 % < bruit 30 %
                 profile={"e50": {"status": "ok", "ns_per_iter": 2.6, "path": "bytecode_f64"}},
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


if __name__ == "__main__":
    try:
        run()
    except AssertionError as e:
        print(f"\n❌ ÉCHEC : {e}")
        sys.exit(1)
