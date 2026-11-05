"""Sous-commandes principales : score, baseline, gaps, journal, adjudicate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
import sys
from pathlib import Path

from .axes_f3 import axis_parity, axis_speed
from .axes_perf import axis_bench, axis_profile
from .axes_ddtruth import axis_ddtruth
from .axes_precision import axis_precision
from .axes_truth import axis_fuzz, axis_goldens, axis_quality
from .config import (
    ADJUDICATIONS, BASELINE, BASELINES_DIR, BENCH, CRASH_JOURNAL, HISTORY_DIR,
    REPO, SCHEMA, SCORECARD, baseline_path_for, load_baseline, machine_slug,
    SEVERITY_LABEL, TOML_DIR, _now_iso, cargo_build, find_f3, git_dirty,
    git_sha, load_compare_f3, machine_info, tier_config,
)
from .gaps import compute_gaps, load_adjudications, self_speed_ratio, sort_gaps
from .guard import (
    check_stale_inflight, default_mem_limit_mb, journal_begin, journal_end,
    load_quarantine, load_slow_safe, reconcile_quarantine_from_journal,
)
from .scorecard import load_history, build_scorecard_md, latest_history, write_history

def cmd_score(args) -> None:
    cfg = tier_config(args.tier)
    cases = ([c.strip() for c in args.cases.split(",")] if args.cases
             else cfg["cases"])
    width = args.width or cfg["width"]
    height = args.height or cfg["height"]
    runs = args.runs or cfg["runs"]
    timeout = args.timeout or cfg["timeout"]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]

    # Garde-fou crash : un breadcrumb résiduel = le run précédent est mort
    # pendant un cas (OOM / plantage OS). On le journalise + quarantaine AVANT
    # de relancer, sinon on replante sur le même cas.
    check_stale_inflight(auto_quarantine=not args.no_quarantine)
    if not args.no_quarantine:
        reconcile_quarantine_from_journal()
    mem_limit_mb = (0 if args.no_mem_limit
                    else args.mem_limit_mb if args.mem_limit_mb is not None
                    else default_mem_limit_mb())
    quarantined = set() if args.no_quarantine else set(load_quarantine())
    slow_safe = set(load_slow_safe())
    if mem_limit_mb:
        print(f"→ cap mémoire/process : {mem_limit_mb} MB "
              f"(RLIMIT_AS ; --no-mem-limit pour désactiver)")
    if quarantined:
        print(f"→ quarantaine active ({len(quarantined)}) : "
              f"{', '.join(sorted(quarantined))}")
    if slow_safe:
        print(f"→ slow-safe (skip, correct hors budget temps) ({len(slow_safe)}) : "
              f"{', '.join(sorted(slow_safe))}")

    if not args.no_rebuild:
        print("→ cargo build --release", flush=True)
        if not cargo_build():
            sys.exit("cargo build a échoué")

    f3_bin = find_f3()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    card: dict = {
        "schema": SCHEMA,
        "meta": {
            "date_utc": now.replace(microsecond=0).isoformat(),
            "stamp": stamp,
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "machine": machine_info(),
            "tier": args.tier,
            "axes": axes,
            "width": width,
            "height": height,
            "quality_width": cfg.get("quality_width", 256),
            "quality_height": cfg.get("quality_height", 256),
            "runs": runs,
            "f3_bin": str(f3_bin) if f3_bin else None,
            "mem_limit_mb": mem_limit_mb,
            "quarantined": sorted(quarantined),
            "slow_safe": sorted(slow_safe),
        },
        "speed": {"status": "skipped"},
        "parity": {"status": "skipped"},
        "quality": {"status": "skipped"},
        "fuzz": {"status": "skipped"},
        "precision": {"status": "skipped"},
        "truth": {"status": "skipped"},
        "goldens": {"status": "skipped"},
        "profile": {"status": "skipped"},
        "bench": {"status": "skipped"},
    }

    print(f"→ tier={args.tier} cases={len(cases)} {width}×{height} "
          f"runs={runs} timeout={timeout}s axes={axes}")
    print(f"→ F3: {f3_bin or 'INDISPONIBLE (speed/parity dégradés)'}")

    if "speed" in axes:
        print("\n== SPEED ==")
        card["speed"] = axis_speed(cases, width, height, runs, timeout, f3_bin,
                                   mem_limit_mb=mem_limit_mb,
                                   quarantined=quarantined, slow_safe=slow_safe)
    if "parity" in axes:
        print("\n== PARITY ==")
        card["parity"] = axis_parity(cases, width, height, timeout, f3_bin,
                                     mem_limit_mb=mem_limit_mb,
                                     iterations_cap=args.parity_iterations)
    if "quality" in axes:
        print("\n== QUALITY ==")
        card["quality"] = axis_quality(
            args.no_rebuild,
            cfg.get("quality_width", 256),
            cfg.get("quality_height", 256),
            mem_limit_mb=mem_limit_mb,
        )
    if "fuzz" in axes:
        print("\n== FUZZ ==")
        card["meta"]["fuzz_seed"] = args.fuzz_seed
        card["fuzz"] = axis_fuzz(cfg.get("fuzz_probes", 3), args.fuzz_seed,
                                 mem_limit_mb=mem_limit_mb)
    if "precision" in axes:
        print("\n== PRECISION (juge orbite ×4) ==")
        card["precision"] = axis_precision(
            cases, cfg.get("precision_size", 96), timeout,
            mem_limit_mb=mem_limit_mb, quarantined=quarantined,
            slow_safe=slow_safe)
    if "truth" in axes:
        print("\n== TRUTH (production vs tier dd) ==")
        card["truth"] = axis_ddtruth(
            cases, cfg.get("precision_size", 96), timeout,
            mem_limit_mb=mem_limit_mb, quarantined=quarantined,
            slow_safe=slow_safe)
    if "goldens" in axes:
        print("\n== GOLDENS ==")
        card["goldens"] = axis_goldens()
    if "profile" in axes:
        print("\n== PROFILE (où va le temps) ==")
        card["profile"] = axis_profile(cases, width, height, timeout,
                                       mem_limit_mb=mem_limit_mb,
                                       quarantined=quarantined,
                                       slow_safe=slow_safe)
    if "bench" in axes:
        print("\n== BENCH (iters/s par technique) ==")
        card["bench"] = axis_bench(mem_limit_mb=mem_limit_mb)

    card["beyond_f3"] = {
        "speed_wins": card.get("speed", {}).get("wins", []),
        "correctness_wins": [
            n for n, c in card.get("parity", {}).get("cases", {}).items()
            if c.get("status") == "f3_degenerate"],
    }

    base = load_baseline(card["meta"]["machine"])
    card["meta"]["baseline_file"] = base.get("_file") if base else None
    card["speed_self"] = self_speed_ratio(card, base) if base else {}
    card["gaps"] = compute_gaps(card, base)

    hist = write_history(card)
    SCORECARD.write_text(build_scorecard_md(card, base))
    print(f"\n✓ history : {hist.relative_to(REPO)}")
    print(f"✓ scorecard : {SCORECARD.relative_to(REPO)}")
    _print_gaps(card["gaps"])

def cmd_render(args) -> None:
    """Régénère SCORECARD.md (gaps recalculés vs baseline courante) depuis un
    history JSON — le dernier par défaut. Sert après un changement du rendu
    scorecard ou de la baseline, sans re-mesurer. N'écrit PAS de history."""
    path = Path(args.history) if args.history else latest_history()
    if path is None or not path.exists():
        sys.exit("history introuvable — lance d'abord `harness.py score`.")
    card = json.loads(path.read_text())
    base = load_baseline(card.get("meta", {}).get("machine"))
    card["meta"]["baseline_file"] = base.get("_file") if base else None
    card["speed_self"] = self_speed_ratio(card, base) if base else {}
    card["gaps"] = compute_gaps(card, base)
    if args.update_history:
        path.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    SCORECARD.write_text(build_scorecard_md(card, base))
    print(f"✓ scorecard régénéré depuis {path.relative_to(REPO)}")
    _print_gaps(card["gaps"])


def cmd_baseline(args) -> None:
    """Fige un history comme baseline de SA machine (`harness/baselines/
    <slug>.json`) — par défaut le dernier history produit sur CETTE machine.
    `--legacy` écrit le repli historique `baseline.json` (machine de dev)."""
    me = machine_info()
    if args.history:
        latest = Path(args.history)
    else:
        mine = load_history(same_machine_as={"meta": {"machine": me}})
        latest = (HISTORY_DIR / mine[-1]["_file"]) if mine else None
        if latest is None:
            sys.exit("Aucun history de CETTE machine — lance d'abord "
                     "`harness.py score` (ou passe un fichier history).")
    if not latest.exists():
        sys.exit(f"history introuvable : {latest}")
    card = json.loads(latest.read_text())
    machine = card.get("meta", {}).get("machine") or me
    if args.legacy:
        target = BASELINE
    else:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        target = baseline_path_for(machine)
    shutil.copyfile(latest, target)
    print(f"✓ baseline {target.relative_to(REPO)} ← {latest.name} "
          f"(tier {card.get('meta', {}).get('tier')}, "
          f"{machine.get('cpu')} · {machine.get('nproc')}t)")
    if not args.legacy and machine_slug(machine) != machine_slug(me):
        print("  ⚠️ ce history vient d'une AUTRE machine que celle-ci — la "
              "baseline écrite est celle de SA machine.")
    print(f"✓ baseline ← {latest.relative_to(REPO)}")

def cmd_gaps(_args) -> None:
    latest = latest_history()
    if latest is None:
        sys.exit("Aucun history — lance d'abord `harness.py score`.")
    card = json.loads(latest.read_text())
    print(f"Gaps du dernier score ({latest.name}) :\n")
    _print_gaps(card.get("gaps", []))

def _print_gaps(gaps: list[dict]) -> None:
    gaps = sort_gaps(gaps)
    if not gaps:
        print("Aucun gap détecté 🎉")
        return
    print(f"\n{len(gaps)} gap(s) — top 10 :")
    for i, g in enumerate(gaps[:10], 1):
        sev = SEVERITY_LABEL[g["severity"]]
        print(f"  {i:>2}. [{g['severity']} {sev:<10}] {g['axis']}/{g['case']} "
              f"{g['metric']}={g['value']} — {g['note']}")

def cmd_journal(_args) -> None:
    if not CRASH_JOURNAL.exists():
        print("Aucun incident journalisé 🎉")
        return
    lines = CRASH_JOURNAL.read_text().splitlines()
    print(f"{len(lines)} incident(s) — {CRASH_JOURNAL.relative_to(REPO)} :\n")
    for ln in lines[-30:]:
        try:
            r = json.loads(ln)
        except Exception:
            continue
        print(f"  [{r.get('ended_utc') or r.get('detected_utc', '?')}] "
              f"{r.get('outcome', '?'):15s} {r.get('phase', '?')}/"
              f"{r.get('case', '?')} "
              f"{('sig=' + str(r['signal'])) if r.get('signal') else ''} "
              f"{r.get('note', '')}")

def cmd_adjudicate(args) -> None:
    """Arbitre 3-voies (fractall-pert / F3 / GMP per-pixel) d'un cas parité,
    et PERSISTE le verdict dans harness/adjudications.json (versionné). Les
    scores suivants annotent les gaps parité avec ce verdict — un cas adjugé
    `f3_wrong` est déclassé (sévérité 2→4) : F3 fautif ≠ gap moteur fractall.
    ⚠️ coût : GMP per-pixel ≈ 1 µs/iter × size² — garder --size petit sur les
    cas profonds (>100k iters)."""
    compare_f3 = load_compare_f3()
    if compare_f3 is None:
        sys.exit("compare_f3 inimportable (numpy + OpenEXR requis : "
                 "pip3 install numpy OpenEXR Pillow)")
    import three_way_gmp as tw
    if tw.c.F3 is None:
        sys.exit("F3 binaire introuvable — bash scripts/build_f3_linux.sh")
    outdir = BENCH / "three_way"
    outdir.mkdir(parents=True, exist_ok=True)
    adjudications = load_adjudications()
    for stem in args.cases:
        toml = TOML_DIR / f"{stem}.toml"
        if not toml.exists():
            print(f"  {stem}: toml absent — skip")
            continue
        t = compare_f3.parse_light_toml(toml)
        iters = int(t.iterations or 1000)
        print(f"→ adjudication {stem} ({args.size}² · {iters} iters · "
              f"zoom {t.zoom})", flush=True)
        jrec = journal_begin("adjudicate", stem, {"size": f"{args.size}"})
        try:
            _, _, fp, f3s, total = tw.compare(
                stem, t.real, t.imag, t.zoom, iters, outdir,
                args.size, args.size)
        except Exception as e:  # rendu F3/fractall/GMP échoué
            journal_end(jrec, "fail", {"note": str(e)})
            print(f"  {stem}: ÉCHEC ({e})")
            continue
        journal_end(jrec, "ok")
        # Verdict sur les erreurs LARGES (>5, débruité du ±1 cross-implé).
        fb, f3b = fp["big"], f3s["big"]
        if fb <= max(2, total // 5000) and f3b > 4 * max(fb, 1):
            verdict = "f3_wrong"
        elif f3b <= max(2, total // 5000) and fb > 4 * max(f3b, 1):
            verdict = "fractall_wrong"
        elif fb > 0 and f3b > 0:
            verdict = "shared"
        else:
            verdict = "both_match_gmp"
        adjudications[stem] = {
            "date": _now_iso(), "git_sha": git_sha(), "size": args.size,
            "iters": iters, "fractall_big": fb, "f3_big": f3b,
            "fractall_max": fp["max"], "f3_max": f3s["max"],
            "verdict": verdict,
        }
        print(f"  {stem}: fractall big={fb} (max {fp['max']}) · "
              f"F3 big={f3b} (max {f3s['max']}) → {verdict}")
    ADJUDICATIONS.write_text(json.dumps(adjudications, indent=1,
                                        ensure_ascii=False) + "\n")
    print(f"\n✓ {ADJUDICATIONS.relative_to(REPO)} mis à jour "
          f"({len(adjudications)} cas adjugés)")
