"""`harness.py ab` — A/B isolé par variable d'environnement ou flag CLI.

Méthode G14 (TODO.md) : chaque optimisation se juge sur un A/B contrôlé —
même cas, même taille, variantes = valeurs d'une env (`FRACTALL_RELIABILITY=
off,observe,escalate`) ou flags CLI (`--dd-tier`), N runs, médiane + spread,
ventilation par phase (profil) ET contrôle que la SORTIE est identique
(octets PNG : encodeur déterministe → mêmes octets ⇔ mêmes pixels). Un gain
de vitesse à sortie différente n'est pas un gain : c'est un tradeoff
correction à adjuger vs GMP (`fractall-quality compare`).

Sorties : `bench/harness/ab/<cas>.{md,json}` + tableau console.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .axes_perf import parse_profile_output, phase_breakdown
from .config import BENCH, CLI, TOML_DIR, _now_iso, cargo_build, git_sha, machine_info
from .guard import _classify_rc, _rlimit_preexec, journal_begin, journal_end


def parse_variants(env_specs: list[str], flag_specs: list[str]) -> list[dict]:
    """`--env K=a,b,c` et/ou `--flags "--x|--y z|"` → liste de variantes
    {name, env, flags}. Une seule dimension à la fois (A/B, pas un plan
    factoriel) ; plusieurs `--env` = produit cartésien simple."""
    variants: list[dict] = [{"name": "", "env": {}, "flags": []}]
    for spec in env_specs:
        if "=" not in spec:
            sys.exit(f"--env attend K=v1,v2,… (reçu {spec!r})")
        key, vals = spec.split("=", 1)
        out = []
        for v in variants:
            for val in vals.split(","):
                val = val.strip()
                nm = (v["name"] + " " if v["name"] else "") + f"{key}={val or '∅'}"
                env = dict(v["env"])
                if val:
                    env[key] = val
                else:
                    env[key] = ""   # vide = variable posée vide (≠ absente)
                out.append({"name": nm, "env": env, "flags": list(v["flags"])})
        variants = out
    for spec in flag_specs:
        out = []
        for v in variants:
            for alt in spec.split("|"):
                alt = alt.strip()
                nm = (v["name"] + " " if v["name"] else "") + (alt or "(sans)")
                out.append({"name": nm, "env": dict(v["env"]),
                            "flags": v["flags"] + (alt.split() if alt else [])})
        variants = out
    return variants


def run_variant(case: str, variant: dict, size: int, runs: int, timeout: float,
                outdir: Path, mem_limit_mb: int) -> dict:
    toml_path = TOML_DIR / f"{case}.toml"
    env = os.environ.copy()
    env.update({"FRACTALL_NO_AUTO_ADJUST": "1", "FRACTALL_NO_PERIOD": "1",
                "FRACTALL_PERTURB_STATS": "1", "FRACTALL_WISDOM": "1"})
    env.update(variant["env"])
    slug = "".join(ch if ch.isalnum() else "_" for ch in variant["name"]) or "base"
    png = outdir / f"{case}__{slug}.png"
    cmd = [str(CLI), "--toml", str(toml_path), "--width", str(size),
           "--height", str(size), "--bailout", "25", "--output", str(png),
           *variant["flags"]]
    walls: list[float] = []
    prof = None
    for _ in range(runs):
        jrec = journal_begin("ab", case, {"variant": variant["name"]})
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, text=True, errors="replace", timeout=timeout,
                               preexec_fn=_rlimit_preexec(mem_limit_mb))
        except subprocess.TimeoutExpired:
            journal_end(jrec, "timeout")
            return {"status": "timeout"}
        wall = time.monotonic() - t0
        st = _classify_rc(p.returncode)
        journal_end(jrec, st)
        if st != "ok":
            return {"status": st, "log": (p.stdout or "")[-2000:]}
        walls.append(wall)
        prof = parse_profile_output(p.stdout or "")   # dernier run = profil
    med = statistics.median(walls)
    prof = prof or {}
    prof.update(phase_breakdown(prof, med))
    try:
        digest = __import__("hashlib").sha256(png.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = None
    return {"status": "ok", "wall_median": round(med, 4),
            "wall_min": round(min(walls), 4),
            "spread": (round((max(walls) - min(walls)) / med, 4)
                       if len(walls) > 1 and med > 0 else None),
            "runs": [round(w, 4) for w in walls], "png_sha256_16": digest,
            "path": prof.get("path"), "ns_per_iter": prof.get("ns_per_iter"),
            "secs": prof.get("secs"), "share_pct": prof.get("share_pct"),
            "reliability_flagged_pct": prof.get("reliability_flagged_pct"),
            "dd_escalation": prof.get("dd_escalation")}


def ab_table(case: str, rows: list[tuple[dict, dict]]) -> list[str]:
    """Markdown : une ligne par variante, ratio vs la PREMIÈRE variante (A)."""
    L = [f"### {case}", "",
         "| Variante | wall méd. (s) | min | bruit | vs A | path | ns/iter | "
         "orbite | pixels | post | fiab. % | sortie ≡ A |",
         "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    ref = next((r for _, r in rows if r.get("status") == "ok"), None)
    for v, r in rows:
        if r.get("status") != "ok":
            L.append(f"| {v['name']} | — ({r.get('status')}) | | | | | | | | | | |")
            continue
        vs = (f"{r['wall_median'] / ref['wall_median']:.2f}×"
              if ref and ref["wall_median"] > 0 else "—")
        same = ("oui" if ref and r["png_sha256_16"] == ref["png_sha256_16"]
                else "**NON**")
        s = r.get("secs") or {}
        sp = r.get("spread")
        rel = r.get("reliability_flagged_pct")
        L.append(f"| {v['name']} | {r['wall_median']:.3f} | {r['wall_min']:.3f} | "
                 f"{(f'{100*sp:.0f} %' if sp is not None else '—')} | {vs} | "
                 f"{r.get('path') or '?'} | "
                 f"{r['ns_per_iter'] if r.get('ns_per_iter') is not None else '—'} | "
                 f"{s.get('orbit', 0):.3f} | {s.get('pixels', 0):.3f} | "
                 f"{s.get('post', 0):.3f} | "
                 f"{(f'{rel:.2f}' if rel is not None else '—')} | {same} |")
    return L


def cmd_ab(args) -> None:
    variants = parse_variants(args.env or [], args.flags or [])
    if len(variants) < 2:
        sys.exit("il faut ≥ 2 variantes : --env K=a,b et/ou --flags '--x|'")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    size, runs, timeout = args.size or 256, args.runs or 3, args.timeout or 600.0
    if not args.no_rebuild and not CLI.exists():
        cargo_build("fractall-cli")
    outdir = BENCH / "ab"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"→ A/B {len(cases)} cas × {len(variants)} variantes · {size}² · "
          f"runs={runs} (médiane) · A = {variants[0]['name']}")
    report = {"date_utc": _now_iso(), "git_sha": git_sha(), "machine": machine_info(),
              "size": size, "runs": runs, "cases": {}}
    md = [f"# A/B — {_now_iso()} (`{git_sha()}`)", "",
          f"- {size}² · runs={runs} (médiane) · machine {machine_info()['cpu']} · "
          f"{machine_info()['nproc']} threads",
          "- `vs A` = wall médian / wall de la première variante ; `sortie ≡ A` = "
          "octets PNG identiques (sinon : tradeoff correction, adjuger vs GMP).", ""]
    for case in cases:
        if not (TOML_DIR / f"{case}.toml").exists():
            print(f"  {case}: toml absent — skip")
            continue
        rows: list[tuple[dict, dict]] = []
        for v in variants:
            print(f"  {case:18s} {v['name']:40s}", end=" ", flush=True)
            r = run_variant(case, v, size, runs, timeout, outdir, 0)
            rows.append((v, r))
            if r.get("status") == "ok":
                print(f"wall={r['wall_median']:.3f}s path={r.get('path')} "
                      f"post={(r.get('secs') or {}).get('post', 0):.3f}s")
            else:
                print(f"✗ {r.get('status')}")
        report["cases"][case] = [{"variant": v["name"], **r} for v, r in rows]
        md += ab_table(case, rows) + [""]
    (outdir / "ab.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (outdir / "ab.md").write_text("\n".join(md) + "\n")
    print()
    print("\n".join(md))
    print(f"✓ rapport : {(outdir / 'ab.md').relative_to(BENCH.parent.parent)}")
