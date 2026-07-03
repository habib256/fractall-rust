#!/usr/bin/env python3
"""Génère `bench/viewer.html` — interface visuelle pour comparer F3 vs Fractall.

Scanne `bench/compare/` pour les triplets `<case>__{f3,fr,diff}.png`, joint avec
les métriques de `_summary.csv` quand dispo, lit `toml/<case>.toml` pour zoom/
iterations. Tout est embarqué dans un seul HTML self-contained ouvrable via
`open bench/viewer.html` (les PNG sont référencés en chemin relatif).

Usage:
  python3 scripts/build_viewer.py

Re-lancer après chaque exécution de `compare_f3.py`.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPARE = ROOT / "bench" / "compare"
TOML_DIR = ROOT / "toml"
OUT = ROOT / "bench" / "viewer.html"


def parse_toml_minimal(path: Path) -> dict:
    """Extrait zoom / iterations / rotate sans dépendance toml."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r'\s*(zoom|iterations|rotate|real|imag)\s*=\s*(.+?)\s*$', line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        out[key] = val
    return out


def load_summary() -> dict[str, dict]:
    csv_path = COMPARE / "_summary.csv"
    if not csv_path.exists():
        return {}
    rows = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["name"]] = row
    return rows


def discover_cases() -> list[str]:
    cases = set()
    for p in COMPARE.glob("*__f3.png"):
        cases.add(p.stem.replace("__f3", ""))
    return sorted(cases)


def build_manifest() -> list[dict]:
    summary = load_summary()
    cases = []
    for name in discover_cases():
        toml = parse_toml_minimal(TOML_DIR / f"{name}.toml")
        m = summary.get(name, {})
        cases.append({
            "name": name,
            "f3": f"compare/{name}__f3.png",
            "fr": f"compare/{name}__fr.png",
            "diff": f"compare/{name}__diff.png",
            "zoom": toml.get("zoom"),
            "iterations": toml.get("iterations"),
            "rotate": toml.get("rotate"),
            "mean_abs_dsi": float(m["mean_abs_dsi"]) if m.get("mean_abs_dsi") else None,
            "max_abs_dsi": float(m["max_abs_dsi"]) if m.get("max_abs_dsi") else None,
            "rms_dsi": float(m["rms_dsi"]) if m.get("rms_dsi") else None,
            "inside_mismatch": int(m["inside_mismatch"]) if m.get("inside_mismatch") else None,
            "px_total": int(m["px_total"]) if m.get("px_total") else None,
            "f3_sec": m.get("f3_sec"),
            "fr_sec": m.get("fr_sec"),
            "iterations_used": int(m["iterations"]) if m.get("iterations") and m["iterations"].isdigit() else None,
        })
    return cases


HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>F3 vs Fractall — viewer</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; background: #111; color: #ddd; }
  #app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  #sidebar { background: #161616; border-right: 1px solid #2a2a2a; overflow-y: auto; }
  #sidebar header { padding: 12px 14px; border-bottom: 1px solid #2a2a2a; position: sticky; top: 0; background: #161616; z-index: 2; }
  #sidebar h1 { margin: 0 0 6px; font-size: 13px; letter-spacing: 0.5px; color: #eee; }
  #sidebar .sub { font-size: 11px; color: #888; }
  #sortbar { padding: 8px 14px; display: flex; gap: 6px; flex-wrap: wrap; border-bottom: 1px solid #2a2a2a; }
  #sortbar button { background: #222; color: #bbb; border: 1px solid #333; padding: 3px 8px; font-size: 11px; cursor: pointer; font-family: inherit; border-radius: 3px; }
  #sortbar button.active { background: #2a4060; color: #fff; border-color: #3a6090; }
  #caselist { list-style: none; margin: 0; padding: 0; }
  #caselist li { padding: 8px 14px; border-bottom: 1px solid #1f1f1f; cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  #caselist li:hover { background: #1d1d1d; }
  #caselist li.active { background: #233048; }
  #caselist .nm { font-size: 12px; color: #ddd; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #caselist .badge { font-size: 10px; padding: 2px 6px; border-radius: 8px; flex-shrink: 0; }
  .b-pass { background: #1a4a1a; color: #8fe88f; }
  .b-warn { background: #5a4a1a; color: #f0d080; }
  .b-fail { background: #5a1a1a; color: #f08080; }
  .b-none { background: #333; color: #888; }
  #main { display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
  #header { padding: 12px 18px; border-bottom: 1px solid #2a2a2a; background: #181818; }
  #header h2 { margin: 0 0 6px; font-size: 14px; color: #eee; }
  #meta { font-size: 11px; color: #aaa; display: flex; gap: 16px; flex-wrap: wrap; }
  #meta b { color: #eee; font-weight: 600; }
  #viewbar { padding: 8px 18px; background: #181818; border-bottom: 1px solid #2a2a2a; display: flex; gap: 8px; align-items: center; }
  #viewbar button { background: #222; color: #bbb; border: 1px solid #333; padding: 4px 10px; font-size: 12px; cursor: pointer; font-family: inherit; border-radius: 3px; }
  #viewbar button.active { background: #2a4060; color: #fff; border-color: #3a6090; }
  #viewbar .hint { font-size: 11px; color: #666; margin-left: auto; }
  #stage { position: relative; overflow: auto; background: #0a0a0a; display: flex; align-items: center; justify-content: center; padding: 20px; }
  .mode-side { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
  .mode-side .pane { display: flex; flex-direction: column; align-items: center; gap: 6px; }
  .mode-side .pane label { font-size: 11px; color: #888; }
  .mode-side .pane img { max-width: 100%; max-height: 80vh; border: 1px solid #333; image-rendering: pixelated; }
  .mode-wipe { position: relative; display: inline-block; line-height: 0; }
  .mode-wipe img { display: block; max-height: 80vh; image-rendering: pixelated; }
  .mode-wipe .top { position: absolute; top: 0; left: 0; clip-path: inset(0 50% 0 0); pointer-events: none; }
  .mode-wipe .divider { position: absolute; top: 0; bottom: 0; width: 2px; background: #ff3; left: 50%; transform: translateX(-50%); pointer-events: none; }
  .mode-wipe .label-l, .mode-wipe .label-r { position: absolute; top: 6px; font-size: 11px; padding: 2px 6px; background: rgba(0,0,0,0.7); color: #fff; pointer-events: none; border-radius: 2px; }
  .mode-wipe .label-l { left: 6px; }
  .mode-wipe .label-r { right: 6px; }
  .mode-diff img, .mode-blink img { max-width: 100%; max-height: 85vh; border: 1px solid #333; image-rendering: pixelated; }
  .mode-blink { position: relative; display: inline-block; line-height: 0; }
  .mode-blink .top { position: absolute; top: 0; left: 0; opacity: 0; animation: blink 1.4s infinite; }
  @keyframes blink { 0%, 49% { opacity: 0; } 50%, 99% { opacity: 1; } }
  .empty { color: #555; padding: 40px; text-align: center; font-size: 13px; }
  kbd { background: #2a2a2a; padding: 1px 5px; border-radius: 3px; font-size: 10px; border: 1px solid #3a3a3a; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <header>
      <h1>F3 vs Fractall</h1>
      <div class="sub" id="case-count"></div>
    </header>
    <div id="sortbar">
      <button data-sort="mean" class="active">Δmean ↓</button>
      <button data-sort="max">Δmax ↓</button>
      <button data-sort="name">A→Z</button>
      <button data-sort="zoom">zoom ↑</button>
    </div>
    <ul id="caselist"></ul>
  </aside>
  <section id="main">
    <div id="header">
      <h2 id="case-title">—</h2>
      <div id="meta"></div>
    </div>
    <div id="viewbar">
      <button data-mode="side" class="active">Side-by-side</button>
      <button data-mode="wipe">Wipe</button>
      <button data-mode="blink">Blink</button>
      <button data-mode="diff">Diff strip</button>
      <span class="hint"><kbd>↑</kbd><kbd>↓</kbd> cas — <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd> mode</span>
    </div>
    <div id="stage"><div class="empty">Sélectionne un cas dans la liste.</div></div>
  </section>
</div>

<script>
const CASES = __MANIFEST__;
let currentIdx = -1;
let currentMode = "side";
let currentSort = "mean";

function fmt(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return n.toFixed(digits);
}

function badge(c) {
  const v = c.mean_abs_dsi;
  if (v == null) return '<span class="badge b-none">—</span>';
  let cls = "b-fail";
  if (v <= 1) cls = "b-pass";
  else if (v <= 100) cls = "b-warn";
  return `<span class="badge ${cls}">${fmt(v)}</span>`;
}

function zoomExp(s) {
  if (!s) return null;
  const m = String(s).match(/[Ee]([+-]?\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

function sortedCases() {
  const cs = [...CASES];
  if (currentSort === "name") cs.sort((a, b) => a.name.localeCompare(b.name));
  else if (currentSort === "zoom") cs.sort((a, b) => (zoomExp(a.zoom) ?? -1) - (zoomExp(b.zoom) ?? -1));
  else if (currentSort === "max") cs.sort((a, b) => (b.max_abs_dsi ?? -1) - (a.max_abs_dsi ?? -1));
  else cs.sort((a, b) => (b.mean_abs_dsi ?? -1) - (a.mean_abs_dsi ?? -1));
  return cs;
}

function renderList() {
  const ul = document.getElementById("caselist");
  const cs = sortedCases();
  ul.innerHTML = cs.map((c, i) => `
    <li data-name="${c.name}">
      <span class="nm">${c.name}</span>
      ${badge(c)}
    </li>`).join("");
  ul.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectByName(li.dataset.name));
  });
  document.getElementById("case-count").textContent = `${cs.length} cas`;
  highlightActive();
}

function highlightActive() {
  const name = currentIdx >= 0 ? CASES[currentIdx].name : null;
  document.querySelectorAll("#caselist li").forEach(li => {
    li.classList.toggle("active", li.dataset.name === name);
  });
}

function selectByName(name) {
  const idx = CASES.findIndex(c => c.name === name);
  if (idx >= 0) selectIdx(idx);
}

function selectIdx(idx) {
  currentIdx = idx;
  const c = CASES[idx];
  document.getElementById("case-title").textContent = c.name;
  const m = document.getElementById("meta");
  const parts = [];
  if (c.zoom) parts.push(`<span><b>zoom</b> ${c.zoom}</span>`);
  if (c.iterations) parts.push(`<span><b>iter</b> ${c.iterations}</span>`);
  if (c.rotate) parts.push(`<span><b>rotate</b> ${c.rotate}°</span>`);
  if (c.mean_abs_dsi != null) parts.push(`<span><b>Δmean</b> ${fmt(c.mean_abs_dsi)}</span>`);
  if (c.max_abs_dsi != null) parts.push(`<span><b>Δmax</b> ${fmt(c.max_abs_dsi)}</span>`);
  if (c.rms_dsi != null) parts.push(`<span><b>rms</b> ${fmt(c.rms_dsi)}</span>`);
  if (c.inside_mismatch != null) parts.push(`<span><b>inside_mm</b> ${c.inside_mismatch}/${c.px_total}</span>`);
  if (c.f3_sec) parts.push(`<span><b>F3</b> ${c.f3_sec}</span>`);
  if (c.fr_sec) parts.push(`<span><b>Fr</b> ${c.fr_sec}</span>`);
  m.innerHTML = parts.join(" ");
  highlightActive();
  renderStage();
}

function renderStage() {
  const stage = document.getElementById("stage");
  if (currentIdx < 0) {
    stage.innerHTML = '<div class="empty">Sélectionne un cas.</div>';
    return;
  }
  const c = CASES[currentIdx];
  if (currentMode === "side") {
    stage.innerHTML = `
      <div class="mode-side">
        <div class="pane"><label>F3 (référence)</label><img src="${c.f3}" alt="F3"></div>
        <div class="pane"><label>Fractall</label><img src="${c.fr}" alt="Fractall"></div>
      </div>`;
  } else if (currentMode === "wipe") {
    stage.innerHTML = `
      <div class="mode-wipe" id="wipe">
        <img src="${c.fr}" alt="Fractall">
        <img src="${c.f3}" alt="F3" class="top">
        <div class="divider"></div>
        <span class="label-l">F3</span>
        <span class="label-r">Fractall</span>
      </div>`;
    setupWipe();
  } else if (currentMode === "blink") {
    stage.innerHTML = `
      <div class="mode-blink">
        <img src="${c.fr}" alt="Fractall">
        <img src="${c.f3}" alt="F3" class="top">
      </div>`;
  } else {
    stage.innerHTML = `<div class="mode-diff"><img src="${c.diff}" alt="diff"></div>`;
  }
}

function setupWipe() {
  const w = document.getElementById("wipe");
  if (!w) return;
  const top = w.querySelector(".top");
  const div = w.querySelector(".divider");
  const move = (e) => {
    const rect = w.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    top.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    div.style.left = pct + "%";
  };
  w.addEventListener("mousemove", move);
  w.addEventListener("touchmove", move, { passive: true });
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll("#viewbar button").forEach(b => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });
  renderStage();
}

function setSort(sort) {
  currentSort = sort;
  document.querySelectorAll("#sortbar button").forEach(b => {
    b.classList.toggle("active", b.dataset.sort === sort);
  });
  renderList();
}

document.querySelectorAll("#viewbar button").forEach(b => {
  b.addEventListener("click", () => setMode(b.dataset.mode));
});
document.querySelectorAll("#sortbar button").forEach(b => {
  b.addEventListener("click", () => setSort(b.dataset.sort));
});

document.addEventListener("keydown", (e) => {
  const cs = sortedCases();
  const visIdx = currentIdx < 0 ? -1 : cs.findIndex(c => c.name === CASES[currentIdx].name);
  if (e.key === "ArrowDown") {
    if (visIdx < cs.length - 1) selectByName(cs[visIdx + 1].name);
    e.preventDefault();
  } else if (e.key === "ArrowUp") {
    if (visIdx > 0) selectByName(cs[visIdx - 1].name);
    e.preventDefault();
  } else if (e.key === "1") setMode("side");
  else if (e.key === "2") setMode("wipe");
  else if (e.key === "3") setMode("blink");
  else if (e.key === "4") setMode("diff");
});

renderList();
if (CASES.length) selectIdx(CASES.findIndex(c => (c.mean_abs_dsi ?? -1) >= 0) >= 0 ? 0 : 0);
</script>
</body>
</html>
"""


def main():
    cases = build_manifest()
    if not cases:
        print(f"Aucun cas trouvé dans {COMPARE}. Lance d'abord scripts/compare_f3.py.")
        return
    payload = json.dumps(cases, ensure_ascii=False)
    html = HTML.replace("__MANIFEST__", payload)
    OUT.write_text(html, encoding="utf-8")
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    print(f"Écrit: {OUT.relative_to(ROOT)} ({len(cases)} cas)")
    print(f"Ouvre avec: {opener} {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
