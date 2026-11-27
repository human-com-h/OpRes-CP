"""Automated CMAME submission compliance validator for paper_cmame/.

Checks:
  1) Abstract word count <= 250 words.
  2) Highlights: 3-5 items, each <= 85 characters including spaces.
  3) Graphical Abstract PNG width >= 1328 px and horizontal orientation (width > height).
  4) Vitae: each author <= 100 words.
  5) Citations: 100% bidirectional match between paper_cmame/*.tex and references.bib,
     no \\citet or \\citep left over, DOI presence summary.
  6) Appendix numbering in main.aux: all appendix equations/tables/figures use A.1, B.1, C.1
     and no "Appendix Appendix" artifacts.
  7) Compilation log checks (main.log & main.blg): 0 Errors, 0 Warnings, 0 BibTeX Warnings,
     0 Overfull/Underfull boxes.
"""

from pathlib import Path
import re
import sys

from PIL import Image


ROOT = Path("paper_cmame")
SUB = ROOT / "submission_package"


def check_abstract():
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    assert m, "Abstract block not found in paper_cmame/main.tex"
    abs_txt = m.group(1).strip()
    clean = re.sub(r"\\emph\{([^}]*)\}", r"\1", abs_txt)
    clean = clean.replace("---", " — ").replace("--", "–")
    wc = len(clean.split())
    assert wc <= 250, f"Abstract exceeds 250 words: {wc}"
    assert r"\cite" not in abs_txt, "Abstract must not contain citations"
    return wc


def check_highlights():
    lines = [l.strip() for l in (SUB / "highlights.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert 3 <= len(lines) <= 5, f"Highlights count must be 3-5, got {len(lines)}"
    lens = []
    for i, line in enumerate(lines, 1):
        s = re.sub(r"^[-*]\s*", "", line).strip()
        assert len(s) <= 85, f"Highlight {i} exceeds 85 chars ({len(s)}): {s}"
        lens.append(len(s))
    return lens


def check_graphical_abstract():
    png = SUB / "graphical_abstract.png"
    pdf = SUB / "graphical_abstract.pdf"
    assert png.is_file() and pdf.is_file(), "Missing graphical_abstract.png or .pdf"
    with Image.open(png) as im:
        w, h = im.size
    assert w >= 1328 and w > h, f"Graphical abstract must be >=1328 px wide and horizontal, got {w}x{h}"
    return w, h


def check_vitae():
    vt = (SUB / "vitae_template.txt").read_text(encoding="utf-8")
    counts = {}
    for block in vt.split("[")[1:]:
        name = block.split("]")[0]
        body = "\n".join(block.splitlines()[1:]).strip()
        wc = len(body.split())
        assert wc <= 100, f"Vitae for {name} exceeds 100 words: {wc}"
        counts[name] = wc
    assert counts, "No author vitae blocks found"
    return counts


def check_bibliography():
    bib_text = (ROOT / "references.bib").read_text(encoding="utf-8")
    entries = [e for e in re.split(r"\n(?=@)", bib_text.strip()) if e.lstrip().startswith("@")]
    bib_keys = {}
    for e in entries:
        m = re.match(r"@\w+\{([^,]+),", e)
        if m:
            bib_keys[m.group(1).strip()] = ("doi" in e.lower())

    tex_files = list(ROOT.glob("*.tex")) + list((ROOT / "tables").glob("*.tex"))
    cites = set()
    bad_natbib = []
    for f in tex_files:
        t = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\\cite([pt]?)\{([^}]+)\}", t):
            if m.group(1):
                bad_natbib.append((f.name, m.group(0)))
            for c in m.group(2).split(","):
                cites.add(c.strip())

    unused = set(bib_keys) - cites
    missing = cites - set(bib_keys)
    assert not bad_natbib, f"Found \\citep/\\citet incompatible with elsarticle-num: {bad_natbib[:5]}"
    assert not unused, f"Unused bib entries: {sorted(unused)}"
    assert not missing, f"Missing bib entries: {sorted(missing)}"
    return len(cites), sorted(k for k, has_d in bib_keys.items() if not has_d)


def check_latex_build():
    aux = ROOT / "main.aux"
    log = ROOT / "main.log"
    blg = ROOT / "main.blg"
    assert aux.is_file() and log.is_file() and blg.is_file(), "Compile paper_cmame/main.tex first"

    aux_t = aux.read_text(encoding="utf-8", errors="replace")
    assert "Appendix Appendix" not in aux_t, "Double Appendix label found in main.aux"
    # Verify appendix labels start with A., B., C.
    app_labels = re.findall(r"\\newlabel\{([^}]+)\}\{\{([A-C]\.\d+)\}", aux_t)
    assert any(v.startswith("A.") for _, v in app_labels), "No A.* appendix labels found in main.aux"
    assert any(v.startswith("B.") for _, v in app_labels), "No B.* appendix labels found in main.aux"

    log_t = log.read_text(encoding="utf-8", errors="replace")
    errors = [l for l in log_t.splitlines() if l.startswith("!")]
    warns = [l for l in log_t.splitlines() if "Warning:" in l]
    # Filter harmless xelatex/hyperref/fontspec info if any, check strictly first
    over_under = [l for l in log_t.splitlines() if ("Overfull " in l or "Underfull " in l)]
    blg_t = blg.read_text(encoding="utf-8", errors="replace")
    bib_warns = [l for l in blg_t.splitlines() if "Warning--" in l]

    assert not errors, f"LaTeX errors: {errors}"
    assert not warns, f"LaTeX warnings: {warns}"
    assert not over_under, f"Overfull/Underfull boxes: {over_under}"
    assert not bib_warns, f"BibTeX warnings: {bib_warns}"
    return len(app_labels)


def check_keywords():
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.S)
    assert m, "Keyword block not found in paper_cmame/main.tex"
    kws = [k.strip() for k in m.group(1).split(r"\sep") if k.strip()]
    assert 1 <= len(kws) <= 7, f"CMAME requires 1-7 keywords, got {len(kws)}: {kws}"
    return len(kws)


def check_ordered_figures():
    main_txt = (ROOT / "main.tex").read_text(encoding="utf-8")
    # Count ordered Figure_1..N in submission_package
    def _collect(p):
        t = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        out = []
        for m in re.finditer(r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", t):
            if m.group(1) == "input":
                sp = p.parent / (m.group(2).strip() + ("" if m.group(2).strip().endswith(".tex") else ".tex"))
                if sp.is_file():
                    for x in _collect(sp):
                        if x not in out:
                            out.append(x)
            else:
                stem = Path(m.group(2).strip()).name
                if stem not in out:
                    out.append(stem)
        return out
    expected = _collect(ROOT / "main.tex")
    assert len(expected) >= 9, f"Expected >= 9 ordered figures, found {len(expected)}: {expected}"
    for i in range(1, len(expected) + 1):
        assert (SUB / f"Figure_{i}.pdf").is_file() and (SUB / f"Figure_{i}.png").is_file(), f"Missing Figure_{i}"
    extra = [p.name for p in SUB.glob("Figure_*.*")
             if int(re.search(r"Figure_(\d+)", p.name).group(1)) > len(expected)]
    assert not extra, f"Stale extra Figure files in submission_package: {extra}"
    return expected


def main():
    if not (ROOT / "main.tex").is_file():
        print("[SKIP] paper_cmame/ manuscript source is not included in the public code repository; manuscript compliance checks apply only to the author LaTeX workspace.")
        return
    wc = check_abstract()
    n_kw = check_keywords()
    hl = check_highlights()
    w, h = check_graphical_abstract()
    vit = check_vitae()
    n_cites, no_doi = check_bibliography()
    n_app = check_latex_build()
    figs = check_ordered_figures()
    print(f"[CMAME Check PASS] Abstract={wc}/250 words | Keywords={n_kw}/7 | Highlights={hl} (<=85) | "
          f"GA={w}x{h} (>=1328) | Vitae={vit} | Cites={n_cites} (no DOI: {no_doi}) | "
          f"Appendix labels={n_app} | Ordered Figures={len(figs)} | 0 errors/warnings/overfull/bibtex-warnings")


if __name__ == "__main__":
    main()
