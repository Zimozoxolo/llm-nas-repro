import zipfile, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = Path.home() / "Downloads"
SKIP_EXT = {".aux", ".log", ".out", ".blg", ".bcf", ".run.xml", ".fls", ".fdb_latexmk", ".toc", ".xdv", ".synctex.gz"}

NOTES = {
 "tmlr": "TMLR submission version (double-blind). Compiler: pdfLaTeX. Main file: main.tex.\nAuthors are hidden in submission mode; use \\usepackage[accepted]{tmlr} for camera-ready.\nFill nothing before review; see README in the repository for open items.\n",
 "gpem": "Genetic Programming and Evolvable Machines (Springer Nature sn-jnl). Compiler: pdfLaTeX. Main file: main.tex.\nFill every [ ... ] placeholder (affiliation, e-mail, Declarations) before submission.\n",
 "rescience": "ReScience C. Compiler: XeLaTeX with Biber (latexmkrc sets this; otherwise Menu > Compiler > XeLaTeX). Main file: article.tex.\nEdit metadata.yaml, then regenerate metadata.tex locally with: python yaml-to-latex.py -i metadata.yaml -o metadata.tex\n(Overleaf cannot run the script; edit metadata.tex by hand there.)\n",
}

def keep(p: Path):
    n = p.name
    if any(n.endswith(e) for e in SKIP_EXT):
        return False
    if n in ("README.md", "Makefile") and p.parent.name == "rescience":
        return n == "Makefile"
    return True

for name, zname in (("tmlr", "LLM-NAS_TMLR_overleaf.zip"), ("gpem", "LLM-NAS_GPEM_overleaf.zip"),
                    ("rescience", "LLM-NAS_ReScienceC_overleaf.zip")):
    d = HERE / name
    if name == "rescience":
        (d / "latexmkrc").write_text("$pdf_mode = 5;\n$bibtex_use = 2;\n", encoding="utf-8")
    zp = OUT / zname
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.rglob("*")):
            if p.is_file() and keep(p) and not p.name.endswith(".pdf"):
                z.write(p, p.relative_to(d).as_posix())
        z.writestr("OVERLEAF_README.txt", NOTES[name])
    print(zname, round(zp.stat().st_size / 1e6, 2), "MB")
