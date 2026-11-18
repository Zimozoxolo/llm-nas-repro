"""Zip the impact-factor journal versions for Overleaf upload and copy their PDFs to Downloads."""
import shutil, sys, zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\siyab\Downloads\LLM-NAS_journal_submissions")
OUT.mkdir(exist_ok=True)
SKIP = (".aux", ".log", ".out", ".blg", ".fls", ".fdb_latexmk", ".toc", ".spl", ".synctex.gz")

JOURNALS = {
    "tevc": ("IEEE_TEVC", "IEEE Transactions on Evolutionary Computation. Compiler: pdfLaTeX. Main file: main.tex.\n"
             "Upload supplementary.pdf (compile supplementary.tex) as supplemental material.\n"),
    "swevo": ("SwarmEvolComput", "Swarm and Evolutionary Computation (Elsevier). Compiler: pdfLaTeX. Main file: main.tex.\n"
              "Upload highlights.txt as the Highlights file.\n"),
    "asoc": ("AppliedSoftComput", "Applied Soft Computing (Elsevier). Compiler: pdfLaTeX. Main file: main.tex.\n"
             "Upload highlights.txt and graphical_abstract.pdf (or .png) as separate files; a graphical abstract is required.\n"),
    "neunet": ("NeuralNetworks", "Neural Networks (Elsevier), author-year references. Compiler: pdfLaTeX. Main file: main.tex.\n"
               "Upload highlights.txt as the Highlights file.\n"),
    "neucom": ("Neurocomputing", "Neurocomputing (Elsevier). Compiler: pdfLaTeX. Main file: main.tex.\n"
               "Upload highlights.txt as the Highlights file.\n"),
    "mlj": ("MachineLearning", "Machine Learning (Springer, sn-jnl author-year). Compiler: pdfLaTeX. Main file: main.tex.\n"
            "No subfolders, as Springer's submission system requires.\n"),
    "gpem": ("GPEM", "Genetic Programming and Evolvable Machines (Springer, sn-jnl numbered). Compiler: pdfLaTeX. Main file: main.tex.\n"
             "No subfolders, as Springer's submission system requires.\n"),
}
NOTE = "Fill every [ ... ] placeholder before submission (author details, declarations, repository URL).\n"

if (HERE / "graphical_abstract" / "graphical_abstract.png").exists():
    shutil.copy2(HERE / "graphical_abstract" / "graphical_abstract.png", HERE / "asoc" / "graphical_abstract.png")

for key, (label, note) in JOURNALS.items():
    d = HERE / key
    zp = OUT / f"LLM-NAS_{label}_overleaf.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            if p.is_file() and not p.name.endswith(SKIP) and p.suffix != ".pdf" or p.name in ("graphical_abstract.pdf",):
                z.write(p, p.name)
        z.writestr("OVERLEAF_README.txt", note + NOTE)
    shutil.copy2(d / "main.pdf", OUT / f"LLM-NAS_{label}.pdf")
    if key == "tevc":
        shutil.copy2(d / "supplementary.pdf", OUT / "LLM-NAS_IEEE_TEVC_supplementary.pdf")
    print(f"{zp.name}: {zp.stat().st_size / 1e6:.2f} MB")
shutil.copy2(HERE / "graphical_abstract" / "graphical_abstract.png", OUT / "LLM-NAS_graphical_abstract.png")
shutil.copy2(HERE / "README_impact_factor_journals.md", OUT / "README.md")
print("->", OUT)
