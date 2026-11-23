"""Second batch of journal versions (journals with a Clarivate Journal Impact Factor).

Targets (all flat folders, no subfolders, as Springer and Elsevier upload systems prefer):
  tevc/    IEEE Transactions on Evolutionary Computation (IEEEtran, two-column, anonymised)
  swevo/   Swarm and Evolutionary Computation (elsarticle, numbered)
  asoc/    Applied Soft Computing (elsarticle, numbered, graphical abstract required)
  neunet/  Neural Networks (elsarticle, author-year / APA-like)
  neucom/  Neurocomputing (elsarticle, numbered)
  mlj/     Machine Learning (Springer sn-jnl, author-year)
  gpem/    Genetic Programming and Evolvable Machines (Springer sn-jnl, numbered) - rebuilt flat
Run from report/journals:  python -I build_more.py
"""
import re, shutil, sys, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_versions as bv  # noqa: E402
import authors as au  # noqa: E402

FIGS = ("cnn_fronts_edgegpu.png", "hv_floor.png", "fronts_cifar100.png")
SEP = " \\sep "
FLAT_PKGS = (bv.PREAMBLE_PKGS.replace("\\graphicspath{{figures/}}\n", "")
             + "\\usepackage{adjustbox}\n\\setlength{\\emergencystretch}{3em}\n")

HIGHLIGHTS = [
    "Independent reimplementation of LLM-NAS from its text, algorithm and prompts",
    "Eleven of twelve reported architectures recovered; one does not exist",
    "Hypervolume rankings hinge on one degenerate point and on normalisation",
    "No LLM run beat random-operator evolution across six devices (0 of 24)",
    "The training-free evaluator, not the LLM, explains the cost savings",
]
assert 3 <= len(HIGHLIGHTS) <= 5 and all(len(h) <= 85 for h in HIGHLIGHTS), [len(h) for h in HIGHLIGHTS]

AI_ELSEVIER = (
    "During the preparation of this work the first author used Claude Code (Anthropic), an AI coding assistant, "
    "to write and run the implementation and analysis code and to draft and edit parts of the manuscript text. "
    "After using this tool, the authors reviewed and edited the content as needed and take full responsibility "
    "for the content of the published article."
)
DATA_STMT = (
    "The benchmark data are public: HW-NAS-Bench, NAS-Bench-Suite-Zero and NATS-Bench. The code, "
    "configurations and archived run logs behind every number are available at [repository URL and archive DOI]."
)

TEVC_TITLE = "Does the Language Model Matter? Reproducing Language-Model-Driven Hardware-Aware Neural Architecture Search"
TEVC_ABSTRACT = (
    "A recent method for hardware-aware neural architecture search replaces the evolutionary variation operator "
    "with a large language model and reports near-optimal accuracy-latency trade-offs from 120 model queries per "
    "search. Because no reference implementation was available, we reimplemented the method from its published "
    "description and logged every model query for exact replay. Validation of the benchmark data recovered eleven "
    "of the twelve architectures reported by the original authors; one reported point does not exist in the "
    "benchmark. We then compared searches driven by five language models with controls that use no language model "
    "but share the same training-free evaluator and complexity niches. Front quality, measured by hypervolume and "
    "inverted generational distance, depended strongly on an unstated normalisation: a single degenerate "
    "architecture on the true front compresses all useful architectures into a small part of the accuracy axis. "
    "Across six hardware devices, no language-model run exceeded evolution with a random variation operator, and "
    "one exceeded random sampling. On this small tabulated search space with a cheap evaluator, the reported cost "
    "savings come from the evaluator rather than from the language model; whether the operator pays off with "
    "expensive evaluations remains open."
)
TEVC_KEYWORDS = ["Neural architecture search", "evolutionary computation", "large language models",
                 "multiobjective optimization", "reproducibility"]

CLAUDE_BIB = """
@misc{claudecode,
  author = {{Anthropic}},
  title = {Claude Code},
  year = {2026},
  howpublished = {AI coding assistant, \\url{https://claude.com/claude-code}}
}
"""


def wc(s):
    return len(re.sub(r"\\[a-zA-Z]+|[{}~$]", " ", s).split())


def prep(name, main, repo_sentence):
    d = HERE / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    for f in FIGS:
        shutil.copy2(bv.FIG / f, d / f)
    body = main.replace(bv.REPO_OLD, repo_sentence)
    old_s1 = ("\\State \\textbf{Stage 1:} $K_k\\gets L(\\text{device, dataset, }S_k,\\text{ parents from }P_k"
              "\\text{ with metrics and rationales})$")
    assert body.count(old_s1) == 1
    body = body.replace(old_s1, "\\State \\textbf{Stage 1:} $K_k\\gets L(\\cdot)$, given the device, dataset, niche $S_k$ "
                                "and the parents from $P_k$ with their metrics and rationales")
    body = fit_wide(body, springer=name in ("mlj", "gpem"))
    return d, body


def split_pm(table):
    """Put the '+/- s.d.' part of each cell on a second row so wide tables fit sn-jnl's text width."""
    out = []
    for line in table.split("\n"):
        if "$\\pm$" in line and line.rstrip().endswith("\\\\"):
            cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
            means = [c.split("$\\pm$")[0] for c in cells]
            sds = ["" if i == 0 else ("$\\pm$" + c.split("$\\pm$")[1] if "$\\pm$" in c else "") for i, c in enumerate(cells)]
            out.append(" & ".join(means) + " \\\\")
            out.append(" & ".join(sds) + " \\\\")
        else:
            out.append(line)
    return "\n".join(out)


def fit_wide(s, springer=False):
    """Shrink (never enlarge) tables and TikZ pictures that are wider than the text column."""
    wrap = lambda m: "\\begin{adjustbox}{max width=\\linewidth}\n" + m.group(1) + "\n\\end{adjustbox}"
    if springer:
        # sn-jnl's table environment breaks when a tabular is boxed, so shrink fonts, padding and
        # fixed column widths instead.
        s = re.sub(r"(\\begin\{table\}(?:\[[^\]]*\])?\\centering)\\small",
                   r"\1\\footnotesize\\setlength{\\tabcolsep}{3pt}", s)
        s = re.sub(r"p\{(\d+(?:\.\d+)?)cm\}", lambda m: "p{%.1fcm}" % (float(m.group(1)) * 0.78), s)
        s = s.replace("\\setlength{\\tabcolsep}{4pt}", "\\setlength{\\tabcolsep}{2pt}")
        s = s.replace("Method & Edge~GPU & Raspberry~Pi~4 & Edge~TPU & Pixel~3 & Eyeriss & FPGA \\\\",
                      "Method & EGPU & Pi4 & ETPU & Pix3 & Eyeriss & FPGA \\\\")
        s = s.replace("\\caption{Hypervolume (default convention) on CIFAR-10, all six devices.",
                      "\\caption{Hypervolume (default convention) on CIFAR-10, all six devices (EGPU: Edge GPU; "
                      "Pi4: Raspberry Pi 4; ETPU: Edge TPU; Pix3: Pixel 3).")
        s = re.sub(r"\\begin\{table\}.*?\\end\{table\}",
                   lambda m: split_pm(m.group(0)) if "Method & EGPU" in m.group(0) else m.group(0), s, flags=re.S)
    else:
        s = re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})", wrap, s, flags=re.S)
    s = re.sub(r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})", wrap, s, flags=re.S)
    return s


# ---------------------------------------------------------------- graphical abstract
GA_TEX = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\begin{document}
\begin{tikzpicture}[font=\sffamily\small,>={Stealth[length=2mm]},
  box/.style={draw,rounded corners=2pt,align=center,fill=gray!8,inner sep=4pt,text width=3.4cm,minimum height=2.2cm}]
\node[box] (a) {\textbf{LLM-NAS}\\[2pt]an LLM proposes architectures; a training-free evaluator scores them};
\node[box,right=6mm of a] (b) {\textbf{Reproduction}\\[2pt]same evaluator and niches for LLM and non-LLM operators, six devices};
\draw[->,thick] (a)--(b);
\begin{scope}[shift={(10.4,-1.1)}]
  \def\s{18}
  \foreach \lab/\v/\c [count=\i] in {GPT-4.1/0.928/blue!55, DeepSeek/0.897/blue!55, Qwen/0.891/blue!55, Haiku/0.888/blue!55,
      {Random sampling}/0.950/orange!80, {Random-operator evolution}/0.952/orange!80} {
    \pgfmathsetmacro{\w}{(\v-0.85)*\s}
    \pgfmathsetmacro{\y}{2.3-0.38*\i}
    \fill[\c] (0,\y-0.14) rectangle (\w,\y+0.14);
    \node[anchor=east,font=\sffamily\scriptsize] at (-0.05,\y) {\lab};
    \node[anchor=west,font=\sffamily\scriptsize] at (\w+0.03,\y) {\v};
  }
  \draw (0,-0.05)--(0,2.15);
  \node[font=\sffamily\scriptsize,anchor=north] at (1.0,-0.05) {hypervolume, mean over six devices (CIFAR-10); axis starts at 0.85};
  \node[font=\sffamily\small\bfseries,anchor=south,align=center] at (1.0,2.2) {0 of 24 LLM runs beat random-operator evolution};
\end{scope}
\draw[->,thick] (b.east)--++(0.9,0);
\end{tikzpicture}
\end{document}
"""


def build_ga():
    d = HERE / "graphical_abstract"
    d.mkdir(exist_ok=True)
    (d / "graphical_abstract.tex").write_text(GA_TEX, encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "graphical_abstract.tex"], cwd=d,
                   stdout=subprocess.DEVNULL, check=False)
    # PNG at 300 dpi; must be >= 1328 px wide
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "graphical_abstract.pdf", "graphical_abstract"],
                   cwd=d, check=False)
    return d


# ---------------------------------------------------------------- Elsevier
def build_elsevier(name, journal, main, glossary, authoryear=False, ga=False, max_kw=7):
    d, body = prep(name, main, "All code, configurations and run logs are in the repository given in the Data availability statement.")
    shutil.copy2(HERE / "refs.bib", d / "refs.bib")
    if authoryear:
        body = body.replace("\\cite{", "\\citep{")
    if ga:
        shutil.copy2(HERE / "graphical_abstract" / "graphical_abstract.pdf", d / "graphical_abstract.pdf")
    kws = bv.KEYWORDS[:max_kw]
    style = "elsarticle-harv" if authoryear else "elsarticle-num"
    opts = "preprint,12pt,authoryear" if authoryear else "preprint,12pt"
    hl = "\n".join(f"\\item {h}" for h in HIGHLIGHTS)
    gab = ("\\begin{graphicalabstract}\n\\includegraphics[width=\\linewidth]{graphical_abstract.pdf}\n"
           "\\end{graphicalabstract}\n\n") if ga else ""
    tex = (
        f"% {journal}: Elsevier elsarticle template. Use the 'review' class option for double spacing if requested.\n"
        f"\\documentclass[{opts}]{{elsarticle}}\n"
        + bv.DECL_NOTE + FLAT_PKGS +
        "\\usepackage{hyperref}\n"
        f"\\journal{{{journal}}}\n\n\\begin{{document}}\n\\begin{{frontmatter}}\n\n"
        f"\\title{{{bv.TITLE}}}\n\n"
        + au.elsarticle() + "\n"
        f"\\begin{{abstract}}\n{bv.ABSTRACT}\n\\end{{abstract}}\n\n"
        + gab +
        f"\\begin{{highlights}}\n{hl}\n\\end{{highlights}}\n\n"
        "\\begin{keyword}\n" + SEP.join(kws) + "\n\\end{keyword}\n\n"
        "\\end{frontmatter}\n\n"
        + body +
        "\n\n\\section*{CRediT authorship contribution statement}\n"
        + au.credit_elsevier() + "\n\n"
        "\\section*{Declaration of competing interest}\n"
        "The authors declare that they have no known competing financial interests or personal relationships "
        "that could have appeared to influence the work reported in this paper.\n\n"
        f"\\section*{{Data availability}}\n{DATA_STMT}\n\n"
        "\\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}\n"
        f"{AI_ELSEVIER}\n\n"
        "\\section*{Acknowledgements}\nThis research received no specific funding.\n\n"
        "\\appendix\n" + glossary +
        f"\n\n\\bibliographystyle{{{style}}}\n\\bibliography{{refs}}\n\n\\end{{document}}\n"
    )
    (d / "main.tex").write_text(tex, encoding="utf-8")
    (d / "highlights.txt").write_text("\n".join("- " + h for h in HIGHLIGHTS) + "\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------- Springer (GPEM numbered, MLJ author-year)
def build_springer(name, main, glossary, style, authoryear, heading):
    d, body = prep(name, main, bv.REPO_NEW["gpem"])
    for f in ("sn-jnl.cls", f"{style}.bst"):
        shutil.copy2(HERE / "sn-template" / f, d / f)
    shutil.copy2(HERE / "refs.bib", d / "refs.bib")
    if authoryear:
        body = body.replace("\\cite{", "\\citep{")
    pre = FLAT_PKGS.replace("\\usepackage{amsmath,amssymb}\n", "")
    tex = (
        f"%Springer Nature LaTeX template (sn-jnl), style {style}.\n"
        f"\\documentclass[pdflatex,{style}]{{sn-jnl}}\n"
        + bv.DECL_NOTE + "\\usepackage{amsmath,amssymb,amsfonts}\n" + pre + "\\raggedbottom\n"
        "\\begin{document}\n\n"
        f"\\title[{bv.TITLE_SHORT}]{{{bv.TITLE}}}\n\n"
        + au.sn_jnl() + "\n"
        f"\\abstract{{{bv.ABSTRACT}}}\n\n\\keywords{{{', '.join(bv.KEYWORDS)}}}\n\n\\maketitle\n\n"
        + body +
        "\n\n\\backmatter\n\n"
        f"\\section*{{{heading}}}\n"
        "\\begin{itemize}\n"
        "\\item \\textbf{Funding.} This research received no specific funding.\n"
        "\\item \\textbf{Competing interests.} The authors have no relevant financial or non-financial interests to disclose.\n"
        "\\item \\textbf{Ethics approval and consent to participate.} Not applicable; no human or animal subjects were involved.\n"
        f"\\item \\textbf{{Data availability.}} {DATA_STMT}\n"
        "\\item \\textbf{Code availability.} [Public repository URL and archive DOI, e.g.\\ Zenodo.]\n"
        "\\item \\textbf{Use of large language models.} Large language models are the object of study and are documented in "
        "Sections~\\ref{sec:background}--\\ref{sec:results}. An AI coding assistant (Claude Code) was used to write the "
        "implementation and to draft and edit parts of the text (Section~\\ref{sec:method}); the authors reviewed all code "
        "and text and are accountable for them.\n"
        "\\item \\textbf{Author contributions.} " + au.credit_springer() + "\n"
        "\\end{itemize}\n\n"
        "\\begin{appendices}\n" + glossary + "\n\\end{appendices}\n\n"
        "\\bibliography{refs}\n\n\\end{document}\n"
    )
    (d / "main.tex").write_text(tex, encoding="utf-8")
    return d


# ---------------------------------------------------------------- IEEE TEVC
def build_tevc(main, glossary):
    d, body = prep("tevc", main, "All code, configurations and run logs are provided as anonymised supplementary material.")
    (d / "refs.bib").write_text((HERE / "refs.bib").read_text(encoding="utf-8") + CLAUDE_BIB, encoding="utf-8")
    body = re.sub(r"\\begin\{table\}(\[[^\]]*\])?", r"\\begin{table*}[!t]", body).replace("\\end{table}", "\\end{table*}")
    body = re.sub(r"\\begin\{figure\}(\[[^\]]*\])?", r"\\begin{figure*}[!t]", body).replace("\\end{figure}", "\\end{figure*}")
    body = body.replace("Appendix~\\ref{sec:glossary}", "the supplementary glossary")
    body = body.replace("(Claude Code)", "(Claude Code \\cite{claudecode})", 1)
    assert "sec:glossary" not in body, "glossary reference left in TEVC body"
    tex = (
        "% IEEE Transactions on Evolutionary Computation: IEEEtran journal, two-column, anonymised for review.\n"
        "\\documentclass[journal]{IEEEtran}\n"
        + bv.DECL_NOTE + FLAT_PKGS +
        "\\usepackage{cite}\n\\usepackage{hyperref}\n"
        "\\begin{document}\n"
        f"\\title{{{TEVC_TITLE}}}\n"
        "% Authors' names and affiliations must not appear in the review version.\n"
        "\\author{Anonymous Authors}\n" + au.ieee_camera_ready() +
        "\\maketitle\n"
        f"\\begin{{abstract}}\n{TEVC_ABSTRACT}\n\\end{{abstract}}\n"
        f"\\begin{{IEEEkeywords}}\n{', '.join(TEVC_KEYWORDS)}\n\\end{{IEEEkeywords}}\n\n"
        + body +
        "\n\n\\section*{Acknowledgment}\n"
        "Text throughout this paper was drafted with the assistance of an AI system, Claude Code \\cite{claudecode}, "
        "which was also used to write the implementation; the authors reviewed and revised all text and code and are "
        "accountable for them. This research received no specific funding.\n\n"
        "\\bibliographystyle{IEEEtran}\n\\bibliography{refs}\n\n\\end{document}\n"
    )
    (d / "main.tex").write_text(tex, encoding="utf-8")
    supp = (
        "\\documentclass[11pt,a4paper]{article}\n\\usepackage[margin=2.4cm]{geometry}\n" + FLAT_PKGS +
        "\\begin{document}\n\\title{Supplementary material: glossary of terms and formulas}\n\\date{}\n\\maketitle\n"
        + glossary.replace("\\section{Glossary of terms and formulas}\\label{sec:glossary}", "") +
        "\n\\end{document}\n"
    )
    (d / "supplementary.tex").write_text(supp, encoding="utf-8")
    return d


if __name__ == "__main__":
    main, glossary = bv.load_body()
    print("abstract words (Elsevier/Springer):", wc(bv.ABSTRACT), "| TEVC abstract:", wc(TEVC_ABSTRACT),
          "| TEVC title words:", len(TEVC_TITLE.split()))
    build_ga()
    build_elsevier("swevo", "Swarm and Evolutionary Computation", main, glossary)
    build_elsevier("asoc", "Applied Soft Computing", main, glossary, ga=True)
    build_elsevier("neunet", "Neural Networks", main, glossary, authoryear=True, max_kw=6)
    build_elsevier("neucom", "Neurocomputing", main, glossary)
    build_springer("mlj", main, glossary, "sn-mathphys-ay", True, "Statements and Declarations")
    build_springer("gpem", main, glossary, "sn-mathphys-num", False, "Declarations")
    build_tevc(main, glossary)
    print("done")
