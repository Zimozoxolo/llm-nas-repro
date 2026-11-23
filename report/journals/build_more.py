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
    "Eleven of twelve reported architectures recovered; one does not exist",
    "Hypervolume rankings hinge on one degenerate point and on normalisation",
    "No LLM search beat NSGA-II or regularised evolution on six devices (0 of 48)",
    "Pre-registered fixes for repeated LLM proposals did not close the gap",
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
    "configurations and archived run logs behind every number are available at \\url{https://github.com/Zimozoxolo/llm-nas-repro}."
)

TEVC_TITLE = "Does the Language Model Matter? Reproducing Language-Model-Driven Hardware-Aware Neural Architecture Search"
TEVC_ABSTRACT = (
    "A recent method for hardware-aware neural architecture search replaces the variation operator of an "
    "evolutionary search with a large language model and reports near-optimal accuracy-latency trade-offs from "
    "120 model queries per search. No reference implementation was available, so we reimplemented the method from "
    "its published description and logged every query for exact replay. Eleven of the twelve architectures "
    "reported by the original authors were recovered; one does not exist in the benchmark. We compared searches "
    "driven by five language models with five controls that use no language model but share the training-free "
    "evaluator and the complexity niches, among them two standard evolutionary algorithms, each run with ten seeds. "
    "Front quality depended strongly on an unstated normalisation: a single degenerate architecture on the true "
    "front compresses all useful architectures into a small part of the accuracy axis. Across six hardware "
    "devices, none of 48 language-model searches exceeded the two evolutionary algorithms or random-operator "
    "evolution with refilling. In a pre-registered test, showing the model the architectures it had already "
    "evaluated and asking again for rejected proposals did not reduce its repeated proposals. On this benchmark, "
    "the reported cost savings come from the evaluator rather than from the language model."
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
    # the source scales some wide tables with \resizebox; unwrap them so each template applies its own rule
    s = re.sub(r"\\resizebox\{\\linewidth\}\{!\}\{(\\begin\{tabular\}.*?\\end\{tabular\})\}", r"\1", s, flags=re.S)
    if springer:
        # sn-jnl's table environment breaks when a tabular is boxed, so shrink fonts, padding and
        # fixed column widths instead.
        s = re.sub(r"(\\begin\{table\}(?:\[[^\]]*\])?\\centering)\\small",
                   r"\1\\footnotesize\\setlength{\\tabcolsep}{3pt}", s)
        s = re.sub(r"p\{(\d+(?:\.\d+)?)cm\}", lambda m: "p{%.1fcm}" % (float(m.group(1)) * 0.78), s)
        s = s.replace("\\setlength{\\tabcolsep}{4pt}", "\\setlength{\\tabcolsep}{2pt}")
        s = s.replace("Method & Edge~GPU & Raspberry~Pi~4 & Edge~TPU & Pixel~3 & Eyeriss & FPGA",
                      "Method & EGPU & Pi4 & ETPU & Pix3 & Eyeriss & FPGA")
        s = s.replace("\\setlength{\\tabcolsep}{2.5pt}", "\\setlength{\\tabcolsep}{2pt}")
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
\begin{scope}[shift={(11.3,-1.1)}]
  \def\s{18}
  \foreach \lab/\v/\c [count=\i] in {GPT-4.1 (1 seed)/0.928/blue!55, Qwen (3)/0.894/blue!55, DeepSeek (3)/0.890/blue!55,
      Haiku (1)/0.888/blue!55, {Random sampling (10)}/0.948/orange!80, {Random-operator evolution (10)}/0.950/orange!80,
      {Same, with refilling (10)}/0.959/orange!80, {Regularised evolution (10)}/0.969/orange!80, {NSGA-II (10)}/0.970/orange!80} {
    \pgfmathsetmacro{\w}{(\v-0.85)*\s}
    \pgfmathsetmacro{\y}{2.3-0.235*\i}
    \fill[\c] (0,\y-0.095) rectangle (\w,\y+0.095);
    \node[anchor=east,font=\sffamily\scriptsize] at (-0.05,\y) {\lab};
    \node[anchor=west,font=\sffamily\scriptsize] at (\w+0.03,\y) {\v};
  }
  \draw (0,-0.05)--(0,2.15);
  \node[font=\sffamily\scriptsize,anchor=north] at (1.0,-0.05) {hypervolume, mean over six devices (CIFAR-10); axis starts at 0.85};
  \node[font=\sffamily\small\bfseries,anchor=south,align=center] at (1.0,2.2) {0 of 48 LLM searches beat the evolutionary controls};
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
        "\\item \\textbf{Code availability.} \\url{https://github.com/Zimozoxolo/llm-nas-repro} (MIT licence).\n"
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
# Moved to the supplementary file so the paper fits TEVC's length limit (about 14 pages).
TEVC_TO_SUPP = [
    "How it was done before language models",
    "Search strategies: advantages and pitfalls",
    "How performance is estimated today",
    "Unspecified details and the choices made",
    "Scope of what was run",
    "Ablation on CIFAR-100 (paper's Table 5)",
    "What the paper reports",
    # added to reach TEVC's 10-page limit for regular papers (over-length pages are charged)
    "Framework",
    "Data",
    "LLM co-evolution of knowledge and architectures",
    "Training-free evaluation",
    "Cost",
    "Why hardware-aware search",
]
TEVC_PARAS_TO_SUPP = ["What other studies report.", "Relation to other work.", "Where an LLM operator could still be justified.",
                      "What the paper offers as justification.", "Evidence 2: cost.", "What would settle the question."]
TEVC_SECTIONS_TO_SUPP = ["Future work", "Reproducibility and artefacts"]
TEVC_FLOATS_TO_SUPP = ["tab:multidevicefloor", "fig:floor", "fig:pipeline", "fig:cnnfronts", "tab:dupfixq1"]


def tevc_trim(body):
    """Cut the listed subsections out of the body; references to their labels point to the supplement."""
    moved = []
    for title in TEVC_TO_SUPP:
        m = re.search(r"\\subsection\{" + re.escape(title) + r"\}.*?(?=\\subsection\{|\\section\{|%=====)", body, flags=re.S)
        assert m, f"TEVC: subsection not found: {title}"
        moved.append(m.group(0))
        body = body[: m.start()] + body[m.end():]
    for lab in TEVC_FLOATS_TO_SUPP:
        m = re.search(r"\\begin\{(table|figure)\}(?:(?!\\end\{(?:table|figure)\}).)*?\\label\{" + re.escape(lab)
                      + r"\}.*?\\end\{\1\}\n?", body, flags=re.S)
        assert m, f"TEVC: float not found: {lab}"
        moved.append(m.group(0))
        body = body[: m.start()] + body[m.end():]
    for title in TEVC_SECTIONS_TO_SUPP:
        m = re.search(r"\\section\{" + re.escape(title) + r"\}.*?(?=\\section\{|%=====|\Z)", body, flags=re.S)
        assert m, f"TEVC: section not found: {title}"
        moved.append(m.group(0).replace("\\section{", "\\subsection{", 1))
        body = body[: m.start()] + body[m.end():]
    for title in TEVC_PARAS_TO_SUPP:
        m = re.search(r"\\paragraph\{" + re.escape(title) + r"\}.*?(?=\\paragraph\{|\\subsection\{|\\section\{|%=====)",
                      body, flags=re.S)
        assert m, f"TEVC: paragraph not found: {title}"
        moved.append(m.group(0))
        body = body[: m.start()] + body[m.end():]
    labels = {l for t in moved for l in re.findall(r"\\label\{([^}]+)\}", t)}
    for l in labels:
        body = re.sub(r"(Tables?|Figures?|Sections?|Sec\.|Fig\.)~\\ref\{" + re.escape(l) + r"\}", "the supplementary material", body)
        body = body.replace("\\ref{" + l + "}", "S")
    # references from the moved text back into the paper
    for i, t in enumerate(moved):
        for l in set(re.findall(r"\\ref\{([^}]+)\}", t)) - labels:
            t = re.sub(r"(Tables?|Figures?|Sections?|Sec\.|Fig\.|Appendix)~\\ref\{" + re.escape(l) + r"\}", "the main paper", t)
            t = t.replace("\\ref{" + l + "}", "the main paper")
        moved[i] = t
    return body, moved


def build_tevc(main, glossary):
    d, body = prep("tevc", main, "All code, configurations and run logs are provided as anonymised supplementary material.")
    roadmap = ("Section~\\ref{sec:discussion} what was discovered, and Section~\\ref{sec:future} next steps. "
               "Appendix~\\ref{sec:glossary} defines all technical terms and formulas.")
    assert body.count(roadmap) == 1
    body = body.replace(roadmap, "and Section~\\ref{sec:discussion} what was discovered. The supplementary material "
                                 "gives further background, settings, results, next steps and a glossary of all terms.")
    body, moved = tevc_trim(body)
    body = re.sub(r"(\\paragraph\{[^}]*\} )the supplementary material", r"\1The supplementary material", body)
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
        "\\usepackage{natbib}\n\\begin{document}\n\\title{Supplementary material}\n\\date{}\n\\maketitle\n"
        "\\section{Extended background and settings}\n"
        + "\n".join(moved) +
        "\n\\section{Glossary of terms and formulas}\n"
        + glossary.replace("\\section{Glossary of terms and formulas}\\label{sec:glossary}", "") +
        "\n\\bibliographystyle{plainnat}\n\\bibliography{refs}\n\\end{document}\n"
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
