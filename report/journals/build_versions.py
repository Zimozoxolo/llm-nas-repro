"""Build journal-formatted versions of the LLM-NAS reproduction study.

Source: ../reproduction_only.tex (body from the Introduction to the end of the glossary).
Targets: tmlr/, gpem/ (Springer Nature sn-jnl), rescience/ (ReScience C template).
Run from report/journals:  python -I build_versions.py
"""
import re, shutil, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import authors as au  # noqa: E402
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "reproduction_only.tex"
FIG = HERE.parent / "figures"

TITLE = "Does the Language Model Matter? A Reproduction Study of LLM-NAS for Hardware-Aware Neural Architecture Search"
TITLE_SHORT = "A Reproduction Study of LLM-NAS"

ABSTRACT = (
    "LLM-NAS uses a large language model (LLM) as the variation operator of a hardware-aware neural "
    "architecture search and reports near-optimal Pareto fronts on HW-NAS-Bench from 120 calls to the "
    "language model per search. No reference implementation was available, so we reimplemented the method "
    "from the paper's text, algorithm and prompts in a configuration-driven framework that logs every call "
    "for exact replay. We first validated the benchmark data: eleven of the paper's twelve reported "
    "architectures were recovered exactly, one reported point does not exist in the benchmark, and 4{,}699 "
    "architectures carry an invalid Edge TPU latency. We then compared searches driven by GPT-4.1, Claude "
    "Opus~5.5, Claude Haiku~4.5, DeepSeek-Coder-V2 and Qwen2.5-7B with non-LLM controls that share the "
    "evaluator and the niche structure, using hypervolume (HV) and inverted generational distance. Under the "
    "default convention GPT-4.1 reached HV~0.964 (paper: 0.997) and the controls 0.989--0.991. The ranking "
    "depends on the normalisation: one degenerate architecture on the true front compresses all sensible "
    "architectures into 9\\% of the accuracy axis, and excluding architectures below 80\\% accuracy places "
    "Claude Opus~5.5 first (0.961) and GPT-4.1 below the controls (0.752). Across six devices, none of 24 LLM "
    "runs exceeded random-operator evolution and one exceeded random sampling. On this small tabulated space "
    "with a cheap evaluator, the method's cost advantage comes from the training-free evaluator rather than "
    "from the language model; whether the LLM operator pays off with expensive evaluations or in larger "
    "spaces remains open."
)
KEYWORDS = ["Neural architecture search", "Hardware-aware NAS", "Large language models",
            "Reproducibility", "Evolutionary computation", "Multi-objective optimisation"]

AI_SENTENCE_OLD = ("The implementation is a Python package with the following properties, all motivated by "
                   "the need to regenerate thesis numbers years later:")
AI_SENTENCE_NEW = ("The implementation is a Python package, written with the assistance of an AI coding tool "
                   "(Claude Code) and reviewed by us; every number in this paper is produced by it and can be "
                   "regenerated from the archived run logs. It has the following properties, all motivated by "
                   "the need to regenerate every number years later:")
REPO_OLD = "All code, configurations and reports are in the \\code{llm-nas-repro} repository."
REPO_NEW = {
    "tmlr": "All code, configurations and run logs are provided as anonymised supplementary material.",
    "gpem": "All code, configurations and run logs are in the public repository given in the Code availability statement.",
    "rescience": "All code, configurations and run logs are in the public repository given in the article metadata.",
}

PREAMBLE_PKGS = r"""\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit}
\usepackage{algorithm}
\usepackage{algpseudocode}
\graphicspath{{figures/}}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\HV}{\mathrm{HV}}
\newcommand{\IGD}{\mathrm{IGD}}
"""


def word_count(s):
    return len(re.sub(r"\\[a-zA-Z]+|[{}~$]", " ", s).split())


def load_body():
    t = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    a = t.index("\\section{Introduction}")
    b = t.index("\\begin{thebibliography}")
    body = t[a:b].rstrip()
    body = body.replace("\\end{document}", "")
    main, glossary = body.split("\\appendix", 1)
    for old in ("this report", "This report"):
        main = main.replace(old, old.replace("report", "paper"))
    assert main.count(AI_SENTENCE_OLD) == 1
    main = main.replace(AI_SENTENCE_OLD, AI_SENTENCE_NEW)
    assert main.count(REPO_OLD) == 1
    # strip separator comment lines
    main = re.sub(r"\n%=+\n", "\n", main)
    glossary = re.sub(r"\n%=+\n", "\n", glossary).strip()
    return main.strip(), glossary


def copy_figs(dst):
    (dst / "figures").mkdir(parents=True, exist_ok=True)
    for f in ("cnn_fronts_edgegpu.png", "hv_floor.png", "fronts_cifar100.png"):
        shutil.copy2(FIG / f, dst / "figures" / f)


DECL_NOTE = "% [TO CONFIRM] statements in square brackets must be checked or completed by the author before submission.\n"


def build_tmlr(main, glossary):
    d = HERE / "tmlr"
    d.mkdir(exist_ok=True)
    for f in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty"):
        shutil.copy2(HERE / "tmlr-style" / f, d / f)
    shutil.copy2(HERE / "refs.bib", d / "refs.bib")
    copy_figs(d)
    body = main.replace(REPO_OLD, REPO_NEW["tmlr"])
    body = body.replace("\\cite{", "\\citep{")
    tex = (
        "\\documentclass[10pt]{article}\n"
        "\\usepackage{tmlr}\n"
        "% Camera-ready: \\usepackage[accepted]{tmlr}; preprint: \\usepackage[preprint]{tmlr}\n"
        "\\usepackage{hyperref}\n"
        + PREAMBLE_PKGS +
        f"\\title{{{TITLE}}}\n"
        "% Authors are hidden by tmlr.sty in submission mode (double-blind review).\n"
        + au.tmlr() +
        "\\def\\month{MM}\n\\def\\year{YYYY}\n\\def\\openreview{\\url{https://openreview.net/forum?id=XXXX}}\n"
        "\\begin{document}\n\\maketitle\n"
        f"\\begin{{abstract}}\n{ABSTRACT}\n\\end{{abstract}}\n\n"
        + body +
        "\n\n\\bibliography{refs}\n\\bibliographystyle{tmlr}\n\n\\appendix\n" + glossary +
        "\n\n\\end{document}\n"
    )
    (d / "main.tex").write_text(tex, encoding="utf-8")


def build_gpem(main, glossary):
    d = HERE / "gpem"
    d.mkdir(exist_ok=True)
    for f in ("sn-jnl.cls", "sn-mathphys-num.bst"):
        shutil.copy2(HERE / "sn-template" / f, d / f)
    shutil.copy2(HERE / "refs.bib", d / "refs.bib")
    copy_figs(d)
    body = main.replace(REPO_OLD, REPO_NEW["gpem"])
    pre = PREAMBLE_PKGS.replace("\\usepackage{amsmath,amssymb}\n", "")
    tex = (
        "%Springer Nature LaTeX template (sn-jnl), Math and Physical Sciences numbered reference style.\n"
        "\\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}\n"
        + DECL_NOTE +
        "\\usepackage{amsmath,amssymb,amsfonts}\n"
        + pre +
        "\\raggedbottom\n"
        "\\begin{document}\n\n"
        f"\\title[{TITLE_SHORT}]{{{TITLE}}}\n\n"
        "\\author*[1]{\\fnm{Siyabonga} \\sur{Mjali}}\\email{[email of corresponding author]}\n"
        "\\affil*[1]{\\orgdiv{[Department]}, \\orgname{[Institution]}, \\orgaddress{\\city{[City]}, \\country{[Country]}}}\n\n"
        f"\\abstract{{{ABSTRACT}}}\n\n"
        f"\\keywords{{{', '.join(KEYWORDS)}}}\n\n"
        "\\maketitle\n\n"
        + body +
        "\n\n\\backmatter\n\n"
        "\\section*{Declarations}\n"
        "\\begin{itemize}\n"
        "\\item \\textbf{Funding.} [State the funding sources, or: The author received no specific funding for this work.]\n"
        "\\item \\textbf{Competing interests.} [The author declares no competing interests.]\n"
        "\\item \\textbf{Ethics approval and consent to participate.} Not applicable; no human or animal subjects were involved.\n"
        "\\item \\textbf{Data availability.} All benchmark data are public: HW-NAS-Bench, NAS-Bench-Suite-Zero and NATS-Bench. "
        "The archived run logs behind every number are available in the code repository.\n"
        "\\item \\textbf{Code availability.} [Public repository URL and archive DOI, e.g.\\ Zenodo.]\n"
        "\\item \\textbf{Use of large language models.} Large language models are the object of study and are documented in "
        "Sections~\\ref{sec:background}--\\ref{sec:results}; an AI coding tool assisted the implementation, as stated in "
        "Section~\\ref{sec:method}. The author reviewed all code and text and is accountable for them.\n"
        "\\item \\textbf{Author contributions.} [S.M.: conceptualisation, software, investigation, analysis, writing.]\n"
        "\\end{itemize}\n\n"
        "\\begin{appendices}\n" + glossary + "\n\\end{appendices}\n\n"
        "\\bibliography{refs}\n\n\\end{document}\n"
    )
    (d / "main.tex").write_text(tex, encoding="utf-8")


DECL_NOTE_PLACEHOLDER = ""


def build_rescience(main, glossary):
    d = HERE / "rescience"
    src = HERE / "rescience-template"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns(".git", "content.tex", "metadata.yaml",
                                                          "bibliography.bib", "metadata.tex", "*.pdf"))
    shutil.copy2(HERE / "refs.bib", d / "bibliography.bib")
    copy_figs(d)
    body = main.replace(REPO_OLD, REPO_NEW["rescience"])
    content = (PREAMBLE_PKGS.split("\\graphicspath")[0].replace("\\usepackage", "%\\usepackage")  # packages go in article.tex
               + "\\graphicspath{{figures/}}\n\\newcommand{\\code}[1]{\\texttt{#1}}\n"
               "\\newcommand{\\HV}{\\mathrm{HV}}\n\\newcommand{\\IGD}{\\mathrm{IGD}}\n\n"
               + body + "\n\n\\appendix\n" + glossary + "\n")
    (d / "content.tex").write_text(content, encoding="utf-8")
    art = (d / "article.tex").read_text(encoding="utf-8")
    art = art.replace("\\documentclass{rescience}\n",
                      "\\documentclass{rescience}\n\\usepackage{booktabs}\n\\usepackage{graphicx}\n"
                      "\\usetikzlibrary{arrows.meta,positioning,fit}\n\\usepackage{algorithm}\n\\usepackage{algpseudocode}\n")
    (d / "article.tex").write_text(art, encoding="utf-8")
    abstract_plain = ABSTRACT.replace("~", " ").replace("{,}", ",").replace("--", "–")
    kw = ", ".join(k.lower() for k in KEYWORDS) + ", python"
    yaml = f"""# To be filled by the author(s) at the time of submission
# -------------------------------------------------------
# [TO CONFIRM] "[¬Re]" marks a replication that did not reproduce the original claims;
# use "[Re]" instead if the editor and you judge the replication successful.
title: "[¬Re] LLM-NAS: LLM-Driven Hardware-Aware Neural Architecture Search"

{au.rescience_yaml()}
keywords: {kw}

code:
  - url: "[public repository URL]"
  - doi:
  - swh:

data:
  - url: https://github.com/GATECH-EIC/HW-NAS-Bench
  - doi:

replication:
 - cite: "Zhu, H., Zhang, G. L. and Huang, S. LLM-NAS: LLM-driven Hardware-Aware Neural Architecture Search. arXiv:2510.01472, 2025."
 - bib: llmnas
 - url: https://arxiv.org/abs/2510.01472
 - doi: 10.48550/arXiv.2510.01472

abstract: >-
  {abstract_plain}

bibliography: bibliography.bib

type: Replication

domain: Machine Learning

language: Python

# To be filled by the author(s) after acceptance
# -----------------------------------------------------------------------------
review:
  - url:

contributors:
  - name:
    orcid:
    role: editor
  - name:
    orcid:
    role: reviewer
  - name:
    orcid:
    role: reviewer

dates:
  - received:
  - accepted:
  - published:

article:
  - number:
  - doi:
  - url:

journal:
  - name:   "ReScience C"
  - issn:   2430-3658
  - volume:
  - issue:
"""
    (d / "metadata.yaml").write_text(yaml, encoding="utf-8")


if __name__ == "__main__":
    main, glossary = load_body()
    print("abstract words:", word_count(ABSTRACT))
    build_tmlr(main, glossary)
    # GPEM is now built (flat, no subfolders) by build_more.py
    build_rescience(main, glossary)
    print("built tmlr/, gpem/, rescience/")
