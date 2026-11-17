# Journal versions of the LLM-NAS reproduction study

Built from `../reproduction_only.tex` by `build_versions.py` (run `python -I build_versions.py` from this folder).
All three versions share `refs.bib` and a 241-word abstract written to meet the 250-word limit of GPEM.

| Folder | Journal | Template | Compiler | Review |
|---|---|---|---|---|
| `tmlr/` | Transactions on Machine Learning Research (Reproducibility Certification) | official TMLR style (github.com/JmlrOrg/tmlr-style-file) | pdfLaTeX + BibTeX | double-blind, OpenReview |
| `gpem/` | Genetic Programming and Evolvable Machines (Springer) | Springer Nature `sn-jnl` (v3, Oct 2023 copy; official current is v3.1, Dec 2024) | pdfLaTeX + BibTeX | full peer review |
| `rescience/` | ReScience C | official ReScience template (github.com/ReScience/template) | XeLaTeX + Biber | open review on GitHub |

## What each journal requires and what was done

**TMLR** — PDF from the TMLR template; double-blind (authors hidden by `tmlr.sty` in submission mode); no page limit; appendix after references; supplementary material anonymised (≤100 MB, PDF or ZIP); Broader Impact statement only if there is risk of harm; Reproducibility Certification needs "significant added value" (our non-LLM controls, multi-LLM comparison and HV-convention analysis).
Done: template, anonymised author block, repository name replaced by "anonymised supplementary material", `\citep` citations, glossary moved after references, table of contents removed.
To do: make an anonymised code archive for the supplementary upload; all authors need OpenReview profiles.

**GPEM** — abstract 150–250 words; 4–6 keywords; numbered references in square brackets with DOIs; at most three heading levels; editable source files; Declarations (competing interests and funding mandatory, data availability for all original research, code availability); use of LLMs documented in the Methods.
Done: sn-jnl numbered style, 241-word abstract, 6 keywords, DOIs in `refs.bib`, Declarations section, AI-tool use stated in Section 3.1 and in the Declarations.
To do: fill every `[ ... ]` placeholder (affiliation, e-mail, funding, competing interests, code URL, author contributions). Run-in `\paragraph` headings in Section 5 may count as a fourth heading level; convert them to `\subsection` if the editor asks.

**ReScience C** — replication written from the published description (true here: no reference implementation existed); public code repository URL is mandatory at submission; metadata in `metadata.yaml`; title prefixed `[Re]` (successful) or `[¬Re]` (failed); a failed replication must show that the original authors were contacted; you cannot replicate your own or close collaborators' work; submission is a GitHub issue at github.com/rescience/submissions.
Done: template, `metadata.yaml` and generated `metadata.tex`, `[¬Re]` prefix (to confirm), CC-BY footer from the template.
To do: make the code repository public and put its URL in `metadata.yaml`; contact the LLM-NAS authors (the paper already lists this as future work) and record it; decide `[Re]` versus `[¬Re]`; regenerate `metadata.tex` with `python yaml-to-latex.py -i metadata.yaml -o metadata.tex`. The template prints no abstract on the PDF by design; the abstract lives in the metadata.

## Overleaf
Each `*_overleaf.zip` in Downloads uploads as a new project (New Project → Upload Project). TMLR and GPEM compile with the default pdfLaTeX. The ReScience project contains a `latexmkrc` that selects XeLaTeX; if Overleaf ignores it, set Menu → Compiler → XeLaTeX.
