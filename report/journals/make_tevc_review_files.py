"""Files for the TEVC portal: ``python -I report/journals/make_tevc_review_files.py`` (after build_more.py and LaTeX).

* LLM-NAS_TEVC_manuscript_latex.zip: main.tex (camera-ready author lines removed), main.bbl, refs.bib, main.pdf;
  checked for identifying text, because reviewers can open the source.
* LLM-NAS_TEVC_title_page.docx and LLM-NAS_TEVC_cover_letter.docx (via pandoc), from authors.json.
"""

import json
import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEVC = HERE / "tevc"
OUT = Path.home() / "Downloads" / "LLM-NAS_journal_submissions"
IDENT = re.compile(r"Zimozoxolo|Mjali|Nyathi|Siyabonga|Thambo|u12167283|up\.ac\.za|Pretoria|siyab", re.I)

tex = (TEVC / "main.tex").read_text(encoding="utf-8")
tex = "\n".join(l for l in tex.splitlines()
                if not (l.lstrip().startswith("%") and (IDENT.search(l) or "code URL" in l or "\\author{" in l))) + "\n"
zpath = OUT / "LLM-NAS_TEVC_manuscript_latex.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for name, data in [("main.tex", tex.encode("utf-8")), ("main.bbl", (TEVC / "main.bbl").read_bytes()),
                       ("refs.bib", (TEVC / "refs.bib").read_bytes()), ("main.pdf", (TEVC / "main.pdf").read_bytes())]:
        if name != "main.pdf":
            hit = IDENT.search(data.decode("utf-8"))
            assert not hit, f"identifying text in {name}: {hit.group(0)}"
        z.writestr(name, data)
print(zpath)

A = json.loads((HERE / "authors.json").read_text(encoding="utf-8"))
aff = A["affiliations"]["up_cs"]
title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)


def docx(md: str, name: str):
    md_path = OUT / (name + ".md")
    md_path.write_text(md, encoding="utf-8")
    subprocess.run(["pandoc", str(md_path), "-o", str(OUT / (name + ".docx"))], check=True)
    md_path.unlink()
    print(OUT / (name + ".docx"))


authors = "\n".join(
    f"- **{a['title']} {a['given']} {a['family']}**{' (corresponding author)' if a['corresponding'] else ''}. "
    f"{aff['department']}, {aff['institution']}, {aff['city']}, {aff['country']}. "
    f"Email: {a['email']}. ORCID: {a['orcid']}." for a in A["authors"])
corr = next(a for a in A["authors"] if a["corresponding"])
docx(f"""# {title}

## Authors

{authors}

## Address for correspondence

{corr['title']} {corr['given']} {corr['family']}, {aff['department']}, {aff['institution']}, {aff['postal_address']}.
Email: {corr['email']}. Telephone: {aff['phone']}.

## Acknowledgments

Text throughout this paper was drafted with the assistance of an AI system, Claude Code (Anthropic), which was also
used to write the implementation; the authors reviewed and revised all text and code and are accountable for them.
This research received no specific funding.

## Declarations

Competing interests: none. Funding: none. Code, configurations and run logs: provided to reviewers as anonymised
supplementary material; public after review.
""", "LLM-NAS_TEVC_title_page")

docx(f"""Dear Editor-in-Chief,

Please consider the enclosed manuscript, "{title}", for publication as a regular paper in the IEEE Transactions on
Evolutionary Computation.

The manuscript is an independent reproduction of a recent method that uses a large language model as the variation
operator of an evolutionary, hardware-aware neural architecture search. No reference implementation was available, so
we reimplemented the method from its published description and compared it, under the same training-free evaluator
and complexity niches, with five non-language-model controls, including NSGA-II and regularised evolution. Across six
hardware devices, none of 48 language-model searches exceeded the evolutionary controls, and a pre-registered test
showed that the obvious remedies for the model's repeated proposals do not change this. We believe the results are of
direct interest to the Transactions' readers, because they bear on whether language-model operators improve on
established evolutionary operators.

The work has not been published elsewhere and is not under review by another publication. All authors have approved
the submission. The code, configurations and the logs of every search, which allow every number to be regenerated
without access to the language models, are provided as anonymised supplementary material. For transparency: the
authors' public code repository also contains the source text of this manuscript, so the similarity check may match
it; this is the authors' own repository. The use of an AI coding assistant is disclosed in the Acknowledgment section.
The authors declare no competing interests and received no specific funding for this work.

Yours sincerely,

{' and '.join(a['given'].split()[0] + ' ' + a['family'] for a in A['authors'])}
""", "LLM-NAS_TEVC_cover_letter")
