# IEEE Transactions on Evolutionary Computation: submission sheet

Checked against the TEVC Information for Authors (cis.ieee.org, read 23 November 2026).

**Portal:** https://ieee.atyponrex.com/journal/TEVC-IEEE (IEEE Author Portal; sign in with an IEEE account).
**Article type:** Regular paper. **Review:** double-anonymous; the files below contain no author names.

## Files to upload (all in `Downloads\LLM-NAS_journal_submissions`)

| Portal file type | File | Notes |
|---|---|---|
| Main manuscript | `LLM-NAS_IEEE_TEVC.pdf` | 10 pages, IEEE two-column, anonymous. 10 pages is the limit before over-length charges. |
| Supplementary material | `LLM-NAS_IEEE_TEVC_supplementary.pdf` | 17 pages: extended background, settings, extra results and tables, future work, glossary. |
| Supplementary material | `LLM-NAS_code_anonymised.zip` | 19.8 MB: code, configurations, reports and every run log, names removed. |
| Supplementary material (graphical abstract) | `LLM-NAS_graphical_abstract.png` | Optional; TEVC encourages one. |

Do **not** upload the `_v2` files in that folder; they predate the 10-page version. LaTeX source
(`LLM-NAS_IEEE_TEVC_overleaf.zip`) is only needed after acceptance.

## Metadata to paste

**Title** (11 words, no acronyms):
Does the Language Model Matter? Reproducing Language-Model-Driven Hardware-Aware Neural Architecture Search

**Abstract** (197 words, no acronyms, as required):
A recent method for hardware-aware neural architecture search replaces the variation operator of an evolutionary search with a large language model and reports near-optimal accuracy-latency trade-offs from 120 model queries per search. No reference implementation was available, so we reimplemented the method from its published description and logged every query for exact replay. Eleven of the twelve architectures reported by the original authors were recovered; one does not exist in the benchmark. We compared searches driven by five language models with five controls that use no language model but share the training-free evaluator and the complexity niches, among them two standard evolutionary algorithms, each run with ten seeds. Front quality depended strongly on an unstated normalisation: a single degenerate architecture on the true front compresses all useful architectures into a small part of the accuracy axis. Across six hardware devices, none of 48 language-model searches exceeded the two evolutionary algorithms or random-operator evolution with refilling. In a pre-registered test, showing the model the architectures it had already evaluated and asking again for rejected proposals did not reduce its repeated proposals. On this benchmark, the reported cost savings come from the evaluator rather than from the language model.

**Keywords** (5): Neural architecture search; evolutionary computation; large language models; multiobjective optimization; reproducibility

## Authors (entered in the portal only, never in the files)

| Order | Name | Affiliation | Email | ORCID | Role |
|---|---|---|---|---|---|
| 1 | Siyabonga Zimozoxolo Mjali | Department of Computer Science, University of Pretoria, South Africa | u12167283@up.ac.za | 0009-0008-9207-5371 | Corresponding author |
| 2 | Thambo Nyathi | Department of Computer Science, University of Pretoria, South Africa | t.nyathi@up.ac.za | 0000-0003-4676-6063 | Co-author |

Both authors need a registered ORCID (they have one). Co-author approval: given by Dr Nyathi, 23 November 2026.

## Questions the portal usually asks

- **Previously published or under review elsewhere?** No. (Do not post a preprint until you decide; IEEE allows arXiv, with an IEEE copyright notice once submitted.)
- **Funding:** none. **Competing interests:** none.
- **Use of AI:** disclosed in the Acknowledgment section, with a citation to the tool, as TEVC requires.
- **Open access:** choose the traditional (no fee) option unless you want open access (US$2,800 in 2026).
- **Suggested or opposed reviewers:** optional. Do not suggest the LLM-NAS authors (conflict of interest); you may list them as opposed reviewers if you wish.
- **Page charges:** a voluntary $110 per page may be requested after acceptance; it is not required for publication. There is no mandatory over-length charge at 10 pages.

## Comments to the Editor-in-Chief (cover letter) — draft, to be checked by you

Dear Editor-in-Chief,

Please consider the enclosed manuscript, "Does the Language Model Matter? Reproducing Language-Model-Driven Hardware-Aware Neural Architecture Search", for publication as a regular paper in the IEEE Transactions on Evolutionary Computation.

The manuscript is an independent reproduction of a recent method that uses a large language model as the variation operator of an evolutionary, hardware-aware neural architecture search. No reference implementation was available, so we reimplemented the method from its published description and compared it, under the same training-free evaluator and complexity niches, with five non-language-model controls, including NSGA-II and regularised evolution. Across six hardware devices, none of 48 language-model searches exceeded the evolutionary controls. A pre-registered test showed that the obvious remedies for the model's repeated proposals do not change this. We believe the results are of direct interest to the Transactions' readers, because they bear on whether language-model operators improve on established evolutionary operators.

The work has not been published elsewhere and is not under review by another publication. All authors have approved the submission. The code, configurations and the logs of every search, which allow every number to be regenerated without access to the language models, are provided as anonymised supplementary material. The use of an AI coding assistant is disclosed in the Acknowledgment section. The authors declare no competing interests and received no specific funding for this work.

Yours sincerely,
The authors
