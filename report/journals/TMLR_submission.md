# Transactions on Machine Learning Research: submission sheet

Checked against the TMLR author guide (jmlr.org/tmlr/author-guide.html, read 23 November 2026).

**Do not submit until IEEE TEVC confirms the withdrawal** (request emailed to s.fishkin@ieee.org on 23 November 2026,
Dr Nyathi in copy). Until then the paper is still under review there.

**Where:** OpenReview, TMLR venue: https://openreview.net/group?id=TMLR → "Submit".
**Review:** double-blind. **Licence:** all TMLR submissions are CC BY 4.0; the authors keep copyright. No fees.

## Before submitting: OpenReview profiles (both authors)

TMLR requires a complete, active OpenReview profile for **every** author: name, email, current affiliation
(University of Pretoria, Department of Computer Science), affiliation history, publication history (DBLP import or
manual), and conflicts of interest. A new profile with an institutional email (`@up.ac.za`) is activated quickly;
other emails can take up to two weeks to be moderated. Dr Nyathi must create or update his own profile; you add
him to the submission by his OpenReview ID or email.

## Files (in `Downloads\LLM-NAS_journal_submissions`)

| Form field | File | Notes |
|---|---|---|
| PDF | `LLM-NAS_TMLR.pdf` | 29 pages, TMLR template in submission mode, anonymous (checked: no names in text or PDF metadata). |
| Supplementary material | `LLM-NAS_code_anonymised.zip` | 19.8 MB (limit 100 MB, PDF or ZIP): code, configurations, reports, every run log, names removed. |

## Metadata to paste

**Title:** Does the Language Model Matter? A Reproduction Study of LLM-NAS for Hardware-Aware Neural Architecture Search

**Abstract:**
LLM-NAS uses a large language model (LLM) as the variation operator of a hardware-aware neural architecture search and reports near-optimal Pareto fronts on HW-NAS-Bench from 120 calls to the language model per search. No reference implementation was available, so we reimplemented the method from the paper's description in a framework that logs every call for exact replay. Eleven of the paper's twelve reported architectures were recovered exactly; one does not exist in the benchmark, and 4,699 architectures carry an invalid Edge TPU latency. We compared searches driven by GPT-4.1, Claude Opus 5.5, Claude Haiku 4.5, DeepSeek-Coder-V2 and Qwen2.5-7B with five non-LLM controls that share the evaluator and the niche structure, among them NSGA-II and regularised evolution, ten seeds each. On Edge GPU, GPT-4.1 reached a hypervolume of 0.964 (paper: 0.997). Across six devices, none of 48 LLM searches exceeded NSGA-II, regularised evolution or refilled random-operator evolution. One degenerate architecture on the true front compresses all sensible architectures into 9% of the accuracy axis; excluding it reorders the methods and puts Claude Opus 5.5 first on Edge GPU, from a single search, but leaves the six-device result unchanged. The LLMs repeat 20–74% of their proposals. In a pre-registered test, showing GPT-4.1 the cells already evaluated and asking again for rejected children did not reduce the repeats, and with the larger budget it scored below every control. On this small tabulated space, the method's cost advantage comes from the training-free evaluator rather than from the language model.

**Authors (OpenReview IDs, entered in the form only):** Siyabonga Zimozoxolo Mjali (corresponding), Thambo Nyathi.

## Questions the form usually asks

- **Submission length:** choose the long-submission option (main text over 12 pages); TMLR allows any length but
  long papers get more review time.
- **Suggested action editors:** pick from TMLR's action-editor list, people working on neural architecture search,
  AutoML or evolutionary computation. Avoid anyone with a conflict (co-authors, same institution, recent collaborators)
  and the LLM-NAS authors.
- **Previously submitted or under review elsewhere:** previously submitted to IEEE TEVC and withdrawn by the authors
  before review (add the date the withdrawal was confirmed). Not under review elsewhere.
- **Competing interests:** none. **Funding:** none. **Human subjects / IRB:** not applicable, no human subjects.
- **Broader impact statement:** not required (no significant risk of harm).
- **Certification:** if the form or the cover note allows, say the paper is a reproduction study and you would like
  it considered for the Reproducibility Certification.
- **Preprint:** allowed at any time, but do not link to a version with your names from the TMLR submission.

## Note to the action editor (optional comments field) — draft, to be checked by you

This manuscript is an independent reproduction of LLM-NAS, a recent method that uses a large language model as the
variation operator of a hardware-aware neural architecture search. No reference implementation was available. We
reimplemented the method, validated the benchmark data against the original paper, and compared five language
models with five non-LLM controls under the same training-free evaluator, including a pre-registered test of two
remedies for the model's repeated proposals. Code, configurations and the logs of every search, which regenerate
every number without access to the language models, are provided as anonymised supplementary material. We would be
grateful if the submission could be considered for the Reproducibility Certification. The manuscript was previously
submitted to IEEE Transactions on Evolutionary Computation and withdrawn by the authors before review.
