# Submission packages for journals with a Clarivate Journal Impact Factor

Built by `build_more.py` from `../reproduction_only.tex`; zipped by `make_zips2.py`.
Requirements were read from each journal's author guide on 18 November 2026. All folders are flat (no subfolders).
Every `[ ... ]` in a manuscript is a placeholder you must fill or confirm.

## Shared by all versions
- 241-word abstract (IEEE uses its own 189-word version without acronyms).
- Bibliography `refs.bib` with DOIs.
- AI disclosure: an AI coding assistant (Claude Code) wrote the implementation and drafted and edited text. Each version states this in the place its publisher requires. **Check the wording matches what you actually did.**
- Placeholders to fill: affiliation, e-mail, funding, competing interests, CRediT roles, and the public repository URL and DOI. A public code repository is needed for every journal's data or code statement.

## IEEE Transactions on Evolutionary Computation (`tevc/`, IF 15.9)
Requirements:
- IEEE two-column format, with no author names or affiliations.
- Abstract of 100–200 words with no acronyms, formulas or references.
- 2–5 keywords.
- Title of 15 words or fewer, with no acronyms.
- At most 10 pages before over-length charges ($200 per page), and at most 15 pages in all.
- Disclosure of AI-generated text in the acknowledgements, with a citation to the AI system.

Done:
- IEEEtran journal class, anonymised.
- 11-word title and 189-word abstract.
- 5 index terms.
- Glossary moved to `supplementary.tex`.
- AI disclosure placed in the acknowledgements, citing Claude Code.

**Issue:** the paper is 15 pages, the maximum allowed. Pages 11–15 would cost about $1,000 in over-length charges. To avoid that, cut roughly 5 pages, for example by moving Sections 6–8 or some tables into the supplement.

## Swarm and Evolutionary Computation (`swevo/`, IF 9.6) and Neurocomputing (`neucom/`, IF 6.7)
Requirements:
- Abstract of 250 words or fewer.
- 3–5 highlights of up to 85 characters each.
- 1–7 keywords.
- Numbered references.
- Single-anonymized review.
- Data deposit and link (Option C).
- Declarations: CRediT roles, competing interests, data availability, and generative AI use.

Done: elsarticle (numbered style), the highlights (also in `highlights.txt`), 6 keywords, and all four declarations.

## Applied Soft Computing (`asoc/`, IF 7.8)
Same requirements as above, plus a **required graphical abstract**: at least 1328 px wide and readable at 13 cm.
Done: `graphical_abstract.pdf` and `.png` (2046 px wide), also placed in the front matter.

## Neural Networks (`neunet/`, IF 7.2)
Same requirements as Swarm and Evolutionary Computation, except **APA author-year references** and 1–6 keywords.
Done: elsarticle author-year style (Elsevier reformats the references to APA at production) and 6 keywords.

## Machine Learning (`mlj/`, IF 4.9) and GPEM (`gpem/`, IF 1.9)
Requirements:
- Abstract of 150–250 words.
- 4–6 keywords.
- A "Statements and Declarations" section (Machine Learning) or "Declarations" (GPEM).
- Machine Learning uses author-year citations; GPEM uses numbered citations.
- LaTeX sources without subfolders.

Done:
- sn-jnl (mathphys-ay for Machine Learning, mathphys-num for GPEM).
- The declarations section.
- Figures moved to the top level.
- Wide tables reflowed so they fit the page: smaller font, and ± values on a second row.
