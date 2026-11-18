"""Add ORCID iDs under the e-mails on the two internal reports."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "authors.json").read_text(encoding="utf-8"))["authors"]
line = "\\small ORCID: " + ", ".join(f"\\href{{https://orcid.org/{a['orcid']}}}{{{a['orcid']}}}" for a in A) + "}\n"
for f in ("reproduction_only.tex", "report.tex"):
    p = HERE.parent / f
    t = p.read_text(encoding="utf-8")
    old = "\\small \\texttt{u12167283@up.ac.za}, \\texttt{t.nyathi@up.ac.za}}\n"
    if "ORCID:" in t:
        print(f, "already has ORCID"); continue
    assert t.count(old) == 1, f
    t = t.replace(old, "\\small \\texttt{u12167283@up.ac.za}, \\texttt{t.nyathi@up.ac.za}\\\\\n" + line)
    p.write_text(t, encoding="utf-8")
    print(f, "ok")
