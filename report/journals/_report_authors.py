"""Put both authors and the affiliation on the two internal reports."""
import re
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
for f in ("reproduction_only.tex", "report.tex"):
    p = BASE / f
    t = p.read_text(encoding="utf-8")
    m = re.search(r"\\author\{Siyabonga Mjali(\\thanks\{.*?\})\}\n", t, re.S)
    assert m, f
    new = ("\\author{Siyabonga Zimozoxolo Mjali" + m.group(1) + " \\and Thambo Nyathi\\\\[4pt]\n"
           "\\small Department of Computer Science, University of Pretoria, South Africa\\\\\n"
           "\\small \\texttt{u12167283@up.ac.za}, \\texttt{t.nyathi@up.ac.za}}\n")
    t = t[:m.start()] + new + t[m.end():]
    p.write_text(t, encoding="utf-8")
    print(f, "ok")
