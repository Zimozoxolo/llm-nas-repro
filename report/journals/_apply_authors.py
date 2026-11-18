"""One-off edit: make build_versions.py and build_more.py take author data from authors.py."""
from pathlib import Path
HERE = Path(__file__).resolve().parent


def edit(fname, pairs):
    p = HERE / fname
    t = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert t.count(a) == 1, (fname, a[:70], t.count(a))
        t = t.replace(a, b)
    p.write_text(t, encoding="utf-8")
    print(fname, "ok")


edit("build_versions.py", [
    ("import re, shutil, sys\n", "import re, shutil, sys\nsys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))\nimport authors as au  # noqa: E402\n"),
    ('        "\\\\author{\\\\name Siyabonga Mjali \\\\email [email] \\\\\\\\ \\\\addr [Department, Institution, Country]}\\n"\n',
     '        + au.tmlr() +\n'),
    ("    build_gpem(main, glossary)\n", "    # GPEM is now built (flat, no subfolders) by build_more.py\n"),
])

edit("build_more.py", [
    ("import build_versions as bv  # noqa: E402\n", "import build_versions as bv  # noqa: E402\nimport authors as au  # noqa: E402\n"),
    ("    \"During the preparation of this work the author used Claude Code (Anthropic), an AI coding assistant, \"",
     "    \"During the preparation of this work the first author used Claude Code (Anthropic), an AI coding assistant, \""),
    ("    \"After using this tool, the author reviewed and edited the content as needed and takes full responsibility \"",
     "    \"After using this tool, the authors reviewed and edited the content as needed and take full responsibility \""),
    ('        "\\\\author[inst1]{Siyabonga Mjali\\\\corref{cor1}}\\n"\n'
     '        "\\\\ead{[email of corresponding author]}\\n"\n'
     '        "\\\\cortext[cor1]{Corresponding author}\\n"\n'
     '        "\\\\affiliation[inst1]{organization={[Department, Institution]}, city={[City]}, country={[Country]}}\\n\\n"\n',
     '        + au.elsarticle() + "\\n"\n'),
    ('        "\\\\textbf{Siyabonga Mjali:} [Conceptualization, Methodology, Software, Investigation, Formal analysis, "\n'
     '        "Data curation, Writing -- original draft, Writing -- review \\\\& editing.]\\n\\n"\n',
     '        + au.credit_elsevier() + "\\n\\n"\n'),
    ("        \"[The author declares that they have no known competing financial interests or personal relationships \"",
     "        \"[The authors declare that they have no known competing financial interests or personal relationships \""),
    ('        "\\\\author*[1]{\\\\fnm{Siyabonga} \\\\sur{Mjali}}\\\\email{[email of corresponding author]}\\n"\n'
     '        "\\\\affil*[1]{\\\\orgdiv{[Department]}, \\\\orgname{[Institution]}, \\\\orgaddress{\\\\city{[City]}, \\\\country{[Country]}}}\\n\\n"\n',
     '        + au.sn_jnl() + "\\n"\n'),
    ("[State the funding sources, or: The author received no specific funding for this work.]",
     "[State the funding sources, or: The authors received no specific funding for this work.]"),
    ("[The author has no relevant financial or non-financial interests to disclose.]",
     "[The authors have no relevant financial or non-financial interests to disclose.]"),
    ("the author reviewed all code \"\n        \"and text and is accountable for them. [TO CONFIRM.]",
     "the authors reviewed all code \"\n        \"and text and are accountable for them. [TO CONFIRM.]"),
    ('        "\\\\item \\\\textbf{Author contributions.} [S.M.: conceptualisation, software, investigation, analysis, writing.]\\n"\n',
     '        "\\\\item \\\\textbf{Author contributions.} " + au.credit_springer() + "\\n"\n'),
    ('        "\\\\author{Anonymous Authors}\\n"\n',
     '        "\\\\author{Anonymous Authors}\\n" + au.ieee_camera_ready() +\n'),
])
