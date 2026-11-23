"""Anonymised code archive for double-blind review (TEVC, TMLR): ``python -I report/journals/make_anon_code.py``.

Packs every tracked file except the paper sources (report/), with author names and the repository URL replaced,
and fails if any identifying string is left. Writes Downloads/LLM-NAS_journal_submissions/LLM-NAS_code_anonymised.zip.
"""

import re
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path.home() / "Downloads" / "LLM-NAS_journal_submissions" / "LLM-NAS_code_anonymised.zip"
SUBS = [("https://github.com/Zimozoxolo/llm-nas-repro.git", "<repository URL>"),
        ("Siyabonga Zimozoxolo Mjali and Thambo Nyathi", "Anonymous Authors")]
IDENT = re.compile(r"Zimozoxolo|Mjali|Nyathi|Siyabonga|Thambo|u12167283|up\.ac\.za|Pretoria|Users[\\/]+siyab", re.I)

files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
files = [f for f in files if not f.startswith("report/")]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in files:
        data = (REPO / f).read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            z.writestr("llm-nas-repro/" + f, data)
            continue
        for a, b in SUBS:
            text = text.replace(a, b)
        assert not IDENT.search(text), f"identifying text left in {f}: {IDENT.search(text).group(0)}"
        z.writestr("llm-nas-repro/" + f, text)
print(f"{OUT}  {len(files)} files  {OUT.stat().st_size / 1e6:.1f} MB")
