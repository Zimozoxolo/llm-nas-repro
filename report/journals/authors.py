"""Render the author block from authors.json for each journal template."""
import json
from pathlib import Path

DATA = json.loads((Path(__file__).resolve().parent / "authors.json").read_text(encoding="utf-8"))
AUTHORS = DATA["authors"]
AFF = DATA["affiliations"]


def full(a):
    return f"{a['given']} {a['family']}"


def initials(a):
    return "".join(p[0] + "." for p in a["given"].split()) + a["family"][0] + "."


def corresponding():
    return next(a for a in AUTHORS if a["corresponding"])


def tmlr():
    parts = []
    for a in AUTHORS:
        f = AFF[a["affiliation"]]
        parts.append(f"\\name {full(a)} \\email {a['email']} \\\\\n      \\addr {f['department']}\\\\\n      "
                     f"{f['institution']}, {f['city']}, {f['country']}")
    return "\\author{" + "\n      \\AND\n      ".join(parts) + "}\n"


def sn_jnl():
    codes = {k: i + 1 for i, k in enumerate(AFF)}
    out = []
    for a in AUTHORS:
        star = "*" if a["corresponding"] else ""
        orcid = ""  # sn-jnl has no ORCID field; Springer collects ORCID iDs in the submission system
        out.append(f"\\author{star}{orcid}[{codes[a['affiliation']]}]{{\\fnm{{{a['given']}}} \\sur{{{a['family']}}}}}"
                   f"\\email{{{a['email']}}}")
    for k, f in AFF.items():
        star = "*" if any(a["corresponding"] and a["affiliation"] == k for a in AUTHORS) else ""
        out.append(f"\\affil{star}[{codes[k]}]{{\\orgdiv{{{f['department']}}}, \\orgname{{{f['institution']}}}, "
                   f"\\orgaddress{{\\street{{{f['street']}}}, \\city{{{f['city']}}}, \\postcode{{{f['postcode']}}}, "
                   f"\\country{{{f['country']}}}}}}}")
    return "\n".join(out) + "\n"


def elsarticle():
    out = []
    for a in AUTHORS:
        cor = "\\corref{cor1}" if a["corresponding"] else ""
        out.append(f"\\author[{a['affiliation']}]{{{full(a)}{cor}}}\n\\ead{{{a['email']}}}")
    out.append("\\cortext[cor1]{Corresponding author}")
    for k, f in AFF.items():
        out.append(f"\\affiliation[{k}]{{organization={{{f['department']}, {f['institution']}}}, "
                   f"addressline={{{f['street']}}}, city={{{f['city']}}}, postcode={{{f['postcode']}}}, "
                   f"country={{{f['country']}}}}}")
    return "\n".join(out) + "\n"


def ieee_camera_ready():
    """IEEE author block for the accepted version (the review version stays anonymous)."""
    names = " and ".join(full(a) for a in AUTHORS)
    f = AFF[AUTHORS[0]["affiliation"]]
    return (f"% \\author{{{names}% <-this % stops a space\n"
            f"% \\thanks{{The authors are with the {f['department']}, {f['institution']}, {f['city']} {f['postcode']}, "
            f"{f['country']} (e-mail: {', '.join(a['email'] for a in AUTHORS)}).}}}}\n")


def credit_elsevier():
    return " ".join(f"\\textbf{{{full(a)}:}} {a['credit']}." for a in AUTHORS)


def credit_springer():
    return "; ".join(f"{initials(a)}: {a['credit']}" for a in AUTHORS) + "."


def rescience_yaml():
    codes = {k: i + 1 for i, k in enumerate(AFF)}
    lines = ["authors:"]
    for a in AUTHORS:
        aff = str(codes[a["affiliation"]]) + (",*" if a["corresponding"] else "")
        lines += [f"  - name: {full(a)}", f"    orcid: \"{a['orcid']}\"", f"    email: {a['email']}",
                  f"    affiliations: {aff}", ""]
    lines.append("affiliations:")
    for k, f in AFF.items():
        lines += [f"  - code:    {codes[k]}", f"    name:    \"{f['department']}, {f['institution']}\"",
                  f"    address: \"{f['street']}, {f['city']} {f['postcode']}, {f['country']}\"", ""]
    return "\n".join(lines)
