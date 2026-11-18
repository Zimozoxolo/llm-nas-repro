"""Command line interface:  python -m llmnas <command> ...

  prepare-data                     download benchmarks + build data/processed/*.csv
  check-data                       sanity checks (paper's Table-2 points, predictor Spearman)
  run <experiment> [filters]       run an experiment grid (resumable)
  report <experiment>              tables + figures -> results/reports/<experiment>/
  fig3                             Figure 3 (parameter histogram by #conv3x3)
  compare-llms                     side-by-side table of all LLM runs (quality, tokens, cost)
  slm-reference                    reference SLMs: parameter-count check + modelled costs
  slm-calibrate --device cpu|cuda  fit the SLM latency model to measurements on this machine
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the repo's .env (git-ignored) without overriding real env vars."""
    import os

    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _filters(a) -> dict:
    return dict(variants=a.variants, datasets=a.datasets, devices=a.devices, seeds=a.seeds)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="llmnas", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare-data").add_argument("--data-dir", default=None)
    sub.add_parser("check-data").add_argument("--data-dir", default=None)

    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("experiment", help="name in configs/experiments/ or a path to a yaml file")
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="override config, e.g. --set llm.model=gpt-4o search.generations=5")
        p.add_argument("--variants", nargs="*")
        p.add_argument("--datasets", nargs="*")
        p.add_argument("--devices", nargs="*")
        p.add_argument("--seeds", nargs="*", type=int)
        if name == "run":
            p.add_argument("--force", action="store_true", help="recompute finished runs")

    sub.add_parser("fig3").add_argument("--out", default="results/reports/fig3_params.png")

    p = sub.add_parser("compare-llms", help="side-by-side table of every LLM run on one dataset/device")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--device", default="edgegpu")
    p.add_argument("--min-acc", type=float, default=80.0, help="also score with architectures below this test accuracy removed")

    p = sub.add_parser("conventions", help="re-score every archived run under 16 hypervolume conventions")
    p.add_argument("--datasets", nargs="*", default=["cifar10", "cifar100"])

    p = sub.add_parser("slm-reference", help="check reference SLMs: parameter counts and modelled costs")
    p.add_argument("--device", default="jetson_orin_nx_16gb")
    p = sub.add_parser("slm-calibrate", help="measure random transformers and fit the latency model")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--space", default="lts_gpt2")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-decoder-m", type=float, default=None, help="skip candidates larger than this")
    p.add_argument("--out", required=True, help="e.g. configs/slm/devices/local_cpu.json")
    a = ap.parse_args(argv)
    load_dotenv()

    if a.cmd == "prepare-data":
        from .data import prepare

        prepare(Path(a.data_dir) if a.data_dir else None)
    elif a.cmd == "check-data":
        from .checks import check_data

        check_data(a.data_dir)
    elif a.cmd in ("run", "report"):
        from .experiments import load_experiment, parse_sets, run_experiment

        exp = load_experiment(a.experiment, parse_sets(a.set))
        if a.cmd == "run":
            run_experiment(exp, force=a.force, **_filters(a))
        elif exp.get("type") == "slm":
            from .slm.report import make_slm_report

            make_slm_report(exp)
        else:
            from .report import make_report

            make_report(exp)
    elif a.cmd == "compare-llms":
        from .compare import compare_llms

        print(compare_llms(a.dataset, a.device, min_acc=a.min_acc))
    elif a.cmd == "conventions":
        from .conventions import report as conventions_report

        print(conventions_report(tuple(a.datasets)))
    elif a.cmd == "slm-reference":
        from .slm.cli import reference

        reference(a.device)
    elif a.cmd == "slm-calibrate":
        from .slm.cli import calibrate_cmd

        calibrate_cmd(a)
    elif a.cmd == "fig3":
        from .report import plot_fig3

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        plot_fig3(Path(a.out))
        print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
