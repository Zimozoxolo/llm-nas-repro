"""Download raw benchmark files and build one tidy table per dataset.

Raw sources (all public):

* HW-NAS-Bench v1.0  - per-device latency for the 15,625 NB201 cells
  https://github.com/RICE-EIC/HW-NAS-Bench
* NAS-Bench-Suite-Zero ``zc_nasbench201.json`` - the 13 zero-cost proxies + val accuracy
  https://github.com/automl/NASLib/tree/zerocost
* NATS-Bench topology space (= NAS-Bench-201) - 200-epoch *test* accuracy, which is
  what the paper reports (e.g. the 94.37% CIFAR-10 optimum in Table 2)
  https://github.com/D-X-Y/NATS-Bench

Output: ``data/processed/nb201_<dataset>.csv`` with columns

    idx, arch_str, code, test_acc, val_acc, <13 zc proxies>, <6 latencies>
"""

from __future__ import annotations

import json
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from .search_space import code_to_str, format_code, naslib_key_to_code, str_to_code

DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
DEVICES = {  # short name -> HW-NAS-Bench metric key (paper's names in comments)
    "edgegpu": "edgegpu_latency",  # Edge GPU (NVIDIA Jetson TX2)
    "raspi4": "raspi4_latency",  # Raspberry Pi 4
    "edgetpu": "edgetpu_latency",  # Edge TPU (Google TPU Dev Board)
    "pixel3": "pixel3_latency",  # Pixel 3
    "eyeriss": "eyeriss_latency",  # ASIC (Eyeriss)
    "fpga": "fpga_latency",  # FPGA
}
DEVICE_LABELS = {
    "edgegpu": "Edge GPU",
    "raspi4": "Raspi 4",
    "edgetpu": "Edge TPU",
    "pixel3": "Pixel 3",
    "eyeriss": "Eyeriss",
    "fpga": "FPGA",
}
ZC_PROXIES = [
    "epe_nas", "fisher", "flops", "grad_norm", "grasp", "jacov", "l2_norm",
    "nwot", "params", "plain", "snip", "synflow", "zen",
]  # fmt: skip

HWNB_URL = "https://github.com/RICE-EIC/HW-NAS-Bench/raw/main/HW-NAS-Bench-v1_0.pickle"
GDRIVE_IDS = {
    "zc_nasbench201.json": "1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H",
    "NATS-tss-v1_0-3ffb9-simple.tar": "17_saCsj_krKjlCBLOJEpNtzPXArMCqxU",
}
NATS_DIR = "NATS-tss-v1_0-3ffb9-simple"


def default_data_dir() -> Path:
    return Path(os.environ.get("LLMNAS_DATA", Path(__file__).resolve().parents[2] / "data"))


def download_raw(data_dir: Path | None = None) -> None:
    data_dir = Path(data_dir or default_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)

    hw = data_dir / "HW-NAS-Bench-v1_0.pickle"
    if not hw.exists():
        print(f"Downloading {hw.name} ...")
        urllib.request.urlretrieve(HWNB_URL, hw)

    import gdown

    for name, fid in GDRIVE_IDS.items():
        target = data_dir / name
        if name.endswith(".tar") and (data_dir / NATS_DIR).exists():
            continue
        if not target.exists():
            print(f"Downloading {name} from Google Drive ...")
            gdown.download(id=fid, output=str(target), quiet=False)
        if name.endswith(".tar"):
            print(f"Extracting {name} ...")
            with tarfile.open(target) as tf:
                tf.extractall(data_dir)
            target.unlink()


def _nats_test_accuracies(data_dir: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """arch_str -> dataset -> (test_acc, valid_acc), 200-epoch schedule, mean over seeds."""
    from nats_bench import create

    api = create(str(data_dir / NATS_DIR), "tss", fast_mode=True, verbose=False)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for i in range(len(api)):
        arch = api.arch(i)
        out[arch] = {}
        for ds in DATASETS:
            # cifar10: train on train+valid, test on ori-test (the 94.37% number);
            # cifar100 / ImageNet16-120: test on x-test.
            info = api.get_more_info(i, ds, hp="200", is_random=False)
            out[arch][ds] = (info["test-accuracy"], info.get("valid-accuracy", float("nan")))
        api.clear_params(i)
        if i % 2000 == 0:
            print(f"  NATS {i}/{len(api)}")
    return out


def build_tables(data_dir: Path | None = None) -> None:
    data_dir = Path(data_dir or default_data_dir())
    out_dir = data_dir / "processed"
    out_dir.mkdir(exist_ok=True)

    hw = pickle.load(open(data_dir / "HW-NAS-Bench-v1_0.pickle", "rb"))["nasbench201"]
    zc = json.load(open(data_dir / "zc_nasbench201.json"))
    print("Reading NATS-Bench test accuracies (takes a few minutes) ...")
    nats = _nats_test_accuracies(data_dir)

    for ds in DATASETS:
        rows = {}
        for idx, cfg in enumerate(hw[ds]["config"]):
            code = str_to_code(cfg["arch_str"])
            assert code_to_str(code) == cfg["arch_str"]
            row = {"idx": idx, "arch_str": cfg["arch_str"], "code": format_code(code)}
            row["test_acc"] = nats[cfg["arch_str"]][ds][0]
            for dev, key in DEVICES.items():
                row[f"lat_{dev}"] = float(hw[ds][key][idx])
            rows[code] = row
        for key, rec in zc[ds].items():
            row = rows[naslib_key_to_code(key)]
            row["val_acc"] = rec["val_accuracy"]
            for p in ZC_PROXIES:
                row[f"zc_{p}"] = rec[p]["score"]
        df = pd.DataFrame(rows.values()).sort_values("idx")
        cols = ["idx", "arch_str", "code", "test_acc", "val_acc"]
        cols += [f"zc_{p}" for p in ZC_PROXIES] + [f"lat_{d}" for d in DEVICES]
        df = df[cols]
        assert len(df) == 15625 and not df.isna().any().any(), f"incomplete table for {ds}"
        path = out_dir / f"nb201_{ds}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}  (best test acc {df.test_acc.max():.2f})")


def prepare(data_dir: Path | None = None) -> None:
    download_raw(data_dir)
    build_tables(data_dir)
