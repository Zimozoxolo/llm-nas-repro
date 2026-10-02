import json
from pathlib import Path

import pytest

from llmnas.search_space import (FULL_SPACE, PAPER_NICHES, all_codes, code_to_str, naslib_key_to_code,
                                 str_to_code)

DATA = Path(__file__).resolve().parents[1] / "data"


def test_roundtrip():
    for c in all_codes()[::97]:
        assert str_to_code(code_to_str(c)) == c


def test_niches_partition_space():
    codes = all_codes()
    assert len(codes) == 15625
    for c in codes:
        assert sum(n.contains(c) for n in PAPER_NICHES) == 1
        assert FULL_SPACE.contains(c)


def test_niche3_prompt_matches_appendix_e():
    txt = PAPER_NICHES[3].constraint_text()
    assert "exactly 2 x nor_conv_3x3" in txt and "0-4 x nor_conv_1x1" in txt


@pytest.mark.skipif(not (DATA / "processed" / "nb201_cifar10.csv").exists(), reason="data not prepared")
def test_naslib_mapping_against_processed_table():
    import pandas as pd

    df = pd.read_csv(DATA / "processed" / "nb201_cifar10.csv")
    val = dict(zip(df.arch_str, df.val_acc))
    zc = json.load(open(DATA / "zc_nasbench201.json"))["cifar10"]
    for key, rec in list(zc.items())[:2000]:
        assert abs(val[code_to_str(naslib_key_to_code(key))] - rec["val_accuracy"]) < 1e-6
