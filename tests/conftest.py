import numpy as np
import pandas as pd
import pytest

from llmnas.data import DEVICES, ZC_PROXIES
from llmnas.search_space import CONV1, CONV3, all_codes, code_to_str, format_code


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """A fake processed table with the real search space and made-up metrics."""
    root = tmp_path_factory.mktemp("data")
    (root / "processed").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, c in enumerate(all_codes()):
        n3, n1 = sum(x == CONV3 for x in c), sum(x == CONV1 for x in c)
        acc = 40 + 8 * n3 + 4 * n1 + rng.normal(0, 2)
        row = dict(idx=i, arch_str=code_to_str(c), code=format_code(c), test_acc=acc, val_acc=acc + rng.normal(0, 1))
        row.update({f"zc_{p}": acc + rng.normal(0, 5) for p in ZC_PROXIES})
        row.update({f"lat_{d}": 1 + 1.5 * n3 + 0.5 * n1 + rng.random() for d in DEVICES})
        rows.append(row)
    df = pd.DataFrame(rows)
    for ds in ["cifar10", "cifar100", "ImageNet16-120"]:
        df.to_csv(root / "processed" / f"nb201_{ds}.csv", index=False)
    return root
