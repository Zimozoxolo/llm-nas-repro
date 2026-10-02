"""Numbers reported in the paper (arXiv:2510.01472v4), for side-by-side comparison.

Baseline rows (LLMatic, FairNAS, PRP-NAS, DARTS) are *quoted*, not re-run: they need
GPU-days of supernet training or training every candidate (Table 4).
"""

DEVICE_ORDER = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]


def _rows(flat: list[float]) -> dict[str, tuple[float, float]]:
    return {d: (flat[2 * i], flat[2 * i + 1]) for i, d in enumerate(DEVICE_ORDER)}


# Table 3: (HV, IGD) per device
TABLE3 = {
    "cifar10": {
        "LLMatic": _rows([0.191, 0.542, 0.549, 0.296, 0.354, 0.514, 0.551, 0.337, 0.512, 0.331, 0.586, 0.370]),
        "FairNAS": _rows([0.892, 0.073, 0.962, 0.035, 0.947, 0.089, 0.971, 0.033, 0.958, 0.068, 0.918, 0.091]),
        "PRP-NAS": _rows([0.843, 0.116, 0.926, 0.133, 0.916, 0.123, 0.926, 0.124, 0.928, 0.145, 0.903, 0.241]),
        "LLM-NAS": _rows([0.997, 0.006, 0.997, 0.013, 0.955, 0.057, 0.996, 0.011, 0.961, 0.037, 0.931, 0.046]),
    },
    "cifar100": {
        "LLMatic": _rows([0.233, 0.571, 0.516, 0.411, 0.455, 0.465, 0.745, 0.256, 0.552, 0.297, 0.598, 0.241]),
        "FairNAS": _rows([0.853, 0.072, 0.930, 0.058, 0.929, 0.102, 0.930, 0.055, 0.952, 0.110, 0.958, 0.117]),
        "PRP-NAS": _rows([0.794, 0.161, 0.824, 0.179, 0.751, 0.190, 0.817, 0.174, 0.863, 0.246, 0.798, 0.317]),
        "LLM-NAS": _rows([0.992, 0.009, 0.994, 0.016, 0.981, 0.017, 0.985, 0.023, 0.962, 0.050, 0.977, 0.032]),
    },
    "ImageNet16-120": {
        "LLMatic": _rows([0.285, 0.566, 0.340, 0.461, 0.279, 0.632, 0.783, 0.193, 0.392, 0.428, 0.678, 0.230]),
        "FairNAS": _rows([0.838, 0.115, 0.894, 0.048, 0.851, 0.122, 0.907, 0.067, 0.912, 0.086, 0.916, 0.079]),
        "PRP-NAS": _rows([0.833, 0.096, 0.857, 0.082, 0.887, 0.116, 0.892, 0.073, 0.879, 0.096, 0.876, 0.113]),
        "LLM-NAS": _rows([0.953, 0.043, 0.988, 0.011, 0.943, 0.033, 0.983, 0.042, 0.945, 0.050, 0.972, 0.028]),
    },
}

# Table 5 (CIFAR-100, averaged over the six devices): (HV mean, HV std, IGD mean, IGD std)
# Note: the mean of Table 3's CIFAR-100 LLM-NAS HVs is 0.982, not 0.978 as printed here.
TABLE5 = {
    "llmnas": (0.978, 0.017, 0.0246, 0.0132),
    "no_partitioning": (0.516, 0.155, 0.3734, 0.1197),
    "no_llm_pea": (0.843, 0.075, 0.1649, 0.0311),
    "synflow_only": (0.819, 0.112, 0.1717, 0.0381),
}

# Table 2 (CIFAR-10): device -> list of (acc, acc_std, latency_ms) rows, LLM-NAS only
TABLE2_LLMNAS = {
    "edgegpu": [(94.37, 0.02, 4.35), (93.88, 0.10, 3.36), (89.18, 0.15, 1.78)],
    "raspi4": [(94.37, 0.15, 69.76), (92.37, 0.07, 18.67), (90.70, 0.12, 7.29)],
    "pixel3": [(94.30, 0.15, 21.59), (93.31, 0.05, 8.98), (90.36, 0.08, 2.57)],
    "fpga": [(94.37, 0.14, 6.68), (93.29, 0.15, 2.91), (89.57, 0.25, 1.65)],
}
TABLE2_BASELINES = {  # method -> device -> (acc, latency)
    "DARTS": {"edgegpu": (68.30, 5.36), "raspi4": (68.30, 45.36), "pixel3": (68.30, 11.4), "fpga": (68.30, 7.32)},
    "FairNAS": {"edgegpu": (93.23, 4.68), "raspi4": (92.51, 34.15), "pixel3": (92.40, 8.65), "fpga": (92.90, 5.12)},
    "PRP-NAS-BA": {"edgegpu": (94.37, 4.35), "raspi4": (93.68, 40.7), "pixel3": (94.20, 5.60), "fpga": (94.37, 6.80)},
    "PRP-NAS-BL": {"edgegpu": (92.34, 2.30), "raspi4": (88.70, 7.60), "pixel3": (89.57, 3.60), "fpga": (91.35, 3.60)},
    "LLMatic": {"edgegpu": (94.26, 6.80), "raspi4": (94.26, 69.06), "pixel3": (94.26, 21.59), "fpga": (94.26, 6.67)},
}

# Table 4: search cost per dataset per device
TABLE4 = {"LLMatic": "17 GPU days", "FairNAS": "10 GPU days", "DARTS": "4 GPU days",
          "PRP-NAS-BA": "2 GPU days", "LLM-NAS": "3 mins (120 API calls)"}  # fmt: skip

# Table 6 (ViT on ImageNet-1k, A6000) - not reproduced here, see REPRODUCTION.md
TABLE6 = {"LLM-NAS-ViT-Tiny": (76.2, 4.0, 6.9), "LLM-NAS-ViT-Small": (79.7, 4.7, 16.1),
          "LLM-NAS-ViT-Base": (82.5, 5.4, 20.2)}  # fmt: skip

HYPERPARAMETERS = {  # stated in Sec. 4.1
    "llm": "GPT-4.1", "generations": 10, "crossover_prob": 0.5, "api_calls": 120,
    "predictor": "XGBoost on 13 NAS-Bench-Suite-Zero proxies (Spearman ~0.90)",
}  # fmt: skip
