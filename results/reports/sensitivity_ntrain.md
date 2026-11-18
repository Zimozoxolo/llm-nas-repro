# Predictor training-set size: sensitivity of the no-LLM controls

HV and IGD under the default convention (front normalisation, reference point 1.1, archived front), mean over 6 devices x 3 seeds. `rank` orders the four controls by HV within each (dataset, n_train). The LLM runs all use n_train = 1,000 and are not repeated here.


## cifar10

| Method | HV (n=200) | HV (n=1000) | HV (n=5000) | IGD (n=200) | IGD (n=1000) | IGD (n=5000) | rank (n=200) | rank (n=1000) | rank (n=5000) |
|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 0.964 | 0.969 | 0.973 | 0.0796 | 0.0648 | 0.0447 | 2 | 1 | 2 |
| PEA (no LLM) | 0.938 | 0.952 | 0.951 | 0.0815 | 0.0612 | 0.0612 | 4 | 3 | 3 |
| Random sampling (no LLM) | 0.944 | 0.950 | 0.948 | 0.0775 | 0.0645 | 0.0609 | 3 | 4 | 4 |
| Reg. evolution (no LLM) | 0.965 | 0.967 | 0.977 | 0.0629 | 0.0553 | 0.0388 | 1 | 2 | 1 |

Predictor Spearman correlation on held-out architectures: n=200: 0.943, n=1000: 0.960, n=5000: 0.969


## cifar100

| Method | HV (n=200) | HV (n=1000) | HV (n=5000) | IGD (n=200) | IGD (n=1000) | IGD (n=5000) | rank (n=200) | rank (n=1000) | rank (n=5000) |
|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 0.946 | 0.959 | 0.972 | 0.0866 | 0.0492 | 0.0411 | 2 | 2 | 2 |
| PEA (no LLM) | 0.918 | 0.946 | 0.947 | 0.0808 | 0.0623 | 0.0569 | 4 | 3 | 3 |
| Random sampling (no LLM) | 0.919 | 0.931 | 0.933 | 0.0891 | 0.0614 | 0.0602 | 3 | 4 | 4 |
| Reg. evolution (no LLM) | 0.948 | 0.962 | 0.972 | 0.0746 | 0.0449 | 0.0371 | 1 | 1 | 1 |

Predictor Spearman correlation on held-out architectures: n=200: 0.944, n=1000: 0.964, n=5000: 0.974


## ImageNet16-120

| Method | HV (n=200) | HV (n=1000) | HV (n=5000) | IGD (n=200) | IGD (n=1000) | IGD (n=5000) | rank (n=200) | rank (n=1000) | rank (n=5000) |
|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 0.944 | 0.949 | 0.959 | 0.0687 | 0.0563 | 0.0441 | 1 | 2 | 1 |
| PEA (no LLM) | 0.897 | 0.910 | 0.921 | 0.0804 | 0.0734 | 0.0576 | 3 | 3 | 4 |
| Random sampling (no LLM) | 0.896 | 0.904 | 0.926 | 0.0844 | 0.0724 | 0.0619 | 4 | 4 | 3 |
| Reg. evolution (no LLM) | 0.937 | 0.953 | 0.959 | 0.0708 | 0.0458 | 0.0417 | 2 | 1 | 2 |

Predictor Spearman correlation on held-out architectures: n=200: 0.929, n=1000: 0.957, n=5000: 0.970

