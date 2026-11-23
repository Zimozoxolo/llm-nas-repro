# Robustness of the results to the hypervolume convention

Every archived run re-scored under 16 conventions (normalisation x reference point x scored set x accuracy floor at 85% of the best accuracy). Device-level values are means over seeds; the tables give the mean over the devices that the method was run on. Higher HV and lower IGD are better. No search was re-run and no LLM was called.

Significance (one-sided, H1: the LLM runs are worse than the control): *pooled sign* = binomial test on the wins, treating every LLM-device pair as independent; *Wilcoxon* = signed-rank test on the same pairs; *device-clustered* = exact sign-flip permutation test on the mean LLM-minus-control difference per device (pairs on one device share the control, so this is the conservative test; with six devices its smallest possible p is 1/64 = 0.016).


## cifar10: mean HV over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 6 | 0.888 | 0.821 | 0.895 | 0.832 | 0.872 | 0.786 | 0.879 | 0.799 | 0.977 | 0.933 | 0.977 | 0.937 | 0.975 | 0.921 | 0.976 | 0.926 |
| claude-opus-5-5 | 1 | 0.980 | 0.961 | 0.981 | 0.976 | 0.993 | 0.956 | 0.995 | 0.971 | 0.990 | 0.989 | 0.990 | 0.993 | 0.997 | 0.987 | 0.997 | 0.992 |
| deepseek-coder-v2-16b-8k | 6 | 0.890 | 0.813 | 0.897 | 0.825 | 0.870 | 0.778 | 0.877 | 0.789 | 0.981 | 0.924 | 0.983 | 0.935 | 0.978 | 0.912 | 0.981 | 0.923 |
| gpt-4.1 | 6 | 0.928 | 0.860 | 0.940 | 0.879 | 0.919 | 0.833 | 0.933 | 0.854 | 0.989 | 0.953 | 0.990 | 0.959 | 0.988 | 0.943 | 0.990 | 0.951 |
| gpt-4.1 [history + re-prompt] | 6 | 0.932 | 0.892 | 0.944 | 0.910 | 0.922 | 0.873 | 0.935 | 0.891 | 0.989 | 0.959 | 0.991 | 0.969 | 0.989 | 0.951 | 0.991 | 0.963 |
| gpt-4.1 [history] | 6 | 0.942 | 0.903 | 0.951 | 0.917 | 0.935 | 0.884 | 0.944 | 0.900 | 0.990 | 0.963 | 0.990 | 0.968 | 0.990 | 0.957 | 0.991 | 0.963 |
| qwen2.5-7b-8k | 6 | 0.894 | 0.815 | 0.904 | 0.836 | 0.879 | 0.784 | 0.888 | 0.805 | 0.975 | 0.923 | 0.978 | 0.937 | 0.972 | 0.911 | 0.975 | 0.926 |
| NSGA-II (no LLM) | 6 | 0.970 | 0.945 | 0.983 | 0.968 | 0.967 | 0.938 | 0.981 | 0.963 | 0.994 | 0.973 | 0.997 | 0.986 | 0.994 | 0.970 | 0.996 | 0.984 |
| PEA (no LLM) | 6 | 0.952 | 0.911 | 0.960 | 0.929 | 0.945 | 0.897 | 0.953 | 0.916 | 0.991 | 0.961 | 0.993 | 0.972 | 0.990 | 0.956 | 0.992 | 0.967 |
| PEA with refill (no LLM) | 6 | 0.959 | 0.923 | 0.971 | 0.943 | 0.953 | 0.910 | 0.965 | 0.930 | 0.993 | 0.967 | 0.996 | 0.981 | 0.992 | 0.963 | 0.995 | 0.977 |
| Random sampling (no LLM) | 6 | 0.948 | 0.907 | 0.956 | 0.922 | 0.940 | 0.891 | 0.948 | 0.906 | 0.991 | 0.961 | 0.993 | 0.971 | 0.990 | 0.955 | 0.992 | 0.965 |
| Reg. evolution (no LLM) | 6 | 0.969 | 0.941 | 0.983 | 0.965 | 0.966 | 0.932 | 0.981 | 0.958 | 0.995 | 0.973 | 0.997 | 0.986 | 0.994 | 0.969 | 0.997 | 0.984 |

### cifar10: rank by HV under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 2 | 3 |
| PEA (no LLM) | 5 | 6 |
| PEA with refill (no LLM) | 3 | 4 |
| Random sampling (no LLM) | 4 | 7 |
| Reg. evolution (no LLM) | 1 | 3 |
| claude-haiku-4-5 | 10 | 12 |
| claude-opus-5-5 | 1 | 8 |
| deepseek-coder-v2-16b-8k | 10 | 12 |
| gpt-4.1 | 9 | 9 |
| gpt-4.1 [history + re-prompt] | 6 | 8 |
| gpt-4.1 [history] | 5 | 8 |
| qwen2.5-7b-8k | 10 | 12 |

### cifar10: LLM runs that beat each control by HV (same device and convention)

| Convention | beats PEA (no LLM) | beats Random sampling (no LLM) | beats NSGA-II (no LLM) | beats Reg. evolution (no LLM) | beats PEA with refill (no LLM) |
|---|---|---|---|---|---|
| front / r=1.1 / archive / no floor | 0/25 | 3/25 | 0/25 | 0/25 | 0/25 |
| front / r=1.1 / archive / floor 85% | 1/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| front / r=1.1 / all / no floor | 1/25 | 3/25 | 0/25 | 0/25 | 0/25 |
| front / r=1.1 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| front / r=1.0 / archive / no floor | 1/25 | 3/25 | 0/25 | 0/25 | 1/25 |
| front / r=1.0 / archive / floor 85% | 1/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| front / r=1.0 / all / no floor | 2/25 | 3/25 | 0/25 | 0/25 | 1/25 |
| front / r=1.0 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.1 / archive / no floor | 1/25 | 3/25 | 0/25 | 0/25 | 0/25 |
| space / r=1.1 / archive / floor 85% | 1/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.1 / all / no floor | 2/25 | 3/25 | 0/25 | 0/25 | 0/25 |
| space / r=1.1 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.0 / archive / no floor | 2/25 | 4/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.0 / archive / floor 85% | 1/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.0 / all / no floor | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.0 / all / floor 85% | 1/25 | 3/25 | 1/25 | 1/25 | 1/25 |

## cifar10: mean IGD over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 6 | 0.089 | 0.127 | 0.081 | 0.120 | 0.089 | 0.127 | 0.081 | 0.120 | 0.044 | 0.074 | 0.041 | 0.070 | 0.044 | 0.074 | 0.041 | 0.070 |
| claude-opus-5-5 | 1 | 0.052 | 0.041 | 0.045 | 0.024 | 0.052 | 0.041 | 0.045 | 0.024 | 0.044 | 0.022 | 0.040 | 0.013 | 0.044 | 0.022 | 0.040 | 0.013 |
| deepseek-coder-v2-16b-8k | 6 | 0.098 | 0.134 | 0.085 | 0.121 | 0.098 | 0.134 | 0.085 | 0.121 | 0.042 | 0.078 | 0.035 | 0.070 | 0.042 | 0.078 | 0.035 | 0.070 |
| gpt-4.1 | 6 | 0.072 | 0.101 | 0.059 | 0.086 | 0.072 | 0.101 | 0.059 | 0.086 | 0.039 | 0.059 | 0.032 | 0.050 | 0.039 | 0.059 | 0.032 | 0.050 |
| gpt-4.1 [history + re-prompt] | 6 | 0.083 | 0.098 | 0.053 | 0.065 | 0.083 | 0.098 | 0.053 | 0.065 | 0.041 | 0.056 | 0.027 | 0.038 | 0.041 | 0.056 | 0.027 | 0.038 |
| gpt-4.1 [history] | 6 | 0.067 | 0.075 | 0.067 | 0.075 | 0.067 | 0.075 | 0.067 | 0.075 | 0.035 | 0.044 | 0.039 | 0.045 | 0.035 | 0.044 | 0.039 | 0.045 |
| qwen2.5-7b-8k | 6 | 0.093 | 0.128 | 0.082 | 0.116 | 0.093 | 0.128 | 0.082 | 0.116 | 0.040 | 0.075 | 0.034 | 0.067 | 0.040 | 0.075 | 0.034 | 0.067 |
| NSGA-II (no LLM) | 6 | 0.057 | 0.068 | 0.030 | 0.037 | 0.057 | 0.068 | 0.030 | 0.037 | 0.029 | 0.040 | 0.015 | 0.022 | 0.029 | 0.040 | 0.015 | 0.022 |
| PEA (no LLM) | 6 | 0.061 | 0.080 | 0.047 | 0.064 | 0.061 | 0.080 | 0.047 | 0.064 | 0.029 | 0.047 | 0.021 | 0.037 | 0.029 | 0.047 | 0.021 | 0.037 |
| PEA with refill (no LLM) | 6 | 0.058 | 0.075 | 0.035 | 0.048 | 0.058 | 0.075 | 0.035 | 0.048 | 0.027 | 0.044 | 0.014 | 0.028 | 0.027 | 0.044 | 0.014 | 0.028 |
| Random sampling (no LLM) | 6 | 0.065 | 0.086 | 0.048 | 0.067 | 0.065 | 0.086 | 0.048 | 0.067 | 0.029 | 0.050 | 0.020 | 0.039 | 0.029 | 0.050 | 0.020 | 0.039 |
| Reg. evolution (no LLM) | 6 | 0.053 | 0.067 | 0.028 | 0.038 | 0.053 | 0.067 | 0.028 | 0.038 | 0.025 | 0.039 | 0.013 | 0.022 | 0.025 | 0.039 | 0.013 | 0.022 |

### cifar10: rank by IGD under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 2 | 4 |
| PEA (no LLM) | 3 | 6 |
| PEA with refill (no LLM) | 2 | 4 |
| Random sampling (no LLM) | 4 | 7 |
| Reg. evolution (no LLM) | 1 | 3 |
| claude-haiku-4-5 | 10 | 12 |
| claude-opus-5-5 | 1 | 12 |
| deepseek-coder-v2-16b-8k | 9 | 12 |
| gpt-4.1 | 7 | 9 |
| gpt-4.1 [history + re-prompt] | 6 | 9 |
| gpt-4.1 [history] | 5 | 10 |
| qwen2.5-7b-8k | 8 | 11 |

### cifar10: LLM runs that beat each control by IGD (same device and convention)

| Convention | beats PEA (no LLM) | beats Random sampling (no LLM) | beats NSGA-II (no LLM) | beats Reg. evolution (no LLM) | beats PEA with refill (no LLM) |
|---|---|---|---|---|---|
| front / r=1.1 / archive / no floor | 2/25 | 2/25 | 3/25 | 1/25 | 1/25 |
| front / r=1.1 / archive / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| front / r=1.1 / all / no floor | 1/25 | 2/25 | 0/25 | 0/25 | 0/25 |
| front / r=1.1 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| front / r=1.0 / archive / no floor | 2/25 | 2/25 | 3/25 | 1/25 | 1/25 |
| front / r=1.0 / archive / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| front / r=1.0 / all / no floor | 1/25 | 2/25 | 0/25 | 0/25 | 0/25 |
| front / r=1.0 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.1 / archive / no floor | 3/25 | 2/25 | 2/25 | 2/25 | 2/25 |
| space / r=1.1 / archive / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.1 / all / no floor | 1/25 | 3/25 | 0/25 | 0/25 | 0/25 |
| space / r=1.1 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.0 / archive / no floor | 3/25 | 2/25 | 2/25 | 2/25 | 2/25 |
| space / r=1.0 / archive / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |
| space / r=1.0 / all / no floor | 1/25 | 3/25 | 0/25 | 0/25 | 0/25 |
| space / r=1.0 / all / floor 85% | 2/25 | 3/25 | 1/25 | 1/25 | 1/25 |

## cifar100: mean HV over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 6 | 0.959 | 0.900 | 0.978 | 0.940 | 0.954 | 0.886 | 0.976 | 0.932 | 0.987 | 0.917 | 0.992 | 0.949 | 0.986 | 0.906 | 0.991 | 0.942 |
| PEA (no LLM) | 6 | 0.946 | 0.849 | 0.953 | 0.871 | 0.939 | 0.823 | 0.945 | 0.846 | 0.982 | 0.888 | 0.986 | 0.914 | 0.980 | 0.870 | 0.984 | 0.898 |
| Random sampling (no LLM) | 6 | 0.931 | 0.830 | 0.940 | 0.849 | 0.919 | 0.804 | 0.928 | 0.824 | 0.981 | 0.877 | 0.984 | 0.896 | 0.978 | 0.858 | 0.981 | 0.879 |
| Reg. evolution (no LLM) | 6 | 0.962 | 0.912 | 0.982 | 0.956 | 0.957 | 0.901 | 0.979 | 0.947 | 0.989 | 0.929 | 0.995 | 0.969 | 0.988 | 0.919 | 0.994 | 0.963 |

### cifar100: rank by HV under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 2 | 2 |
| PEA (no LLM) | 3 | 3 |
| Random sampling (no LLM) | 4 | 4 |
| Reg. evolution (no LLM) | 1 | 1 |

## cifar100: mean IGD over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 6 | 0.049 | 0.087 | 0.032 | 0.058 | 0.049 | 0.087 | 0.032 | 0.058 | 0.028 | 0.071 | 0.018 | 0.047 | 0.028 | 0.071 | 0.018 | 0.047 |
| PEA (no LLM) | 6 | 0.062 | 0.115 | 0.049 | 0.091 | 0.062 | 0.115 | 0.049 | 0.091 | 0.037 | 0.094 | 0.029 | 0.075 | 0.037 | 0.094 | 0.029 | 0.075 |
| Random sampling (no LLM) | 6 | 0.061 | 0.121 | 0.053 | 0.111 | 0.061 | 0.121 | 0.053 | 0.111 | 0.033 | 0.100 | 0.028 | 0.091 | 0.033 | 0.100 | 0.028 | 0.091 |
| Reg. evolution (no LLM) | 6 | 0.045 | 0.083 | 0.024 | 0.048 | 0.045 | 0.083 | 0.024 | 0.048 | 0.025 | 0.068 | 0.013 | 0.040 | 0.025 | 0.068 | 0.013 | 0.040 |

### cifar100: rank by IGD under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 2 | 2 |
| PEA (no LLM) | 3 | 4 |
| Random sampling (no LLM) | 3 | 4 |
| Reg. evolution (no LLM) | 1 | 1 |

## ImageNet16-120: mean HV over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 6 | 0.949 | 0.792 | 0.974 | 0.881 | 0.944 | 0.773 | 0.971 | 0.872 | 0.973 | 0.826 | 0.985 | 0.904 | 0.971 | 0.808 | 0.984 | 0.895 |
| PEA (no LLM) | 6 | 0.910 | 0.656 | 0.928 | 0.738 | 0.894 | 0.612 | 0.914 | 0.701 | 0.962 | 0.759 | 0.973 | 0.838 | 0.956 | 0.732 | 0.969 | 0.819 |
| Random sampling (no LLM) | 6 | 0.904 | 0.628 | 0.921 | 0.705 | 0.888 | 0.571 | 0.907 | 0.658 | 0.961 | 0.758 | 0.970 | 0.829 | 0.955 | 0.726 | 0.965 | 0.805 |
| Reg. evolution (no LLM) | 6 | 0.953 | 0.814 | 0.970 | 0.875 | 0.946 | 0.796 | 0.965 | 0.862 | 0.978 | 0.852 | 0.987 | 0.917 | 0.975 | 0.837 | 0.985 | 0.908 |

### ImageNet16-120: rank by HV under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 1 | 2 |
| PEA (no LLM) | 3 | 3 |
| Random sampling (no LLM) | 4 | 4 |
| Reg. evolution (no LLM) | 1 | 2 |

## ImageNet16-120: mean IGD over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 6 | 0.056 | 0.172 | 0.031 | 0.106 | 0.056 | 0.172 | 0.031 | 0.106 | 0.032 | 0.130 | 0.019 | 0.082 | 0.032 | 0.130 | 0.019 | 0.082 |
| PEA (no LLM) | 6 | 0.073 | 0.240 | 0.056 | 0.173 | 0.073 | 0.240 | 0.056 | 0.173 | 0.040 | 0.165 | 0.030 | 0.122 | 0.040 | 0.165 | 0.030 | 0.122 |
| Random sampling (no LLM) | 6 | 0.072 | 0.226 | 0.061 | 0.189 | 0.072 | 0.226 | 0.061 | 0.189 | 0.040 | 0.165 | 0.034 | 0.133 | 0.040 | 0.165 | 0.034 | 0.133 |
| Reg. evolution (no LLM) | 6 | 0.046 | 0.149 | 0.026 | 0.097 | 0.046 | 0.149 | 0.026 | 0.097 | 0.025 | 0.115 | 0.014 | 0.077 | 0.025 | 0.115 | 0.014 | 0.077 |

### ImageNet16-120: rank by IGD under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 2 | 2 |
| PEA (no LLM) | 3 | 4 |
| Random sampling (no LLM) | 3 | 4 |
| Reg. evolution (no LLM) | 1 | 1 |

## Significance of the head-to-head results

Default convention (front / r=1.1 / archive / no floor), then the range over all 16 conventions.

| Dataset | Metric | Control | LLM wins (default) | pooled sign p | Wilcoxon p | devices won | device-clustered p | wins over 16 conv. | max device-clustered p over 16 conv. |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 | HV | PEA (no LLM) | 0/25 | 3.0e-08 | 3.0e-08 | 0/6 | 0.016 | 0-2 of 25 | 0.016 |
| cifar10 | HV | Random sampling (no LLM) | 3/25 | 7.8e-05 | 7.5e-07 | 0/6 | 0.016 | 3-4 of 25 | 0.016 |
| cifar10 | HV | NSGA-II (no LLM) | 0/25 | 3.0e-08 | 3.0e-08 | 0/6 | 0.016 | 0-1 of 25 | 0.016 |
| cifar10 | HV | Reg. evolution (no LLM) | 0/25 | 3.0e-08 | 3.0e-08 | 0/6 | 0.016 | 0-1 of 25 | 0.016 |
| cifar10 | HV | PEA with refill (no LLM) | 0/25 | 3.0e-08 | 3.0e-08 | 0/6 | 0.016 | 0-1 of 25 | 0.016 |
| cifar10 | IGD | PEA (no LLM) | 2/25 | 9.7e-06 | 7.5e-07 | 0/6 | 0.016 | 1-3 of 25 | 0.031 |
| cifar10 | IGD | Random sampling (no LLM) | 2/25 | 9.7e-06 | 2.1e-06 | 0/6 | 0.016 | 2-3 of 25 | 0.016 |
| cifar10 | IGD | NSGA-II (no LLM) | 3/25 | 7.8e-05 | 1.3e-06 | 0/6 | 0.016 | 0-3 of 25 | 0.016 |
| cifar10 | IGD | Reg. evolution (no LLM) | 1/25 | 7.7e-07 | 6.0e-08 | 0/6 | 0.016 | 0-2 of 25 | 0.016 |
| cifar10 | IGD | PEA with refill (no LLM) | 1/25 | 7.7e-07 | 8.9e-08 | 0/6 | 0.016 | 0-2 of 25 | 0.016 |
