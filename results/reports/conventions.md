# Robustness of the results to the hypervolume convention

Every archived run re-scored under 16 conventions (normalisation x reference point x scored set x accuracy floor at 85% of the best accuracy). Device-level values are means over seeds; the table gives the mean over devices that the method was run on. LLM rows are single runs per device. No search was re-run.


## cifar10: mean HV over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4-5 | 6 | 0.888 | 0.821 | 0.895 | 0.832 | 0.872 | 0.786 | 0.879 | 0.799 | 0.977 | 0.933 | 0.977 | 0.937 | 0.975 | 0.921 | 0.976 | 0.926 |
| claude-opus-5-5 | 1 | 0.980 | 0.961 | 0.981 | 0.976 | 0.993 | 0.956 | 0.995 | 0.971 | 0.990 | 0.989 | 0.990 | 0.993 | 0.997 | 0.987 | 0.997 | 0.992 |
| deepseek-coder-v2-16b-8k | 6 | 0.897 | 0.846 | 0.904 | 0.853 | 0.879 | 0.817 | 0.887 | 0.826 | 0.976 | 0.935 | 0.977 | 0.940 | 0.974 | 0.925 | 0.975 | 0.930 |
| gpt-4.1 | 6 | 0.928 | 0.860 | 0.940 | 0.879 | 0.919 | 0.833 | 0.933 | 0.854 | 0.989 | 0.953 | 0.990 | 0.959 | 0.988 | 0.943 | 0.990 | 0.951 |
| qwen2.5-7b-8k | 6 | 0.891 | 0.807 | 0.898 | 0.820 | 0.878 | 0.771 | 0.885 | 0.786 | 0.961 | 0.918 | 0.963 | 0.925 | 0.959 | 0.905 | 0.960 | 0.913 |
| NSGA-II (no LLM) | 6 | 0.969 | 0.943 | 0.984 | 0.968 | 0.966 | 0.937 | 0.982 | 0.963 | 0.994 | 0.972 | 0.997 | 0.986 | 0.993 | 0.968 | 0.996 | 0.984 |
| PEA (no LLM) | 6 | 0.952 | 0.911 | 0.960 | 0.929 | 0.945 | 0.897 | 0.953 | 0.916 | 0.991 | 0.961 | 0.993 | 0.972 | 0.990 | 0.956 | 0.992 | 0.967 |
| Random sampling (no LLM) | 6 | 0.950 | 0.907 | 0.956 | 0.922 | 0.941 | 0.889 | 0.948 | 0.907 | 0.992 | 0.963 | 0.993 | 0.971 | 0.991 | 0.957 | 0.992 | 0.965 |

### cifar10: LLM runs that beat each control (same device and convention)

| Convention | beats PEA (no LLM) | beats Random sampling (no LLM) | beats NSGA-II (no LLM) |
|---|---|---|---|
| front / r=1.1 / archive / no floor | 0/25 | 1/25 | 0/25 |
| front / r=1.1 / archive / floor 85% | 1/25 | 3/25 | 1/25 |
| front / r=1.1 / all / no floor | 1/25 | 1/25 | 0/25 |
| front / r=1.1 / all / floor 85% | 2/25 | 3/25 | 1/25 |
| front / r=1.0 / archive / no floor | 1/25 | 2/25 | 0/25 |
| front / r=1.0 / archive / floor 85% | 1/25 | 3/25 | 1/25 |
| front / r=1.0 / all / no floor | 2/25 | 2/25 | 0/25 |
| front / r=1.0 / all / floor 85% | 2/25 | 3/25 | 1/25 |
| space / r=1.1 / archive / no floor | 2/25 | 2/25 | 0/25 |
| space / r=1.1 / archive / floor 85% | 2/25 | 3/25 | 1/25 |
| space / r=1.1 / all / no floor | 2/25 | 1/25 | 0/25 |
| space / r=1.1 / all / floor 85% | 2/25 | 3/25 | 1/25 |
| space / r=1.0 / archive / no floor | 3/25 | 3/25 | 1/25 |
| space / r=1.0 / archive / floor 85% | 2/25 | 3/25 | 1/25 |
| space / r=1.0 / all / no floor | 2/25 | 2/25 | 1/25 |
| space / r=1.0 / all / floor 85% | 1/25 | 3/25 | 1/25 |

### cifar10: rank of each method under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 1 | 2 |
| PEA (no LLM) | 3 | 4 |
| Random sampling (no LLM) | 2 | 4 |
| claude-haiku-4-5 | 6 | 8 |
| claude-opus-5-5 | 1 | 4 |
| deepseek-coder-v2-16b-8k | 6 | 7 |
| gpt-4.1 | 5 | 5 |
| qwen2.5-7b-8k | 7 | 8 |

## cifar100: mean HV over devices

| Method | Devices | front / r=1.1 / archive / no floor | front / r=1.1 / archive / floor 85% | front / r=1.1 / all / no floor | front / r=1.1 / all / floor 85% | front / r=1.0 / archive / no floor | front / r=1.0 / archive / floor 85% | front / r=1.0 / all / no floor | front / r=1.0 / all / floor 85% | space / r=1.1 / archive / no floor | space / r=1.1 / archive / floor 85% | space / r=1.1 / all / no floor | space / r=1.1 / all / floor 85% | space / r=1.0 / archive / no floor | space / r=1.0 / archive / floor 85% | space / r=1.0 / all / no floor | space / r=1.0 / all / floor 85% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NSGA-II (no LLM) | 6 | 0.959 | 0.900 | 0.978 | 0.940 | 0.954 | 0.886 | 0.976 | 0.932 | 0.987 | 0.917 | 0.992 | 0.949 | 0.986 | 0.906 | 0.991 | 0.942 |
| PEA (no LLM) | 6 | 0.946 | 0.849 | 0.953 | 0.871 | 0.939 | 0.823 | 0.945 | 0.846 | 0.982 | 0.888 | 0.986 | 0.914 | 0.980 | 0.870 | 0.984 | 0.898 |
| Random sampling (no LLM) | 6 | 0.931 | 0.830 | 0.940 | 0.849 | 0.919 | 0.804 | 0.928 | 0.824 | 0.981 | 0.877 | 0.984 | 0.896 | 0.978 | 0.858 | 0.981 | 0.879 |

### cifar100: LLM runs that beat each control (same device and convention)

| Convention | beats PEA (no LLM) | beats Random sampling (no LLM) | beats NSGA-II (no LLM) |
|---|---|---|---|
| front / r=1.1 / archive / no floor | 0/0 | 0/0 | 0/0 |
| front / r=1.1 / archive / floor 85% | 0/0 | 0/0 | 0/0 |
| front / r=1.1 / all / no floor | 0/0 | 0/0 | 0/0 |
| front / r=1.1 / all / floor 85% | 0/0 | 0/0 | 0/0 |
| front / r=1.0 / archive / no floor | 0/0 | 0/0 | 0/0 |
| front / r=1.0 / archive / floor 85% | 0/0 | 0/0 | 0/0 |
| front / r=1.0 / all / no floor | 0/0 | 0/0 | 0/0 |
| front / r=1.0 / all / floor 85% | 0/0 | 0/0 | 0/0 |
| space / r=1.1 / archive / no floor | 0/0 | 0/0 | 0/0 |
| space / r=1.1 / archive / floor 85% | 0/0 | 0/0 | 0/0 |
| space / r=1.1 / all / no floor | 0/0 | 0/0 | 0/0 |
| space / r=1.1 / all / floor 85% | 0/0 | 0/0 | 0/0 |
| space / r=1.0 / archive / no floor | 0/0 | 0/0 | 0/0 |
| space / r=1.0 / archive / floor 85% | 0/0 | 0/0 | 0/0 |
| space / r=1.0 / all / no floor | 0/0 | 0/0 | 0/0 |
| space / r=1.0 / all / floor 85% | 0/0 | 0/0 | 0/0 |

### cifar100: rank of each method under each convention (1 = best)

| Method | best rank | worst rank |
|---|---|---|
| NSGA-II (no LLM) | 1 | 1 |
| PEA (no LLM) | 2 | 2 |
| Random sampling (no LLM) | 3 | 3 |
