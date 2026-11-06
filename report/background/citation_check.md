# Citation check for the rewritten Introduction (2026-11-06)

Method: every new reference was compared with the arXiv record (title, full author list, venue comment) and, where arXiv gave no venue, with the paper's own reference lists in other papers I had read (NAS-Bench-201, NATS-Bench) or the OpenAlex record. Claims were re-checked against the text of the cited paper.

Reference corrections made
- LLMatic: author order was wrong (now Nasir, Earle, Cleghorn, James, Togelius); venue GECCO 2024 added.
- CoLLM-NAS: authors were missing (Li, Lin, Wang); venue added (CVPR 2026 Workshop on NAS, oral, per the arXiv comment).
- BRP-NAS: NeurIPS 2020 added (arXiv comment).
- Li and Talwalkar: UAI 2019 added (arXiv journal reference).
- Elsken et al.: JMLR 20 added.
- LAPT: arXiv id added.
- Verified without change: all other author lists and titles in the new and existing entries; Yu et al. is ICLR 2020 (cited that way in NAS-Bench-201 and NATS-Bench); Hernandez-Lobato et al. and Reagen et al. author lists match the papers and the ISLPED record.
- Not independently confirmed: LiteTransformerSearch as NeurIPS 2022 (arXiv gives no venue); Hernandez-Lobato et al. as a NIPS 2016 BayesOpt workshop paper (taken from the workshop site).

Claim corrections made
- NATS-Bench: the text says random search trails regularised evolution on CIFAR-100 in the topology space; for ImageNet-16-120 it says BOHB converges fastest and states no ordering for random search. The size space says random search is worst on CIFAR-100 and ImageNet-16-120. The Introduction now says exactly that. (The earlier wording overstated the topology result; the same overstatement in Part 3 is corrected.)
- Zero-cost proxies: Abdelfattah et al. report "up to 4x for NAS-Bench-101 compared to current state-of-the-art" and a 5.6x speed-up for aging evolution on NAS-Bench-201; the Introduction now says "by up to four times over the previous best result on NAS-Bench-101".
- NAS-Bench-301: boosted trees were the best surrogate alongside a graph network; the Introduction no longer calls boosted trees "the standard way".
- NB201 and NATS-Bench: DARTS collapses to an all-skip cell (NB201) and a cell with many skip connections (NATS-Bench); wording is now "dominated by skip connections".

Numbers re-checked in the source text: 575M/564M FLOPs and 113/183 ms (MnasNet); 19% fewer MACs and 29% longer latency (NetAdapt); Kendall FLOPs-latency 0.18 (Edge TPU) to 0.83 (FPGA) (HW-NAS-Bench Table 3).
