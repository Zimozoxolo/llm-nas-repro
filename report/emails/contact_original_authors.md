# Draft: email to the LLM-NAS authors (for you to check and send yourself)

**To:** Hengyi Zhu, Grace Li Zhang, Shaoyi Huang (corresponding author's address from the arXiv PDF)
**Cc:** Dr Thambo Nyathi
**Subject:** Reproduction of LLM-NAS (arXiv:2510.01472) – results and a few questions

Dear Dr Huang, Dr Zhang and Mr Zhu,

I am a researcher in the Department of Computer Science at the University of Pretoria, working with
Dr Thambo Nyathi. We have reimplemented LLM-NAS from the description in your paper (arXiv:2510.01472v4),
because we could not find a public implementation, and we plan to submit the reproduction for publication.

In brief:

- 11 of the 12 architectures reported in your paper match HW-NAS-Bench; one code does not occur in the
  benchmark.
- On CIFAR-10 / Edge GPU, our GPT-4.1 runs reach a hypervolume of 0.964, against the 0.997 reported.
- The hypervolume values depend strongly on the normalisation and reference point, which the paper does not
  state, so part of this gap may come from scoring rather than from the search.

Before we submit, we would be grateful for your help with any of the following:

1. The hypervolume settings you used (normalisation, reference point, and whether the archived front or the final
   population was scored).
2. The prompts, or the code if you are able to share it.
3. Anything in our setup that differs from yours and would explain the difference.

Our code, settings and logs of every LLM call are available to you, and we will gladly send the draft. We
would like to describe your methods accurately, and we will acknowledge any help you give.

Kind regards,
Siyabonga Mjali
Department of Computer Science, University of Pretoria
