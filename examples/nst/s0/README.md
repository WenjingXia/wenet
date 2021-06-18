# Performance Record

## Unified Conformer Result

* Feature info: using fbank feature, with cmvn, speed perturb.
* Training info: lr 0.002, batch size 16, 7 gpus, acc_grad 1, 120 epochs for Aishell-1, 60 epochs per tuning, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 
* Model link: 

| attention rescoring w/o lm | aishell1_test | aishell2_test   | selected   |
|                            | full | 16   | full  | 16    | duration | cer   | avg_token_len |
|----------------------------|------|------|-------|-------|----------|-------|---------------|
| baseline(separately)       | 5.05 | 5.45 | 6.08  | 6.46  | -        | -     | -             |
| conformer                  | 5.19 | 5.73 | 13.21 | 14.02 | -        | -     | -             |
|  +cutoff_1.0               | 5.19 | 5.73 | 13.00 | 13.86 | 6.29h    | 13.76 | 1.98          |
|   +cutoff_0.5              | 4.85 | 5.23 | 11.38 | 11.91 | 450.38h  | 1.88  | 10.95         |
|    +cutoff_0.0             | 4.67 | 4.94 | 10.73 | 11.18 | 697.63h  | 3.48  | 10.84         |
|     +cutoff_-1.0           | 4.55 | 4.74 | 10.46 | 10.93 | 858.95h  | 5.26  | 10.84         |
|      +cutoff_-∞            | 4.46 | 4.64 | 10.21 | 10.68 | 1000.75h | 7.59  | 10.90         |
