# Performance Record

## Unified Conformer Result

* Feature info: using fbank feature, with cmvn, speed perturb.
* Training info: lr 0.002, batch size 16, 7 gpus, acc_grad 1, 120 epochs for Aishell-1, 40 epochs per tuning, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 
* Model link: 

| attention rescoring w/o lm | aishell1_test | aishell2_test   | selected   |
|                            | full | 16   | full  | 16    | duration | cer   | avg_token_len |
|----------------------------|------|------|-------|-------|----------|-------|---------------|
| baseline(separately)       | 5.05 | 5.45 | 6.08  | 6.46  | -        | -     | -             |
| conformer                  | 5.19 | 5.73 | 13.21 | 14.02 | -        | -     | -             |
|  +cutoff_1.0               | 5.14 | 5.64 | 12.88 | 13.65 | 6.3h     | 13.76 | 1.98          |
|   +cutoff_0.5              | 4.59 | 4.89 | 10.81 | 11.32 | 460.2h   | 1.93  | 10.88         |
|    +cutoff_0.0             | 4.40 | 4.67 | 10.18 | 10.65 | 712.0h   | 3.52  | 10.82         |
|     +cutoff_-1.0           | 4.25 | 4.44 | 9.92  | 10.32 | 866.9h   | 5.18  | 10.84         |
|      +cutoff_-∞            | 4.19 | 4.33 | 9.70  | 10.13 | 1000.75h | 7.36  | 10.90         |
|      +cutoff_-∞            | 4.18 | 4.29 | 9.57  | 9.90  | 1000.75h | 7.26  | 10.90         |
