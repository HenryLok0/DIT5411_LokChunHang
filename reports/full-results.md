# Full-dataset test results

All three classifiers were evaluated on the same unaugmented holdout.

- Output classes: 13,065; test coverage: 13,022 classes
- Original training images: 522,127; original test images: 162,550
- Augmented training draws per epoch: 2,613,000
- Training: 3 epochs, batch 256, Adam learning rate 0.001, float32, seed 42
- Dataset ZIP SHA-256: `74607b8d642a4dfd640586efd221cd935bdf16c93c83f2eb9e5092befde60420`

| Model | Parameters | Test loss | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| A | 4,630,953 | 9.4780 | 0.01% | 0.04% |
| B | 1,917,569 | 9.4780 | 0.01% | 0.04% |
| C | 886,361 | 9.4780 | 0.01% | 0.04% |

Highest observed test top-1: **A, B, C**.
Top-1 counts exact first-choice predictions; top-5 counts true labels among the five highest scores.
These are image-weighted scores, and classes without held-out samples are not measured.
