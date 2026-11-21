# Real-data CPU smoke test

Development subset only. These are not full-dataset assignment scores.

- Output classes: 32; test coverage: 31 classes
- Original training images: 1,256; original test images: 396
- Augmented training draws per epoch: 6,400
- Training: 5 epochs, batch 128, Adam learning rate 0.001, float32, seed 42
- Dataset ZIP SHA-256: `74607b8d642a4dfd640586efd221cd935bdf16c93c83f2eb9e5092befde60420`

| Model | Parameters | Test loss | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| A | 1,281,472 | 0.5886 | 83.59% | 96.46% |
| B | 236,312 | 1.1438 | 66.16% | 89.65% |
| C | 39,216 | 1.8957 | 39.65% | 83.84% |

Highest observed test top-1: **A**.
Top-1 counts exact first-choice predictions; top-5 counts true labels among the five highest scores.
These are image-weighted scores, and classes without held-out samples are not measured.
