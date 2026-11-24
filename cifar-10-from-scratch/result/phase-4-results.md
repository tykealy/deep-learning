# Phase 4 — CIFAR-10 training experiments

Seed: 42. Device: mps. PyTorch: 2.15.0.
One run per setting. Accuracy is shown as a percentage; CSV accuracy is a fraction.
Epoch time includes training and validation. At a fixed epoch count, larger batches do fewer optimizer steps.
RSS and MPS are the largest samples after forward and after the optimizer step on every tenth training batch, starting with the first. These are not true peaks. CUDA reports peak allocated tensor memory over the epoch. RSS is for the whole notebook process.

## 1. Batch size

| Batch | Steps/epoch | Train loss (1 → 10) | Val loss (1 → 10) | Train acc (10) | Val acc (10) | Mean s/epoch | Sampled RSS MiB | Sampled MPS MiB | CUDA peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 1407 | 1.8464 → 0.7915 | 1.6071 → 0.8327 | 72.49% | 71.24% | 12.57 | 801.2 | 14.1 | — |
| 64 | 704 | 1.8999 → 0.9484 | 1.6681 → 0.9159 | 66.72% | 68.18% | 8.58 | 467.7 | 27.0 | — |
| 128 | 352 | 1.9828 → 1.0692 | 1.7942 → 1.0168 | 62.28% | 64.08% | 7.38 | 498.3 | 53.8 | — |
| 256 | 176 | 2.0512 → 1.2631 | 1.8607 → 1.2167 | 54.49% | 56.00% | 7.19 | 560.9 | 104.3 | — |

## 2. Learning rate

| Learning rate | Val acc (1) | Val acc (5) | Val acc (10) | Best val acc | Val loss (1 → 10) | Val loss increases |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 9.94% | 10.60% | 10.34% | 10.60% (epoch 5) | 2.3031 → 2.3039 | 5/9 |
| 0.001 | 40.56% | 65.06% | 71.24% | 71.50% (epoch 9) | 1.6071 → 0.8327 | 1/9 |
| 0.0001 | 28.98% | 39.24% | 41.48% | 41.48% (epoch 10) | 1.9222 → 1.6044 | 2/9 |

## 3. Optimizer

| Optimizer | Val acc (1) | Val acc (5) | Val acc (10) | Best val acc | Val loss (1 → 10) | Val loss increases |
| --- | --- | --- | --- | --- | --- | --- |
| Adam | 40.56% | 65.06% | 71.24% | 71.50% (epoch 9) | 1.6071 → 0.8327 | 1/9 |
| SGD (momentum 0.9) | 15.36% | 28.34% | 35.12% | 35.86% (epoch 9) | 2.3004 → 1.7415 | 1/9 |

Both optimizers use LR 0.001 here. Adam rescales updates using running gradient statistics, while momentum SGD accumulates gradients without that adaptive rescaling. The same LR can give very different effective updates, so tune LR (and momentum for SGD) separately before comparing their best performance.

## 4. Epoch budget

| Epochs | Train acc | Val acc | Change in val acc | Val loss | Best val acc so far | Total seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 63.40% | 65.06% | — | 0.9862 | 65.06% (epoch 5) | 64.3 |
| 10 | 72.49% | 71.24% | +6.18 pp | 0.8327 | 71.50% (epoch 9) | 125.7 |
| 20 | 79.62% | 77.12% | +5.88 pp | 0.7105 | 78.32% (epoch 19) | 239.9 |

Small validation gains alongside falling training loss can indicate a plateau. Falling validation accuracy or rising validation loss while training improves can indicate overfitting. Check the full per-epoch history in `phase-4-metrics.csv`.
