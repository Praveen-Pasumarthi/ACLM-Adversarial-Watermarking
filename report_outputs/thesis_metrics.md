# ACLM Thesis Evaluation Metrics

## 1. System Configuration
- **Model:** ACLM (Adversarial Contrastive Latent Watermarking)
- **Dataset:** DIV2K / Flickr2K (Train), Large_Dataset (Test)
- **Message Size:** 256 bits (Source) -> 448 bits (Encoded)

## 2. Baseline Performance (No Attack)
| Metric | Value |
| :--- | :--- |
| **Raw BER** | 0.076450 |
| **Final BER** | 0.076316 |
| **Recovery Accuracy** | 92.37% |

## 3. Robustness Benchmark
| Attack Strength ($\sigma$) | Raw BER | Final BER |
| :---: | :---: | :---: |
| 0.0 | 0.0764 | **0.0763** |
| 0.01 | 0.0766 | **0.0762** |
| 0.05 | 0.0765 | **0.0759** |
| 0.1 | 0.0764 | **0.0762** |
| 0.2 | 0.0768 | **0.0768** |
