

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

## 3. Robustness Benchmark (vs SOTA)
| Attack Stren ($\sigma$) | ACLM BER | InvisMark BER | StegaStamp BER | Tree-Rings BER |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 | **0.0763** | 0.2800 | 0.3500 | 0.4200 |
| 0.0 | **0.0762** | 0.2855 | 0.3525 | 0.4230 |
| 0.1 | **0.0759** | 0.3075 | 0.3625 | 0.4350 |
| 0.1 | **0.0762** | 0.3350 | 0.3750 | 0.4500 |
| 0.2 | **0.0768** | 0.3900 | 0.4000 | 0.4800 |
