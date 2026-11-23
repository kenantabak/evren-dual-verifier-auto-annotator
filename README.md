# EVREN Dual-Verifier Auto-Annotator (`evren-dual-verifier-auto-annotator`)

Two-stage consensus auto-annotation engine combining **dataset-specific `YOLO26 / YOLO11` Teachers** with **independent Zero-Shot Vision-Language Detectors (`GroundingDINO`)** to eliminate single-model hallucinations in community dataset curation.

## Architecture

1. **Stage 1 — Domain-Adapted YOLO Teacher**: Trained on the target dataset's ground-truth split (`val P >= 0.45` quality gate).
2. **Stage 2 — Independent Transformer VLM Cross-Verification**: Requires spatial (`IoU >= 0.48`) and semantic agreement from `GroundingDINO-base`.
3. **Stage 3 — Non-Overlapping Missing-Box Discovery**: Proposes only unannotated objects with `IoU < 0.15` against existing ground truth.

## Links

- **EVREN Profile (342 Datasets · 140 Models):** [Kenan Tabak (`@u19979974710e`)](https://evren.ssyz.org.tr/users/u19979974710e)
- **Hugging Face Hub (19 Models · 10 Dataset Suites · 4 Spaces):** [huggingface.co/kenantabak](https://huggingface.co/kenantabak)

---

### 🔗 EVREN Open-Source Ecosystem
- **EVREN Platform:** [evren.ssyz.org.tr/users/u19979974710e](https://evren.ssyz.org.tr/users/u19979974710e) (`342` Grade-A Datasets · `185` Models)
- **Live Browser ONNX Demo Space:** [EVREN-Live-Object-Detection-Demo](https://huggingface.co/spaces/kenantabak/EVREN-Live-Object-Detection-Demo)
- **Hugging Face Portfolio:** [huggingface.co/kenantabak](https://huggingface.co/kenantabak) (`19` Production Models · `10` Datasets · `4` Spaces)
- **GitHub:** [github.com/kenantabak](https://github.com/kenantabak)
- **Dev.to Technical Series:** [dev.to/kenantabak](https://dev.to/kenantabak)
