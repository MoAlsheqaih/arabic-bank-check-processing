# Arabic Bank Check Amount Extraction and Processing

**Course:** ICS472 – Natural Language Processing  
**Instructor:** Dr. Irfan Ahmad  
**Team:** Mohammed Al Sheqaih · Abdulrhman Ammar

---

## Overview

This project implements an end-to-end pipeline for automatically processing Arabic bank checks. Given a scanned check image, the system:

1. **Detects** the courtesy amount and legal amount regions.
2. **Recognizes** the courtesy amount (digit sequence, e.g. `80000`).
3. **Recognizes** the legal amount (Arabic handwritten text, e.g. `ثمانون ألف ريال فقط`).
4. **Verifies** consistency between the two amounts using a rule-based converter.

---

## Dataset

The dataset consists of scanned Arabic bank check images with ground-truth annotations for bounding boxes and recognized amounts. **The dataset is private and licensed — do not share or upload it publicly.**

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 1,800  | Bounding boxes + courtesy + legal amounts |
| Test  | 600    | Bounding boxes + courtesy + legal amounts |

**Directory structure (not committed — clone separately):**
```
CheckImages-Train/          # 1,800 grayscale TIFF images (ac00000.tif …)
CheckImages-Test/           # 600 grayscale TIFF images  (ac03000.tif …)
CheckAnnotation-Train/
├── BoundingBox/            # YOLO-format labels (class cx cy w h), class 0=legal, 1=courtesy
├── CourtesyAmounts.txt     # digit sequences per image
└── LegalAmounts.txt        # Arabic sub-word token sequences per image
CheckAnnotations-Test/      # same structure as train
```

---

## Project Pipeline

```
Check Image
    │
    ▼
┌─────────────────────────────┐
│  Part A – Object Detection  │  YOLOv8s  →  courtesy crop + legal crop
└─────────────────────────────┘
         │                │
         ▼                ▼
┌───────────────┐  ┌──────────────────┐
│    Part B     │  │     Part C       │
│  Courtesy     │  │   Legal Amount   │
│  Recognition  │  │   Recognition    │
│ CNN+BiLSTM+   │  │  CNN+BiLSTM+CTC  │
│    CTC        │  │  (Arabic sub-    │
│ → digit seq   │  │   word tokens)   │
└───────────────┘  └──────────────────┘
         │                │
         └────────┬───────┘
                  ▼
    ┌─────────────────────────┐
    │  Part D – Verification  │  Rule-based Arabic→digits + match
    └─────────────────────────┘
                  │
                  ▼
         Verified / Failed
```

---

## Codebase Structure

```
codebase/
├── utils.py                       # Shared: paths, label parsing, metrics, vocab, transforms
├── 00_EDA.ipynb                   # Exploratory data analysis & dataset sanity checks
├── 01_Part_A_Detection.ipynb      # YOLOv8 training & evaluation (IoU @ 0.5 / 0.75 / 0.9)
├── 02_Part_B_Courtesy.ipynb       # Courtesy digit recognition (CNN+BiLSTM+CTC)
├── 03_Part_C_Legal.ipynb          # Legal Arabic recognition (CNN+BiLSTM+CTC)
└── 04_Part_D_Verification.ipynb   # End-to-end pipeline + final verification
```

---

## Evaluation Metrics

| Part | Metric |
|------|--------|
| A – Detection | Accuracy @ IoU ≥ {0.50, 0.75, 0.90} · Mean IoU |
| B – Courtesy  | Digit-level accuracy `(1 − (I+D+S)/N) × 100` · % amounts with 0 / 1 / ≥2 errors |
| C – Legal     | CER and WER `((I+D+S)/N) × 100` |
| D – Verification | % of checks where legal amount (converted) matches courtesy amount |

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd arabic-bank-check-processing

# Install dependencies
pip install -r codebase/requirements.txt

# Place the dataset folders in the repo root (see Dataset section above)
```

**Hardware:** Training was performed on an NVIDIA RTX 5060 (16 GB VRAM). Inference runs on CPU or Apple Silicon (MPS).

---

## References

- Redmon & Farhadi, *YOLOv3* (2018) — basis for Ultralytics YOLOv8
- Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (CRNN, 2016)
- Graves et al., *Connectionist Temporal Classification* (CTC, 2006)
- Course reference: Arabic Check Processing (Dr. Irfan Ahmad, ICS472)
