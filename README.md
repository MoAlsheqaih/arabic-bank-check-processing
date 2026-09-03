# Arabic Bank Check Amount Extraction and Verification

An end-to-end deep learning pipeline that reads the two monetary amounts written on a Saudi bank check and decides whether they agree.

Every check carries the same amount twice: the **courtesy amount** in handwritten digits, and the **legal amount** in handwritten Arabic words. Banks verify a check by confirming the two match. This system does that automatically, from a raw scanned image to a verified / failed verdict.

**82.0% end-to-end verification accuracy** on a held-out set of 600 checks.

Authors: Mohammed Al Sheqaih, Abdulrhman Ammar
Supervisor: Dr. Irfan Ahmad
Natural Language Processing, King Fahd University of Petroleum and Minerals

---

## The pipeline

```
                        Scanned check image
                                 │
                    ┌────────────┴────────────┐
                    │   YOLOv8s detection     │   mean IoU 63.35%
                    └────────────┬────────────┘
                    ┌────────────┴────────────┐
                    ▼                         ▼
            courtesy crop                legal crop
          (handwritten digits)      (handwritten Arabic)
                    │                         │
            ┌───────┴───────┐         ┌───────┴───────┐
            │  CRNN + CTC   │         │  CRNN + CTC   │
            │  95.17% digit │         │  9.02% CER    │
            │   accuracy    │         │               │
            └───────┬───────┘         └───────┬───────┘
                    │                         │
                    │                 Arabic words → integer
                    │                  (rule-based parser)
                    └────────────┬────────────┘
                                 ▼
                        amounts agree?
                                 │
                        VERIFIED / FAILED
                            82.0%
```

---

## Results

| Stage | Metric | Result |
|:---|:---|:---|
| Region detection | Mean IoU | 63.35% |
| Courtesy amount | Digit accuracy | 95.17% |
| Legal amount | Character error rate | 9.02% |
| Legal amount | Exact matches | 430 / 600 |
| End-to-end | Verification accuracy | **82.0%** (492 / 600) |

The published academic result on this dataset is 78.5%. Our comparable figure is 80.8%, and 82.0% is reached with a more permissive fallback than the paper's strict courtesy-versus-legal integer match. So the pipeline is above the published benchmark, but the two protocols are not identical and the margin should not be quoted as a precise number.

---

## Three findings worth the reading

**Character-level tokenization cut the Arabic error rate 5.6×.** The dataset labels Arabic amounts as sub-word fragments — *ريال* arrives as `['ر', 'يا', 'ل']` — producing a 288-token vocabulary over a few thousand training samples. Rebuilding the vocabulary at the character level, 24 Arabic letters plus a space, collapsed the class count by more than an order of magnitude and gave every class far more examples. Character error rate fell from 50.5% to **9.02%**.

**The dictionary post-processing made things dramatically worse, and we kept the measurement.** A minimum-edit-distance correction layer was built to repair broken CTC output against a dictionary of known amount words. It raised character error rate from 9.02% to **80.49%**. The raw CTC output is the reported result. The failure is instructive: CTC output is not merely misspelled, it is often mis-segmented, so nearest-dictionary-word correction confidently replaces partial words with wrong whole ones. A correction step needs to know when to decline.

**The newer architecture lost to the simpler one.** After building the CRNN pipeline we went back and tested the assumption behind it, replacing YOLOv8 with RT-DETR and the CRNN with TrOCR. TrOCR reached **78.82%** digit accuracy against the CRNN's 95.17%. That run is included in this repository rather than hidden, along with what went wrong in it — see `06-rt-detr-trocr-exploration.ipynb` and the caveats below.

---

## Notebooks

Run in order. Each writes artifacts the next one reads.

| Notebook | What it does |
|:---|:---|
| `01-exploratory-data-analysis.ipynb` | Dataset integrity checks, label alignment, vocabulary construction |
| `02-amount-region-detection-yolov8.ipynb` | YOLOv8s fine-tuned to locate both amount regions; IoU evaluation; crop extraction |
| `03-courtesy-amount-recognition-crnn.ipynb` | CNN + BiLSTM + CTC over digit crops |
| `04-legal-amount-recognition-crnn.ipynb` | The same architecture over Arabic text crops, plus the tokenization redesign and the post-processing experiment |
| `05-cross-verification.ipynb` | Rule-based Arabic-to-integer conversion and the final match |
| `06-rt-detr-trocr-exploration.ipynb` | The transformer alternative, and why it did not win |
| `07-pipeline-architecture.ipynb` | Architecture diagrams and the full written description |

All notebooks live in `codebase/`. `utils.py`, alongside them, holds everything shared: label parsing, vocabulary building, IoU and edit-distance metrics, and the Arabic-numeral converter.

---

## Honest caveats

These are stated here rather than left for a reader to find.

**The legal vocabulary is built over train and test together.** 288 tokens covering both splits, so there are zero out-of-vocabulary tokens at inference. This was deliberate — the label files were available and the vocabulary is coverage, not supervision — but it means the recognizer is never tested against a token it has not seen, which flatters the result relative to deployment.

**The transformer exploration has a real defect.** TrOCR digit fine-tuning reports `train_loss: nan` for all 30 epochs while evaluation loss and CER continue to be computed. The run was never diagnosed. Its 78.82% is therefore evidence that *this* training run underperformed, not a fair verdict on TrOCR. The Arabic half of that notebook was interrupted before finishing.

**Detection at high IoU is weak.** Mean IoU of 63.35% is adequate for cropping a region that a recognizer will then read, but precise localization degrades sharply at stricter thresholds. In the RT-DETR comparison, mAP falls from 0.63 at IoU 0.50 to 0.02 at IoU 0.90.

**Training was interrupted in the legal recognition notebook** and resumed from the best checkpoint. The stored output records this.

---

## Dataset

Scanned Saudi bank checks, 1,800 training and 600 test images, grayscale TIFF at roughly 1152 × 422, with bounding-box annotations and both amount transcriptions.

**The dataset is private and licensed. It is not in this repository and cannot be redistributed.** The notebooks expect it at the repository root:

```
CheckImages-Train/       CheckAnnotation-Train/
CheckImages-Test/        CheckAnnotations-Test/
```

One image, `ac00048`, has no bounding-box annotation and is skipped throughout, leaving 1,799 usable training images.

---

## Running this

```bash
pip install -r codebase/requirements.txt
# place the dataset folders at the repository root
cd codebase && jupyter notebook
```

Training was done on an NVIDIA RTX 5060. The notebooks auto-detect CUDA, then MPS, then CPU.

---

## References

- Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (CRNN, 2016)
- Graves et al., *Connectionist Temporal Classification* (2006)
- Jocher et al., *Ultralytics YOLOv8*
- Li et al., *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models* (2021)
