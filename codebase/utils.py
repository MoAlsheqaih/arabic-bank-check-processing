# ICS472 – Natural Language Processing
# Arabic Bank Check Processing
# Mohammed Al Sheqaih · Abdulrhman Ammar

import os
import ast
import re
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGES   = ROOT / "CheckImages-Train"
TEST_IMAGES    = ROOT / "CheckImages-Test"
TRAIN_BBOX     = ROOT / "CheckAnnotation-Train" / "BoundingBox"
TEST_BBOX      = ROOT / "CheckAnnotations-Test" / "BoundingBox"
TRAIN_CA       = ROOT / "CheckAnnotation-Train" / "CourtesyAmounts.txt"
TRAIN_LA       = ROOT / "CheckAnnotation-Train" / "LegalAmounts.txt"
TEST_CA        = ROOT / "CheckAnnotations-Test" / "CourtesyAmounts.txt"
TEST_LA        = ROOT / "CheckAnnotations-Test" / "LegalAmounts.txt"
ARTIFACTS      = ROOT / "codebase" / "artifacts"

ARTIFACTS.mkdir(parents=True, exist_ok=True)


# ─── Label Parsing ────────────────────────────────────────────────────────────

def parse_bbox(txt_path):
    # class 0 = legal amount region, class 1 = courtesy amount region
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
    return rows


def parse_courtesy_amounts(txt_path):
    result = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            fname = parts[0].strip()
            raw = ast.literal_eval(parts[1].strip())
            result[fname] = [_normalise_ca_token(t) for t in raw]
    return result


def _normalise_ca_token(t):
    if t == 10:
        return '<BOS/EOS>'
    if t == 11 or t == '/':
        return '<SEP>'  # both mean the riyal/halalas separator across train and test
    if t == '.':
        return '.'
    return str(t)


def parse_legal_amounts(txt_path):
    result = {}
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # the label files have invisible RTL/LTR control characters — strip them
            line = re.sub(r'[‎‏‪-‮]', '', line)
            idx = line.find('[')
            if idx == -1:
                continue
            fname = line[:idx].strip().split()[0]
            try:
                tokens = ast.literal_eval(line[idx:])
                result[fname] = [str(t) for t in tokens]
            except Exception:
                pass
    return result


# ─── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(token_lists, extra_tokens=None):
    # index 0 is always the CTC blank — don't change this
    tokens = set()
    for lst in token_lists:
        tokens.update(lst)
    if extra_tokens:
        tokens.update(extra_tokens)
    vocab = ['<blank>'] + sorted(tokens)
    token2idx = {t: i for i, t in enumerate(vocab)}
    idx2token = {i: t for t, i in token2idx.items()}
    return vocab, token2idx, idx2token


# ─── Image Utilities ──────────────────────────────────────────────────────────

def load_image(path):
    return Image.open(path).convert('L')


def crop_region(img, cx, cy, w, h):
    # converts normalised YOLO box to pixel coords then crops
    W, H = img.size
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return img.crop((x1, y1, x2, y2))


# ─── IoU & Detection Metrics ─────────────────────────────────────────────────

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [x1, y1, x2, y2]


def detection_metrics(predictions, ground_truths, thresholds=(0.5, 0.75, 0.9)):
    # predictions and ground_truths are lists of dicts: {'courtesy': [x1,y1,x2,y2], 'legal': [...]}
    ious = {'courtesy': [], 'legal': []}
    for pred, gt in zip(predictions, ground_truths):
        for cls in ('courtesy', 'legal'):
            if pred.get(cls) and gt.get(cls):
                ious[cls].append(iou(pred[cls], gt[cls]))
            else:
                ious[cls].append(0.0)

    all_ious = ious['courtesy'] + ious['legal']
    results = {}
    for t in thresholds:
        acc = sum(v >= t for v in all_ious) / len(all_ious) * 100
        results[f'acc@{t}'] = round(acc, 2)
    results['mean_iou'] = round(float(np.mean(all_ious)) * 100, 2)
    return results


# ─── Sequence Metrics ─────────────────────────────────────────────────────────

def edit_distance(ref, hyp):
    # space-efficient DP, O(min(n,m)) memory
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def digit_accuracy(ref_seq, hyp_seq):
    N = len(ref_seq)
    if N == 0:
        return 1.0
    return max(0.0, 1 - edit_distance(ref_seq, hyp_seq) / N)


def cer(ref_tokens, hyp_tokens):
    N = len(ref_tokens)
    if N == 0:
        return 0.0
    return edit_distance(ref_tokens, hyp_tokens) / N


def wer(ref_words, hyp_words):
    N = len(ref_words)
    if N == 0:
        return 0.0
    return edit_distance(ref_words, hyp_words) / N


def courtesy_summary(refs, hyps):
    total_tokens, total_edits = 0, 0
    errors = []
    for ref, hyp in zip(refs, hyps):
        dist = edit_distance(ref, hyp)
        errors.append(dist)
        total_tokens += len(ref)
        total_edits  += dist

    n = len(errors)
    digit_acc = max(0, 1 - total_edits / max(total_tokens, 1)) * 100
    return {
        'digit_accuracy': round(digit_acc, 2),
        'pct_no_error':   round(sum(e == 0 for e in errors) / n * 100, 2),
        'pct_one_error':  round(sum(e == 1 for e in errors) / n * 100, 2),
        'pct_two_plus':   round(sum(e >= 2 for e in errors) / n * 100, 2),
    }


def legal_summary(refs, hyps, token_level=True, word_level=True):
    results = {}
    if token_level:
        total_N, total_edits = 0, 0
        for ref, hyp in zip(refs, hyps):
            total_edits += edit_distance(ref, hyp)
            total_N     += len(ref)
        results['CER'] = round(total_edits / max(total_N, 1) * 100, 2)

    if word_level:
        # join sub-word tokens back into words, then measure word-level distance
        total_N, total_edits = 0, 0
        for ref, hyp in zip(refs, hyps):
            ref_words = ''.join(ref).split()
            hyp_words = ''.join(hyp).split()
            total_edits += edit_distance(ref_words, hyp_words)
            total_N     += len(ref_words)
        results['WER'] = round(total_edits / max(total_N, 1) * 100, 2)

    return results


# ─── Arabic Amount → Digits (Part D) ─────────────────────────────────────────

# listed large-to-small so the greedy loop hits thousands before hundreds before units
_AMOUNT_MAP = [
    ('مئة',        100), ('مائة',       100), ('مائه',       100),
    ('مئتان',      200), ('مائتان',     200), ('مئتين',      200), ('مائتين',    200),
    ('مئتا',       200), ('مائتا',      200),
    ('ثلاثمائة',   300), ('ثلاثمائه',   300), ('ثلثمائة',    300),
    ('أربعمائة',   400), ('اربعمائة',   400), ('أربعمائه',   400), ('اربعمائه',  400),
    ('خمسمائة',    500), ('خمسمائه',    500),
    ('ستمائة',     600), ('ستمائه',     600), ('ستمئة',      600),
    ('سبعمائة',    700), ('سبعمائه',    700),
    ('ثمانمائة',   800), ('ثمانمائه',   800), ('ثمانيمائة',  800),
    ('تسعمائة',    900), ('تسعمائه',    900),
    ('ألف',       1000), ('الف',       1000), ('آلاف',      1000), ('ألفان',    2000),
    ('ألفين',     2000), ('الفين',     2000), ('ألفا',      1000), ('الفا',     1000),
    ('عشرون',       20), ('عشرين',       20), ('عشرة',        10), ('عشر',        10),
    ('أحد عشر',    11), ('إحدى عشرة',  11),
    ('اثنا عشر',   12), ('اثني عشر',   12), ('اثنتا عشرة', 12), ('إثنا عشر',  12),
    ('ثلاثة عشر',  13), ('ثلاث عشرة',  13),
    ('أربعة عشر',  14), ('أربع عشرة',  14),
    ('خمسة عشر',   15), ('خمس عشرة',   15),
    ('ستة عشر',    16), ('ست عشرة',    16),
    ('سبعة عشر',   17), ('سبع عشرة',   17),
    ('ثمانية عشر', 18), ('ثماني عشرة', 18),
    ('تسعة عشر',   19), ('تسع عشرة',   19),
    ('ثلاثون',      30), ('ثلاثين',      30),
    ('أربعون',      40), ('أربعين',      40),
    ('خمسون',       50), ('خمسين',       50),
    ('ستون',        60), ('ستين',        60),
    ('سبعون',       70), ('سبعين',       70),
    ('ثمانون',      80), ('ثمانين',      80),
    ('تسعون',       90), ('تسعين',       90),
    ('واحد',         1), ('احد',          1), ('إحدى',        1), ('احدى',        1),
    ('اثنان',        2), ('اثنين',        2), ('إثنان',       2), ('إثنين',       2),
    ('ثلاثة',        3), ('ثلاث',         3),
    ('أربعة',        4), ('اربعة',        4), ('أربع',        4), ('اربع',        4),
    ('خمسة',         5), ('خمس',          5),
    ('ستة',          6), ('ست',           6),
    ('سبعة',         7), ('سبع',          7),
    ('ثمانية',       8), ('ثماني',        8), ('ثمانٍ',       8),
    ('تسعة',         9), ('تسع',          9),
]

# words that carry no numeric value — skip them during parsing
def _normalize_arabic(s):
    """Normalize Arabic: unify alef variants (آأإ → ا) and taa marbuta (ة → ه)."""
    for c in 'آأإ':
        s = s.replace(c, 'ا')
    s = s.replace('ة', 'ه')
    return s


# Amount map normalized + sorted longest-first for greedy substring scan
_NORM_AMOUNT_MAP = sorted(
    [(_normalize_arabic(a), v) for a, v in _AMOUNT_MAP],
    key=lambda x: -len(x[0])
)

# Noise substrings to strip before scanning (normalized, longer entries first)
_NOISE_SUBSTRINGS = [
    _normalize_arabic(s) for s in [
        'وقدرة', 'وقدره', 'قدره', 'سعودية', 'سعودي', 'فقط',
        'لاغير', 'ريالاً', 'ريالا', 'ريالات', 'رياله', 'ريال',
        'هللات', 'هللة', 'هلله', 'هلل',
    ]
]


def legal_text_to_digits(token_list):
    """Convert a list of Arabic sub-word tokens to an integer amount.

    Joins the sub-word fragments (no spaces), normalizes Arabic script variants,
    strips noise words, then performs a greedy longest-match scan against the
    amount vocabulary.  Returns None if no numeric value can be extracted.
    """
    # 1. Join all sub-word fragments and normalize script variants
    text = _normalize_arabic(''.join(token_list))

    # 2. Strip noise substrings (replace with space so adjacent words don't merge)
    for noise in _NOISE_SUBSTRINGS:
        text = text.replace(noise, ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Greedy longest-match character scan
    total = 0
    current = 0
    i = 0
    n = len(text)

    while i < n:
        if text[i] in ' \t':
            i += 1
            continue

        matched = False
        for arabic_norm, val in _NORM_AMOUNT_MAP:
            L = len(arabic_norm)
            if text[i:i + L] == arabic_norm:
                if val == 1000:
                    current = max(current, 1) * 1000
                    total += current
                    current = 0
                else:
                    current += val
                i += L
                matched = True
                break

        if not matched:
            i += 1  # skip unrecognised character

    total += current
    return total if total > 0 else None
