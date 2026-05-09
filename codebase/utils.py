"""
Shared utilities for Arabic Bank Check Processing project.
ICS472 – Natural Language Processing
Team: Mohammed Al Sheqaih · Abdulrhman Ammar
"""

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
    """
    Returns list of (class_id, cx, cy, w, h) floats from a YOLO label file.
    class 0 = legal amount, class 1 = courtesy amount.
    """
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
    """
    Returns dict  filename -> list of tokens  from a CourtesyAmounts.txt.
    Tokens are normalised: int digits 0-9, '.' for decimal, '<SEP>' for
    the slash/11 separator, '<BOS>'/<EOS>' for boundary markers (token 10).
    """
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
    """Map raw token to canonical string."""
    if t == 10:
        return '<BOS/EOS>'
    if t == 11 or t == '/':
        return '<SEP>'      # fractional separator (riyal / halalas)
    if t == '.':
        return '.'
    return str(t)           # digits 0-9 as strings


def parse_legal_amounts(txt_path):
    """
    Returns dict  filename -> list of sub-word token strings.
    Strips Unicode RTL/LTR markers before parsing.
    """
    result = {}
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # strip invisible RTL/LTR control chars
            line = re.sub(r'[‎‏‪-‮]', '', line)
            idx = line.find('[')
            if idx == -1:
                continue
            fname_part = line[:idx].strip()
            fname = fname_part.split()[0]
            try:
                tokens = ast.literal_eval(line[idx:])
                result[fname] = [str(t) for t in tokens]
            except Exception:
                pass
    return result


# ─── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(token_lists, extra_tokens=None):
    """
    Builds char/token -> index mapping.
    Index 0 is reserved for CTC blank.
    """
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
    """Load TIFF (or any format) as a grayscale PIL image."""
    return Image.open(path).convert('L')


def crop_region(img, cx, cy, w, h):
    """
    Crop normalised YOLO bbox (cx,cy,w,h) from a PIL image.
    Returns a PIL image of the cropped region.
    """
    W, H = img.size
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return img.crop((x1, y1, x2, y2))


# ─── IoU ──────────────────────────────────────────────────────────────────────

def iou(boxA, boxB):
    """
    Compute IoU between two boxes in [x1,y1,x2,y2] pixel format.
    """
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
    """Convert normalised YOLO box to pixel [x1,y1,x2,y2]."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [x1, y1, x2, y2]


def detection_metrics(predictions, ground_truths, thresholds=(0.5, 0.75, 0.9)):
    """
    Compute accuracy at each IoU threshold and mean IoU for detection.

    predictions / ground_truths: list of dicts with keys
        {'courtesy': [x1,y1,x2,y2], 'legal': [x1,y1,x2,y2]}
    """
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
    """Standard Levenshtein edit distance between two lists/strings."""
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
    """
    Digit-level accuracy for courtesy amounts.
    Returns accuracy in [0,1].
    """
    N = len(ref_seq)
    if N == 0:
        return 1.0
    dist = edit_distance(ref_seq, hyp_seq)
    return max(0.0, 1 - dist / N)


def cer(ref_tokens, hyp_tokens):
    """Character Error Rate (list of chars or sub-word tokens)."""
    N = len(ref_tokens)
    if N == 0:
        return 0.0
    return edit_distance(ref_tokens, hyp_tokens) / N


def wer(ref_words, hyp_words):
    """Word Error Rate (list of words)."""
    N = len(ref_words)
    if N == 0:
        return 0.0
    return edit_distance(ref_words, hyp_words) / N


def courtesy_summary(refs, hyps):
    """
    Aggregate courtesy metrics over a dataset.
    refs / hyps: list of token-lists (strings, no BOS/EOS markers).
    Returns dict with overall digit accuracy, % 0-error, % 1-error, % ≥2-error.
    """
    total_tokens, total_edits = 0, 0
    errors = []
    for ref, hyp in zip(refs, hyps):
        dist = edit_distance(ref, hyp)
        errors.append(dist)
        total_tokens += len(ref)
        total_edits  += dist

    n = len(errors)
    digit_acc = max(0, 1 - total_edits / max(total_tokens, 1)) * 100
    pct_0 = sum(e == 0 for e in errors) / n * 100
    pct_1 = sum(e == 1 for e in errors) / n * 100
    pct_2p = sum(e >= 2 for e in errors) / n * 100
    return {
        'digit_accuracy': round(digit_acc, 2),
        'pct_no_error':   round(pct_0, 2),
        'pct_one_error':  round(pct_1, 2),
        'pct_two_plus':   round(pct_2p, 2),
    }


def legal_summary(refs, hyps, token_level=True, word_level=True):
    """
    CER and WER for legal amounts.
    refs/hyps: list of sub-word token lists.
    """
    results = {}
    if token_level:
        total_N, total_edits = 0, 0
        for ref, hyp in zip(refs, hyps):
            total_edits += edit_distance(ref, hyp)
            total_N     += len(ref)
        results['CER'] = round(total_edits / max(total_N, 1) * 100, 2)

    if word_level:
        # join sub-words to words then split on spaces for word-level
        total_N, total_edits = 0, 0
        for ref, hyp in zip(refs, hyps):
            ref_words = ''.join(ref).split()
            hyp_words = ''.join(hyp).split()
            total_edits += edit_distance(ref_words, hyp_words)
            total_N     += len(ref_words)
        results['WER'] = round(total_edits / max(total_N, 1) * 100, 2)

    return results


# ─── Arabic Amount → Digits (Part D) ─────────────────────────────────────────

# Ordered large-to-small so greedy matching works correctly.
_AMOUNT_MAP = [
    # sub-word fragments that together form a word — handled by joining first
    # Full-word mappings (post-join)
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

_IGNORE_WORDS = {
    'ريال', 'ريالا', 'ريالاً', 'رياله', 'ريالات',
    'هللة', 'هلله', 'هللات', 'هلل',
    'فقط', 'لاغير', 'لا', 'غير', 'وقدره', 'وقدرة', 'قدره',
    'سعودي', 'سعودية', 'و', 'ا', 'ة', 'ه', 'ن', 'ي',
}


def legal_text_to_digits(token_list):
    """
    Convert a list of Arabic sub-word tokens to an integer amount.
    Returns int or None if conversion fails.
    Strategy: join tokens → clean → greedy match amount map.
    """
    text = ''.join(token_list)
    # strip common noise words
    for noise in ('فقط', 'لاغير', 'وقدره', 'وقدرة', 'سعودي', 'سعودية'):
        text = text.replace(noise, ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    total = 0
    current = 0
    words = text.split()

    i = 0
    while i < len(words):
        word = words[i]

        # skip noise single tokens
        if word in _IGNORE_WORDS:
            i += 1
            continue

        matched = False
        # try two-word phrases first
        if i + 1 < len(words):
            two = word + ' ' + words[i + 1]
            for arabic, val in _AMOUNT_MAP:
                if two == arabic:
                    if val == 1000:
                        current = max(current, 1) * 1000
                    elif val >= 100:
                        current += val
                    else:
                        current += val
                    i += 2
                    matched = True
                    break

        if not matched:
            for arabic, val in _AMOUNT_MAP:
                if word == arabic:
                    if val == 1000:
                        current = max(current, 1) * 1000
                        total += current
                        current = 0
                    elif val >= 100:
                        current += val
                    else:
                        current += val
                    i += 1
                    matched = True
                    break

        if not matched:
            i += 1  # unknown token, skip

    total += current
    return total if total > 0 else None
