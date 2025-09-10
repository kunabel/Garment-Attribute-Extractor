# heuristic_service/brand_detector.py
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz
import os

# Brand lexicon (extend as needed or load from file via env)
DEFAULT_BRANDS = [
    "Nike", "Adidas", "Puma", "Reebok", "New Balance", "Under Armour",
    "Zara", "H&M", "Uniqlo", "Levi's", "Gucci", "Prada", "Louis Vuitton",
    "Chanel", "Balenciaga", "Burberry", "Hermes", "Ralph Lauren",
    "Tommy Hilfiger", "Calvin Klein", "The North Face", "Patagonia",
    "Columbia", "Carhartt", "Diesel", "Stone Island", "Moncler",
    "Armani", "Versace", "Off-White", "Guess", "Mango", "COS", "Massimo Dutti", "Moss",
    "Joseph", "Erdem", "Maje", "Monsoon", "MB", "Versace", "J. Lindeberg",
    "Claudie Pierlot"
]

_STOPWORDS = {
    "made","in","size","small","medium","large","xl","xxl","xxxl",
    "wash","cold","warm","dry","clean","iron","bleach",
    "cotton","polyester","nylon","wool","silk","leather","lining",
    "shell","fill","down","rn","ca","no","number",
    "china","italy","usa","uk","eur","japan","india","turkey",
    "imported","exclusive","of","trims","care","instructions","see",
    "brand","registered","trademark","since","est","co","inc","ltd","gmbh","sa","spa"
}

def _load_brand_lexicon() -> List[str]:
    path = os.getenv("BRAND_LEXICON_PATH")
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            brands = [ln.strip() for ln in f if ln.strip()]
        return brands
    return DEFAULT_BRANDS

_PUNCT_TO_SPACE = str.maketrans({c: " " for c in r"\/|_.,:;!?'`~^*()[]{}<>=+&@"})

def _normalize_text(s: str) -> str:
    return " ".join(s.translate(_PUNCT_TO_SPACE).split())

def _is_candidate_token(tok: str) -> bool:
    t = tok.strip()
    if len(t) < 2:
        return False
    if t.lower() in _STOPWORDS:
        return False
    if t.upper() in {"XS","S","M","L","XL","XXL","XXXL"}:
        return False
    return any(c.isalpha() for c in t)

def _n_grams(tokens: List[str], n_min=1, n_max=3) -> List[str]:
    spans: List[str] = []
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            spans.append(" ".join(tokens[i:i+n]))
    return spans

# ------- PaddleOCR singleton -------
_OCR: PaddleOCR | None = None

def init_ocr(lang: str = "en"):
    """
    Initialize PaddleOCR once on CPU.
    lang='en' uses English recognition; add 'en' only to keep model small.
    """
    global _OCR
    if _OCR is None:
        # use_angle_cls helps rotated/tilted text; det/rec models auto-fetched and cached
        _OCR = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)

def detect_brand_from_label(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Detect brand from a label image using PaddleOCR (CPU) + fuzzy matching.
    Returns dict with best brand, confidence, raw_text, and debug fields.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"brand": None, "confidence": 0.0, "raw_text": "", "candidates": [], "match_debug": []}

    if _OCR is None:
        raise RuntimeError("OCR not initialized. Call init_ocr() at app startup.")

    # Slight upscale helps OCR on small labels
    h, w = img_bgr.shape[:2]
    if max(h, w) < 600:
        img_bgr = cv2.resize(img_bgr, (int(w*1.3), int(h*1.3)), interpolation=cv2.INTER_CUBIC)

    # PaddleOCR accepts either file path or numpy image (BGR ok)
    ocr_result = _OCR.ocr(img_bgr, cls=True)  # returns list per image

    lines: List[Tuple[str, float]] = []
    raw_text_parts: List[str] = []

    # ocr_result is [[ [box, (text, conf)], ... ]] for a single image
    if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
        for entry in ocr_result[0]:
            if len(entry) >= 2:
                text, conf = entry[1][0], float(entry[1][1])
                text = text.strip()
                if text:
                    lines.append((text, conf))
                    raw_text_parts.append(text)

    raw_text = " | ".join(raw_text_parts)

    if not lines:
        return {"brand": None, "confidence": 0.0, "raw_text": raw_text, "candidates": [], "match_debug": []}

    # Build candidates: tokens + n-grams up to 3
    candidates: List[str] = []
    for txt, _ in lines:
        tokens = [_ for _ in _normalize_text(txt).split() if _is_candidate_token(_)]
        candidates.extend(tokens)
        candidates.extend(_n_grams(tokens, 2, 3))

    # Deduplicate (case-insensitive)
    seen = set()
    cand_strings: List[str] = []
    for c in candidates:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            cand_strings.append(c)

    if not cand_strings:
        return {"brand": None, "confidence": 0.0, "raw_text": raw_text, "candidates": lines, "match_debug": []}

    brands = _load_brand_lexicon()
    matches = []
    for c in cand_strings:
        best = process.extractOne(c, brands, scorer=fuzz.token_set_ratio)
        if best:
            brand, score, _ = best
            matches.append((c, brand, score))

    if not matches:
        return {"brand": None, "confidence": 0.0, "raw_text": raw_text, "candidates": lines, "match_debug": []}

    matches.sort(key=lambda x: x[2], reverse=True)
    best_cand, best_brand, best_score = matches[0]

    # Mix fuzzy score with OCR conf of the line that contained the candidate (if any)
    ocr_conf_for_best = 0.7
    for txt, conf in lines:
        if best_cand.lower() in _normalize_text(txt).lower():
            ocr_conf_for_best = max(ocr_conf_for_best, conf)

    confidence = 0.6 * (best_score / 100.0) + 0.4 * max(0.0, min(1.0, ocr_conf_for_best))

    return {
        "brand": best_brand,
        "confidence": round(float(confidence), 3),
        "raw_text": raw_text,
        "candidates": [(t, float(c)) for t, c in lines][:50],
        "match_debug": matches[:10]
    }
