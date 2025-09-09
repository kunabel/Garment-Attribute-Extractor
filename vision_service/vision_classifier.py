import torch
import torch.nn.functional as F
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import cv2

# Prompt templates
TEMPLATES_CATEGORY = [
    "a catalog photo of a {}.",
    "a product photo of a {}.",
    "an ecommerce listing photo of a {}.",
    "a studio photo of a {}.",
]
TEMPLATES_ATTR_GENERIC = [
    "a catalog photo of {} clothing.",
    "a product photo of {} apparel.",
    "an ecommerce listing photo of {} clothing.",
    "a studio photo of {} clothing.",
]
TEMPLATES_ATTR_WITH_CAT = [
    "a catalog photo of a {} {}.",
    "a product photo of a {} {}.",
    "an ecommerce listing photo of a {} {}.",
    "a studio photo of a {} {}.",
]

CATEGORIES = [
    "t-shirt", "shirt", "blouse", "dress", "skirt", "jeans", "trousers",
    "shorts", "jacket", "coat", "sweater", "hoodie", "cardigan", "leggings",
    "sneakers", "heels", "boots", "sandals", "bag", "hat"
]
COLORS = [
    "black","white","gray","navy","blue","light blue","red","pink","beige",
    "brown","green","olive","yellow","orange","purple","cream","burgundy"
]
MATERIALS = [
    "cotton","denim","wool","cashmere","linen","silk","polyester",
    "leather","suede","down","nylon","viscose","acrylic"
]
CONDITIONS = ["new","like new","gently used","used","vintage"]
STYLES = ["casual","streetwear","sporty","formal","business","elegant","boho","minimal"]
GENDERS = ["womens","mens","unisex","kids"]
SEASONS = ["summer","winter","spring","autumn","all-season"]
PATTERNS = ["solid","striped","checked","plaid","floral","polka dot","graphic","animal print","colorblock"]
SLEEVE_LENGTHS = ["sleeveless","short sleeve","3/4 sleeve","long sleeve"]
NECKLINES = ["crew neck","v-neck","scoop neck","turtleneck","collared","off-shoulder","square neck"]
CLOSURES = ["zipper","buttons","snap","tie","slip-on","lace-up","hook and eye","buckle"]
FITS = ["slim fit","regular fit","relaxed fit","oversized","tapered","loose"]

DEVICE = "cpu"

# -------------------------
# Helper dataclasses
# -------------------------

@dataclass
class AxisResult:
    axis: str
    top1: Tuple[str, float]
    topk: List[Tuple[str, float]]

# -------------------------
# Image conversion
# -------------------------

def to_pil_image(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    """
    Accepts a PIL.Image or an OpenCV/NumPy image.
    - If ndarray: expects BGR (OpenCV) or grayscale. Converts to RGB PIL.Image.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # grayscale
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR -> RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            # BGRA -> RGBA -> drop alpha to RGB
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported ndarray shape for image: {img.shape}")
        return Image.fromarray(rgb)

    raise TypeError(f"Unsupported image type: {type(img)}")

# -------------------------
# Model loading
# -------------------------

def load_model(model_name: str, device: str):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

# -------------------------
# Core scoring utilities
# -------------------------

def _score_prompts(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_pil: Image.Image,
    prompts: List[str],
    device: str,
) -> torch.Tensor:
    """Return probs over prompts for a single image."""
    inputs = processor(text=prompts, images=image_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # shape (1, N)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (N,)
    return probs

def _ensemble_prompts(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_pil: Image.Image,
    labels: List[str],
    templates: List[str],
    device: str,
) -> torch.Tensor:
    """
    Template ensembling: for each label, average probability over templated prompts.
    Returns a tensor of shape (len(labels),).
    """
    prompt_list = []
    spans = []  # (start_idx, end_idx) per label
    for lab in labels:
        start = len(prompt_list)
        for t in templates:
            prompt_list.append(t.format(lab))
        spans.append((start, len(prompt_list)))

    probs_over_prompts = _score_prompts(model, processor, image_pil, prompt_list, device)
    label_probs = []
    for s, e in spans:
        label_probs.append(probs_over_prompts[s:e].mean().unsqueeze(0))
    return torch.cat(label_probs, dim=0)

def classify_axis_generic(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_pil: Image.Image,
    axis_name: str,
    labels: List[str],
    templates: List[str],
    top_k: int = 5,
    device: str = DEVICE,
) -> AxisResult:
    probs = _ensemble_prompts(model, processor, image_pil, labels, templates, device)
    vals, idx = torch.topk(probs, k=min(top_k, len(labels)))
    topk = [(labels[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    return AxisResult(axis=axis_name, top1=topk[0], topk=topk)

def classify_axis_with_category_context(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_pil: Image.Image,
    axis_name: str,
    labels: List[str],
    category: str,
    templates_with_cat: List[str],
    top_k: int = 5,
    device: str = DEVICE,
) -> AxisResult:
    prompts = []
    spans = []
    for lab in labels:
        start = len(prompts)
        for t in templates_with_cat:
            prompts.append(t.format(lab, category))
        spans.append((start, len(prompts)))
    probs_over_prompts = _score_prompts(model, processor, image_pil, prompts, device)
    label_probs = []
    for s, e in spans:
        label_probs.append(probs_over_prompts[s:e].mean().unsqueeze(0))
    probs = torch.cat(label_probs, dim=0)
    vals, idx = torch.topk(probs, k=min(top_k, len(labels)))
    topk = [(labels[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
    return AxisResult(axis=axis_name, top1=topk[0], topk=topk)

# -------------------------
# Public API (now accepts OpenCV image)
# -------------------------

def predict_fashion_attributes_from_cv(
    img_bgr: np.ndarray,
    top_k: int = 5,
    refine_with_category: bool = True,
    model=None,
    processor=None,
    device: str = DEVICE,
) -> Dict[str, AxisResult]:
    if img_bgr is None:
        raise ValueError("img_bgr is None. Make sure read_image(url) returned a valid image.")

    image_pil = to_pil_image(img_bgr)

    if model is None or processor is None:
        raise RuntimeError("Model not loaded. Did you forget to call load_model on startup?")

    results: Dict[str, AxisResult] = {}

    # 1) Category first
    cat_res = classify_axis_generic(
        model, processor, image_pil, "category", CATEGORIES, TEMPLATES_CATEGORY, top_k, device
    )
    results["category"] = cat_res
    category_top = cat_res.top1[0]

    # 2) Remaining axes (optionally using category as context)
    axes = [
        ("color", COLORS),
        ("material", MATERIALS),
        ("condition", CONDITIONS),
        ("style", STYLES),
        ("gender", GENDERS),
        ("season", SEASONS),
        ("pattern", PATTERNS),
        ("sleeve length", SLEEVE_LENGTHS),
        ("neckline", NECKLINES),
        ("closure type", CLOSURES),
        ("fit", FITS),
    ]

    for axis_name, labels in axes:
        if refine_with_category:
            res = classify_axis_with_category_context(
                model, processor, image_pil, axis_name, labels, category_top, TEMPLATES_ATTR_WITH_CAT, top_k, device
            )
        else:
            res = classify_axis_generic(
                model, processor, image_pil, axis_name, labels, TEMPLATES_ATTR_GENERIC, top_k, device
            )
        results[axis_name] = res

    return results
