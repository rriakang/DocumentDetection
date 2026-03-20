import re
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

# OCR 엔진 싱글턴
_easyocr_reader = None
_paddle_reader = None
_paddle_cache: Dict[int, List[Tuple]] = {}


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _easyocr_reader


def _get_paddle():
    global _paddle_reader
    if _paddle_reader is None:
        import logging
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        logging.getLogger("paddleocr").setLevel(logging.WARNING)
        from paddleocr import PaddleOCR
        _paddle_reader = PaddleOCR(lang="korean", use_textline_orientation=True)
    return _paddle_reader


def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """OCR 정확도를 위한 전처리: 업스케일 + 대비 강화."""
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]

    min_h = 80
    if h < min_h:
        scale = min_h / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


def _find_label_boundary(image: Image.Image) -> float:
    """행 이미지에서 첫 번째 세로 구분선 x좌표를 찾아 라벨 영역 경계를 결정한다."""
    img = np.array(image.convert("L"))
    h, w = img.shape

    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 10)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    col_sums = np.sum(v_lines, axis=0) / 255
    threshold = h * 0.4

    x_start = int(w * 0.10)
    x_end = int(w * 0.60)

    for x in range(x_start, x_end):
        if col_sums[x] > threshold:
            return x / w

    return 0.30


# ──────────────────────────────────────────────
# PaddleOCR 전체 이미지 OCR (보정용)
# ──────────────────────────────────────────────

def _paddle_ocr_full_image(image: Image.Image) -> List[Tuple]:
    """PaddleOCR로 전체 이미지 OCR → [(cx, cy, text, conf), ...]"""
    img_id = id(image)
    if img_id in _paddle_cache:
        return _paddle_cache[img_id]

    reader = _get_paddle()
    img_np = np.array(image.convert("RGB"))
    result = reader.predict(img_np)

    parsed = []
    if result:
        for page in result:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("rec_polys", page.get("dt_polys", []))
            for i, text in enumerate(texts):
                conf = scores[i] if i < len(scores) else 0.0
                if conf < 0.3 or not text.strip():
                    continue
                if i < len(polys) and len(polys[i]) >= 4:
                    xs = [p[0] for p in polys[i]]
                    ys = [p[1] for p in polys[i]]
                    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                else:
                    cx, cy = 0, 0
                parsed.append((cx, cy, text.strip(), conf))

    _paddle_cache[img_id] = parsed
    return parsed


def _paddle_get_label_for_row(full_image: Image.Image, row_bbox: list) -> str:
    """PaddleOCR 전체 이미지 결과에서 해당 행의 라벨 영역 텍스트를 가져온다."""
    all_results = _paddle_ocr_full_image(full_image)
    x1, y1, x2, y2 = row_bbox
    w = full_image.width

    row_crop = full_image.crop((max(0, x1), max(0, y1), min(w, x2), min(full_image.height, y2)))
    label_ratio = _find_label_boundary(row_crop)
    label_x_abs = x1 + (x2 - x1) * label_ratio

    margin = 3
    labels = []
    for cx, cy, text, conf in all_results:
        if cy < y1 - margin or cy > y2 + margin:
            continue
        if cx < label_x_abs:
            labels.append(text)

    label = _clean_ocr_text(" ".join(labels))
    label = re.sub(r"[*+'\"`|]+$", "", label).strip()
    label = re.sub(r"^[*+'\"`|]+", "", label).strip()
    return label


def clear_ocr_cache():
    _paddle_cache.clear()


# ──────────────────────────────────────────────
# 메인 OCR 함수
# ──────────────────────────────────────────────

def extract_text_from_image(image: Image.Image) -> str:
    """PIL 이미지에서 한국어+영어 텍스트를 추출한다."""
    reader = _get_easyocr()
    processed = _preprocess_for_ocr(image)
    results = reader.readtext(processed, detail=0, paragraph=True)
    return " ".join(results).strip()


def extract_row_label_and_content(image: Image.Image, row_bbox: list = None,
                                   full_image: Image.Image = None) -> dict:
    """행 이미지에서 라벨(왼쪽)과 내용(오른쪽)을 분리 추출.

    하이브리드 방식:
    1) EasyOCR로 crop 기반 OCR (기본)
    2) EasyOCR 라벨이 빈칸이거나 너무 짧으면 PaddleOCR 전체 이미지 결과로 보정
    """
    # crop 준비
    if full_image is not None and row_bbox is not None:
        bbox = row_bbox
        crop = full_image.crop((
            max(0, bbox[0]), max(0, bbox[1] - 2),
            min(full_image.width, bbox[2]), min(full_image.height, bbox[3] + 2),
        ))
    else:
        crop = image
        full_image = None

    # EasyOCR 기본 추출
    result = _easyocr_extract(crop)

    # EasyOCR 라벨이 부실하면 PaddleOCR로 보정
    if full_image is not None and row_bbox is not None:
        easy_label = result["label"]
        if len(easy_label.replace(" ", "")) <= 1:
            paddle_label = _paddle_get_label_for_row(full_image, row_bbox)
            if len(paddle_label) > len(easy_label):
                result["label"] = paddle_label

    return result


def _easyocr_extract(image: Image.Image) -> dict:
    """EasyOCR로 crop 이미지에서 라벨/내용 추출."""
    reader = _get_easyocr()
    processed = _preprocess_for_ocr(image)
    proc_h, proc_w = processed.shape[:2]

    label_ratio = _find_label_boundary(image)
    results = reader.readtext(processed, detail=1)

    labels = []
    contents = []

    for bbox, text, conf in results:
        if conf < 0.15:
            continue
        text = text.strip()
        if not text:
            continue

        xs = [p[0] for p in bbox]
        center_x = sum(xs) / len(xs)
        ratio = center_x / proc_w

        if ratio < label_ratio:
            labels.append(text)
        else:
            contents.append(text)

    label_text = _clean_ocr_text(" ".join(labels))
    content_text = _clean_ocr_text(" ".join(contents))
    full_text = _clean_ocr_text(" ".join(t for _, t, _ in results if t.strip()))

    label_text = re.sub(r"[*+'\"`|]+$", "", label_text).strip()
    label_text = re.sub(r"^[*+'\"`|]+", "", label_text).strip()

    return {
        "label": label_text,
        "content": content_text,
        "full_text": full_text,
        "label_boundary": round(label_ratio, 2),
    }


def _clean_ocr_text(text: str) -> str:
    """OCR 후처리: 노이즈 제거 및 일반적 오류 수정."""
    if not text:
        return ""

    text = " ".join(text.split())
    text = "".join(c for c in text if c.isprintable())

    text = re.sub(r"[*+'\"`|]+$", "", text)
    text = re.sub(r"^[*+'\"`|]+", "", text)
    text = text.strip()

    if len(text) <= 10:
        text = re.sub(r'(?<=[\uAC00-\uD7A3])\s(?=[\uAC00-\uD7A3])', '', text)

    return text.strip()


def is_field_row(ocr_result: dict) -> bool:
    """OCR 결과를 기반으로 해당 행이 실제 입력 필드인지 판별한다."""
    full_text = ocr_result.get("full_text", "")

    if not full_text:
        return False

    if len(full_text.replace(" ", "")) < 3:
        return False

    skip_exact = [
        "표시 필수입력",
        "필수입력",
    ]

    text_clean = full_text.replace("(", "").replace(")", "").replace("*", "").strip()
    for pat in skip_exact:
        if pat in text_clean:
            return False

    return True
