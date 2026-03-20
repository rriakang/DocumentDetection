import cv2
import numpy as np
from PIL import Image, ImageFilter

# easyocr Reader를 싱글턴으로 유지 (초기 로딩이 느리므로)
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _reader


def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """OCR 정확도를 위한 전처리: 업스케일 + 대비 강화 + 이진화.

    easyocr는 작은 텍스트에서 오독이 잦음.
    최소 높이 64px로 업스케일하고, CLAHE 대비 강화 적용.
    """
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]

    # 1) 행 높이가 작으면 업스케일 (최소 높이 80px)
    min_h = 80
    if h < min_h:
        scale = min_h / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2) 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3) CLAHE 대비 강화 (로컬 히스토그램 평활화)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4) 가벼운 디노이즈
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # 5) 다시 3채널 RGB로 (easyocr 입력)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


def _find_label_boundary(image: Image.Image) -> float:
    """행 이미지에서 첫 번째 세로 구분선 x좌표를 찾아 라벨 영역 경계를 결정한다.

    세로선이 없으면 폭의 30%를 기본값으로 사용.
    """
    img = np.array(image.convert("L"))
    h, w = img.shape

    # 세로선 감지: 높이 방향으로 긴 직선 찾기
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 2, 10)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # 각 x좌표별 세로선 픽셀 수
    col_sums = np.sum(v_lines, axis=0) / 255
    threshold = h * 0.4

    # 왼쪽 10%~60% 범위에서 첫 번째 세로선 찾기
    x_start = int(w * 0.10)
    x_end = int(w * 0.60)

    for x in range(x_start, x_end):
        if col_sums[x] > threshold:
            return x / w  # 비율로 반환

    return 0.30  # 세로선 못 찾으면 30%


def extract_text_from_image(image: Image.Image) -> str:
    """PIL 이미지에서 한국어+영어 텍스트를 추출한다."""
    reader = _get_reader()
    processed = _preprocess_for_ocr(image)
    results = reader.readtext(processed, detail=0, paragraph=True)
    return " ".join(results).strip()


def extract_row_label_and_content(image: Image.Image) -> dict:
    """행 이미지에서 라벨(왼쪽)과 내용(오른쪽)을 분리 추출한다.

    1) 세로 구분선을 찾아 라벨 영역 경계를 결정
    2) 전처리(업스케일+대비강화) 후 OCR
    3) 텍스트 위치 기반으로 라벨/내용 분리
    """
    reader = _get_reader()
    processed = _preprocess_for_ocr(image)
    proc_h, proc_w = processed.shape[:2]

    # 세로선 기반 라벨 경계 찾기
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

        # bbox 중심 x좌표
        xs = [p[0] for p in bbox]
        center_x = sum(xs) / len(xs)
        ratio = center_x / proc_w

        if ratio < label_ratio:
            labels.append(text)
        else:
            contents.append(text)

    # 후처리: 일반적인 OCR 오류 정리
    label_text = _clean_ocr_text(" ".join(labels))
    content_text = _clean_ocr_text(" ".join(contents))
    full_text = _clean_ocr_text(" ".join(t for _, t, _ in results if t.strip()))

    # 라벨에서 필수 표시 * 및 노이즈 문자 제거
    import re
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

    # 불필요한 공백 정리
    text = " ".join(text.split())

    # 일반적인 잘못된 문자 제거 (제어 문자 등)
    text = "".join(c for c in text if c.isprintable())

    # 서류 필드 라벨에서 흔한 OCR 노이즈 제거
    import re
    # 끝에 붙는 노이즈 문자 제거: *, +, ', ", `, |
    text = re.sub(r"[*+'\"`|]+$", "", text)
    text = re.sub(r"^[*+'\"`|]+", "", text)
    text = text.strip()

    # 띄어쓰기 낀 한국어 라벨 정규화 (성 명 → 성명, 주 소 → 주소)
    # 한글 사이의 단일 공백 제거 (짧은 라벨에 해당)
    if len(text) <= 10:
        text = re.sub(r'(?<=[\uAC00-\uD7A3])\s(?=[\uAC00-\uD7A3])', '', text)

    return text.strip()


def is_field_row(ocr_result: dict) -> bool:
    """OCR 결과를 기반으로 해당 행이 실제 입력 필드인지 판별한다.

    최소한의 필터링만 적용 — 명백한 비필드만 제거.
    동의서, 참여이력, 서명란 등은 유지.
    """
    full_text = ocr_result.get("full_text", "")

    # 텍스트가 아예 없으면 비필드
    if not full_text:
        return False

    # 3글자 이하의 짧은 텍스트는 장식/빈칸일 가능성 높음
    if len(full_text.replace(" ", "")) < 3:
        return False

    # 명백한 비필드: 안내 헤더만 필터링 (최소한으로)
    skip_exact = [
        "표시 필수입력",       # "(*)표시 필수입력"
        "필수입력",
    ]

    text_clean = full_text.replace("(", "").replace(")", "").replace("*", "").strip()
    for pat in skip_exact:
        if pat in text_clean:
            return False

    return True
