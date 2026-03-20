from typing import List, Tuple
from PIL import Image, ImageEnhance, ImageFilter
from src.config import MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, VLM_IMAGE_MAX_SIZE


def resize_image_if_needed(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= MAX_IMAGE_WIDTH and height <= MAX_IMAGE_HEIGHT:
        return image

    image = image.copy()
    image.thumbnail((MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT))
    return image


def resize_for_vlm(image: Image.Image) -> Image.Image:
    """VLM 전송용으로 이미지 크기를 줄인다. 박스 감지에는 영향 없음."""
    width, height = image.size
    if width <= VLM_IMAGE_MAX_SIZE and height <= VLM_IMAGE_MAX_SIZE:
        return image
    image = image.copy()
    image.thumbnail((VLM_IMAGE_MAX_SIZE, VLM_IMAGE_MAX_SIZE))
    return image


def clamp_box(box: Tuple[int, int, int, int], width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def crop_with_padding(image: Image.Image, box, padding: int = 12) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding
    x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), width, height)
    return image.crop((x1, y1, x2, y2))


def sort_boxes(boxes: List[dict]) -> List[dict]:
    return sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0]))


def enhance_crop_for_vlm(crop: Image.Image, min_height: int = 80) -> Image.Image:
    """행 크롭 이미지를 VLM이 한글을 잘 읽을 수 있도록 품질 향상.

    - 너무 작은 크롭은 업스케일
    - 대비/샤프닝 강화
    - 흰색 여백 추가
    """
    w, h = crop.size

    # 1) 높이가 너무 작으면 업스케일 (한글 글자가 선명하게 보이도록)
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        crop = crop.resize((new_w, min_height), Image.LANCZOS)

    # 2) 대비 강화
    enhancer = ImageEnhance.Contrast(crop)
    crop = enhancer.enhance(1.4)

    # 3) 샤프닝
    crop = crop.filter(ImageFilter.SHARPEN)

    # 4) 흰색 여백 추가 (VLM이 텍스트 경계를 더 잘 인식)
    pad = 20
    w, h = crop.size
    padded = Image.new("RGB", (w + pad * 2, h + pad * 2), (255, 255, 255))
    padded.paste(crop, (pad, pad))

    return padded