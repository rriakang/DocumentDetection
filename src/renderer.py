from PIL import Image, ImageDraw, ImageFont


COLORS = [
    "#FF4444", "#4488FF", "#44BB44", "#FF8800",
    "#AA44FF", "#FF44AA", "#44DDDD", "#FFCC00",
    "#8844FF", "#FF6644", "#44FF88", "#4444FF",
    "#DD44DD", "#44FFFF", "#FFAA44", "#88FF44",
    "#FF4488", "#4488AA", "#AA8844", "#44AA88",
]


def _load_font(size: int = 16) -> ImageFont.FreeTypeFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_boxes_on_image(image: Image.Image, boxes: list) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = _load_font(16)

    for i, box in enumerate(boxes):
        bbox = box.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        idx = box.get("id", i + 1)
        color = COLORS[i % len(COLORS)]

        # 박스 테두리
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 번호 라벨 배경
        label = str(idx)
        label_w, label_h = 26, 22
        lx = x1
        ly = max(0, y1 - label_h - 2)
        draw.rectangle([lx, ly, lx + label_w, ly + label_h], fill=color)
        draw.text((lx + 5, ly + 1), label, fill="white", font=font)

    return image
