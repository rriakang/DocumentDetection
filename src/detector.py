from typing import List, Dict
import cv2
import numpy as np
from PIL import Image

from src.config import MAX_FIELDS


def _find_lines_with_params(binary: np.ndarray, h: int, w: int,
                             kernel_w: int, threshold_pct: float) -> List[int]:
    """주어진 파라미터로 가로선 y좌표를 찾는다."""
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    row_sums = np.sum(h_lines, axis=1) / 255
    threshold = w * threshold_pct
    line_ys = np.where(row_sums > threshold)[0]
    return [int(y) for y in line_ys]


def _group_line_positions(line_ys: list, gap: int = 6) -> List[int]:
    """인접한 y좌표를 하나의 선으로 그룹화."""
    if len(line_ys) < 1:
        return []

    sorted_ys = sorted(set(line_ys))
    groups: List[int] = []
    current = [sorted_ys[0]]
    for y in sorted_ys[1:]:
        if y - current[-1] <= gap:
            current.append(y)
        else:
            groups.append(int(np.mean(current)))
            current = [y]
    groups.append(int(np.mean(current)))
    return groups


def _find_horizontal_line_positions(binary: np.ndarray, h: int, w: int) -> List[int]:
    """테이블의 가로선 y좌표들을 찾는다."""

    # 기본 감지: 넓은 커널 + 적절한 임계값
    all_ys = _find_lines_with_params(binary, h, w, max(w // 4, 100), 0.15)
    base_lines = _group_line_positions(all_ys)

    # 기본 감지 결과가 충분하면 그대로 사용
    if len(base_lines) >= 8:
        return base_lines

    # 부족하면 더 짧은 커널과 낮은 임계값으로 보강
    for kernel_ratio, thresh in [(6, 0.10), (8, 0.08)]:
        extra_ys = _find_lines_with_params(binary, h, w, max(w // kernel_ratio, 60), thresh)
        all_ys.extend(extra_ys)

    # 끊긴 선 복구 후 재시도
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1))
    repaired = cv2.dilate(binary, repair_kernel, iterations=1)
    repaired = cv2.erode(repaired, repair_kernel, iterations=1)
    extra_ys = _find_lines_with_params(repaired, h, w, max(w // 6, 60), 0.10)
    all_ys.extend(extra_ys)

    return _group_line_positions(all_ys)


def _has_label_text(gray: np.ndarray, y1: int, y2: int, w: int) -> bool:
    """행의 왼쪽 라벨 영역(~25%)에 유의미한 텍스트가 있는지 확인.

    텍스트가 없으면 이전 행의 연속(continuation)으로 판단.
    """
    label_w = int(w * 0.22)
    margin = 4
    roi = gray[y1 + margin:y2 - margin, margin:label_w]
    if roi.size == 0:
        return False

    # 어두운 픽셀(텍스트) 비율 계산 — 경계선 제외
    dark_ratio = float(np.mean(roi < 150))
    return dark_ratio > 0.02


def _merge_continuation_rows(rows: List[Dict], gray: np.ndarray, w: int) -> List[Dict]:
    """라벨 영역에 텍스트가 없는 연속 행을 이전 행과 병합."""
    if len(rows) < 2:
        return rows

    merged = [rows[0]]
    for row in rows[1:]:
        y1, y2 = row["bbox"][1], row["bbox"][3]
        if not _has_label_text(gray, y1, y2, w):
            # 이전 행과 병합: y2를 확장
            merged[-1]["bbox"][3] = y2
        else:
            merged.append(row)

    return merged


def _detect_bottom_content(gray: np.ndarray, last_table_y: int, h: int, w: int) -> List[Dict]:
    """테이블 아래 영역(서명란, 날짜 등)에 텍스트가 있으면 추가 행으로 반환."""
    if last_table_y >= h - 30:
        return []

    bottom_region = gray[last_table_y:h, 0:w]
    if bottom_region.size == 0:
        return []

    # 각 행의 텍스트 밀도 체크 — 테이블 경계선(dark > 0.5)은 제외
    col_dark = np.mean(bottom_region < 150, axis=1)
    text_ys = np.where((col_dark > 0.01) & (col_dark < 0.5))[0]
    if len(text_ys) == 0:
        return []

    content_start = last_table_y + int(text_ys[0])
    content_end = min(h, last_table_y + int(text_ys[-1]) + 10)

    if content_end - content_start < 20:
        return []

    return [{"bbox": [0, content_start, w, content_end]}]


def detect_form_rows(image: Image.Image) -> List[Dict]:
    """서류 이미지에서 테이블 행(row)을 감지한다."""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    binary1 = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8,
    )

    # Otsu threshold (실제 사진에 더 강건)
    _, binary2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 두 방법의 결과를 합침
    binary = cv2.bitwise_or(binary1, binary2)

    line_positions = _find_horizontal_line_positions(binary, h, w)

    if len(line_positions) < 2:
        line_positions = _find_horizontal_line_positions(binary1, h, w)

    if len(line_positions) < 2:
        return []

    rows: List[Dict] = []
    for i in range(len(line_positions) - 1):
        y1 = line_positions[i]
        y2 = line_positions[i + 1]
        row_h = y2 - y1

        if row_h < 20:
            continue

        if row_h > h * 0.40:
            continue

        roi = gray[y1:y2, 0:w]
        if roi.size > 0:
            white_ratio = float(np.mean(roi > 200))
            if white_ratio < 0.10:
                continue

        rows.append({"bbox": [0, y1, w, y2]})

    # 연속 행 병합 (라벨 영역에 텍스트 없는 행 → 이전 행과 합침)
    rows = _merge_continuation_rows(rows, gray, w)

    # 테이블 아래 서명/날짜 영역 추가
    if rows:
        last_table_y = rows[-1]["bbox"][3]
        bottom_rows = _detect_bottom_content(gray, last_table_y, h, w)
        rows.extend(bottom_rows)

    rows = rows[:MAX_FIELDS]

    for idx, item in enumerate(rows, start=1):
        item["id"] = idx

    return rows
