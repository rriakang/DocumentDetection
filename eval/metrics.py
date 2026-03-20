"""
정량 평가 메트릭 모듈.

논문/벤치마크 기반 메트릭:
- Layer 1 (Row Detection): FUNSD/PubLayNet 방식 IoU, Precision, Recall, F1
- Layer 2 (OCR Quality): SROIE/IC15 방식 CER, NED, Label Accuracy
- Layer 3 (LLM Generation): DocVQA 방식 ANLS, JSON Success Rate
- Layer 4 (End-to-End): 전체 파이프라인 성능
"""

from typing import List, Tuple, Dict, Optional


# ──────────────────────────────────────────────
# Layer 1: Row Detection Metrics
# 참고: FUNSD (Jaume et al., ICDAR 2019), PubLayNet
# ──────────────────────────────────────────────

def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """두 바운딩박스의 IoU (Intersection over Union) 계산.

    PASCAL VOC / COCO 표준 방식.
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def match_boxes(
    pred_boxes: List[List[int]],
    gt_boxes: List[List[int]],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """예측 박스와 GT 박스를 IoU 기반 그리디 매칭.

    Returns:
        matched: (pred_idx, gt_idx) 쌍 리스트
        unmatched_pred: 매칭 안 된 예측 인덱스
        unmatched_gt: 매칭 안 된 GT 인덱스
    """
    if not pred_boxes or not gt_boxes:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    # 모든 (pred, gt) 쌍의 IoU 계산
    iou_pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou >= iou_threshold:
                iou_pairs.append((iou, pi, gi))

    # IoU 높은 순으로 그리디 매칭
    iou_pairs.sort(key=lambda x: -x[0])
    matched = []
    used_pred = set()
    used_gt = set()

    for iou, pi, gi in iou_pairs:
        if pi not in used_pred and gi not in used_gt:
            matched.append((pi, gi))
            used_pred.add(pi)
            used_gt.add(gi)

    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]

    return matched, unmatched_pred, unmatched_gt


def evaluate_detection(
    pred_boxes: List[List[int]],
    gt_boxes: List[List[int]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """행 검출 평가 (FUNSD/PubLayNet 표준 방식).

    Returns:
        precision, recall, f1, mean_iou, row_count_error
    """
    matched, unmatched_pred, unmatched_gt = match_boxes(
        pred_boxes, gt_boxes, iou_threshold
    )

    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 매칭된 박스들의 평균 IoU
    mean_iou = 0.0
    if matched:
        ious = [compute_iou(pred_boxes[pi], gt_boxes[gi]) for pi, gi in matched]
        mean_iou = sum(ious) / len(ious)

    # 행 개수 오차
    row_count_error = abs(len(pred_boxes) - len(gt_boxes))

    # 행 경계 오차 (y좌표 기준, 매칭된 박스만)
    boundary_errors = []
    for pi, gi in matched:
        y1_err = abs(pred_boxes[pi][1] - gt_boxes[gi][1])
        y2_err = abs(pred_boxes[pi][3] - gt_boxes[gi][3])
        boundary_errors.append((y1_err + y2_err) / 2)
    mean_boundary_error = (
        sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "row_count_error": row_count_error,
        "mean_boundary_error_px": round(mean_boundary_error, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ──────────────────────────────────────────────
# Layer 2: OCR Quality Metrics
# 참고: SROIE (ICDAR 2019), IC13/IC15
# ──────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """레벤슈타인 편집 거리 계산."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(prediction: str, reference: str) -> float:
    """Character Error Rate (CER).

    CER = edit_distance(pred, ref) / len(ref)
    SROIE/IC15 표준 OCR 평가 메트릭.
    """
    if not reference:
        return 0.0 if not prediction else 1.0
    return _levenshtein(prediction, reference) / len(reference)


def compute_ned(prediction: str, reference: str) -> float:
    """Normalized Edit Distance (1 - NED).

    DocVQA에서 ANLS 계산에 사용하는 방식.
    1.0 = 완벽 일치, 0.0 = 완전 불일치.
    """
    if not reference and not prediction:
        return 1.0
    max_len = max(len(prediction), len(reference))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(prediction, reference)
    return 1.0 - dist / max_len


def compute_anls(prediction: str, reference: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ANLS).

    DocVQA (Mathew et al., WACV 2021) 공식 메트릭.
    OCR 오류에 관대한 소프트 매칭.
    """
    ned = compute_ned(prediction, reference)
    return ned if ned >= threshold else 0.0


def evaluate_ocr(
    pred_labels: List[str],
    gt_labels: List[str],
) -> Dict[str, float]:
    """OCR 품질 평가.

    Args:
        pred_labels: OCR로 추출한 라벨 리스트
        gt_labels: 정답 라벨 리스트 (1:1 매칭 가정)
    """
    if not gt_labels:
        return {"cer": 0.0, "ned": 1.0, "anls": 1.0, "exact_match": 1.0}

    n = min(len(pred_labels), len(gt_labels))
    cers = []
    neds = []
    anlss = []
    exact_matches = 0

    for i in range(n):
        pred = pred_labels[i].strip()
        ref = gt_labels[i].strip()

        cers.append(compute_cer(pred, ref))
        neds.append(compute_ned(pred, ref))
        anlss.append(compute_anls(pred, ref))
        if pred == ref:
            exact_matches += 1

    return {
        "cer": round(sum(cers) / len(cers), 4) if cers else 0.0,
        "ned": round(sum(neds) / len(neds), 4) if neds else 0.0,
        "anls": round(sum(anlss) / len(anlss), 4) if anlss else 0.0,
        "exact_match": round(exact_matches / len(gt_labels), 4),
        "n_evaluated": n,
        "n_gt": len(gt_labels),
    }


# ──────────────────────────────────────────────
# Layer 3: LLM Generation Quality Metrics
# 참고: G-Eval (Liu et al., 2023), BERTScore
# ──────────────────────────────────────────────

def evaluate_generation(
    predictions: List[Dict],
    ground_truths: List[Dict],
    language: str = "en",
) -> Dict[str, float]:
    """LLM 생성 품질 평가.

    Args:
        predictions: [{"field_name_ko": ..., "field_name": ..., "description": ...}, ...]
        ground_truths: [{"label_ko": ..., "label_en": ..., "description_en": ...}, ...]
    """
    lang_key = f"label_{language}"
    desc_key = f"description_{language}"

    n = min(len(predictions), len(ground_truths))
    if n == 0:
        return {
            "field_name_translation_ned": 0.0,
            "field_name_ko_accuracy": 0.0,
            "description_anls": 0.0,
            "json_success_rate": 0.0,
        }

    name_neds = []
    ko_matches = 0
    desc_anlss = []
    json_successes = 0

    for i in range(n):
        pred = predictions[i]
        gt = ground_truths[i]

        # JSON 파싱 성공 여부
        if pred.get("field_name") and pred.get("description"):
            json_successes += 1

        # 한국어 필드명 정확도
        pred_ko = pred.get("field_name_ko", "").strip()
        gt_ko = gt.get("label_ko", "").strip()
        if pred_ko == gt_ko:
            ko_matches += 1

        # 번역된 필드명 유사도 (NED)
        pred_name = pred.get("field_name", "").strip()
        gt_name = gt.get(lang_key, "").strip()
        if gt_name:
            name_neds.append(compute_ned(pred_name.lower(), gt_name.lower()))

        # 설명 유사도 (ANLS)
        pred_desc = pred.get("description", "").strip()
        gt_desc = gt.get(desc_key, "").strip()
        if gt_desc:
            desc_anlss.append(compute_anls(pred_desc.lower(), gt_desc.lower(), threshold=0.3))

    return {
        "field_name_translation_ned": round(
            sum(name_neds) / len(name_neds), 4
        ) if name_neds else 0.0,
        "field_name_ko_accuracy": round(ko_matches / n, 4),
        "description_anls": round(
            sum(desc_anlss) / len(desc_anlss), 4
        ) if desc_anlss else 0.0,
        "json_success_rate": round(json_successes / n, 4),
        "n_evaluated": n,
    }


# ──────────────────────────────────────────────
# Layer 4: End-to-End System Metrics
# ──────────────────────────────────────────────

def evaluate_end_to_end(
    detection_results: Dict,
    ocr_results: Dict,
    generation_results: Dict,
    latency_seconds: float,
) -> Dict[str, float]:
    """전체 파이프라인 종합 평가.

    에러 전파 분석 포함 (FUNSD/CORD 벤치마크 방식).
    """
    # 단계별 성공률 추정
    det_success = detection_results.get("recall", 0)
    ocr_success = ocr_results.get("anls", 0)
    gen_success = generation_results.get("json_success_rate", 0)

    # 에러 전파: 각 단계가 독립적으로 실패할 수 있음
    estimated_e2e_success = det_success * ocr_success * gen_success

    # 병목 분석
    stages = {
        "detection": det_success,
        "ocr": ocr_success,
        "generation": gen_success,
    }
    bottleneck = min(stages, key=stages.get)

    return {
        "detection_recall": round(det_success, 4),
        "ocr_anls": round(ocr_success, 4),
        "generation_success": round(gen_success, 4),
        "estimated_e2e_success": round(estimated_e2e_success, 4),
        "bottleneck_stage": bottleneck,
        "latency_seconds": round(latency_seconds, 2),
        "latency_per_field": round(
            latency_seconds / max(detection_results.get("tp", 1), 1), 2
        ),
    }
