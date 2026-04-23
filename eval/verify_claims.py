"""주장 검증 스크립트:
- #11: 다중행 필드 병합 성공률 (체크박스/서브행이 제대로 병합되는지)
- #12: 12개 필드 기준 10개 이상 정확 식별
"""
import json
import sys
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from src.graph import build_graph


def _similar(a: str, b: str) -> float:
    """문자열 유사도 (0~1)."""
    a_clean = a.replace(" ", "").replace("*", "").replace("(", "").replace(")", "")
    b_clean = b.replace(" ", "").replace("*", "").replace("(", "").replace(")", "")
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _match_gt(gt_label: str, predicted_labels: list) -> tuple:
    """GT 라벨과 예측 라벨 중 최고 매칭 찾기."""
    best_score = 0.0
    best_pred = None
    for p in predicted_labels:
        score = _similar(gt_label, p)
        # 부분 포함도 인정
        if gt_label.replace(" ", "") in p.replace(" ", "") or \
           p.replace(" ", "") in gt_label.replace(" ", ""):
            score = max(score, 0.9)
        if score > best_score:
            best_score = score
            best_pred = p
    return best_pred, best_score


def verify_field_accuracy(gt_path: str, threshold: float = 0.7):
    """12개 필드 기준 정확 식별률 검증."""
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    image_path = Path(__file__).parent.parent / gt["image_path"]
    image = Image.open(image_path)

    graph = build_graph()
    state = {
        "input_type": "image",
        "image": image,
        "language": "en",
    }
    result = graph.invoke(state)

    field_infos = result.get("field_infos", [])
    predicted_labels = [f.get("field_name_ko", "") for f in field_infos]

    print(f"\n{'='*60}")
    print(f"Field Accuracy Verification — {gt.get('description', '')}")
    print(f"{'='*60}")
    print(f"GT fields: {len(gt['rows'])}")
    print(f"Predicted fields: {len(predicted_labels)}")
    print(f"Threshold for match: {threshold}")
    print()

    correct = 0
    unmatched_gt = []
    for row in gt["rows"]:
        gt_label = row["label_ko"]
        pred, score = _match_gt(gt_label, predicted_labels)
        status = "OK " if score >= threshold else "MISS"
        print(f"  [{status}] GT: {gt_label:20s} → pred: {pred!s:25s} (sim={score:.2f})")
        if score >= threshold:
            correct += 1
        else:
            unmatched_gt.append(gt_label)

    print()
    print(f"Result: {correct}/{len(gt['rows'])} fields correctly identified")
    claim_target = 10  # "12개 중 10개 이상"
    passed = correct >= claim_target
    print(f"Claim '10 out of 12': {'PASS' if passed else 'FAIL'}")
    if unmatched_gt:
        print(f"Missed: {unmatched_gt}")

    return correct, len(gt["rows"]), predicted_labels


def verify_merge_success_rate():
    """다중행 필드 병합 성공률 검증.

    complex_form.png의 경우:
    - 학교명: 고등학교/대학교 2행 → 1개로 병합돼야 함
    - 희망지역: 체크박스 2행(서울~충남, 울산~제주) → 1개로 병합돼야 함
    - 일경험 참여이력: 헤더 + 데이터 행 → 1개로 병합돼야 함
    - 개인정보 제공: 본문 + 체크박스 → 1개로 병합돼야 함

    병합 대상(merge_targets) 중 실제 1개로 병합된 비율.
    """
    image_path = Path(__file__).parent.parent / "eval/images/complex_form.png"
    image = Image.open(image_path)

    graph = build_graph()
    state = {
        "input_type": "image",
        "image": image,
        "language": "en",
    }
    result = graph.invoke(state)

    field_infos = result.get("field_infos", [])
    labels = [f.get("field_name_ko", "") for f in field_infos]
    print(f"\n{'='*60}")
    print(f"Multi-Row Merge Success Rate — complex_form.png")
    print(f"{'='*60}")
    print(f"Predicted: {labels}")

    # 병합 체크: 각 대상이 1개 필드로만 나왔는지
    merge_targets = [
        ("학교명",    ["학교명", "고등학교", "대학교"]),
        ("희망지역",  ["희망지역", "서울", "부산", "제주"]),
        ("일경험 참여이력", ["일경험", "참여이력", "참여 프로그램", "참여 기간", "참여 회사"]),
        ("개인정보 동의", ["개인정보", "동의"]),
    ]

    correct = 0
    total = len(merge_targets)
    for target_name, keywords in merge_targets:
        # 각 필드 라벨에서 키워드 중 하나라도 포함하는 필드 개수 계산
        matched = [l for l in labels if any(kw in l for kw in keywords)]
        # 성공 = 정확히 1개 필드로 병합
        ok = len(matched) == 1
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {target_name}: {len(matched)}개 필드로 표현 → {matched}")
        if ok:
            correct += 1

    rate = correct / total
    print()
    print(f"Merge success rate: {correct}/{total} = {rate:.0%}")
    print(f"Claim '91%': {'APPROX PASS' if rate >= 0.85 else 'FAIL'}")
    return correct, total


if __name__ == "__main__":
    # Claim #12: 12개 필드 중 10개 이상
    correct, total, _ = verify_field_accuracy(
        "eval/ground_truth/real_form_12fields.json"
    )

    # Claim #11: 다중행 병합 성공률
    merge_correct, merge_total = verify_merge_success_rate()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Claim #12 (10/12 field accuracy): {correct}/{total}")
    print(f"Claim #11 (multi-row merge):      {merge_correct}/{merge_total}")
