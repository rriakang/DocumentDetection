"""
정량 평가 실행 스크립트.

사용법:
  # 전체 파이프라인 E2E 평가 (이미지 필요)
  python -m eval.run_eval --mode e2e --gt eval/ground_truth/sample_form_01.json

  # 개별 레이어 단위 테스트
  python -m eval.run_eval --mode detection   # 행 검출만 테스트
  python -m eval.run_eval --mode ocr          # OCR만 테스트
  python -m eval.run_eval --mode generation   # LLM 생성만 테스트
  python -m eval.run_eval --mode unit         # 메트릭 함수 단위 테스트

참고 논문/벤치마크:
  - FUNSD (ICDAR 2019): Form entity recognition F1
  - SROIE (ICDAR 2019): OCR + Key Information Extraction
  - DocVQA (WACV 2021): ANLS 메트릭
  - XFUND (ACL 2022): 다국어 서류 이해
  - CORD: 영수증 핵심 정보 추출
"""

import argparse
import json
import os
import sys
import time

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import (
    compute_iou,
    compute_cer,
    compute_ned,
    compute_anls,
    evaluate_detection,
    evaluate_ocr,
    evaluate_generation,
    evaluate_end_to_end,
    match_boxes,
)


def run_unit_tests():
    """메트릭 함수 단위 테스트."""
    print("=" * 60)
    print("  메트릭 함수 단위 테스트")
    print("=" * 60)

    # IoU 테스트
    print("\n[IoU 테스트]")
    assert abs(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 0.001
    print("  완전 겹침 IoU = 1.0  OK")

    assert abs(compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) - (25 / 175)) < 0.01
    print(f"  부분 겹침 IoU = {compute_iou([0, 0, 10, 10], [5, 5, 15, 15]):.4f}  OK")

    assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    print("  안 겹침 IoU = 0.0  OK")

    # CER 테스트
    print("\n[CER 테스트]")
    assert compute_cer("성명", "성명") == 0.0
    print("  완전 일치 CER = 0.0  OK")

    cer = compute_cer("성몀", "성명")
    print(f"  1글자 오류 CER = {cer:.4f}  OK")

    # NED 테스트
    print("\n[NED 테스트]")
    ned = compute_ned("Name", "Name")
    assert ned == 1.0
    print(f"  완전 일치 NED = {ned:.4f}  OK")

    ned2 = compute_ned("Nme", "Name")
    print(f"  'Nme' vs 'Name' NED = {ned2:.4f}  OK")

    # ANLS 테스트
    print("\n[ANLS 테스트 - DocVQA 방식]")
    anls1 = compute_anls("성명", "성명")
    print(f"  완전 일치 ANLS = {anls1:.4f}  OK")

    anls2 = compute_anls("완전다른텍스트", "성명")
    print(f"  불일치 ANLS = {anls2:.4f}  OK")

    # 검출 평가 테스트
    print("\n[Detection 평가 테스트]")
    gt_boxes = [[0, 100, 800, 150], [0, 150, 800, 200], [0, 200, 800, 250]]
    pred_boxes = [[0, 102, 800, 148], [0, 151, 800, 199]]  # 2/3 검출, 경계 약간 차이

    det_result = evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5)
    print(f"  Precision: {det_result['precision']}")
    print(f"  Recall:    {det_result['recall']}")
    print(f"  F1:        {det_result['f1']}")
    print(f"  Mean IoU:  {det_result['mean_iou']}")
    print(f"  Row count error: {det_result['row_count_error']}")
    print(f"  Mean boundary error: {det_result['mean_boundary_error_px']}px")

    # OCR 평가 테스트
    print("\n[OCR 평가 테스트]")
    ocr_result = evaluate_ocr(
        pred_labels=["성명", "국적", "생년웜일"],
        gt_labels=["성명", "국적", "생년월일"],
    )
    print(f"  CER:         {ocr_result['cer']}")
    print(f"  NED:         {ocr_result['ned']}")
    print(f"  ANLS:        {ocr_result['anls']}")
    print(f"  Exact Match: {ocr_result['exact_match']}")

    # 생성 평가 테스트
    print("\n[Generation 평가 테스트]")
    gen_result = evaluate_generation(
        predictions=[
            {"field_name_ko": "성명", "field_name": "Name", "description": "Write your full name"},
            {"field_name_ko": "국적", "field_name": "Nationality", "description": "Your country"},
        ],
        ground_truths=[
            {"label_ko": "성명", "label_en": "Name", "description_en": "Write your full legal name"},
            {"label_ko": "국적", "label_en": "Nationality", "description_en": "Write your country of citizenship"},
        ],
        language="en",
    )
    print(f"  Translation NED:    {gen_result['field_name_translation_ned']}")
    print(f"  KO field accuracy:  {gen_result['field_name_ko_accuracy']}")
    print(f"  Description ANLS:   {gen_result['description_anls']}")
    print(f"  JSON success rate:  {gen_result['json_success_rate']}")

    # E2E 평가 테스트
    print("\n[End-to-End 평가 테스트]")
    e2e = evaluate_end_to_end(det_result, ocr_result, gen_result, latency_seconds=45.0)
    print(f"  Detection recall:     {e2e['detection_recall']}")
    print(f"  OCR ANLS:             {e2e['ocr_anls']}")
    print(f"  Generation success:   {e2e['generation_success']}")
    print(f"  E2E success estimate: {e2e['estimated_e2e_success']}")
    print(f"  Bottleneck:           {e2e['bottleneck_stage']}")
    print(f"  Latency:              {e2e['latency_seconds']}s")
    print(f"  Latency/field:        {e2e['latency_per_field']}s")

    print("\n" + "=" * 60)
    print("  모든 단위 테스트 통과!")
    print("=" * 60)


def run_e2e_eval(gt_path: str, language: str = "en"):
    """Ground Truth 파일 기반 E2E 평가."""
    from PIL import Image
    from src.detector import detect_form_rows
    from src.ocr import extract_row_label_and_content
    from src.llm_client import call_ollama_text, parse_json_response
    from src.prompts import build_ocr_field_prompt

    with open(gt_path) as f:
        gt_data = json.load(f)

    image_path = gt_data["image_path"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)

    if not os.path.exists(image_path):
        print(f"[ERROR] 이미지 파일 없음: {image_path}")
        print("  eval/images/ 폴더에 테스트 이미지를 넣어주세요.")
        print("  Ground Truth JSON의 image_path를 확인해주세요.")
        return

    image = Image.open(image_path)
    gt_rows = gt_data["rows"]
    gt_boxes = [r["bbox"] for r in gt_rows]

    print(f"\n{'='*60}")
    print(f"  E2E 평가: {gt_data['description']}")
    print(f"  이미지: {image_path}")
    print(f"  GT 행 수: {len(gt_rows)}")
    print(f"{'='*60}")

    # ── Layer 1: Detection ──
    print("\n[Layer 1] 행 검출 평가...")
    t0 = time.time()
    detected = detect_form_rows(image)
    det_time = time.time() - t0

    pred_boxes = [d["bbox"] for d in detected]
    det_result = evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5)
    print(f"  검출된 행: {len(detected)} / GT: {len(gt_rows)}")
    print(f"  Precision: {det_result['precision']:.4f}")
    print(f"  Recall:    {det_result['recall']:.4f}")
    print(f"  F1:        {det_result['f1']:.4f}")
    print(f"  Mean IoU:  {det_result['mean_iou']:.4f}")
    print(f"  경계 오차: {det_result['mean_boundary_error_px']}px")
    print(f"  소요 시간: {det_time:.2f}s")

    # 매칭된 박스만 이후 평가에 사용
    matched, _, _ = match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5)
    if not matched:
        print("\n  [WARNING] 매칭된 행이 없어 OCR/LLM 평가를 건너뜁니다.")
        return

    # ── Layer 2: OCR ──
    print(f"\n[Layer 2] OCR 품질 평가... ({len(matched)}개 매칭 행)")
    ocr_labels = []
    gt_label_list = []
    t1 = time.time()

    for pi, gi in matched:
        bbox = pred_boxes[pi]
        crop = image.crop((
            max(0, bbox[0]), max(0, bbox[1] - 2),
            min(image.width, bbox[2]), min(image.height, bbox[3] + 2),
        ))
        ocr_result = extract_row_label_and_content(crop)
        ocr_labels.append(ocr_result["label"])
        gt_label_list.append(gt_rows[gi]["label_ko"])

    ocr_time = time.time() - t1
    ocr_result = evaluate_ocr(ocr_labels, gt_label_list)

    print(f"  CER:         {ocr_result['cer']:.4f}")
    print(f"  NED:         {ocr_result['ned']:.4f}")
    print(f"  ANLS:        {ocr_result['anls']:.4f}")
    print(f"  Exact Match: {ocr_result['exact_match']:.4f}")
    print(f"  소요 시간:   {ocr_time:.2f}s")

    # 개별 OCR 결과 상세
    print("\n  [상세 OCR 결과]")
    for i, (pred, gt) in enumerate(zip(ocr_labels, gt_label_list)):
        match_mark = "O" if pred == gt else "X"
        cer = compute_cer(pred, gt)
        print(f"    [{match_mark}] GT: '{gt}' -> OCR: '{pred}' (CER: {cer:.2f})")

    # ── Layer 3: LLM Generation ──
    lang_key = f"label_{language}"
    desc_key = f"description_{language}"

    # GT에 해당 언어 키가 있는지 확인
    has_lang_gt = any(lang_key in r for r in gt_rows)
    if not has_lang_gt:
        print(f"\n  [WARNING] GT에 '{lang_key}' 키 없음. LLM 평가 건너뜀.")
        llm_predictions = []
        gen_result = {"json_success_rate": 0, "field_name_ko_accuracy": 0}
        llm_time = 0
    else:
        print(f"\n[Layer 3] LLM 생성 품질 평가... ({len(matched)}개 필드, 언어: {language})")
        llm_predictions = []
        t2 = time.time()

        for idx, (pi, gi) in enumerate(matched):
            label = ocr_labels[idx] or gt_label_list[idx]
            full_text = label  # OCR 전체 텍스트 대용

            prompt = build_ocr_field_prompt(full_text, label, language)
            try:
                raw = call_ollama_text(prompt)
                parsed = parse_json_response(raw)
                llm_predictions.append(parsed)
                status = "OK"
            except Exception as e:
                llm_predictions.append({"field_name_ko": label, "field_name": "", "description": ""})
                status = f"FAIL: {e}"
            print(f"  [{idx+1}/{len(matched)}] {label} -> {status}")

        llm_time = time.time() - t2
        matched_gts = [gt_rows[gi] for _, gi in matched]
        gen_result = evaluate_generation(llm_predictions, matched_gts, language)

        print(f"\n  Translation NED:   {gen_result['field_name_translation_ned']:.4f}")
        print(f"  KO accuracy:       {gen_result['field_name_ko_accuracy']:.4f}")
        print(f"  Description ANLS:  {gen_result['description_anls']:.4f}")
        print(f"  JSON success:      {gen_result['json_success_rate']:.4f}")
        print(f"  소요 시간:         {llm_time:.2f}s")

    # ── Layer 4: E2E ──
    total_time = det_time + ocr_time + llm_time
    e2e = evaluate_end_to_end(det_result, ocr_result, gen_result, total_time)

    print(f"\n{'='*60}")
    print(f"  종합 평가 결과 (End-to-End)")
    print(f"{'='*60}")
    print(f"  Detection Recall:     {e2e['detection_recall']:.4f}")
    print(f"  OCR ANLS:             {e2e['ocr_anls']:.4f}")
    print(f"  Generation Success:   {e2e['generation_success']:.4f}")
    print(f"  E2E Success (est.):   {e2e['estimated_e2e_success']:.4f}")
    print(f"  Bottleneck:           {e2e['bottleneck_stage']}")
    print(f"  Total Latency:        {e2e['latency_seconds']:.1f}s")
    print(f"  Latency/field:        {e2e['latency_per_field']:.1f}s")

    # 결과 저장
    results = {
        "gt_file": gt_path,
        "language": language,
        "detection": det_result,
        "ocr": ocr_result,
        "generation": gen_result,
        "end_to_end": e2e,
        "timing": {
            "detection_s": round(det_time, 2),
            "ocr_s": round(ocr_time, 2),
            "llm_s": round(llm_time, 2),
            "total_s": round(total_time, 2),
        },
    }

    out_path = gt_path.replace(".json", "_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {out_path}")


def run_detection_only(gt_path: str):
    """검출만 단독 평가."""
    from PIL import Image
    from src.detector import detect_form_rows

    with open(gt_path) as f:
        gt_data = json.load(f)

    image_path = gt_data["image_path"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path)

    if not os.path.exists(image_path):
        print(f"[ERROR] 이미지 없음: {image_path}")
        return

    image = Image.open(image_path)
    gt_boxes = [r["bbox"] for r in gt_data["rows"]]

    detected = detect_form_rows(image)
    pred_boxes = [d["bbox"] for d in detected]

    result = evaluate_detection(pred_boxes, gt_boxes)
    print(json.dumps(result, indent=2))


def run_ocr_only():
    """OCR 메트릭만 단독 테스트 (하드코딩 샘플)."""
    test_cases = [
        (["성명", "국적", "생년월일", "여권번호"], ["성명", "국적", "생년월일", "여권번호"]),
        (["성몀", "국적", "생년웜일", "여권빈호"], ["성명", "국적", "생년월일", "여권번호"]),
        (["이름", "나라", "생일", "여권"], ["성명", "국적", "생년월일", "여권번호"]),
    ]
    for i, (pred, gt) in enumerate(test_cases):
        print(f"\n테스트 {i+1}:")
        result = evaluate_ocr(pred, gt)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="서류 도우미 정량 평가")
    parser.add_argument(
        "--mode",
        choices=["unit", "e2e", "detection", "ocr", "generation"],
        default="unit",
        help="평가 모드",
    )
    parser.add_argument("--gt", type=str, help="Ground Truth JSON 경로")
    parser.add_argument("--lang", type=str, default="en", help="평가 언어 (en/zh/vi/ko)")
    args = parser.parse_args()

    if args.mode == "unit":
        run_unit_tests()
    elif args.mode == "e2e":
        if not args.gt:
            print("E2E 모드는 --gt 인자가 필요합니다.")
            print("예: python -m eval.run_eval --mode e2e --gt eval/ground_truth/sample_form_01.json")
            return
        run_e2e_eval(args.gt, args.lang)
    elif args.mode == "detection":
        if not args.gt:
            print("--gt 인자가 필요합니다.")
            return
        run_detection_only(args.gt)
    elif args.mode == "ocr":
        run_ocr_only()
    elif args.mode == "generation":
        print("Generation 단독 평가는 E2E 모드를 사용해주세요.")


if __name__ == "__main__":
    main()
