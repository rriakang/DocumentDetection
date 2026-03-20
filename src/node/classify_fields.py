import re
from src.state import FormGuideState
from src.llm_client import call_ollama_with_image, call_ollama_text, parse_json_response, parse_json_list_response
from src.prompts import build_row_ocr_prompt, build_row_classify_prompt, build_batch_classify_prompt
from src.utils import enhance_crop_for_vlm


def _is_continuation_row(text: str) -> bool:
    """라벨 없이 체크박스/옵션만 있는 연속 행인지 판별."""
    text = text.strip()
    if not text or text == "(empty)" or text == "(unreadable)":
        return False
    # □ 로 시작하는 행 = 이전 필드의 연속 옵션
    if text.startswith("□"):
        return True
    # "대학교", "대 학 교" 등으로 시작 = 학교명 필드의 하위 행
    if re.match(r"^(□\s*)?(대\s*학\s*교|고\s*등\s*학\s*교)", text):
        return True
    return False


def _is_table_subrow(text: str) -> bool:
    """테이블 서브컬럼 행인지 판별 (이전 행의 세부 항목)."""
    text = text.strip()
    if not text:
        return False
    # "참여 프로그램  참여 기간  참여 회사(기관)명" 같은 서브헤더
    sub_keywords = ["참여 프로그램", "참여 기간", "참여 기업", "참여 회사", "기관명", "기관)명"]
    if any(kw in text for kw in sub_keywords):
        return True
    # '*'도 없고 '□'도 없는데 2~3개 단어로만 구성 (서브컬럼 헤더)
    # 예: "프로그램명  기간  기관명"
    parts = [p for p in re.split(r'\s{2,}', text) if p.strip()]
    if len(parts) >= 2 and all(len(p) <= 10 for p in parts) and '*' not in text:
        return True
    return False


def _is_skip_text(text: str) -> bool:
    """비필드 텍스트인지 판별 (조직명, 푸터, 안내문 등)."""
    if any(kw in text for kw in [
        "귀중", "귀하", "(사)", "(재)",
        "구비서류", "본인은 위 기재한",
        "(*)표시", "필수입력",
    ]):
        return True
    # 의미없는 짧은 텍스트 (VLM 할루시네이션) 필터
    cleaned = re.sub(r'[^가-힣a-zA-Z]', '', text)
    if len(cleaned) < 2:
        return True
    return False


def _merge_continuation_rows(
    row_texts: list[str], rows: list[dict] | None = None,
) -> tuple[list[str], list[list[int]]]:
    """연속 행을 사전 병합."""
    merged_texts = []
    merged_indices = []

    for i, text in enumerate(row_texts):
        if i > 0 and _is_continuation_row(text) and merged_texts:
            merged_texts[-1] += "\n" + text
            merged_indices[-1].append(i)
        else:
            merged_texts.append(text)
            merged_indices.append([i])

    return merged_texts, merged_indices


def _ocr_row(image, item, ocr_prompt, padding=4):
    """단일 행을 VLM으로 OCR. 실패 시 크롭을 넓혀서 재시도."""
    x1, y1, x2, y2 = item["bbox"]
    crop = image.crop((
        max(0, x1),
        max(0, y1 - padding),
        min(image.width, x2),
        min(image.height, y2 + padding),
    ))

    try:
        enhanced = enhance_crop_for_vlm(crop)
        text = call_ollama_with_image(ocr_prompt, enhanced).strip()
        if text:
            return text
    except Exception:
        pass

    # 빈 응답 → 크롭을 위아래로 넓혀서 재시도 (더 많은 문맥 제공)
    extended_pad = max(padding, (y2 - y1) // 3)
    crop2 = image.crop((
        max(0, x1),
        max(0, y1 - extended_pad),
        min(image.width, x2),
        min(image.height, y2 + extended_pad),
    ))

    try:
        enhanced2 = enhance_crop_for_vlm(crop2)
        text2 = call_ollama_with_image(ocr_prompt, enhanced2).strip()
        if text2:
            return text2
    except Exception:
        pass

    return "(empty)"


def _classify_rows(merged_texts, merged_indices, language, rows=None):
    """병합된 행 텍스트를 분류하여 field_infos 생성. 이미지/파일 공용."""
    field_infos = []
    field_id = 0

    for text, orig_indices in zip(merged_texts, merged_indices):
        if not text.strip() or text in ("(empty)", "(unreadable)"):
            continue

        if _is_skip_text(text.strip()):
            continue

        try:
            classify_prompt = build_row_classify_prompt(text, language)
            raw = call_ollama_text(classify_prompt)

            # 한 행에 여러 필드가 있을 수 있으므로 배열도 시도
            try:
                parsed_list = parse_json_list_response(raw)
            except Exception:
                parsed_list = [parse_json_response(raw)]

            for parsed in parsed_list:
                ko_name = parsed.get("field_name_ko", "").strip()
                name = parsed.get("field_name", "").strip()
                if not ko_name and not name:
                    continue

                # bbox 계산 (이미지 입력인 경우만)
                bbox = [0, 0, 0, 0]
                if rows:
                    min_idx = min(orig_indices)
                    max_idx = max(orig_indices)
                    if min_idx < len(rows) and max_idx < len(rows):
                        bbox = [
                            rows[min_idx]["bbox"][0],
                            rows[min_idx]["bbox"][1],
                            rows[max_idx]["bbox"][2],
                            rows[max_idx]["bbox"][3],
                        ]

                field_id += 1
                field_infos.append({
                    "id": field_id,
                    "bbox": bbox,
                    "field_name_ko": ko_name,
                    "field_name": name or ko_name,
                    "description": parsed.get("description", ""),
                    "example": parsed.get("example", ""),
                    "warning": parsed.get("warning", ""),
                })
        except Exception:
            continue

    return field_infos


def classify_fields_node(state: FormGuideState) -> FormGuideState:
    if state.get("error"):
        return state

    language = state.get("language", "en")

    # ── 파일 입력: row_texts가 이미 추출되어 있음 ──
    if state.get("input_type") == "file" and state.get("row_texts"):
        row_texts = state["row_texts"]

        if not row_texts:
            state["error"] = "파일에서 텍스트를 추출하지 못했습니다."
            return state

        # 1) 사전 필터: 비필드 행 제거 (파일은 설명문이 많으므로 공격적 필터)
        filtered = []
        for text in row_texts:
            t = text.strip()
            if not t or len(t) < 2:
                continue
            if _is_skip_text(t):
                continue
            # 파일용 추가 필터: 긴 설명문/안내문 제거 (30자 이상이면서 *나 □가 없으면 안내문)
            if len(t) > 40 and '*' not in t and '□' not in t and '［' not in t:
                continue
            filtered.append(t)

        merged_texts, _ = _merge_continuation_rows(filtered)

        # 2) 청크 단위 배치 (최대 25행씩 — LLM이 안정적으로 처리)
        CHUNK_SIZE = 25
        all_field_infos = []

        for start in range(0, len(merged_texts), CHUNK_SIZE):
            chunk = merged_texts[start:start + CHUNK_SIZE]
            batch_prompt = build_batch_classify_prompt(chunk, language)
            try:
                raw = call_ollama_text(batch_prompt, max_tokens=4096)
                parsed_list = parse_json_list_response(raw)
            except Exception:
                # 배치 실패 시 이 청크는 스킵
                continue

            for parsed in parsed_list:
                ko_name = parsed.get("field_name_ko", "").strip()
                name = parsed.get("field_name", "").strip()
                if not ko_name and not name:
                    continue
                all_field_infos.append({
                    "id": 0,
                    "bbox": [0, 0, 0, 0],
                    "field_name_ko": ko_name,
                    "field_name": name or ko_name,
                    "description": parsed.get("description", ""),
                    "example": parsed.get("example", ""),
                    "warning": parsed.get("warning", ""),
                })

        # id 재정렬
        for i, f in enumerate(all_field_infos):
            f["id"] = i + 1

        state["field_infos"] = all_field_infos
        return state

    # ── 이미지 입력: 기존 VLM OCR 파이프라인 ──
    image = state["preprocessed_image"]
    rows = state.get("detected_boxes", [])

    if not rows:
        state["error"] = "입력 칸을 감지하지 못했습니다. 다른 이미지를 시도해주세요."
        return state

    # 1단계: VLM으로 각 행의 텍스트 추출
    ocr_prompt = build_row_ocr_prompt()
    row_texts = [_ocr_row(image, item, ocr_prompt) for item in rows]

    # 2단계: 연속 행 사전 병합
    merged_texts, merged_indices = _merge_continuation_rows(row_texts)

    # 3단계: 분류
    state["field_infos"] = _classify_rows(
        merged_texts, merged_indices, language, rows,
    )
    return state
