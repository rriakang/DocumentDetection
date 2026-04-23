from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from src.state import FormGuideState
from src.llm_client import (
    call_ollama_text,
    parse_json_list_response,
)
from src.ocr import extract_row_label_and_content
from src.prompts import build_batch_classify_prompt

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# OCR 기반 필드 분류 (규칙 제거, LLM이 전부 판단)
# ──────────────────────────────────────────────
# 기존 문제: 규칙 기반으로 연속행/서브행/스킵행을 판별하니 케이스마다 깨짐
# 새 방식:
#   1) OCR로 모든 행의 라벨+내용을 추출
#   2) 전체 텍스트를 텍스트 LLM에 한번에 넘김
#   3) LLM이 필드 분류, 병합, 스킵을 알아서 판단
#   4) LLM 결과의 source_rows로 bbox 매칭


def _ocr_all_rows(image, rows: list[dict]) -> list[dict]:
    """모든 행을 OCR하여 라벨/내용/full_text를 추출.

    OCR 하이브리드 실패 시 VLM(qwen3-vl) 폴백 활성화:
    - EasyOCR + PaddleOCR로 전부 빈 응답이면 VLM으로 재추출
    - VLM 응답이 비었을 경우 크롭 영역 확장 재시도
    """
    results = []
    for row in rows:
        try:
            ocr = extract_row_label_and_content(
                image=image, row_bbox=row["bbox"], full_image=image,
                use_vlm_fallback=True,
            )
            results.append({
                "label": ocr.get("label", ""),
                "content": ocr.get("content", ""),
                "full_text": ocr.get("full_text", ""),
                "bbox": row["bbox"],
            })
        except Exception:
            results.append({
                "label": "", "content": "", "full_text": "",
                "bbox": row["bbox"],
            })
    return results


def _build_ocr_batch_prompt(ocr_rows: list[dict], language: str) -> str:
    """OCR 결과를 텍스트 LLM에 보내는 프롬프트.

    규칙을 코드로 하드코딩하지 않고, LLM이 직접 판단하게 함.
    """
    from src.config import SUPPORTED_LANGUAGES
    lang = SUPPORTED_LANGUAGES.get(language, "English")

    rows_block = ""
    for i, row in enumerate(ocr_rows):
        label = row["label"] or "(no label)"
        full = row["full_text"] or "(empty)"
        rows_block += f"  Row {i+1} [label: \"{label}\"]: \"{full}\"\n"

    return f"""You are a Korean government form expert.
Below are ALL rows detected from a Korean form, with OCR text.

{rows_block}
Identify every INPUT FIELD the user must fill in.

CRITICAL — MERGE these into ONE field:
- Checkbox rows spanning multiple lines: ALL "□" options for the same label = ONE field.
  Example: "희망지역 □서울 □인천" and "□울산 □부산" = ONE "희망지역" field
- Education sub-rows: "고등학교 ..." and "대학교 ..." under "학교명" = ONE "학교명" field
- Table fields: "일경험 참여이력" + "참여 프로그램 / 참여 기간 / 참여 회사" + any data rows = ONE "일경험 참여이력" field. Do NOT split sub-columns (참여 프로그램, 참여 기간, 참여 회사) into separate fields.
- "자격/면허" rows near 일경험 = part of the same "일경험 참여이력" table
- Consent text + checkbox (동의/부동의): = ONE "개인정보 제공 동의 여부" field
- Any row with NO label that continues the previous field's options = merge it

SPLIT into SEPARATE fields:
- "성명  주민등록번호" on one row = TWO fields

SKIP completely (NOT input fields):
- Title/header (e.g., "인턴형 일경험 참여신청서", "접수번호")
- Instructions (e.g., "(*)표시 필수입력", "*해당시")
- Long paragraphs of consent/description text (these are NOT fields, they are explanations)
- Footer (e.g., "...협회 귀중", "본인은 위 기재한...")
- "구비서류" rows

INCLUDE these as fields (the user fills them in):
- Date fields at the bottom (e.g., "2026년 월 일" → "작성일" field)
- Signature fields (e.g., "신청인", "(인 또는 서명)" → "신청인 서명" field)
- Age field next to 주민등록번호 (e.g., "(만  세)" → "만 나이" field)

Output JSON array:
[
  {{
    "field_name_ko": "Korean label",
    "field_name": "Translation in {lang}",
    "description": "Short instruction in {lang} (under 15 words). Start with Write/Enter/Select.",
    "example": "Realistic example",
    "warning": "Note for foreigners, or empty",
    "source_rows": [row numbers, 1-based]
  }}
]

Output ONLY JSON array, nothing else."""


def _merge_subfields(field_infos: list[dict]) -> list[dict]:
    """LLM이 쪼개버린 하위 필드를 상위 필드에 병합.

    예: '참여 프로그램', '참여 기간', '참여 회사' → '일경험 참여이력'에 흡수
        'E-mail' → '연락처'에 흡수
        '자격/면허' 가 일경험 근처에 있으면 → '일경험 참여이력'에 흡수
    """
    # 병합 규칙: (상위 필드 키워드, 하위 필드 키워드들)
    merge_rules = [
        ("일경험", ["참여 프로그램", "참여 기간", "참여 회사", "참여 기관", "자격", "면허"]),
        ("연락처", ["E-mail", "e-mail", "이메일", "휴대폰", "휴대돈", "tel", "Tel", "TEL"]),
    ]

    # 1) 먼저 상위 필드가 하위 필드를 흡수하도록 마킹
    absorbed = set()

    for parent_kw, child_kws in merge_rules:
        # 상위 필드 찾기
        parent_idx = None
        for i, field in enumerate(field_infos):
            if parent_kw in field.get("field_name_ko", ""):
                parent_idx = i
                break

        if parent_idx is None:
            continue

        parent = field_infos[parent_idx]
        for j, other in enumerate(field_infos):
            if j == parent_idx or j in absorbed:
                continue
            other_ko = other.get("field_name_ko", "")
            if any(ckw in other_ko for ckw in child_kws):
                sr = parent.get("source_rows", [])
                osr = other.get("source_rows", [])
                parent["source_rows"] = sorted(set(sr + osr))
                absorbed.add(j)

    # 2) 흡수되지 않은 필드만 남기기
    merged = [f for i, f in enumerate(field_infos) if i not in absorbed]

    # 하위 필드만 있고 상위 필드가 없는 경우 처리

    # Case 1: 일경험 참여이력
    sub_kws = ["참여 프로그램", "참여 기간", "참여 회사", "참여 기관"]
    has_parent = any("일경험" in f.get("field_name_ko", "") for f in merged)
    if not has_parent:
        sub_fields = [f for f in merged if any(kw in f.get("field_name_ko", "") for kw in sub_kws)]
        if sub_fields:
            parent = sub_fields[0]
            parent["field_name_ko"] = "일경험 참여이력"
            parent["field_name"] = "Work Experience History"
            parent["description"] = "Enter program name, period, and company for each experience."
            for sf in sub_fields[1:]:
                sr = parent.get("source_rows", [])
                osr = sf.get("source_rows", [])
                parent["source_rows"] = sorted(set(sr + osr))
                merged.remove(sf)

    # Case 2: 연락처 — "이름" + "tel" 이 있는데 "연락처"가 없으면 묶기
    contact_kws = ["이름", "tel", "Tel", "TEL", "E-mail", "e-mail", "이메일", "휴대폰"]
    has_contact = any("연락처" in f.get("field_name_ko", "") for f in merged)
    if not has_contact:
        contact_fields = [f for f in merged if any(kw in f.get("field_name_ko", "") for kw in contact_kws)]
        if len(contact_fields) >= 2:
            parent = contact_fields[0]
            parent["field_name_ko"] = "연락처"
            parent["field_name"] = "Contact Information"
            parent["description"] = "Enter your name and phone number."
            for sf in contact_fields[1:]:
                sr = parent.get("source_rows", [])
                osr = sf.get("source_rows", [])
                parent["source_rows"] = sorted(set(sr + osr))
                merged.remove(sf)

    return merged


def _match_source_rows_to_bbox(field_infos: list[dict],
                                 ocr_rows: list[dict]) -> list[dict]:
    """source_rows 정보를 사용하여 bbox를 할당."""
    for field in field_infos:
        source = field.pop("source_rows", [])
        if not source:
            field["bbox"] = [0, 0, 0, 0]
            continue

        # 1-based → 0-based
        indices = [s - 1 for s in source if 0 < s <= len(ocr_rows)]
        if not indices:
            field["bbox"] = [0, 0, 0, 0]
            continue

        bboxes = [ocr_rows[i]["bbox"] for i in indices]
        field["bbox"] = [
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        ]

    return field_infos


# ──────────────────────────────────────────────
# 파일 입력용 (기존 텍스트 배치 방식)
# ──────────────────────────────────────────────

def _is_skip_text(text: str) -> bool:
    if any(kw in text for kw in [
        "귀중", "귀하", "(사)", "(재)",
        "구비서류", "본인은 위 기재한",
        "(*)표시 필수입력",
    ]):
        return True
    cleaned = re.sub(r'[^가-힣a-zA-Z]', '', text)
    if len(cleaned) < 2:
        return True
    return False


def _post_filter_fields(field_infos: list[dict]) -> list[dict]:
    """LLM이 걸러주지 못한 행정/제목/스킵 필드를 후처리로 제거."""
    skip_names = {
        "접수번호", "접수일자", "접수 번호", "접수 일자",
        "제목", "제  목", "제    목",
        "과제개요", "과제개요 (필수입력)",
        "결재", "산학협력단 결재",
        "수신", "참조",
    }
    result = []
    for f in field_infos:
        ko = f.get("field_name_ko", "").strip()
        # 공백 제거 후 비교
        ko_nospace = ko.replace(" ", "")
        if ko_nospace in {s.replace(" ", "") for s in skip_names}:
            continue
        result.append(f)
    return result


def _classify_file_rows(row_texts: list[str], language: str) -> list[dict]:
    filtered = []
    for text in row_texts:
        t = text.strip()
        if not t or len(t) < 2:
            continue
        if _is_skip_text(t):
            continue
        filtered.append(t)

    CHUNK_SIZE = 25
    all_field_infos = []

    for start in range(0, len(filtered), CHUNK_SIZE):
        chunk = filtered[start:start + CHUNK_SIZE]
        batch_prompt = build_batch_classify_prompt(chunk, language)
        try:
            raw = call_ollama_text(batch_prompt, max_tokens=4096)
            parsed_list = parse_json_list_response(raw)
        except Exception:
            continue

        for parsed in parsed_list:
            ko_name = parsed.get("field_name_ko", "").strip()
            name = parsed.get("field_name", "").strip()
            if not ko_name and not name:
                continue
            all_field_infos.append({
                "id": 0, "bbox": [0, 0, 0, 0],
                "field_name_ko": ko_name,
                "field_name": name or ko_name,
                "description": parsed.get("description", ""),
                "example": parsed.get("example", ""),
                "warning": parsed.get("warning", ""),
            })

    # 하위 필드 병합 (이름/tel → 연락처 등)
    all_field_infos = _merge_subfields(all_field_infos)
    # 행정/제목 필드 후처리 제거
    all_field_infos = _post_filter_fields(all_field_infos)

    for i, f in enumerate(all_field_infos):
        f["id"] = i + 1
    return all_field_infos


# ──────────────────────────────────────────────
# 메인 노드
# ──────────────────────────────────────────────

def classify_fields_node(state: FormGuideState) -> FormGuideState:
    if state.get("error"):
        return state

    language = state.get("language", "en")

    # ── 파일 입력 ──
    if state.get("input_type") == "file" and state.get("row_texts"):
        row_texts = state["row_texts"]
        if not row_texts:
            state["error"] = "파일에서 텍스트를 추출하지 못했습니다."
            return state
        state["field_infos"] = _classify_file_rows(row_texts, language)
        return state

    # ── 이미지 입력: OCR + 텍스트 LLM 방식 ──
    image = state["preprocessed_image"]
    rows = state.get("detected_boxes", [])

    if not rows:
        state["error"] = "입력 칸을 감지하지 못했습니다. 다른 이미지를 시도해주세요."
        return state

    # 1단계: OCR로 모든 행의 텍스트 추출
    logger.info(f"OCR scanning {len(rows)} detected rows...")
    ocr_rows = _ocr_all_rows(image, rows)
    logger.info(f"OCR labels: {[r['label'] for r in ocr_rows]}")

    # 2단계: 텍스트 LLM에 전체 OCR 결과를 넘겨서 필드 분류
    prompt = _build_ocr_batch_prompt(ocr_rows, language)
    try:
        raw = call_ollama_text(prompt, max_tokens=4096)
        logger.info(f"LLM classify response length: {len(raw)}")
        parsed_list = parse_json_list_response(raw)
    except Exception as e:
        logger.error(f"LLM field classification failed: {e}")
        state["error"] = "필드 분류에 실패했습니다. 다시 시도해주세요."
        return state

    # 3단계: 파싱 → field_infos
    field_infos = []
    for parsed in parsed_list:
        ko_name = parsed.get("field_name_ko", "").strip()
        name = parsed.get("field_name", "").strip()
        if not ko_name and not name:
            continue
        field_infos.append({
            "id": 0,
            "bbox": [0, 0, 0, 0],
            "field_name_ko": ko_name,
            "field_name": name or ko_name,
            "description": parsed.get("description", ""),
            "example": parsed.get("example", ""),
            "warning": parsed.get("warning", ""),
            "source_rows": parsed.get("source_rows", []),
        })

    # 4단계: 하위 필드 병합 (LLM이 쪼갠 것 후처리)
    field_infos = _merge_subfields(field_infos)

    # 5단계: source_rows → bbox 매칭
    field_infos = _match_source_rows_to_bbox(field_infos, ocr_rows)

    for i, f in enumerate(field_infos):
        f["id"] = i + 1

    logger.info(f"Final {len(field_infos)} fields: "
                f"{[f['field_name_ko'] for f in field_infos]}")

    state["field_infos"] = field_infos
    return state
