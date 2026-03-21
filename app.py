from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
from PIL import Image, ImageDraw, ImageFont
from src.graph import build_graph
from src.llm_client import call_ollama_with_image, call_ollama_text
from src.prompts import build_document_overview_prompt
from src.utils import resize_for_vlm
from src.file_filler import fill_document

graph = build_graph()

# 지원하는 파일 확장자
SUPPORTED_FILE_EXTS = {".pdf", ".docx", ".hwp", ".hwpx", ".hwtx"}

# -- UI 텍스트 ----------------------------------------------------------

LANG_LABELS = {
    "ko": {
        "analyzing": "서류를 분석하고 있어요... 잠시만 기다려주세요!",
        "input_prompt": "아래에 입력해주세요!",
        "done_field": "입력 완료!",
        "all_done": "모든 항목 입력이 끝났어요!",
        "summary_title": "입력 내용 요약",
        "field": "칸",
        "example": "예시",
        "warning": "주의",
        "no_fields": "입력 칸을 찾지 못했습니다. 다른 이미지를 시도해주세요.",
        "found_fields": "총 {n}개의 입력 칸을 찾았어요! 하나씩 안내해드릴게요.",
        "upload_first": "왼쪽에 서류 사진을 먼저 올려주세요!",
        "session_done": "모든 입력이 완료되었어요. 새 서류를 분석하려면 이미지를 올리고 다시 분석해주세요!",
    },
    "en": {
        "analyzing": "Analyzing your form... Please wait!",
        "input_prompt": "Please type your answer below!",
        "done_field": "Got it!",
        "all_done": "All fields completed!",
        "summary_title": "Input Summary",
        "field": "Field",
        "example": "Example",
        "warning": "Note",
        "no_fields": "No input fields found. Please try a different image.",
        "found_fields": "Found {n} input fields! I'll guide you through each one.",
        "upload_first": "Please upload a form image on the left first!",
        "session_done": "All done! Upload a new form image and click Analyze to start again.",
    },
    "zh": {
        "analyzing": "正在分析表格...请稍候！",
        "input_prompt": "请在下方输入！",
        "done_field": "已记录！",
        "all_done": "所有字段已完成！",
        "summary_title": "输入摘要",
        "field": "字段",
        "example": "示例",
        "warning": "注意",
        "no_fields": "未找到输入字段。请尝试其他图片。",
        "found_fields": "找到 {n} 个输入字段！我会逐一指导您。",
        "upload_first": "请先在左侧上传表格图片！",
        "session_done": "全部完成！上传新表格并点击分析即可重新开始。",
    },
    "vi": {
        "analyzing": "Đang phân tích biểu mẫu... Vui lòng chờ!",
        "input_prompt": "Vui lòng nhập bên dưới!",
        "done_field": "Đã ghi nhận!",
        "all_done": "Đã hoàn thành tất cả!",
        "summary_title": "Tóm tắt",
        "field": "Mục",
        "example": "Ví dụ",
        "warning": "Lưu ý",
        "no_fields": "Không tìm thấy trường nhập. Vui lòng thử ảnh khác.",
        "found_fields": "Tìm thấy {n} trường nhập! Tôi sẽ hướng dẫn từng bước.",
        "upload_first": "Vui lòng tải ảnh biểu mẫu lên bên trái trước!",
        "session_done": "Hoàn tất! Tải ảnh mới và nhấn Phân tích để bắt đầu lại.",
    },
}


def _lbl(language: str) -> dict:
    return LANG_LABELS.get(language, LANG_LABELS["en"])


# -- 메시지 포매팅 -------------------------------------------------------

def format_field_guidance(field: dict, language: str) -> str:
    """필드 안내를 상담사 말풍선 스타일로 포맷."""
    lbl = _lbl(language)
    fid = field.get("id", "?")
    ko_name = field.get("field_name_ko", "")
    name = field.get("field_name", ko_name)

    if ko_name and ko_name != name:
        header = f"**{lbl['field']} {fid}: {ko_name} → {name}**"
    else:
        header = f"**{lbl['field']} {fid}: {name}**"

    parts = [header, ""]

    if field.get("description"):
        parts.append(field["description"])
        parts.append("")

    if field.get("example"):
        parts.append(f"💡 {lbl['example']}: `{field['example']}`")

    if field.get("warning"):
        parts.append(f"⚠️ {lbl['warning']}: {field['warning']}")

    parts.append("")
    parts.append(f"👇 {lbl['input_prompt']}")

    return "\n".join(parts)


def _parse_checkbox_options(field: dict) -> list[str]:
    """필드에서 체크박스(□) 옵션을 추출. 없으면 빈 리스트."""
    options = []
    field_label = field.get("field_name_ko", "").replace("*", "").strip()

    # raw_text, field_name_ko, description, example 모두에서 □ 옵션 탐색
    for key in ["raw_text", "field_name_ko", "description", "example"]:
        text = field.get(key, "")
        if "□" not in text and "☐" not in text:
            continue
        # "□ 서울  □ 인천·경기  □ 강원" → ["서울", "인천·경기", "강원"]
        parts = re.split(r"[□☐]", text)
        for p in parts:
            opt = p.strip().rstrip("*").strip()
            # 깨끗하게 정리 — 앞뒤 구두점 제거
            opt = opt.strip(",;.·").strip()
            if not opt or len(opt) > 30:
                continue
            if opt in options:
                continue
            # 필드명 자체는 옵션이 아님
            if opt == field_label:
                continue
            # "예시:" 같은 메타 텍스트 제외
            if any(kw in opt for kw in ["예시", "Example", "example"]):
                continue
            options.append(opt)
        if options:  # 첫 번째로 옵션을 찾으면 종료 (중복 방지)
            break
    return options


def _is_checkbox_field(field: dict) -> bool:
    """이 필드가 체크박스 선택 필드인지 판별."""
    return len(_parse_checkbox_options(field)) >= 2


def format_summary(field_infos: list, user_inputs: dict, language: str) -> str:
    """모든 입력 내용을 요약 테이블로 포맷."""
    lbl = _lbl(language)
    lines = [f"## {lbl['summary_title']}", ""]

    for field in field_infos:
        fid = field.get("id")
        ko_name = field.get("field_name_ko", "")
        name = field.get("field_name", ko_name)
        user_val = user_inputs.get(fid, "-")

        if ko_name and ko_name != name:
            display = f"{ko_name} ({name})"
        else:
            display = name

        lines.append(f"**{fid}. {display}**: {user_val}")

    return "\n".join(lines)


def _load_korean_font(size: int):
    """한국어 지원 폰트 로드. 실패 시 기본 폰트 사용."""
    font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_field_cards(
    field_infos: list,
    current_field: int = 0,
    user_inputs: dict = None,
) -> Image.Image:
    """필드 목록을 시각적 카드 이미지로 렌더링.
    - 현재 필드: 파란색 하이라이트
    - 완료 필드: 회색 (컬러 제거)
    - 대기 필드: 흰색
    """
    user_inputs = user_inputs or {}

    if not field_infos:
        img = Image.new("RGB", (400, 100), "#f8f9fa")
        draw = ImageDraw.Draw(img)
        font = _load_korean_font(16)
        draw.text((20, 35), "감지된 필드 없음 / No fields detected", fill="#999", font=font)
        return img

    # 레이아웃 설정
    card_w = 420
    card_h = 64
    card_gap = 8
    padding_x = 16
    padding_top = 50
    padding_bottom = 16

    total_h = padding_top + len(field_infos) * (card_h + card_gap) + padding_bottom
    img = Image.new("RGB", (card_w + padding_x * 2, total_h), "#f0f2f5")
    draw = ImageDraw.Draw(img)

    font_title = _load_korean_font(15)
    font_name = _load_korean_font(17)
    font_sub = _load_korean_font(13)
    font_header = _load_korean_font(14)

    # 헤더
    draw.text(
        (padding_x, 16),
        f"📋 감지된 필드 ({len(user_inputs)}/{len(field_infos)} 완료)",
        fill="#333",
        font=font_header,
    )

    # 색상 정의
    COLOR_CURRENT_BG = "#e3f2fd"      # 연한 파란 배경
    COLOR_CURRENT_BORDER = "#1976d2"  # 진한 파란 테두리
    COLOR_CURRENT_TEXT = "#0d47a1"    # 진한 파란 텍스트
    COLOR_DONE_BG = "#f5f5f5"         # 연한 회색 배경
    COLOR_DONE_BORDER = "#e0e0e0"     # 연한 회색 테두리
    COLOR_DONE_TEXT = "#9e9e9e"        # 회색 텍스트
    COLOR_PENDING_BG = "#ffffff"      # 흰색 배경
    COLOR_PENDING_BORDER = "#e0e0e0"  # 연한 테두리
    COLOR_PENDING_TEXT = "#424242"     # 진한 회색 텍스트

    for i, f in enumerate(field_infos):
        fid = f.get("id", i + 1)
        ko = f.get("field_name_ko", "")
        en = f.get("field_name", ko)
        is_done = fid in user_inputs
        is_current = (i == current_field) and not is_done

        y = padding_top + i * (card_h + card_gap)
        x = padding_x

        # 상태별 색상
        if is_current:
            bg, border, text_color = COLOR_CURRENT_BG, COLOR_CURRENT_BORDER, COLOR_CURRENT_TEXT
            border_w = 3
        elif is_done:
            bg, border, text_color = COLOR_DONE_BG, COLOR_DONE_BORDER, COLOR_DONE_TEXT
            border_w = 1
        else:
            bg, border, text_color = COLOR_PENDING_BG, COLOR_PENDING_BORDER, COLOR_PENDING_TEXT
            border_w = 1

        # 카드 배경 (둥근 모서리 효과)
        draw.rounded_rectangle(
            [x, y, x + card_w, y + card_h],
            radius=8,
            fill=bg,
            outline=border,
            width=border_w,
        )

        # 왼쪽 번호 원
        circle_r = 14
        circle_x = x + 20
        circle_y = y + card_h // 2

        if is_done:
            # 완료: 회색 체크마크 원
            draw.ellipse(
                [circle_x - circle_r, circle_y - circle_r,
                 circle_x + circle_r, circle_y + circle_r],
                fill="#bdbdbd", outline="#9e9e9e",
            )
            draw.text((circle_x - 5, circle_y - 8), "✓", fill="white", font=font_sub)
        elif is_current:
            # 현재: 파란 원 + 번호
            draw.ellipse(
                [circle_x - circle_r, circle_y - circle_r,
                 circle_x + circle_r, circle_y + circle_r],
                fill="#1976d2", outline="#0d47a1",
            )
            num_text = str(fid)
            draw.text(
                (circle_x - (len(num_text) * 4), circle_y - 7),
                num_text, fill="white", font=font_sub,
            )
        else:
            # 대기: 빈 원 + 번호
            draw.ellipse(
                [circle_x - circle_r, circle_y - circle_r,
                 circle_x + circle_r, circle_y + circle_r],
                fill="white", outline="#bdbdbd",
            )
            num_text = str(fid)
            draw.text(
                (circle_x - (len(num_text) * 4), circle_y - 7),
                num_text, fill="#757575", font=font_sub,
            )

        # 텍스트 영역
        text_x = circle_x + circle_r + 14

        if is_done:
            # 완료: 필드명 + 입력값 (회색 취소선 효과)
            user_val = user_inputs.get(fid, "")
            display_val = user_val if len(user_val) <= 20 else user_val[:18] + "…"
            draw.text((text_x, y + 12), ko or en, fill=COLOR_DONE_TEXT, font=font_name)
            draw.text((text_x, y + 36), f"→ {display_val}", fill="#bdbdbd", font=font_sub)
        elif is_current:
            # 현재: 필드명 볼드 + "입력 중" 표시
            draw.text((text_x, y + 10), ko or en, fill=COLOR_CURRENT_TEXT, font=font_name)
            # 영문명이 다르면 같이 표시
            sub_text = f"{en}" if en and en != ko else "입력해주세요"
            draw.text((text_x, y + 35), f"▶ {sub_text}", fill="#42a5f5", font=font_sub)
            # 오른쪽에 깜빡이는 효과를 위한 화살표
            draw.text((x + card_w - 35, y + card_h // 2 - 8), "◀", fill="#1976d2", font=font_name)
        else:
            # 대기: 필드명만
            draw.text((text_x, y + 12), ko or en, fill=COLOR_PENDING_TEXT, font=font_name)
            if en and en != ko:
                draw.text((text_x, y + 36), en, fill="#9e9e9e", font=font_sub)

    return img


# -- 핵심 로직 -----------------------------------------------------------

def _build_file_overview(row_texts: list, language: str) -> str:
    """파일에서 추출된 텍스트로 문서 개요를 LLM으로 생성."""
    from src.config import SUPPORTED_LANGUAGES
    lang = SUPPORTED_LANGUAGES.get(language, "English")

    # 앞부분 텍스트만 사용 (제목 + 초반 내용)
    preview = "\n".join(row_texts[:15])
    prompt = f"""You are a friendly counselor helping a foreigner understand a Korean government form.
The following text was extracted from a Korean document:

{preview}

Based on this text, explain in {lang}:
- The document title and its translation
- What this form is used for
- Who typically fills this out

Respond in {lang} in a warm, friendly counselor tone. Keep it concise (3-5 sentences)."""

    try:
        return call_ollama_text(prompt, max_tokens=512)
    except Exception:
        return "서류를 분석했습니다." if language == "ko" else "Form analyzed."


def _classify_one_row(text: str, language: str) -> dict | None:
    """단일 행 텍스트를 LLM으로 분류하여 field_info 반환. 비필드면 None."""
    from src.prompts import build_row_classify_prompt
    from src.llm_client import call_ollama_text, parse_json_response, parse_json_list_response

    try:
        raw = call_ollama_text(build_row_classify_prompt(text, language))
        try:
            parsed_list = parse_json_list_response(raw)
        except Exception:
            parsed_list = [parse_json_response(raw)]

        results = []
        for parsed in parsed_list:
            ko = parsed.get("field_name_ko", "").strip()
            name = parsed.get("field_name", "").strip()
            if not ko and not name:
                continue
            results.append({
                "id": 0,
                "bbox": [0, 0, 0, 0],
                "field_name_ko": ko,
                "field_name": name or ko,
                "description": parsed.get("description", ""),
                "example": parsed.get("example", ""),
                "warning": parsed.get("warning", ""),
                # 체크박스 파싱용: LLM이 반환한 raw_text 또는 원본 입력
                "raw_text": parsed.get("raw_text", "") or text,
            })
        return results if results else None
    except Exception:
        return None


def _prepare_file_rows(file_path: str) -> list[str]:
    """파일에서 텍스트 추출하여 원본 행과 분류용 행을 반환."""
    from src.file_extractor import extract_rows_from_file

    row_texts = extract_rows_from_file(file_path)

    # 필터링은 _classify_file_rows에서 처리하므로 원본 그대로 반환
    return row_texts, row_texts


def _checkbox_updates_for_field(field: dict):
    """필드가 체크박스면 (cb_group, cb_btn, is_checkbox) 반환."""
    options = _parse_checkbox_options(field)
    if len(options) >= 2:
        return (
            gr.update(choices=options, value=[], visible=True, label=f"☑ {field.get('field_name_ko', '')}"),
            gr.update(visible=True),
            True,
        )
    return (
        gr.update(visible=False, choices=[], value=[]),
        gr.update(visible=False),
        False,
    )


def _hide_checkbox():
    """체크박스 UI 숨김."""
    return (
        gr.update(visible=False, choices=[], value=[]),
        gr.update(visible=False),
    )


def start_analysis(image, file, language, session_state):
    """이미지 또는 파일 업로드 후 '분석 시작' 클릭 시 호출."""
    lbl = _lbl(language)

    has_file = file is not None
    has_image = image is not None

    # 공통 에러/빈 반환 헬퍼 (8개 output)
    def _err(chat, img=None):
        cb_h = _hide_checkbox()
        return chat, img, gr.update(visible=False, value=""), {}, gr.update(interactive=False), gr.update(value=None, visible=False), cb_h[0], cb_h[1]

    if not has_file and not has_image:
        chat = [{"role": "assistant", "content": lbl["upload_first"]}]
        return _err(chat)

    # ── 파일 입력: 배치 방식 (추출→문서분리→개요→전체 필드 분류) ──
    if has_file:
        file_path = file if isinstance(file, str) else str(file)
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in SUPPORTED_FILE_EXTS:
            chat = [{"role": "assistant", "content":
                     f"❌ 지원하지 않는 파일 형식입니다: {ext}\n"
                     f"지원 형식: PDF, DOCX, HWP, HWPX"}]
            return _err(chat)

        # 1) 텍스트 추출
        try:
            raw_texts, _ = _prepare_file_rows(file_path)
        except Exception as e:
            chat = [{"role": "assistant", "content": f"❌ 파일 처리 실패: {e}"}]
            return _err(chat)

        if not raw_texts:
            chat = [{"role": "assistant", "content": lbl["no_fields"]}]
            return _err(chat)

        # 2) 합본 파일 문서 분리
        from src.file_extractor import split_documents
        documents = split_documents(raw_texts)

        # 여러 문서가 있으면 각각 분류 후 섹션별로 안내
        if len(documents) > 1:
            doc_titles = [d["title"] for d in documents]
            chat = [{"role": "assistant", "content":
                     f"📑 이 파일에 **{len(documents)}개 문서**가 포함되어 있습니다:\n" +
                     "\n".join(f"  {i+1}. {t}" for i, t in enumerate(doc_titles)) +
                     "\n\n모든 문서의 입력 필드를 순서대로 안내해드릴게요!"}]
            target_rows = raw_texts  # 전체를 LLM에 넘김
        else:
            target_rows = raw_texts
            chat = []

        # 3) 문서 개요 생성 (LLM 1회)
        overview = _build_file_overview(target_rows, language)
        chat.append({"role": "assistant", "content": f"📄 {overview}"})

        # 4) 전체 필드 배치 분류 (LLM 1회)
        from src.node.classify_fields import _classify_file_rows
        all_fields = _classify_file_rows(target_rows, language)

        if not all_fields:
            chat.append({"role": "assistant", "content": lbl["no_fields"]})
            return _err(chat)

        # 첫 필드 안내
        guidance = format_field_guidance(all_fields[0], language)
        intro = lbl["found_fields"].format(n=len(all_fields))
        chat.append({"role": "assistant", "content":
                     f"{intro}\n\n{guidance}"})

        new_state = {
            "field_infos": all_fields,
            "pending_rows": [],
            "current_field": 0,
            "user_inputs": {},
            "language": language,
            "input_type": "file",
            "file_path": file_path,
            "done": False,
        }

        # 필드 카드 이미지 생성
        card_img = _render_field_cards(all_fields, current_field=0)

        # 첫 필드가 체크박스면 체크박스 UI 표시, 텍스트 입력 숨김
        cb_grp, cb_btn, is_cb = _checkbox_updates_for_field(all_fields[0])
        txt_update = gr.update(interactive=True, visible=not is_cb)
        return chat, card_img, gr.update(visible=False), new_state, txt_update, gr.update(value=None, visible=False), cb_grp, cb_btn

    # ── 이미지 입력: 기존 파이프라인 (전체 처리) ──
    result = graph.invoke({
        "image": image,
        "input_type": "image",
        "language": language,
    })

    if result.get("error"):
        chat = [{"role": "assistant", "content": f"❌ {result['error']}"}]
        return _err(chat)

    field_infos = result.get("field_infos", [])
    annotated = result.get("annotated_image")

    try:
        overview_prompt = build_document_overview_prompt(language)
        resized = resize_for_vlm(image)
        overview = call_ollama_with_image(overview_prompt, resized)
    except Exception:
        overview = "서류를 분석했습니다." if language == "ko" else "Form analyzed."

    chat = [{"role": "assistant", "content": f"📄 {overview}"}]

    if not field_infos:
        chat.append({"role": "assistant", "content": lbl["no_fields"]})
        cb_h = _hide_checkbox()
        return chat, annotated, gr.update(visible=False, value=""), {}, gr.update(interactive=False), gr.update(value=None, visible=False), cb_h[0], cb_h[1]

    intro = lbl["found_fields"].format(n=len(field_infos))
    guidance = format_field_guidance(field_infos[0], language)
    chat.append({"role": "assistant", "content": f"{intro}\n\n{guidance}"})

    new_state = {
        "field_infos": field_infos,
        "current_field": 0,
        "user_inputs": {},
        "language": language,
        "input_type": "image",
        "done": False,
    }

    # 첫 필드가 체크박스면 체크박스 UI 표시
    cb_grp, cb_btn, is_cb = _checkbox_updates_for_field(field_infos[0])
    txt_update = gr.update(interactive=True, visible=not is_cb)
    return chat, annotated, gr.update(visible=False, value=""), new_state, txt_update, gr.update(value=None, visible=False), cb_grp, cb_btn


def _try_fill_document(session_state: dict) -> str | None:
    """세션에서 파일 경로, 필드정보, 입력값으로 채워진 문서를 생성. 실패 시 None."""
    file_path = session_state.get("file_path")
    if not file_path:
        return None
    try:
        return fill_document(
            file_path,
            session_state.get("field_infos", []),
            session_state.get("user_inputs", {}),
        )
    except Exception:
        return None


def _advance_to_next_field(chat_history, session_state, lbl, language):
    """다음 필드로 이동하고 (chat, session_state, next_field_or_None, download_update) 반환."""
    field_infos = session_state["field_infos"]
    current = session_state["current_field"]
    download_update = gr.update()
    next_field = None

    # ── 파일 입력: on-demand 분류 ──
    if session_state.get("input_type") == "file":
        pending = session_state.get("pending_rows", [])

        if current < len(field_infos):
            next_field = field_infos[current]
            logger.info(f"[NEXT] 이미 분류된 필드 사용: {next_field.get('field_name_ko')}")
        elif pending:
            logger.info(f"[CLASSIFY] 남은 행 {len(pending)}개, 다음 행 분류 시작...")
            while pending and next_field is None:
                row = pending.pop(0)
                logger.info(f"[CLASSIFY] 분류 중: {row[:50]}...")
                result = _classify_one_row(row, language)
                logger.info(f"[CLASSIFY] 결과: {result}")
                if result:
                    for f in result:
                        f["id"] = len(field_infos) + 1
                        field_infos.append(f)
                    next_field = result[0]
            session_state["pending_rows"] = pending
            if next_field:
                session_state["current_field"] = field_infos.index(next_field)
        else:
            logger.info("[DONE] pending도 없고 field_infos도 끝")

        if next_field:
            n_done = len(session_state["user_inputs"])
            guidance = format_field_guidance(next_field, language)
            msg = f"✅ {lbl['done_field']} ({n_done}번째 완료)\n\n{guidance}"
            chat_history.append({"role": "assistant", "content": msg})
        else:
            session_state["done"] = True
            summary = format_summary(field_infos, session_state["user_inputs"], language)
            filled_path = _try_fill_document(session_state)
            if filled_path:
                msg = f"🎉 {lbl['all_done']}\n\n{summary}\n\n📥 **작성된 파일을 아래에서 다운로드하세요!**"
                download_update = gr.update(value=filled_path, visible=True)
            else:
                msg = f"🎉 {lbl['all_done']}\n\n{summary}"
            chat_history.append({"role": "assistant", "content": msg})

    # ── 이미지 입력 ──
    else:
        if current < len(field_infos):
            next_field = field_infos[current]
            progress = f"({current}/{len(field_infos)})"
            guidance = format_field_guidance(next_field, language)
            msg = f"✅ {lbl['done_field']} {progress}\n\n{guidance}"
            chat_history.append({"role": "assistant", "content": msg})
        else:
            session_state["done"] = True
            summary = format_summary(field_infos, session_state["user_inputs"], language)
            msg = f"🎉 {lbl['all_done']}\n\n{summary}"
            chat_history.append({"role": "assistant", "content": msg})

    return chat_history, session_state, next_field, download_update


def _build_return(chat_history, session_state, next_field, download_update):
    """handle_user_input / handle_checkbox_input 공용 반환값 구성 (8개).
    Returns: (chatbot, session_state, user_input, annotated, field_list, download, checkbox_group, checkbox_send_btn)
    """
    field_infos = session_state.get("field_infos", [])

    # 필드 카드 이미지 업데이트 (파일 입력인 경우)
    if session_state.get("input_type") == "file":
        card_img = _render_field_cards(
            field_infos, session_state.get("current_field", 0), session_state.get("user_inputs", {})
        )
    else:
        card_img = gr.update()

    # 다음 필드 체크박스 여부
    if next_field:
        cb_grp, cb_btn, is_checkbox = _checkbox_updates_for_field(next_field)
        if is_checkbox:
            # 체크박스 필드: 텍스트 입력 숨기고 체크박스 표시
            user_input_update = gr.update(value="", visible=False)
        else:
            # 일반 필드: 텍스트 입력 표시
            user_input_update = gr.update(value="", visible=True, interactive=True)
    else:
        cb_h = _hide_checkbox()
        cb_grp, cb_btn = cb_h
        user_input_update = gr.update(value="", interactive=False, visible=True)

    return chat_history, session_state, user_input_update, card_img, gr.update(), download_update, cb_grp, cb_btn


def handle_user_input(user_msg, chat_history, session_state):
    """사용자가 필드 값을 입력할 때 호출."""
    logger.info(f"[INPUT] user_msg='{user_msg[:50]}', done={session_state.get('done') if session_state else 'N/A'}")
    if not user_msg.strip():
        cb_h = _hide_checkbox()
        return chat_history, session_state, "", gr.update(), gr.update(), gr.update(), cb_h[0], cb_h[1]

    if not session_state or session_state.get("done"):
        lbl = _lbl(session_state.get("language", "en") if session_state else "en")
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": lbl["session_done"]})
        cb_h = _hide_checkbox()
        return chat_history, session_state, "", gr.update(), gr.update(), gr.update(), cb_h[0], cb_h[1]

    field_infos = session_state["field_infos"]
    current = session_state["current_field"]
    language = session_state["language"]
    lbl = _lbl(language)

    # 현재 필드에 사용자 입력 저장
    field = field_infos[current]
    chat_history.append({"role": "user", "content": user_msg})
    session_state["user_inputs"][field["id"]] = user_msg

    # 다음 필드로 이동
    current += 1
    session_state["current_field"] = current

    chat_history, session_state, next_field, download_update = _advance_to_next_field(
        chat_history, session_state, lbl, language
    )

    return _build_return(chat_history, session_state, next_field, download_update)


def handle_checkbox_input(selected, chat_history, session_state):
    """체크박스 선택 완료 시 호출."""
    logger.info(f"[CHECKBOX] selected={selected}, done={session_state.get('done') if session_state else 'N/A'}")
    if not selected:
        cb_h = _hide_checkbox()
        return chat_history, session_state, "", gr.update(), gr.update(), gr.update(), cb_h[0], cb_h[1]

    if not session_state or session_state.get("done"):
        cb_h = _hide_checkbox()
        return chat_history, session_state, "", gr.update(), gr.update(), gr.update(), cb_h[0], cb_h[1]

    field_infos = session_state["field_infos"]
    current = session_state["current_field"]
    language = session_state["language"]
    lbl = _lbl(language)

    # 선택값을 쉼표로 합침
    user_msg = ", ".join(selected)

    # 현재 필드에 저장
    field = field_infos[current]
    chat_history.append({"role": "user", "content": f"☑ {user_msg}"})
    session_state["user_inputs"][field["id"]] = user_msg

    # 다음 필드
    current += 1
    session_state["current_field"] = current

    chat_history, session_state, next_field, download_update = _advance_to_next_field(
        chat_history, session_state, lbl, language
    )

    return _build_return(chat_history, session_state, next_field, download_update)


# -- Gradio UI -----------------------------------------------------------

CUSTOM_CSS = """
/* ── 전체 ── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── 헤더 ── */
#header-area {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 12px;
    text-align: center;
}
#header-area h1 {
    color: white !important;
    font-size: 1.5em !important;
    margin: 0 !important;
}
#header-area p {
    color: rgba(255,255,255,0.85) !important;
    margin: 4px 0 0 0 !important;
    font-size: 0.9em;
}

/* ── 왼쪽 패널 ── */
#left-panel {
    background: #f8f9fc;
    border-radius: 12px;
    border: 1px solid #e8eaf0;
    padding: 10px;
}

/* ── 분석 버튼 ── */
#analyze-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1em !important;
    padding: 10px !important;
    border-radius: 10px !important;
}
#analyze-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* ── 채팅 영역 ── */
#chatbot {
    border-radius: 12px !important;
    border: 1px solid #e8eaf0 !important;
}

/* ── 입력 행 ── */
#input-row {
    display: flex !important;
    gap: 8px !important;
    align-items: stretch !important;
}
#user-input {
    flex: 1 !important;
}
#user-input textarea {
    border-radius: 10px !important;
    border: 2px solid #e8eaf0 !important;
    padding: 10px 14px !important;
}
#user-input textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
}
#send-btn {
    background: #667eea !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    min-width: 50px !important;
    max-width: 60px !important;
    font-size: 1.2em !important;
}

/* ── 체크박스 ── */
#cb-send-btn {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%) !important;
    color: #1a1a2e !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
}
#cb-group {
    background: #f0fdf4 !important;
    border: 2px solid #86efac !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ── 감지된 필드 이미지 ── */
#detected-fields {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── 다운로드 ── */
#download-area {
    background: #f0fdf4;
    border: 2px solid #86efac;
    border-radius: 12px;
}

/* ── 스크롤바 ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #c5c8d4; border-radius: 3px; }
"""

WELCOME_MSG = (
    "안녕하세요! 서류 작성을 도와드리는 AI 상담사입니다 😊\n\n"
    "**사용법:**\n"
    "1️⃣ 왼쪽에 서류 사진 또는 파일을 올려주세요\n"
    "2️⃣ 언어를 선택하고 **분석 시작**을 눌러주세요\n"
    "3️⃣ 하나씩 안내해드릴게요!\n\n"
    "---\n"
    "Hello! I'm your AI form-filling counselor.\n"
    "Upload your form on the left, select a language, and click **Start Analysis**!"
)

with gr.Blocks(
    title="서류 작성 도우미 / Form Filing Assistant",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
) as demo:
    session_state = gr.State({})

    # ── 헤더 ──
    gr.HTML("""
    <div id="header-area">
        <h1>📋 서류 작성 도우미 &nbsp;|&nbsp; Form Filing Assistant</h1>
        <p>외국인을 위한 한국 서류 작성 AI 상담 서비스</p>
    </div>
    """)

    with gr.Row(equal_height=True):
        # ──────── 왼쪽: 업로드 + 감지 결과 ────────
        with gr.Column(scale=2, min_width=320, elem_id="left-panel"):
            with gr.Tab("📷 사진 / Image"):
                image_input = gr.Image(
                    type="pil",
                    label="서류 사진을 올려주세요",
                    height=220,
                )
            with gr.Tab("📄 파일 / File"):
                file_input = gr.File(
                    label="PDF, DOCX, HWP, HWPX 파일",
                    type="filepath",
                )

            with gr.Row():
                language_input = gr.Dropdown(
                    choices=[
                        ("한국어", "ko"),
                        ("English", "en"),
                        ("中文", "zh"),
                        ("Tiếng Việt", "vi"),
                    ],
                    value="en",
                    label="🌐 언어",
                    scale=1,
                )
                analyze_btn = gr.Button(
                    "🔍 분석 시작",
                    variant="primary",
                    scale=2,
                    elem_id="analyze-btn",
                )

            annotated_output = gr.Image(
                type="pil",
                label="감지된 필드",
                interactive=False,
                elem_id="detected-fields",
                show_label=True,
            )

            field_list_output = gr.Markdown(
                value="", visible=False,
            )

            download_output = gr.File(
                label="📥 작성 완료 파일 다운로드",
                visible=False,
                interactive=False,
                elem_id="download-area",
            )

        # ──────── 오른쪽: 상담 채팅 ────────
        with gr.Column(scale=3, min_width=400):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": WELCOME_MSG}],
                type="messages",
                label="💬 AI 상담",
                height=480,
                layout="bubble",
                elem_id="chatbot",
            )

            # 체크박스 선택 UI (체크박스 필드일 때만 표시)
            checkbox_group = gr.CheckboxGroup(
                choices=[],
                label="☑ 선택하세요",
                visible=False,
                interactive=True,
                elem_id="cb-group",
            )
            checkbox_send_btn = gr.Button(
                "✅ 선택 완료",
                visible=False,
                variant="primary",
                elem_id="cb-send-btn",
            )

            with gr.Row(elem_id="input-row"):
                user_input = gr.Textbox(
                    placeholder="여기에 입력 후 Enter ↵ / Type here...",
                    show_label=False,
                    scale=6,
                    interactive=False,
                    lines=1,
                    max_lines=3,
                    elem_id="user-input",
                )
                send_btn = gr.Button(
                    "➤",
                    scale=1,
                    variant="primary",
                    min_width=50,
                    elem_id="send-btn",
                )

    # ── 이벤트 연결 ──
    analysis_outputs = [chatbot, annotated_output, field_list_output, session_state, user_input, download_output, checkbox_group, checkbox_send_btn]
    handle_outputs = [chatbot, session_state, user_input, annotated_output, field_list_output, download_output, checkbox_group, checkbox_send_btn]

    analyze_btn.click(
        fn=start_analysis,
        inputs=[image_input, file_input, language_input, session_state],
        outputs=analysis_outputs,
    )

    send_btn.click(
        fn=handle_user_input,
        inputs=[user_input, chatbot, session_state],
        outputs=handle_outputs,
    )

    user_input.submit(
        fn=handle_user_input,
        inputs=[user_input, chatbot, session_state],
        outputs=handle_outputs,
    )

    checkbox_send_btn.click(
        fn=handle_checkbox_input,
        inputs=[checkbox_group, chatbot, session_state],
        outputs=handle_outputs,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_api=False,
    )
