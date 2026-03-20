from src.state import FormGuideState

LABELS = {
    "ko": {
        "title": "입력 항목 안내",
        "original": "원본",
        "desc": "설명",
        "example": "예시",
        "warning": "주의사항",
    },
    "en": {
        "title": "Form Field Guide",
        "original": "Korean",
        "desc": "Description",
        "example": "Example",
        "warning": "Warning",
    },
    "zh": {
        "title": "填写指南",
        "original": "韩文原名",
        "desc": "说明",
        "example": "示例",
        "warning": "注意事项",
    },
    "vi": {
        "title": "Huong dan dien bieu mau",
        "original": "Tieng Han",
        "desc": "Mo ta",
        "example": "Vi du",
        "warning": "Luu y",
    },
}


def finalize_node(state: FormGuideState) -> FormGuideState:
    if state.get("error"):
        state["final_text"] = state["error"]
        return state

    language = state.get("language", "en")
    field_infos = state.get("field_infos", [])
    lbl = LABELS.get(language, LABELS["en"])

    lines = [lbl["title"], "=" * 40, ""]

    for field in field_infos:
        ko_name = field.get("field_name_ko", "")
        name = field.get("field_name", ko_name)
        fid = field.get("id", "?")

        # 한국어 원본명 + 번역명
        if ko_name and ko_name != name:
            lines.append(f"[{fid}] {ko_name}  ->  {name}")
        else:
            lines.append(f"[{fid}] {name}")

        if field.get("description"):
            lines.append(f"    {lbl['desc']}: {field['description']}")
        if field.get("example"):
            lines.append(f"    {lbl['example']}: {field['example']}")
        if field.get("warning"):
            lines.append(f"    {lbl['warning']}: {field['warning']}")
        lines.append("")

    state["final_text"] = "\n".join(lines).strip()
    return state
