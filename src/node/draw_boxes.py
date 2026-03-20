from src.state import FormGuideState
from src.renderer import draw_boxes_on_image


def draw_boxes_node(state: FormGuideState) -> FormGuideState:
    if state.get("error"):
        return state

    # 파일 입력인 경우: 이미지가 없으므로 스킵
    if state.get("input_type") == "file":
        return state

    image = state["preprocessed_image"]
    # 필터링된 field_infos만 그려서 비필드 행(제목, 안내문)은 표시 안 함
    fields = state.get("field_infos", [])

    if fields:
        annotated = draw_boxes_on_image(image, fields)
    else:
        annotated = image.copy()

    state["annotated_image"] = annotated
    return state
