from src.state import FormGuideState
from src.detector import detect_form_rows


def detect_boxes_node(state: FormGuideState) -> FormGuideState:
    # 파일 입력인 경우: 박스 감지 불필요
    if state.get("input_type") == "file":
        return state

    image = state["preprocessed_image"]
    rows = detect_form_rows(image)
    state["detected_boxes"] = rows
    return state
