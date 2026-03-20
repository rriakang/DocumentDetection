from src.state import FormGuideState
from src.utils import resize_image_if_needed
from src.file_extractor import extract_rows_from_file


def preprocess_node(state: FormGuideState) -> FormGuideState:
    # 파일 입력인 경우: 텍스트 추출
    if state.get("input_type") == "file" and state.get("file_path"):
        try:
            row_texts = extract_rows_from_file(state["file_path"])
            state["row_texts"] = row_texts
        except Exception as e:
            state["error"] = f"파일 처리 실패: {e}"
        return state

    # 이미지 입력인 경우: 기존 로직
    image = state["image"]
    preprocessed = resize_image_if_needed(image)
    state["preprocessed_image"] = preprocessed
    return state
