from typing import Any, Dict, List, TypedDict
from PIL import Image


class FormGuideState(TypedDict, total=False):
    image: Image.Image
    language: str

    # 파일 입력 관련
    file_path: str              # 업로드된 파일 경로
    input_type: str             # "image" 또는 "file"
    row_texts: List[str]        # 파일에서 추출된 행 텍스트 목록

    preprocessed_image: Image.Image
    document_type: str

    detected_boxes: List[Dict[str, Any]]
    field_infos: List[Dict[str, Any]]

    annotated_image: Image.Image
    final_text: str
    error: str