from PIL import Image

from src.config import SUPPORTED_LANGUAGES
from src.llm_client import call_ollama_with_image, parse_json_response
from src.prompts import build_form_helper_prompt
from src.utils import resize_image_if_needed, safe_list, safe_str


def run_form_helper(image: Image.Image, target_language: str = "ko") -> dict:
    image = resize_image_if_needed(image)

    target_language_name = SUPPORTED_LANGUAGES.get(target_language, "한국어")
    prompt = build_form_helper_prompt(target_language_name=target_language_name)

    raw_output = call_ollama_with_image(prompt=prompt, image=image)
    parsed = parse_json_response(raw_output)

    fields = []
    for item in safe_list(parsed.get("fields")):
        fields.append({
            "field_name": safe_str(item.get("field_name"), "확인되지 않음"),
            "description": safe_str(item.get("description"), ""),
            "example": safe_str(item.get("example"), ""),
            "warning": safe_str(item.get("warning"), ""),
        })

    result = {
        "document_type": safe_str(parsed.get("document_type"), "문서 종류를 확인하지 못했습니다"),
        "summary": safe_str(parsed.get("summary"), "문서 요약을 생성하지 못했습니다"),
        "fields": fields,
        "translated_text": safe_str(parsed.get("translated_text"), ""),
    }

    if not result["fields"]:
        result["fields"] = [
            {
                "field_name": "확인 필요",
                "description": "입력 항목을 충분히 식별하지 못했습니다. 문서 원본을 다시 확인해주세요.",
                "example": "",
                "warning": "서류의 해상도와 밝기를 높여 다시 촬영해보세요.",
            }
        ]

    return result