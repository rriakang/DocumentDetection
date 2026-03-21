import base64
import io
import json
import requests
from PIL import Image

from src.config import OLLAMA_VLM_MODEL, OLLAMA_TEXT_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL

# qwen3 thinking 모드 우회: 빈 think 블록을 넣으면 thinking을 건너뜀
_RAW_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n</think>\n"
)


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_ollama_with_image(prompt: str, image: Image.Image,
                           max_tokens: int = 512) -> str:
    image_b64 = _image_to_base64(image)

    payload = {
        "model": OLLAMA_VLM_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
            "num_ctx": 8192,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    raw = response.json().get("response", "").strip()

    # qwen3-vl thinking 모드 응답에서 <think>...</think> 블록 제거
    if "<think>" in raw:
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return raw


def call_ollama_text(prompt: str, max_tokens: int = 4096) -> str:
    """텍스트 전용 LLM (qwen3:8b) 호출. 이미지 없이 텍스트만 처리."""
    payload = {
        "model": OLLAMA_TEXT_MODEL,
        "prompt": _RAW_TEMPLATE.format(prompt=prompt),
        "raw": True,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
            "num_ctx": 8192,
            "repeat_penalty": 1.3,
            "repeat_last_n": 128,
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def parse_json_response(raw_text: str) -> dict:
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw_text = raw_text[start : end + 1]

    return json.loads(raw_text)


def parse_json_list_response(raw_text: str) -> list:
    """JSON 배열 응답을 파싱한다. LLM 출력의 흔한 오류를 자동 수정."""
    import re

    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw_text = raw_text[start : end + 1]

    # LLM이 만드는 흔한 JSON 오류 수정
    # "field," → "field_name" 같은 키 잘림 수정
    raw_text = re.sub(r'"field,"', '"field_name"', raw_text)
    # 후행 쉼표 제거 (마지막 항목 뒤의 ,])
    raw_text = re.sub(r',\s*]', ']', raw_text)
    raw_text = re.sub(r',\s*}', '}', raw_text)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # 잘린 JSON 복구 시도: 마지막 완전한 } 까지만 사용
        last_brace = raw_text.rfind("}")
        if last_brace != -1:
            raw_text = raw_text[:last_brace + 1] + "]"
            result = json.loads(raw_text)
        else:
            raise

    if not isinstance(result, list):
        raise ValueError("Expected JSON array")
    return result
