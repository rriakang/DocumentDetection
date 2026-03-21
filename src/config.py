import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen3-vl:8b")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# OpenCV 박스 감지용 (원본 품질 유지)
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 1280

# VLM에 보내는 이미지 크기 (작을수록 추론 빠름)
VLM_IMAGE_MAX_SIZE = 800

SUPPORTED_LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "zh": "中文",
    "vi": "Tiếng Việt",
}

MAX_FIELDS = 30
