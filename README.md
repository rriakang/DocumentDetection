# DocumentDetection - 서류 작성 도우미

한국 관공서 서류를 외국인이 쉽게 작성할 수 있도록 도와주는 AI 기반 서류 분석 시스템입니다.

서류 이미지 또는 파일(PDF, DOCX, HWP)을 업로드하면 입력 칸을 자동으로 감지하고, 각 항목에 무엇을 써야 하는지 선택한 언어로 안내해줍니다.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 주요 기능

- **서류 입력 칸 자동 감지** - OpenCV 기반 테이블 행 검출 + 연속 행 자동 병합
- **OCR 텍스트 추출** - EasyOCR 한국어/영어 인식 + 노이즈 후처리
- **다국어 안내 생성** - LLM이 각 필드를 외국인 관점에서 쉽게 설명
- **대화형 입력** - 챗봇 형태로 하나씩 안내하며 입력 수집
- **체크박스 지원** - 선택형 필드 자동 감지 및 UI 제공
- **파일 자동 작성** - 입력 완료 후 원본 파일에 값을 채워서 다운로드
- **다국어 지원** - 한국어, English, 中文, Tiếng Việt

## 데모

```
서류 이미지 업로드 → 칸 감지 → OCR → LLM 안내 생성 → 대화형 입력 → 완성 파일 다운로드
```

## 시스템 아키텍처

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  이미지/파일  │ ──▶ │  행 검출      │ ──▶ │  OCR 추출    │ ──▶ │  LLM 분류    │
│  업로드      │     │  (OpenCV)    │     │  (EasyOCR)  │     │  + 안내 생성  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                     │
                    ┌──────────────┐     ┌─────────────┐             ▼
                    │  파일 작성    │ ◀── │  대화형 입력  │ ◀── ┌──────────────┐
                    │  + 다운로드   │     │  수집        │     │  Gradio UI   │
                    └──────────────┘     └─────────────┘     └──────────────┘
```

### 파이프라인 (LangGraph)

| 단계 | 모듈 | 설명 |
|------|------|------|
| 전처리 | `src/node/preprocess.py` | 이미지 리사이즈, 품질 보정 |
| 행 검출 | `src/detector.py` | OpenCV 가로선 감지 + 연속 행 병합 |
| OCR | `src/ocr.py` | EasyOCR + 전처리(업스케일, CLAHE) + 후처리 |
| 필드 분류 | `src/node/classify_fields.py` | LLM 기반 필드 판별 및 안내 생성 |
| 렌더링 | `src/renderer.py` | 감지된 칸에 번호 표시 |
| 파일 채우기 | `src/file_filler.py` | DOCX/HWP 파일에 사용자 입력 자동 기입 |

## 설치 및 실행

### 요구사항

- Python 3.11+
- [Ollama](https://ollama.ai/) (로컬 LLM 서버)

### 1. 저장소 클론 및 환경 설정

```bash
git clone https://github.com/rriakang/DocumentDetection.git
cd DocumentDetection

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama 모델 설치 및 실행

```bash
# 모델 다운로드
ollama pull qwen3-vl:8b   # Vision-Language 모델 (이미지 분석용)
ollama pull qwen3:8b       # Text 모델 (필드 분류/안내 생성용)

# 서버 실행
ollama serve
```

### 3. 앱 실행

```bash
python app.py
```

브라우저에서 `http://localhost:7860` 으로 접속합니다.

### 환경 변수 (선택)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_VLM_MODEL` | `qwen3-vl:8b` | Vision 모델 |
| `OLLAMA_TEXT_MODEL` | `qwen3:8b` | Text 모델 |
| `OLLAMA_TIMEOUT` | `180` | 요청 타임아웃 (초) |

## 사용 방법

1. 왼쪽 패널에서 **서류 사진** 또는 **파일**(PDF, DOCX, HWP, HWPX)을 업로드
2. **언어**를 선택 (English, 中文, Tiếng Việt, 한국어)
3. **분석 시작** 클릭
4. 챗봇이 각 필드를 하나씩 안내 - 안내에 따라 값 입력
5. 모든 입력 완료 시 요약 표시 + 파일 입력의 경우 **작성된 파일 다운로드**

## 평가 (Evaluation)

논문/벤치마크 기반 정량 평가 프레임워크를 포함하고 있습니다.

```bash
# 메트릭 함수 단위 테스트
python -m eval.run_eval --mode unit

# E2E 전체 파이프라인 평가
python -m eval.run_eval --mode e2e --gt eval/ground_truth/synth_form_01.json --lang en
python -m eval.run_eval --mode e2e --gt eval/ground_truth/complex_form.json --lang en
```

### 평가 메트릭

| Layer | 메트릭 | 참고 논문/벤치마크 |
|-------|--------|-------------------|
| Detection | IoU, Precision, Recall, F1 | FUNSD (ICDAR 2019), PubLayNet |
| OCR | CER, NED, ANLS, Exact Match | SROIE (ICDAR 2019), IC15 |
| Generation | Translation NED, Description ANLS | DocVQA (WACV 2021), G-Eval |
| E2E | Error Propagation, Bottleneck Analysis | FUNSD, CORD |

### 현재 성능

| 메트릭 | Simple Form | Complex Form |
|--------|-------------|--------------|
| Detection F1 | 1.000 | 0.909 |
| Detection Recall | 1.000 | 1.000 |
| OCR ANLS | 1.000 | 0.940 |
| OCR Exact Match | 1.000 | 0.700 |
| LLM JSON Success | 1.000 | 1.000 |
| E2E Success | 1.000 | 0.940 |

## 프로젝트 구조

```
DocumentDetection/
├── app.py                  # Gradio 메인 앱
├── requirements.txt
├── src/
│   ├── config.py           # 설정 (모델, 이미지 크기 등)
│   ├── detector.py         # OpenCV 행 검출 + 연속 행 병합
│   ├── ocr.py              # EasyOCR 텍스트 추출 + 후처리
│   ├── llm_client.py       # Ollama API 클라이언트
│   ├── prompts.py          # LLM 프롬프트 템플릿
│   ├── graph.py            # LangGraph 파이프라인 정의
│   ├── state.py            # 파이프라인 상태 정의
│   ├── file_extractor.py   # PDF/DOCX/HWP 텍스트 추출
│   ├── file_filler.py      # 파일에 사용자 입력 기입
│   ├── renderer.py         # 이미지에 번호 렌더링
│   ├── utils.py            # 유틸리티 함수
│   └── node/               # LangGraph 노드
│       ├── preprocess.py
│       ├── detect_boxes.py
│       ├── classify_fields.py
│       ├── generate_guidance.py
│       ├── draw_boxes.py
│       └── finalize.py
└── eval/
    ├── run_eval.py          # 평가 실행 스크립트
    ├── metrics.py           # 평가 메트릭 (IoU, CER, ANLS 등)
    ├── ground_truth/        # GT 데이터
    └── images/              # 테스트 이미지
```

## 기술 스택

- **UI**: Gradio 5.0
- **파이프라인**: LangGraph
- **행 검출**: OpenCV (Adaptive Threshold + Morphology)
- **OCR**: EasyOCR (한국어/영어)
- **LLM**: Ollama (qwen3-vl:8b, qwen3:8b)
- **파일 처리**: pdfplumber, python-docx
