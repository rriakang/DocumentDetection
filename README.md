# K Form Helper

서류 이미지에서 입력 칸을 찾아 번호를 붙이고,
각 항목에 무엇을 써야 하는지 설명해주는 MVP입니다.

## 기능
- 서류 이미지 업로드
- 입력 칸 탐지
- 칸별 설명 생성
- 번호가 표시된 이미지 출력
- 선택 언어 출력

## 실행 방법

### 1. Ollama 실행
```bash
ollama pull qwen3-vl:8b
ollama serve