from src.config import SUPPORTED_LANGUAGES


def build_document_overview_prompt(language_code: str) -> str:
    """VLM에게 문서 전체 이미지를 보여주고 문서 유형/목적을 설명하게 하는 프롬프트."""
    lang = SUPPORTED_LANGUAGES.get(language_code, "English")

    return f"""You are a friendly counselor helping a foreigner understand a Korean government form.

CRITICAL INSTRUCTIONS:
1. Read the TITLE text at the very top of this document image character by character.
2. The title is usually the largest text, often centered, near the top.
3. Do NOT assume or guess what the document is. ONLY report what you actually see written.
4. If you cannot read the title clearly, say so honestly.

After identifying the title, explain in {lang}:
- The exact document title in Korean (copy it precisely) and its translation
- What this form is used for and when someone needs it
- Who typically fills this out (especially relevant for foreigners in Korea)

Respond in {lang} in a warm, friendly counselor tone.
Keep it concise (3-5 sentences). Do NOT output JSON — use natural text only.
Do NOT invent information you cannot see in the image."""


def build_row_ocr_prompt() -> str:
    """VLM용 — 행 이미지에서 텍스트만 추출 (짧은 프롬프트로 정확도 향상)."""
    return "Read all Korean and English text in this image. Output ONLY the exact text you see, nothing else."


def build_row_classify_prompt(ocr_text: str, language_code: str) -> str:
    """텍스트 LLM용 — 행 OCR 결과를 기반으로 필드 분류 + 설명 생성."""
    lang = SUPPORTED_LANGUAGES.get(language_code, "English")

    return f"""You are helping a foreigner fill out a Korean government form.
The following text was extracted from one section of the form:

"{ocr_text}"

STEP 1: Decide if this is an input field.
NOT input fields (output empty JSON):
- Headers, titles, instructions (e.g., "(*)표시 필수입력")
- Organization names, footer text (e.g., "...협회 귀중")
- Empty or decorative rows

These ARE input fields — do NOT skip:
- Table-style fields with sub-columns (e.g., "참여이력 / 참여 프로그램 / 참여 기간")
- Signature/date fields (e.g., "신청인", "서명", "년/월/일")
- Any field with * marker
- Checkbox fields with □ markers

STEP 2: If this IS an input field, check if there are MULTIPLE separate fields.
For example, "성명* 주민등록번호* - (만 세)" contains TWO fields: 성명 and 주민등록번호.
If multiple fields exist, output a JSON ARRAY of field objects.
If only one field, output a single JSON object.

IMPORTANT — Table-style composite fields:
If the text contains a main label with sub-columns (e.g., "일경험 참여이력\n참여 프로그램  참여 기간  참여 회사(기관)명"),
treat it as ONE field. The field_name_ko should be the main label (e.g., "일경험 참여이력").
In the description, explain ALL sub-columns the user needs to fill in.
Example: "Please provide your work experience: program name, duration, and company name. Format: Program / Period / Company"

Single field format:
{{
  "field_name_ko": "Korean label from the text",
  "field_name": "Accurate translation in {lang}",
  "description": "Clear instructions for a foreigner, in {lang}",
  "example": "Realistic example in {lang}",
  "warning": "Important notes for foreigners, or empty string"
}}

Multiple fields format:
[
  {{"field_name_ko": "...", "field_name": "...", "description": "...", "example": "...", "warning": "..."}},
  {{"field_name_ko": "...", "field_name": "...", "description": "...", "example": "...", "warning": "..."}}
]

Empty (not an input field):
{{"field_name_ko": "", "field_name": "", "description": "", "example": "", "warning": ""}}

Rules:
- field_name_ko = the label part only (e.g., "성 명*" not the whole row text), WITHOUT the * marker
- description must be ONE short sentence (under 15 words) starting with "Write", "Enter", or "Select" — e.g., "Write your full legal name."
- Do NOT add extra explanation, formatting tips, or multi-sentence instructions
- If checkboxes (□) or options exist, list ALL of them with translations in description
- For checkbox fields: include the FULL original text with □ markers in a "raw_text" field for parsing
- For table-style rows (with sub-columns like 참여 프로그램, 참여 기간), describe all sub-columns as ONE field
- All text in {lang} except field_name_ko
- Use realistic but fake examples

Additional field for checkbox fields:
  "raw_text": "full original text with □ markers, e.g. 희망지역*  □ 서울  □ 인천·경기  □ 강원"
Only include raw_text if the field contains □ checkboxes."""


def build_batch_classify_prompt(row_texts: list[str], language_code: str) -> str:
    """텍스트 LLM용 — 전체 행 OCR 결과를 한번에 분석하여 필드 목록 생성."""
    lang = SUPPORTED_LANGUAGES.get(language_code, "English")

    rows_block = ""
    for i, text in enumerate(row_texts):
        rows_block += f"  Row {i+1}: \"{text}\"\n"

    return f"""You are a Korean government form expert helping foreigners.
Below are ALL rows extracted from a Korean form, in order from top to bottom:

{rows_block}
Analyze these rows and identify every INPUT FIELD where the user needs to write or select something.

IMPORTANT RULES:
1. SKIP non-input rows: titles, headers, instructions (e.g., "(*)표시 필수입력"), footer text, organization names, empty rows.
2. MERGE multi-row fields: If a row has NO field label and only contains options/checkboxes (e.g., "□ 울산·대구·경북 □ 부산·경남"), it is a CONTINUATION of the previous field — merge it. Similarly if "고등학교" and "대학교" rows are both under "학교명*", combine into ONE "학교명" field. Include ALL options from ALL rows in the description.
   NEVER skip a field that has a label with * (required marker).
3. SPLIT multi-field rows: If one row contains multiple separate input fields (e.g., "성명*" and "주민등록번호*" on the same row), output them as SEPARATE fields.
4. For checkbox fields: list ALL available options from ALL related rows with translations.
5. For date fields (년/월/일) and signature fields (신청인, 서명): include them as fields.
6. For table-style fields (일경험 참여이력 with sub-columns): describe what sub-columns need to be filled.

Output a JSON array of fields. Each field:
{{
  "field_name_ko": "Exact Korean label",
  "field_name": "Accurate translation in {lang}",
  "description": "Clear, specific instructions for a foreigner, in {lang}",
  "example": "A realistic example value in {lang}",
  "warning": "Important notes or common mistakes for foreigners, or empty string",
  "source_rows": [list of row numbers this field spans]
}}

Output ONLY the JSON array, no other text."""


def build_ocr_field_prompt(ocr_text: str, label: str, language_code: str) -> str:
    """OCR로 추출된 텍스트를 기반으로 텍스트 LLM에 보내는 프롬프트 (레거시)."""
    lang = SUPPORTED_LANGUAGES.get(language_code, "English")

    return f"""You are helping a foreigner fill out a Korean government/official form.
OCR extracted the following text from one row of the form:

Field label (left side): "{label}"
Full row text: "{ocr_text}"

Based on this Korean form field, explain what the user should write.

Output ONLY this JSON (no other text):
{{
  "field_name_ko": "{label}",
  "field_name": "Translated field name in {lang}",
  "description": "Clear instructions on what to write in this field, in {lang}",
  "example": "Example value in {lang} (use fake but realistic data)",
  "warning": "Important notes or common mistakes in {lang}, or empty string"
}}

Rules:
- field_name_ko must be exactly "{label}"
- Translate the field name accurately into {lang}
- description must be ONE short sentence (under 15 words) starting with "Write" or "Enter" — e.g., "Write your full legal name as it appears on your passport."
- Do NOT add extra explanation, formatting tips, or multi-sentence instructions
- If the field involves checkboxes or options, list and translate them all
- Use fake but realistic examples (e.g., fake name, fake address)
- All text except field_name_ko must be in {lang}"""
