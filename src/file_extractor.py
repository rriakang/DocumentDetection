"""PDF, DOCX, HWP/HWPX 파일에서 텍스트를 행 단위로 추출."""

import os
import re
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional


def extract_rows_from_file(file_path: str) -> List[str]:
    """파일 형식에 따라 적절한 추출기를 호출하여 행 텍스트 목록을 반환."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _extract_from_pdf(file_path)
    elif ext == ".docx":
        return _extract_from_docx(file_path)
    elif ext == ".hwpx":
        return _extract_from_hwpx(file_path)
    elif ext == ".hwp":
        return _extract_from_hwp(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def _extract_from_pdf(file_path: str) -> List[str]:
    """pdfplumber로 PDF에서 텍스트+테이블 추출."""
    import pdfplumber

    rows = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # 테이블이 있으면 테이블 우선 추출
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            cells = [c.strip() if c else "" for c in row]
                            text = "  ".join(c for c in cells if c)
                            if text.strip():
                                rows.append(text.strip())
            else:
                # 테이블 없으면 일반 텍스트 추출
                text = page.extract_text()
                if text:
                    for line in text.split("\n"):
                        line = line.strip()
                        if line:
                            rows.append(line)
    return rows


def _extract_from_docx(file_path: str) -> List[str]:
    """python-docx로 Word 문서에서 텍스트+테이블 추출."""
    from docx import Document

    doc = Document(file_path)
    rows = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "tbl":
            # 테이블 처리
            for table in doc.tables:
                if table._element is element:
                    for row in table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        text = "  ".join(c for c in cells if c)
                        if text.strip():
                            rows.append(text.strip())
                    break
        elif tag == "p":
            # 일반 문단 처리
            for para in doc.paragraphs:
                if para._element is element:
                    text = para.text.strip()
                    if text:
                        rows.append(text)
                    break

    return rows


def _extract_from_hwpx(file_path: str) -> List[str]:
    """HWPX (ZIP + XML) 파일에서 텍스트 추출."""
    rows = []

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            # HWPX 내부의 section XML 파일들을 찾음
            section_files = sorted([
                n for n in zf.namelist()
                if n.startswith("Contents/") and n.endswith(".xml")
                and "section" in n.lower()
            ])

            if not section_files:
                # section 파일이 없으면 모든 XML에서 텍스트 추출
                section_files = [
                    n for n in zf.namelist()
                    if n.startswith("Contents/") and n.endswith(".xml")
                ]

            for sf in section_files:
                xml_data = zf.read(sf)
                rows.extend(_parse_hwpx_xml(xml_data))

    except zipfile.BadZipFile:
        raise ValueError("HWPX 파일이 손상되었습니다.")

    return rows


def _parse_hwpx_xml(xml_data: bytes) -> List[str]:
    """HWPX의 section XML에서 텍스트를 추출."""
    rows = []

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return rows

    # 모든 텍스트 노드를 수집
    # HWPX namespace 처리
    ns_pattern = re.compile(r"\{[^}]+\}")

    current_para = []
    for elem in root.iter():
        local_tag = ns_pattern.sub("", elem.tag)

        if local_tag == "t" and elem.text:
            current_para.append(elem.text)
        elif local_tag in ("p", "para") and current_para:
            text = "".join(current_para).strip()
            if text:
                rows.append(text)
            current_para = []

    # 마지막 문단 처리
    if current_para:
        text = "".join(current_para).strip()
        if text:
            rows.append(text)

    return rows


def _extract_from_hwp(file_path: str) -> List[str]:
    """구버전 HWP → LibreOffice로 PDF 변환 후 추출."""
    # LibreOffice 실행 파일 찾기
    lo_path = _find_libreoffice()
    if not lo_path:
        raise RuntimeError(
            "HWP 파일을 처리하려면 LibreOffice가 필요합니다.\n"
            "설치: brew install --cask libreoffice (macOS)\n"
            "또는: sudo apt install libreoffice (Linux)"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # HWP → PDF 변환
        result = subprocess.run(
            [lo_path, "--headless", "--convert-to", "pdf",
             "--outdir", tmpdir, file_path],
            capture_output=True, timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"HWP→PDF 변환 실패: {result.stderr.decode()}")

        # 변환된 PDF 찾기
        pdf_files = list(Path(tmpdir).glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError("HWP→PDF 변환 결과물을 찾을 수 없습니다.")

        return _extract_from_pdf(str(pdf_files[0]))


def _find_libreoffice() -> Optional[str]:
    """시스템에 설치된 LibreOffice 경로를 찾는다."""
    candidates = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS
        "/usr/bin/libreoffice",  # Linux
        "/usr/bin/soffice",  # Linux alternative
        "/snap/bin/libreoffice",  # Snap
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # PATH에서 찾기
    try:
        result = subprocess.run(
            ["which", "libreoffice"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return None
