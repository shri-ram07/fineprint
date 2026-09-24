"""Turn uploaded files into plain text.

Every byte that reaches this module is untrusted, so each parser is bounded
(input size, decompressed size, extracted characters) and every failure is
reported as a `DocumentError` with a message the user can act on.
"""

import codecs
import io
import logging
import re
import zipfile
from typing import Literal
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import fromstring as parse_xml
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError

logger = logging.getLogger(__name__)

DocumentKind = Literal["pdf", "docx", "txt", "md"]
ErrorCode = Literal["empty", "no_text_layer", "encrypted", "too_large", "corrupt"]

# ~75k tokens: comfortably inside the model's context together with the prompt and output.
MAX_DOCUMENT_CHARS = 300_000
# Word XML carries 5-20x markup per character of text, so this is a zip-bomb ceiling,
# deliberately not derived from the text limit.
MAX_XML_BYTES = 32 * 1024 * 1024

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


class DocumentError(Exception):
    """A file could not be turned into usable text. `message` is safe to show to the user."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _too_large(max_chars: int) -> DocumentError:
    return DocumentError(
        "too_large",
        f"This document is longer than the {max_chars:,}-character limit. "
        "Try uploading only the section you need help with.",
    )


def extract_text(kind: DocumentKind, data: bytes, max_chars: int = MAX_DOCUMENT_CHARS) -> str:
    """Extract readable text from an uploaded file.

    Raises:
        DocumentError: the file is empty, unreadable, encrypted, scanned, or too long.
    """
    if not data:
        raise DocumentError("empty", "The file is empty.")

    if kind in ("txt", "md"):
        text = _decode_text(data)
    else:
        try:
            text = _pdf_text(data, max_chars) if kind == "pdf" else _docx_text(data)
        except DocumentError:
            raise
        except FileNotDecryptedError:
            raise DocumentError(
                "encrypted",
                "This PDF is password-protected. Remove the password (for example with "
                "'Print to PDF') and upload it again.",
            ) from None
        except Exception as exc:
            # The parsers' exception trees are open-ended (zlib.error, RuntimeError, pypdf
            # errors, ...) and the bytes are untrusted, so any failure means "unreadable".
            logger.warning("Could not parse %s upload: %s", kind, type(exc).__name__)
            raise DocumentError(
                "corrupt",
                f"The file could not be read. It may be damaged or not a real {kind} file.",
            ) from None

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise DocumentError("empty", "No text was found in this file.")
    if len(text) > max_chars:
        raise _too_large(max_chars)
    return text


def _decode_text(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Plain-text files saved by older Windows editors are usually cp1252.
        return data.decode("cp1252", errors="replace")


def _pdf_text(data: bytes, max_chars: int) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    found_text = False
    length = 0
    for number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        found_text = found_text or bool(page_text)
        # Mark unreadable pages (typically a scanned signature page or exhibit) so the
        # user and the model both know something is missing.
        pages.append(page_text or f"[Page {number}: no readable text - scanned image?]")
        length += len(pages[-1])
        # Bounds accumulation across pages. A single page's streams are capped by pypdf
        # (75 MB decompressed each), and the web layer parses at most two files at once.
        if length > max_chars:
            raise _too_large(max_chars)
    if not found_text:
        raise DocumentError(
            "no_text_layer",
            "This PDF has no selectable text; it is probably a scanned image. "
            "Text recognition (OCR) isn't supported, so please paste the text instead.",
        )
    return "\n\n".join(pages)


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive, archive.open("word/document.xml") as member:
        # Read at most one byte past the ceiling: never trust the size the archive declares.
        xml = member.read(MAX_XML_BYTES + 1)
    if len(xml) > MAX_XML_BYTES:
        raise DocumentError("too_large", "This Word document is too large to process.")

    parts: list[str] = []
    # defusedxml refuses DTDs and entity expansion, the classic XML attacks on uploads.
    _collect_docx_text(parse_xml(xml), parts)
    return "".join(parts)


def _collect_docx_text(node: Element, parts: list[str]) -> None:
    """Walk the document once, in order. Paragraphs nested in text boxes are visited once,
    and Word's legacy copy of each text box (mc:Fallback) is skipped so nothing repeats."""
    for child in node:
        if child.tag == _FALLBACK:
            continue
        if child.tag == f"{_WORD_NS}t":
            parts.append(child.text or "")
        elif child.tag == f"{_WORD_NS}tab":
            parts.append("\t")
        elif child.tag == f"{_WORD_NS}noBreakHyphen":
            parts.append("-")
        elif child.tag in (f"{_WORD_NS}br", f"{_WORD_NS}cr"):
            parts.append("\n")
        elif child.tag == f"{_WORD_NS}p" and parts and not parts[-1].endswith("\n"):
            parts.append("\n")  # a text box starts mid-paragraph: give it its own line
        _collect_docx_text(child, parts)
        if child.tag == f"{_WORD_NS}p":
            parts.append("\n")
