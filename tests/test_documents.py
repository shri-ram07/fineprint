import io
import zipfile

import pytest
from pypdf import PdfReader, PdfWriter

from fineprint import documents
from fineprint.documents import DocumentError, extract_text


def make_pdf(*pages: str) -> bytes:
    """Build a tiny valid PDF with one line of Helvetica text per page ("" = blank page)."""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "",  # page tree, filled in once the page ids are known
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET" if text else ""
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {len(objects)} 0 R"
            " /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    out += trailer.encode()
    return out


def encrypted_pdf(user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf("Rent is due monthly"))))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_docx(document_xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def error_code(kind, data, **kwargs) -> str:
    with pytest.raises(DocumentError) as raised:
        extract_text(kind, data, **kwargs)
    return raised.value.code


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("Tenant pays £900 – on time".encode(), "Tenant pays £900 – on time"),
        ("\ufeffLease".encode(), "Lease"),
        ("Lease £900".encode("utf-16"), "Lease £900"),
        ("Caf\xe9 lease".encode("cp1252"), "Café lease"),
    ],
    ids=["utf-8", "utf-8-bom", "utf-16-bom", "cp1252-fallback"],
)
def test_text_encodings(data, expected):
    assert extract_text("txt", data) == expected


def test_collapses_runs_of_blank_lines():
    assert extract_text("md", b"\n\nClause 1\n\n\n\n\nClause 2\n\n") == "Clause 1\n\nClause 2"


def test_docx_paragraphs_tabs_and_breaks():
    xml = (
        f"<w:document {W}><w:body>"
        "<w:p><w:r><w:t>1. Term</w:t><w:tab/><w:t>12 months</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Line one</w:t><w:br/><w:t>Line two</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    assert extract_text("docx", make_docx(xml)) == "1. Term\t12 months\nLine one\nLine two"


def test_docx_decompressed_size_is_bounded(monkeypatch):
    monkeypatch.setattr(documents, "MAX_XML_BYTES", 1_000)
    bomb = make_docx(f"<w:document {W}>" + " " * 5_000 + "</w:document>")
    assert len(bomb) < 1_000  # compresses well: only the decompressed size can catch it
    assert error_code("docx", bomb) == "too_large"


def test_docx_without_document_part_is_corrupt():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
    assert error_code("docx", buffer.getvalue()) == "corrupt"


def test_docx_with_damaged_compressed_data_is_corrupt():
    damaged = bytearray(make_docx(f"<w:document {W}>" + "<w:p/>" * 200 + "</w:document>"))
    damaged[80:120] = b"\xff" * 40
    assert error_code("docx", bytes(damaged)) == "corrupt"


def test_pdf_text_is_extracted():
    assert "Rent is due monthly" in extract_text("pdf", make_pdf("Rent is due monthly"))


def test_pdf_blank_pages_are_marked():
    text = extract_text("pdf", make_pdf("Signed by the Tenant", ""))
    assert "Signed by the Tenant" in text
    assert "[Page 2: no readable text" in text


def test_scanned_pdf_has_no_text_layer():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    assert error_code("pdf", buffer.getvalue()) == "no_text_layer"


def test_owner_password_pdf_opens_without_a_password():
    assert "Rent is due monthly" in extract_text("pdf", encrypted_pdf(user_password=""))


def test_user_password_pdf_is_encrypted():
    assert error_code("pdf", encrypted_pdf(user_password="secret")) == "encrypted"


def test_garbage_pdf_is_corrupt():
    assert error_code("pdf", b"%PDF-1.7 garbage") == "corrupt"


@pytest.mark.parametrize(
    ("kind", "data"),
    [("txt", b"A" * 50), ("pdf", make_pdf("A" * 50))],
)
def test_text_longer_than_limit_is_too_large(kind, data):
    assert error_code(kind, data, max_chars=10) == "too_large"


@pytest.mark.parametrize("data", [b"", b"\n\n  \n"])
def test_empty_input(data):
    assert error_code("txt", data) == "empty"
