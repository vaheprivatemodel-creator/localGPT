"""
Document-to-Markdown converter.

Despite the legacy module name (`pdf_converter`), this now handles **PDF and
DOCX** through docling, plus scanned-PDF OCR via macOS Vision Framework
(``OcrMacOptions``). The class is still named ``PDFConverter`` so existing
import sites (`IndexingPipeline`) keep working unchanged — but its
``convert_to_markdown`` method routes by extension.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import fitz  # PyMuPDF for quick text inspection


_DOCX_EXTS = {".docx"}
_PDF_EXTS = {".pdf"}


class PDFConverter:
    """Convert PDF/DOCX files to structured Markdown via docling.

    Three converters are pre-initialised so we never pay model-load latency
    on the per-document hot path:

    * ``converter_no_ocr``  – PDFs that already have a text layer
    * ``converter_ocr``     – scanned PDFs (forced full-page OCR via Vision)
    * ``converter_docx``    – Word documents
    """

    def __init__(self):
        try:
            pipeline_no_ocr = PdfPipelineOptions()
            pipeline_no_ocr.do_ocr = False
            self.converter_no_ocr = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_no_ocr)}
            )

            pipeline_ocr = PdfPipelineOptions()
            pipeline_ocr.do_ocr = True
            pipeline_ocr.ocr_options = OcrMacOptions(force_full_page_ocr=True)
            self.converter_ocr = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_ocr)}
            )

            # DOCX has no OCR/pipeline knobs to tune; docling handles it natively.
            self.converter_docx = DocumentConverter(allowed_formats=[InputFormat.DOCX])

            print("docling DocumentConverter(s) initialized (PDF+OCR, PDF no-OCR, DOCX).")
        except Exception as e:
            print(f"Error initializing docling DocumentConverter(s): {e}")
            self.converter_no_ocr = None
            self.converter_ocr = None
            self.converter_docx = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pdf_has_text(path: str) -> bool:
        """Cheap heuristic: returns True when at least one PDF page already
        exposes a text layer, in which case OCR can be skipped."""
        try:
            doc = fitz.open(path)
            for page in doc:
                if page.get_text("text").strip():
                    return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert_to_markdown(self, file_path: str) -> List[Tuple[str, Dict[str, Any], Any]]:
        """Convert a PDF or DOCX file to a single Markdown string.

        Returns a list with one ``(markdown, metadata, docling_doc)`` tuple to
        match the existing IndexingPipeline contract. The third element is the
        underlying ``DoclingDocument`` so downstream chunkers (e.g. the docling
        token chunker) can walk the element tree instead of re-parsing text.
        """
        if not (self.converter_no_ocr and self.converter_ocr and self.converter_docx):
            print("docling converters not available. Skipping conversion.")
            return []

        ext = os.path.splitext(file_path)[1].lower()
        if ext in _DOCX_EXTS:
            return self._convert_docx(file_path)
        if ext in _PDF_EXTS:
            return self._convert_pdf(file_path)

        print(f"⚠️  Unsupported file type '{ext}' for '{file_path}'. Skipping.")
        return []

    # ------------------------------------------------------------------
    # Internal converters
    # ------------------------------------------------------------------
    def _convert_pdf(self, pdf_path: str) -> List[Tuple[str, Dict[str, Any], Any]]:
        use_ocr = not self._pdf_has_text(pdf_path)
        converter = self.converter_ocr if use_ocr else self.converter_no_ocr
        ocr_msg = "(OCR enabled)" if use_ocr else "(no OCR)"

        print(f"Converting PDF {pdf_path} via docling {ocr_msg}...")
        try:
            result = converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            metadata = {"source": pdf_path, "mime_type": "application/pdf", "ocr_used": use_ocr}
            print(f"Successfully converted {pdf_path} with docling {ocr_msg}.")
            return [(markdown_content, metadata, result.document)]
        except Exception as e:
            print(f"Error processing PDF {pdf_path} with docling: {e}")
            return []

    def _convert_docx(self, docx_path: str) -> List[Tuple[str, Dict[str, Any], Any]]:
        print(f"Converting DOCX {docx_path} via docling...")
        try:
            result = self.converter_docx.convert(docx_path)
            markdown_content = result.document.export_to_markdown()
            metadata = {
                "source": docx_path,
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "ocr_used": False,
            }
            print(f"Successfully converted {docx_path}.")
            return [(markdown_content, metadata, result.document)]
        except Exception as e:
            print(f"Error processing DOCX {docx_path} with docling: {e}")
            return []
