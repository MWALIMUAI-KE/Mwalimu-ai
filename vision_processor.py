"""
Mwalimu AI - Visual Processing Layer

Handles:
- Text-based PDFs
- Scanned/image-only PDF pages
- Image-based student scripts
- Page-order preservation
- Vision analysis of handwritten work
- Temporary image cleanup

This module does NOT store API keys. The OpenAI client is supplied by
the calling application, which should obtain credentials from secrets
/environment variables.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class VisualPage:
    """Represents one page extracted from a student script."""

    page_number: int
    image_bytes: bytes
    mime_type: str = "image/png"
    source_name: str = ""
    extracted_text: str = ""
    vision_analysis: str = ""

    @property
    def data_url(self) -> str:
        """Return the page image as a data URL for vision processing."""
        encoded = base64.b64encode(self.image_bytes).decode("utf-8")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass
class VisualDocument:
    """Container for all pages belonging to one uploaded document."""

    source_name: str
    pages: List[VisualPage] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class VisionProcessor:
    """
    Converts PDF/image inputs into ordered visual pages and optionally
    sends those pages to an OpenAI vision-capable model.

    The processor deliberately keeps file handling separate from the
    marking engine so the existing marking functionality can continue
    to work independently.
    """

    def __init__(
        self,
        openai_client=None,
        vision_model: Optional[str] = None,
        render_scale: float = 2.0,
    ):
        self.client = openai_client
        self.vision_model = (
            vision_model
            or os.getenv("OPENAI_VISION_MODEL")
            or "gpt-4.1-mini"
        )
        self.render_scale = render_scale

    # ------------------------------------------------------------------
    # PDF PROCESSING
    # ------------------------------------------------------------------

    def process_pdf(
        self,
        pdf_bytes: bytes,
        source_name: str = "uploaded_script.pdf",
    ) -> VisualDocument:
        """
        Render every PDF page as an image while preserving page order.

        The PDF is processed entirely from memory where possible.
        No uploaded source PDF is permanently written to disk.
        """

        if not pdf_bytes:
            raise ValueError("The supplied PDF is empty.")

        document = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages: List[VisualPage] = []

        try:
            for index, page in enumerate(document):
                page_number = index + 1

                matrix = fitz.Matrix(
                    self.render_scale,
                    self.render_scale,
                )

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image_bytes = pixmap.tobytes("png")

                pages.append(
                    VisualPage(
                        page_number=page_number,
                        image_bytes=image_bytes,
                        mime_type="image/png",
                        source_name=source_name,
                    )
                )
        finally:
            document.close()

        return VisualDocument(
            source_name=source_name,
            pages=pages,
        )

    # ------------------------------------------------------------------
    # IMAGE PROCESSING
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_bytes: bytes,
        source_name: str = "uploaded_script.png",
        mime_type: str = "image/png",
    ) -> VisualDocument:
        """Create a one-page visual document from an uploaded image."""

        if not image_bytes:
            raise ValueError("The supplied image is empty.")

        return VisualDocument(
            source_name=source_name,
            pages=[
                VisualPage(
                    page_number=1,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    source_name=source_name,
                )
            ],
        )

    # ------------------------------------------------------------------
    # VISION ANALYSIS
    # ------------------------------------------------------------------

    def analyze_page(
        self,
        page: VisualPage,
        instructions: Optional[str] = None,
    ) -> str:
        """
        Analyze one page using an OpenAI vision-capable model.

        The model is instructed to preserve mathematical working,
        diagrams, graphs, chemistry notation and uncertainty instead
        of inventing unreadable content.
        """

        if self.client is None:
            raise RuntimeError(
                "An OpenAI client is required for visual analysis."
            )

        prompt = instructions or self.default_analysis_prompt()

        response = self.client.responses.create(
            model=self.vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": page.data_url,
                        },
                    ],
                }
            ],
        )

        result = getattr(response, "output_text", None)

        if not result:
            return ""

        page.vision_analysis = result.strip()
        return page.vision_analysis

    def analyze_document(
        self,
        document: VisualDocument,
        instructions: Optional[str] = None,
    ) -> VisualDocument:
        """
        Analyze all pages sequentially.

        Page order is preserved. A failure on one page does not silently
        change the numbering of the remaining pages.
        """

        for page in document.pages:
            try:
                self.analyze_page(
                    page,
                    instructions=instructions,
                )
            except Exception as exc:
                page.vision_analysis = (
                    f"[VISION_PROCESSING_ERROR: {type(exc).__name__}]"
                )

        return document

    # ------------------------------------------------------------------
    # DEFAULT VISION INSTRUCTIONS
    # ------------------------------------------------------------------

    @staticmethod
    def default_analysis_prompt() -> str:
        return """
You are the visual-processing component of Mwalimu AI, a Kenyan
educational assessment system.

Examine this examination/student-script page carefully.

Your task is to TRANSCRIBE AND DESCRIBE what is actually visible.
Do not invent missing answers, marks, symbols, diagrams or calculations.

Preserve, where readable:

1. Question numbers and sub-question labels.
2. The student's handwritten or printed response.
3. Mathematical calculations and intermediate working.
4. Fractions, indices, roots, equations and algebraic notation.
5. Chemical formulae, equations, charges, state symbols and structures.
6. Graphs, tables, diagrams and labelled figures.
7. Crossed-out work where it affects interpretation.
8. Units and numerical values.
9. Any visible examiner annotations or marks, while clearly
   distinguishing them from the student's work.

For mathematics, use LaTeX where it improves accuracy.

For diagrams and graphs:
- describe the visible structure;
- record labels and plotted information;
- do not guess values that cannot be read.

If something is unclear, explicitly state:
[UNCLEAR]

If something is not visible, state:
[NOT_VISIBLE]

Do not award marks.
Do not create a marking scheme.
Do not correct the student's work.

Your output will be passed to a separate Mwalimu AI marking engine.
""".strip()

    # ------------------------------------------------------------------
    # TEXT EXTRACTION HELPER
    # ------------------------------------------------------------------

    @staticmethod
    def extract_embedded_text(pdf_bytes: bytes) -> str:
        """
        Extract selectable text from a PDF.

        This is deliberately retained as a helper so the visual layer
        can support both ordinary text PDFs and scanned PDFs.
        """

        if not pdf_bytes:
            return ""

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

        try:
            text_parts = []

            for page in document:
                text = page.get_text("text")

                if text:
                    text_parts.append(text)

            return "\n".join(text_parts).strip()

        finally:
            document.close()

    # ------------------------------------------------------------------
    # INPUT CLASSIFICATION
    # ------------------------------------------------------------------

    @staticmethod
    def is_probably_text_pdf(pdf_bytes: bytes) -> bool:
        """
        Determine whether a PDF contains a meaningful amount of
        selectable text.

        This is only a routing heuristic, not a final OCR decision.
        """

        text = VisionProcessor.extract_embedded_text(pdf_bytes)

        # A very small amount of text may simply be page metadata,
        # headers or an image-based document with minimal embedded text.
        return len(text.strip()) >= 80

    # ------------------------------------------------------------------
    # TEMPORARY FILE SAFETY
    # ------------------------------------------------------------------

    @staticmethod
    def temporary_file(
        data: bytes,
        suffix: str = ".bin",
    ):
        """
        Create a temporary file for libraries that require a filesystem
        path.

        The caller should use the returned path within the context
        manager. The file is automatically removed afterwards.
        """

        return tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=True,
        )

    @staticmethod
    def validate_upload(
        file_bytes: bytes,
        filename: str,
        max_size_mb: int = 25,
    ) -> None:
        """Basic upload validation before processing."""

        if not file_bytes:
            raise ValueError("The uploaded file is empty.")

        max_bytes = max_size_mb * 1024 * 1024

        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"File exceeds the {max_size_mb} MB upload limit."
            )

        extension = Path(filename).suffix.lower()

        allowed = {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        if extension not in allowed:
            raise ValueError(
                "Unsupported file type. "
                "Use PDF, PNG, JPG, JPEG or WEBP."
)
