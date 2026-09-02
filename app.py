import os
import io
import re
import json
import base64
from typing import List, Dict, Any

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI


# ============================================================
# MWALIMU AI — MVP3.1 VISUAL MARKING ENGINE
# ============================================================
#
# GOLDEN MVP3 PRINCIPLE:
# The uploaded PDF is the visual source of truth.
#
# The original PDF pages are rendered and displayed directly.
# AI never reconstructs the original examination paper.
#
# MVP3.1 — SINGLE CONTROLLED CHANGE:
# Each detected question now receives a visual bounding box.
# The original question image is cropped from the original
# rendered page and displayed immediately before its own
# AI working / answer / marking scheme.
#
# EVERYTHING ELSE REMAINS AS IN MVP3.
# ============================================================


st.set_page_config(
    page_title="Mwalimu AI — Visual Marking",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mwalimu AI")
st.caption(
    "MVP3 Visual Marking Engine — Original question preserved, "
    "AI workings and marking scheme added underneath."
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

MAX_PAGE_DIMENSION = 1800
JPEG_QUALITY = 88


# ------------------------------------------------------------
# OPENAI CLIENT
# ------------------------------------------------------------

api_key = None

try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    pass

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Add it to Streamlit Secrets before running the app."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if "page_images" not in st.session_state:
    st.session_state.page_images = []

if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""


# ------------------------------------------------------------
# PDF → PAGE IMAGES
# ------------------------------------------------------------

def render_pdf_pages(pdf_bytes: bytes) -> List[bytes]:
    """
    Render every original PDF page to a high-resolution JPEG.

    The rendered image is the visual source of truth for the
    displayed examination paper.

    Nothing is reconstructed.
    """

    pages = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page in document:

        rect = page.rect

        # Render at approximately 1800px on the longest side.
        scale = MAX_PAGE_DIMENSION / max(
            rect.width,
            rect.height
        )

        # Prevent unnecessary enlargement.
        scale = min(scale, 2.5)

        matrix = fitz.Matrix(
            scale,
            scale
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image_bytes = pix.tobytes(
            "jpeg",
            jpg_quality=JPEG_QUALITY
        )

        pages.append(image_bytes)

    document.close()

    return pages


# ------------------------------------------------------------
# OPTIONAL TEXT EXTRACTION
# ------------------------------------------------------------

def extract_page_texts(
    pdf_bytes: bytes
) -> List[str]:
    """
    Extract text only for supporting AI reasoning.

    IMPORTANT:
    This text is NEVER used as the displayed version
    of the original question.
    """

    texts = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page in document:

        try:
            text = page.get_text("text")
        except Exception:
            text = ""

        texts.append(
            text.strip()
        )

    document.close()

    return texts


# ------------------------------------------------------------
# IMAGE → DATA URL
# ------------------------------------------------------------

def image_to_data_url(
    image_bytes: bytes
) -> str:

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ------------------------------------------------------------
# PAGE ANALYSIS PROMPT
# ------------------------------------------------------------

PAGE_ANALYSIS_PROMPT = """
You are the visual analysis engine for Mwalimu AI.

You are analysing an ORIGINAL Kenyan secondary-school
mathematics examination paper page.

CRITICAL RULE:

The original page image is the authoritative source.

DO NOT invent, redraw, simplify or replace diagrams.

Your task is to identify what is actually visible on the page.

In addition to identifying each question, identify the approximate
bounding box of each question on the supplied page image.

The bounding box must cover the COMPLETE visible question,
including its text, mathematical expressions, diagrams, tables,
graphs, constructions and answer choices where applicable.

Use pixel coordinates relative to the supplied image:

x = horizontal position from the left
y = vertical position from the top
width = width of the box
height = height of the box

Return JSON only.

Required JSON structure:

{
  "page_number": integer,
  "image_width": integer,
  "image_height": integer,
  "questions": [
    {
      "number": "question number",
      "visible_text_summary": "faithful summary of what is visible",

      "bbox": {
        "x": integer,
        "y": integer,
        "width": integer,
        "height": integer
      },

      "has_diagram": true/false,
      "diagram_description": "describe the actual visible diagram, if any",
      "has_graph": true/false,
      "has_table": true/false,
      "has_construction": true/false,
      "has_special_math_notation": true/false,

      "important_visual_features": [
        "list important visible features"
      ]
    }
  ],

  "visual_warnings": [
    "anything that is difficult to read or visually ambiguous"
  ]
}

IMPORTANT BOUNDING-BOX RULES:

1. Coordinates must refer to the supplied page image.
2. Do not invent coordinates for questions that are not visible.
3. Include all visible parts of a question in its bounding box.
4. Include diagrams belonging to the question.
5. Include tables, graphs and constructions belonging to the question.
6. If two questions are very close together, leave a small amount
   of surrounding context rather than cutting off mathematical
   information.
7. Do not solve the questions here.
8. Do not manufacture missing information.
9. If a diagram is present, explicitly say so.
"""


# ------------------------------------------------------------
# QUESTION SOLUTION PROMPT
# ------------------------------------------------------------

SOLUTION_PROMPT = """
You are the senior mathematics examiner for Mwalimu AI.

The original examination page is supplied as an image.

The image is authoritative.

You must solve the question using the actual visible information.

IMPORTANT:

1. Do NOT recreate the original question.
2. Do NOT omit a diagram from your reasoning.
3. If a diagram is essential, explicitly refer to the labels,
   dimensions, angles, coordinates or other information visible
   in that diagram.
4. Show complete mathematical working.
5. Give a final answer.
6. Give a concise marking scheme.
7. Do not award marks for unsupported invented work.
8. If the image is unclear, say so instead of guessing.

Return JSON only:

{
  "question_number": "...",
  "method": "brief description of method",
  "working": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "final_answer": "...",
  "marking_scheme": [
    {
      "marks": integer,
      "point": "what earns the mark"
    }
  ],
  "visual_dependency": "none | low | medium | high",
  "visual_check": "explain what visual information was used",
  "confidence": "high | medium | low",
  "warning": ""
}
"""


# ------------------------------------------------------------
# SAFE JSON PARSER
# ------------------------------------------------------------

def parse_json_response(
    text: str
) -> Dict[str, Any]:

    if not text:
        return {}

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        try:
            return json.loads(
                text[start:end + 1]
            )
        except Exception:
            return {}

    return {}


# ------------------------------------------------------------
# NORMALISE BOUNDING BOX
# ------------------------------------------------------------

def normalise_bbox(
    bbox: Any,
    image_width: int,
    image_height: int
) -> Dict[str, int] | None:
    """
    Safely constrain an AI-produced bounding box to the
    actual image dimensions.

    This function never changes the original page.
    It only controls which portion is displayed as the
    question crop.
    """

    if not isinstance(bbox, dict):
        return None

    try:
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        width = int(bbox.get("width", 0))
        height = int(bbox.get("height", 0))
    except Exception:
        return None

    if image_width <= 0 or image_height <= 0:
        return None

    # Reject clearly invalid boxes.
    if width <= 5 or height <= 5:
        return None

    # Clamp starting coordinates.
    x = max(
        0,
        min(x, image_width - 1)
    )

    y = max(
        0,
        min(y, image_height - 1)
    )

    # Clamp dimensions.
    width = min(
        width,
        image_width - x
    )

    height = min(
        height,
        image_height - y
    )

    if width <= 5 or height <= 5:
        return None

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }


# ------------------------------------------------------------
# CROP ORIGINAL QUESTION
# ------------------------------------------------------------

def crop_original_question(
    image_bytes: bytes,
    bbox: Dict[str, int]
) -> bytes | None:
    """
    Crop the question directly from the ORIGINAL rendered page.

    This is not OCR reconstruction.

    The pixels displayed here originate from the original
    examination page image.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        image_width, image_height = image.size

        safe_bbox = normalise_bbox(
            bbox,
            image_width,
            image_height
        )

        if not safe_bbox:
            return None

        x = safe_bbox["x"]
        y = safe_bbox["y"]
        width = safe_bbox["width"]
        height = safe_bbox["height"]

        # Add a small visual margin around the original question.
        margin_x = max(
            12,
            int(width * 0.015)
        )

        margin_y = max(
            12,
            int(height * 0.015)
        )

        left = max(
            0,
            x - margin_x
        )

        top = max(
            0,
            y - margin_y
        )

        right = min(
            image_width,
            x + width + margin_x
        )

        bottom = min(
            image_height,
            y + height + margin_y
        )

        cropped = image.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

        output = io.BytesIO()

        cropped.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY
        )

        return output.getvalue()

    except Exception:
        return None


# ------------------------------------------------------------
# CALL VISION MODEL
# ------------------------------------------------------------

def analyse_page(
    image_bytes: bytes,
    page_number: int,
    extracted_text: str
) -> Dict[str, Any]:

    image_url = image_to_data_url(
        image_bytes
    )

    supporting_text = extracted_text[:12000]

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            PAGE_ANALYSIS_PROMPT
                            + "\n\nPAGE NUMBER: "
                            + str(page_number)
                            + "\n\nOPTIONAL EXTRACTED TEXT "
                              "(supporting evidence only):\n"
                            + supporting_text
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    result = parse_json_response(
        response.output_text
    )

    return result


# ------------------------------------------------------------
# SOLVE ONE QUESTION
# ------------------------------------------------------------

def solve_question(
    image_bytes: bytes,
    page_number: int,
    question_number: str,
    question_summary: str,
    diagram_description: str,
    extracted_text: str
) -> Dict[str, Any]:

    image_url = image_to_data_url(
        image_bytes
    )

    prompt = SOLUTION_PROMPT + f"""

PAGE NUMBER: {page_number}

QUESTION NUMBER: {question_number}

QUESTION SUMMARY:
{question_summary}

DIAGRAM DESCRIPTION:
{diagram_description}

EXTRACTED TEXT FROM THE ORIGINAL PDF:
{extracted_text[:16000]}

Remember:

The displayed original page is authoritative.

Use the image to verify the mathematics before solving.
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    result = parse_json_response(
        response.output_text
    )

    if not result:

        return {
            "question_number": question_number,
            "method": "",
            "working": [],
            "final_answer": response.output_text,
            "marking_scheme": [],
            "visual_dependency": "unknown",
            "visual_check": "",
            "confidence": "low",
            "warning": (
                "Model response could not be parsed as JSON."
            )
        }

    return result


# ------------------------------------------------------------
# VISUAL COMPLETENESS CHECK
# ------------------------------------------------------------

def completeness_check(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    expected = []

    for page_result in results:

        for question in page_result.get(
            "questions",
            []
        ):

            number = str(
                question.get(
                    "number",
                    ""
                )
            ).strip()

            if number:
                expected.append(number)

    solved = []

    for page_result in results:

        for solution in page_result.get(
            "solutions",
            []
        ):

            number = str(
                solution.get(
                    "question_number",
                    ""
                )
            ).strip()

            if number:
                solved.append(number)

    expected_set = set(expected)
    solved_set = set(solved)

    missing = sorted(
        expected_set - solved_set,
        key=lambda x: (
            int(
                re.sub(
                    r"\D",
                    "",
                    x
                ) or 999
            ),
            x
        )
    )

    return {
        "questions_detected": len(
            expected_set
        ),
        "questions_solved": len(
            solved_set
        ),
        "missing_questions": missing,
        "complete": len(missing) == 0
    }


# ------------------------------------------------------------
# DISPLAY ORIGINAL PAGE
# ------------------------------------------------------------

def display_original_page(
    image_bytes: bytes,
    page_number: int
):

    st.image(
        image_bytes,
        caption=(
            f"Original examination page "
            f"{page_number}"
        ),
        width="stretch"
    )

    st.caption(
        "🔒 Original page preserved — this image is not "
        "reconstructed from AI-generated text."
    )


# ------------------------------------------------------------
# DISPLAY QUESTION + SOLUTION
# ------------------------------------------------------------

def display_question_with_solution(
    image_bytes: bytes,
    question: Dict[str, Any],
    solution: Dict[str, Any]
):
    """
    DISPLAY-ONLY CHANGE FOR MVP3.1.

    The question image is cropped directly from the original
    rendered page.

    No OCR text is used to recreate the question.
    """

    question_number = str(
        question.get(
            "number",
            solution.get(
                "question_number",
                "Unknown"
            )
        )
    )

    st.markdown(
        f"### 📌 Question {question_number}"
    )

    bbox = question.get(
        "bbox"
    )

    question_crop = None

    if bbox:
        question_crop = crop_original_question(
            image_bytes,
            bbox
        )

    if question_crop:

        st.image(
            question_crop,
            caption=(
                f"Original Question "
                f"{question_number}"
            ),
            width="stretch"
        )

        st.caption(
            "🔒 Original question image preserved — "
            "cropped directly from the scanned page."
        )

    else:

        st.warning(
            "The original question region could not be "
            "isolated reliably. The full original page above "
            "remains the authoritative source."
        )

    # --------------------------------------------------------
    # AI SOLUTION
    # --------------------------------------------------------

    st.markdown(
        f"#### ✏️ Question {question_number} — "
        "AI Marking Scheme"
    )

    confidence = solution.get(
        "confidence",
        ""
    )

    if confidence:

        if confidence == "high":

            st.success(
                f"Confidence: {confidence}"
            )

        elif confidence == "medium":

            st.warning(
                f"Confidence: {confidence}"
            )

        else:

            st.error(
                f"Confidence: {confidence}"
            )

    visual_dependency = solution.get(
        "visual_dependency",
        "none"
    )

    if visual_dependency in (
        "medium",
        "high"
    ):

        st.info(
            f"👁️ Visual dependency: "
            f"**{visual_dependency}**\n\n"
            + solution.get(
                "visual_check",
                ""
            )
        )

    method = solution.get(
        "method"
    )

    if method:

        st.markdown(
            "**Method**"
        )

        st.write(
            method
        )

    st.markdown(
        "**Working**"
    )

    working = solution.get(
        "working",
        []
    )

    if isinstance(
        working,
        list
    ):

        for index, step in enumerate(
            working,
            start=1
        ):

            st.markdown(
                f"**{index}.** {step}"
            )

    else:

        st.write(
            working
        )

    final_answer = solution.get(
        "final_answer"
    )

    if final_answer:

        st.markdown(
            "**Final Answer**"
        )

        st.success(
            str(final_answer)
        )

    marking = solution.get(
        "marking_scheme",
        []
    )

    if marking:

        st.markdown(
            "**Mark Allocation**"
        )

        for item in marking:

            marks = item.get(
                "marks",
                ""
            )

            point = item.get(
                "point",
                ""
            )

            st.markdown(
                f"- **{marks} mark(s):** "
                f"{point}"
            )

    warning = solution.get(
        "warning"
    )

    if warning:

        st.warning(
            f"⚠️ Examiner/AI warning: "
            f"{warning}"
        )


# ------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload the original examination paper",
    type=["pdf"],
    help=(
        "Upload the original PDF. Mwalimu AI will preserve "
        "its visual appearance and analyse the pages directly."
    )
)


if uploaded:

    pdf_bytes = uploaded.getvalue()

    st.success(
        f"Loaded: {uploaded.name}"
    )

    if st.button(
        "🔍 Scan Original Paper & Generate Marking Scheme",
        type="primary",
        use_container_width=True
    ):

        st.session_state.analysis_results = []

        st.session_state.page_images = []

        st.session_state.paper_name = (
            uploaded.name
        )

        # ----------------------------------------------------
        # STEP 1 — RENDER ORIGINAL PDF
        # ----------------------------------------------------

        with st.spinner(
            "Rendering the original PDF pages..."
        ):

            try:

                page_images = render_pdf_pages(
                    pdf_bytes
                )

                page_texts = extract_page_texts(
                    pdf_bytes
                )

            except Exception as exc:

                st.error(
                    "Could not read the PDF."
                )

                st.exception(
                    exc
                )

                st.stop()

        st.session_state.page_images = (
            page_images
        )

        st.success(
            f"Rendered {len(page_images)} original pages."
        )

        # ----------------------------------------------------
        # STEP 2 — VISUAL ANALYSIS
        # ----------------------------------------------------

        progress = st.progress(
            0
        )

        all_results = []

        for index, image_bytes in enumerate(
            page_images
        ):

            page_number = index + 1

            with st.status(
                f"Visually analysing page "
                f"{page_number}...",
                expanded=False
            ):

                try:

                    page_analysis = analyse_page(
                        image_bytes=image_bytes,
                        page_number=page_number,
                        extracted_text=page_texts[index]
                    )

                except Exception as exc:

                    page_analysis = {
                        "page_number": page_number,
                        "questions": [],
                        "visual_warnings": [
                            str(exc)
                        ]
                    }

            page_result = {
                "page_number": page_number,
                "questions": page_analysis.get(
                    "questions",
                    []
                ),
                "visual_warnings": page_analysis.get(
                    "visual_warnings",
                    []
                ),
                "solutions": []
            }

            # ------------------------------------------------
            # STEP 3 — SOLVE QUESTIONS ON THIS PAGE
            # ------------------------------------------------

            questions = page_result[
                "questions"
            ]

            for question in questions:

                question_number = str(
                    question.get(
                        "number",
                        ""
                    )
                ).strip()

                if not question_number:
                    continue

                with st.status(
                    f"Solving Question "
                    f"{question_number}...",
                    expanded=False
                ):

                    try:

                        solution = solve_question(
                            image_bytes=image_bytes,
                            page_number=page_number,
                            question_number=question_number,
                            question_summary=question.get(
                                "visible_text_summary",
                                ""
                            ),
                            diagram_description=question.get(
                                "diagram_description",
                                ""
                            ),
                            extracted_text=page_texts[index]
                        )

                    except Exception as exc:

                        solution = {
                            "question_number":
                                question_number,
                            "method": "",
                            "working": [],
                            "final_answer": "",
                            "marking_scheme": [],
                            "visual_dependency":
                                "unknown",
                            "visual_check": "",
                            "confidence": "low",
                            "warning": str(exc)
                        }

                page_result[
                    "solutions"
                ].append(
                    solution
                )

            all_results.append(
                page_result
            )

            progress.progress(
                int(
                    (
                        (index + 1)
                        / len(page_images)
                    )
                    * 100
                )
            )

        st.session_state.analysis_results = (
            all_results
        )

        st.success(
            "Visual scan and marking-scheme generation completed."
        )


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results = st.session_state.analysis_results


if results:

    st.divider()

    st.header(
        "📄 Original Paper + Question-by-Question "
        "Marking Scheme"
    )

    st.info(
        "The original paper pages below are the actual "
        "scanned pages. Mwalimu AI does not redraw the "
        "questions. Each question is taken directly from "
        "the original page and its generated working is "
        "shown immediately underneath."
    )

    # --------------------------------------------------------
    # COMPLETENESS
    # --------------------------------------------------------

    check = completeness_check(
        results
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Questions detected",
            check[
                "questions_detected"
            ]
        )

    with col2:

        st.metric(
            "Questions solved",
            check[
                "questions_solved"
            ]
        )

    with col3:

        st.metric(
            "Missing",
            len(
                check[
                    "missing_questions"
                ]
            )
        )

    if check[
        "missing_questions"
    ]:

        st.warning(
            "Questions not yet solved: "
            + ", ".join(
                check[
                    "missing_questions"
                ]
            )
        )

    else:

        st.success(
            "✅ Completeness check passed: "
            "all detected questions have a solution."
        )

    # --------------------------------------------------------
    # PAGE-BY-PAGE OUTPUT
    # --------------------------------------------------------

    for page_result in results:

        page_number = page_result[
            "page_number"
        ]

        st.divider()

        st.header(
            f"📄 Page {page_number}"
        )

        # ----------------------------------------------------
        # ORIGINAL FULL PAGE FIRST
        # ----------------------------------------------------

        if (
            page_number
            <= len(
                st.session_state.page_images
            )
        ):

            original_page = (
                st.session_state.page_images[
                    page_number - 1
                ]
            )

            display_original_page(
                original_page,
                page_number
            )

        # ----------------------------------------------------
        # VISUAL WARNINGS
        # ----------------------------------------------------

        warnings = page_result.get(
            "visual_warnings",
            []
        )

        if warnings:

            with st.expander(
                "👁️ Visual analysis notes"
            ):

                for warning in warnings:

                    st.warning(
                        warning
                    )

        # ----------------------------------------------------
        # QUESTION-BY-QUESTION OUTPUT
        # ----------------------------------------------------

        questions = page_result.get(
            "questions",
            []
        )

        solutions = page_result.get(
            "solutions",
            []
        )

        # Match solutions to their original question
        # rather than relying blindly on list position.

        solution_map = {}

        for solution in solutions:

            number = str(
                solution.get(
                    "question_number",
                    ""
                )
            ).strip()

            if number:

                solution_map[
                    number
                ] = solution

        if questions:

            for question in questions:

                number = str(
                    question.get(
                        "number",
                        ""
                    )
                ).strip()

                if not number:
                    continue

                solution = solution_map.get(
                    number
                )

                if solution is None:

                    solution = {
                        "question_number":
                            number,
                        "method": "",
                        "working": [],
                        "final_answer": "",
                        "marking_scheme": [],
                        "visual_dependency":
                            "unknown",
                        "visual_check": "",
                        "confidence": "low",
                        "warning":
                            "No solution was generated."
                    }

                st.divider()

                display_question_with_solution(
                    image_bytes=original_page,
                    question=question,
                    solution=solution
                )

        else:

            st.caption(
                "No numbered questions were detected "
                "on this page."
            )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    st.divider()

    st.header(
        "📊 Visual Marking Report"
    )

    diagram_questions = []

    for page_result in results:

        for question in page_result.get(
            "questions",
            []
        ):

            if question.get(
                "has_diagram",
                False
            ):

                diagram_questions.append(
                    str(
                        question.get(
                            "number",
                            ""
                        )
                    )
                )

    if diagram_questions:

        st.success(
            "Diagram/figure questions detected: "
            + ", ".join(
                diagram_questions
            )
        )

    else:

        st.info(
            "No diagram-dependent questions were detected "
            "by the visual analyser."
        )

    # --------------------------------------------------------
    # DOWNLOADABLE ANALYSIS JSON
    # --------------------------------------------------------

    json_output = json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        "⬇️ Download AI analysis (JSON)",
        data=json_output,
        file_name=(
            "mwalimu_ai_visual_marking_analysis.json"
        ),
        mime="application/json",
        use_container_width=True
    )


else:

    st.markdown(
        """
        ### How this MVP3 version works

        **1. Upload the original paper**

        **2. Mwalimu AI renders the original pages**

        **3. Vision analysis reads the actual page**

        **4. The original page remains untouched**

        **5. AI generates the mathematical working**

        **6. Each original question is shown with its
        working immediately underneath**

        This is deliberately different from an OCR-only pipeline.
        """
    )
