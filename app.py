import os
import io
import re
import json
import base64
from typing import List, Dict, Any

import streamlit as st
import fitz
from PIL import Image
from openai import OpenAI


# ============================================================
# MWALIMU AI — MVP3.2 CLEAN MARKING SCHEME TEST
# ============================================================
#
# ORIGINAL PDF = VISUAL SOURCE OF TRUTH
#
# LOW CONFIDENCE IS ONLY ALLOWED FOR:
#   1. Genuine mathematical error
#   2. Genuine essential visual ambiguity
#
# Technical/API/JSON/formatting issues NEVER automatically
# produce LOW confidence.
#
# This version does NOT add construction or graph generation yet.
# ============================================================


st.set_page_config(
    page_title="Mwalimu AI — Visual Marking",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mwalimu AI")

st.caption(
    "MVP3.2 Visual Marking Engine — Original question preserved, "
    "AI workings and marking scheme added underneath."
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna"
)

MAX_PAGE_DIMENSION = 1800
JPEG_QUALITY = 88


# ============================================================
# OPENAI
# ============================================================

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
        "Add it to Streamlit Secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

if "page_images" not in st.session_state:
    st.session_state.page_images = []

if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""


# ============================================================
# PDF RENDERING
# ============================================================

def render_pdf_pages(pdf_bytes: bytes) -> List[bytes]:

    pages = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page in document:

        rect = page.rect

        scale = MAX_PAGE_DIMENSION / max(
            rect.width,
            rect.height
        )

        scale = min(
            scale,
            2.5
        )

        pix = page.get_pixmap(
            matrix=fitz.Matrix(
                scale,
                scale
            ),
            alpha=False
        )

        pages.append(
            pix.tobytes(
                "jpeg",
                jpg_quality=JPEG_QUALITY
            )
        )

    document.close()

    return pages


# ============================================================
# OPTIONAL PDF TEXT
# ============================================================

def extract_page_texts(
    pdf_bytes: bytes
) -> List[str]:

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


# ============================================================
# IMAGE → DATA URL
# ============================================================

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


# ============================================================
# PAGE ANALYSIS
# ============================================================

PAGE_ANALYSIS_PROMPT = r"""
You are the visual examination-paper analyser for Mwalimu AI.

The supplied image is the ORIGINAL mathematics examination page.

The image is authoritative.

Identify EVERY visible numbered question.

Do not solve the questions.

For every question provide an approximate bounding box in pixels.

The box must include the COMPLETE visible question, including:

- question text
- fractions
- powers
- roots
- mathematical symbols
- diagrams
- graphs
- constructions
- tables
- answer choices

Return JSON only:

{
  "page_number": 1,
  "image_width": 0,
  "image_height": 0,
  "questions": [
    {
      "number": "1",
      "visible_text_summary": "...",

      "bbox": {
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0
      },

      "has_diagram": false,
      "diagram_description": "",

      "has_graph": false,
      "has_table": false,
      "has_construction": false,
      "has_special_math_notation": false,

      "important_visual_features": []
    }
  ],

  "visual_warnings": []
}

Do not invent missing information.

Do not solve questions.

If a diagram is visible, identify it explicitly.
"""


# ============================================================
# SOLUTION PROMPT
# ============================================================

SOLUTION_PROMPT = r"""
You are the senior mathematics examiner for Mwalimu AI.

Solve the question using the ORIGINAL examination page image.

The image is authoritative.

Show complete mathematical working.

Give:

1. Method
2. Step-by-step working
3. Final answer
4. Concise marking scheme

If a diagram is involved, use the ACTUAL visible labels,
dimensions, angles, coordinates and other information.

Do not invent missing information.

============================================================
MATHEMATICAL NOTATION
============================================================

Use proper LaTeX.

Examples:

\frac{a}{b}

\sqrt{x}

x^2

x^3

x_n

\pm

\times

\div

30^\circ

\pi

\sin\theta

\cos\theta

\tan\theta

\vec{AB}

\parallel

\perp

\therefore

\sim

\cong

Use \( ... \) for inline mathematics.

Use \[ ... \] for displayed mathematics.

Do not use plain-text substitutes when proper notation is
appropriate.

============================================================
CONFIDENCE
============================================================

Do NOT use LOW confidence merely because:

- the question is difficult;
- a diagram exists;
- advanced mathematics is used;
- LaTeX is used;
- the model is generally uncertain;
- JSON formatting is imperfect;
- a technical issue occurred.

LOW confidence is allowed ONLY when there is:

- a genuine mathematical error;
OR
- genuine essential visual ambiguity.

If the mathematics is correct and the necessary visual
information is readable, confidence should normally be HIGH.

Return JSON only:

{
  "question_number": "...",

  "method": "...",

  "working": [
    "...",
    "...",
    "..."
  ],

  "final_answer": "...",

  "marking_scheme": [
    {
      "marks": 1,
      "point": "..."
    }
  ],

  "visual_dependency": "none | low | medium | high",

  "visual_check": "...",

  "confidence": "high | medium | low",

  "confidence_reason": "...",

  "warning": ""
}
"""


# ============================================================
# VERIFICATION PROMPT
# ============================================================

VERIFICATION_PROMPT = r"""
You are the FINAL verification examiner for Mwalimu AI.

Compare the generated mathematics solution against the ORIGINAL
examination page image.

Check:

1. Was the question interpreted correctly?
2. Were the visible numbers correct?
3. Were mathematical symbols interpreted correctly?
4. Were diagram labels and measurements used correctly?
5. Is the mathematical working correct?
6. Does the final answer follow from the working?
7. Does the marking scheme match the working?

============================================================
CRITICAL CONFIDENCE RULE
============================================================

LOW confidence MUST NOT be used merely because:

- the question is difficult;
- a diagram exists;
- advanced mathematics is involved;
- LaTeX is present;
- the model has general uncertainty;
- formatting is imperfect.

LOW is ONLY permitted when:

A. There is a genuine mathematical error.

OR

B. Essential visual information is genuinely unreadable
or ambiguous.

If the solution is mathematically correct and the visual
information is sufficiently clear, return HIGH.

Return JSON only:

{
  "verified": true,
  "confidence": "high | medium | low",
  "reason": "...",
  "mathematical_error": false,
  "visual_ambiguity": false,
  "recommended_action": "accept | review"
}
"""


# ============================================================
# JSON PARSER
# ============================================================

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
            pass

    return {}


# ============================================================
# SAFE BOUNDING BOX
# ============================================================

def normalise_bbox(
    bbox: Any,
    image_width: int,
    image_height: int
):

    if not isinstance(
        bbox,
        dict
    ):
        return None

    try:

        x = int(
            bbox.get(
                "x",
                0
            )
        )

        y = int(
            bbox.get(
                "y",
                0
            )
        )

        width = int(
            bbox.get(
                "width",
                0
            )
        )

        height = int(
            bbox.get(
                "height",
                0
            )
        )

    except Exception:

        return None

    if width <= 5 or height <= 5:
        return None

    x = max(
        0,
        min(
            x,
            image_width - 1
        )
    )

    y = max(
        0,
        min(
            y,
            image_height - 1
        )
    )

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


# ============================================================
# ORIGINAL QUESTION CROP
# ============================================================

def crop_original_question(
    image_bytes: bytes,
    bbox: Dict[str, int]
):

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        ).convert("RGB")

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

        margin_x = max(
            12,
            int(width * 0.015)
        )

        margin_y = max(
            12,
            int(height * 0.015)
        )

        crop = image.crop(
            (
                max(
                    0,
                    x - margin_x
                ),
                max(
                    0,
                    y - margin_y
                ),
                min(
                    image_width,
                    x + width + margin_x
                ),
                min(
                    image_height,
                    y + height + margin_y
                )
            )
        )

        output = io.BytesIO()

        crop.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY
        )

        return output.getvalue()

    except Exception:

        return None


# ============================================================
# ANALYSE PAGE
# ============================================================

def analyse_page(
    image_bytes: bytes,
    page_number: int,
    extracted_text: str
):

    response = client.responses.create(

        model=MODEL,

        input=[

            {
                "role": "user",

                "content": [

                    {
                        "type": "input_text",

                        "text":
                            PAGE_ANALYSIS_PROMPT
                            + "\n\nPAGE NUMBER:\n"
                            + str(page_number)
                            + "\n\nSUPPORTING PDF TEXT:\n"
                            + extracted_text[:12000]
                    },

                    {
                        "type": "input_image",

                        "image_url":
                            image_to_data_url(
                                image_bytes
                            ),

                        "detail":
                            "high"
                    }
                ]
            }
        ]
    )

    return parse_json_response(
        response.output_text
    )


# ============================================================
# SOLVE QUESTION
# ============================================================

def solve_question(
    image_bytes: bytes,
    page_number: int,
    question: Dict[str, Any],
    extracted_text: str
):

    question_number = str(
        question.get(
            "number",
            ""
        )
    ).strip()

    prompt = SOLUTION_PROMPT + """

PAGE NUMBER:
""" + str(page_number) + """

QUESTION NUMBER:
""" + question_number + """

VISIBLE QUESTION SUMMARY:
""" + str(
        question.get(
            "visible_text_summary",
            ""
        )
    ) + """

DIAGRAM DESCRIPTION:
""" + str(
        question.get(
            "diagram_description",
            ""
        )
    ) + """

SUPPORTING EXTRACTED TEXT:
""" + extracted_text[:16000]

    try:

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

                            "image_url":
                                image_to_data_url(
                                    image_bytes
                                ),

                            "detail":
                                "high"
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

                "question_number":
                    question_number,

                "method":
                    "",

                "working":
                    [],

                "final_answer":
                    response.output_text,

                "marking_scheme":
                    [],

                "visual_dependency":
                    "unknown",

                "visual_check":
                    "",

                # IMPORTANT:
                # Technical parser problem is MEDIUM,
                # never LOW.
                "confidence":
                    "medium",

                "confidence_reason":
                    "Response formatting could not be parsed; "
                    "this is a technical issue, not evidence of "
                    "a mathematical error.",

                "warning":
                    "Model response could not be parsed as JSON."
            }

        result[
            "question_number"
        ] = question_number

        return result

    except Exception as exc:

        return {

            "question_number":
                question_number,

            "method":
                "",

            "working":
                [],

            "final_answer":
                "",

            "marking_scheme":
                [],

            "visual_dependency":
                "unknown",

            "visual_check":
                "",

            # IMPORTANT:
            # API failure never becomes LOW.
            "confidence":
                "medium",

            "confidence_reason":
                "Solution service was unavailable.",

            "warning":
                str(exc)
        }


# ============================================================
# VERIFY QUESTION
# ============================================================

def verify_solution(
    image_bytes: bytes,
    page_number: int,
    question: Dict[str, Any],
    solution: Dict[str, Any]
):

    prompt = (

        VERIFICATION_PROMPT

        + "\n\nPAGE NUMBER:\n"
        + str(page_number)

        + "\n\nQUESTION:\n"
        + json.dumps(
            question,
            ensure_ascii=False,
            indent=2
        )

        + "\n\nGENERATED SOLUTION:\n"
        + json.dumps(
            solution,
            ensure_ascii=False,
            indent=2
        )
    )

    try:

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

                            "image_url":
                                image_to_data_url(
                                    image_bytes
                                ),

                            "detail":
                                "high"
                        }
                    ]
                }
            ]
        )

        result = parse_json_response(
            response.output_text
        )

        if result:

            return result

        return {

            "verified":
                False,

            "confidence":
                "medium",

            "reason":
                "Verification response formatting issue.",

            "mathematical_error":
                False,

            "visual_ambiguity":
                False,

            "recommended_action":
                "accept"
        }

    except Exception as exc:

        return {

            "verified":
                False,

            "confidence":
                "medium",

            "reason":
                "Verification service unavailable.",

            "mathematical_error":
                False,

            "visual_ambiguity":
                False,

            "recommended_action":
                "accept",

            "technical_error":
                str(exc)
        }


# ============================================================
# APPLY CLEAN CONFIDENCE RULE
# ============================================================

def apply_verification(
    solution: Dict[str, Any],
    verification: Dict[str, Any]
):

    mathematical_error = (
        verification.get(
            "mathematical_error",
            False
        )
        is True
    )

    visual_ambiguity = (
        verification.get(
            "visual_ambiguity",
            False
        )
        is True
    )

    verifier_confidence = str(
        verification.get(
            "confidence",
            ""
        )
    ).lower().strip()

    # --------------------------------------------------------
    # THE IMPORTANT PART
    # --------------------------------------------------------
    #
    # LOW is impossible unless there is actual evidence.
    # --------------------------------------------------------

    if (
        mathematical_error
        or visual_ambiguity
    ):

        final_confidence = "low"

    elif verifier_confidence == "high":

        final_confidence = "high"

    else:

        # Technical or unresolved situations are MEDIUM.
        final_confidence = "medium"

    reason = str(
        verification.get(
            "reason",
            ""
        )
    ).strip()

    if not reason:

        if final_confidence == "high":

            reason = (
                "Verification found no genuine mathematical "
                "error or essential visual ambiguity."
            )

        elif final_confidence == "low":

            reason = (
                "Verification identified a genuine issue "
                "requiring review."
            )

        else:

            reason = (
                "Verification did not establish sufficient "
                "evidence for a high-confidence result."
            )

    solution[
        "confidence"
    ] = final_confidence

    solution[
        "confidence_reason"
    ] = reason

    solution[
        "verification"
    ] = {

        "verified":
            bool(
                verification.get(
                    "verified",
                    False
                )
            ),

        "confidence":
            final_confidence,

        "reason":
            reason,

        "mathematical_error":
            mathematical_error,

        "visual_ambiguity":
            visual_ambiguity,

        "recommended_action":
            verification.get(
                "recommended_action",
                "accept"
            )
    }

    return solution


# ============================================================
# LATEX RENDERING
# ============================================================

def render_math_text(
    text: Any
):

    if text is None:
        return

    text = str(
        text
    ).strip()

    if not text:
        return

    text = text.replace(
        "\\[",
        "$$"
    )

    text = text.replace(
        "\\]",
        "$$"
    )

    text = text.replace(
        "\\(",
        "$"
    )

    text = text.replace(
        "\\)",
        "$"
    )

    st.markdown(
        text
    )


# ============================================================
# COMPLETENESS CHECK
# ============================================================

def completeness_check(
    results: List[Dict[str, Any]]
):

    expected = []
    solved = []

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
                expected.append(
                    number
                )

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
                solved.append(
                    number
                )

    expected_set = set(
        expected
    )

    solved_set = set(
        solved
    )

    missing = sorted(
        expected_set - solved_set,

        key=lambda x: (
            int(
                re.sub(
                    r"\D",
                    "",
                    x
                )
                or 999
            ),
            x
        )
    )

    return {

        "questions_detected":
            len(expected_set),

        "questions_solved":
            len(solved_set),

        "missing_questions":
            missing,

        "complete":
            len(missing) == 0
    }


# ============================================================
# DISPLAY QUESTION
# ============================================================

def display_question(
    image_bytes: bytes,
    question: Dict[str, Any],
    solution: Dict[str, Any]
):

    number = str(
        question.get(
            "number",
            solution.get(
                "question_number",
                "Unknown"
            )
        )
    )

    st.markdown(
        f"### 📌 Question {number}"
    )

    bbox = question.get(
        "bbox"
    )

    crop = None

    if bbox:

        crop = crop_original_question(
            image_bytes,
            bbox
        )

    if crop:

        st.image(
            crop,
            caption=(
                f"Original Question {number}"
            ),
            width="stretch"
        )

        st.caption(
            "🔒 Original question preserved directly "
            "from the examination page."
        )

    else:

        st.caption(
            "Original question crop unavailable. "
            "Refer to the full original page above."
        )

    st.markdown(
        f"#### ✏️ Question {number} — "
        "AI Marking Scheme"
    )

    confidence = str(
        solution.get(
            "confidence",
            "medium"
        )
    ).lower()

    if confidence == "high":

        st.success(
            "Confidence: high"
        )

    elif confidence == "medium":

        st.warning(
            "Confidence: medium"
        )

    else:

        st.error(
            "Confidence: low"
        )

    reason = solution.get(
        "confidence_reason",
        ""
    )

    if reason:

        st.caption(
            "Verification: "
            + str(reason)
        )

    verification = solution.get(
        "verification",
        {}
    )

    if isinstance(
        verification,
        dict
    ):

        if verification.get(
            "verified",
            False
        ):

            st.caption(
                "✓ Solution verification completed."
            )

        if verification.get(
            "mathematical_error",
            False
        ):

            st.error(
                "Mathematical verification found a "
                "possible error requiring review."
            )

        if verification.get(
            "visual_ambiguity",
            False
        ):

            st.warning(
                "Visual verification found essential "
                "information that may be ambiguous."
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
            + str(
                solution.get(
                    "visual_check",
                    ""
                )
            )
        )

    if solution.get(
        "method"
    ):

        st.markdown(
            "**Method**"
        )

        render_math_text(
            solution["method"]
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
            1
        ):

            st.markdown(
                f"**{index}.**"
            )

            render_math_text(
                step
            )

    else:

        render_math_text(
            working
        )

    final_answer = solution.get(
        "final_answer"
    )

    if final_answer:

        st.markdown(
            "**Final Answer**"
        )

        render_math_text(
            final_answer
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
                f"- **{marks} mark(s):**"
            )

            render_math_text(
                point
            )

    warning = solution.get(
        "warning"
    )

    if warning:

        st.warning(
            "⚠️ "
            + str(warning)
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

uploaded = st.file_uploader(
    "Upload the original examination paper",
    type=["pdf"],
    help=(
        "Upload the original PDF. "
        "Mwalimu AI analyses the actual pages."
    )
)


if uploaded:

    pdf_bytes = uploaded.getvalue()

    st.success(
        f"Loaded: {uploaded.name}"
    )

    if st.button(
        "🔍 Scan Original Paper & Generate "
        "Clean Marking Scheme",
        type="primary",
        use_container_width=True
    ):

        st.session_state.analysis_results = []

        st.session_state.page_images = []

        st.session_state.paper_name = (
            uploaded.name
        )

        # ----------------------------------------------------
        # RENDER
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
            f"Rendered {len(page_images)} "
            "original pages."
        )

        progress = st.progress(
            0
        )

        all_results = []

        # ----------------------------------------------------
        # EACH PAGE
        # ----------------------------------------------------

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
                        image_bytes,
                        page_number,
                        page_texts[index]
                    )

                except Exception as exc:

                    page_analysis = {

                        "questions":
                            [],

                        "visual_warnings":
                            [str(exc)]
                    }

            page_result = {

                "page_number":
                    page_number,

                "questions":
                    page_analysis.get(
                        "questions",
                        []
                    ),

                "visual_warnings":
                    page_analysis.get(
                        "visual_warnings",
                        []
                    ),

                "solutions":
                    []
            }

            # ------------------------------------------------
            # QUESTIONS
            # ------------------------------------------------

            for question in page_result[
                "questions"
            ]:

                number = str(
                    question.get(
                        "number",
                        ""
                    )
                ).strip()

                if not number:
                    continue

                with st.status(
                    f"Solving Question "
                    f"{number}...",
                    expanded=False
                ):

                    solution = solve_question(
                        image_bytes,
                        page_number,
                        question,
                        page_texts[index]
                    )

                # --------------------------------------------
                # VERIFICATION
                # --------------------------------------------

                with st.status(
                    f"Verifying Question "
                    f"{number}...",
                    expanded=False
                ):

                    verification = verify_solution(
                        image_bytes,
                        page_number,
                        question,
                        solution
                    )

                    solution = apply_verification(
                        solution,
                        verification
                    )

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
            "✅ Clean marking-scheme generation "
            "and verification completed."
        )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.analysis_results
)


if results:

    st.divider()

    st.header(
        "📄 Original Paper + "
        "Question-by-Question Marking Scheme"
    )

    st.info(
        "The original examination pages remain "
        "the visual source of truth. Mwalimu AI "
        "generates the working underneath them."
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
            "✅ Completeness check passed."
        )

    # --------------------------------------------------------
    # PAGE OUTPUT
    # --------------------------------------------------------

    for page_result in results:

        page_number = page_result[
            "page_number"
        ]

        st.divider()

        st.header(
            f"📄 Page {page_number}"
        )

        original_page = (
            st.session_state.page_images[
                page_number - 1
            ]
        )

        st.image(
            original_page,
            caption=(
                f"Original examination page "
                f"{page_number}"
            ),
            width="stretch"
        )

        st.caption(
            "🔒 Original page preserved — "
            "not reconstructed by AI."
        )

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

        solution_map = {}

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

                solution_map[
                    number
                ] = solution

        questions = page_result.get(
            "questions",
            []
        )

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

                    "confidence":
                        "medium",

                    "confidence_reason":
                        "No solution was generated.",

                    "working":
                        [],

                    "marking_scheme":
                        [],

                    "warning":
                        "No solution was generated."
                }

            st.divider()

            display_question(
                original_page,
                question,
                solution
            )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    st.divider()

    st.header(
        "📊 Visual Marking Report"
    )

    visual_questions = []

    for page_result in results:

        for question in page_result.get(
            "questions",
            []
        ):

            if (
                question.get(
                    "has_diagram",
                    False
                )
                or question.get(
                    "has_graph",
                    False
                )
                or question.get(
                    "has_construction",
                    False
                )
            ):

                visual_questions.append(
                    str(
                        question.get(
                            "number",
                            ""
                        )
                    )
                )

    if visual_questions:

        st.success(
            "Visual questions detected: "
            + ", ".join(
                visual_questions
            )
        )

    else:

        st.info(
            "No diagram/graph/construction "
            "questions were detected."
        )

    # --------------------------------------------------------
    # JSON DOWNLOAD
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
            "mwalimu_ai_clean_visual_marking_analysis.json"
        ),
        mime="application/json",
        use_container_width=True
    )


else:

    st.markdown(
        """
        ### 🧪 Clean Marking-Scheme Test

        **1. Upload the original paper**

        **2. Mwalimu AI reads the actual page image**

        **3. Questions are identified**

        **4. Complete mathematical working is generated**

        **5. The solution is verified against the original page**

        **6. Low confidence is reserved for genuine
        mathematical errors or essential visual ambiguity**

        **7. Original pages remain untouched**

        **8. Marking points are displayed under each question**
        """
    )
