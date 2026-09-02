import os
import io
import re
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import fitz
from PIL import Image
from openai import OpenAI


# ============================================================
# MWALIMU AI — MVP3.2 VISUAL MARKING
# FINAL MATHS POLISH
# ============================================================
#
# MAIN TARGETS:
#   1. SURDS
#   2. INTEGRATION
#   3. MAKING THE SUBJECT OF A FORMULA
#
# ORIGINAL PDF = VISUAL SOURCE OF TRUTH
#
# The original page is preserved.
# Individual questions are isolated visually.
# The solver receives:
#   - the isolated original question
#   - the complete original page
#   - supporting PDF text
#
# The verifier independently checks the solution.
#
# LOW CONFIDENCE is ONLY allowed for:
#   - genuine mathematical error
#   - genuine essential visual ambiguity
#
# Technical/API/parser problems NEVER automatically become LOW.
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

# Extra crop margin helps preserve nearby labels,
# diagram dimensions and mathematical notation.
QUESTION_CROP_MARGIN = 0.06


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

def render_pdf_pages(
    pdf_bytes: bytes
) -> List[bytes]:

    pages = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:

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

    finally:

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

    try:

        for page in document:

            try:
                text = page.get_text("text")
            except Exception:
                text = ""

            texts.append(
                text.strip()
            )

    finally:

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
# PAGE ANALYSIS PROMPT
# ============================================================

PAGE_ANALYSIS_PROMPT = r"""
You are the visual examination-paper analyser for Mwalimu AI.

The supplied image is the ORIGINAL mathematics examination page.

The image is authoritative.

Identify EVERY visible numbered question.

Do not solve the questions.

For every question provide an approximate bounding box in pixels.

The bounding box must include the COMPLETE visible question,
including:

- question number
- question text
- fractions
- numerators and denominators
- powers and indices
- roots and surds
- algebraic expressions
- brackets
- mathematical signs
- equations
- diagrams
- graphs
- constructions
- tables
- coordinates
- angle labels
- dimensions
- answer choices

IMPORTANT:

For mathematical questions, preserve the visual structure.

For example, distinguish:

    sqrt(2x)

from:

    sqrt(2) x

and distinguish:

    x^2

from:

    2x

and distinguish:

    1/(x+2)

from:

    1/x + 2

Do not invent missing information.

If a diagram is visible, identify it explicitly.

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
"""


# ============================================================
# SPECIALISED MATHEMATICS SOLUTION PROMPT
# ============================================================

SOLUTION_PROMPT = r"""
You are the SENIOR MATHEMATICS EXAMINER for Mwalimu AI.

You must solve the ORIGINAL examination question shown in the
supplied image.

The ORIGINAL IMAGE is authoritative.

The question image is the primary source.
The complete examination page is secondary visual context.

DO NOT guess.

DO NOT silently replace an unreadable expression with a likely one.

First identify the exact mathematical structure of the question.

Then solve it completely.

============================================================
MANDATORY MATHEMATICAL READING
============================================================

Pay particular attention to:

- fractions
- numerator/denominator boundaries
- brackets
- powers
- indices
- roots
- surds
- negative signs
- multiplication signs
- division signs
- algebraic coefficients
- variables
- integration symbols
- limits
- constants
- degrees
- angles
- coordinates
- diagram labels
- units

Never confuse:

    sqrt(a+b)

with:

    sqrt(a) + b

Never confuse:

    (a+b)^2

with:

    a+b^2

Never confuse:

    1/(a+b)

with:

    1/a+b

Never confuse:

    ∫x^n dx

with:

    x^n

============================================================
SURDS
============================================================

If the question involves surds:

1. Identify the exact surd expression.
2. Simplify perfect-square factors.
3. Combine like surds where appropriate.
4. Rationalise denominators where required.
5. Show every important algebraic step.
6. Do not turn an exact surd answer into a decimal unless requested.
7. Check the final result algebraically.

Examples of proper notation:

\sqrt{12}=2\sqrt{3}

\frac{1}{\sqrt{3}}
=
\frac{\sqrt{3}}{3}

\frac{1}{a+\sqrt{b}}

Use the actual expression from the image,
not an assumed textbook version.

============================================================
INTEGRATION
============================================================

If the question involves integration:

1. Identify the exact integrand.
2. Rewrite it into a suitable form if necessary.
3. Apply the correct integration rule.
4. Show the power-rule step explicitly where appropriate.
5. Include the constant of integration for indefinite integration.
6. Apply limits correctly for definite integration.
7. Simplify the exact answer.
8. Check the result by differentiation when useful.

For example:

\[
\int x^n\,dx
=
\frac{x^{n+1}}{n+1}+C
\]

provided \(n\neq -1\).

For a definite integral:

\[
\int_a^b f(x)\,dx
=
F(b)-F(a)
\]

Do not omit \(C\) from an indefinite integral.

Do not invent limits.

============================================================
MAKING THE SUBJECT OF A FORMULA
============================================================

If the question asks to make a particular variable the subject:

1. Identify the requested subject exactly.
2. State the original formula.
3. Rearrange one operation at a time.
4. Keep both sides mathematically balanced.
5. Deal carefully with fractions.
6. Deal carefully with powers and roots.
7. Show sign choices where square roots are involved.
8. State restrictions if mathematically necessary.
9. Substitute back or otherwise verify the rearrangement.

For example, do not jump directly from:

\[
y=\frac{x+a}{b}
\]

to the answer without showing the algebraic rearrangement.

============================================================
GENERAL MATHEMATICS
============================================================

Show complete examination-quality working.

Give:

1. Method
2. Step-by-step working
3. Final answer
4. Concise marking scheme

The working must be sufficient for a teacher to award marks.

Do not give only the final answer.

============================================================
DIAGRAM QUESTIONS
============================================================

If a diagram is involved:

- use the ACTUAL visible labels;
- use the ACTUAL visible measurements;
- use the ACTUAL visible angles;
- use the ACTUAL coordinates;
- use the actual relationships shown;
- do not invent missing dimensions;
- do not assume a diagram is to scale unless the question says so.

============================================================
CONFIDENCE
============================================================

Do NOT use LOW confidence merely because:

- the question is difficult;
- the mathematics is advanced;
- a diagram exists;
- surds are present;
- integration is present;
- algebra is complicated;
- LaTeX is used;
- the answer is long;
- the model is generally uncertain;
- JSON formatting is imperfect;
- an API/technical problem occurred.

LOW confidence is allowed ONLY when:

A. There is a genuine mathematical error.

OR

B. Essential visual information is genuinely unreadable
or ambiguous.

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
You are the FINAL INDEPENDENT MATHEMATICS VERIFICATION EXAMINER
for Mwalimu AI.

Compare the generated solution against the ORIGINAL QUESTION
IMAGE and the COMPLETE ORIGINAL PAGE.

The original image is authoritative.

Do not merely agree with the generated solution.

Independently check the mathematics.

============================================================
CHECK 1 — QUESTION READING
============================================================

Confirm that:

- the correct question was solved;
- all visible numbers were read correctly;
- fractions were read correctly;
- signs were read correctly;
- powers and indices were read correctly;
- roots and surds were read correctly;
- brackets were read correctly;
- integration symbols and limits were read correctly;
- diagram labels were read correctly.

============================================================
CHECK 2 — SURDS
============================================================

If surds are present:

- verify simplification;
- verify multiplication/division;
- verify collection of like surds;
- verify rationalisation;
- verify the final exact answer.

============================================================
CHECK 3 — INTEGRATION
============================================================

If integration is present:

- verify the integrand;
- verify the integration rule;
- verify powers;
- verify coefficients;
- verify the constant of integration where required;
- verify limits for definite integrals;
- differentiate the result mentally/algebraically where useful.

============================================================
CHECK 4 — MAKING SUBJECT OF FORMULA
============================================================

If algebraic rearrangement is present:

- verify every rearrangement;
- verify signs;
- verify denominators;
- verify powers;
- verify roots;
- verify that the requested variable is actually the subject.

============================================================
CHECK 5 — DIAGRAM
============================================================

If a diagram is involved:

- verify visible labels;
- verify dimensions;
- verify angles;
- verify coordinates;
- verify geometric relationships;
- do not invent missing information.

============================================================
CONFIDENCE RULE
============================================================

LOW confidence MUST NOT be used merely because:

- the question is difficult;
- a diagram exists;
- surds are present;
- integration is present;
- algebra is complicated;
- LaTeX is present;
- the solution is lengthy;
- the verifier has general uncertainty;
- formatting is imperfect;
- a technical service failed.

LOW is ONLY permitted when:

A. There is a genuine mathematical error.

OR

B. Essential visual information is genuinely unreadable
or ambiguous.

If the mathematics is correct and the visual information is
sufficiently clear, return HIGH.

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

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        try:

            parsed = json.loads(
                text[start:end + 1]
            )

            if isinstance(parsed, dict):
                return parsed

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
) -> Optional[Dict[str, int]]:

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
) -> Optional[bytes]:

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

        # Generous margin is deliberate.
        # It protects against cutting off:
        #   - a denominator
        #   - a superscript
        #   - a radical
        #   - an equation continuation
        #   - diagram labels
        #   - angle measurements
        margin_x = max(
            18,
            int(width * QUESTION_CROP_MARGIN)
        )

        margin_y = max(
            18,
            int(height * QUESTION_CROP_MARGIN)
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
) -> Dict[str, Any]:

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
    page_image_bytes: bytes,
    question_image_bytes: Optional[bytes],
    page_number: int,
    question: Dict[str, Any],
    extracted_text: str
) -> Dict[str, Any]:

    question_number = str(
        question.get(
            "number",
            ""
        )
    ).strip()

    prompt = (
        SOLUTION_PROMPT
        + "\n\nPAGE NUMBER:\n"
        + str(page_number)
        + "\n\nQUESTION NUMBER:\n"
        + question_number
        + "\n\nVISIBLE QUESTION SUMMARY:\n"
        + str(
            question.get(
                "visible_text_summary",
                ""
            )
        )
        + "\n\nDIAGRAM DESCRIPTION:\n"
        + str(
            question.get(
                "diagram_description",
                ""
            )
        )
        + "\n\nIMPORTANT VISUAL FEATURES:\n"
        + json.dumps(
            question.get(
                "important_visual_features",
                []
            ),
            ensure_ascii=False
        )
        + "\n\nSUPPORTING EXTRACTED TEXT:\n"
        + extracted_text[:16000]
    )

    content = [

        {
            "type": "input_text",
            "text": prompt
        }
    ]

    # The isolated question is the PRIMARY image.
    if question_image_bytes:

        content.append(
            {
                "type": "input_text",
                "text": (
                    "PRIMARY QUESTION IMAGE: "
                    "Read the exact mathematical expression "
                    "from this isolated original question."
                )
            }
        )

        content.append(
            {
                "type": "input_image",

                "image_url":
                    image_to_data_url(
                        question_image_bytes
                    ),

                "detail":
                    "high"
            }
        )

    # Full page remains available as visual context.
    content.append(
        {
            "type": "input_text",
            "text": (
                "COMPLETE ORIGINAL PAGE: "
                "Use this to verify context, question continuation "
                "and surrounding diagram information."
            )
        }
    )

    content.append(
        {
            "type": "input_image",

            "image_url":
                image_to_data_url(
                    page_image_bytes
                ),

            "detail":
                "high"
        }
    )

    try:

        response = client.responses.create(

            model=MODEL,

            input=[
                {
                    "role": "user",
                    "content": content
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
                    "medium",

                "visual_check":
                    "",

                "confidence":
                    "medium",

                "confidence_reason":
                    "The mathematical response was generated, "
                    "but its JSON structure could not be parsed. "
                    "This is a technical issue, not evidence of "
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
                "medium",

            "visual_check":
                "",

            "confidence":
                "medium",

            "confidence_reason":
                "The solution service was unavailable. "
                "This is not evidence of a mathematical error.",

            "warning":
                str(exc)
        }


# ============================================================
# VERIFY QUESTION
# ============================================================

def verify_solution(
    page_image_bytes: bytes,
    question_image_bytes: Optional[bytes],
    page_number: int,
    question: Dict[str, Any],
    solution: Dict[str, Any]
) -> Dict[str, Any]:

    prompt = (

        VERIFICATION_PROMPT

        + "\n\nPAGE NUMBER:\n"
        + str(page_number)

        + "\n\nQUESTION METADATA:\n"
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

    content = [

        {
            "type": "input_text",
            "text": prompt
        }
    ]

    # Primary verification image.
    if question_image_bytes:

        content.append(
            {
                "type": "input_text",
                "text": (
                    "PRIMARY ORIGINAL QUESTION: "
                    "Independently read this exact question "
                    "before judging the generated solution."
                )
            }
        )

        content.append(
            {
                "type": "input_image",

                "image_url":
                    image_to_data_url(
                        question_image_bytes
                    ),

                "detail":
                    "high"
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "COMPLETE ORIGINAL PAGE: "
                "Use this for context and visual verification."
            )
        }
    )

    content.append(
        {
            "type": "input_image",

            "image_url":
                image_to_data_url(
                    page_image_bytes
                ),

            "detail":
                "high"
        }
    )

    try:

        response = client.responses.create(

            model=MODEL,

            input=[
                {
                    "role": "user",
                    "content": content
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
                "Verification response formatting issue. "
                "No mathematical error was established.",

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
                "Verification service unavailable. "
                "No mathematical error was established.",

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
) -> Dict[str, Any]:

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

    verified = (
        verification.get(
            "verified",
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

    # ========================================================
    # STRICT CONFIDENCE GATE
    # ========================================================
    #
    # LOW requires actual evidence.
    #
    # Technical problems cannot create LOW.
    #
    # ========================================================

    if (
        mathematical_error
        or visual_ambiguity
    ):

        final_confidence = "low"

    elif (
        verified
        and verifier_confidence == "high"
    ):

        final_confidence = "high"

    else:

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
                "Independent verification found no genuine "
                "mathematical error or essential visual ambiguity."
            )

        elif final_confidence == "low":

            reason = (
                "Independent verification identified a genuine "
                "mathematical or essential visual issue."
            )

        else:

            reason = (
                "The result was not given high confidence by "
                "independent verification. No genuine mathematical "
                "error was established."
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
            verified,

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
) -> Dict[str, Any]:

    expected = []
    solved = []

    for page_result in results:

        page_number = str(
            page_result.get(
                "page_number",
                ""
            )
        )

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
                    f"{page_number}:{number}"
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
                    f"{page_number}:{number}"
                )

    expected_set = set(
        expected
    )

    solved_set = set(
        solved
    )

    missing_keys = expected_set - solved_set

    missing_questions = []

    for key in sorted(
        missing_keys
    ):

        try:

            page, number = key.split(
                ":",
                1
            )

            missing_questions.append(
                f"Page {page}, Q{number}"
            )

        except Exception:

            missing_questions.append(
                key
            )

    return {

        "questions_detected":
            len(expected_set),

        "questions_solved":
            len(
                expected_set
                & solved_set
            ),

        "missing_questions":
            missing_questions,

        "complete":
            len(missing_questions) == 0
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
            "Confidence: HIGH"
        )

    elif confidence == "medium":

        st.warning(
            "Confidence: MEDIUM"
        )

    else:

        st.error(
            "Confidence: LOW"
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
                "✓ Independent solution verification completed."
            )

        if verification.get(
            "mathematical_error",
            False
        ):

            st.error(
                "Mathematical verification found a "
                "possible genuine error requiring review."
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

            if not isinstance(
                item,
                dict
            ):
                continue

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

        # ====================================================
        # RENDER
        # ====================================================

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

        if not page_images:

            st.error(
                "No pages were found in the PDF."
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

        # ====================================================
        # EACH PAGE
        # ====================================================

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
                            [
                                str(exc)
                            ]
                    }

            questions = page_analysis.get(
                "questions",
                []
            )

            page_result = {

                "page_number":
                    page_number,

                "questions":
                    questions,

                "visual_warnings":
                    page_analysis.get(
                        "visual_warnings",
                        []
                    ),

                "solutions":
                    []
            }

            # =================================================
            # QUESTIONS
            # =================================================

            for question in questions:

                number = str(
                    question.get(
                        "number",
                        ""
                    )
                ).strip()

                if not number:
                    continue

                # ---------------------------------------------
                # CREATE ORIGINAL QUESTION CROP
                # ---------------------------------------------

                question_crop = None

                bbox = question.get(
                    "bbox"
                )

                if bbox:

                    question_crop = (
                        crop_original_question(
                            image_bytes,
                            bbox
                        )
                    )

                # ---------------------------------------------
                # SOLVE
                # ---------------------------------------------

                with st.status(
                    f"Solving Question "
                    f"{number}...",
                    expanded=False
                ):

                    solution = solve_question(
                        page_image_bytes=image_bytes,
                        question_image_bytes=question_crop,
                        page_number=page_number,
                        question=question,
                        extracted_text=page_texts[index]
                    )

                # ---------------------------------------------
                # VERIFY
                # ---------------------------------------------

                with st.status(
                    f"Verifying Question "
                    f"{number}...",
                    expanded=False
                ):

                    verification = verify_solution(
                        page_image_bytes=image_bytes,
                        question_image_bytes=question_crop,
                        page_number=page_number,
                        question=question,
                        solution=solution
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
            "and independent verification completed."
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
        "generates the mathematical working underneath them."
    )

    # ========================================================
    # COMPLETENESS
    # ========================================================

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

    # ========================================================
    # PAGE OUTPUT
    # ========================================================

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

    # ========================================================
    # FINAL REPORT
    # ========================================================

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

    # ========================================================
    # MATHS DIAGNOSTIC
    # ========================================================

    maths_features = {
        "surds": 0,
        "integration": 0,
        "formula_subject": 0
    }

    for page_result in results:

        for question in page_result.get(
            "questions",
            []
        ):

            summary = (
                str(
                    question.get(
                        "visible_text_summary",
                        ""
                    )
                )
                .lower()
            )

            features = " ".join(
                str(x).lower()
                for x in question.get(
                    "important_visual_features",
                    []
                )
            )

            combined = (
                summary
                + " "
                + features
            )

            if any(
                word in combined
                for word in [
                    "surd",
                    "root",
                    "radical",
                    "√"
                ]
            ):

                maths_features[
                    "surds"
                ] += 1

            if any(
                word in combined
                for word in [
                    "integrat",
                    "∫",
                    "integral"
                ]
            ):

                maths_features[
                    "integration"
                ] += 1

            if any(
                phrase in combined
                for phrase in [
                    "make",
                    "subject",
                    "formula"
                ]
            ):

                maths_features[
                    "formula_subject"
                ] += 1

    if any(
        value > 0
        for value in maths_features.values()
    ):

        st.markdown(
            "### 🧮 Mathematics focus detected"
        )

        m1, m2, m3 = st.columns(
            3
        )

        with m1:

            st.metric(
                "Surd questions",
                maths_features[
                    "surds"
                ]
            )

        with m2:

            st.metric(
                "Integration questions",
                maths_features[
                    "integration"
                ]
            )

        with m3:

            st.metric(
                "Formula questions",
                maths_features[
                    "formula_subject"
                ]
            )

    # ========================================================
    # JSON DOWNLOAD
    # ========================================================

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

        **4. Each question is visually isolated**

        **5. Complete mathematical working is generated**

        **6. Surds, integration and algebraic rearrangement
        receive dedicated mathematical instructions**

        **7. The solution is independently verified
        against the original question**

        **8. Low confidence is reserved for genuine
        mathematical errors or essential visual ambiguity**

        **9. Original pages remain untouched**

        **10. Marking points are displayed under each question**
        """
    )
