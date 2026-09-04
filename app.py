import os
import io
import re
import json
import base64
import math
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import fitz
from PIL import Image
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import sympy as sp


# ============================================================
# MWALIMU AI — MVP3.2 VISUAL MARKING
# FINAL MATHS POLISH + CONSTRUCTION V1
# ============================================================

st.set_page_config(
    page_title="Mwalimu AI — Visual Marking",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Mwalimu AI")
st.caption(
    "MVP3.2 Visual Marking Engine — Original question preserved, "
    "AI workings, graphs and mathematical constructions added underneath."
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

MAX_PAGE_DIMENSION = 1800
JPEG_QUALITY = 88
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

def render_pdf_pages(pdf_bytes: bytes) -> List[bytes]:
    pages = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        for page in document:
            rect = page.rect

            scale = MAX_PAGE_DIMENSION / max(
                rect.width,
                rect.height,
            )

            scale = min(scale, 2.5)

            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )

            pages.append(
                pix.tobytes(
                    "jpeg",
                    jpg_quality=JPEG_QUALITY,
                )
            )
    finally:
        document.close()

    return pages


# ============================================================
# OPTIONAL PDF TEXT
# ============================================================

def extract_page_texts(pdf_bytes: bytes) -> List[str]:
    texts = []

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        for page in document:
            try:
                text = page.get_text("text")
            except Exception:
                text = ""

            texts.append(text.strip())
    finally:
        document.close()

    return texts


# ============================================================
# IMAGE → DATA URL
# ============================================================

def image_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


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

The bounding box must include the COMPLETE visible question, including:

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

Distinguish:
sqrt(2x) from sqrt(2)x
x^2 from 2x
1/(x+2) from 1/x + 2

Do not invent missing information.

If a diagram is visible, identify it explicitly.

If a geometric construction is visible or required, set
has_construction to true.

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
# CONSTRUCTION DEFAULT SPEC
# ============================================================

def default_construction_spec() -> Dict[str, Any]:
    return {
        "required": False,
        "construction_type": "none",
        "points": {},
        "lengths": {},
        "angles": {},
        "radius": None,
        "line_start": "A",
        "line_end": "B",
        "point": "P",
        "label_points": True,
        "show_construction_lines": True,
        "show_arcs": True,
        "title": "",
    }


# ============================================================
# SPECIALISED MATHEMATICS SOLUTION PROMPT
# ============================================================

SOLUTION_PROMPT = r"""
You are the SENIOR MATHEMATICS EXAMINER for Mwalimu AI.

You must solve the ORIGINAL examination question shown in the supplied image.

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

sqrt(a+b) with sqrt(a)+b
(a+b)^2 with a+b^2
1/(a+b) with 1/a+b
∫x^n dx with x^n

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

Use proper mathematical notation.

============================================================
INTEGRATION
============================================================

If the question involves integration:

1. Identify the exact integrand.
2. Rewrite it into a suitable form if necessary.
3. Apply the correct integration rule.
4. Show the power-rule step explicitly where appropriate.
5. Include C for indefinite integration.
6. Apply limits correctly for definite integration.
7. Simplify the exact answer.
8. Check by differentiation where useful.

Do not invent limits.

============================================================
MAKING THE SUBJECT OF A FORMULA
============================================================

If the question asks to make a variable the subject:

1. Identify the requested subject exactly.
2. State the original formula.
3. Rearrange one operation at a time.
4. Keep both sides mathematically balanced.
5. Deal carefully with fractions.
6. Deal carefully with powers and roots.
7. Show sign choices where square roots are involved.
8. State restrictions if necessary.
9. Verify the rearrangement.

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

============================================================
DIAGRAM QUESTIONS
============================================================

If a diagram is involved:

- use actual visible labels;
- use actual visible measurements;
- use actual visible angles;
- use actual visible coordinates;
- use actual relationships shown;
- do not invent missing dimensions;
- do not assume a diagram is to scale unless the question says so.

============================================================
GRAPHING
============================================================

If the question requires a graph, identify:

- equation/function
- graph type
- domain
- supplied coordinates
- table values
- intercepts
- axis labels
- point requirements
- joining requirements
- shading requirements
- specified scale or axis limits

Do NOT claim that a graph has already been drawn by AI.

Return a graph_spec so Python can generate it mathematically.

Allowed graph types:

- line
- quadratic
- function
- points

Canonical expressions:

2*x+3
x**2-4*x+3
sin(x)
cos(x)

Do not place arbitrary Python code inside graph_spec.

============================================================
GEOMETRIC CONSTRUCTION
============================================================

If the question requires a mathematical construction, identify it
and return a construction_spec.

IMPORTANT:

Do NOT attempt to draw the construction using text or ASCII.

Do NOT invent arbitrary dimensions.

Python will generate the actual construction from this specification.

Recognised construction types:

- angle_bisector
- perpendicular_bisector
- perpendicular_from_point
- triangle_sides
- triangle_base_angles
- locus_circle
- locus_perpendicular_bisector

Use supplied lengths and angles only.

If exact coordinates are NOT given, use a canonical orientation
based on the supplied measurements/angles.

The coordinates do NOT need to preserve the physical scale of
the examination drawing.

Do not invent information merely to make a construction possible.

construction_spec structure:

{
  "required": false,
  "construction_type": "none",
  "points": {
    "A": [0, 0],
    "B": [10, 0],
    "C": [4, 7],
    "P": [5, 5]
  },
  "lengths": {
    "AB": 10,
    "BC": 8,
    "CA": 7
  },
  "angles": {
    "A": 60,
    "B": 45,
    "C": 75
  },
  "radius": null,
  "line_start": "A",
  "line_end": "B",
  "point": "P",
  "label_points": true,
  "show_construction_lines": true,
  "show_arcs": true,
  "title": ""
}

For an angle bisector, identify the vertex and the two rays.

For a perpendicular bisector, identify the two endpoints of the
segment.

For a perpendicular from a point, identify the line and external
point.

For triangle_sides, use the three supplied side lengths.

For triangle_base_angles, use the supplied base and base angles.

For a locus circle, use the actual supplied centre and radius.

For a perpendicular-bisector locus, use the supplied segment.

The generated construction is mathematical and does not claim to
preserve the exact scale of the original examination drawing.

============================================================
CONFIDENCE
============================================================

LOW confidence is allowed ONLY for:

A. genuine mathematical error

OR

B. genuine essential visual ambiguity.

Technical/API/parser/rendering problems are NOT mathematical errors.

Return JSON only:

{
  "question_number": "...",
  "method": "...",
  "working": [],
  "final_answer": "...",
  "marking_scheme": [],
  "visual_dependency": "none | low | medium | high",
  "visual_check": "...",

  "graph_spec": {
    "required": false,
    "graph_type": "none",
    "expression": "",
    "x_values": [],
    "y_values": [],
    "x_min": -10,
    "x_max": 10,
    "y_min": null,
    "y_max": null,
    "x_label": "x",
    "y_label": "y",
    "show_points": false,
    "connect_points": false,
    "shade": false,
    "shade_direction": "",
    "angle_unit": "radians",
    "title": ""
  },

  "construction_spec": {
    "required": false,
    "construction_type": "none",
    "points": {},
    "lengths": {},
    "angles": {},
    "radius": null,
    "line_start": "A",
    "line_end": "B",
    "point": "P",
    "label_points": true,
    "show_construction_lines": true,
    "show_arcs": true,
    "title": ""
  },

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

Compare the generated solution against the ORIGINAL QUESTION IMAGE
and the COMPLETE ORIGINAL PAGE.

The original image is authoritative.

Do not merely agree with the generated solution.

Independently check the mathematics.

============================================================
QUESTION READING
============================================================

Confirm:

- correct question
- visible numbers
- fractions
- signs
- powers
- indices
- roots
- surds
- brackets
- integration symbols
- limits
- diagram labels
- dimensions
- angles
- coordinates

============================================================
SURDS
============================================================

Verify:

- simplification
- multiplication/division
- collection of like surds
- rationalisation
- final exact answer

============================================================
INTEGRATION
============================================================

Verify:

- integrand
- integration rule
- powers
- coefficients
- constant of integration
- limits
- final result

============================================================
MAKING SUBJECT OF FORMULA
============================================================

Verify:

- requested variable
- rearrangement
- signs
- denominators
- powers
- roots
- final subject

============================================================
DIAGRAM
============================================================

Verify:

- labels
- dimensions
- angles
- coordinates
- geometric relationships

Do not invent missing information.

============================================================
GRAPH
============================================================

If required, verify:

- equation
- graph type
- domain
- coordinates
- intercepts
- axis labels
- joining
- mathematical shape

A technical failure to render a graph is NOT a mathematical error.

============================================================
CONSTRUCTION
============================================================

If a construction is required, independently verify:

- construction type
- supplied lengths
- supplied angles
- point/line relationships
- construction parameters
- labels
- whether the construction specification corresponds to the
  actual question

Do NOT penalise the solution because Python fails to render the
construction technically.

A wrong construction type or wrong construction parameters IS a
genuine mathematical/visual error.

Do not accept invented dimensions or angles.

============================================================
CONFIDENCE
============================================================

LOW is ONLY permitted when:

A. genuine mathematical error

OR

B. essential visual information is genuinely unreadable/ambiguous.

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

def parse_json_response(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = str(text).strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
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
    image_height: int,
) -> Optional[Dict[str, int]]:

    if not isinstance(bbox, dict):
        return None

    try:
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        width = int(bbox.get("width", 0))
        height = int(bbox.get("height", 0))
    except Exception:
        return None

    if width <= 5 or height <= 5:
        return None

    x = max(
        0,
        min(x, image_width - 1),
    )

    y = max(
        0,
        min(y, image_height - 1),
    )

    width = min(
        width,
        image_width - x,
    )

    height = min(
        height,
        image_height - y,
    )

    if width <= 5 or height <= 5:
        return None

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


# ============================================================
# ORIGINAL QUESTION CROP
# ============================================================

def crop_original_question(
    image_bytes: bytes,
    bbox: Dict[str, int],
) -> Optional[bytes]:

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image_width, image_height = image.size

        safe_bbox = normalise_bbox(
            bbox,
            image_width,
            image_height,
        )

        if not safe_bbox:
            return None

        x = safe_bbox["x"]
        y = safe_bbox["y"]
        width = safe_bbox["width"]
        height = safe_bbox["height"]

        margin_x = max(
            18,
            int(width * QUESTION_CROP_MARGIN),
        )

        margin_y = max(
            18,
            int(height * QUESTION_CROP_MARGIN),
        )

        crop = image.crop(
            (
                max(0, x - margin_x),
                max(0, y - margin_y),
                min(
                    image_width,
                    x + width + margin_x,
                ),
                min(
                    image_height,
                    y + height + margin_y,
                ),
            )
        )

        output = io.BytesIO()

        crop.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
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
    extracted_text: str,
) -> Dict[str, Any]:

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
                            + "\n\nPAGE NUMBER:\n"
                            + str(page_number)
                            + "\n\nSUPPORTING PDF TEXT:\n"
                            + extracted_text[:12000]
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            image_bytes
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
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
    extracted_text: str,
) -> Dict[str, Any]:

    question_number = str(
        question.get("number", "")
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
                "",
            )
        )
        + "\n\nDIAGRAM DESCRIPTION:\n"
        + str(
            question.get(
                "diagram_description",
                "",
            )
        )
        + "\n\nIMPORTANT VISUAL FEATURES:\n"
        + json.dumps(
            question.get(
                "important_visual_features",
                [],
            ),
            ensure_ascii=False,
        )
        + "\n\nSUPPORTING EXTRACTED TEXT:\n"
        + extracted_text[:16000]
    )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    if question_image_bytes:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "PRIMARY QUESTION IMAGE: "
                    "Read the exact mathematical expression "
                    "from this isolated original question."
                ),
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(
                    question_image_bytes
                ),
                "detail": "high",
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "COMPLETE ORIGINAL PAGE: "
                "Use this to verify context, question continuation "
                "and surrounding diagram information."
            ),
        }
    )

    content.append(
        {
            "type": "input_image",
            "image_url": image_to_data_url(
                page_image_bytes
            ),
            "detail": "high",
        }
    )

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
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
                "visual_dependency": "medium",
                "visual_check": "",
                "graph_spec": {
                    "required": False,
                    "graph_type": "none",
                    "expression": "",
                    "x_values": [],
                    "y_values": [],
                    "x_min": -10,
                    "x_max": 10,
                    "y_min": None,
                    "y_max": None,
                    "x_label": "x",
                    "y_label": "y",
                    "show_points": False,
                    "connect_points": False,
                    "shade": False,
                    "shade_direction": "",
                    "angle_unit": "radians",
                    "title": "",
                },
                "construction_spec": default_construction_spec(),
                "confidence": "medium",
                "confidence_reason": (
                    "The mathematical response was generated, "
                    "but its JSON structure could not be parsed. "
                    "This is a technical issue, not evidence of "
                    "a mathematical error."
                ),
                "warning": (
                    "Model response could not be parsed as JSON."
                ),
            }

        result["question_number"] = question_number

        if not isinstance(
            result.get("construction_spec"),
            dict,
        ):
            result["construction_spec"] = (
                default_construction_spec()
            )

        if not isinstance(
            result.get("graph_spec"),
            dict,
        ):
            result["graph_spec"] = {
                "required": False,
                "graph_type": "none",
            }

        return result

    except Exception as exc:
        return {
            "question_number": question_number,
            "method": "",
            "working": [],
            "final_answer": "",
            "marking_scheme": [],
            "visual_dependency": "medium",
            "visual_check": "",
            "graph_spec": {
                "required": False,
                "graph_type": "none",
            },
            "construction_spec": default_construction_spec(),
            "confidence": "medium",
            "confidence_reason": (
                "The solution service was unavailable. "
                "This is not evidence of a mathematical error."
            ),
            "warning": str(exc),
        }


# ============================================================
# VERIFY QUESTION
# ============================================================

def verify_solution(
    page_image_bytes: bytes,
    question_image_bytes: Optional[bytes],
    page_number: int,
    question: Dict[str, Any],
    solution: Dict[str, Any],
) -> Dict[str, Any]:

    prompt = (
        VERIFICATION_PROMPT
        + "\n\nPAGE NUMBER:\n"
        + str(page_number)
        + "\n\nQUESTION METADATA:\n"
        + json.dumps(
            question,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nGENERATED SOLUTION:\n"
        + json.dumps(
            solution,
            ensure_ascii=False,
            indent=2,
        )
    )

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    if question_image_bytes:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "PRIMARY ORIGINAL QUESTION: "
                    "Independently read this exact question "
                    "before judging the generated solution."
                ),
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(
                    question_image_bytes
                ),
                "detail": "high",
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "COMPLETE ORIGINAL PAGE: "
                "Use this for context and visual verification."
            ),
        }
    )

    content.append(
        {
            "type": "input_image",
            "image_url": image_to_data_url(
                page_image_bytes
            ),
            "detail": "high",
        }
    )

    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        result = parse_json_response(
            response.output_text
        )

        if result:
            return result

        return {
            "verified": False,
            "confidence": "medium",
            "reason": (
                "Verification response formatting issue. "
                "No mathematical error was established."
            ),
            "mathematical_error": False,
            "visual_ambiguity": False,
            "recommended_action": "accept",
        }

    except Exception as exc:
        return {
            "verified": False,
            "confidence": "medium",
            "reason": (
                "Verification service unavailable. "
                "No mathematical error was established."
            ),
            "mathematical_error": False,
            "visual_ambiguity": False,
            "recommended_action": "accept",
            "technical_error": str(exc),
        }


# ============================================================
# APPLY CLEAN CONFIDENCE RULE
# ============================================================

def apply_verification(
    solution: Dict[str, Any],
    verification: Dict[str, Any],
) -> Dict[str, Any]:

    mathematical_error = (
        verification.get(
            "mathematical_error",
            False,
        )
        is True
    )

    visual_ambiguity = (
        verification.get(
            "visual_ambiguity",
            False,
        )
        is True
    )

    verified = (
        verification.get(
            "verified",
            False,
        )
        is True
    )

    verifier_confidence = str(
        verification.get(
            "confidence",
            "",
        )
    ).lower().strip()

    if mathematical_error or visual_ambiguity:
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
            "",
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

    solution["confidence"] = final_confidence
    solution["confidence_reason"] = reason

    solution["verification"] = {
        "verified": verified,
        "confidence": final_confidence,
        "reason": reason,
        "mathematical_error": mathematical_error,
        "visual_ambiguity": visual_ambiguity,
        "recommended_action": verification.get(
            "recommended_action",
            "accept",
        ),
    }

    return solution


# ============================================================
# LATEX / MATHEMATICAL RENDERING
# ============================================================

def _clean_math_text(text: str) -> str:
    text = str(text).strip()

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"^\s*```(?:latex|tex|math)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    text = text.replace("", "$$")
    text = text.replace("", "$$")
    text = text.replace("", "$")
    text = text.replace("", "$")

    text = re.sub(
        r"<br\s*/?>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def render_math_text(text: Any):
    if text is None:
        return

    text = _clean_math_text(text)

    if not text:
        return

    replacements = [
        ("−", "-"),
        ("×", r"\times "),
        ("÷", r"\div "),
        ("°", r"^\circ"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    text = re.sub(
        r"√\s*([A-Za-z0-9]+)",
        r"$\\sqrt{\1}$",
        text,
    )

    text = re.sub(
        r"(?<!\^)([A-Za-z])\^(-?\d+)(?!\})",
        r"$\1^{\2}$",
        text,
    )

    text = re.sub(
        r"(?<![_\\])([A-Za-z])_([0-9]+)",
        r"$\1_{\2}$",
        text,
    )

    lines = text.split("\n")
    rendered_lines = []

    math_indicators = (
        r"\frac",
        r"\dfrac",
        r"\tfrac",
        r"\sqrt",
        r"\int",
        r"\sum",
        r"\prod",
        r"\lim",
        r"\sin",
        r"\cos",
        r"\tan",
        r"\log",
        r"\pi",
        r"\pm",
        r"\leq",
        r"\geq",
        r"\neq",
        r"\times",
        r"\div",
        r"\cdot",
    )

    for line in lines:
        stripped = line.strip()

        if not stripped:
            rendered_lines.append("")
            continue

        if (
            "$" in stripped
            or any(
                indicator in stripped
                for indicator in math_indicators
            )
        ):
            rendered_lines.append(line)
            continue

        looks_mathematical = bool(
            re.search(
                r"(=|≤|≥|≠|\+|-|\*|/|\^|√|\b\d+[a-zA-Z]\b)",
                stripped,
            )
            and re.search(
                r"[A-Za-z0-9]",
                stripped,
            )
        )

        if (
            looks_mathematical
            and len(stripped) < 180
            and re.search(
                r"^[A-Za-z\s]+$",
                stripped,
            ) is None
        ):
            rendered_lines.append(
                "$" + stripped + "$"
            )
            continue

        rendered_lines.append(line)

    st.markdown(
        "\n".join(rendered_lines)
    )


# ============================================================
# GRAPH ENGINE — PRESERVED
# ============================================================

GRAPH_SAFE_LOCALS = {
    "x": sp.symbols("x"),
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "pi": sp.pi,
    "E": sp.E,
    "Abs": sp.Abs,
}


def _safe_graph_expression(expression: str):
    if not expression:
        return None

    expression = str(expression).strip()

    expression = re.sub(
        r"^\s*y\s*=\s*",
        "",
        expression,
        flags=re.IGNORECASE,
    )

    expression = expression.replace("^", "**")
    expression = expression.replace("$", "")

    try:
        return sp.sympify(
            expression,
            locals=GRAPH_SAFE_LOCALS,
        )
    except Exception:
        return None


def render_graph(
    graph_spec: Dict[str, Any],
) -> Optional[bytes]:

    if not isinstance(graph_spec, dict):
        return None

    if not graph_spec.get("required", False):
        return None

    graph_type = str(
        graph_spec.get(
            "graph_type",
            "none",
        )
    ).lower().strip()

    if graph_type == "none":
        return None

    try:
        x = sp.symbols("x")

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        try:
            x_min = float(
                graph_spec.get(
                    "x_min",
                    -10,
                )
            )
        except Exception:
            x_min = -10.0

        try:
            x_max = float(
                graph_spec.get(
                    "x_max",
                    10,
                )
            )
        except Exception:
            x_max = 10.0

        if x_min >= x_max:
            x_min, x_max = -10.0, 10.0

        if graph_type in (
            "line",
            "quadratic",
            "function",
        ):
            expression = graph_spec.get(
                "expression",
                "",
            )

            sympy_expr = _safe_graph_expression(
                expression
            )

            if sympy_expr is None:
                plt.close(fig)
                return None

            function = sp.lambdify(
                x,
                sympy_expr,
                modules="numpy",
            )

            x_data = np.linspace(
                x_min,
                x_max,
                1000,
            )

            angle_unit = str(
                graph_spec.get(
                    "angle_unit",
                    "radians",
                )
            ).lower()

            evaluation_x = (
                np.pi / 180.0 * x_data
                if angle_unit == "degrees"
                else x_data
            )

            try:
                y_data = function(
                    evaluation_x
                )

                y_data = np.asarray(
                    y_data,
                    dtype=float,
                )
            except Exception:
                plt.close(fig)
                return None

            y_data = np.where(
                np.isfinite(y_data),
                y_data,
                np.nan,
            )

            ax.plot(
                x_data,
                y_data,
                linewidth=2,
            )

        elif graph_type == "points":
            x_values = graph_spec.get(
                "x_values",
                [],
            )

            y_values = graph_spec.get(
                "y_values",
                [],
            )

            if (
                not isinstance(x_values, list)
                or not isinstance(y_values, list)
                or len(x_values) == 0
                or len(x_values) != len(y_values)
            ):
                plt.close(fig)
                return None

            x_data = np.asarray(
                [float(v) for v in x_values]
            )

            y_data = np.asarray(
                [float(v) for v in y_values]
            )

            ax.scatter(
                x_data,
                y_data,
                s=45,
            )

            if graph_spec.get(
                "connect_points",
                False,
            ):
                ax.plot(
                    x_data,
                    y_data,
                    linewidth=2,
                )

        else:
            plt.close(fig)
            return None

        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.set_xlabel(
            str(
                graph_spec.get(
                    "x_label",
                    "x",
                )
            )
        )

        ax.set_ylabel(
            str(
                graph_spec.get(
                    "y_label",
                    "y",
                )
            )
        )

        y_min = graph_spec.get("y_min")
        y_max = graph_spec.get("y_max")

        try:
            if (
                y_min is not None
                and y_max is not None
            ):
                y_min = float(y_min)
                y_max = float(y_max)

                if y_min < y_max:
                    ax.set_ylim(
                        y_min,
                        y_max,
                    )
        except Exception:
            pass

        ax.set_xlim(
            x_min,
            x_max,
        )

        title = str(
            graph_spec.get(
                "title",
                "",
            )
        ).strip()

        if title:
            ax.set_title(title)

        plt.tight_layout()

        output = io.BytesIO()

        fig.savefig(
            output,
            format="png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        return output.getvalue()

    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

        return None


# ============================================================
# CONSTRUCTION ENGINE V1
# ============================================================

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    try:
        return float(value)
    except Exception:
        return default


def _point_from_spec(
    points: Dict[str, Any],
    name: str,
    default: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[float, float]]:

    value = points.get(name)

    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
    ):
        x = _safe_float(value[0])
        y = _safe_float(value[1])

        if x is not None and y is not None:
            return (x, y)

    return default


def _distance(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:

    return math.hypot(
        b[0] - a[0],
        b[1] - a[1],
    )


def _unit(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[float, float]:

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    length = math.hypot(dx, dy)

    if length == 0:
        return (1.0, 0.0)

    return (
        dx / length,
        dy / length,
    )


def _add(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[float, float]:

    return (
        a[0] + b[0],
        a[1] + b[1],
    )


def _scale(
    a: Tuple[float, float],
    scalar: float,
) -> Tuple[float, float]:

    return (
        a[0] * scalar,
        a[1] * scalar,
    )


def _draw_segment(
    ax,
    a: Tuple[float, float],
    b: Tuple[float, float],
    **kwargs,
):

    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        **kwargs,
    )


def _draw_arc(
    ax,
    center: Tuple[float, float],
    radius: float,
    theta1: float,
    theta2: float,
    **kwargs,
):

    patch = Arc(
        center,
        2 * radius,
        2 * radius,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        **kwargs,
    )

    ax.add_patch(patch)


def _label_point(
    ax,
    name: str,
    point: Tuple[float, float],
):

    ax.annotate(
        name,
        point,
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
    )


def _circle_intersections(
    c1: Tuple[float, float],
    r1: float,
    c2: Tuple[float, float],
    r2: float,
) -> List[Tuple[float, float]]:

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]

    d = math.hypot(dx, dy)

    if d == 0:
        return []

    if d > r1 + r2:
        return []

    if d < abs(r1 - r2):
        return []

    if d == 0:
        return []

    a = (
        r1 * r1
        - r2 * r2
        + d * d
    ) / (2 * d)

    h_squared = r1 * r1 - a * a

    if h_squared < -1e-9:
        return []

    h = math.sqrt(
        max(0.0, h_squared)
    )

    xm = c1[0] + a * dx / d
    ym = c1[1] + a * dy / d

    rx = -dy * h / d
    ry = dx * h / d

    p1 = (
        xm + rx,
        ym + ry,
    )

    p2 = (
        xm - rx,
        ym - ry,
    )

    return [p1, p2]


def _canonical_points_for_angle_bisector(
    spec: Dict[str, Any],
):

    points = spec.get("points", {})

    A = _point_from_spec(
        points,
        "A",
        (0.0, 0.0),
    )

    B = _point_from_spec(
        points,
        "B",
        (8.0, 0.0),
    )

    C = _point_from_spec(
        points,
        "C",
        (4.0, 6.0),
    )

    return A, B, C


def render_construction(
    construction_spec: Dict[str, Any],
) -> Optional[bytes]:

    if not isinstance(
        construction_spec,
        dict,
    ):
        return None

    if not construction_spec.get(
        "required",
        False,
    ):
        return None

    construction_type = str(
        construction_spec.get(
            "construction_type",
            "none",
        )
    ).lower().strip()

    if construction_type == "none":
        return None

    try:
        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        points = construction_spec.get(
            "points",
            {},
        )

        lengths = construction_spec.get(
            "lengths",
            {},
        )

        angles = construction_spec.get(
            "angles",
            {},
        )

        show_arcs = construction_spec.get(
            "show_arcs",
            True,
        )

        show_lines = construction_spec.get(
            "show_construction_lines",
            True,
        )

        label_points = construction_spec.get(
            "label_points",
            True,
        )

        # ----------------------------------------------------
        # ANGLE BISECTOR
        # ----------------------------------------------------

        if construction_type == "angle_bisector":

            A, B, C = (
                _canonical_points_for_angle_bisector(
                    construction_spec
                )
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            _draw_segment(
                ax,
                A,
                C,
                linewidth=2,
            )

            radius = min(
                _distance(A, B),
                _distance(A, C),
            ) * 0.35

            if radius <= 0:
                raise ValueError(
                    "Invalid angle-bisector geometry."
                )

            u1 = _unit(A, B)
            u2 = _unit(A, C)

            D = _add(
                A,
                _scale(u1, radius),
            )

            E = _add(
                A,
                _scale(u2, radius),
            )

            intersections = _circle_intersections(
                D,
                radius,
                E,
                radius,
            )

            if not intersections:
                raise ValueError(
                    "Could not construct angle bisector."
                )

            # Select intersection furthest from A.
            F = max(
                intersections,
                key=lambda p: _distance(A, p),
            )

            if show_arcs:
                theta_b = math.degrees(
                    math.atan2(
                        B[1] - A[1],
                        B[0] - A[0],
                    )
                )

                theta_c = math.degrees(
                    math.atan2(
                        C[1] - A[1],
                        C[0] - A[0],
                    )
                )

                _draw_arc(
                    ax,
                    A,
                    radius,
                    min(theta_b, theta_c),
                    max(theta_b, theta_c),
                    linewidth=1.5,
                )

                for centre in (D, E):
                    theta_f = math.degrees(
                        math.atan2(
                            F[1] - centre[1],
                            F[0] - centre[0],
                        )
                    )

                    _draw_arc(
                        ax,
                        centre,
                        radius,
                        theta_f - 35,
                        theta_f + 35,
                        linewidth=1.2,
                    )

            _draw_segment(
                ax,
                A,
                F,
                linewidth=2,
                linestyle="--" if show_lines else "-",
            )

            if label_points:
                for name, p in (
                    ("A", A),
                    ("B", B),
                    ("C", C),
                ):
                    _label_point(
                        ax,
                        name,
                        p,
                    )

        # ----------------------------------------------------
        # PERPENDICULAR BISECTOR
        # ----------------------------------------------------

        elif construction_type == "perpendicular_bisector":

            A = _point_from_spec(
                points,
                "A",
                (0.0, 0.0),
            )

            B = _point_from_spec(
                points,
                "B",
                None,
            )

            if B is None:
                length = _safe_float(
                    lengths.get("AB"),
                    10.0,
                )

                B = (
                    length,
                    0.0,
                )

            AB = _distance(A, B)

            if AB <= 0:
                raise ValueError(
                    "Invalid segment."
                )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            radius = max(
                AB * 0.7,
                AB * 0.55,
            )

            intersections = _circle_intersections(
                A,
                radius,
                B,
                radius,
            )

            if len(intersections) < 2:
                raise ValueError(
                    "Could not construct perpendicular bisector."
                )

            P1, P2 = intersections[:2]

            _draw_segment(
                ax,
                P1,
                P2,
                linewidth=2,
                linestyle="--",
            )

            if show_arcs:
                for centre in (A, B):
                    _draw_arc(
                        ax,
                        centre,
                        radius,
                        -70,
                        70,
                        linewidth=1.2,
                    )

                    _draw_arc(
                        ax,
                        centre,
                        radius,
                        110,
                        250,
                        linewidth=1.2,
                    )

            if label_points:
                _label_point(ax, "A", A)
                _label_point(ax, "B", B)

        # ----------------------------------------------------
        # PERPENDICULAR FROM POINT
        # ----------------------------------------------------

        elif construction_type == "perpendicular_from_point":

            A = _point_from_spec(
                points,
                "A",
                (0.0, 0.0),
            )

            B = _point_from_spec(
                points,
                "B",
                (10.0, 0.0),
            )

            P = _point_from_spec(
                points,
                "P",
                (4.0, 5.0),
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            # Foot of perpendicular.
            dx = B[0] - A[0]
            dy = B[1] - A[1]

            denominator = dx * dx + dy * dy

            if denominator == 0:
                raise ValueError(
                    "Invalid construction line."
                )

            t = (
                (P[0] - A[0]) * dx
                + (P[1] - A[1]) * dy
            ) / denominator

            H = (
                A[0] + t * dx,
                A[1] + t * dy,
            )

            PH = _distance(P, H)

            if PH <= 0:
                raise ValueError(
                    "Point must not lie on the construction line."
                )

            # Compass-style auxiliary points on AB.
            r = min(
                PH * 0.55,
                max(
                    0.5,
                    _distance(A, B) * 0.25,
                ),
            )

            line_unit = _unit(A, B)

            Q = _add(
                H,
                _scale(line_unit, r),
            )

            R = _add(
                H,
                _scale(line_unit, -r),
            )

            # Intersections of equal arcs from Q and R.
            arc_radius = max(
                _distance(Q, P),
                _distance(R, P),
            )

            intersections = _circle_intersections(
                Q,
                arc_radius,
                R,
                arc_radius,
            )

            if intersections:
                construction_point = max(
                    intersections,
                    key=lambda x: _distance(P, x),
                )
            else:
                construction_point = (
                    P[0] + (P[0] - H[0]),
                    P[1] + (P[1] - H[1]),
                )

            if show_arcs:
                # Arc from P crossing the line.
                theta_q = math.degrees(
                    math.atan2(
                        Q[1] - P[1],
                        Q[0] - P[0],
                    )
                )

                theta_r = math.degrees(
                    math.atan2(
                        R[1] - P[1],
                        R[0] - P[0],
                    )
                )

                _draw_arc(
                    ax,
                    P,
                    r,
                    min(theta_q, theta_r),
                    max(theta_q, theta_r),
                    linewidth=1.2,
                )

                for centre in (Q, R):
                    theta_p = math.degrees(
                        math.atan2(
                            P[1] - centre[1],
                            P[0] - centre[0],
                        )
                    )

                    _draw_arc(
                        ax,
                        centre,
                        arc_radius,
                        theta_p - 30,
                        theta_p + 30,
                        linewidth=1.2,
                    )

            _draw_segment(
                ax,
                P,
                H,
                linewidth=2,
            )

            if label_points:
                _label_point(ax, "A", A)
                _label_point(ax, "B", B)
                _label_point(ax, "P", P)
                _label_point(ax, "H", H)

        # ----------------------------------------------------
        # TRIANGLE FROM SIDES
        # ----------------------------------------------------

        elif construction_type == "triangle_sides":

            A = _point_from_spec(
                points,
                "A",
                (0.0, 0.0),
            )

            B = _point_from_spec(
                points,
                "B",
                None,
            )

            AB = _safe_float(
                lengths.get("AB"),
                8.0,
            )

            BC = _safe_float(
                lengths.get("BC"),
                6.0,
            )

            CA = _safe_float(
                lengths.get("CA"),
                5.0,
            )

            if B is None:
                B = (
                    AB,
                    0.0,
                )

            intersections = _circle_intersections(
                A,
                CA,
                B,
                BC,
            )

            if not intersections:
                raise ValueError(
                    "The supplied side lengths cannot form a triangle."
                )

            C = max(
                intersections,
                key=lambda p: p[1],
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            _draw_segment(
                ax,
                B,
                C,
                linewidth=2,
            )

            _draw_segment(
                ax,
                C,
                A,
                linewidth=2,
            )

            if show_arcs:
                theta = np.linspace(
                    0,
                    360,
                    361,
                )

                ax.plot(
                    A[0] + CA * np.cos(
                        np.radians(theta)
                    ),
                    A[1] + CA * np.sin(
                        np.radians(theta)
                    ),
                    linewidth=1,
                    linestyle="--",
                )

                ax.plot(
                    B[0] + BC * np.cos(
                        np.radians(theta)
                    ),
                    B[1] + BC * np.sin(
                        np.radians(theta)
                    ),
                    linewidth=1,
                    linestyle="--",
                )

            if label_points:
                _label_point(ax, "A", A)
                _label_point(ax, "B", B)
                _label_point(ax, "C", C)

        # ----------------------------------------------------
        # TRIANGLE FROM BASE ANGLES
        # ----------------------------------------------------

        elif construction_type == "triangle_base_angles":

            A = _point_from_spec(
                points,
                "A",
                (0.0, 0.0),
            )

            B = _point_from_spec(
                points,
                "B",
                None,
            )

            base = _safe_float(
                lengths.get("AB"),
                10.0,
            )

            if B is None:
                B = (
                    base,
                    0.0,
                )

            angle_A = _safe_float(
                angles.get("A"),
                60.0,
            )

            angle_B = _safe_float(
                angles.get("B"),
                45.0,
            )

            rad_A = math.radians(
                angle_A
            )

            rad_B = math.radians(
                180.0 - angle_B
            )

            direction_A = (
                math.cos(rad_A),
                math.sin(rad_A),
            )

            direction_B = (
                math.cos(rad_B),
                math.sin(rad_B),
            )

            denom = (
                direction_A[0]
                * direction_B[1]
                - direction_A[1]
                * direction_B[0]
            )

            if abs(denom) < 1e-9:
                raise ValueError(
                    "Base angles do not form a valid triangle."
                )

            bx = B[0] - A[0]
            by = B[1] - A[1]

            t = (
                bx * direction_B[1]
                - by * direction_B[0]
            ) / denom

            C = (
                A[0] + t * direction_A[0],
                A[1] + t * direction_A[1],
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            _draw_segment(
                ax,
                A,
                C,
                linewidth=2,
            )

            _draw_segment(
                ax,
                B,
                C,
                linewidth=2,
            )

            if show_lines:
                # Construction rays extended slightly.
                ext = max(
                    base * 0.15,
                    1.0,
                )

                A_ext = (
                    A[0] + direction_A[0] * ext,
                    A[1] + direction_A[1] * ext,
                )

                B_ext = (
                    B[0] + direction_B[0] * ext,
                    B[1] + direction_B[1] * ext,
                )

                _draw_segment(
                    ax,
                    A,
                    A_ext,
                    linestyle="--",
                    linewidth=1,
                )

                _draw_segment(
                    ax,
                    B,
                    B_ext,
                    linestyle="--",
                    linewidth=1,
                )

            if label_points:
                _label_point(ax, "A", A)
                _label_point(ax, "B", B)
                _label_point(ax, "C", C)

        # ----------------------------------------------------
        # LOCUS CIRCLE
        # ----------------------------------------------------

        elif construction_type == "locus_circle":

            centre = _point_from_spec(
                points,
                "A",
                None,
            )

            if centre is None:
                centre = _point_from_spec(
                    points,
                    "O",
                    (0.0, 0.0),
                )

            radius = _safe_float(
                construction_spec.get(
                    "radius"
                ),
                None,
            )

            if radius is None:
                radius = _safe_float(
                    lengths.get("r"),
                    5.0,
                )

            if radius <= 0:
                raise ValueError(
                    "Invalid circle radius."
                )

            theta = np.linspace(
                0,
                2 * np.pi,
                500,
            )

            ax.plot(
                centre[0]
                + radius * np.cos(theta),
                centre[1]
                + radius * np.sin(theta),
                linewidth=2,
            )

            ax.scatter(
                [centre[0]],
                [centre[1]],
                s=35,
            )

            if label_points:
                _label_point(
                    ax,
                    "O",
                    centre,
                )

        # ----------------------------------------------------
        # LOCUS PERPENDICULAR BISECTOR
        # ----------------------------------------------------

        elif (
            construction_type
            == "locus_perpendicular_bisector"
        ):

            A = _point_from_spec(
                points,
                "A",
                (0.0, 0.0),
            )

            B = _point_from_spec(
                points,
                "B",
                None,
            )

            if B is None:
                length = _safe_float(
                    lengths.get("AB"),
                    10.0,
                )

                B = (
                    length,
                    0.0,
                )

            AB = _distance(A, B)

            if AB <= 0:
                raise ValueError(
                    "Invalid segment."
                )

            midpoint = (
                (A[0] + B[0]) / 2,
                (A[1] + B[1]) / 2,
            )

            u = _unit(A, B)

            perpendicular = (
                -u[1],
                u[0],
            )

            extension = AB * 0.8

            P1 = _add(
                midpoint,
                _scale(
                    perpendicular,
                    extension,
                ),
            )

            P2 = _add(
                midpoint,
                _scale(
                    perpendicular,
                    -extension,
                ),
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            _draw_segment(
                ax,
                P1,
                P2,
                linewidth=2,
                linestyle="--",
            )

            radius = AB * 0.65

            if show_arcs:
                _draw_arc(
                    ax,
                    A,
                    radius,
                    0,
                    180,
                    linewidth=1.2,
                )

                _draw_arc(
                    ax,
                    B,
                    radius,
                    0,
                    180,
                    linewidth=1.2,
                )

                _draw_arc(
                    ax,
                    A,
                    radius,
                    180,
                    360,
                    linewidth=1.2,
                )

                _draw_arc(
                    ax,
                    B,
                    radius,
                    180,
                    360,
                    linewidth=1.2,
                )

            if label_points:
                _label_point(ax, "A", A)
                _label_point(ax, "B", B)

        else:
            plt.close(fig)
            return None

        # ----------------------------------------------------
        # FINISH CONSTRUCTION FIGURE
        # ----------------------------------------------------

        title = str(
            construction_spec.get(
                "title",
                "",
            )
        ).strip()

        if title:
            ax.set_title(title)

        ax.set_aspect(
            "equal",
            adjustable="datalim",
        )

        ax.grid(
            True,
            alpha=0.15,
        )

        ax.set_xlabel(
            "Construction geometry"
        )

        ax.set_ylabel(
            ""
        )

        plt.tight_layout()

        output = io.BytesIO()

        fig.savefig(
            output,
            format="png",
            dpi=170,
            bbox_inches="tight",
        )

        plt.close(fig)

        return output.getvalue()

    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

        return None


# ============================================================
# COMPLETENESS CHECK
# ============================================================

def completeness_check(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    expected = []
    solved = []

    for page_result in results:

        page_number = str(
            page_result.get(
                "page_number",
                "",
            )
        )

        for question in page_result.get(
            "questions",
            [],
        ):

            number = str(
                question.get(
                    "number",
                    "",
                )
            ).strip()

            if number:
                expected.append(
                    f"{page_number}:{number}"
                )

        for solution in page_result.get(
            "solutions",
            [],
        ):

            number = str(
                solution.get(
                    "question_number",
                    "",
                )
            ).strip()

            if number:
                solved.append(
                    f"{page_number}:{number}"
                )

    expected_set = set(expected)
    solved_set = set(solved)

    missing_keys = (
        expected_set
        - solved_set
    )

    missing_questions = []

    for key in sorted(missing_keys):
        try:
            page, number = key.split(
                ":",
                1,
            )

            missing_questions.append(
                f"Page {page}, Q{number}"
            )
        except Exception:
            missing_questions.append(key)

    return {
        "questions_detected": len(
            expected_set
        ),
        "questions_solved": len(
            expected_set & solved_set
        ),
        "missing_questions": missing_questions,
        "complete": len(
            missing_questions
        ) == 0,
    }


# ============================================================
# DISPLAY QUESTION
# ============================================================

def display_question(
    image_bytes: bytes,
    question: Dict[str, Any],
    solution: Dict[str, Any],
):

    number = str(
        question.get(
            "number",
            solution.get(
                "question_number",
                "Unknown",
            ),
        )
    )

    st.markdown(
        f"### 📌 Question {number}"
    )

    bbox = question.get("bbox")

    crop = None

    if bbox:
        crop = crop_original_question(
            image_bytes,
            bbox,
        )

    if crop:
        st.image(
            crop,
            caption=f"Original Question {number}",
            width="stretch",
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
            "medium",
        )
    ).lower()

    if confidence == "high":
        st.success("Confidence: HIGH")
    elif confidence == "medium":
        st.warning("Confidence: MEDIUM")
    else:
        st.error("Confidence: LOW")

    reason = solution.get(
        "confidence_reason",
        "",
    )

    if reason:
        st.caption(
            "Verification: "
            + str(reason)
        )

    verification = solution.get(
        "verification",
        {},
    )

    if isinstance(
        verification,
        dict,
    ):

        if verification.get(
            "verified",
            False,
        ):
            st.caption(
                "✓ Independent solution verification completed."
            )

        if verification.get(
            "mathematical_error",
            False,
        ):
            st.error(
                "Mathematical verification found a "
                "possible genuine error requiring review."
            )

        if verification.get(
            "visual_ambiguity",
            False,
        ):
            st.warning(
                "Visual verification found essential "
                "information that may be ambiguous."
            )

    visual_dependency = solution.get(
        "visual_dependency",
        "none",
    )

    if visual_dependency in (
        "medium",
        "high",
    ):
        st.info(
            f"👁️ Visual dependency: "
            f"**{visual_dependency}**\n\n"
            + str(
                solution.get(
                    "visual_check",
                    "",
                )
            )
        )

    if solution.get("method"):
        st.markdown("**Method**")
        render_math_text(
            solution["method"]
        )

    st.markdown("**Working**")

    working = solution.get(
        "working",
        [],
    )

    if isinstance(
        working,
        list,
    ):
        for index, step in enumerate(
            working,
            1,
        ):
            st.markdown(
                f"**{index}.**"
            )

            render_math_text(step)

    else:
        render_math_text(working)

    final_answer = solution.get(
        "final_answer"
    )

    if final_answer:
        st.markdown("**Final Answer**")
        render_math_text(final_answer)

    # ========================================================
    # GENERATED GRAPH
    # ========================================================

    graph_spec = solution.get(
        "graph_spec",
        {},
    )

    if (
        isinstance(graph_spec, dict)
        and graph_spec.get(
            "required",
            False,
        )
    ):

        st.markdown(
            "### 📈 Generated Graph"
        )

        graph_png = render_graph(
            graph_spec
        )

        if graph_png:
            st.image(
                graph_png,
                caption=(
                    f"Mathematical graph for "
                    f"Question {number}"
                ),
                width="stretch",
            )

            st.caption(
                "📐 Graph generated mathematically "
                "from the question's verified "
                "graph specification."
            )

        else:
            st.warning(
                "The question requires a graph, "
                "but the graph specification could "
                "not be rendered."
            )

    # ========================================================
    # CONSTRUCTION V1
    # ========================================================

    construction_spec = solution.get(
        "construction_spec",
        {},
    )

    if (
        isinstance(
            construction_spec,
            dict,
        )
        and construction_spec.get(
            "required",
            False,
        )
    ):

        st.markdown(
            "### 📐 Generated Construction"
        )

        construction_png = render_construction(
            construction_spec
        )

        if construction_png:

            st.image(
                construction_png,
                caption=(
                    f"Geometric construction for "
                    f"Question {number}"
                ),
                width="stretch",
            )

            st.caption(
                "📐 Construction generated mathematically "
                "from the question's verified construction "
                "specification."
            )

        else:

            st.warning(
                "The question requires a construction, "
                "but the construction specification "
                "could not be rendered."
            )

    # ========================================================
    # MARKING SCHEME
    # ========================================================

    marking = solution.get(
        "marking_scheme",
        [],
    )

    if marking:

        st.markdown(
            "**Mark Allocation**"
        )

        for item in marking:

            if not isinstance(
                item,
                dict,
            ):
                continue

            marks = item.get(
                "marks",
                "",
            )

            point = item.get(
                "point",
                "",
            )

            st.markdown(
                f"- **{marks} mark(s):**"
            )

            render_math_text(point)

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
    ),
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
        use_container_width=True,
    ):

        st.session_state.analysis_results = []
        st.session_state.page_images = []
        st.session_state.paper_name = uploaded.name

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

                st.exception(exc)

                st.stop()

        if not page_images:

            st.error(
                "No pages were found in the PDF."
            )

            st.stop()

        st.session_state.page_images = page_images

        st.success(
            f"Rendered {len(page_images)} "
            "original pages."
        )

        progress = st.progress(0)

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
                expanded=False,
            ):

                try:

                    page_analysis = analyse_page(
                        image_bytes,
                        page_number,
                        page_texts[index],
                    )

                except Exception as exc:

                    page_analysis = {
                        "questions": [],
                        "visual_warnings": [
                            str(exc)
                        ],
                    }

            questions = page_analysis.get(
                "questions",
                [],
            )

            page_result = {
                "page_number": page_number,
                "questions": questions,
                "visual_warnings": page_analysis.get(
                    "visual_warnings",
                    [],
                ),
                "solutions": [],
            }

            # =================================================
            # QUESTIONS
            # =================================================

            for question in questions:

                number = str(
                    question.get(
                        "number",
                        "",
                    )
                ).strip()

                if not number:
                    continue

                # ---------------------------------------------
                # ORIGINAL QUESTION CROP
                # ---------------------------------------------

                question_crop = None

                bbox = question.get(
                    "bbox"
                )

                if bbox:

                    question_crop = (
                        crop_original_question(
                            image_bytes,
                            bbox,
                        )
                    )

                # ---------------------------------------------
                # SOLVE
                # ---------------------------------------------

                with st.status(
                    f"Solving Question "
                    f"{number}...",
                    expanded=False,
                ):

                    solution = solve_question(
                        page_image_bytes=image_bytes,
                        question_image_bytes=question_crop,
                        page_number=page_number,
                        question=question,
                        extracted_text=page_texts[index],
                    )

                # ---------------------------------------------
                # VERIFY
                # ---------------------------------------------

                with st.status(
                    f"Verifying Question "
                    f"{number}...",
                    expanded=False,
                ):

                    verification = verify_solution(
                        page_image_bytes=image_bytes,
                        question_image_bytes=question_crop,
                        page_number=page_number,
                        question=question,
                        solution=solution,
                    )

                    solution = apply_verification(
                        solution,
                        verification,
                    )

                page_result[
                    "solutions"
                ].append(solution)

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
            "✅ Clean marking-scheme generation, "
            "independent verification and visual "
            "construction processing completed."
        )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.analysis_results


if results:

    st.divider()

    st.header(
        "📄 Original Paper + "
        "Question-by-Question Marking Scheme"
    )

    st.info(
        "The original examination pages remain "
        "the visual source of truth. Mwalimu AI "
        "generates mathematical working, graphs "
        "and constructions underneath them."
    )

    # ========================================================
    # COMPLETENESS
    # ========================================================

    check = completeness_check(results)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Questions detected",
            check["questions_detected"],
        )

    with col2:
        st.metric(
            "Questions solved",
            check["questions_solved"],
        )

    with col3:
        st.metric(
            "Missing",
            len(
                check[
                    "missing_questions"
                ]
            ),
        )

    if check["missing_questions"]:

        st.warning(
            "Questions not yet solved: "
            + ", ".join(
                check["missing_questions"]
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
            width="stretch",
        )

        st.caption(
            "🔒 Original page preserved — "
            "not reconstructed by AI."
        )

        warnings = page_result.get(
            "visual_warnings",
            [],
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
            [],
        ):

            number = str(
                solution.get(
                    "question_number",
                    "",
                )
            ).strip()

            if number:
                solution_map[number] = solution

        questions = page_result.get(
            "questions",
            [],
        )

        for question in questions:

            number = str(
                question.get(
                    "number",
                    "",
                )
            ).strip()

            if not number:
                continue

            solution = solution_map.get(
                number
            )

            if solution is None:

                solution = {
                    "question_number": number,
                    "confidence": "medium",
                    "confidence_reason": (
                        "No solution was generated."
                    ),
                    "working": [],
                    "marking_scheme": [],
                    "construction_spec": (
                        default_construction_spec()
                    ),
                    "warning": (
                        "No solution was generated."
                    ),
                }

            st.divider()

            display_question(
                original_page,
                question,
                solution,
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    st.divider()

    st.header(
        "📊 Visual Marking Report"
    )

    visual_questions = []

    construction_questions = []

    for page_result in results:

        for question in page_result.get(
            "questions",
            [],
        ):

            number = str(
                question.get(
                    "number",
                    "",
                )
            )

            if (
                question.get(
                    "has_diagram",
                    False,
                )
                or question.get(
                    "has_graph",
                    False,
                )
                or question.get(
                    "has_construction",
                    False,
                )
            ):

                visual_questions.append(
                    number
                )

            if question.get(
                "has_construction",
                False,
            ):

                construction_questions.append(
                    number
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

    if construction_questions:

        st.info(
            "📐 Construction questions detected: "
            + ", ".join(
                construction_questions
            )
        )

    # ========================================================
    # MATHS DIAGNOSTIC
    # ========================================================

    maths_features = {
        "surds": 0,
        "integration": 0,
        "formula_subject": 0,
    }

    for page_result in results:

        for question in page_result.get(
            "questions",
            [],
        ):

            summary = str(
                question.get(
                    "visible_text_summary",
                    "",
                )
            ).lower()

            features = " ".join(
                str(x).lower()
                for x in question.get(
                    "important_visual_features",
                    [],
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
                    "√",
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
                    "integral",
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
                    "formula",
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

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Surd questions",
                maths_features[
                    "surds"
                ],
            )

        with m2:
            st.metric(
                "Integration questions",
                maths_features[
                    "integration"
                ],
            )

        with m3:
            st.metric(
                "Formula questions",
                maths_features[
                    "formula_subject"
                ],
            )

    # ========================================================
    # JSON DOWNLOAD
    # ========================================================

    json_output = json.dumps(
        results,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "⬇️ Download AI analysis (JSON)",
        data=json_output,
        file_name=(
            "mwalimu_ai_clean_visual_marking_analysis.json"
        ),
        mime="application/json",
        use_container_width=True,
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

        **7. Graphs are generated mathematically when required**

        **8. Geometric constructions are generated
        mathematically when required**

        **9. The solution is independently verified
        against the original question**

        **10. Low confidence is reserved for genuine
        mathematical errors or essential visual ambiguity**

        **11. Original pages remain untouched**

        **12. Marking points are displayed under each question**
        """
    )
