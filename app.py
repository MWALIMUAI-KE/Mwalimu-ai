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
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# ============================================================
# MWALIMU AI — MVP3.2 VISUAL MARKING
# FINAL MATHS POLISH + CONSTRUCTION V1
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
        filetype="pdf"
    )

    try:
        for page in document:
            rect = page.rect

            scale = MAX_PAGE_DIMENSION / max(
                rect.width,
                rect.height
            )

            scale = min(scale, 2.5)

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

def extract_page_texts(pdf_bytes: bytes) -> List[str]:
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

            texts.append(text.strip())
    finally:
        document.close()

    return texts


# ============================================================
# IMAGE → DATA URL
# ============================================================

def image_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

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

If a mathematical construction is visible or requested, identify it explicitly.

A construction may include:
- angle bisector
- perpendicular bisector
- perpendicular from a point to a line
- triangle construction
- loci
- arcs used in construction
- parallel/perpendicular construction
- construction of an angle
- division of a line
- circle/arc construction

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
      "construction_description": "",
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

- use the ACTUAL visible labels
- use the ACTUAL visible measurements
- use the ACTUAL visible angles
- use the ACTUAL visible coordinates
- use the actual relationships shown
- do not invent missing dimensions
- do not assume a diagram is to scale unless the question says so


============================================================
MATHEMATICAL CONSTRUCTIONS
============================================================

If the question requires a mathematical construction:

DO NOT attempt to draw the construction using text characters.

Instead, identify the construction mathematically and return a
construction_spec so that the application can generate the geometry.

The application supports Construction V1:

- angle_bisector
- perpendicular_bisector
- perpendicular_from_point
- triangle_from_sides
- angle_construction
- line_division
- circle_or_arc

For a construction:

1. Identify the construction type.
2. Identify all visible/given points.
3. Identify all visible/given lengths.
4. Identify all visible/given angles.
5. Identify the line or segment being constructed.
6. Identify construction arcs where appropriate.
7. Identify labels.
8. Do not invent dimensions.
9. If the construction is insufficiently specified, set
   "required": false and explain why in the construction description.
10. If sufficiently specified, return a construction_spec.

The construction_spec structure is:

{
  "required": false,
  "construction_type": "none",
  "point_labels": [],
  "points": [],
  "lengths": [],
  "angles": [],
  "target_point": "",
  "base_line": [],
  "arc_data": [],
  "show_construction_arcs": true,
  "show_construction_lines": true,
  "show_labels": true,
  "title": ""
}

Point format:

{
  "label": "A",
  "x": 0,
  "y": 0
}

For triangle_from_sides, use:

{
  "required": true,
  "construction_type": "triangle_from_sides",
  "point_labels": ["A","B","C"],
  "points": [
    {"label":"A","x":0,"y":0},
    {"label":"B","x":5,"y":0}
  ],
  "lengths": [
    {"from":"A","to":"B","value":5},
    {"from":"A","to":"C","value":4},
    {"from":"B","to":"C","value":3}
  ],
  "angles": [],
  "target_point": "C",
  "base_line": ["A","B"],
  "arc_data": [],
  "show_construction_arcs": true,
  "show_construction_lines": true,
  "show_labels": true,
  "title": "Construction of triangle ABC"
}

For an angle bisector, identify the vertex and the two rays.

For a perpendicular bisector, identify the two endpoints of the
segment.

For a perpendicular from a point to a line, identify the point and
the two endpoints of the line segment where possible.

For construction arcs, the Python engine will calculate their
geometry from the supplied points and lengths.

Never place arbitrary Python code inside construction_spec.


============================================================
GRAPHING
============================================================

If the question requires a graph, independently identify and preserve:

- the equation or function
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

Do NOT claim that a graph has already been drawn by the AI.

Instead, return a graph_spec that allows the application to generate
the graph mathematically.

For PP2 V1, use only these graph types:

- line
- quadratic
- function
- points

Use canonical mathematical expressions such as:

2*x+3
x**2-4*x+3
sin(x)
cos(x)

Do not place arbitrary Python code inside graph_spec.

If the question does NOT require a graph:

"required": false

If it DOES require a graph:

"required": true


============================================================
CONFIDENCE
============================================================

Do NOT use LOW confidence merely because:

- the question is difficult
- the mathematics is advanced
- a diagram exists
- surds are present
- integration is present
- algebra is complicated
- LaTeX is used
- the answer is long
- the model is generally uncertain
- JSON formatting is imperfect
- an API/technical problem occurred

LOW confidence is allowed ONLY when:

A. There is a genuine mathematical error.

OR

B. Essential visual information is genuinely unreadable or ambiguous.

If the mathematics is correct and the necessary visual information is
readable, confidence should normally be HIGH.


============================================================
RETURN JSON ONLY
============================================================

{
  "question_number": "...",
  "method": "...",
  "working": ["...", "...", "..."],
  "final_answer": "...",

  "marking_scheme": [
    {
      "marks": 1,
      "point": "..."
    }
  ],

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
    "point_labels": [],
    "points": [],
    "lengths": [],
    "angles": [],
    "target_point": "",
    "base_line": [],
    "arc_data": [],
    "show_construction_arcs": true,
    "show_construction_lines": true,
    "show_labels": true,
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
CHECK 1 — QUESTION READING
============================================================

Confirm that:

- the correct question was solved
- all visible numbers were read correctly
- fractions were read correctly
- signs were read correctly
- powers and indices were read correctly
- roots and surds were read correctly
- brackets were read correctly
- integration symbols and limits were read correctly
- diagram labels were read correctly


============================================================
CHECK 2 — SURDS
============================================================

If surds are present:

- verify simplification
- verify multiplication/division
- verify collection of like surds
- verify rationalisation
- verify the final exact answer


============================================================
CHECK 3 — INTEGRATION
============================================================

If integration is present:

- verify the integrand
- verify the integration rule
- verify powers
- verify coefficients
- verify the constant of integration where required
- verify limits for definite integrals
- differentiate the result mentally/algebraically where useful


============================================================
CHECK 4 — MAKING SUBJECT OF FORMULA
============================================================

If algebraic rearrangement is present:

- verify every rearrangement
- verify signs
- verify denominators
- verify powers
- verify roots
- verify that the requested variable is actually the subject


============================================================
CHECK 5 — DIAGRAM
============================================================

If a diagram is involved:

- verify visible labels
- verify dimensions
- verify angles
- verify coordinates
- verify geometric relationships
- do not invent missing information


============================================================
CHECK 6 — CONSTRUCTION
============================================================

If a mathematical construction is involved:

Independently verify:

- construction type
- point labels
- supplied lengths
- supplied angles
- base line
- target point
- geometric relationships
- construction arcs
- perpendicular or bisector relationships
- triangle side relationships

A technical failure to render a construction is NOT itself a
mathematical error.

A wrong construction type, wrong supplied measurement, wrong
geometric relationship, or mathematically incorrect construction_spec
IS a genuine mathematical error.


============================================================
GRAPH VERIFICATION
============================================================

If the question requires a graph, independently verify:

- equation/function
- graph type
- domain
- supplied coordinates
- table values
- intercepts
- axis labels
- point requirements
- joining requirements
- general mathematical shape

A technical failure to render a graph is NOT itself a mathematical
error.


============================================================
CONFIDENCE RULE
============================================================

LOW confidence MUST NOT be used merely because:

- the question is difficult
- a diagram exists
- surds are present
- integration is present
- algebra is complicated
- LaTeX is present
- the solution is lengthy
- the verifier has general uncertainty
- formatting is imperfect
- a technical service failed

LOW is ONLY permitted when:

A. There is a genuine mathematical error.

OR

B. Essential visual information is genuinely unreadable or ambiguous.

If the mathematics is correct and the visual information is
sufficiently clear, return HIGH.


============================================================
RETURN JSON ONLY
============================================================

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

    text = text.strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
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
        min(x, image_width - 1)
    )

    y = max(
        0,
        min(y, image_height - 1)
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
            io.BytesIO(image_bytes)
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
            18,
            int(width * QUESTION_CROP_MARGIN)
        )

        margin_y = max(
            18,
            int(height * QUESTION_CROP_MARGIN)
        )

        crop = image.crop(
            (
                max(0, x - margin_x),
                max(0, y - margin_y),
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
                        "text": (
                            PAGE_ANALYSIS_PROMPT
                            + "\n\nPAGE NUMBER:\n"
                            + str(page_number)
                            + "\n\nSUPPORTING PDF TEXT:\n"
                            + extracted_text[:12000]
                        )
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            image_bytes
                        ),
                        "detail": "high"
                    }
                ]
            }
        ]
    )

    return parse_json_response(
        response.output_text
    )


# ============================================================
# DEFAULT GRAPH SPEC
# ============================================================

def default_graph_spec() -> Dict[str, Any]:
    return {
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
        "title": ""
    }


# ============================================================
# DEFAULT CONSTRUCTION SPEC
# ============================================================

def default_construction_spec() -> Dict[str, Any]:
    return {
        "required": False,
        "construction_type": "none",
        "point_labels": [],
        "points": [],
        "lengths": [],
        "angles": [],
        "target_point": "",
        "base_line": [],
        "arc_data": [],
        "show_construction_arcs": True,
        "show_construction_lines": True,
        "show_labels": True,
        "title": ""
    }


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
        + "\n\nCONSTRUCTION DESCRIPTION:\n"
        + str(
            question.get(
                "construction_description",
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

    if question_image_bytes:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "PRIMARY QUESTION IMAGE: "
                    "Read the exact mathematical expression "
                    "and construction information from this "
                    "isolated original question."
                )
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(
                    question_image_bytes
                ),
                "detail": "high"
            }
        )

    content.append(
        {
            "type": "input_text",
            "text": (
                "COMPLETE ORIGINAL PAGE: "
                "Use this to verify context, question "
                "continuation and surrounding diagram "
                "or construction information."
            )
        }
    )

    content.append(
        {
            "type": "input_image",
            "image_url": image_to_data_url(
                page_image_bytes
            ),
            "detail": "high"
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
                "question_number": question_number,
                "method": "",
                "working": [],
                "final_answer": response.output_text,
                "marking_scheme": [],
                "visual_dependency": "medium",
                "visual_check": "",
                "graph_spec": default_graph_spec(),
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
                )
            }

        result["question_number"] = question_number

        if not isinstance(
            result.get("graph_spec"),
            dict
        ):
            result["graph_spec"] = default_graph_spec()

        if not isinstance(
            result.get("construction_spec"),
            dict
        ):
            result["construction_spec"] = (
                default_construction_spec()
            )

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
            "graph_spec": default_graph_spec(),
            "construction_spec": default_construction_spec(),
            "confidence": "medium",
            "confidence_reason": (
                "The solution service was unavailable. "
                "This is not evidence of a mathematical error."
            ),
            "warning": str(exc)
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
                "image_url": image_to_data_url(
                    question_image_bytes
                ),
                "detail": "high"
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
            "image_url": image_to_data_url(
                page_image_bytes
            ),
            "detail": "high"
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
            "verified": False,
            "confidence": "medium",
            "reason": (
                "Verification response formatting issue. "
                "No mathematical error was established."
            ),
            "mathematical_error": False,
            "visual_ambiguity": False,
            "recommended_action": "accept"
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
            "technical_error": str(exc)
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
        ) is True
    )

    visual_ambiguity = (
        verification.get(
            "visual_ambiguity",
            False
        ) is True
    )

    verified = (
        verification.get(
            "verified",
            False
        ) is True
    )

    verifier_confidence = str(
        verification.get(
            "confidence",
            ""
        )
    ).lower().strip()

    if mathematical_error or visual_ambiguity:
        final_confidence = "low"

    elif verified and verifier_confidence == "high":
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
            "accept"
        )
    }

    return solution


# ============================================================
# LATEX / MATHEMATICAL RENDERING
# ============================================================

def _clean_math_text(text: str) -> str:
    """
    Clean common model-generated mathematical wrappers
    while preserving valid LaTeX.
    """

    text = str(text).strip()

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    ).replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"^\s*```(?:latex|tex|math)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    )

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

    text = re.sub(
        r"<br\s*/?>",
        "\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\$\$\s+",
        "$$",
        text
    )

    text = re.sub(
        r"\s+\$\$",
        "$$",
        text
    )

    return text.strip()


def render_math_text(text: Any):
    """
    Render AI-generated mathematical text safely.
    """

    if text is None:
        return

    text = _clean_math_text(text)

    if not text:
        return

    replacements = [
        (
            "−",
            "-"
        ),
        (
            "×",
            r"\times "
        ),
        (
            "÷",
            r"\div "
        ),
        (
            "°",
            r"^\circ"
        )
    ]

    for old, new in replacements:
        text = text.replace(
            old,
            new
        )

    # Simple square-root notation.
    text = re.sub(
        r"√\s*([A-Za-z0-9]+)",
        r"$\\sqrt{\1}$",
        text
    )

    # Simple exponent patterns.
    text = re.sub(
        r"(?<!\^)([A-Za-z])\^(-?\d+)(?!\})",
        r"$\1^{\2}$",
        text
    )

    # Simple subscript patterns.
    text = re.sub(
        r"(?<![_\\])([A-Za-z])_([0-9]+)",
        r"$\1_{\2}$",
        text
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
                stripped
            )
            and bool(
                re.search(
                    r"[A-Za-z0-9]",
                    stripped
                )
            )
        )

        if (
            looks_mathematical
            and len(stripped) < 180
        ):
            if re.search(
                r"^[A-Za-z\s]+$",
                stripped
            ) is None:
                rendered_lines.append(
                    "$" + stripped + "$"
                )
                continue

        rendered_lines.append(line)

    text = "\n".join(rendered_lines)

    st.markdown(text)


# ============================================================
# 📈 PP2 GRAPHING ENGINE V1
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
    """Safely convert a mathematical expression into SymPy."""

    if not expression:
        return None

    expression = str(
        expression
    ).strip()

    expression = re.sub(
        r"^\s*y\s*=\s*",
        "",
        expression,
        flags=re.IGNORECASE
    )

    expression = expression.replace(
        "^",
        "**"
    )

    expression = expression.replace(
        "$",
        ""
    )

    try:
        return sp.sympify(
            expression,
            locals=GRAPH_SAFE_LOCALS
        )
    except Exception:
        return None


def render_graph(
    graph_spec: Dict[str, Any]
) -> Optional[bytes]:

    if not isinstance(
        graph_spec,
        dict
    ):
        return None

    if not graph_spec.get(
        "required",
        False
    ):
        return None

    graph_type = str(
        graph_spec.get(
            "graph_type",
            "none"
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
                    -10
                )
            )
        except Exception:
            x_min = -10.0

        try:
            x_max = float(
                graph_spec.get(
                    "x_max",
                    10
                )
            )
        except Exception:
            x_max = 10.0

        if x_min >= x_max:
            x_min, x_max = -10.0, 10.0

        if graph_type in (
            "line",
            "quadratic",
            "function"
        ):

            expression = graph_spec.get(
                "expression",
                ""
            )

            sympy_expr = _safe_graph_expression(
                expression
            )

            if sympy_expr is None:
                plt.close(fig)
                return None

            try:
                function = sp.lambdify(
                    x,
                    sympy_expr,
                    modules="numpy"
                )
            except Exception:
                plt.close(fig)
                return None

            x_data = np.linspace(
                x_min,
                x_max,
                1000
            )

            angle_unit = str(
                graph_spec.get(
                    "angle_unit",
                    "radians"
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
                    dtype=float
                )
            except Exception:
                plt.close(fig)
                return None

            y_data = np.where(
                np.isfinite(y_data),
                y_data,
                np.nan
            )

            ax.plot(
                x_data,
                y_data,
                linewidth=2
            )

        elif graph_type == "points":

            x_values = graph_spec.get(
                "x_values",
                []
            )

            y_values = graph_spec.get(
                "y_values",
                []
            )

            if (
                not isinstance(
                    x_values,
                    list
                )
                or not isinstance(
                    y_values,
                    list
                )
            ):
                plt.close(fig)
                return None

            if (
                len(x_values) == 0
                or len(x_values) != len(y_values)
            ):
                plt.close(fig)
                return None

            try:
                x_data = np.asarray(
                    [float(v) for v in x_values]
                )

                y_data = np.asarray(
                    [float(v) for v in y_values]
                )
            except Exception:
                plt.close(fig)
                return None

            ax.scatter(
                x_data,
                y_data,
                s=45
            )

            if graph_spec.get(
                "connect_points",
                False
            ):
                ax.plot(
                    x_data,
                    y_data,
                    linewidth=2
                )

        else:
            plt.close(fig)
            return None

        ax.axhline(
            0,
            linewidth=1
        )

        ax.axvline(
            0,
            linewidth=1
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.set_xlabel(
            str(
                graph_spec.get(
                    "x_label",
                    "x"
                )
            )
        )

        ax.set_ylabel(
            str(
                graph_spec.get(
                    "y_label",
                    "y"
                )
            )
        )

        y_min = graph_spec.get(
            "y_min"
        )

        y_max = graph_spec.get(
            "y_max"
        )

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
                        y_max
                    )
        except Exception:
            pass

        ax.set_xlim(
            x_min,
            x_max
        )

        title = str(
            graph_spec.get(
                "title",
                ""
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
            bbox_inches="tight"
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
# 📐 MATHEMATICAL CONSTRUCTION ENGINE V1
# ============================================================

def _construction_point_map(
    construction_spec: Dict[str, Any]
) -> Dict[str, Tuple[float, float]]:

    point_map = {}

    points = construction_spec.get(
        "points",
        []
    )

    if not isinstance(
        points,
        list
    ):
        return point_map

    for point in points:
        if not isinstance(
            point,
            dict
        ):
            continue

        label = str(
            point.get(
                "label",
                ""
            )
        ).strip()

        if not label:
            continue

        try:
            px = float(
                point.get(
                    "x",
                    0
                )
            )

            py = float(
                point.get(
                    "y",
                    0
                )
            )

            point_map[label] = (
                px,
                py
            )

        except Exception:
            continue

    return point_map


def _distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:

    return float(
        np.hypot(
            p2[0] - p1[0],
            p2[1] - p1[1]
        )
    )


def _circle_intersections(
    c1: Tuple[float, float],
    r1: float,
    c2: Tuple[float, float],
    r2: float
) -> List[Tuple[float, float]]:

    x1, y1 = c1
    x2, y2 = c2

    dx = x2 - x1
    dy = y2 - y1

    d = float(
        np.hypot(
            dx,
            dy
        )
    )

    if d <= 1e-9:
        return []

    if d > r1 + r2 + 1e-9:
        return []

    if d < abs(r1 - r2) - 1e-9:
        return []

    a = (
        r1**2
        - r2**2
        + d**2
    ) / (2 * d)

    h_sq = r1**2 - a**2

    if h_sq < -1e-9:
        return []

    h = np.sqrt(
        max(
            0,
            h_sq
        )
    )

    xm = x1 + a * dx / d
    ym = y1 + a * dy / d

    rx = -dy * h / d
    ry = dx * h / d

    p1 = (
        xm + rx,
        ym + ry
    )

    p2 = (
        xm - rx,
        ym - ry
    )

    if _distance(p1, p2) < 1e-9:
        return [p1]

    return [
        p1,
        p2
    ]


def _draw_segment(
    ax,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    linewidth: float = 2,
    linestyle: str = "-"
):
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linewidth=linewidth,
        linestyle=linestyle
    )


def _draw_arc(
    ax,
    centre: Tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    linewidth: float = 1.4,
    linestyle: str = "--"
):

    angles = np.linspace(
        np.deg2rad(start_angle),
        np.deg2rad(end_angle),
        180
    )

    xs = (
        centre[0]
        + radius * np.cos(angles)
    )

    ys = (
        centre[1]
        + radius * np.sin(angles)
    )

    ax.plot(
        xs,
        ys,
        linewidth=linewidth,
        linestyle=linestyle
    )


def _label_points(
    ax,
    point_map: Dict[str, Tuple[float, float]]
):

    for label, point in point_map.items():
        ax.scatter(
            [point[0]],
            [point[1]],
            s=30
        )

        ax.annotate(
            label,
            (
                point[0],
                point[1]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold"
        )


def render_construction(
    construction_spec: Dict[str, Any]
) -> Optional[bytes]:

    if not isinstance(
        construction_spec,
        dict
    ):
        return None

    if not construction_spec.get(
        "required",
        False
    ):
        return None

    construction_type = str(
        construction_spec.get(
            "construction_type",
            "none"
        )
    ).lower().strip()

    if construction_type == "none":
        return None

    try:
        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        point_map = _construction_point_map(
            construction_spec
        )

        # ----------------------------------------------------
        # TRIANGLE FROM SIDES
        # ----------------------------------------------------

        if construction_type == "triangle_from_sides":

            lengths = construction_spec.get(
                "lengths",
                []
            )

            if not isinstance(
                lengths,
                list
            ):
                plt.close(fig)
                return None

            base_line = construction_spec.get(
                "base_line",
                []
            )

            if (
                not isinstance(
                    base_line,
                    list
                )
                or len(base_line) < 2
            ):
                plt.close(fig)
                return None

            a_label = str(
                base_line[0]
            )

            b_label = str(
                base_line[1]
            )

            if (
                a_label not in point_map
                or b_label not in point_map
            ):
                plt.close(fig)
                return None

            A = point_map[a_label]
            B = point_map[b_label]

            side_lookup = {}

            for item in lengths:
                if not isinstance(
                    item,
                    dict
                ):
                    continue

                p1 = str(
                    item.get(
                        "from",
                        ""
                    )
                )

                p2 = str(
                    item.get(
                        "to",
                        ""
                    )
                )

                try:
                    value = float(
                        item.get(
                            "value"
                        )
                    )
                except Exception:
                    continue

                side_lookup[
                    tuple(
                        sorted(
                            [p1, p2]
                        )
                    )
                ] = value

            target_label = str(
                construction_spec.get(
                    "target_point",
                    ""
                )
            ).strip()

            if not target_label:
                target_label = "C"

            key_ac = tuple(
                sorted(
                    [a_label, target_label]
                )
            )

            key_bc = tuple(
                sorted(
                    [b_label, target_label]
                )
            )

            if (
                key_ac not in side_lookup
                or key_bc not in side_lookup
            ):
                plt.close(fig)
                return None

            AC = side_lookup[key_ac]
            BC = side_lookup[key_bc]

            AB = _distance(
                A,
                B
            )

            if AB <= 0:
                plt.close(fig)
                return None

            intersections = _circle_intersections(
                A,
                AC,
                B,
                BC
            )

            if not intersections:
                plt.close(fig)
                return None

            # Prefer the point above the base.
            C_candidates = [
                p for p in intersections
                if p[1] >= 0
            ]

            if C_candidates:
                C = C_candidates[0]
            else:
                C = intersections[0]

            point_map[target_label] = C

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2.2
            )

            _draw_segment(
                ax,
                A,
                C,
                linewidth=2.2
            )

            _draw_segment(
                ax,
                B,
                C,
                linewidth=2.2
            )

            if construction_spec.get(
                "show_construction_arcs",
                True
            ):
                _draw_arc(
                    ax,
                    A,
                    AC,
                    -65,
                    65
                )

                angle_B = np.rad2deg(
                    np.arctan2(
                        C[1] - B[1],
                        C[0] - B[0]
                    )
                )

                _draw_arc(
                    ax,
                    B,
                    BC,
                    angle_B - 55,
                    angle_B + 55
                )

        # ----------------------------------------------------
        # PERPENDICULAR BISECTOR
        # ----------------------------------------------------

        elif construction_type == "perpendicular_bisector":

            base_line = construction_spec.get(
                "base_line",
                []
            )

            if (
                not isinstance(
                    base_line,
                    list
                )
                or len(base_line) < 2
            ):
                plt.close(fig)
                return None

            a_label = str(base_line[0])
            b_label = str(base_line[1])

            if (
                a_label not in point_map
                or b_label not in point_map
            ):
                plt.close(fig)
                return None

            A = point_map[a_label]
            B = point_map[b_label]

            length = _distance(
                A,
                B
            )

            if length <= 0:
                plt.close(fig)
                return None

            # Extend line to form the perpendicular bisector.
            midpoint = (
                (A[0] + B[0]) / 2,
                (A[1] + B[1]) / 2
            )

            dx = B[0] - A[0]
            dy = B[1] - A[1]

            nx = -dy / length
            ny = dx / length

            extent = length * 0.8

            P1 = (
                midpoint[0] - nx * extent,
                midpoint[1] - ny * extent
            )

            P2 = (
                midpoint[0] + nx * extent,
                midpoint[1] + ny * extent
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2.2
            )

            if construction_spec.get(
                "show_construction_lines",
                True
            ):
                _draw_segment(
                    ax,
                    P1,
                    P2,
                    linewidth=1.8
                )

            if construction_spec.get(
                "show_construction_arcs",
                True
            ):
                _draw_arc(
                    ax,
                    A,
                    length * 0.72,
                    0,
                    360
                )

                _draw_arc(
                    ax,
                    B,
                    length * 0.72,
                    0,
                    360
                )

            point_map["M"] = midpoint

        # ----------------------------------------------------
        # ANGLE BISECTOR
        # ----------------------------------------------------

        elif construction_type == "angle_bisector":

            base_line = construction_spec.get(
                "base_line",
                []
            )

            if (
                not isinstance(
                    base_line,
                    list
                )
                or len(base_line) < 3
            ):
                plt.close(fig)
                return None

            vertex_label = str(
                base_line[0]
            )

            ray1_label = str(
                base_line[1]
            )

            ray2_label = str(
                base_line[2]
            )

            if not all(
                label in point_map
                for label in [
                    vertex_label,
                    ray1_label,
                    ray2_label
                ]
            ):
                plt.close(fig)
                return None

            V = point_map[vertex_label]
            P = point_map[ray1_label]
            Q = point_map[ray2_label]

            v1 = np.array(
                [
                    P[0] - V[0],
                    P[1] - V[1]
                ],
                dtype=float
            )

            v2 = np.array(
                [
                    Q[0] - V[0],
                    Q[1] - V[1]
                ],
                dtype=float
            )

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 <= 1e-9 or n2 <= 1e-9:
                plt.close(fig)
                return None

            u1 = v1 / n1
            u2 = v2 / n2

            bisector = u1 + u2

            bisector_norm = np.linalg.norm(
                bisector
            )

            if bisector_norm <= 1e-9:
                plt.close(fig)
                return None

            bisector = (
                bisector
                / bisector_norm
            )

            ray_length = max(
                n1,
                n2
            )

            end = (
                V[0] + bisector[0] * ray_length,
                V[1] + bisector[1] * ray_length
            )

            _draw_segment(
                ax,
                V,
                P,
                linewidth=2.2
            )

            _draw_segment(
                ax,
                V,
                Q,
                linewidth=2.2
            )

            if construction_spec.get(
                "show_construction_lines",
                True
            ):
                _draw_segment(
                    ax,
                    V,
                    end,
                    linewidth=1.8
                )

            arc_radius = min(
                n1,
                n2
            ) * 0.28

            if (
                construction_spec.get(
                    "show_construction_arcs",
                    True
                )
                and arc_radius > 0
            ):

                angle1 = np.rad2deg(
                    np.arctan2(
                        v1[1],
                        v1[0]
                    )
                )

                angle2 = np.rad2deg(
                    np.arctan2(
                        v2[1],
                        v2[0]
                    )
                )

                start_angle = min(
                    angle1,
                    angle2
                )

                end_angle = max(
                    angle1,
                    angle2
                )

                if end_angle - start_angle > 180:
                    start_angle, end_angle = (
                        end_angle,
                        start_angle + 360
                    )

                _draw_arc(
                    ax,
                    V,
                    arc_radius,
                    start_angle,
                    end_angle
                )

        # ----------------------------------------------------
        # PERPENDICULAR FROM POINT
        # ----------------------------------------------------

        elif construction_type == "perpendicular_from_point":

            base_line = construction_spec.get(
                "base_line",
                []
            )

            target_label = str(
                construction_spec.get(
                    "target_point",
                    ""
                )
            ).strip()

            if (
                not isinstance(
                    base_line,
                    list
                )
                or len(base_line) < 2
                or target_label not in point_map
            ):
                plt.close(fig)
                return None

            A_label = str(base_line[0])
            B_label = str(base_line[1])

            if (
                A_label not in point_map
                or B_label not in point_map
            ):
                plt.close(fig)
                return None

            A = np.array(
                point_map[A_label],
                dtype=float
            )

            B = np.array(
                point_map[B_label],
                dtype=float
            )

            P = np.array(
                point_map[target_label],
                dtype=float
            )

            AB = B - A

            denominator = np.dot(
                AB,
                AB
            )

            if denominator <= 1e-9:
                plt.close(fig)
                return None

            t = np.dot(
                P - A,
                AB
            ) / denominator

            foot = A + t * AB

            _draw_segment(
                ax,
                tuple(A),
                tuple(B),
                linewidth=2.2
            )

            _draw_segment(
                ax,
                tuple(P),
                tuple(foot),
                linewidth=1.8
            )

            point_map["H"] = (
                float(foot[0]),
                float(foot[1])
            )

            if construction_spec.get(
                "show_construction_arcs",
                True
            ):
                radius = _distance(
                    tuple(P),
                    tuple(foot)
                )

                if radius > 0:
                    _draw_arc(
                        ax,
                        tuple(P),
                        radius,
                        0,
                        360
                    )

        # ----------------------------------------------------
        # GENERIC ANGLE CONSTRUCTION
        # ----------------------------------------------------

        elif construction_type == "angle_construction":

            base_line = construction_spec.get(
                "base_line",
                []
            )

            if (
                not isinstance(
                    base_line,
                    list
                )
                or len(base_line) < 2
            ):
                plt.close(fig)
                return None

            A_label = str(base_line[0])
            B_label = str(base_line[1])

            if (
                A_label not in point_map
                or B_label not in point_map
            ):
                plt.close(fig)
                return None

            A = point_map[A_label]
            B = point_map[B_label]

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2.2
            )

            angles = construction_spec.get(
                "angles",
                []
            )

            angle_value = None

            if isinstance(
                angles,
                list
            ):
                for item in angles:
                    if isinstance(
                        item,
                        dict
                    ):
                        try:
                            angle_value = float(
                                item.get(
                                    "value"
                                )
                            )
                            break
                        except Exception:
                            pass

            if angle_value is not None:

                length = max(
                    _distance(A, B),
                    4.0
                )

                theta = np.deg2rad(
                    angle_value
                )

                C = (
                    A[0] + length * np.cos(theta),
                    A[1] + length * np.sin(theta)
                )

                _draw_segment(
                    ax,
                    A,
                    C,
                    linewidth=2.2
                )

                if construction_spec.get(
                    "show_construction_arcs",
                    True
                ):
                    _draw_arc(
                        ax,
                        A,
                        length * 0.3,
                        0,
                        angle_value
                    )

                target_label = str(
                    construction_spec.get(
                        "target_point",
                        "C"
                    )
                ).strip()

                if target_label:
                    point_map[target_label] = C

        # ----------------------------------------------------
        # LINE DIVISION
        # ----------------------------------------------------

        elif construction_type == "line_division":

            base_line = construction_spec.get(
                "base_line",
                []
            )

            if (
                not isinstance(
                    base_line,
                    list
                )
                or len(base_line) < 2
            ):
                plt.close(fig)
                return None

            A_label = str(base_line[0])
            B_label = str(base_line[1])

            if (
                A_label not in point_map
                or B_label not in point_map
            ):
                plt.close(fig)
                return None

            A = np.array(
                point_map[A_label],
                dtype=float
            )

            B = np.array(
                point_map[B_label],
                dtype=float
            )

            _draw_segment(
                ax,
                tuple(A),
                tuple(B),
                linewidth=2.2
            )

            target_label = str(
                construction_spec.get(
                    "target_point",
                    "M"
                )
            ).strip()

            ratio = 0.5

            lengths = construction_spec.get(
                "lengths",
                []
            )

            if isinstance(
                lengths,
                list
            ):
                for item in lengths:
                    if isinstance(
                        item,
                        dict
                    ):
                        try:
                            ratio = float(
                                item.get(
                                    "ratio",
                                    0.5
                                )
                            )
                            break
                        except Exception:
                            pass

            ratio = min(
                max(ratio, 0.0),
                1.0
            )

            M = (
                A
                + ratio * (B - A)
            )

            point_map[target_label] = (
                float(M[0]),
                float(M[1])
            )

            ax.scatter(
                [M[0]],
                [M[1]],
                s=40
            )

        # ----------------------------------------------------
        # CIRCLE / ARC
        # ----------------------------------------------------

        elif construction_type == "circle_or_arc":

            if not point_map:
                plt.close(fig)
                return None

            centre_label = str(
                construction_spec.get(
                    "target_point",
                    ""
                )
            ).strip()

            if centre_label not in point_map:
                centre_label = next(
                    iter(point_map)
                )

            centre = point_map[
                centre_label
            ]

            radius = None

            lengths = construction_spec.get(
                "lengths",
                []
            )

            if isinstance(
                lengths,
                list
            ):
                for item in lengths:
                    if isinstance(
                        item,
                        dict
                    ):
                        try:
                            radius = float(
                                item.get(
                                    "value"
                                )
                            )
                            break
                        except Exception:
                            pass

            if radius is None:
                if len(point_map) >= 2:
                    other = [
                        p
                        for label, p in point_map.items()
                        if label != centre_label
                    ][0]

                    radius = _distance(
                        centre,
                        other
                    )

            if radius is None or radius <= 0:
                plt.close(fig)
                return None

            _draw_arc(
                ax,
                centre,
                radius,
                0,
                360,
                linewidth=2.0,
                linestyle="-"
            )

        else:
            plt.close(fig)
            return None

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        if construction_spec.get(
            "show_labels",
            True
        ):
            _label_points(
                ax,
                point_map
            )

        # ----------------------------------------------------
        # AXIS / VIEW
        # ----------------------------------------------------

        all_points = list(
            point_map.values()
        )

        if all_points:
            xs = [
                p[0]
                for p in all_points
            ]

            ys = [
                p[1]
                for p in all_points
            ]

            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)

            width = max(
                xmax - xmin,
                1
            )

            height = max(
                ymax - ymin,
                1
            )

            margin = 0.35 * max(
                width,
                height
            )

            ax.set_xlim(
                xmin - margin,
                xmax + margin
            )

            ax.set_ylim(
                ymin - margin,
                ymax + margin
            )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.axis("off")

        title = str(
            construction_spec.get(
                "title",
                ""
            )
        ).strip()

        if title:
            ax.set_title(
                title
            )

        plt.tight_layout()

        output = io.BytesIO()

        fig.savefig(
            output,
            format="png",
            dpi=170,
            bbox_inches="tight"
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

    expected_set = set(expected)
    solved_set = set(solved)

    missing_keys = (
        expected_set
        - solved_set
    )

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
        "questions_detected": len(
            expected_set
        ),
        "questions_solved": len(
            expected_set & solved_set
        ),
        "missing_questions": missing_questions,
        "complete": len(
            missing_questions
        ) == 0
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

    # ========================================================
    # 📈 GENERATED GRAPH
    # ========================================================

    graph_spec = solution.get(
        "graph_spec",
        {}
    )

    if (
        isinstance(
            graph_spec,
            dict
        )
        and graph_spec.get(
            "required",
            False
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
                width="stretch"
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
    # 📐 GENERATED MATHEMATICAL CONSTRUCTION
    # ========================================================

    construction_spec = solution.get(
        "construction_spec",
        {}
    )

    if (
        isinstance(
            construction_spec,
            dict
        )
        and construction_spec.get(
            "required",
            False
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
                    f"Mathematical construction for "
                    f"Question {number}"
                ),
                width="stretch"
            )

            st.caption(
                "📐 Construction generated mathematically "
                "from the verified construction specification."
            )

        else:

            st.warning(
                "The question requires a construction, "
                "but the construction specification "
                "could not be rendered."
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

                st.exception(
                    exc
                )

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
                        "questions": [],
                        "visual_warnings": [
                            str(exc)
                        ]
                    }

            questions = page_analysis.get(
                "questions",
                []
            )

            page_result = {
                "page_number": page_number,
                "questions": questions,
                "visual_warnings": page_analysis.get(
                    "visual_warnings",
                    []
                ),
                "solutions": []
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
            check["questions_detected"]
        )

    with col2:
        st.metric(
            "Questions solved",
            check["questions_solved"]
        )

    with col3:
        st.metric(
            "Missing",
            len(
                check["missing_questions"]
            )
        )

    if check[
        "missing_questions"
    ]:

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
                    "question_number": number,
                    "confidence": "medium",
                    "confidence_reason": (
                        "No solution was generated."
                    ),
                    "working": [],
                    "marking_scheme": [],
                    "graph_spec": default_graph_spec(),
                    "construction_spec": (
                        default_construction_spec()
                    ),
                    "warning": (
                        "No solution was generated."
                    )
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

    construction_questions = []

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
            )

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
                    number
                )

            if question.get(
                "has_construction",
                False
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

        st.markdown(
            "### 📐 Mathematical constructions detected"
        )

        st.success(
            "Construction questions: "
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
        "formula_subject": 0
    }

    for page_result in results:

        for question in page_result.get(
            "questions",
            []
        ):

            summary = str(
                question.get(
                    "visible_text_summary",
                    ""
                )
            ).lower()

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

        **7. Mathematical constructions can be identified
        and generated using a dedicated geometry engine**

        **8. The solution is independently verified
        against the original question**

        **9. Low confidence is reserved for genuine
        mathematical errors or essential visual ambiguity**

        **10. Original pages remain untouched**

        **11. Marking points are displayed under each question**
        """
    )
