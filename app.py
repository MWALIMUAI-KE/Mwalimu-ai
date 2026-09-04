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
from matplotlib.patches import Arc, Circle

import sympy as sp


# ============================================================
# MWALIMU AI — MVP3.2 VISUAL MARKING
# CONSTRUCTION ENGINE V1
#
# FINAL MATHS POLISH PRESERVED
#
# MAIN TARGETS:
# 1. SURDS
# 2. INTEGRATION
# 3. MAKING THE SUBJECT OF A FORMULA
# 4. RELIABLE MATHEMATICAL RENDERING
# 5. DETERMINISTIC GEOMETRIC CONSTRUCTIONS
#
# ORIGINAL PDF = VISUAL SOURCE OF TRUTH
#
# The original page is preserved.
# Individual questions are isolated visually.
#
# The solver receives:
# - the isolated original question
# - the complete original page
# - supporting PDF text
#
# The verifier independently checks the solution.
#
# LOW CONFIDENCE is ONLY allowed for:
# - genuine mathematical error
# - genuine essential visual ambiguity
#
# Technical/API/parser problems NEVER automatically become LOW.
# ============================================================


st.set_page_config(
    page_title="Mwalimu AI — Visual Marking",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Mwalimu AI")

st.caption(
    "MVP3.2 Visual Marking Engine — Original question preserved, "
    "AI workings, marking scheme, graphs and geometric constructions "
    "added underneath."
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

            scale = min(
                scale,
                2.5,
            )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    scale,
                    scale,
                ),
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

            texts.append(
                text.strip()
            )

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

If a geometric construction is requested, identify it explicitly.

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

Use proper mathematical notation.

Examples:

\sqrt{12}=2\sqrt{3}

\frac{1}{\sqrt{3}}=\frac{\sqrt{3}}{3}

\frac{1}{a+\sqrt{b}}

Use the actual expression from the image, not an assumed textbook version.

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

\int x^n dx =
\frac{x^{n+1}}{n+1}+C

provided n != -1.

For a definite integral:

\int_a^b f(x)dx=F(b)-F(a)

Do not omit C from an indefinite integral.

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
- use the ACTUAL visible coordinates;
- use the actual relationships shown;
- do not invent missing dimensions;
- do not assume a diagram is to scale unless the question says so.

============================================================
GRAPHING
============================================================

If the question requires a graph, independently identify and preserve:

- equation or function
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

Instead, return a graph_spec that allows the application to generate the graph mathematically.

For PP2 V1, use only:

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

Return this graph_spec structure:

{
  "required": false,
  "graph_type": "none | line | quadratic | function | points",
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
}

Never invent a domain, range, scale, coordinate or graph requirement that is not supported by the question.

============================================================
GEOMETRIC CONSTRUCTIONS
============================================================

If the question requires a geometric construction, identify the exact
construction required from the ORIGINAL IMAGE.

Possible construction types for Construction Engine V1 are:

- none
- angle_bisector
- perpendicular_bisector
- perpendicular_from_point
- triangle_sides
- triangle_base_angles
- locus_circle
- locus_perpendicular_bisector

IMPORTANT:

The AI must NOT attempt to draw the construction itself.

Instead, return a precise construction_spec.

Python will perform the actual mathematical construction.

Use the ACTUAL visible lengths and angles.

Do not invent dimensions.

If exact coordinates are not supplied, use normalized/canonical geometry
derived from the supplied lengths or angles.

Do not claim that the construction is to scale unless the question provides
enough information for scale.

The construction_spec must use this structure:

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
    "AB": null,
    "BC": null,
    "CA": null
  },
  "angles": {
    "A": null,
    "B": null,
    "C": null
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

Rules:

1. Use supplied lengths/angles only.
2. Do not invent arbitrary measurements.
3. If exact coordinates are visible, use them.
4. If coordinates are not visible, Python may construct canonical geometry
   from the supplied lengths and/or angles.
5. Preserve the construction relationships.
6. Show construction arcs/lines where appropriate.
7. Label visible named points where appropriate.
8. The generated construction is mathematical geometry, not an AI sketch.

For an angle bisector, identify the vertex and the two rays.

For a perpendicular bisector, identify the two endpoints of the segment.

For a perpendicular from a point to a line, identify the line endpoints
and the external point.

For a triangle from three sides, use the supplied side lengths.

For a triangle from base angles, use the supplied base and base angles.

For a locus circle, identify the centre and radius.

For a locus perpendicular bisector, identify the relevant segment.

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

B. Essential visual information is genuinely unreadable or ambiguous.

If the mathematics is correct and the necessary visual information is readable,
confidence should normally be HIGH.

Return JSON only:

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
You are the FINAL INDEPENDENT MATHEMATICS VERIFICATION EXAMINER for
Mwalimu AI.

Compare the generated solution against the ORIGINAL QUESTION IMAGE
and the COMPLETE ORIGINAL PAGE.

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
CHECK 6 — CONSTRUCTION
============================================================

If a geometric construction is required, independently verify:

- the construction type;
- the relevant points;
- supplied lengths;
- supplied angles;
- geometric relationships;
- whether the construction_spec corresponds to the ORIGINAL question;
- whether the construction uses only information supported by the question;
- whether labels correspond to the actual question.

A technical failure to render a construction is NOT itself a mathematical error.

A wrong construction type, wrong supplied length, wrong angle,
wrong geometric relationship, or invented essential data IS a genuine
mathematical/visual error.

The construction is generated mathematically by the application.
Do not judge a rendering service failure as a mathematical error.

============================================================
GRAPH VERIFICATION
============================================================

If the question requires a graph, independently verify:

- equation/function;
- graph type;
- domain;
- supplied coordinates;
- table values;
- intercepts;
- axis labels;
- point requirements;
- joining requirements;
- general mathematical shape.

A technical failure to render a graph is NOT itself a mathematical error.

A wrong equation, wrong coordinates, wrong domain, wrong graph type,
or mathematically incorrect graph specification IS a genuine mathematical error.

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

B. Essential visual information is genuinely unreadable or ambiguous.

If the mathematics is correct and the visual information is sufficiently clear,
return HIGH.

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

    text = text.strip()

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
        min(
            x,
            image_width - 1,
        ),
    )

    y = max(
        0,
        min(
            y,
            image_height - 1,
        ),
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
            int(
                width * QUESTION_CROP_MARGIN
            ),
        )

        margin_y = max(
            18,
            int(
                height * QUESTION_CROP_MARGIN
            ),
        )

        crop = image.crop(
            (
                max(
                    0,
                    x - margin_x,
                ),
                max(
                    0,
                    y - margin_y,
                ),
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
        "title": "",
    }


# ============================================================
# DEFAULT CONSTRUCTION SPEC
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
        question.get(
            "number",
            "",
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
                ),
            }

        result["question_number"] = question_number

        if not isinstance(
            result.get("graph_spec"),
            dict,
        ):
            result["graph_spec"] = default_graph_spec()

        if not isinstance(
            result.get("construction_spec"),
            dict,
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
    """
    Clean common model-generated mathematical wrappers while
    preserving valid LaTeX.
    """

    text = str(text).strip()

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

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

    # Handle common math delimiters.
    text = text.replace(
        r"\[",
        "$$",
    )

    text = text.replace(
        r"\]",
        "$$",
    )

    text = text.replace(
        r"\(",
        "$",
    )

    text = text.replace(
        r"\)",
        "$",
    )

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
        (
            "−",
            "-",
        ),
        (
            "×",
            r"\times ",
        ),
        (
            "÷",
            r"\div ",
        ),
        (
            "°",
            r"^\circ",
        ),
    ]

    for old, new in replacements:
        text = text.replace(
            old,
            new,
        )

    # Simple square-root notation.
    text = re.sub(
        r"√\s*([A-Za-z0-9]+)",
        r"$\\sqrt{\1}$",
        text,
    )

    # Plain exponent notation.
    text = re.sub(
        r"(?<!\^)([A-Za-z])\^(-?\d+)(?!\})",
        r"$\1^{\2}$",
        text,
    )

    # Common subscript notation.
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
            and bool(
                re.search(
                    r"[A-Za-z0-9]",
                    stripped,
                )
            )
        )

        if (
            looks_mathematical
            and len(stripped) < 180
        ):

            if re.search(
                r"^[A-Za-z\s]+$",
                stripped,
            ) is None:

                rendered_lines.append(
                    "$"
                    + stripped
                    + "$"
                )

                continue

        rendered_lines.append(line)

    text = "\n".join(
        rendered_lines
    )

    st.markdown(text)


# ============================================================
# PP2 GRAPHING ENGINE V1
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


def _safe_graph_expression(
    expression: str,
):
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
        flags=re.IGNORECASE,
    )

    expression = expression.replace(
        "^",
        "**",
    )

    expression = expression.replace(
        "$",
        "",
    )

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

    if not isinstance(
        graph_spec,
        dict,
    ):
        return None

    if not graph_spec.get(
        "required",
        False,
    ):
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

        # ----------------------------------------------------
        # X RANGE
        # ----------------------------------------------------

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
            x_min, x_max = (
                -10.0,
                10.0,
            )

        # ----------------------------------------------------
        # FUNCTION / LINE / QUADRATIC
        # ----------------------------------------------------

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

            try:

                function = sp.lambdify(
                    x,
                    sympy_expr,
                    modules="numpy",
                )

            except Exception:

                plt.close(fig)
                return None

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
                np.pi
                / 180.0
                * x_data
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

        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------

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
                not isinstance(
                    x_values,
                    list,
                )
                or not isinstance(
                    y_values,
                    list,
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
                    [
                        float(v)
                        for v in x_values
                    ]
                )

                y_data = np.asarray(
                    [
                        float(v)
                        for v in y_values
                    ]
                )

            except Exception:

                plt.close(fig)
                return None

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

        # ----------------------------------------------------
        # AXES
        # ----------------------------------------------------

        ax.axhline(
            0,
            linewidth=1,
        )

        ax.axvline(
            0,
            linewidth=1,
        )

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

        # ----------------------------------------------------
        # Y RANGE
        # ----------------------------------------------------

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
                        y_max,
                    )

        except Exception:
            pass

        ax.set_xlim(
            x_min,
            x_max,
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

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

        if value is None:
            return default

        return float(value)

    except Exception:
        return default


def _point_from_spec(
    points: Dict[str, Any],
    name: str,
) -> Optional[np.ndarray]:

    if not isinstance(
        points,
        dict,
    ):
        return None

    value = points.get(name)

    if (
        not isinstance(
            value,
            (list, tuple),
        )
        or len(value) < 2
    ):
        return None

    try:

        return np.array(
            [
                float(value[0]),
                float(value[1]),
            ],
            dtype=float,
        )

    except Exception:
        return None


def _distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.linalg.norm(
            b - a
        )
    )


def _unit(
    vector: np.ndarray,
) -> Optional[np.ndarray]:

    length = float(
        np.linalg.norm(vector)
    )

    if length <= 1e-9:
        return None

    return vector / length


def _draw_segment(
    ax,
    a: np.ndarray,
    b: np.ndarray,
    **kwargs,
):

    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        **kwargs,
    )


def _draw_arc(
    ax,
    center: np.ndarray,
    radius: float,
    theta1: float,
    theta2: float,
    **kwargs,
):

    patch = Arc(
        (
            center[0],
            center[1],
        ),
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
    point: np.ndarray,
    label: str,
    offset: Tuple[float, float] = (0.12, 0.12),
):

    ax.text(
        point[0] + offset[0],
        point[1] + offset[1],
        label,
        fontsize=12,
        fontweight="bold",
    )


def _canonical_triangle_from_sides(
    ab: float,
    bc: float,
    ca: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    if (
        ab <= 0
        or bc <= 0
        or ca <= 0
    ):
        return None

    # Triangle inequality.
    if (
        ab + bc <= ca
        or ab + ca <= bc
        or bc + ca <= ab
    ):
        return None

    A = np.array(
        [0.0, 0.0]
    )

    B = np.array(
        [ab, 0.0]
    )

    x = (
        ca * ca
        + ab * ab
        - bc * bc
    ) / (2.0 * ab)

    y_squared = (
        ca * ca
        - x * x
    )

    if y_squared <= 0:
        return None

    C = np.array(
        [
            x,
            math.sqrt(y_squared),
        ]
    )

    return A, B, C


def _canonical_triangle_from_angles(
    ab: float,
    angle_a: float,
    angle_b: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    if ab <= 0:
        return None

    if (
        angle_a <= 0
        or angle_b <= 0
        or angle_a + angle_b >= 180
    ):
        return None

    A = np.array(
        [0.0, 0.0]
    )

    B = np.array(
        [ab, 0.0]
    )

    theta_a = math.radians(
        angle_a
    )

    theta_b = math.radians(
        180.0 - angle_b
    )

    direction_a = np.array(
        [
            math.cos(theta_a),
            math.sin(theta_a),
        ]
    )

    direction_b = np.array(
        [
            math.cos(theta_b),
            math.sin(theta_b),
        ]
    )

    matrix = np.column_stack(
        (
            direction_a,
            -direction_b,
        )
    )

    rhs = B - A

    try:

        t, s = np.linalg.solve(
            matrix,
            rhs,
        )

    except Exception:
        return None

    if t <= 0 or s <= 0:
        return None

    C = (
        A
        + t * direction_a
    )

    return A, B, C


def _setup_construction_axes(
    ax,
    points: List[np.ndarray],
):

    valid = [
        p
        for p in points
        if p is not None
    ]

    if not valid:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return

    coords = np.vstack(valid)

    min_x = float(np.min(coords[:, 0]))
    max_x = float(np.max(coords[:, 0]))
    min_y = float(np.min(coords[:, 1]))
    max_y = float(np.max(coords[:, 1]))

    span_x = max(
        max_x - min_x,
        1.0,
    )

    span_y = max(
        max_y - min_y,
        1.0,
    )

    margin_x = 0.18 * span_x
    margin_y = 0.18 * span_y

    ax.set_xlim(
        min_x - margin_x,
        max_x + margin_x,
    )

    ax.set_ylim(
        min_y - margin_y,
        max_y + margin_y,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.25,
    )


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

        if not isinstance(
            points,
            dict,
        ):
            points = {}

        lengths = construction_spec.get(
            "lengths",
            {},
        )

        if not isinstance(
            lengths,
            dict,
        ):
            lengths = {}

        angles = construction_spec.get(
            "angles",
            {},
        )

        if not isinstance(
            angles,
            dict,
        ):
            angles = {}

        show_lines = bool(
            construction_spec.get(
                "show_construction_lines",
                True,
            )
        )

        show_arcs = bool(
            construction_spec.get(
                "show_arcs",
                True,
            )
        )

        label_points = bool(
            construction_spec.get(
                "label_points",
                True,
            )
        )

        used_points = []

        # ====================================================
        # ANGLE BISECTOR
        # ====================================================

        if construction_type == "angle_bisector":

            A = _point_from_spec(
                points,
                "A",
            )

            B = _point_from_spec(
                points,
                "B",
            )

            C = _point_from_spec(
                points,
                "C",
            )

            if (
                A is None
                or B is None
                or C is None
            ):

                A = np.array(
                    [0.0, 0.0]
                )

                B = np.array(
                    [8.0, 0.0]
                )

                C = np.array(
                    [4.0, 6.0]
                )

            used_points.extend(
                [
                    A,
                    B,
                    C,
                ]
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

            unit_ab = _unit(
                B - A
            )

            unit_ac = _unit(
                C - A
            )

            if (
                unit_ab is None
                or unit_ac is None
            ):

                plt.close(fig)
                return None

            base_length = min(
                _distance(A, B),
                _distance(A, C),
            )

            compass_radius = 0.32 * base_length

            D = (
                A
                + unit_ab * compass_radius
            )

            E = (
                A
                + unit_ac * compass_radius
            )

            if show_arcs:

                angle_b = math.degrees(
                    math.atan2(
                        unit_ab[1],
                        unit_ab[0],
                    )
                )

                angle_c = math.degrees(
                    math.atan2(
                        unit_ac[1],
                        unit_ac[0],
                    )
                )

                _draw_arc(
                    ax,
                    A,
                    compass_radius,
                    min(
                        angle_b,
                        angle_c,
                    ),
                    max(
                        angle_b,
                        angle_c,
                    ),
                    linewidth=1.5,
                )

            # Find the internal angle-bisector direction.
            bisector_direction = _unit(
                unit_ab + unit_ac
            )

            if bisector_direction is None:

                plt.close(fig)
                return None

            # Compass arc from D and E.
            DE = _distance(
                D,
                E,
            )

            intersection_radius = (
                max(
                    DE,
                    compass_radius,
                )
                * 1.15
            )

            # Intersection of circles centered D and E.
            midpoint = (
                D + E
            ) / 2.0

            chord = _distance(
                D,
                E,
            )

            if chord <= 1e-9:

                plt.close(fig)
                return None

            if (
                intersection_radius
                <= chord / 2.0
            ):

                intersection_radius = (
                    chord / 2.0
                    + 0.2 * base_length
                )

            h_squared = (
                intersection_radius ** 2
                - (chord / 2.0) ** 2
            )

            if h_squared <= 0:

                plt.close(fig)
                return None

            perpendicular = np.array(
                [
                    -(E - D)[1],
                    (E - D)[0],
                ]
            )

            perpendicular = _unit(
                perpendicular
            )

            if perpendicular is None:

                plt.close(fig)
                return None

            F = (
                midpoint
                + perpendicular
                * math.sqrt(h_squared)
            )

            # Choose intersection lying in the
            # internal-bisector direction.
            if np.dot(
                F - A,
                bisector_direction,
            ) < 0:

                F = (
                    midpoint
                    - perpendicular
                    * math.sqrt(h_squared)
                )

            if show_arcs:

                theta_d = math.degrees(
                    math.atan2(
                        F[1] - D[1],
                        F[0] - D[0],
                    )
                )

                theta_e = math.degrees(
                    math.atan2(
                        F[1] - E[1],
                        F[0] - E[0],
                    )
                )

                _draw_arc(
                    ax,
                    D,
                    intersection_radius,
                    theta_d - 24,
                    theta_d + 24,
                    linewidth=1.2,
                )

                _draw_arc(
                    ax,
                    E,
                    intersection_radius,
                    theta_e - 24,
                    theta_e + 24,
                    linewidth=1.2,
                )

            _draw_segment(
                ax,
                A,
                F,
                linewidth=2.5,
            )

            if label_points:

                _label_point(
                    ax,
                    A,
                    "A",
                )

                _label_point(
                    ax,
                    B,
                    "B",
                )

                _label_point(
                    ax,
                    C,
                    "C",
                )

            used_points.append(F)

        # ====================================================
        # PERPENDICULAR BISECTOR
        # ====================================================

        elif construction_type == "perpendicular_bisector":

            A = _point_from_spec(
                points,
                "A",
            )

            B = _point_from_spec(
                points,
                "B",
            )

            if (
                A is None
                or B is None
            ):

                ab = _safe_float(
                    lengths.get(
                        "AB"
                    ),
                    10.0,
                )

                A = np.array(
                    [0.0, 0.0]
                )

                B = np.array(
                    [ab, 0.0]
                )

            used_points.extend(
                [
                    A,
                    B,
                ]
            )

            segment_length = _distance(
                A,
                B,
            )

            if segment_length <= 1e-9:

                plt.close(fig)
                return None

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            midpoint = (
                A + B
            ) / 2.0

            radius = 0.65 * segment_length

            if radius <= segment_length / 2.0:
                radius = (
                    segment_length * 0.6
                )

            if show_arcs:

                angle_ab = math.degrees(
                    math.atan2(
                        B[1] - A[1],
                        B[0] - A[0],
                    )
                )

                _draw_arc(
                    ax,
                    A,
                    radius,
                    angle_ab - 65,
                    angle_ab + 65,
                    linewidth=1.4,
                )

                _draw_arc(
                    ax,
                    B,
                    radius,
                    angle_ab + 115,
                    angle_ab + 245,
                    linewidth=1.4,
                )

            direction = _unit(
                B - A
            )

            if direction is None:

                plt.close(fig)
                return None

            perpendicular = np.array(
                [
                    -direction[1],
                    direction[0],
                ]
            )

            half_height = math.sqrt(
                max(
                    radius * radius
                    - (
                        segment_length / 2.0
                    ) ** 2,
                    0.0,
                )
            )

            U = (
                midpoint
                + perpendicular
                * half_height
            )

            V = (
                midpoint
                - perpendicular
                * half_height
            )

            _draw_segment(
                ax,
                U,
                V,
                linewidth=2.5,
            )

            if label_points:

                _label_point(
                    ax,
                    A,
                    "A",
                )

                _label_point(
                    ax,
                    B,
                    "B",
                )

                _label_point(
                    ax,
                    midpoint,
                    "M",
                )

            used_points.extend(
                [
                    U,
                    V,
                ]
            )

        # ====================================================
        # PERPENDICULAR FROM POINT
        # ====================================================

        elif construction_type == "perpendicular_from_point":

            A = _point_from_spec(
                points,
                "A",
            )

            B = _point_from_spec(
                points,
                "B",
            )

            P = _point_from_spec(
                points,
                "P",
            )

            if (
                A is None
                or B is None
                or P is None
            ):

                A = np.array(
                    [0.0, 0.0]
                )

                B = np.array(
                    [10.0, 0.0]
                )

                P = np.array(
                    [4.0, 6.0]
                )

            used_points.extend(
                [
                    A,
                    B,
                    P,
                ]
            )

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            line_direction = _unit(
                B - A
            )

            if line_direction is None:

                plt.close(fig)
                return None

            # Projection of P onto AB.
            AP = P - A

            projection_distance = np.dot(
                AP,
                line_direction,
            )

            H = (
                A
                + projection_distance
                * line_direction
            )

            distance_to_line = _distance(
                P,
                H,
            )

            if distance_to_line <= 1e-9:

                plt.close(fig)
                return None

            # Compass construction:
            # First circle centered at P intersects AB.
            compass_radius = max(
                distance_to_line * 1.25,
                0.3 * _distance(A, B),
            )

            # Ensure the circle reaches the line.
            if compass_radius <= distance_to_line:
                compass_radius = (
                    distance_to_line * 1.5
                )

            perpendicular_to_line = np.array(
                [
                    -line_direction[1],
                    line_direction[0],
                ]
            )

            foot_offset = math.sqrt(
                max(
                    compass_radius ** 2
                    - distance_to_line ** 2,
                    0.0,
                )
            )

            D = (
                H
                - line_direction
                * foot_offset
            )

            E = (
                H
                + line_direction
                * foot_offset
            )

            if show_arcs:

                theta_p = math.degrees(
                    math.atan2(
                        H[1] - P[1],
                        H[0] - P[0],
                    )
                )

                _draw_arc(
                    ax,
                    P,
                    compass_radius,
                    theta_p - 65,
                    theta_p + 65,
                    linewidth=1.3,
                )

            # Second pair of arcs from D/E.
            second_radius = (
                0.75 * _distance(D, E)
            )

            if second_radius <= _distance(D, E) / 2:
                second_radius = (
                    0.6 * _distance(D, E)
                )

            midpoint_de = (
                D + E
            ) / 2.0

            de_length = _distance(
                D,
                E,
            )

            h2 = math.sqrt(
                max(
                    second_radius ** 2
                    - (de_length / 2.0) ** 2,
                    0.0,
                )
            )

            upper = (
                midpoint_de
                + perpendicular_to_line
                * h2
            )

            lower = (
                midpoint_de
                - perpendicular_to_line
                * h2
            )

            # Choose intersection on same side as P.
            if np.dot(
                upper - H,
                P - H,
            ) > 0:

                F = upper

            else:

                F = lower

            if show_arcs:

                theta_d = math.degrees(
                    math.atan2(
                        F[1] - D[1],
                        F[0] - D[0],
                    )
                )

                theta_e = math.degrees(
                    math.atan2(
                        F[1] - E[1],
                        F[0] - E[0],
                    )
                )

                _draw_arc(
                    ax,
                    D,
                    second_radius,
                    theta_d - 28,
                    theta_d + 28,
                    linewidth=1.2,
                )

                _draw_arc(
                    ax,
                    E,
                    second_radius,
                    theta_e - 28,
                    theta_e + 28,
                    linewidth=1.2,
                )

            _draw_segment(
                ax,
                P,
                H,
                linewidth=2.5,
            )

            if label_points:

                _label_point(
                    ax,
                    A,
                    "A",
                )

                _label_point(
                    ax,
                    B,
                    "B",
                )

                _label_point(
                    ax,
                    P,
                    "P",
                )

                _label_point(
                    ax,
                    H,
                    "H",
                )

            used_points.extend(
                [
                    D,
                    E,
                    F,
                    H,
                ]
            )

        # ====================================================
        # TRIANGLE FROM THREE SIDES
        # ====================================================

        elif construction_type == "triangle_sides":

            A = _point_from_spec(
                points,
                "A",
            )

            B = _point_from_spec(
                points,
                "B",
            )

            C = _point_from_spec(
                points,
                "C",
            )

            ab = _safe_float(
                lengths.get("AB"),
            )

            bc = _safe_float(
                lengths.get("BC"),
            )

            ca = _safe_float(
                lengths.get("CA"),
            )

            if (
                A is None
                or B is None
                or C is None
            ):

                if (
                    ab is None
                    or bc is None
                    or ca is None
                ):

                    plt.close(fig)
                    return None

                canonical = (
                    _canonical_triangle_from_sides(
                        ab,
                        bc,
                        ca,
                    )
                )

                if canonical is None:

                    plt.close(fig)
                    return None

                A, B, C = canonical

            else:

                if ab is None:
                    ab = _distance(A, B)

                if bc is None:
                    bc = _distance(B, C)

                if ca is None:
                    ca = _distance(C, A)

            used_points.extend(
                [
                    A,
                    B,
                    C,
                ]
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

                theta_ab = math.degrees(
                    math.atan2(
                        B[1] - A[1],
                        B[0] - A[0],
                    )
                )

                _draw_arc(
                    ax,
                    A,
                    ca,
                    theta_ab - 45,
                    theta_ab + 45,
                    linewidth=1.3,
                )

                theta_ba = math.degrees(
                    math.atan2(
                        A[1] - B[1],
                        A[0] - B[0],
                    )
                )

                _draw_arc(
                    ax,
                    B,
                    bc,
                    theta_ba - 45,
                    theta_ba + 45,
                    linewidth=1.3,
                )

            if label_points:

                _label_point(
                    ax,
                    A,
                    "A",
                )

                _label_point(
                    ax,
                    B,
                    "B",
                )

                _label_point(
                    ax,
                    C,
                    "C",
                )

        # ====================================================
        # TRIANGLE FROM BASE ANGLES
        # ====================================================

        elif construction_type == "triangle_base_angles":

            A = _point_from_spec(
                points,
                "A",
            )

            B = _point_from_spec(
                points,
                "B",
            )

            C = _point_from_spec(
                points,
                "C",
            )

            ab = _safe_float(
                lengths.get(
                    "AB"
                ),
            )

            angle_a = _safe_float(
                angles.get(
                    "A"
                ),
            )

            angle_b = _safe_float(
                angles.get(
                    "B"
                ),
            )

            if (
                A is None
                or B is None
                or C is None
            ):

                if (
                    ab is None
                    or angle_a is None
                    or angle_b is None
                ):

                    plt.close(fig)
                    return None

                canonical = (
                    _canonical_triangle_from_angles(
                        ab,
                        angle_a,
                        angle_b,
                    )
                )

                if canonical is None:

                    plt.close(fig)
                    return None

                A, B, C = canonical

            else:

                if ab is None:
                    ab = _distance(A, B)

                if angle_a is None:
                    angle_a = 60.0

                if angle_b is None:
                    angle_b = 60.0

            used_points.extend(
                [
                    A,
                    B,
                    C,
                ]
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

            # Extend base-angle rays slightly to show
            # the construction.
            if show_lines:

                direction_a = _unit(
                    C - A
                )

                direction_b = _unit(
                    C - B
                )

                if (
                    direction_a is not None
                    and direction_b is not None
                ):

                    extension = 0.25 * ab

                    _draw_segment(
                        ax,
                        A - direction_a * extension,
                        C,
                        linewidth=1.1,
                        linestyle="--",
                    )

                    _draw_segment(
                        ax,
                        B - direction_b * extension,
                        C,
                        linewidth=1.1,
                        linestyle="--",
                    )

            if label_points:

                _label_point(
                    ax,
                    A,
                    "A",
                )

                _label_point(
                    ax,
                    B,
                    "B",
                )

                _label_point(
                    ax,
                    C,
                    "C",
                )

        # ====================================================
        # LOCUS CIRCLE
        # ====================================================

        elif construction_type == "locus_circle":

            O = _point_from_spec(
                points,
                "O",
            )

            if O is None:

                O = _point_from_spec(
                    points,
                    "A",
                )

            radius = _safe_float(
                construction_spec.get(
                    "radius"
                )
            )

            if radius is None:

                radius = _safe_float(
                    lengths.get(
                        "radius"
                    )
                )

            if (
                O is None
                or radius is None
                or radius <= 0
            ):

                plt.close(fig)
                return None

            used_points.append(O)

            circle = Circle(
                (
                    O[0],
                    O[1],
                ),
                radius,
                fill=False,
                linewidth=2,
            )

            ax.add_patch(circle)

            ax.scatter(
                [O[0]],
                [O[1]],
                s=35,
            )

            if label_points:
                _label_point(
                    ax,
                    O,
                    "O",
                )

            radius_point = (
                O
                + np.array(
                    [
                        radius,
                        0.0,
                    ]
                )
            )

            _draw_segment(
                ax,
                O,
                radius_point,
                linewidth=1.5,
            )

            used_points.append(
                radius_point
            )

        # ====================================================
        # LOCUS PERPENDICULAR BISECTOR
        # ====================================================

        elif construction_type == "locus_perpendicular_bisector":

            A = _point_from_spec(
                points,
                "A",
            )

            B = _point_from_spec(
                points,
                "B",
            )

            if (
                A is None
                or B is None
            ):

                ab = _safe_float(
                    lengths.get(
                        "AB"
                    ),
                    10.0,
                )

                A = np.array(
                    [0.0, 0.0]
                )

                B = np.array(
                    [ab, 0.0]
                )

            used_points.extend(
                [
                    A,
                    B,
                ]
            )

            segment_length = _distance(
                A,
                B,
            )

            if segment_length <= 1e-9:

                plt.close(fig)
                return None

            _draw_segment(
                ax,
                A,
                B,
                linewidth=2,
            )

            midpoint = (
                A + B
            ) / 2.0

            direction = _unit(
                B - A
            )

            if direction is None:

                plt.close(fig)
                return None

            perpendicular = np.array(
                [
                    -direction[1],
                    direction[0],
                ]
            )

            extension = 0.75 * segment_length

            U = (
                midpoint
                + perpendicular * extension
            )

            V = (
                midpoint
                - perpendicular * extension
            )

            _draw_segment(
                ax,
                U,
                V,
                linewidth=2.5,
            )

            if show_arcs:

                radius = 0.7 * segment_length

                angle_ab = math.degrees(
                    math.atan2(
                        B[1] - A[1],
                        B[0] - A[0],
                    )
                )

                _draw_arc(
                    ax,
                    A,
                    radius,
                    angle_ab - 60,
                    angle_ab + 60,
                    linewidth=1.2,
                )

                _draw_arc(
                    ax,
                    B,
                    radius,
                    angle_ab + 120,
                    angle_ab + 240,
                    linewidth=1.2,
                )

            if label_points:

                _label_point(
                    ax,
                    A,
                    "A",
                )

                _label_point(
                    ax,
                    B,
                    "B",
                )

                _label_point(
                    ax,
                    midpoint,
                    "M",
                )

        else:

            plt.close(fig)
            return None

        # ====================================================
        # AXIS / TITLE
        # ====================================================

        _setup_construction_axes(
            ax,
            used_points,
        )

        ax.set_xlabel(
            "Construction geometry"
        )

        ax.set_ylabel(
            ""
        )

        title = str(
            construction_spec.get(
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

    expected_set = set(
        expected
    )

    solved_set = set(
        solved
    )

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
                1,
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
            expected_set
            & solved_set
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

    bbox = question.get(
        "bbox"
    )

    crop = None

    if bbox:

        crop = crop_original_question(
            image_bytes,
            bbox,
        )

    if crop:

        st.image(
            crop,
            caption=(
                f"Original Question {number}"
            ),
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
    # GENERATED GRAPH
    # ========================================================

    graph_spec = solution.get(
        "graph_spec",
        {},
    )

    if (
        isinstance(
            graph_spec,
            dict,
        )
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
    # GENERATED CONSTRUCTION
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
                "but the construction specification could "
                "not be rendered."
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
                ].append(
                    solution
                )

            all_results.append(
                page_result
            )

            progress.progress(
                int(
                    (
                        (
                            index + 1
                        )
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
        "generates mathematical working, graphs "
        "and constructions underneath them."
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
            ],
        )

    with col2:

        st.metric(
            "Questions solved",
            check[
                "questions_solved"
            ],
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

                solution_map[
                    number
                ] = solution

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
                    "graph_spec": default_graph_spec(),
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

    graph_questions = []

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

            if question.get(
                "has_graph",
                False,
            ):

                graph_questions.append(
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

    if graph_questions:

        st.info(
            "📈 Graph questions detected: "
            + ", ".join(
                graph_questions
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

        m1, m2, m3 = st.columns(
            3
        )

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

        **7. The solution is independently verified
        against the original question**

        **8. Low confidence is reserved for genuine
        mathematical errors or essential visual ambiguity**

        **9. Original pages remain untouched**

        **10. Marking points are displayed under each question**

        **11. Required graphs are generated mathematically**

        **12. Required geometric constructions are generated
        mathematically from the verified construction specification**
        """
    )
