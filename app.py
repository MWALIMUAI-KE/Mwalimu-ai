import os
import io
import re
import base64

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI


# ============================================================
# MWALIMU AI — VISUAL MVP3
# BATTLE 1: DIAGRAMS ONLY
# ============================================================

st.set_page_config(
    page_title="Mwalimu AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Mwalimu AI")
st.caption(
    "Teacher AI assistant — Visual MVP3 | Diagram understanding upgrade"
)


# ============================================================
# OPENAI
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Add it as a deployment secret/environment variable."
    )
    st.stop()

client = OpenAI(api_key=api_key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# ============================================================
# PDF FUNCTIONS
# ============================================================

def get_pdf_bytes(uploaded_file):
    """
    Read the uploaded PDF once.

    The original PDF bytes are preserved and sent to the
    vision-capable model. This is important because extracted
    text alone cannot reliably preserve diagrams.
    """
    return uploaded_file.getvalue()


def get_page_count(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return 0


def extract_pdf_text(pdf_bytes):
    """
    Extract machine-readable text as SUPPLEMENTARY information.

    This is deliberately not the authoritative source.
    The original PDF is authoritative because it contains
    the actual visual page.
    """

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))

        pages = []

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""

            pages.append(
                f"--- PAGE {i} ---\n{text}"
            )

        return "\n\n".join(pages)

    except Exception as exc:
        return (
            "[TEXT EXTRACTION ERROR]\n"
            f"{exc}"
        )


# ============================================================
# QUESTION INVENTORY
# ============================================================

def split_question_numbers(text):
    """
    Best-effort question-number detection.

    This is ONLY a completeness aid.
    It must never override what the model sees in the PDF.
    """

    patterns = [
        r"(?im)^\s*(?:question\s*)?(\d{1,2})[.):-]\s+",
        r"(?im)^\s*q\s*(\d{1,2})[.):-]\s+",
    ]

    found = []

    for pattern in patterns:
        found.extend(re.findall(pattern, text))

    return list(dict.fromkeys(found))


# ============================================================
# ORIGINAL PDF DISPLAY
# ============================================================

def display_original_pdf(pdf_bytes):

    st.subheader("📄 Original Question Paper")

    st.caption(
        "The original PDF is preserved exactly as uploaded. "
        "Mwalimu AI uses the original page as the visual authority."
    )

    encoded = base64.b64encode(pdf_bytes).decode("utf-8")

    components.html(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded}"
            width="100%"
            height="850"
            style="
                border:1px solid #cccccc;
                border-radius:8px;
            ">
        </iframe>
        """,
        height=870,
        scrolling=True,
    )


# ============================================================
# DIAGRAM DETECTION HINTS
# ============================================================

def visual_keywords(subject):

    common = (
        "diagram, figure, graph, chart, table, drawing, "
        "illustration, apparatus, specimen, labelled, "
        "shown, image, construction"
    )

    if subject.lower() == "biology":
        return (
            common
            + ", biological drawing, specimen, cell, organ, "
              "tissue, structure, life cycle"
        )

    if subject.lower() == "physics":
        return (
            common
            + ", circuit, ray diagram, apparatus, force diagram, "
              "motion diagram, electrical diagram"
        )

    if subject.lower() == "chemistry":
        return (
            common
            + ", apparatus, laboratory setup, structure, "
              "organic structure, molecular diagram"
        )

    if subject.lower() == "mathematics":
        return (
            common
            + ", triangle, circle, angle, geometry, coordinate plane, "
              "shape, transformation, graph"
        )

    return common


# ============================================================
# VISUAL PROMPT
# ============================================================

def build_visual_prompt(
    subject,
    level,
    extracted_text,
):

    inventory = split_question_numbers(
        extracted_text
    )

    inventory_text = (
        ", ".join(inventory)
        if inventory
        else "Not reliably detected from text."
    )

    return f"""
You are Mwalimu AI, a rigorous Kenyan teacher-support
assistant.

SUBJECT:
{subject}

LEVEL:
{level}

PRELIMINARY TEXT QUESTION INVENTORY:
{inventory_text}

VISUAL ELEMENTS TO WATCH FOR:
{visual_keywords(subject)}


============================================================
PRIMARY SOURCE
============================================================

The ORIGINAL PDF attached to this request is the
authoritative examination paper.

You must visually inspect the actual PDF pages.

The extracted text supplied separately is supplementary
only.

Do NOT rely exclusively on OCR or PDF text extraction.

If extracted text conflicts with the visible PDF page,
trust the visible PDF page.


============================================================
DIAGRAM-FIRST MISSION
============================================================

THIS VERSION IS SPECIFICALLY TESTING DIAGRAM UNDERSTANDING.

Before generating the marking scheme, perform a visual
inspection of EVERY PAGE.

For each page, look carefully for:

- diagrams
- figures
- graphs
- tables
- charts
- apparatus
- specimens
- biological drawings
- circuit diagrams
- ray diagrams
- geometric figures
- labelled structures
- arrows
- lines
- measurements
- scales
- angles
- shaded regions
- construction marks
- captions
- visual symbols


============================================================
CONNECT VISUALS TO QUESTIONS
============================================================

For every visual element determine:

1. Which question it belongs to.
2. Which sub-question it belongs to, if applicable.
3. What visible labels are present.
4. What measurements are visible.
5. What arrows or lines are present.
6. What shapes or structures are visible.
7. What relationships are visually shown.
8. What information from the visual is required to answer
   the question.


============================================================
NEVER GUESS
============================================================

Do not invent:

- labels
- measurements
- coordinates
- dimensions
- angles
- biological structures
- apparatus
- graph values
- chemical structures

If the visual is genuinely unclear, explicitly say so.

Identify the exact question affected.

For example:

"Question 4(b): the diagram is present but the label at
the lower-right structure cannot be read confidently."


============================================================
MARKING SCHEME
============================================================

After completing the visual inspection, generate the normal
MVP3-style marking scheme.

Process EVERY question.

Process EVERY sub-question.

Keep the original question numbering.

Do not silently skip questions.

For calculation questions:

- show the formula
- substitute values
- show essential working
- give the final answer
- include units where required

For theory questions:

- give the expected answer
- include relevant marking points

For diagram questions:

- use information actually visible in the diagram
- explicitly refer to relevant labels/features
- do not substitute an invented diagram


============================================================
ORIGINAL DIAGRAM POLICY
============================================================

DO NOT recreate the original question paper.

DO NOT redraw the question diagram using ASCII art.

DO NOT replace the original diagram with an invented
description.

The original PDF displayed in the application remains the
visual reference.

The marking scheme should explain the answer using the
actual visual information that was inspected.


============================================================
BIOLOGY VISUALS
============================================================

For biological diagrams:

- inspect the actual drawing
- identify visible labels
- identify structures only when supported by the visual
- distinguish similar structures carefully
- use correct biological terminology

Do not invent missing labels.


============================================================
PHYSICS VISUALS
============================================================

For Physics diagrams:

- inspect arrows
- direction
- apparatus
- circuit connections
- ray paths
- measurements
- labels
- forces
- distances
- angles

Use only visible information.


============================================================
MATHEMATICS VISUALS
============================================================

For Mathematics diagrams:

- inspect shapes
- labels
- angles
- lengths
- coordinates
- lines
- curves
- shaded regions

Do not guess measurements that are not visible.


============================================================
CHEMISTRY VISUALS
============================================================

For Chemistry diagrams:

- inspect apparatus
- labels
- laboratory arrangement
- structures
- visible chemical notation

Do not invent chemical structures from an unclear image.


============================================================
SECOND VISUAL PASS
============================================================

Before finalising, perform a second pass through the PDF.

Ask:

- Did I inspect every page?
- Did I identify every diagram?
- Did I associate each diagram with the correct question?
- Did I use the visible information?
- Did I accidentally rely on OCR where the image contradicted it?
- Did I skip any question because its diagram was difficult?


============================================================
DIAGRAM AUDIT
============================================================

End with exactly:

[DIAGRAM AUDIT]

Pages visually inspected:
...

Diagram/visual questions detected:
...

Diagram/visual questions successfully interpreted:
...

Questions with unclear visual information:
...

Important visual information used:
...

[/DIAGRAM AUDIT]


============================================================
COMPLETENESS CHECK
============================================================

Then finish with:

[COMPLETENESS CHECK]

Questions identified:
...

Questions answered:
...

Sub-questions answered:
...

Unresolved questions:
...

No question was intentionally omitted.

[/COMPLETENESS CHECK]


IMPORTANT:

This is a diagram-understanding test.

Do NOT introduce new specialist-tool integrations.

Do NOT redesign the mathematics rendering.

Do NOT redesign chemistry notation.

Do NOT introduce GeoGebra.

Do NOT change the existing MVP3 marking philosophy.

The only objective of this upgrade is to improve the AI's
ability to SEE and USE diagrams from the original PDF.
"""


# ============================================================
# AI GENERATION
# ============================================================

def generate_marking_scheme(
    pdf_bytes,
    filename,
    subject,
    level,
    extracted_text,
):

    system_prompt = build_visual_prompt(
        subject,
        level,
        extracted_text,
    )

    encoded_pdf = base64.b64encode(
        pdf_bytes
    ).decode("utf-8")

    user_content = [
        {
            "type": "input_text",
            "text": (
                f"FILE: {filename}\n"
                f"PAGE COUNT: {get_page_count(pdf_bytes)}\n\n"
                "Inspect the attached original PDF visually, "
                "page by page. This is the DIAGRAM TEST version "
                "of Mwalimu AI."
            ),
        },
        {
            "type": "input_file",
            "filename": filename,
            "file_data": (
                f"data:application/pdf;base64,{encoded_pdf}"
            ),
        },
        {
            "type": "input_text",
            "text": (
                "SUPPLEMENTARY MACHINE-READABLE TEXT.\n"
                "THIS TEXT IS NOT AUTHORITATIVE.\n\n"
                f"{extracted_text[:60000]}"
            ),
        },
    ]

    response = client.responses.create(
        model=MODEL,
        temperature=0.1,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    return response.output_text


# ============================================================
# DISPLAY HELPERS
# ============================================================

def show_diagram_audit(scheme):

    match = re.search(
        r"\[DIAGRAM AUDIT\](.*?)\[/DIAGRAM AUDIT\]",
        scheme,
        flags=re.S | re.I,
    )

    if match:

        with st.expander(
            "🖼️ Diagram audit",
            expanded=True,
        ):
            st.markdown(
                match.group(1).strip()
            )


def show_completeness_check(scheme):

    match = re.search(
        r"\[COMPLETENESS CHECK\](.*?)\[/COMPLETENESS CHECK\]",
        scheme,
        flags=re.S | re.I,
    )

    if match:

        with st.expander(
            "🔍 Completeness check",
            expanded=False,
        ):
            st.markdown(
                match.group(1).strip()
            )


def clean_internal_blocks(scheme):

    scheme = re.sub(
        r"\[DIAGRAM AUDIT\].*?\[/DIAGRAM AUDIT\]",
        "",
        scheme,
        flags=re.S | re.I,
    )

    scheme = re.sub(
        r"\[COMPLETENESS CHECK\].*?\[/COMPLETENESS CHECK\]",
        "",
        scheme,
        flags=re.S | re.I,
    )

    return scheme.strip()


# ============================================================
# USER INTERFACE
# ============================================================

subject = st.selectbox(
    "Subject",
    [
        "Mathematics",
        "Chemistry",
        "Biology",
        "Physics",
        "Agriculture",
        "English",
        "Kiswahili",
        "IRE",
        "Other",
    ],
)

level = st.text_input(
    "Level / Grade",
    "Grade 10",
)

uploaded = st.file_uploader(
    "Upload a question paper (PDF)",
    type=["pdf"],
)


# ============================================================
# MAIN WORKFLOW
# ============================================================

if uploaded:

    pdf_bytes = get_pdf_bytes(
        uploaded
    )

    page_count = get_page_count(
        pdf_bytes
    )

    st.success(
        f"Loaded: {uploaded.name} — "
        f"{page_count} page(s)"
    )

    # Keep the original paper visible.
    display_original_pdf(
        pdf_bytes
    )

    if st.button(
        "🚀 Generate Marking Scheme",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Mwalimu AI is visually scanning the original paper..."
        ):

            try:

                extracted_text = extract_pdf_text(
                    pdf_bytes
                )

                if not extracted_text.strip():
                    extracted_text = (
                        "[No reliable machine-readable text "
                        "was extracted. Use the original PDF "
                        "as the primary source.]"
                    )

                scheme = generate_marking_scheme(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded.name,
                    subject=subject,
                    level=level,
                    extracted_text=extracted_text,
                )

                st.subheader(
                    "📝 Generated Marking Scheme"
                )

                st.markdown(
                    clean_internal_blocks(
                        scheme
                    )
                )

                show_diagram_audit(
                    scheme
                )

                show_completeness_check(
                    scheme
                )

                st.download_button(
                    "⬇️ Download marking scheme",
                    scheme,
                    file_name=(
                        "mwalimu_ai_marking_scheme.md"
                    ),
                    mime="text/markdown",
                    use_container_width=True,
                )

            except Exception as exc:

                st.error(
                    f"Generation failed: {exc}"
                )

                st.exception(
                    exc
                )


# ============================================================
# STATUS
# ============================================================

st.divider()

st.caption(
    "Mwalimu AI — Visual MVP3 | "
    "BATTLE 1: DIAGRAMS ONLY | "
    "Original PDF is authoritative; extracted text is supplementary."
)
