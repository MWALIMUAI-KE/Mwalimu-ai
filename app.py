import os
import io
import re
import base64

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI


# ============================================================
# MWALIMU AI — VISUAL MVP3 UPGRADE
# BATTLE 1: REAL VISUAL QUESTION UNDERSTANDING
#
# MVP3 PRINCIPLE:
# Original PDF = authoritative source
# Extracted text = supplementary source
#
# NEW:
# PASS 1 = visual evidence scan
# PASS 2 = marking scheme using visual evidence + original PDF
# ============================================================


st.set_page_config(
    page_title="Mwalimu AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Mwalimu AI")
st.caption(
    "Teacher AI assistant — Visual MVP3 | "
    "Two-pass diagram understanding"
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

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini"
)


# ============================================================
# PDF HELPERS
# ============================================================

def get_pdf_bytes(uploaded_file):
    """
    Read the uploaded PDF once and preserve the exact bytes.
    """

    return uploaded_file.getvalue()


def get_page_count(pdf_bytes):
    try:
        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )
        return len(reader.pages)

    except Exception:
        return 0


def extract_pdf_text(pdf_bytes):
    """
    Extract machine-readable text.

    IMPORTANT:
    This is supplementary only.

    The original PDF remains authoritative because
    diagrams and visual information may not appear in
    extracted text.
    """

    try:
        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            pages.append(
                f"--- PAGE {page_number} ---\n{text}"
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

    This is NOT authoritative.
    It is only a completeness aid.
    """

    patterns = [
        r"(?im)^\s*(?:question\s*)?"
        r"(\d{1,2})[.):-]\s+",

        r"(?im)^\s*q\s*"
        r"(\d{1,2})[.):-]\s+",
    ]

    found = []

    for pattern in patterns:

        found.extend(
            re.findall(
                pattern,
                text
            )
        )

    return list(
        dict.fromkeys(found)
    )


# ============================================================
# VISUAL KEYWORDS
# ============================================================

def visual_keywords(subject):

    common = (
        "diagram, figure, graph, chart, table, drawing, "
        "illustration, apparatus, specimen, labelled, "
        "shown, image, construction, arrow, scale"
    )

    subject = subject.lower()

    if subject == "mathematics":

        return (
            common
            + ", triangle, circle, angle, geometry, "
              "coordinate plane, shape, transformation, "
              "curve, straight line, shaded region, "
              "construction"
        )

    if subject == "biology":

        return (
            common
            + ", biological drawing, specimen, cell, "
              "organ, tissue, structure, life cycle, "
              "microscope"
        )

    if subject == "physics":

        return (
            common
            + ", circuit, ray diagram, apparatus, "
              "force diagram, motion diagram, electrical "
              "diagram, velocity, vector"
        )

    if subject == "chemistry":

        return (
            common
            + ", laboratory apparatus, setup, molecular "
              "structure, organic structure, reaction "
              "diagram"
        )

    return common


# ============================================================
# PASS 1 — VISUAL EVIDENCE PROMPT
# ============================================================

def build_visual_audit_prompt(
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
        else "Not reliably detected from extracted text."
    )

    return f"""
You are Mwalimu AI, a rigorous Kenyan teacher-support
assistant.

SUBJECT:
{subject}

LEVEL:
{level}

PRELIMINARY QUESTION INVENTORY:
{inventory_text}

VISUAL ELEMENTS TO WATCH FOR:
{visual_keywords(subject)}


============================================================
MISSION — PASS 1
============================================================

This is a VISUAL EXAMINATION PAPER.

Your first task is NOT to generate the marking scheme.

Your first task is to VISUALLY INSPECT THE ORIGINAL PDF.

Inspect EVERY PAGE.

The original PDF is the authoritative source.

Extracted text is supplementary only.


============================================================
PAGE-BY-PAGE VISUAL INSPECTION
============================================================

For every page:

1. Identify all visible questions.
2. Identify all visible sub-questions.
3. Identify every diagram.
4. Identify every graph.
5. Identify every table.
6. Identify every construction.
7. Identify every labelled figure.
8. Identify arrows and directions.
9. Identify visible measurements.
10. Identify angles.
11. Identify coordinates.
12. Identify graph axes and values.
13. Identify shaded regions.
14. Identify apparatus.
15. Identify biological structures.
16. Identify circuit connections.
17. Identify chemical structures or notation.


============================================================
QUESTION ↔ VISUAL CONNECTION
============================================================

For every diagram or visual element state:

QUESTION:
...

SUB-QUESTION:
...

PAGE:
...

VISUAL TYPE:
...

VISIBLE INFORMATION:
...

HOW THE VISUAL RELATES TO THE QUESTION:
...

CONFIDENCE:
HIGH / MEDIUM / LOW


============================================================
DO NOT GUESS
============================================================

Never invent:

- labels
- measurements
- coordinates
- angles
- dimensions
- graph values
- apparatus
- biological structures
- chemical structures

If something is unclear, say exactly what is unclear.

Do not turn an unclear visual into a confident answer.


============================================================
MATHEMATICS
============================================================

Pay particular attention to:

- geometry
- diagrams
- constructions
- graphs
- curves
- coordinates
- angles
- lengths
- perpendicular lines
- parallel lines
- transformations
- shaded regions
- scale drawings

Do NOT assume a diagram is decorative.

Determine whether it supplies information needed to solve
the question.


============================================================
PHYSICS
============================================================

Pay particular attention to:

- circuits
- apparatus
- force arrows
- ray diagrams
- measurements
- distances
- angles
- directions
- motion diagrams


============================================================
BIOLOGY
============================================================

Pay particular attention to:

- biological drawings
- labels
- cells
- tissues
- organs
- specimens
- structures
- arrows
- life-cycle diagrams


============================================================
CHEMISTRY
============================================================

Pay particular attention to:

- laboratory apparatus
- experimental arrangements
- chemical structures
- reaction diagrams
- labels
- visible chemical notation


============================================================
OUTPUT
============================================================

Return a structured VISUAL EVIDENCE MAP.

Use exactly this structure:

[VISUAL EVIDENCE MAP]

PAGE 1
Visuals detected:
...

Questions affected:
...

Important visible information:
...

PAGE 2
Visuals detected:
...

Questions affected:
...

Important visible information:
...

Continue for every page.

Then:

[QUESTION VISUAL MAP]

Question 1:
Visual: YES/NO
Page:
Relevant visual information:

Question 2:
Visual: YES/NO
Page:
Relevant visual information:

Continue for every detected question.

Then:

[UNCLEAR VISUALS]

...

[/UNCLEAR VISUALS]

Then:

[VISUAL CONFIDENCE]

Overall:
...

Questions requiring special visual attention:
...

[/VISUAL CONFIDENCE]

[/VISUAL EVIDENCE MAP]


IMPORTANT:

Do not solve the paper yet.

This pass exists specifically to make sure the visual
information is not lost before the marking scheme is created.
"""


# ============================================================
# PASS 1 — RUN VISUAL AUDIT
# ============================================================

def run_visual_audit(
    pdf_bytes,
    filename,
    subject,
    level,
    extracted_text,
):

    system_prompt = build_visual_audit_prompt(
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
                "VISUAL PASS 1.\n"
                "Inspect the original PDF page by page.\n"
                "Do not solve the questions yet."
            ),
        },

        {
            "type": "input_file",
            "filename": filename,
            "file_data": (
                "data:application/pdf;base64,"
                + encoded_pdf
            ),
        },

        {
            "type": "input_text",
            "text": (
                "SUPPLEMENTARY EXTRACTED TEXT.\n"
                "NOT AUTHORITATIVE.\n\n"
                + extracted_text[:60000]
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
# PASS 2 — MARKING SCHEME PROMPT
# ============================================================

def build_marking_prompt(
    subject,
    level,
    extracted_text,
    visual_audit,
):

    inventory = split_question_numbers(
        extracted_text
    )

    inventory_text = (
        ", ".join(inventory)
        if inventory
        else "Not reliably detected."
    )

    return f"""
You are Mwalimu AI, a rigorous Kenyan teacher-support
assistant creating a professional marking scheme.

SUBJECT:
{subject}

LEVEL:
{level}

AUTOMATIC TEXT INVENTORY:
{inventory_text}


============================================================
PRIMARY SOURCE
============================================================

The ORIGINAL PDF is the authoritative examination paper.

The visual evidence map is an additional analysis of the
original PDF.

Extracted text is supplementary only.

If anything conflicts:

1. Original visible PDF
2. Visual evidence
3. Extracted text


============================================================
VISUAL EVIDENCE FROM PASS 1
============================================================

{visual_audit}


============================================================
MISSION — PASS 2
============================================================

Now generate the COMPLETE marking scheme.

You must process EVERY question.

You must process EVERY sub-question.

Do not silently skip a question.

Do not stop because a question contains a diagram.

Use the visual evidence map to connect diagrams to their
correct questions.


============================================================
QUESTION-FIRST FORMAT
============================================================

For every question reproduce enough of the ORIGINAL QUESTION
TEXT to make the marking scheme traceable.

Then immediately provide:

ANSWER / WORKING

Then:

MARKING POINTS

The basic pattern should be:

QUESTION 1(a)
[faithful question text]

ANSWER / WORKING:
...

MARKING:
- ...
- ...

QUESTION 1(b)
[faithful question text]

ANSWER / WORKING:
...

MARKING:
- ...
- ...


============================================================
DIAGRAM QUESTIONS
============================================================

For a question containing a diagram:

Do NOT invent a replacement diagram.

Do NOT pretend the diagram does not exist.

Explicitly acknowledge it.

For example:

QUESTION 4(b)
[faithful question text]

VISUAL USED:
Diagram on page 2.

OBSERVED:
- ...
- ...
- ...

ANSWER / WORKING:
...

MARKING:
...


If the diagram contains labels, measurements, angles,
coordinates, arrows or other information required to solve
the question, use them.

Do not use guessed values.


============================================================
MATHEMATICS WORKING
============================================================

For calculations:

1. State the formula.
2. Substitute values.
3. Show essential working.
4. Simplify correctly.
5. Give final answer.
6. Include units where required.

Use LaTeX for mathematical expressions.

Use:

\\( ... \\)

for inline mathematics.

Use:

\\[
...
\\]

for displayed mathematics.


============================================================
GEOMETRY / CONSTRUCTIONS / GRAPHS
============================================================

If a diagram is required to solve the question:

Use the actual visible diagram.

Identify:

- points
- lines
- lengths
- angles
- coordinates
- curves
- axes
- scale
- shaded areas
- construction relationships

Do not invent missing measurements.

If a construction is required, describe the actual
construction steps needed for marking.


============================================================
CHEMISTRY
============================================================

Preserve:

- chemical formulae
- subscripts
- charges
- state symbols
- balanced equations
- oxidation states where relevant

Show essential working for mole calculations and
stoichiometry.


============================================================
BIOLOGY
============================================================

For labelled diagrams:

- identify only structures supported by the visual
- use correct terminology
- associate each label with the actual structure
- do not invent unreadable labels


============================================================
COMPLETENESS
============================================================

Before finishing, check:

- every main question processed
- every sub-question processed
- every diagram question processed
- every graph question processed
- every construction processed
- every calculation has working
- every final answer is visible
- no question silently omitted


============================================================
FINAL AUDIT
============================================================

Finish with:

[COMPLETENESS CHECK]

Questions identified:
...

Questions answered:
...

Sub-questions answered:
...

Diagram questions:
...

Graph/construction questions:
...

Unresolved questions:
...

[/COMPLETENESS CHECK]


Then:

[VISUAL MARKING CHECK]

Visual questions actually used in answering:
...

Questions where the diagram was unclear:
...

Important visual information used:
...

[/VISUAL MARKING CHECK]


IMPORTANT:

Do NOT introduce GeoGebra.

Do NOT introduce Formulai.

Do NOT introduce MathType.

Do NOT introduce ChemType.

Do NOT redesign the application.

This is a controlled VISUAL MVP3 test.

The objective is:

SEE THE PAPER
→ SEE THE DIAGRAM
→ CONNECT DIAGRAM TO QUESTION
→ SOLVE USING THE VISUAL
→ PRODUCE THE MARKING SCHEME.
"""


# ============================================================
# PASS 2 — GENERATE MARKING SCHEME
# ============================================================

def generate_marking_scheme(
    pdf_bytes,
    filename,
    subject,
    level,
    extracted_text,
    visual_audit,
):

    system_prompt = build_marking_prompt(
        subject,
        level,
        extracted_text,
        visual_audit,
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
                "PASS 2: Generate the complete marking scheme.\n"
                "Use the original PDF visually and the visual "
                "evidence map below."
            ),
        },

        {
            "type": "input_file",
            "filename": filename,
            "file_data": (
                "data:application/pdf;base64,"
                + encoded_pdf
            ),
        },

        {
            "type": "input_text",
            "text": (
                "VISUAL EVIDENCE MAP FROM PASS 1:\n\n"
                + visual_audit[:60000]
            ),
        },

        {
            "type": "input_text",
            "text": (
                "SUPPLEMENTARY EXTRACTED TEXT:\n\n"
                + extracted_text[:60000]
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
# ORIGINAL PDF DISPLAY
# ============================================================

def display_original_pdf(pdf_bytes):

    st.subheader(
        "📄 Original Question Paper"
    )

    st.caption(
        "The original PDF remains visible and is the "
        "authoritative visual source."
    )

    encoded = base64.b64encode(
        pdf_bytes
    ).decode("utf-8")

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
# AUDIT DISPLAY
# ============================================================

def show_visual_audit(audit):

    if not audit:
        return

    st.subheader(
        "👁️ Visual Scan Report"
    )

    with st.expander(
        "What Mwalimu AI actually saw",
        expanded=True,
    ):

        st.markdown(
            audit
        )


# ============================================================
# CLEAN INTERNAL CHECK BLOCKS
# ============================================================

def clean_internal_blocks(text):

    text = re.sub(
        r"\[COMPLETENESS CHECK\].*?"
        r"\[/COMPLETENESS CHECK\]",
        "",
        text,
        flags=re.S | re.I,
    )

    text = re.sub(
        r"\[VISUAL MARKING CHECK\].*?"
        r"\[/VISUAL MARKING CHECK\]",
        "",
        text,
        flags=re.S | re.I,
    )

    return text.strip()


# ============================================================
# SHOW FINAL MARKING SCHEME
# ============================================================

def show_marking_scheme(scheme):

    st.subheader(
        "📝 Generated Marking Scheme"
    )

    cleaned = clean_internal_blocks(
        scheme
    )

    st.markdown(
        cleaned
    )

    with st.expander(
        "🔍 Final visual/completeness verification",
        expanded=False,
    ):

        visual_match = re.search(
            r"\[VISUAL MARKING CHECK\](.*?)"
            r"\[/VISUAL MARKING CHECK\]",
            scheme,
            flags=re.S | re.I,
        )

        completeness_match = re.search(
            r"\[COMPLETENESS CHECK\](.*?)"
            r"\[/COMPLETENESS CHECK\]",
            scheme,
            flags=re.S | re.I,
        )

        if visual_match:

            st.markdown(
                "### Visual marking check"
            )

            st.markdown(
                visual_match.group(1).strip()
            )

        if completeness_match:

            st.markdown(
                "### Completeness check"
            )

            st.markdown(
                completeness_match.group(1).strip()
            )


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

    # --------------------------------------------------------
    # ALWAYS SHOW ORIGINAL
    # --------------------------------------------------------

    display_original_pdf(
        pdf_bytes
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "🚀 Scan Visually & Generate Marking Scheme",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ------------------------------------------------
            # EXTRACT SUPPLEMENTARY TEXT
            # ------------------------------------------------

            with st.spinner(
                "Step 1/2 — Reading the paper and "
                "building visual evidence..."
            ):

                extracted_text = extract_pdf_text(
                    pdf_bytes
                )

                if not extracted_text.strip():

                    extracted_text = (
                        "[No reliable machine-readable "
                        "text was extracted. The original "
                        "PDF remains the primary source.]"
                    )

                # ------------------------------------------------
                # PASS 1
                # ------------------------------------------------

                visual_audit = run_visual_audit(

                    pdf_bytes=pdf_bytes,
                    filename=uploaded.name,
                    subject=subject,
                    level=level,
                    extracted_text=extracted_text,
                )

            # ------------------------------------------------
            # SHOW VISUAL REPORT
            # ------------------------------------------------

            show_visual_audit(
                visual_audit
            )

            # ------------------------------------------------
            # PASS 2
            # ------------------------------------------------

            with st.spinner(
                "Step 2/2 — Using the visual evidence to "
                "solve every question..."
            ):

                scheme = generate_marking_scheme(

                    pdf_bytes=pdf_bytes,
                    filename=uploaded.name,
                    subject=subject,
                    level=level,
                    extracted_text=extracted_text,
                    visual_audit=visual_audit,
                )

            # ------------------------------------------------
            # FINAL OUTPUT
            # ------------------------------------------------

            show_marking_scheme(
                scheme
            )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            st.download_button(

                "⬇️ Download marking scheme",

                scheme,

                file_name=(
                    "mwalimu_ai_visual_mvp3_marking_scheme.md"
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
    "Mwalimu AI — Visual MVP3 Upgrade | "
    "BATTLE 1: TWO-PASS VISUAL UNDERSTANDING | "
    "Original PDF authoritative | "
    "Extracted text supplementary"
)
