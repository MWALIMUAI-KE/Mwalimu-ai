import os
import io
import re
import base64
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI

# ============================================================
# MWALIMU AI — MVP 4
# Visual + Formula + GeoGebra
# ============================================================

st.set_page_config(
    page_title="Mwalimu AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mwalimu AI")
st.caption(
    "Teacher AI assistant — MVP 4: Visual paper analysis + Formula rendering + GeoGebra"
)

# ------------------------------------------------------------
# API
# ------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Add it as a deployment secret/environment variable."
    )
    st.stop()

client = OpenAI(api_key=api_key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# ------------------------------------------------------------
# PDF TEXT EXTRACTION — FALLBACK / TRACEABILITY
# ------------------------------------------------------------

def extract_pdf_text(pdf_bytes):
    """
    Extract machine-readable text when available.
    This is NOT treated as the primary visual-analysis method.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))

        pages = []

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""

            pages.append(
                f"\n--- PAGE {i} TEXT ---\n{text.strip()}"
            )

        return "\n".join(pages)

    except Exception as exc:
        return f"[TEXT EXTRACTION ERROR: {exc}]"


# ------------------------------------------------------------
# QUESTION INVENTORY
# ------------------------------------------------------------

def split_question_numbers(text):
    """
    Best-effort question inventory.
    The AI visual analysis remains authoritative when numbering
    is complex or diagrams interrupt the text.
    """

    patterns = [
        r"(?im)^\s*(?:question\s*)?(\d{1,2})[\.\):\-]\s+",
        r"(?im)^\s*(\d{1,2})\.\s+"
    ]

    found = []

    for pattern in patterns:
        found.extend(re.findall(pattern, text))

    # preserve order while removing duplicates
    result = []

    for number in found:
        if number not in result:
            result.append(number)

    return result


# ------------------------------------------------------------
# SPECIALIST NEED DETECTION
# ------------------------------------------------------------

def detect_needs(text, subject):
    t = text.lower()

    return {
        "geometry": bool(
            re.search(
                r"\b("
                r"construct|construction|locus|bisect|"
                r"perpendicular|parallel|bearing|circle|"
                r"tangent|transformation|reflection|rotation|"
                r"translation|enlargement|vector|geometry"
                r")\b",
                t
            )
        ),

        "graph": bool(
            re.search(
                r"\b("
                r"graph|plot|curve|coordinate|gradient|"
                r"line|quadratic|cubic|function|sketch"
                r")\b",
                t
            )
        ),

        "formula": subject.lower() in {
            "mathematics",
            "physics",
            "chemistry"
        } or bool(
            re.search(
                r"\b("
                r"equation|formula|fraction|indices|surds|"
                r"matrix|matrices|logarithm|differentiation|"
                r"integration|probability|quadratic|"
                r"velocity|acceleration|force|energy|"
                r"mole|ionic|reaction"
                r")\b",
                t
            )
        ),

        "chemistry": subject.lower() == "chemistry" or bool(
            re.search(
                r"\b("
                r"chemical|mole|ionic|organic|reaction|"
                r"equation|oxidation|reduction|electrolysis"
                r")\b",
                t
            )
        ),

        "diagram": bool(
            re.search(
                r"\b("
                r"diagram|figure|illustration|apparatus|"
                r"shown|shown below|shown above|sketch"
                r")\b",
                t
            )
        )
    }


# ------------------------------------------------------------
# OPENAI VISUAL ANALYSIS
# ------------------------------------------------------------

def analyse_paper_visually(pdf_bytes, filename, subject, level, extracted_text):
    """
    Send the actual PDF to the multimodal model.

    The PDF is supplied as a file input so the model can reason over
    the document itself rather than relying only on extracted text.
    """

    uploaded = client.files.create(
        file=(filename, pdf_bytes),
        purpose="user_data"
    )

    question_inventory = split_question_numbers(extracted_text)

    needs = detect_needs(extracted_text, subject)

    inventory = (
        ", ".join(question_inventory)
        if question_inventory
        else "Numbering could not be reliably extracted."
    )

    prompt = f"""
You are Mwalimu AI, a rigorous Kenyan teacher-support and examination
marking assistant.

SUBJECT:
{subject}

LEVEL:
{level}

PRELIMINARY TEXT INVENTORY:
{inventory}

SPECIALIST INDICATORS:
Geometry/construction: {needs["geometry"]}
Graphing: {needs["graph"]}
Formula-heavy: {needs["formula"]}
Chemistry notation: {needs["chemistry"]}
Diagrams/figures: {needs["diagram"]}

IMPORTANT:
The uploaded PDF itself is the authoritative question paper.

Do NOT rely only on extracted text.

VISUALLY INSPECT THE ACTUAL DOCUMENT.

You must pay attention to:
- diagrams
- graphs
- tables
- geometric figures
- circuit diagrams
- apparatus
- arrows
- labels
- axes
- scales
- fractions
- roots
- superscripts
- subscripts
- Greek letters
- mathematical symbols
- chemical formulae
- state symbols
- charges
- handwritten/printed annotations
- information contained inside figures

============================================================
CORE REQUIREMENTS
============================================================

1. Process EVERY question.

2. Process EVERY sub-question.

3. Preserve the original numbering exactly.

4. Do not silently skip questions.

5. If a question contains a diagram, use the diagram when solving it.

6. If a diagram is unclear, identify the exact question and say what
   information is unreadable. DO NOT invent the missing information.

7. Show essential working for numerical questions.

8. Give marks beside important marking points/steps where appropriate.

9. Preserve mathematical notation.

10. Use LaTeX for mathematical expressions.

11. Preserve chemical formulae, charges and state symbols correctly.

12. For graphs, give the mathematical solution AND a structured
    GeoGebra section.

13. For constructions, give the mathematical/geometrical reasoning
    AND structured GeoGebra commands where appropriate.

14. Never say merely "use GeoGebra".

15. Never claim GeoGebra was automatically executed unless it actually
    was executed.

16. Never claim FormulAI was automatically executed unless an actual
    FormulAI service was used.

17. FormulAI-style output means the formula should be clean,
    structured and suitable for conversion to Word.

============================================================
FORMULA FORMAT
============================================================

Use:

\\( ... \\)

for inline mathematics.

Use:

\\[
...
\\]

for displayed mathematics.

Examples:

\\[
x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}
\\]

\\[
F = ma
\\]

\\[
n = \\frac{{m}}{{M}}
\\]

Do NOT replace mathematical symbols with crude text such as:

sqrt(x)
1/2
x2

when proper notation is possible.

============================================================
CHEMISTRY
============================================================

Write chemistry correctly.

Examples:

\\[
2H_2 + O_2 \\rightarrow 2H_2O
\\]

\\[
Cu^{{2+}} + 2OH^- \\rightarrow Cu(OH)_2
\\]

Include state symbols when they are given or required:

\\[
Zn(s) + 2HCl(aq) \\rightarrow ZnCl_2(aq) + H_2(g)
\\]

Do not use ChemType tags.

Do not mention ChemType.

============================================================
GEOGEBRA
============================================================

When a graph/construction/geometry task requires GeoGebra, produce:

[GEOGEBRA]
Question: ...
Purpose: ...
Input/Commands:
1. ...
2. ...
3. ...
Expected result: ...
[/GEOGEBRA]

Use legitimate GeoGebra-style commands when appropriate, for example:

Point(A)
Line(A,B)
Segment(A,B)
Circle(A,B)
PerpendicularLine(A,g)
ParallelLine(A,g)
Intersect(g,h)
Polygon(A,B,C)
Reflect(A,g)
Rotate(A,alpha)
Translate(A,v)

Do not invent coordinates or dimensions that are absent from the paper.

============================================================
VISUAL VERIFICATION
============================================================

For every question involving a figure, include a short:

[VISUAL CHECK]
Question: ...
Visual information used: ...
[/VISUAL CHECK]

This is essential for proving that the diagram was actually considered.

============================================================
COMPLETENESS CHECK
============================================================

Finish with:

[COMPLETENESS CHECK]

Questions identified:
...

Questions processed:
...

Sub-questions processed:
...

Visual questions checked:
...

GeoGebra tasks:
...

Unresolved/unclear items:
...

[/COMPLETENESS CHECK]

If everything was processed successfully, explicitly state:

"No question or sub-question was intentionally omitted."

============================================================
QUALITY STANDARD
============================================================

Correctness > verbosity.

Traceability > generic explanations.

The final result must be usable by a Kenyan teacher as a marking scheme.
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": uploaded.id
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    return response.output_text


# ------------------------------------------------------------
# FORMULA / LATEX RENDERING
# ------------------------------------------------------------

def render_scheme(text):
    """
    Streamlit's Markdown renderer supports LaTeX.
    The model is instructed to use \\(...\\) and \\[...\\].
    """

    st.markdown(text)


# ------------------------------------------------------------
# SPECIALIST BLOCKS
# ------------------------------------------------------------

def show_geogebra_blocks(scheme):

    blocks = re.findall(
        r"\[GEOGEBRA\](.*?)\[/GEOGEBRA\]",
        scheme,
        flags=re.S | re.I
    )

    if not blocks:
        return

    st.subheader("📐 GeoGebra tasks")

    for i, block in enumerate(blocks, 1):

        with st.expander(
            f"GeoGebra task {i}",
            expanded=True
        ):

            st.code(
                block.strip(),
                language="text"
            )

            st.caption(
                "The commands above are generated from the question "
                "paper and can be reproduced in GeoGebra."
            )


def show_visual_checks(scheme):

    blocks = re.findall(
        r"\[VISUAL CHECK\](.*?)\[/VISUAL CHECK\]",
        scheme,
        flags=re.S | re.I
    )

    if not blocks:
        return

    st.subheader("👁️ Visual verification")

    for i, block in enumerate(blocks, 1):

        with st.expander(
            f"Visual check {i}",
            expanded=False
        ):

            st.write(block.strip())


def show_completeness(scheme):

    match = re.search(
        r"\[COMPLETENESS CHECK\](.*?)\[/COMPLETENESS CHECK\]",
        scheme,
        flags=re.S | re.I
    )

    if not match:
        return

    st.subheader("✅ Completeness check")

    st.info(match.group(1).strip())


# ------------------------------------------------------------
# FORMULAI HANDOFF
# ------------------------------------------------------------

def formulai_panel(scheme):

    st.subheader("∑ FormulAI — formula workspace")

    st.caption(
        "Mwalimu AI generates clean mathematical notation. "
        "Use FormulAI when you need a Word-ready editable equation."
    )

    st.link_button(
        "Open FormulAI Formula Generator",
        "https://formulai.io/formula-generator"
    )

    st.info(
        "FormulAI currently provides a browser-based formula workflow "
        "rather than a public developer API. Mwalimu AI therefore does "
        "not pretend to call a private FormulAI API. The generated "
        "LaTeX/formula notation can be copied into FormulAI for Word-ready "
        "equations."
    )

    # Extract displayed mathematics for easy copying.
    display_math = re.findall(
        r"\\\[(.*?)\\\]",
        scheme,
        flags=re.S
    )

    inline_math = re.findall(
        r"\\\((.*?)\\\)",
        scheme,
        flags=re.S
    )

    formulas = display_math + inline_math

    if formulas:

        with st.expander(
            f"📋 Extracted formulas ({len(formulas)})",
            expanded=False
        ):

            for i, formula in enumerate(formulas, 1):

                st.markdown(f"**Formula {i}**")

                st.code(
                    formula.strip(),
                    language="text"
                )


# ------------------------------------------------------------
# GEOGEBRA PANEL
# ------------------------------------------------------------

def geogebra_panel():

    st.subheader("📐 GeoGebra")

    st.caption(
        "Use GeoGebra for graphing, geometry and constructions. "
        "Generated commands appear above when a question requires them."
    )

    tabs = st.tabs(
        [
            "Geometry",
            "Graphing",
            "Calculator"
        ]
    )

    urls = [
        "https://www.geogebra.org/geometry",
        "https://www.geogebra.org/graphing",
        "https://www.geogebra.org/calculator"
    ]

    for tab, url in zip(tabs, urls):

        with tab:

            components.iframe(
                url,
                height=650,
                scrolling=True
            )


# ------------------------------------------------------------
# QUESTION PAPER PREVIEW
# ------------------------------------------------------------

def show_text_trace(text):

    with st.expander(
        "🔎 Machine-readable text trace",
        expanded=False
    ):

        st.text(
            text[:30000]
        )

        if len(text) > 30000:
            st.caption(
                "Text trace truncated for display. "
                "The visual PDF remains the authoritative source."
            )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

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
        "Other"
    ]
)

level = st.text_input(
    "Level / Grade",
    "Grade 10"
)

uploaded = st.file_uploader(
    "Upload a question paper (PDF)",
    type=["pdf"]
)


if uploaded:

    st.success(
        f"Loaded: {uploaded.name}"
    )

    pdf_bytes = uploaded.getvalue()

    text = extract_pdf_text(pdf_bytes)

    show_text_trace(text)

    if st.button(
        "🚀 Generate Visual Marking Scheme",
        type="primary"
    ):

        with st.spinner(
            "Mwalimu AI is visually analysing the question paper..."
        ):

            try:

                scheme = analyse_paper_visually(
                    pdf_bytes,
                    uploaded.name,
                    subject,
                    level,
                    text
                )

                st.subheader(
                    "Generated Marking Scheme"
                )

                render_scheme(scheme)

                show_visual_checks(scheme)

                show_geogebra_blocks(scheme)

                show_completeness(scheme)

                formulai_panel(scheme)

                st.download_button(
                    "⬇️ Download marking scheme",
                    scheme,
                    file_name="mwalimu_ai_marking_scheme.md",
                    mime="text/markdown"
                )

            except Exception as exc:

                st.error(
                    f"Generation failed: {exc}"
                )

                st.exception(exc)


# ------------------------------------------------------------
# DEVELOPMENT TOOLS
# ------------------------------------------------------------

with st.expander(
    "🧪 Mwalimu AI specialist workspace"
):

    st.markdown(
        """
### Current MVP 4 architecture

**Paper**
→ Visual PDF analysis  
→ Text + diagrams + formulas + tables  
→ Question inventory  
→ Question-by-question reasoning  
→ Mathematical notation  
→ GeoGebra instructions  
→ Completeness verification  
→ Marking scheme

**MathType:** removed  
**ChemType:** removed  
**FormulAI:** Word-ready formula handoff  
**GeoGebra:** graph/construction workspace
"""
    )

    geogebra_panel()


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Mwalimu AI MVP 4 — Visual + Formula + GeoGebra road-test build."
)
