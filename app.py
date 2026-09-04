import os
import io
import re
import base64

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI

try:
    import fitz
except ImportError:
    fitz = None


# ============================================================
# MWALIMU AI — MVP 4.2 VISUAL MARKING
# ORIGINAL PAPER + VISUAL ANALYSIS + VISUAL QUESTION CROPS
# FORMULA RENDERING + GEOGEBRA
# ============================================================

st.set_page_config(
    page_title="Mwalimu AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mwalimu AI")
st.caption(
    "Teacher AI assistant — MVP 4.2: Original paper + "
    "Visual analysis + Original visual question rendering + "
    "Formula rendering + GeoGebra"
)


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

key = os.getenv("OPENAI_API_KEY")

if not key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Add it as a deployment secret."
    )
    st.stop()

client = OpenAI(api_key=key)

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.4-mini"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(data):
    try:
        reader = PdfReader(io.BytesIO(data))

        pages = []

        for i, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            pages.append(
                f"--- PAGE {i} TEXT ---\n{page_text}"
            )

        return "\n".join(pages)

    except Exception as e:
        return f"[TEXT EXTRACTION ERROR: {e}]"


# ============================================================
# ORIGINAL FULL-PAGE RENDERING
# ============================================================

def render_pages(data):
    """
    Render the original PDF pages as PNG images.

    This is deliberately based on the ORIGINAL PDF.
    Nothing is reconstructed by OCR.
    """

    if fitz is None:
        return []

    try:
        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )

        out = []

        matrix = fitz.Matrix(
            150 / 72,
            150 / 72
        )

        for n, page in enumerate(doc, 1):

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            out.append(
                (
                    n,
                    pix.tobytes("png")
                )
            )

        doc.close()

        return out

    except Exception:
        return []


def show_original(data):
    """
    Display the complete original question paper.
    """

    st.subheader("📄 Original Question Paper")

    st.caption(
        "The original uploaded paper is shown here. "
        "It is NOT reconstructed by OCR."
    )

    pages = render_pages(data)

    if pages:

        for n, img in pages:

            st.image(
                img,
                caption=(
                    f"Original question paper — page {n}"
                ),
                use_container_width=True
            )

    else:

        b64 = base64.b64encode(data).decode()

        components.html(
            f"""
            <iframe
                src="data:application/pdf;base64,{b64}"
                width="100%"
                height="900"
                style="border:1px solid #ddd;">
            </iframe>
            """,
            height=920,
            scrolling=True
        )


# ============================================================
# QUESTION DETECTION
# ============================================================

QUESTION_PATTERN = re.compile(
    r"^\s*(?:QUESTION\s*)?(\d{1,2})[.\):\-]\s+",
    re.IGNORECASE
)


def question_numbers(text):
    """
    Extract question numbers from machine-readable text.
    """

    found = re.findall(
        r"(?im)^\s*(?:question\s*)?(\d{1,2})[.\):\-]\s+",
        text
    )

    return list(
        dict.fromkeys(found)
    )


# ============================================================
# VISUAL CUE DETECTION
# ============================================================

VISUAL_KEYWORDS = [
    "diagram",
    "figure",
    "fig.",
    "graph",
    "plot",
    "curve",
    "sketch",
    "construction",
    "construct",
    "locus",
    "apparatus",
    "shown",
    "illustration",
    "table",
    "chart",
    "draw",
    "drawing",
    "map",
    "circle",
    "triangle",
    "quadrilateral",
    "polygon",
    "angle",
    "bearing",
    "coordinate",
    "axes",
    "grid",
    "solubility curve",
    "bar chart",
    "histogram",
    "pie chart",
    "frequency polygon",
    "frequency curve"
]


def contains_visual_cue(text):
    """
    Determine whether a question is likely to contain
    an original visual element.
    """

    t = text.lower()

    for keyword in VISUAL_KEYWORDS:

        if keyword in t:
            return True

    return False


# ============================================================
# QUESTION REGION DETECTION
# ============================================================

def detect_question_regions(data):
    """
    Locate question starts directly inside the ORIGINAL PDF.

    Returns regions of the form:

        {
            "question": "12",
            "page": 3,
            "y0": ...,
            "y1": ...,
            "text": ...
        }

    This lets Mwalimu AI display the actual original
    visual region belonging to a question.
    """

    if fitz is None:
        return []

    regions = []

    try:

        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )

        # ----------------------------------------------------
        # First pass: find every question start
        # ----------------------------------------------------

        starts = []

        for page_index, page in enumerate(doc):

            blocks = page.get_text(
                "blocks"
            )

            # Sort top-to-bottom, left-to-right
            blocks = sorted(
                blocks,
                key=lambda b: (
                    round(b[1], 2),
                    round(b[0], 2)
                )
            )

            for block in blocks:

                block_text = block[4] or ""

                lines = block_text.splitlines()

                for line_index, line in enumerate(lines):

                    match = QUESTION_PATTERN.match(
                        line
                    )

                    if match:

                        question_number = (
                            match.group(1)
                        )

                        # Estimate y-position of the
                        # matching line within the block.
                        block_height = (
                            block[3] - block[1]
                        )

                        line_count = max(
                            len(lines),
                            1
                        )

                        line_height = (
                            block_height / line_count
                        )

                        y0 = (
                            block[1]
                            +
                            line_index * line_height
                        )

                        starts.append(
                            {
                                "question": question_number,
                                "page": page_index,
                                "y0": y0,
                                "x0": block[0],
                                "text": block_text
                            }
                        )

        # ----------------------------------------------------
        # Remove accidental duplicate detections
        # ----------------------------------------------------

        cleaned = []

        seen = set()

        for item in starts:

            key_tuple = (
                item["page"],
                item["question"],
                round(item["y0"], 1)
            )

            if key_tuple in seen:
                continue

            seen.add(key_tuple)

            cleaned.append(item)

        starts = sorted(
            cleaned,
            key=lambda x: (
                x["page"],
                x["y0"]
            )
        )

        # ----------------------------------------------------
        # Build question regions
        # ----------------------------------------------------

        for i, start in enumerate(starts):

            page_index = start["page"]

            page = doc[page_index]

            page_height = page.rect.height
            page_width = page.rect.width

            current_y0 = start["y0"]

            # Find next question
            next_start = None

            if i + 1 < len(starts):
                next_start = starts[i + 1]

            if (
                next_start is not None
                and next_start["page"] == page_index
            ):

                current_y1 = (
                    next_start["y0"]
                    - 6
                )

            else:

                current_y1 = (
                    page_height
                    - 5
                )

            # Safety limits
            current_y0 = max(
                0,
                current_y0 - 10
            )

            current_y1 = min(
                page_height,
                current_y1 + 10
            )

            if current_y1 <= current_y0:
                continue

            # ------------------------------------------------
            # Extract text belonging approximately to region
            # ------------------------------------------------

            region_text_parts = []

            blocks = page.get_text(
                "blocks"
            )

            for block in blocks:

                bx0, by0, bx1, by1 = block[:4]
                btext = block[4] or ""

                # Block overlaps question region
                if (
                    by1 >= current_y0
                    and by0 <= current_y1
                ):

                    region_text_parts.append(
                        btext
                    )

            region_text = "\n".join(
                region_text_parts
            )

            regions.append(
                {
                    "question": start["question"],
                    "page": page_index,
                    "y0": current_y0,
                    "y1": current_y1,
                    "x0": 0,
                    "x1": page_width,
                    "text": region_text
                }
            )

        doc.close()

        return regions

    except Exception:
        return []


# ============================================================
# ORIGINAL VISUAL QUESTION CROPS
# ============================================================

def render_question_crop(
    data,
    page_number,
    y0,
    y1,
    x0=0,
    x1=None
):
    """
    Render an actual cropped portion of the original PDF.

    This is the critical visual-preservation layer.
    """

    if fitz is None:
        return None

    try:

        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )

        page = doc[page_number]

        if x1 is None:
            x1 = page.rect.width

        # Clamp crop to page
        x0 = max(
            0,
            min(x0, page.rect.width)
        )

        x1 = max(
            x0 + 1,
            min(x1, page.rect.width)
        )

        y0 = max(
            0,
            min(y0, page.rect.height)
        )

        y1 = max(
            y0 + 1,
            min(y1, page.rect.height)
        )

        clip = fitz.Rect(
            x0,
            y0,
            x1,
            y1
        )

        # Higher resolution for diagrams,
        # graphs and mathematical notation.
        matrix = fitz.Matrix(
            2.0,
            2.0
        )

        pix = page.get_pixmap(
            matrix=matrix,
            clip=clip,
            alpha=False
        )

        image = pix.tobytes(
            "png"
        )

        doc.close()

        return image

    except Exception:
        return None


def show_visual_question_regions(data):
    """
    Display the ORIGINAL visual portions of questions.

    This does not redraw diagrams.
    It crops the real PDF itself.
    """

    regions = detect_question_regions(
        data
    )

    if not regions:
        return

    visual_regions = []

    for region in regions:

        if contains_visual_cue(
            region["text"]
        ):

            visual_regions.append(
                region
            )

    # --------------------------------------------------------
    # If visual keywords were not extracted reliably,
    # don't falsely claim there are no diagrams.
    # --------------------------------------------------------

    if not visual_regions:

        st.subheader(
            "🖼️ Original Visual Question Regions"
        )

        st.info(
            "No visual question was reliably identified "
            "from the extracted text. The full original "
            "paper above remains the authoritative visual source."
        )

        return

    st.subheader(
        "🖼️ Original Visual Question Regions"
    )

    st.caption(
        "These are crops taken directly from the uploaded "
        "PDF. Diagrams, graphs, tables and sketches are "
        "therefore preserved exactly as they appear in the "
        "original paper."
    )

    for region in visual_regions:

        q = region["question"]

        page_no = region["page"] + 1

        with st.expander(
            f"Question {q} — original visual region "
            f"(page {page_no})",
            expanded=True
        ):

            image = render_question_crop(
                data,
                region["page"],
                region["y0"],
                region["y1"],
                region["x0"],
                region["x1"]
            )

            if image:

                st.image(
                    image,
                    caption=(
                        f"Original Question {q} "
                        f"visual region — page {page_no}"
                    ),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Unable to render this question crop. "
                    "The original page remains available above."
                )


# ============================================================
# SUBJECT-SPECIFIC VISUAL / MATHEMATICAL NEEDS
# ============================================================

def needs(text, subject):

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
                r"quadratic|cubic|function|sketch"
                r")\b",
                t
            )
        ),

        "formula": (
            subject.lower()
            in {
                "mathematics",
                "physics",
                "chemistry"
            }
            or
            bool(
                re.search(
                    r"\b("
                    r"equation|formula|fraction|indices|"
                    r"surds|matrix|logarithm|differentiation|"
                    r"integration|probability|quadratic"
                    r")\b",
                    t
                )
            )
        ),

        "diagram": bool(
            re.search(
                r"\b("
                r"diagram|figure|apparatus|shown|"
                r"sketch|graph|table|chart|draw|"
                r"construction|curve"
                r")\b",
                t
            )
        )
    }


# ============================================================
# AI VISUAL ANALYSIS + MARKING SCHEME
# ============================================================

def analyse(
    data,
    filename,
    subject,
    level,
    text
):

    # Upload the ORIGINAL PDF
    f = client.files.create(
        file=(
            filename,
            data
        ),
        purpose="user_data"
    )

    ns = needs(
        text,
        subject
    )

    nums = ", ".join(
        question_numbers(text)
    )

    if not nums:
        nums = (
            "Not reliably extracted. "
            "Determine numbering visually from the PDF."
        )

    prompt = f"""
You are Mwalimu AI, a rigorous Kenyan examination
marking assistant.

Subject: {subject}
Level: {level}

Preliminary question inventory:
{nums}

Geometry:
{ns["geometry"]}

Graphing:
{ns["graph"]}

Formula-heavy:
{ns["formula"]}

Diagrams:
{ns["diagram"]}


============================================================
AUTHORITATIVE SOURCE
============================================================

The uploaded PDF is the AUTHORITATIVE question paper.

You MUST visually inspect the actual PDF.

Do not rely only on machine-readable OCR/text extraction.

Inspect:

- diagrams
- sketches
- graphs
- tables
- geometric figures
- apparatus
- labels
- arrows
- axes
- scales
- fractions
- mathematical symbols
- superscripts
- subscripts
- chemical structures
- chemical ions
- charges
- reaction schemes
- any information contained visually in figures


============================================================
IMPORTANT VISUAL RULE
============================================================

The application displays the ORIGINAL question paper
to the teacher.

The application also displays original visual regions
cropped directly from the uploaded PDF.

Therefore:

DO NOT recreate the question.

DO NOT rewrite the question.

DO NOT paraphrase the question.

DO NOT invent a replacement diagram.

DO NOT substitute a textual description for information
that can be read from the original visual.

Generate ONLY the answers / marking scheme that belong
underneath the original questions.


============================================================
QUESTION COMPLETENESS
============================================================

Process EVERY question.

Process EVERY sub-question.

Do not silently skip any question.

Do not silently skip any sub-question.

Preserve the original numbering.

If a question contains:

(a), (b), (c)

or

(i), (ii), (iii)

process each part separately.


============================================================
WORKING
============================================================

For Mathematics and other calculation-heavy subjects:

Show essential working.

Do not provide only the final answer.

Award marks logically according to the method shown.

Use proper LaTeX for mathematical notation.


============================================================
VISUAL QUESTIONS
============================================================

For every question involving a diagram, graph, figure,
construction, table, apparatus or other visual element:

Actually inspect the visual information.

Use the visual information when solving the question.

If the visual is unclear or unreadable:

identify the exact question.

State exactly what information is unclear.

Do NOT invent missing information.


============================================================
GEOMETRY / CONSTRUCTION / GRAPH QUESTIONS
============================================================

For geometry, construction and graph questions, provide
useful GeoGebra commands in this exact block:

[GEOGEBRA]
Question: ...
Purpose: ...
Input/Commands:
1. ...
2. ...
Expected result: ...
[/GEOGEBRA]


============================================================
VISUAL VERIFICATION
============================================================

For visually inspected questions, use:

[VISUAL CHECK]
Question: ...
Visual information used: ...
[/VISUAL CHECK]


============================================================
COMPLETENESS CHECK
============================================================

Finish with:

[COMPLETENESS CHECK]
Questions identified: ...
Questions processed: ...
Sub-questions processed: ...
Visual questions checked: ...
GeoGebra tasks: ...
Unresolved/unclear items: ...
[/COMPLETENESS CHECK]

If complete, explicitly state:

No question or sub-question was intentionally omitted.


============================================================
FINAL RULES
============================================================

Check calculations.

Check units.

Check signs.

Check chemical formulae.

Check ionic charges.

Check superscripts and subscripts.

Check fractions.

Check mathematical notation.

Check graph interpretation.

Check diagram interpretation.

Do not mention ChemType.

Do not mention MathType.

Do not claim that a diagram has been generated if
you are merely describing the original diagram.

The original visual belongs to the uploaded paper.
The marking scheme should answer it.
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": f.id
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


# ============================================================
# MAIN MARKING SCHEME DISPLAY
# ============================================================

def main_scheme(s):

    for tag in (
        "GEOGEBRA",
        "VISUAL CHECK",
        "COMPLETENESS CHECK"
    ):

        s = re.sub(
            rf"\[{tag}\].*?\[/{tag}\]",
            "",
            s,
            flags=re.S | re.I
        )

    return s.strip()


# ============================================================
# SPECIALIST VISUAL / GEOGEBRA / COMPLETENESS OUTPUT
# ============================================================

def specialist(s):

    visuals = re.findall(
        r"\[VISUAL CHECK\](.*?)\[/VISUAL CHECK\]",
        s,
        flags=re.S | re.I
    )

    geo = re.findall(
        r"\[GEOGEBRA\](.*?)\[/GEOGEBRA\]",
        s,
        flags=re.S | re.I
    )

    complete = re.search(
        r"\[COMPLETENESS CHECK\](.*?)\[/COMPLETENESS CHECK\]",
        s,
        flags=re.S | re.I
    )

    if visuals:

        st.subheader(
            "👁️ Visual verification"
        )

        for i, x in enumerate(
            visuals,
            1
        ):

            with st.expander(
                f"Visual check {i}"
            ):

                st.write(
                    x.strip()
                )

    if geo:

        st.subheader(
            "📐 GeoGebra tasks"
        )

        for i, x in enumerate(
            geo,
            1
        ):

            with st.expander(
                f"GeoGebra task {i}",
                expanded=True
            ):

                st.code(
                    x.strip()
                )

    if complete:

        st.subheader(
            "✅ Completeness check"
        )

        st.info(
            complete.group(1).strip()
        )


# ============================================================
# FORMULA WORKSPACE
# ============================================================

def formula_workspace(s):

    st.subheader(
        "∑ FormulAI — formula workspace"
    )

    st.caption(
        "Mwalimu AI produces clean LaTeX notation. "
        "FormulAI can convert equations to editable "
        "Word-ready formulas."
    )

    st.link_button(
        "Open FormulAI Formula Generator",
        "https://formulai.io/formula-generator"
    )

    fs = (
        re.findall(
            r"\\\[(.*?)\\\]",
            s,
            flags=re.S
        )
        +
        re.findall(
            r"\\\((.*?)\\\)",
            s,
            flags=re.S
        )
    )

    if fs:

        with st.expander(
            f"📋 Extracted formulas ({len(fs)})"
        ):

            for i, formula in enumerate(
                fs,
                1
            ):

                st.code(
                    formula.strip()
                )


# ============================================================
# GEOGEBRA
# ============================================================

def geogebra():

    st.subheader(
        "📐 GeoGebra"
    )

    a, b, c = st.tabs(
        [
            "Geometry",
            "Graphing",
            "Calculator"
        ]
    )

    urls = (
        "https://www.geogebra.org/geometry",
        "https://www.geogebra.org/graphing",
        "https://www.geogebra.org/calculator"
    )

    for tab, url in zip(
        (a, b, c),
        urls
    ):

        with tab:

            components.iframe(
                url,
                height=650,
                scrolling=True
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


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded:

    data = uploaded.getvalue()

    st.success(
        f"Loaded: {uploaded.name}"
    )

    # --------------------------------------------------------
    # ORIGINAL PAPER
    # --------------------------------------------------------

    show_original(
        data
    )

    # --------------------------------------------------------
    # NEW:
    # ORIGINAL QUESTION VISUAL CROPS
    # --------------------------------------------------------

    show_visual_question_regions(
        data
    )

    # --------------------------------------------------------
    # MACHINE-READABLE TEXT TRACE
    # --------------------------------------------------------

    text = extract_pdf_text(
        data
    )

    with st.expander(
        "🔎 Machine-readable text trace"
    ):

        st.text(
            text[:30000]
        )

    # --------------------------------------------------------
    # GENERATE MARKING SCHEME
    # --------------------------------------------------------

    if st.button(
        "📝 Generate Marking Scheme",
        type="primary"
    ):

        with st.spinner(
            "Analysing the original paper, including "
            "diagrams, graphs, tables and mathematical notation..."
        ):

            try:

                st.session_state[
                    "scheme"
                ] = analyse(
                    data,
                    uploaded.name,
                    subject,
                    level,
                    text
                )

            except Exception as e:

                st.error(
                    f"Unable to generate marking scheme: {e}"
                )

    # --------------------------------------------------------
    # DISPLAY GENERATED SCHEME
    # --------------------------------------------------------

    if st.session_state.get(
        "scheme"
    ):

        s = st.session_state[
            "scheme"
        ]

        st.divider()

        st.subheader(
            "📝 Answers / Marking Scheme"
        )

        st.caption(
            "The original paper and original visual regions "
            "are above. The answers and workings below "
            "correspond to those original questions."
        )

        st.markdown(
            main_scheme(s)
        )

        # ----------------------------------------------------
        # VISUAL CHECKS / GEOGEBRA / COMPLETENESS
        # ----------------------------------------------------

        specialist(
            s
        )

        # ----------------------------------------------------
        # FORMULA WORKSPACE
        # ----------------------------------------------------

        formula_workspace(
            s
        )

        st.divider()

        # ----------------------------------------------------
        # GEOGEBRA
        # ----------------------------------------------------

        geogebra()
