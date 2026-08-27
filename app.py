import os
import io
import re
import base64

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mwalimu AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mwalimu AI")
st.caption(
    "Teacher AI assistant — Visual MVP3: "
    "original paper + visual analysis + workings + formulas + diagrams"
)


# ============================================================
# OPENAI
# ============================================================

key = os.getenv("OPENAI_API_KEY")

if not key:
    st.error(
        "OPENAI_API_KEY is not configured. "
        "Add it as a deployment secret."
    )
    st.stop()

client = OpenAI(api_key=key)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(data):
    """
    Extract machine-readable text.

    This is supplementary information only.
    The visual page images remain authoritative.
    """

    try:
        reader = PdfReader(io.BytesIO(data))

        pages = []

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""

            pages.append(
                f"--- PAGE {i} TEXT ---\n{text}"
            )

        return "\n\n".join(pages)

    except Exception as e:
        return f"[TEXT EXTRACTION ERROR: {e}]"


# ============================================================
# PDF PAGE RENDERING
# ============================================================

def render_pages(data, dpi=150):
    """
    Render every PDF page to PNG.

    This is the critical visual layer.
    """

    if fitz is None:
        return []

    try:
        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )

        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)

        pages = []

        for page_number, page in enumerate(doc, 1):

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            png = pix.tobytes("png")

            pages.append(
                {
                    "page": page_number,
                    "image": png
                }
            )

        doc.close()

        return pages

    except Exception:
        return []


# ============================================================
# DISPLAY ORIGINAL PAPER
# ============================================================

def show_original(data):

    st.subheader("📄 Original Question Paper")

    st.caption(
        "This is the actual uploaded paper. "
        "Mwalimu AI does not reconstruct the question paper."
    )

    pages = render_pages(data)

    if pages:

        for item in pages:

            st.image(
                item["image"],
                caption=f"Original paper — page {item['page']}",
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

def question_numbers(text):

    patterns = [
        r"(?im)^\s*(?:question\s*)?(\d{1,2})[.\):\-]\s+",
        r"(?im)^\s*(\d{1,2})\s*[.)]\s+"
    ]

    found = []

    for pattern in patterns:
        found.extend(
            re.findall(pattern, text)
        )

    return list(dict.fromkeys(found))


# ============================================================
# DETECT VISUAL / MATHEMATICAL CONTENT
# ============================================================

def detect_features(text, subject):

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
                r"quadratic|cubic|function|sketch|axis"
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
            or bool(
                re.search(
                    r"\b("
                    r"equation|formula|fraction|indices|"
                    r"surds|matrix|logarithm|differentiation|"
                    r"integration|probability|quadratic|"
                    r"calculate|calculate the"
                    r")\b",
                    t
                )
            )
        ),

        "diagram": bool(
            re.search(
                r"\b("
                r"diagram|figure|apparatus|shown|"
                r"sketch|graph|illustration|drawing|"
                r"construction"
                r")\b",
                t
            )
        )
    }


# ============================================================
# BUILD VISUAL INPUT
# ============================================================

def build_visual_inputs(pages):

    content = []

    for item in pages:

        encoded = base64.b64encode(
            item["image"]
        ).decode("utf-8")

        content.append(
            {
                "type": "input_text",
                "text": (
                    f"\n\n===== ORIGINAL PAPER PAGE "
                    f"{item['page']} =====\n"
                    "Inspect this page visually. "
                    "All diagrams, graphs, tables, "
                    "symbols, figures, constructions and "
                    "layout on this page are authoritative."
                )
            }
        )

        content.append(
            {
                "type": "input_image",
                "image_url": (
                    f"data:image/png;base64,{encoded}"
                )
            }
        )

    return content


# ============================================================
# MARKING SCHEME PROMPT
# ============================================================

def build_prompt(
    subject,
    level,
    text,
    features,
    numbers
):

    return f"""
You are Mwalimu AI, a rigorous Kenyan examination
marking-scheme assistant.

SUBJECT:
{subject}

LEVEL / GRADE:
{level}

PRELIMINARY QUESTION NUMBERS:
{numbers}

VISUAL FEATURES DETECTED:

Geometry:
{features["geometry"]}

Graphs:
{features["graph"]}

Formula-heavy:
{features["formula"]}

Diagrams:
{features["diagram"]}


============================================================
MOST IMPORTANT INSTRUCTION
============================================================

The ORIGINAL PAGE IMAGES supplied with this request are
the authoritative question paper.

You must visually inspect the actual pages.

DO NOT rely only on OCR or extracted text.

The extracted text is supplementary.

When text extraction conflicts with the original page image,
trust the original page image.


============================================================
VISUAL UNDERSTANDING
============================================================

You must inspect:

• diagrams
• graphs
• geometric constructions
• tables
• axes
• labels
• arrows
• measurements
• fractions
• mathematical symbols
• indices
• roots
• equations
• chemical formulae
• chemical structures
• apparatus
• biological drawings
• maps
• charts
• shaded regions
• lines and angles
• labels attached to diagrams


If a question refers to:

"the diagram below"

"the figure"

"the graph"

"the apparatus shown"

"the construction"

or similar wording,

you MUST inspect the corresponding visual material
before answering.


============================================================
QUESTION COMPLETENESS
============================================================

Process EVERY question.

Process EVERY sub-question.

Do NOT silently omit questions.

Preserve the original numbering.

If the paper contains:

1(a)
1(b)
1(c)

then answer:

1(a)
1(b)
1(c)

Do not merge them.


============================================================
OUTPUT STYLE
============================================================

Generate a professional teacher-ready marking scheme.

For each question use:

QUESTION 1(a)

Answer:
...

Working:
...

Marks:
...

QUESTION 1(b)

Answer:
...

Working:
...

Marks:
...


For theory questions, give the precise expected answer.

For calculation questions, show the essential working.

For Mathematics and Physics:

• show equations clearly
• show substitution
• show calculations
• show units
• give the final answer
• use LaTeX where appropriate

For Chemistry:

• write chemical formulae correctly
• balance equations correctly
• show calculations
• preserve charges, subscripts and states where relevant

For Biology:

• use correct biological terminology
• identify structures accurately
• use the actual diagram where relevant

For geometry:

• identify the construction required
• explain construction steps
• use the actual measurements/labels visible in the diagram
• do not invent measurements

For graph questions:

• identify the axes
• identify scales
• identify coordinates/data
• explain plotting
• calculate gradient where required
• identify intercepts where required
• describe the curve correctly


============================================================
DIAGRAMS
============================================================

DO NOT attempt to recreate an existing diagram using
ordinary text.

Instead, describe what the diagram shows accurately
when necessary and solve the question using the actual
visual.

If the diagram contains information essential to the
answer, explicitly state that information.


============================================================
UNCLEAR VISUAL INFORMATION
============================================================

If a diagram or symbol genuinely cannot be read:

DO NOT GUESS.

Write:

"Visual information unclear — please inspect the original
page."

Then identify the exact question affected.


============================================================
FORMULAS
============================================================

Use proper mathematical notation.

Examples:

$$
a^2+b^2=c^2
$$

$$
v = u + at
$$

$$
\\frac{{a}}{{b}}
$$

Do not replace mathematical notation with crude OCR such as:

a2 + b2 = c2

when proper notation is possible.


============================================================
MARKS
============================================================

Award marks according to the likely examination logic.

Do not invent excessive marks.

Where workings are required, distribute marks logically
between method and final answer.


============================================================
FINAL COMPLETENESS AUDIT
============================================================

At the end output:

[COMPLETENESS CHECK]

Questions identified:
...

Questions answered:
...

Sub-questions answered:
...

Visual questions inspected:
...

Diagram questions:
...

Graph questions:
...

Formula/calculation questions:
...

Unclear questions:
...

No question or sub-question was intentionally omitted.

[/COMPLETENESS CHECK]


Do not mention:

ChemType
MathType
internal software
OCR limitations

unless specifically necessary to explain an unresolved
visual issue.
"""


# ============================================================
# CALL VISION MODEL
# ============================================================

def analyse_paper(
    data,
    filename,
    subject,
    level,
    text
):

    pages = render_pages(
        data,
        dpi=150
    )

    if not pages:

        raise RuntimeError(
            "The PDF pages could not be rendered. "
            "Please ensure PyMuPDF is installed."
        )

    features = detect_features(
        text,
        subject
    )

    numbers = (
        ", ".join(question_numbers(text))
        or "Not reliably extracted"
    )

    prompt = build_prompt(
        subject=subject,
        level=level,
        text=text,
        features=features,
        numbers=numbers
    )

    visual_content = build_visual_inputs(
        pages
    )

    content = [

        {
            "type": "input_text",
            "text": prompt
        },

        {
            "type": "input_text",
            "text": (
                "\n\n===== SUPPLEMENTARY MACHINE-READABLE "
                "TEXT =====\n"
                f"{text[:50000]}"
            )
        }

    ]

    content.extend(
        visual_content
    )

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.output_text


# ============================================================
# CLEAN MARKING SCHEME
# ============================================================

def extract_completeness(text):

    match = re.search(
        r"\[COMPLETENESS CHECK\](.*?)"
        r"\[/COMPLETENESS CHECK\]",
        text,
        flags=re.S | re.I
    )

    return (
        match.group(1).strip()
        if match
        else None
    )


def clean_scheme(text):

    text = re.sub(
        r"\[COMPLETENESS CHECK\].*?"
        r"\[/COMPLETENESS CHECK\]",
        "",
        text,
        flags=re.S | re.I
    )

    return text.strip()


# ============================================================
# FORMULA DISPLAY
# ============================================================

def formula_workspace(text):

    st.subheader(
        "∑ Formula & Mathematical Notation"
    )

    st.caption(
        "Mwalimu AI preserves mathematical notation "
        "using LaTeX. FormulAI can be used when an "
        "editable Word-ready formula is required."
    )

    st.link_button(
        "Open FormulAI Formula Generator",
        "https://formulai.io/formula-generator"
    )

    formulas = []

    formulas.extend(
        re.findall(
            r"\$\$(.*?)\$\$",
            text,
            flags=re.S
        )
    )

    formulas.extend(
        re.findall(
            r"\\\[(.*?)\\\]",
            text,
            flags=re.S
        )
    )

    formulas.extend(
        re.findall(
            r"\\\((.*?)\\\)",
            text,
            flags=re.S
        )
    )

    if formulas:

        with st.expander(
            f"📋 Detected formulas ({len(formulas)})"
        ):

            for i, formula in enumerate(
                formulas,
                1
            ):

                st.markdown(
                    f"**Formula {i}**"
                )

                st.code(
                    formula.strip()
                )


# ============================================================
# GEOGEBRA
# ============================================================

def geogebra_workspace():

    st.subheader("📐 GeoGebra Workspace")

    st.caption(
        "Useful for geometry, constructions and graphing."
    )

    geometry, graphing, calculator = st.tabs(
        [
            "Geometry",
            "Graphing",
            "Calculator"
        ]
    )

    with geometry:

        components.iframe(
            "https://www.geogebra.org/geometry",
            height=650,
            scrolling=True
        )

    with graphing:

        components.iframe(
            "https://www.geogebra.org/graphing",
            height=650,
            scrolling=True
        )

    with calculator:

        components.iframe(
            "https://www.geogebra.org/calculator",
            height=650,
            scrolling=True
        )


# ============================================================
# VISUAL AUDIT
# ============================================================

def visual_audit(text):

    visual_items = re.findall(
        r"\[VISUAL CHECK\](.*?)"
        r"\[/VISUAL CHECK\]",
        text,
        flags=re.S | re.I
    )

    if visual_items:

        st.subheader(
            "👁️ Visual Analysis"
        )

        for i, item in enumerate(
            visual_items,
            1
        ):

            with st.expander(
                f"Visual analysis {i}"
            ):

                st.write(
                    item.strip()
                )


# ============================================================
# MAIN APPLICATION
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


if uploaded:

    data = uploaded.getvalue()

    st.success(
        f"Loaded: {uploaded.name}"
    )

    # --------------------------------------------------------
    # ORIGINAL PAPER
    # --------------------------------------------------------

    show_original(data)

    # --------------------------------------------------------
    # TEXT TRACE
    # --------------------------------------------------------

    text = extract_pdf_text(data)

    with st.expander(
        "🔎 Supplementary machine-readable text"
    ):

        st.text(
            text[:30000]
        )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "📝 Generate Visual Marking Scheme",
        type="primary"
    ):

        with st.spinner(
            "Mwalimu AI is visually inspecting every "
            "page, diagram, graph, formula and question..."
        ):

            try:

                scheme = analyse_paper(
                    data=data,
                    filename=uploaded.name,
                    subject=subject,
                    level=level,
                    text=text
                )

                st.session_state[
                    "scheme"
                ] = scheme

            except Exception as e:

                st.error(
                    f"Unable to generate marking scheme: {e}"
                )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get("scheme"):

    scheme = st.session_state["scheme"]

    st.divider()

    st.subheader(
        "📝 Generated Marking Scheme"
    )

    st.caption(
        "The original question paper remains above. "
        "The answers and workings below are generated "
        "from both the extracted text and the actual "
        "visual pages."
    )

    st.markdown(
        clean_scheme(scheme)
    )

    # --------------------------------------------------------
    # VISUAL AUDIT
    # --------------------------------------------------------

    visual_audit(scheme)

    # --------------------------------------------------------
    # COMPLETENESS
    # --------------------------------------------------------

    completeness = extract_completeness(
        scheme
    )

    if completeness:

        st.subheader(
            "✅ Completeness Audit"
        )

        st.info(
            completeness
        )

    # --------------------------------------------------------
    # FORMULAS
    # --------------------------------------------------------

    formula_workspace(
        scheme
    )

    # --------------------------------------------------------
    # GEOGEBRA
    # --------------------------------------------------------

    st.divider()

    geogebra_workspace()
