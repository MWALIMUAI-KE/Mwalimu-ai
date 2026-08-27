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
# MWALIMU AI — MVP 4.2
# Original Paper + Page Visual Analysis + Formula Support
# + GeoGebra + Completeness Tracking
# ============================================================

st.set_page_config(
    page_title="Mwalimu AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mwalimu AI")
st.caption(
    "Teacher AI assistant — MVP 4.2: "
    "Original paper + visual question analysis + formulas + GeoGebra"
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

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.4-mini"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(data):
    """
    Extract whatever machine-readable text exists in the PDF.

    This is NOT treated as authoritative for diagrams, formulae,
    symbols or page layout. The rendered page image is authoritative
    for visual information.
    """

    try:
        reader = PdfReader(io.BytesIO(data))

        pages = []

        for i, page in enumerate(reader.pages, 1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""

            pages.append(
                f"--- PAGE {i} TEXT ---\n{txt}"
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

    These images are used for:
      - displaying the real paper
      - visual inspection
      - page/question mapping
    """

    if fitz is None:
        return []

    try:
        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )

        pages = []

        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)

        for page_number, page in enumerate(doc, 1):

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_bytes = pix.tobytes("png")

            pages.append(
                {
                    "page": page_number,
                    "image": image_bytes
                }
            )

        doc.close()

        return pages

    except Exception:
        return []


# ============================================================
# ORIGINAL PAPER DISPLAY
# ============================================================

def show_original(data):

    st.subheader("📄 Original Question Paper")

    st.caption(
        "The original uploaded paper is displayed unchanged. "
        "Mwalimu AI does not reconstruct the question paper."
    )

    pages = render_pages(data)

    if pages:

        for item in pages:

            st.image(
                item["image"],
                caption=(
                    f"Original question paper — "
                    f"page {item['page']}"
                ),
                use_container_width=True
            )

    else:

        encoded = base64.b64encode(data).decode()

        components.html(
            f"""
            <iframe
                src="data:application/pdf;base64,{encoded}"
                width="100%"
                height="900"
                style="border:1px solid #ddd;"
            ></iframe>
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
        r"(?im)^\s*(\d{1,2})\s*[.\)]\s+",
    ]

    found = []

    for pattern in patterns:

        try:
            found.extend(
                re.findall(pattern, text)
            )
        except Exception:
            pass

    return list(
        dict.fromkeys(found)
    )


# ============================================================
# VISUAL / SUBJECT NEED DETECTION
# ============================================================

def needs(text, subject):

    t = text.lower()

    geometry = bool(
        re.search(
            r"\b("
            r"construct|construction|locus|bisect|"
            r"perpendicular|parallel|bearing|circle|"
            r"tangent|transformation|reflection|rotation|"
            r"translation|enlargement|vector|geometry"
            r")\b",
            t
        )
    )

    graph = bool(
        re.search(
            r"\b("
            r"graph|plot|curve|coordinate|gradient|"
            r"quadratic|cubic|function|sketch|axis"
            r")\b",
            t
        )
    )

    formula = (
        subject.lower()
        in {
            "mathematics",
            "physics",
            "chemistry"
        }
        or bool(
            re.search(
                r"\b("
                r"equation|formula|fraction|indices|surds|"
                r"matrix|logarithm|differentiation|"
                r"integration|probability|quadratic|"
                r"velocity|acceleration|force|energy|"
                r"mole|concentration"
                r")\b",
                t
            )
        )
    )

    diagram = bool(
        re.search(
            r"\b("
            r"diagram|figure|apparatus|shown|"
            r"sketch|graph|illustration|drawing|"
            r"image|shape"
            r")\b",
            t
        )
    )

    return {
        "geometry": geometry,
        "graph": graph,
        "formula": formula,
        "diagram": diagram
    }


# ============================================================
# VISUAL PAGE ANALYSIS
# ============================================================

def build_visual_pages(data):

    pages = render_pages(
        data,
        dpi=160
    )

    return pages


def analyse_visual_pages(
    data,
    filename,
    subject,
    level,
    text
):
    """
    Sends the ORIGINAL PDF to the model together with rendered
    page images.

    The page images are explicitly labelled so the model is asked
    to inspect diagrams, equations, tables and graphical content
    rather than relying exclusively on OCR/PDF extraction.
    """

    file_obj = client.files.create(
        file=(filename, data),
        purpose="user_data"
    )

    page_images = build_visual_pages(data)

    ns = needs(
        text,
        subject
    )

    nums = ", ".join(
        question_numbers(text)
    )

    if not nums:
        nums = "Not reliably extracted from text."


    # --------------------------------------------------------
    # Visual page inventory
    # --------------------------------------------------------

    page_inventory = []

    for item in page_images:

        page_inventory.append(
            f"PAGE {item['page']}: "
            f"Rendered visual page available for inspection."
        )

    page_inventory_text = "\n".join(
        page_inventory
    )


    # --------------------------------------------------------
    # MASTER PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are Mwalimu AI, a rigorous Kenyan examination marking assistant.

SUBJECT:
{subject}

LEVEL:
{level}

PRELIMINARY TEXT QUESTION INVENTORY:
{nums}

DETECTED NEEDS:
Geometry = {ns["geometry"]}
Graphing = {ns["graph"]}
Formula-heavy = {ns["formula"]}
Diagrams = {ns["diagram"]}

VISUAL PAGE INVENTORY:
{page_inventory_text}

============================================================
CORE RULE
============================================================

The uploaded PDF is the AUTHORITATIVE ORIGINAL QUESTION PAPER.

The original paper must NOT be reconstructed.

The application separately displays the original PDF to the
teacher.

Your job is to analyse the original examination paper and
produce ONLY the answers / marking scheme corresponding to
the original questions.

DO NOT rewrite the question paper.

DO NOT paraphrase the question into a replacement question.

DO NOT invent diagrams.

DO NOT invent numbers, labels, dimensions or symbols.

DO NOT silently omit questions.

============================================================
VISUAL-FIRST INSTRUCTION
============================================================

Inspect the actual visual pages carefully.

For every page, look for:

- diagrams
- geometric figures
- constructions
- graphs
- axes
- tables
- apparatus
- labelled drawings
- arrows
- measurements
- fractions
- mathematical symbols
- chemical formulae
- subscripts
- superscripts
- indices
- roots
- equations
- matrices
- signs
- units
- handwritten-style markings if present
- shaded regions
- curves
- angles
- bearings
- labels attached to diagrams

The rendered page image is especially important whenever
ordinary text extraction is incomplete.

If text extraction says one thing but the visible page clearly
shows something different, trust the visible page.

If something is genuinely unreadable, report it rather than
guessing.

============================================================
QUESTION MAPPING
============================================================

For every question:

1. Identify the page containing it.
2. Identify the question number.
3. Identify all sub-questions.
4. Identify whether it contains visual material.
5. Identify any diagram/graph/table/formula needed to solve it.
6. Solve it.
7. Provide the marking points.
8. Preserve the original numbering.

Do not combine unrelated questions.

Do not move an answer to another question.

============================================================
MATHEMATICS
============================================================

For mathematical questions:

- Show essential working.
- Show substitutions.
- Show intermediate calculations where useful.
- Give the final answer clearly.
- Use proper LaTeX.
- Preserve fractions.
- Preserve roots.
- Preserve powers and indices.
- Preserve mathematical symbols.
- Do not replace equations with vague prose.

For graph questions:

- identify axes
- identify scale if visible
- identify coordinates
- identify plotted points
- identify the curve
- identify intercepts where relevant
- explain the required graph result
- provide GeoGebra commands where useful

For construction questions:

- identify the exact construction required
- identify visible points/labels
- give construction steps
- provide GeoGebra commands where useful

============================================================
CHEMISTRY
============================================================

For Chemistry:

- preserve chemical formulae
- preserve subscripts
- balance equations
- show mole calculations where required
- show substitutions
- show units
- identify observations
- distinguish observations from conclusions
- do not invent apparatus or diagram labels

============================================================
PHYSICS
============================================================

For Physics:

- state the relevant formula
- substitute values
- show working
- include units
- give final answer
- inspect diagrams and circuit/force/optics drawings
- interpret graphs visually

============================================================
BIOLOGY
============================================================

For Biology:

- inspect biological diagrams
- identify labels from the original figure
- answer according to the visible figure
- do not invent labels
- include marking points
- preserve sequence where processes are involved

============================================================
AGRICULTURE AND OTHER SUBJECTS
============================================================

Use the same principle:

The visible original paper is authoritative.

Answer every question and every sub-question.

============================================================
VISUAL CHECK BLOCK
============================================================

For every question that relies on visual information, output:

[VISUAL CHECK]
Question: ...
Page: ...
Visual information used: ...
Confidence: High / Medium / Low
[/VISUAL CHECK]

If the visual is unclear:

[VISUAL CHECK]
Question: ...
Page: ...
Visual information used: ...
Confidence: Low
Unclear element: ...
[/VISUAL CHECK]

NEVER invent missing visual information.

============================================================
GEOGEBRA BLOCK
============================================================

For geometry, construction and graph questions where GeoGebra
would help, output:

[GEOGEBRA]
Question: ...
Purpose: ...
Input/Commands:
1. ...
2. ...
3. ...
Expected result: ...
[/GEOGEBRA]

Commands must be practical GeoGebra commands.

============================================================
FORMULA BLOCK
============================================================

For important mathematical/scientific equations output:

[FORMULA]
Question: ...
Formula:
LaTeX: ...
Meaning / substitution: ...
[/FORMULA]

============================================================
COMPLETENESS
============================================================

At the end output:

[COMPLETENESS CHECK]
Questions identified: ...
Questions processed: ...
Sub-questions processed: ...
Visual questions checked: ...
Formula questions checked: ...
GeoGebra tasks: ...
Unresolved / unclear items: ...
[/COMPLETENESS CHECK]

If complete, state:

No question or sub-question was intentionally omitted.

============================================================
IMPORTANT
============================================================

Do not mention ChemType.

Do not mention MathType.

Do not recreate the original paper.

Do not generate a replacement version of the questions.

Generate answers and marking information only.

Be rigorous.
"""


    # --------------------------------------------------------
    # Build multimodal request
    # --------------------------------------------------------

    content = [
        {
            "type": "input_file",
            "file_id": file_obj.id
        },
        {
            "type": "input_text",
            "text": prompt
        }
    ]


    # --------------------------------------------------------
    # Add rendered pages as visual inputs
    # --------------------------------------------------------

    #
    # The original PDF remains the primary document.
    # Rendered pages provide an additional visual representation
    # so diagrams and mathematical notation are less dependent
    # on PDF text extraction.
    #

    MAX_VISUAL_PAGES = 40

    for item in page_images[:MAX_VISUAL_PAGES]:

        encoded = base64.b64encode(
            item["image"]
        ).decode()

        content.append(
            {
                "type": "input_text",
                "text": (
                    f"\n===== VISUAL PAGE "
                    f"{item['page']} =====\n"
                    f"Inspect this page image carefully."
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


    # --------------------------------------------------------
    # Call model
    # --------------------------------------------------------

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
# REMOVE SPECIAL BLOCKS FROM MAIN MARKING SCHEME
# ============================================================

def main_scheme(text):

    tags = [
        "GEOGEBRA",
        "VISUAL CHECK",
        "FORMULA",
        "COMPLETENESS CHECK"
    ]

    result = text

    for tag in tags:

        result = re.sub(
            rf"\[{tag}\].*?\[/{tag}\]",
            "",
            result,
            flags=re.S | re.I
        )

    return result.strip()


# ============================================================
# SPECIALIST INFORMATION
# ============================================================

def specialist(text):

    visuals = re.findall(
        r"\[VISUAL CHECK\](.*?)\[/VISUAL CHECK\]",
        text,
        flags=re.S | re.I
    )

    geo = re.findall(
        r"\[GEOGEBRA\](.*?)\[/GEOGEBRA\]",
        text,
        flags=re.S | re.I
    )

    formulas = re.findall(
        r"\[FORMULA\](.*?)\[/FORMULA\]",
        text,
        flags=re.S | re.I
    )

    completeness = re.search(
        r"\[COMPLETENESS CHECK\](.*?)\[/COMPLETENESS CHECK\]",
        text,
        flags=re.S | re.I
    )


    # --------------------------------------------------------
    # VISUAL VERIFICATION
    # --------------------------------------------------------

    if visuals:

        st.subheader(
            "👁️ Visual verification"
        )

        for i, item in enumerate(
            visuals,
            1
        ):

            with st.expander(
                f"Visual check {i}"
            ):

                st.write(
                    item.strip()
                )


    # --------------------------------------------------------
    # FORMULAS
    # --------------------------------------------------------

    if formulas:

        st.subheader(
            "∑ Formula verification"
        )

        for i, item in enumerate(
            formulas,
            1
        ):

            with st.expander(
                f"Formula {i}"
            ):

                st.markdown(
                    item.strip()
                )


    # --------------------------------------------------------
    # GEOGEBRA
    # --------------------------------------------------------

    if geo:

        st.subheader(
            "📐 GeoGebra tasks"
        )

        for i, item in enumerate(
            geo,
            1
        ):

            with st.expander(
                f"GeoGebra task {i}",
                expanded=True
            ):

                st.code(
                    item.strip()
                )


    # --------------------------------------------------------
    # COMPLETENESS
    # --------------------------------------------------------

    if completeness:

        st.subheader(
            "✅ Completeness check"
        )

        st.info(
            completeness.group(1).strip()
        )


# ============================================================
# FORMULAI WORKSPACE
# ============================================================

def formula_workspace(text):

    st.subheader(
        "∑ FormulAI — formula workspace"
    )

    st.caption(
        "Mwalimu AI uses LaTeX for accurate mathematical and "
        "scientific notation. FormulAI can be used to convert "
        "equations into editable Word-ready formulae."
    )

    st.link_button(
        "Open FormulAI Formula Generator",
        "https://formulai.io/formula-generator"
    )


    # --------------------------------------------------------
    # Capture common LaTeX forms
    # --------------------------------------------------------

    formulas = []

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
            f"📋 Extracted formulas ({len(formulas)})"
        ):

            for i, formula in enumerate(
                formulas,
                1
            ):

                st.code(
                    formula.strip()
                )


# ============================================================
# GEOGEBRA WORKSPACE
# ============================================================

def geogebra():

    st.subheader(
        "📐 GeoGebra"
    )

    geometry, graphing, calculator = st.tabs(
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

    for tab, url in zip(
        (
            geometry,
            graphing,
            calculator
        ),
        urls
    ):

        with tab:

            components.iframe(
                url,
                height=650,
                scrolling=True
            )


# ============================================================
# SESSION RESET WHEN A NEW PAPER IS UPLOADED
# ============================================================

def paper_identity(uploaded):

    if uploaded is None:
        return None

    return (
        uploaded.name,
        len(uploaded.getvalue())
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

    current_identity = (
        uploaded.name,
        len(data)
    )


    # Clear old result if a different paper is uploaded

    if (
        st.session_state.get(
            "paper_identity"
        )
        != current_identity
    ):

        st.session_state[
            "paper_identity"
        ] = current_identity

        st.session_state.pop(
            "scheme",
            None
        )


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
    # MACHINE TEXT
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
    # VISUAL PAGE COUNT
    # --------------------------------------------------------

    rendered_pages = render_pages(
        data,
        dpi=160
    )

    if rendered_pages:

        st.caption(
            f"👁️ Visual analysis ready: "
            f"{len(rendered_pages)} page(s)"
        )

    else:

        st.warning(
            "Page rendering is unavailable. "
            "The PDF will still be analysed using the "
            "original uploaded document and extracted text."
        )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "📝 Generate Marking Scheme",
        type="primary"
    ):

        with st.spinner(
            "Analysing every page visually and "
            "building the marking scheme..."
        ):

            try:

                scheme = analyse_visual_pages(
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
                    "Unable to generate marking scheme: "
                    f"{e}"
                )


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    if st.session_state.get(
        "scheme"
    ):

        scheme = st.session_state[
            "scheme"
        ]

        st.divider()

        st.subheader(
            "📝 Answers / Marking Scheme"
        )

        st.caption(
            "The original question paper is shown above. "
            "The material below contains the answers, "
            "working and marking information corresponding "
            "to those original questions."
        )


        # Main answer scheme

        main = main_scheme(
            scheme
        )

        if main:

            st.markdown(
                main
            )


        # Specialist visual/formula/GeoGebra information

        specialist(
            scheme
        )


        # FormulAI workspace

        st.divider()

        formula_workspace(
            scheme
        )


        # GeoGebra

        st.divider()

        geogebra()
