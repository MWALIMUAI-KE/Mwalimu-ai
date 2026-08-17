import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="Mwalimu AI", page_icon="🤖", layout="wide")
st.title("🤖 Mwalimu AI")
st.caption("Teacher AI assistant — targeted road-test upgrade")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not configured. Add it as a deployment secret/environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

def extract_pdf(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"\n--- PAGE {i} ---\n{text}")
    return "\n".join(pages)

def split_question_blocks(paper_text):
    """Best-effort question inventory for completeness checking."""
    matches = list(re.finditer(
        r"(?im)^\s*(?:question\s*)?(\d{1,2})[\.\):\-]\s+",
        paper_text
    ))
    if not matches:
        return []
    return [m.group(1) for m in matches]

def detect_specialist_needs(paper_text, subject):
    """Classify likely questions that benefit from specialist rendering/tools."""
    text = paper_text.lower()
    return {
        "geogebra": bool(re.search(
            r"\\b(construct|construction|locus|bisect|perpendicular|parallel|"
            r"transformation|reflection|rotation|translation|enlargement|"
            r"coordinate geometry|plot|graph|draw the graph|sketch|vector|circle)\\b",
            text)),
        "mathtype": subject.lower() == "mathematics" or bool(re.search(
            r"(equation|fraction|indices|surds|matrix|matrices|logarithm|"
            r"differentiation|integration|vector|probability|quadratic)", text)),
        "chemtype": subject.lower() == "chemistry" or bool(re.search(
            r"(chemical equation|balance|mole|formula|ionic|organic|reaction)", text)),
    }

def generate_marking_scheme(paper_text, subject, level):
    question_numbers = split_question_blocks(paper_text)
    inventory = ", ".join(question_numbers) if question_numbers else "Unable to determine automatically."
    needs = detect_specialist_needs(paper_text, subject)

    routing = f"""
SPECIALIST TOOL ROUTING:
- GeoGebra likely relevant: {needs['geogebra']}
- MathType likely relevant: {needs['mathtype']}
- ChemType likely relevant: {needs['chemtype']}

For a genuine graph/construction/geometry task, do NOT merely say "use GeoGebra".
Produce:
[GEOGEBRA]
Purpose: ...
Commands:
1. ...
2. ...
Expected result: ...
[/GEOGEBRA]

Use GeoGebra-style commands where appropriate, such as Point(A), Line(A,B),
PerpendicularLine(A,g), Circle(A,B), Polygon(A,B,C), Reflect(A,g),
Rotate(A,alpha), Translate(A,v), Intersect(g,h). Do not invent missing data.

For mathematics, use LaTeX delimiters \\( ... \\) and \\[ ... \\].
For chemistry, use correct formulae, charges and state symbols. Where structured
chemistry notation is useful, add:
[CHEMTYPE]
...formula/equation...
[/CHEMTYPE]

Never claim a specialist tool was automatically used unless you actually generated
the corresponding structured block.
"""

    system = """You are Mwalimu AI, a Kenyan teacher-support assistant.
Create a rigorous, complete marking scheme from the supplied question paper.

NON-NEGOTIABLE RULES:
1. Preserve original question and sub-question numbering.
2. Process EVERY question and sub-question in order. Never silently skip one.
3. Show essential calculation working and allocate marks beside steps where appropriate.
4. Preserve mathematical notation with LaTeX.
5. Preserve chemistry formulae, charges and state symbols.
6. For graphs/constructions, generate actionable specialist-tool instructions.
7. If the source is genuinely unclear, identify the exact question and missing information. Never invent.
8. Finish with a COMPLETENESS CHECK listing every question processed and unresolved items.
9. Correctness and traceability matter more than verbosity.
""" + routing

    user = f"""Subject: {subject}
Level: {level}

AUTOMATIC QUESTION INVENTORY (best effort): {inventory}

QUESTION PAPER:
{paper_text}"""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
    )
    return response.choices[0].message.content

def math_display_html():
    return """
    <script>
    window.MathJax = {
      tex: {
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']]
      },
      svg: {fontCache: 'global'}
    };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    """

def show_math_rendered(markdown_text):
    # Streamlit markdown handles the authored content; MathJax is also loaded
    # in the page to provide a stronger math-rendering path for a later HTML renderer.
    st.markdown(markdown_text)

def show_specialist_outputs(scheme):
    """Surface actionable specialist blocks generated by the AI."""
    geo = re.findall(r"\\[GEOGEBRA\\](.*?)\\[/GEOGEBRA\\]", scheme, flags=re.S|re.I)
    chem = re.findall(r"\\[CHEMTYPE\\](.*?)\\[/CHEMTYPE\\]", scheme, flags=re.S|re.I)

    if geo:
        st.subheader("📐 GeoGebra instructions generated from the paper")
        for i, block in enumerate(geo, 1):
            with st.expander(f"GeoGebra task {i}", expanded=True):
                st.code(block.strip(), language="text")
                st.caption("Use these commands in the GeoGebra panel to reproduce the construction/graph.")

    if chem:
        st.subheader("🧪 Structured chemistry notation")
        for i, block in enumerate(chem, 1):
            with st.expander(f"ChemType item {i}", expanded=True):
                st.code(block.strip(), language="text")

def geogebra_panel():
    st.subheader("📐 GeoGebra — graphs & constructions")
    st.caption("Development integration: use GeoGebra here for constructions, geometry and graphing while we refine automatic insertion into generated schemes.")
    tabs = st.tabs(["Geometry", "Graphing", "Calculator Suite"])
    urls = [
        "https://www.geogebra.org/geometry",
        "https://www.geogebra.org/graphing",
        "https://www.geogebra.org/calculator",
    ]
    for tab, url in zip(tabs, urls):
        with tab:
            components.iframe(url, height=600, scrolling=True)

def mathtype_demo_panel():
    st.subheader("➗ MathType / ChemType — notation lab")
    st.caption("Development/evaluation panel. The WIRIS Generic Integration supports both MathType and ChemType; production licensing is handled separately.")
    st.markdown("**MathType developer demo:**")
    st.link_button("Open MathType/ChemType demo", "https://demo.wiris.com/mathtype/en/developers.php")
    st.info("The current MVP is text-first. This panel keeps the legitimate WIRIS development demo available while we wait for the commercial integration quotation and refine the editable HTML rendering layer.")

subject = st.selectbox(
    "Subject",
    ["Mathematics","Chemistry","Biology","Physics","Agriculture",
     "English","Kiswahili","IRE","Other"]
)
level = st.text_input("Level / Grade", "Grade 10")
uploaded = st.file_uploader("Upload a question paper (PDF)", type=["pdf"])

if uploaded:
    st.success(f"Loaded: {uploaded.name}")
    if st.button("🚀 Generate Marking Scheme", type="primary"):
        with st.spinner("Mwalimu AI is analysing the paper..."):
            try:
                text = extract_pdf(uploaded)
                if not text.strip():
                    st.error("No readable text was extracted. OCR for scanned papers is still required for image-only papers.")
                    st.stop()

                scheme = generate_marking_scheme(text, subject, level)

                st.subheader("Generated Marking Scheme")
                show_math_rendered(scheme)
                show_specialist_outputs(scheme)

                st.download_button(
                    "Download marking scheme",
                    scheme,
                    file_name="mwalimu_ai_marking_scheme.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")

with st.expander("🧪 Specialist tools — development test"):
    mathtype_demo_panel()
    geogebra_panel()

st.divider()
st.caption("Road-test V2 development build. Production layers still include login, subscriptions/M-Pesa, usage metering, curriculum retrieval, OCR, automatic graph/diagram generation, database and multi-user security.")
    
