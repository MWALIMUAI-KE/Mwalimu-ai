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

def generate_marking_scheme(paper_text, subject, level):
    question_numbers = split_question_blocks(paper_text)
    inventory = ", ".join(question_numbers) if question_numbers else "Unable to determine automatically."

    system = """You are Mwalimu AI, a Kenyan teacher-support assistant.
Create a rigorous, complete marking scheme from the supplied question paper.

NON-NEGOTIABLE RULES:
1. Preserve the original question and sub-question numbering.
2. Process EVERY question and sub-question in order. Never silently skip one.
3. For calculations, show all essential step-by-step working and put marks beside the steps where appropriate.
4. Preserve mathematical notation using LaTeX delimiters: use \\(...\\) for inline maths and \\[...\\] for displayed equations. Do not replace fractions, roots, indices, Greek letters or operators with crude plain text.
5. For chemistry, use clear chemical formulae/equations with correct subscripts, charges and state symbols.
6. For graphs/constructions/diagrams, provide a precise specification of what must appear and a concise construction/plotting description. If the application supplies a graphical tool, use it rather than merely saying 'draw a graph'.
7. If the source is genuinely unclear or unreadable, identify the exact question and say what is missing. Do not invent content.
8. Finish with a COMPLETENESS CHECK listing every question number you processed and any unresolved item.
9. This is an examination marking scheme: correctness and traceability matter more than verbosity."""

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
