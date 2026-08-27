import os, io, re, base64
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from openai import OpenAI

try:
    import fitz
except ImportError:
    fitz = None

st.set_page_config(page_title="Mwalimu AI", page_icon="🤖", layout="wide")
st.title("🤖 Mwalimu AI")
st.caption("Teacher AI assistant — MVP 4.1: Original paper + Visual analysis + Formula rendering + GeoGebra")

key = os.getenv("OPENAI_API_KEY")
if not key:
    st.error("OPENAI_API_KEY is not configured. Add it as a deployment secret.")
    st.stop()

client = OpenAI(api_key=key)
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

def extract_pdf_text(data):
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(
            f"--- PAGE {i} TEXT ---\n{page.extract_text() or ''}"
            for i, page in enumerate(reader.pages, 1)
        )
    except Exception as e:
        return f"[TEXT EXTRACTION ERROR: {e}]"

def render_pages(data):
    if fitz is None:
        return []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        out = []
        matrix = fitz.Matrix(150/72, 150/72)
        for n, page in enumerate(doc, 1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out.append((n, pix.tobytes("png")))
        doc.close()
        return out
    except Exception:
        return []

def show_original(data):
    st.subheader("📄 Original Question Paper")
    st.caption("The original uploaded paper is shown here. It is NOT reconstructed by OCR.")
    pages = render_pages(data)
    if pages:
        for n, img in pages:
            st.image(img, caption=f"Original question paper — page {n}", use_container_width=True)
    else:
        b64 = base64.b64encode(data).decode()
        components.html(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="900" style="border:1px solid #ddd;"></iframe>',
            height=920, scrolling=True
        )

def question_numbers(text):
    found = re.findall(r"(?im)^\s*(?:question\s*)?(\d{1,2})[.\):\-]\s+", text)
    return list(dict.fromkeys(found))

def needs(text, subject):
    t = text.lower()
    return {
        "geometry": bool(re.search(r"\b(construct|construction|locus|bisect|perpendicular|parallel|bearing|circle|tangent|transformation|reflection|rotation|translation|enlargement|vector|geometry)\b", t)),
        "graph": bool(re.search(r"\b(graph|plot|curve|coordinate|gradient|quadratic|cubic|function|sketch)\b", t)),
        "formula": subject.lower() in {"mathematics","physics","chemistry"} or bool(re.search(r"\b(equation|formula|fraction|indices|surds|matrix|logarithm|differentiation|integration|probability|quadratic)\b", t)),
        "diagram": bool(re.search(r"\b(diagram|figure|apparatus|shown|sketch|graph)\b", t))
    }

def analyse(data, filename, subject, level, text):
    f = client.files.create(file=(filename, data), purpose="user_data")
    ns = needs(text, subject)
    nums = ", ".join(question_numbers(text)) or "Not reliably extracted."
    prompt = f"""
You are Mwalimu AI, a rigorous Kenyan examination marking assistant.
Subject: {subject}
Level: {level}
Preliminary question inventory: {nums}
Geometry: {ns["geometry"]}; Graphing: {ns["graph"]}; Formula-heavy: {ns["formula"]}; Diagrams: {ns["diagram"]}

The uploaded PDF is the AUTHORITATIVE question paper. Visually inspect the actual PDF, including diagrams, graphs, tables, fractions and symbols.

CRITICAL: The application already displays the ORIGINAL PDF. DO NOT recreate, rewrite, paraphrase or typeset the questions. Generate ONLY the answers/marking scheme that belong underneath the original questions.

Process EVERY question and EVERY sub-question. Preserve numbering. Never silently skip anything. Show essential working. Use proper LaTeX. Check calculations. If a visual is unclear, identify the exact question and say what is unclear; NEVER invent missing information.

For geometry, construction and graph questions, provide useful GeoGebra commands in this exact block:
[GEOGEBRA]
Question: ...
Purpose: ...
Input/Commands:
1. ...
2. ...
Expected result: ...
[/GEOGEBRA]

For visually inspected questions, use:
[VISUAL CHECK]
Question: ...
Visual information used: ...
[/VISUAL CHECK]

Finish with:
[COMPLETENESS CHECK]
Questions identified: ...
Questions processed: ...
Sub-questions processed: ...
Visual questions checked: ...
GeoGebra tasks: ...
Unresolved/unclear items: ...
[/COMPLETENESS CHECK]

If complete, state: No question or sub-question was intentionally omitted.
Do not mention ChemType or MathType.
"""
    r = client.responses.create(model=MODEL, input=[{"role":"user","content":[
        {"type":"input_file","file_id":f.id},
        {"type":"input_text","text":prompt}
    ]}])
    return r.output_text

def main_scheme(s):
    for tag in ("GEOGEBRA","VISUAL CHECK","COMPLETENESS CHECK"):
        s = re.sub(rf"\[{tag}\].*?\[/{tag}\]", "", s, flags=re.S|re.I)
    return s.strip()

def specialist(s):
    visuals = re.findall(r"\[VISUAL CHECK\](.*?)\[/VISUAL CHECK\]", s, flags=re.S|re.I)
    geo = re.findall(r"\[GEOGEBRA\](.*?)\[/GEOGEBRA\]", s, flags=re.S|re.I)
    complete = re.search(r"\[COMPLETENESS CHECK\](.*?)\[/COMPLETENESS CHECK\]", s, flags=re.S|re.I)

    if visuals:
        st.subheader("👁️ Visual verification")
        for i, x in enumerate(visuals, 1):
            with st.expander(f"Visual check {i}"):
                st.write(x.strip())
    if geo:
        st.subheader("📐 GeoGebra tasks")
        for i, x in enumerate(geo, 1):
            with st.expander(f"GeoGebra task {i}", expanded=True):
                st.code(x.strip())
    if complete:
        st.subheader("✅ Completeness check")
        st.info(complete.group(1).strip())

def formula_workspace(s):
    st.subheader("∑ FormulAI — formula workspace")
    st.caption("Mwalimu AI produces clean LaTeX notation. FormulAI can convert equations to editable Word-ready formulas.")
    st.link_button("Open FormulAI Formula Generator", "https://formulai.io/formula-generator")
    fs = re.findall(r"\\\[(.*?)\\\]", s, flags=re.S) + re.findall(r"\\\((.*?)\\\)", s, flags=re.S)
    if fs:
        with st.expander(f"📋 Extracted formulas ({len(fs)})"):
            for i, f in enumerate(fs, 1):
                st.code(f.strip())

def geogebra():
    st.subheader("📐 GeoGebra")
    a,b,c = st.tabs(["Geometry","Graphing","Calculator"])
    for tab,url in zip((a,b,c),("https://www.geogebra.org/geometry","https://www.geogebra.org/graphing","https://www.geogebra.org/calculator")):
        with tab:
            components.iframe(url, height=650, scrolling=True)

subject = st.selectbox("Subject", ["Mathematics","Chemistry","Biology","Physics","Agriculture","English","Kiswahili","IRE","Other"])
level = st.text_input("Level / Grade", "Grade 10")
uploaded = st.file_uploader("Upload a question paper (PDF)", type=["pdf"])

if uploaded:
    data = uploaded.getvalue()
    st.success(f"Loaded: {uploaded.name}")

    # IMPORTANT: display the real paper, not an OCR recreation.
    show_original(data)

    text = extract_pdf_text(data)
    with st.expander("🔎 Machine-readable text trace"):
        st.text(text[:30000])

    if st.button("📝 Generate Marking Scheme", type="primary"):
        with st.spinner("Analysing the original paper, including diagrams, graphs, tables and mathematical notation..."):
            try:
                st.session_state["scheme"] = analyse(data, uploaded.name, subject, level, text)
            except Exception as e:
                st.error(f"Unable to generate marking scheme: {e}")

    if st.session_state.get("scheme"):
        s = st.session_state["scheme"]
        st.divider()
        st.subheader("📝 Answers / Marking Scheme")
        st.caption("The original paper is above. The answers and workings below correspond to those original questions.")
        st.markdown(main_scheme(s))
        specialist(s)
        formula_workspace(s)
        st.divider()
        geogebra()
