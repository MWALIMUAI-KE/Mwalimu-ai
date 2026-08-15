import os, io
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="Mwalimu AI", page_icon="🤖", layout="wide")
st.title("🤖 Mwalimu AI")
st.caption("Teacher AI assistant — first road-test MVP")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not configured. Add it as a deployment secret/environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

def extract_pdf(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "\n".join(
        f"\n--- PAGE {i} ---\n{page.extract_text() or ''}"
        for i, page in enumerate(reader.pages, 1)
    )

def generate_marking_scheme(paper_text, subject, level):
    system = """You are Mwalimu AI, a Kenyan teacher-support assistant.
Create a rigorous marking scheme from the supplied question paper.
Do not invent questions. Preserve question numbering.
For each question provide the expected answer/working, marks allocated,
acceptable alternatives where justified, and brief marking notes.
For calculations, show step-by-step working and mark allocation.
For graphs/diagrams, state what should be shown and how marks are awarded.
If the source is unclear or unreadable, flag it instead of guessing."""
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content":
             f"Subject: {subject}\nLevel: {level}\n\nQUESTION PAPER:\n{paper_text}"}
        ],
    )
    return response.choices[0].message.content

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
                    st.error("No readable text was extracted. OCR for scanned papers is a next-stage feature.")
                    st.stop()
                scheme = generate_marking_scheme(text, subject, level)
                st.subheader("Generated Marking Scheme")
                st.markdown(scheme)
                st.download_button(
                    "Download marking scheme",
                    scheme,
                    file_name="mwalimu_ai_marking_scheme.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")

st.divider()
st.caption("MVP. Production layers still include login, 7-day trial, subscriptions/M-Pesa, usage metering, curriculum retrieval, OCR, graph/diagram handling, database and multi-user security.")
