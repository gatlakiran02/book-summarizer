import streamlit as st
import os
import io
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from database import DatabaseManager
from preprocessor import TextPreprocessor
from cloud_summarizer import CloudLLMSummarizer

st.set_page_config(page_title="Intelligent Book Summarizer", page_icon="📚", layout="wide")
db = DatabaseManager()

def generate_pdf(title: str, author: str, summary_text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=8)
    sub_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=12, textColor='#555555', spaceAfter=16)
    body_style = ParagraphStyle('DocBody', parent=styles['BodyText'], fontSize=10, leading=14, spaceAfter=8)

    elements = [
        Paragraph(f"Book Summary: {title}", title_style),
        Paragraph(f"Author: {author or 'Unknown'}", sub_style),
        Spacer(1, 10)
    ]
    for paragraph in summary_text.splitlines():
        if paragraph.strip():
            elements.append(Paragraph(paragraph.strip(), body_style))
            elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer

with st.sidebar:
    st.header("⚙️ Model & API Settings")
    provider = st.selectbox("LLM Provider", ["Gemini", "OpenAI"], index=0)
    api_key_env = os.getenv("GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY", "")
    api_key = st.text_input(f"{provider} API Key", value=api_key_env, type="password", help="Enter your API key or configure in .env file")
    
    if provider == "Gemini":
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    else:
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])

    st.markdown("---")
    st.header("📋 Summary Settings")
    summary_style = st.selectbox("Format Style", ["Paragraph", "Bullet Points"])
    summary_length = st.select_slider("Length", options=["Concise", "Medium", "Detailed"], value="Medium")
    processing_mode = st.radio("Processing Strategy", ["Single-Pass (Direct Context)", "Map-Reduce (Chunked)"], index=0)

st.title("📚 Intelligent Book Summarization Platform")
st.caption("AI-powered summarization and key idea extraction using Cloud LLM APIs.")

col_input, col_output = st.columns(2, gap="large")

with col_input:
    st.subheader("1. Book Input & Metadata")
    uploaded_file = st.file_uploader("Select or drop a book file (PDF, TXT)", type=["pdf", "txt"])
    title = st.text_input("Title", placeholder="e.g., Carrion Comfort")
    author = st.text_input("Author", placeholder="e.g., Dan Simmons")
    chapter = st.text_input("Chapter / Section (Optional)", placeholder="e.g., Chapter 1")

    raw_text = ""
    file_name = ""
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            raw_text = "
".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        words = len(raw_text.split())
        st.info(f"Loaded **{file_name}** ({words:,} words)")
    else:
        raw_text = st.text_area("Or paste book text directly:", height=220, placeholder="Paste book excerpts or chapters here...")

with col_output:
    st.subheader("2. Generated Summary")
    generate_btn = st.button("Generate Summary", type="primary", use_container_width=True)

    if generate_btn:
        if not api_key:
            st.error(f"Please enter your {provider} API key in the sidebar.")
        elif not raw_text.strip():
            st.error("Please upload a file or paste book text first.")
        else:
            status_text = st.empty()
            try:
                status_text.text("Connecting to Cloud LLM service...")
                summarizer = CloudLLMSummarizer(provider=provider, api_key=api_key, model_name=model_name)
                
                status_text.text("Preprocessing and cleaning text...")
                cleaned_text = TextPreprocessor.clean_text(raw_text)

                status_text.text(f"Executing {model_name} summarization pipeline...")
                if processing_mode.startswith("Single-Pass"):
                    summary_result = summarizer.summarize_single_pass(cleaned_text, summary_length, summary_style, title=title or "Untitled", author=author or "")
                else:
                    summary_result = summarizer.summarize_map_reduce(cleaned_text, summary_length, summary_style, title=title or "Untitled", author=author or "")

                book_id = db.save_book(1, title or "Untitled", author or "Unknown", chapter, file_name or "Direct Input", raw_text)
                db.save_summary(book_id, summary_result, summary_style, summary_length, f"{provider}:{model_name}")

                status_text.empty()
                st.success("Summary Generated Successfully!")
                with st.container(border=True):
                    st.markdown(summary_result)

                st.markdown("#### Export & Actions")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="📄 Download as .txt",
                        data=summary_result,
                        file_name=f"{(title or 'Summary').replace(' ', '_')}_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with btn_col2:
                    pdf_data = generate_pdf(title or "Untitled", author or "Unknown", summary_result)
                    st.download_button(
                        label="📑 Download as .pdf",
                        data=pdf_data,
                        file_name=f"{(title or 'Summary').replace(' ', '_')}_summary.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                status_text.empty()
                st.error(f"Error: {str(e)}")
