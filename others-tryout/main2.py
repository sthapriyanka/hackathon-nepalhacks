import streamlit as st
from io import BytesIO
import pytesseract
from PIL import Image
import docx
import pdfplumber
import ollama
import json
import pandas as pd

# Improved QnA generation function
def generate_qna_pairs(text):
    prompt = (
        "Read the following security policy or compliance-related text and generate the most important, clear, "
        "and insightful question-answer pairs. Avoid vague terms like 'this' or 'that'. Do not just summarize, "
        "but explore implications, best practices, responsibilities, and reasoning as well. "
        "Output in JSON list format, where each item has 'question' and 'answer' keys.\n\n"
        f"{text}"
    )

    print("Ollama started")
    response = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": prompt}
    ])

    raw_output = response['message']['content']

    try:
        qna_list = json.loads(raw_output)
        return qna_list
    except json.JSONDecodeError:
        print("⚠️ Error: Output is not valid JSON.")
        print("Raw Output:\n", raw_output)
        return None

# Text extraction
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    else:
        return "Unsupported file type"

# Excel exporter
def convert_to_excel(qna_pairs):
    df = pd.DataFrame(qna_pairs)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='QnA')
        writer.save()
    output.seek(0)
    return output

# def save_qna_to_excel(qna_list, filename="qna_output.xlsx"):
#     df = pd.DataFrame(qna_list)
#     df.to_excel(filename, index=False)
#     print(f"📁 Q&A saved to Excel file: {filename}")

# Streamlit UI
st.set_page_config(page_title="Security Q&A Extractor", layout="wide")
st.title("📄 Security Document Q&A Extractor")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or image)", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_file(uploaded_file)
        st.success("Text extracted successfully.")
    
    st.subheader("📃 Extracted Text")
    with st.expander("View Raw Text"):
        st.text_area("Extracted Content", text, height=200)

    if st.button("🔍 Generate Q&A Pairs"):
        with st.spinner("Generating Q&A pairs..."):
            qna_pairs = generate_qna_pairs(text)

        if qna_pairs:
            st.subheader("📌 Generated Q&A Pairs")
            for i, pair in enumerate(qna_pairs):
                st.markdown(f"*Q{i+1}:* {pair['question']}")
                st.markdown(f"*A{i+1}:* {pair['answer'] if pair['answer'] else '[No answer found]'}")
                st.markdown("---")

            # Download Excel
            excel_file = convert_to_excel(qna_pairs)
            st.download_button(
                label="📥 Download Q&A as Excel",
                data=excel_file,
                file_name="qa_pairs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Failed to generate Q&A pairs. Please check the input or try again.")
