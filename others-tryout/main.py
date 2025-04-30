import streamlit as st
from io import StringIO
import pytesseract
from PIL import Image
import docx
import pdfplumber
import ollama
import json

# Placeholder for QA generation (to be replaced with actual models)
def generate_qna_pairs(text):
    prompt = f"Generate {5} clear question-answer pairs from the following policy text:\n\n{text}\n\nOutput in JSON format with keys: 'question' and 'answer' so that i can extract it to file json."

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
    # Dummy logic — replace with model-based QnA extraction
    return [
        {"question": "What is the data retention policy?", "answer": "Retain data for 7 years.", "confidence": 0.95},
        {"question": "Who handles access control?", "answer": "", "confidence": 0.40},  # Incomplete answer
    ]

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

        st.subheader("📌 Generated Q&A Pairs")
        for i, pair in enumerate(qna_pairs):
            flag = ""
            # if pair['confidence'] < 0.6 or pair['answer'].strip() == "":
                # flag = "⚠ Low confidence or incomplete"
            with st.container():
                st.markdown(f"*Q{i+1}:* {pair['question']}")
                st.markdown(f"*A{i+1}:* {pair['answer'] if pair['answer'] else '[No answer found]'}")
                # st.markdown(f"*Confidence:* {pair['confidence']:.2f} {flag}")
                st.markdown("---")