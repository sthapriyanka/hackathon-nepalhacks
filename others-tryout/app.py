import streamlit as st
import tempfile
import os
import json
import pandas as pd
import base64
from PIL import Image
import pytesseract
import pdfplumber
import docx2txt
from io import BytesIO
import torch
import re
import spacy
from concurrent.futures import ThreadPoolExecutor
import ollama

# Load spaCy model for NER and text processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Download if not installed
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Set page title and configuration
st.set_page_config(
    page_title="Document-to-QnA Pair Converter",
    page_icon="❓",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2563EB;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
    }
    .qa-question {
        font-weight: bold;
        color: #1E3A8A;
    }
    .qa-answer {
        color: #1F2937;
    }
    .qa-confidence {
        font-size: 0.8rem;
        color: #6B7280;
    }
    .qa-flag {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Document-to-QnA Pair Converter</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Convert security policies, compliance documents, and evidence files into structured question-answer pairs.</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown('<h2 class="sub-header">Configuration</h2>', unsafe_allow_html=True)

# Model selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    ["LLaMA-2-7B", "LLaMA-3-8B"]
)

# QA Generation settings
st.sidebar.markdown('<h3>QA Generation Settings</h3>', unsafe_allow_html=True)
min_confidence = st.sidebar.slider("Minimum Confidence Score", 0.0, 1.0, 0.7)
max_qa_pairs = st.sidebar.slider("Maximum QA Pairs", 5, 50, 20)
advanced_mode = st.sidebar.checkbox("Advanced Mode", False)

if advanced_mode:
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.95)
    ocr_resolution = st.sidebar.slider("OCR Resolution (DPI)", 150, 600, 300)
else:
    temperature = 0.2
    top_p = 0.95
    ocr_resolution = 300

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using pdfplumber."""
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Extract text while preserving some layout
            extracted_text += page.extract_text() + "\n\n"
    return extracted_text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file."""
    return docx2txt.process(docx_file)

def extract_text_from_image(image_file):
    """Extract text from image using OCR."""
    image = Image.open(image_file)
    # Increase resolution for better OCR results
    text = pytesseract.image_to_string(image, config=f'--dpi {ocr_resolution}')
    return text

def extract_text(file_path, file_type):
    """Extract text based on file type."""
    if file_type == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_type in ["docx", "doc"]:
        return extract_text_from_docx(file_path)
    elif file_type in ["jpg", "jpeg", "png"]:
        return extract_text_from_image(file_path)
    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Unsupported file format"

def clean_text(text):
    """Clean extracted text for better processing."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere with model
    text = re.sub(r'[^\w\s.,;:?!()\[\]{}\-\'""`]', '', text)
    return text.strip()

def segment_text(text, max_length=1000):
    """
    Segment long text into manageable, semantically meaningful chunks using spaCy.
    """
    doc = nlp(text[:1000000])  # Limit to 1 million characters for safety

    segments = []
    current_segment = ""

    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(current_segment) + len(sentence) + 1 > max_length:
            segments.append(current_segment.strip())
            current_segment = sentence
        else:
            current_segment += " " + sentence if current_segment else sentence

    if current_segment:
        segments.append(current_segment.strip())

    return segments

def generate_qa_pairs(text_segment, max_pairs=5):
    """Generate QA pairs from text segment using LLaMA model."""
    # For demonstration purposes, we'll simulate the LLaMA model response
    # In a real implementation, you would use the model to generate QA pairs
    
    # Simulate QA pair generation
    qa_pairs = []
    prompt = f"""
    You are an expert at analyzing security and compliance documents and extracting question-answer pairs.
    Given the following text from a document, create {max_pairs} clear question-answer pairs related to security policies, 
    compliance requirements, and standards. Focus on extracting factual information that would be useful for security assessments.
    
    TEXT:
    {text_segment}
    
    For each pair, provide:
    1. A clear, direct question about a security policy, requirement, or practice
    2. A concise but complete answer based only on the text
    3. A confidence score between 0 and 1
    4. Any flags for ambiguity or incompleteness
    
    Format your response as JSON:
    [
        {{
            "question": "Question text here?",
            "answer": "Answer text here.",
            "confidence": 0.85,
            "flags": ["flag 1", "flag 2"]
        }},
        ...
    ]
    """

    response = ollama.chat(
        model='qwen3',
        messages=[{ "role": "assistant", "content": prompt }],
        options={
            "temperature": temperature,
            "top_p": top_p
        }
    )

    content = response['message']['content']
    
    # Extract JSON from response
    json_match = re.search(r'\[\s*{\s*"question"', content)
    qa_pairs = []

    if json_match:
        json_str = content[json_match.start():]
        try:
            qa_pairs = json.loads(json_str)
        except:
            # Fallback to regex extraction if JSON parsing fails
            questions = re.findall(r'"question":\s*"([^"]+)"', json_str)
            answers = re.findall(r'"answer":\s*"([^"]+)"', json_str)
            confidences = re.findall(r'"confidence":\s*([\d.]+)', json_str)

            for i in range(min(len(questions), len(answers), len(confidences))):
                qa_pairs.append({
                    "question": questions[i],
                    "answer": answers[i],
                    "confidence": float(confidences[i]),
                    "flags": []
                })

    return qa_pairs

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def process_document():
    """Main document processing pipeline."""
    # File uploader
    st.markdown('<h2 class="sub-header">Upload Document</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or image file",
        type=["pdf", "docx", "doc", "jpg", "jpeg", "png", "txt"]
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {"Filename": uploaded_file.name, "File size": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        with st.spinner("Processing document..."):
            try:
                # Extract text based on file type
                file_type = uploaded_file.name.split(".")[-1].lower()
                extracted_text = extract_text(temp_path, file_type)
                
                # Clean and segment text
                cleaned_text = clean_text(extracted_text)
                text_segments = segment_text(cleaned_text)
                
                # Show text extraction results in an expandable section
                with st.expander("View Extracted Text", expanded=False):
                    st.text_area("Extracted text", extracted_text, height=300)
                
                # Load LLM model
                # tokenizer, model = load_llm_model()
                
                # Process segments to generate QA pairs
                all_qa_pairs = []
                
                with st.spinner("Generating QA pairs... This may take a few minutes."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Process each segment
                    for i, segment in enumerate(text_segments):
                        segment_qa_pairs = generate_qa_pairs(
                            segment, 
                            max_pairs=max(1, int(max_qa_pairs/len(text_segments)))
                        )
                        all_qa_pairs.extend(segment_qa_pairs)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(text_segments))
                
                # Filter by confidence score
                filtered_qa_pairs = [qa for qa in all_qa_pairs if qa["confidence"] >= min_confidence]
                
                # Limit to max_qa_pairs
                filtered_qa_pairs = filtered_qa_pairs[:max_qa_pairs]
                
                # Show QA pairs
                st.markdown('<h2 class="sub-header">Generated QA Pairs</h2>', unsafe_allow_html=True)
                st.write(f"Generated {len(filtered_qa_pairs)} QA pairs (filtered from {len(all_qa_pairs)} total)")
                
                # Convert to DataFrame for display
                df = pd.DataFrame(filtered_qa_pairs)
                
                # Add filtering and sorting options
                col1, col2 = st.columns(2)
                with col1:
                    filter_option = st.selectbox(
                        "Filter QA Pairs",
                        ["All", "High Confidence (>0.8)", "Flagged Items", "No Flags"]
                    )
                with col2:
                    sort_option = st.selectbox(
                        "Sort QA Pairs",
                        ["Confidence (High to Low)", "Confidence (Low to High)"]
                    )
                
                # Apply filtering
                display_df = df.copy()
                if filter_option == "High Confidence (>0.8)":
                    display_df = display_df[display_df["confidence"] > 0.8]
                elif filter_option == "Flagged Items":
                    display_df = display_df[display_df["flags"].apply(lambda x: len(x) > 0)]
                elif filter_option == "No Flags":
                    display_df = display_df[display_df["flags"].apply(lambda x: len(x) == 0)]
                
                # Apply sorting
                if sort_option == "Confidence (High to Low)":
                    display_df = display_df.sort_values("confidence", ascending=False)
                else:
                    display_df = display_df.sort_values("confidence", ascending=True)
                
                # Display QA pairs
                for i, row in display_df.iterrows():
                    print(row)
                    with st.container():
                        st.markdown(f"""
                        <div style="padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem; border: 1px solid #E5E7EB;">
                            <p class="qa-question">Q: {row['question']}</p>
                            <p class="qa-answer">A: {row['answer']}</p>
                            <p class="qa-confidence">Confidence: {row['confidence_score']:.2f}</p>
                            {f'<p class="qa-flag">Flags: {", ".join(row["flags"])}</p>' if row["flags"] else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download options
                st.markdown('<h3>Export Results</h3>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(get_download_link(df, "qa_pairs.csv", "Download CSV"), unsafe_allow_html=True)
                with col2:
                    # JSON download
                    json_str = json.dumps(filtered_qa_pairs, indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    st.markdown(
                        f'<a href="data:file/json;base64,{b64}" download="qa_pairs.json">Download JSON</a>',
                        unsafe_allow_html=True
                    )
            
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            
            finally:
                # Clean up the temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

# Cleanup temp files on session end
def cleanup():
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith('tmp'):
            try:
                os.unlink(os.path.join(temp_dir, file))
            except:
                pass

# Main function
def main():
    process_document()
    
    # Add info about usage
    with st.expander("How to Use This Tool", expanded=False):
        st.markdown("""
        ### Instructions
        1. **Upload Document**: Select a PDF, DOCX, or image file containing policy or compliance text.
        2. **Configure Settings**: Adjust model parameters and filtering options in the sidebar.
        3. **View Results**: Explore generated QA pairs and their confidence scores.
        4. **Export**: Download results in CSV or JSON format for further use.
        
        ### Tips for Best Results
        - For complex documents, consider splitting them into smaller parts before uploading.
        - Adjust the confidence threshold to filter out low-quality pairs.
        - Check flagged items manually, as they may require human verification.
        - For images, higher resolution may improve OCR quality but increases processing time.
        """)
    
    # Add example
    with st.expander("Example Input & Output", expanded=False):
        st.markdown("""
        ### Example Input Text
        ```
        The Acceptable Use Policy (AUP) outlines the requirements for proper use of company 
        information and technology resources. All employees must sign the AUP during onboarding 
        and annually thereafter. The policy covers electronic transmission security including 
        email, web, and file transfer services protocols. All users are responsible for ensuring 
        the security of their assigned devices including mobile phones and laptops.
        ```
        
        ### Example QA Pairs Generated
        1. **Q**: Is there an acceptable use policy for company information and technology resources?  
           **A**: The Acceptable Use Policy (AUP) outlines the requirements for proper use of company information and technology resources.  
           **Confidence**: 0.95
           
        2. **Q**: Are employees required to sign the Acceptable Use Policy?  
           **A**: All employees must sign the AUP during onboarding and annually thereafter.  
           **Confidence**: 0.92
           
        3. **Q**: Does the policy cover electronic transmission security requirements?  
           **A**: The policy covers electronic transmission security including email, web, and file transfer services protocols.  
           **Confidence**: 0.89
           
        4. **Q**: Who is responsible for ensuring device security?  
           **A**: All users are responsible for ensuring the security of their assigned devices including mobile phones and laptops.  
           **Confidence**: 0.85
        """)

if __name__ == "__main__":
    main()
    # Register cleanup function
    import atexit
    atexit.register(cleanup)