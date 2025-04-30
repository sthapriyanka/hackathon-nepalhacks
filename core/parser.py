import pdfplumber
from PIL import Image
import pytesseract
import docx2txt
import re

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

def extract_text_from_image(image_file, ocr_resolution=300):
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

def segment_text(nlp, text, max_length=1000):
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